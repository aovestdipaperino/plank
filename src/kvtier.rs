// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Volatility-tiered KV cache planning (issues #60, #64).
//!
//! The prompt prefix is a hierarchy of tiers ordered **most stable first**, each
//! an extension checkpoint of the one above it:
//!
//! | Tier | Content | Key | Storage |
//! |------|---------|-----|---------|
//! | 1 | system prompt (+ global MCP tool defs, incl. cached ones) | `fp1 = sha(model ‖ system)` | `sysprompt-*.kv` (model-global) |
//! | 2 | project-stable context: `AGENTS.md`/`CLAUDE.md` set, memory, **local** MCP tool defs | `fp2 = tier(fp1, stable-hash ‖ local tool defs)` | `<project-key>/project-<fp2>.kv` |
//! | 3 | session-volatile context: git status, date, hook output | — | never cached |
//! | 4 | conversation turns | `tier(fp2, transcript)` | `<session>.payload` |
//!
//! Tier 1's text depends on the set of global MCP servers **present in
//! `~/.plank/.mcp.json`**, not the set that successfully handshook: a global
//! server that fails to start is rendered from its cached advertisement
//! ([`crate::tools::mcp_advert`]) so a flaky server cannot invalidate the most
//! expensive tier. Tier 2 is unaffected — [`crate::tools::mcp::local_tool_defs`]
//! sees only project-local servers, which never get cached advertisements.
//!
//! This module owns the **pure** part of that machinery: building the tier list
//! below Tier 1, canonicalizing the local MCP tool definitions that key Tier 2,
//! and deciding — via [`warm`] — which tier to restore from and where prefill
//! must resume. The walk itself is backend-agnostic: it drives the engine only
//! through `warm_reset`/`warm_append`/`warm_sync`/`get_kv`/`set_kv`, so it is always compiled
//! and unit tested against a spy engine.
//!
//! The walk rule is the one already used for `sysprompt.kv`, generalized:
//! *reuse KV until the first fingerprint mismatch, then prefill only from
//! there*. Because each tier's key embeds its parent's fingerprint, a valid
//! tier implies every ancestor is valid too — so "deepest valid" and "last of
//! the leading valid run" coincide, and a stale checkpoint is rebuilt, never
//! trusted.

use std::path::Path;

use crate::engine::{Engine, EngineError, EngineEvent};
use crate::session::KvKey;
use crate::session::SessionStore;
use crate::session::tier_fingerprint;

/// Changed lines shown in the first-change snippet that explains a Tier 1
/// (system-prompt) cache miss. Small on purpose: the point is to recognize the
/// change, not to read the whole diff.
const SYSPROMPT_SNIPPET_LINES: usize = 6;

/// Which volatility tier a [`TierSpec`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierKind {
    /// Tier 1: model + system prompt. Cached globally, shared across projects.
    System,
    /// Tier 2: project-stable context, shared by every session of a project.
    ProjectStable,
    /// Tier 3: session-volatile context. Prefill-only, never checkpointed.
    SessionVolatile,
}

/// One tier of the prefix: the text it contributes, its chained fingerprint,
/// and where (if anywhere) its KV checkpoint lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierSpec {
    /// Which tier this is.
    pub kind: TierKind,
    /// Chained fingerprint `sha(parent ‖ NUL ‖ material)`.
    pub fingerprint: String,
    /// The text this tier contributes, injected as its own user message so the
    /// tier boundary falls on a clean chat-template message boundary.
    pub text: String,
    /// Where this tier's KV checkpoint is stored, or `None` for a prefill-only
    /// tier (Tier 3) or when there is no session store.
    pub key: Option<KvKey>,
}

impl TierSpec {
    /// Whether this tier is persisted as a KV checkpoint. Tier 3 never is.
    #[must_use]
    pub fn cacheable(&self) -> bool {
        self.key.is_some()
    }
}

/// Tier 1's fingerprint: `sha1(model ‖ NUL ‖ system)`.
///
/// The single definition of the system-prompt checkpoint key, shared by the
/// engine (which writes `sysprompt-*.kv` with it) and by the tier planner
/// (which chains `fp2` off it). Keeping one implementation is what guarantees
/// the two never disagree and silently invalidate the whole chain.
#[must_use]
pub fn system_fingerprint(model: &str, system: &str) -> String {
    let mut data = model.as_bytes().to_vec();
    data.push(0);
    data.extend_from_slice(system.as_bytes());
    crate::session::sha1_hex(&data)
}

/// Canonical, order-independent serialization of a scope's MCP tool
/// definitions, used as key material for the tier that owns that scope
/// (local definitions → Tier 2's `fp2`).
///
/// Servers are sorted by name and each server's tools by tool name, so the
/// fingerprint does not depend on config or handshake ordering. Each tool is
/// rendered as `server/tool\u{1}schema` on its own line, so a schema change is
/// as invalidating as a tool appearing or disappearing.
#[must_use]
pub fn tool_defs_material(defs: &[(String, Vec<(String, String)>)]) -> String {
    let mut servers: Vec<&(String, Vec<(String, String)>)> = defs.iter().collect();
    servers.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = String::new();
    for (server, tools) in servers {
        let mut tools: Vec<&(String, String)> = tools.iter().collect();
        tools.sort_by(|a, b| a.0.cmp(&b.0));
        for (tool, schema) in tools {
            out.push_str(server);
            out.push('/');
            out.push_str(tool);
            out.push('\u{1}');
            out.push_str(schema);
            out.push('\n');
        }
    }
    out
}

/// Key material for Tier 2: the stable context hash and the canonical local MCP
/// tool definitions, `NUL`-separated so the two fields cannot be confused for
/// one another.
#[must_use]
fn tier2_material(stable_hash: &str, local_tool_defs: &str) -> Vec<u8> {
    let mut data = stable_hash.as_bytes().to_vec();
    data.push(0);
    data.extend_from_slice(local_tool_defs.as_bytes());
    data
}

/// Builds the tier list, ordered most-stable-first, with Tier 1 (the system
/// prompt) leading it.
///
/// `fp1` is the Tier 1 (system prompt) fingerprint the chain hangs off;
/// `project_dir` is the project directory that keys Tier 2's checkpoint
/// (`None` disables Tier 2 caching, e.g. when there is no session store).
/// Tiers whose text is empty are omitted entirely, so a project with no
/// `AGENTS.md` simply has no Tier 2 and pays nothing for it.
#[must_use]
pub fn plan(
    fp1: &str,
    system: &str,
    stable: &str,
    volatile: &str,
    local_tool_defs: &str,
    project_dir: Option<&Path>,
) -> Vec<TierSpec> {
    // Tier 1 leads the list so the warm walk is one uniform loop over tiers
    // rather than a system-prompt phase plus a tier phase. Its text is NOT
    // trimmed: the system prompt is tokenized as a `system`-role message by
    // `build_system_tokens`, not by the user-message path that `parse_sections`
    // trims, so trimming here would change `fp1` and invalidate every existing
    // system checkpoint for no reason.
    let mut tiers = vec![TierSpec {
        kind: TierKind::System,
        fingerprint: fp1.to_owned(),
        text: system.to_owned(),
        key: Some(KvKey::System { fp: fp1.to_owned() }),
    }];
    // Canonicalize each tier to the exact text the turn will tokenize. A tier
    // becomes one user message, and the turn rebuilds its tokens from the
    // rendered transcript, where `parse_sections` trims every message's
    // trailing whitespace. Tokenizing the untrimmed text at warm time would
    // therefore diverge from the turn at the first tier and re-prefill
    // everything below it — `stable_context()` ends with a newline, so that is
    // the default case, not an edge case (#64).
    let stable = stable.trim_end();
    let volatile = volatile.trim_end();
    // Tier 2 exists only when there is stable text to cache. Local MCP tool
    // definitions key it but are *not* prefix text — they already reached the
    // model through the system prompt; folding them into fp2 (rather than fp1)
    // is what keeps Tier 1 genuinely cross-project.
    let fp2 = if stable.is_empty() {
        fp1.to_owned()
    } else {
        let stable_hash = crate::session::sha1_hex(stable.as_bytes());
        let fp = tier_fingerprint(fp1, &tier2_material(&stable_hash, local_tool_defs));
        tiers.push(TierSpec {
            kind: TierKind::ProjectStable,
            key: project_dir.map(|d| KvKey::Project {
                dir: d.to_path_buf(),
                fp: fp.clone(),
            }),
            fingerprint: fp.clone(),
            text: stable.to_owned(),
        });
        fp
    };
    // Tier 3 is prefill-only: it always differs, so caching it would only ever
    // write a checkpoint no launch could reuse.
    if !volatile.is_empty() {
        tiers.push(TierSpec {
            kind: TierKind::SessionVolatile,
            fingerprint: tier_fingerprint(&fp2, volatile.as_bytes()),
            text: volatile.to_owned(),
            key: None,
        });
    }
    tiers
}

/// Warms the KV cache over `tiers` (built by [`plan`], most-stable-first).
///
/// Walks deepest-first for the first tier whose checkpoint loads clean,
/// restores it, then prefills only the tiers below it — persisting each
/// cacheable tier exactly at its own boundary. Returns `true` when any prefill
/// ran.
///
/// Deepest-first is sound without independently revalidating ancestors: each
/// tier's fingerprint chains its parent's (see [`tier_fingerprint`]), so a deep
/// tier matching *proves* its ancestors match.
///
/// A checkpoint that keys correctly but fails to load is a miss, not an error:
/// the walk simply falls back to the previous tier. No corrupt cache file can
/// abort startup.
///
/// # Errors
/// Returns [`EngineError`] only when the engine itself fails to restore or
/// prefill. Cache IO failures are best-effort and never propagate.
pub fn warm(
    engine: &mut dyn Engine,
    store: Option<&SessionStore>,
    tiers: &[TierSpec],
    on_event: &mut dyn FnMut(EngineEvent),
) -> Result<bool, EngineError> {
    let Some(system) = tiers.first().filter(|t| t.kind == TierKind::System) else {
        return Ok(false);
    };
    engine.warm_reset(&system.text)?;

    // Restore: deepest tier whose checkpoint loads clean.
    let mut resume = 0;
    for (i, t) in tiers.iter().enumerate().rev() {
        let Some(cache) = t
            .key
            .as_ref()
            .zip(store)
            .and_then(|(key, store)| store.kv_load(key))
        else {
            continue;
        };
        if engine.set_kv(&cache).is_ok() {
            resume = i + 1;
            break;
        }
        // Keyed correctly but the bytes would not load into this build: say so
        // and keep walking upward rather than trusting them.
        on_event(EngineEvent::Notice(
            "context cache is incompatible with this build; rebuilding it".to_owned(),
        ));
    }

    // A Tier 1 miss is the expensive one — the system prompt is the largest
    // prefix and everything below it re-prefills too — so explain it before the
    // bar starts moving, diffing against the prompt text behind the previous
    // checkpoint so a benign change (a ticking MCP tool count, a new date) is
    // obvious at a glance. Only Tier 1 gets this treatment: the cheaper tiers
    // are not worth a sidecar of their own.
    if resume == 0
        && let Some(old) = store.and_then(SessionStore::system_prompt_note)
        && let Some(snip) =
            crate::tools::diff::first_change_snippet(&old, &system.text, SYSPROMPT_SNIPPET_LINES)
    {
        on_event(EngineEvent::Notice(format!(
            "system prompt changed; rebuilding cache\n{snip}"
        )));
    }

    // Extend: prefill each remaining tier, persisting cacheable ones at their
    // own boundary. A snapshot captures the *whole* session, so the capture
    // must happen while the cursor sits at the end of this tier and no further
    // — persisting after the next sync would store the next tier's KV under
    // this tier's key, which fingerprints cannot detect because the key would
    // be genuinely correct.
    let mut prefilled = false;
    for (i, t) in tiers.iter().enumerate() {
        // Append for EVERY tier, including the ones already restored above: the
        // engine's cumulative token buffer must describe the *whole* restored
        // prefix. Skipping the append for restored tiers would leave a buffer
        // with a hole in it, and the next sync — seeing a common prefix shorter
        // than the buffer — would rewrite the session's checkpoint from that
        // truncated buffer, throwing the restored KV away and making a deep hit
        // strictly worse than a cold start.
        //
        // The system tier's tokens are already in the warm buffer from
        // `warm_reset`; every other tier appends its text as a user message.
        let text = (i > 0).then_some(t.text.as_str());
        engine.warm_append(text)?;
        if i < resume {
            // Already in KV via the restore above; extend the buffer, do not
            // sync — and do not re-persist a checkpoint we just read.
            continue;
        }
        prefilled |= engine.warm_sync(on_event)?;
        if let Some(key) = &t.key
            && let Some(store) = store
            && let Some(cache) = engine.get_kv()
        {
            // Tier checkpoints intentionally carry an **empty**
            // `TokenTranscript`: warming never touches `self.transcript`, and
            // nothing needs it — `reconcile` rebuilds spans from text and the
            // C-side common-prefix probe does the real matching, exactly as the
            // older transcript-less checkpoint format did. Do not start
            // trusting `cache.transcript()` for these.
            let _ = store.kv_store(key, &cache);
        }
    }
    // Refresh the sidecar only after a Tier 1 rebuild actually completed: a
    // prefill that fails leaves no new checkpoint, so the *old* text is still
    // the one the next launch will be missing against.
    if resume == 0
        && let Some(store) = store
    {
        let _ = store.store_system_prompt_note(&system.text);
    }
    Ok(prefilled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn plan_emits_the_system_tier_first_and_keys_each_tier() {
        let fp1 = system_fingerprint("model-a", "SYSTEM");
        let tiers = plan(
            &fp1,
            "SYSTEM",
            "agents\n",
            "git status\n",
            "",
            Some(Path::new("/p")),
        );

        assert_eq!(tiers.len(), 3, "system + project-stable + volatile");
        assert_eq!(tiers[0].kind, TierKind::System);
        assert_eq!(tiers[0].text, "SYSTEM");
        assert_eq!(tiers[0].fingerprint, fp1);
        assert_eq!(tiers[0].key, Some(KvKey::System { fp: fp1.clone() }));

        assert_eq!(tiers[1].kind, TierKind::ProjectStable);
        assert_eq!(
            tiers[1].key,
            Some(KvKey::Project {
                dir: "/p".into(),
                fp: tiers[1].fingerprint.clone()
            })
        );

        // Tier 3 stays prefill-only: keying it on volatile bytes would write a
        // checkpoint no launch could ever hit.
        assert_eq!(tiers[2].kind, TierKind::SessionVolatile);
        assert_eq!(tiers[2].key, None);
    }

    #[test]
    fn plan_without_a_project_dir_still_caches_the_system_tier() {
        let fp1 = system_fingerprint("m", "SYSTEM");
        let tiers = plan(&fp1, "SYSTEM", "agents\n", "", "", None);
        assert_eq!(tiers[0].key, Some(KvKey::System { fp: fp1 }));
        assert_eq!(tiers[1].key, None, "no store, no project checkpoint");
    }

    /// Regression (#64): a tier's text is tokenized twice — once at warm time
    /// (`warm`, verbatim) and once per turn, where it has been through
    /// `render_transcript` → `parse_sections`, which trims each message's
    /// trailing whitespace. Trailing whitespace on a tier therefore tokenizes
    /// differently in the two paths, the KV common-prefix probe diverges at
    /// that tier, and every tier from there down is re-prefilled on the first
    /// question. `stable_context()` ends with a newline, so this bit exactly
    /// as soon as it became a message of its own.
    #[test]
    fn tier_text_is_canonical_for_the_transcript_round_trip() {
        let content = crate::context::ContextContent {
            git_content: Some("GIT-STATUS".to_string()),
            agents_md_content: Some("AGENTS".to_string()),
            memory_content: Some("MEM".to_string()),
            date_content: "DATE".to_string(),
        };
        // Guard the premise: if this stops being true the bug is gone for a
        // different reason and this test should be revisited, not deleted.
        assert!(
            content.stable_context().ends_with('\n'),
            "premise: stable_context ends with a newline"
        );

        let tiers = plan(
            "fp1",
            "SYSTEM",
            &content.stable_context(),
            &content.volatile_context(),
            "",
            None,
        );
        assert!(!tiers.is_empty(), "expected tiers for non-empty context");
        for t in &tiers {
            assert_eq!(
                t.text,
                t.text.trim_end(),
                "{:?} tier text must match what parse_sections yields for it",
                t.kind
            );
        }
    }

    #[test]
    fn plan_orders_stable_before_volatile_and_only_caches_tier2() {
        let tiers = plan(
            "fp1",
            "SYSTEM",
            "agents\n",
            "git + date\n",
            "",
            Some(Path::new("/p")),
        );
        assert_eq!(tiers.len(), 3);
        assert_eq!(tiers[1].kind, TierKind::ProjectStable);
        // Trailing whitespace is trimmed: each tier is one user message, and
        // the turn rebuilds it from the transcript, which trims (#64).
        assert_eq!(tiers[1].text, "agents");
        assert!(tiers[1].cacheable(), "Tier 2 is checkpointed");
        assert_eq!(tiers[2].kind, TierKind::SessionVolatile);
        assert_eq!(tiers[2].text, "git + date");
        assert!(
            !tiers[2].cacheable(),
            "Tier 3 is prefill-only and must never be checkpointed"
        );
        // Splitting adds no text of its own: the tiers rejoined by the single
        // newline `render_transcript` puts between messages are exactly the
        // combined context as the turn sees it (i.e. after its own trim).
        assert_eq!(
            tiers[1..]
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            "agents\ngit + date\n".trim_end()
        );
    }

    #[test]
    fn plan_omits_empty_tiers() {
        let tiers = plan("fp1", "SYSTEM", "", "git\n", "", Some(Path::new("/p")));
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[1].kind, TierKind::SessionVolatile);

        let tiers = plan("fp1", "SYSTEM", "agents\n", "", "", Some(Path::new("/p")));
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[1].kind, TierKind::ProjectStable);

        assert_eq!(
            plan("fp1", "SYSTEM", "", "", "", Some(Path::new("/p"))).len(),
            1,
            "only the system tier remains"
        );
    }

    #[test]
    fn tier2_key_chains_off_tier1_and_folds_in_local_tool_defs() {
        let base = plan("fp1", "SYSTEM", "agents\n", "v", "", Some(Path::new("/p")));
        let other_parent = plan(
            "fp1-changed",
            "SYSTEM",
            "agents\n",
            "v",
            "",
            Some(Path::new("/p")),
        );
        let other_stable = plan("fp1", "SYSTEM", "agents2\n", "v", "", Some(Path::new("/p")));
        let other_tools = plan(
            "fp1",
            "SYSTEM",
            "agents\n",
            "v",
            "srv/tool\u{1}{}\n",
            Some(Path::new("/p")),
        );

        assert_ne!(base[1].fingerprint, other_parent[1].fingerprint);
        assert_ne!(base[1].fingerprint, other_stable[1].fingerprint);
        assert_ne!(
            base[1].fingerprint, other_tools[1].fingerprint,
            "local MCP tool definitions must key Tier 2"
        );
        // Deterministic.
        assert_eq!(
            base[1].fingerprint,
            plan("fp1", "SYSTEM", "agents\n", "v", "", Some(Path::new("/p")))[1].fingerprint
        );
        // And the checkpoint key follows the fingerprint.
        assert_eq!(
            base[1].key,
            Some(KvKey::Project {
                dir: "/p".into(),
                fp: base[1].fingerprint.clone()
            })
        );
    }

    #[test]
    fn volatile_tier_never_gets_a_checkpoint_even_with_a_path_fn() {
        let tiers = plan("fp1", "SYSTEM", "s", "v", "", Some(Path::new("/p")));
        assert!(
            tiers.iter().all(|t| t.cacheable()
                == (t.kind == TierKind::System || t.kind == TierKind::ProjectStable))
        );
    }

    #[test]
    fn tier2_is_uncached_without_a_checkpoint_path() {
        let tiers = plan("fp1", "SYSTEM", "agents\n", "v", "", None);
        assert_eq!(tiers[1].kind, TierKind::ProjectStable);
        assert!(!tiers[1].cacheable());
        // Still keyed, so a caller that later gains a store can reuse the fp.
        assert!(!tiers[1].fingerprint.is_empty());
    }

    #[test]
    fn tool_defs_material_is_order_independent_and_content_keyed() {
        let a = vec![
            (
                "beta".to_owned(),
                vec![
                    ("z".to_owned(), "{}".to_owned()),
                    ("a".to_owned(), "{}".to_owned()),
                ],
            ),
            ("alpha".to_owned(), vec![("t".to_owned(), "{}".to_owned())]),
        ];
        let b = vec![
            ("alpha".to_owned(), vec![("t".to_owned(), "{}".to_owned())]),
            (
                "beta".to_owned(),
                vec![
                    ("a".to_owned(), "{}".to_owned()),
                    ("z".to_owned(), "{}".to_owned()),
                ],
            ),
        ];
        assert_eq!(tool_defs_material(&a), tool_defs_material(&b));
        assert_eq!(
            tool_defs_material(&a),
            "alpha/t\u{1}{}\nbeta/a\u{1}{}\nbeta/z\u{1}{}\n"
        );
        // A schema change is invalidating.
        let c = vec![(
            "alpha".to_owned(),
            vec![("t".to_owned(), "{\"x\":1}".to_owned())],
        )];
        assert_ne!(tool_defs_material(&c), tool_defs_material(&a));
        assert_eq!(tool_defs_material(&[]), "");
    }

    #[test]
    fn system_fingerprint_is_model_and_prompt_keyed() {
        let a = system_fingerprint("model-a", "sys");
        assert_eq!(a, system_fingerprint("model-a", "sys"));
        assert_ne!(a, system_fingerprint("model-b", "sys"));
        assert_ne!(a, system_fingerprint("model-a", "sys2"));
        // The NUL separator keeps the two fields from bleeding together.
        assert_ne!(system_fingerprint("ab", "c"), system_fingerprint("a", "bc"));
        // Matches the historical `sha1(model ‖ NUL ‖ system)` layout the
        // existing sysprompt checkpoints were written with.
        let mut expect = b"model-a".to_vec();
        expect.push(0);
        expect.extend_from_slice(b"sys");
        assert_eq!(a, crate::session::sha1_hex(&expect));
    }
}

/// Records what a warm walk asked the engine to do, so the walk's logic can be
/// tested with no model and no filesystem-backed engine.
#[cfg(test)]
#[derive(Default)]
struct SpyEngine {
    reset_to: Option<String>,
    /// Every `warm_append` call, in order — the model of the engine's
    /// cumulative token buffer. Distinct from `synced` on purpose: a restored
    /// tier must appear here and *not* there.
    appended: Vec<Option<String>>,
    synced: Vec<Option<String>>,
    restored: Vec<Vec<u8>>,
    supports_kv: bool,
}

#[cfg(test)]
impl std::fmt::Debug for SpyEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SpyEngine")
    }
}

#[cfg(test)]
impl crate::engine::Engine for SpyEngine {
    fn generate(
        &mut self,
        _p: crate::engine::Prompt<'_>,
        _o: &crate::engine::GenerationOptions,
        _i: &dyn Fn() -> bool,
        _g: &dyn Fn() -> bool,
        _e: &mut dyn FnMut(crate::engine::EngineEvent),
    ) -> Result<crate::engine::GenerationStats, crate::engine::EngineError> {
        unreachable!("warm never generates")
    }
    fn ctx_size(&self) -> i32 {
        4096
    }
    fn get_kv(&mut self) -> Option<crate::kvcache::KVCache> {
        if !self.supports_kv {
            return None;
        }
        // The byte is *how far the session has been synced*, not a call
        // counter: a call counter would still increase if every tier were
        // persisted in a deferred second pass, so it could not detect a capture
        // that slipped past its tier's boundary.
        let synced = u8::try_from(self.synced.len()).unwrap_or(u8::MAX);
        Some(crate::kvcache::KVCache::new(
            vec![synced],
            crate::ds4tokens::TokenTranscript::new(),
        ))
    }
    fn set_kv(&mut self, c: &crate::kvcache::KVCache) -> Result<(), crate::engine::EngineError> {
        self.restored.push(c.kv().to_vec());
        Ok(())
    }
    fn warm_reset(&mut self, system: &str) -> Result<(), crate::engine::EngineError> {
        self.reset_to = Some(system.to_owned());
        Ok(())
    }
    fn warm_append(&mut self, text: Option<&str>) -> Result<(), crate::engine::EngineError> {
        self.appended.push(text.map(str::to_owned));
        Ok(())
    }
    fn warm_sync(
        &mut self,
        _e: &mut dyn FnMut(crate::engine::EngineEvent),
    ) -> Result<bool, crate::engine::EngineError> {
        // A sync flushes whatever the matching append just put in the buffer,
        // so record that text — it keeps `synced` readable as "which tiers were
        // actually prefilled".
        self.synced.push(self.appended.last().cloned().flatten());
        Ok(true)
    }
}

#[cfg(test)]
mod warm_tests {
    use super::{TierSpec, plan, system_fingerprint, warm};
    use std::path::Path;

    use super::SpyEngine;

    fn spy_store(name: &str) -> (crate::session::SessionStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("plank-warm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (crate::session::SessionStore::open(&dir).unwrap(), dir)
    }

    fn tiers_for(system: &str, stable: &str, volatile: &str) -> Vec<TierSpec> {
        let fp1 = system_fingerprint("m", system);
        plan(&fp1, system, stable, volatile, "", Some(Path::new("/p")))
    }

    #[test]
    fn cold_warm_prefills_every_tier_and_persists_the_cacheable_ones() {
        let (store, dir) = spy_store("cold");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };

        assert!(warm(&mut e, Some(&store), &tiers, &mut |_| {}).unwrap());
        assert_eq!(e.reset_to.as_deref(), Some("SYSTEM"));
        // System tier syncs with no appended message; the rest append their text.
        assert_eq!(
            e.synced,
            vec![None, Some("agents".into()), Some("git".into())]
        );
        assert!(e.restored.is_empty(), "nothing on disk, nothing to restore");
        // Both cacheable tiers were persisted; the volatile tier was not.
        assert!(store.kv_load(tiers[0].key.as_ref().unwrap()).is_some());
        assert!(store.kv_load(tiers[1].key.as_ref().unwrap()).is_some());
        assert_eq!(tiers[2].key, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The system prompt is the priciest prefix to rebuild, so a Tier 1 miss
    /// must say *what changed* rather than silently re-prefilling: the first
    /// cold warm records the prompt text, and the next warm with a different
    /// prompt diffs against it.
    #[test]
    fn a_changed_system_prompt_reports_the_first_change() {
        let (store, dir) = spy_store("sysdiff");
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };

        // First run: nothing to diff against, so no notice — but the prompt text
        // is recorded for next time.
        let mut notices = Vec::new();
        warm(
            &mut e,
            Some(&store),
            &tiers_for("You are plank. Tools: 7", "", ""),
            &mut |ev| {
                if let crate::engine::EngineEvent::Notice(m) = ev {
                    notices.push(m);
                }
            },
        )
        .unwrap();
        assert!(notices.is_empty(), "first run has nothing to compare");
        assert_eq!(
            store.system_prompt_note().as_deref(),
            Some("You are plank. Tools: 7")
        );

        // Second run, one character different: the notice names the change and
        // carries a -/+ pair the warm-up display colors like a diff card.
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(
            &mut e,
            Some(&store),
            &tiers_for("You are plank. Tools: 8", "", ""),
            &mut |ev| {
                if let crate::engine::EngineEvent::Notice(m) = ev {
                    notices.push(m);
                }
            },
        )
        .unwrap();
        assert_eq!(notices.len(), 1, "one notice for the changed prompt");
        let msg = &notices[0];
        assert!(
            msg.starts_with("system prompt changed; rebuilding cache\n"),
            "{msg}"
        );
        assert!(
            msg.lines()
                .any(|l| l.starts_with("- ") && l.contains("Tools: 7")),
            "{msg}"
        );
        assert!(
            msg.lines()
                .any(|l| l.starts_with("+ ") && l.contains("Tools: 8")),
            "{msg}"
        );
        assert_eq!(
            store.system_prompt_note().as_deref(),
            Some("You are plank. Tools: 8"),
            "the sidecar advances to the prompt just cached"
        );

        // Third run, unchanged: a Tier 1 hit, so no notice at all.
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(
            &mut e,
            Some(&store),
            &tiers_for("You are plank. Tools: 8", "", ""),
            &mut |ev| {
                if let crate::engine::EngineEvent::Notice(m) = ev {
                    notices.push(m);
                }
            },
        )
        .unwrap();
        assert_eq!(notices.len(), 1, "a hit explains nothing");
        assert_eq!(
            e.synced,
            Vec::<Option<String>>::new(),
            "nothing re-prefilled"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_full_hit_restores_the_deepest_tier_and_prefills_only_below_it() {
        let (store, dir) = spy_store("hit");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        // Seed both cacheable tiers, with distinguishable bytes.
        store
            .kv_store(
                tiers[0].key.as_ref().unwrap(),
                &crate::kvcache::KVCache::new(vec![10], crate::ds4tokens::TokenTranscript::new()),
            )
            .unwrap();
        store
            .kv_store(
                tiers[1].key.as_ref().unwrap(),
                &crate::kvcache::KVCache::new(vec![20], crate::ds4tokens::TokenTranscript::new()),
            )
            .unwrap();

        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(&mut e, Some(&store), &tiers, &mut |_| {}).unwrap();

        // Deepest cacheable tier wins: the project checkpoint, not the system one.
        assert_eq!(
            e.restored,
            vec![vec![20]],
            "restore the deepest valid tier only"
        );
        // Only the volatile tier below it is prefilled.
        assert_eq!(e.synced, vec![Some("git".into())]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression: a restored tier is skipped for *prefill* only. Its tokens
    /// must still be appended to the engine's cumulative warm buffer, because
    /// the buffer is what the next `warm_sync` hands the backend. Drop the
    /// restored tiers from it and the backend sees a shorter common prefix,
    /// rewrites its checkpoint from the truncated buffer, and discards the KV
    /// the restore just paid a disk read for — making a deep hit strictly worse
    /// than a cold start.
    #[test]
    fn a_restored_tier_is_still_appended_to_the_token_buffer() {
        let (store, dir) = spy_store("appended");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        for (t, byte) in tiers.iter().zip([10_u8, 20]) {
            store
                .kv_store(
                    t.key.as_ref().unwrap(),
                    &crate::kvcache::KVCache::new(
                        vec![byte],
                        crate::ds4tokens::TokenTranscript::new(),
                    ),
                )
                .unwrap();
        }

        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(&mut e, Some(&store), &tiers, &mut |_| {}).unwrap();

        // Premise: the project tier really was restored, so tiers 0 and 1 are
        // the "skipped" ones this test is about.
        assert_eq!(e.restored, vec![vec![20]], "deepest tier restored");
        assert_eq!(e.synced, vec![Some("git".into())], "only tier 2 prefilled");

        // The point: every tier reached the buffer, restored or not. The system
        // tier appends `None` (its tokens came from `warm_reset`).
        assert_eq!(
            e.appended,
            vec![None, Some("agents".into()), Some("git".into())],
            "restored tiers must still extend the cumulative token buffer"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_deep_checkpoint_falls_back_to_the_shallower_tier() {
        let (store, dir) = spy_store("stale");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        store
            .kv_store(
                tiers[0].key.as_ref().unwrap(),
                &crate::kvcache::KVCache::new(vec![10], crate::ds4tokens::TokenTranscript::new()),
            )
            .unwrap();
        // Corrupt the project checkpoint: right path, unreadable content.
        let key = tiers[1].key.clone().unwrap();
        let crate::session::KvKey::Project { dir: pdir, fp } = &key else {
            unreachable!()
        };
        let cpath = store.project_checkpoint_path(pdir, fp);
        std::fs::create_dir_all(cpath.parent().unwrap()).unwrap();
        std::fs::write(cpath, b"garbage").unwrap();

        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(&mut e, Some(&store), &tiers, &mut |_| {}).unwrap();

        assert_eq!(e.restored, vec![vec![10]], "fall back to the system tier");
        assert_eq!(e.synced, vec![Some("agents".into()), Some("git".into())]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_engine_without_kv_support_prefills_and_persists_nothing() {
        // Remote engines: get_kv returns None. Warming must still prefill every
        // tier and must never write an empty checkpoint.
        let (store, dir) = spy_store("nokv");
        let tiers = tiers_for("SYSTEM", "agents", "");
        let mut e = SpyEngine {
            supports_kv: false,
            ..Default::default()
        };

        warm(&mut e, Some(&store), &tiers, &mut |_| {}).unwrap();
        assert_eq!(e.synced, vec![None, Some("agents".into())]);
        assert!(store.kv_load(tiers[0].key.as_ref().unwrap()).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn each_tier_is_snapshotted_at_its_own_boundary() {
        // The invariant that fingerprints cannot protect: persisting tier i after
        // prefilling tier i+1 would store tier i+1's KV under tier i's key, and the
        // key would be genuinely correct.
        //
        // `SpyEngine::get_kv` keys its byte to `self.synced.len()` — how far the
        // *session* has been prefilled — and that specific choice is what gives
        // this test teeth. A plain get_kv call counter would also increase from
        // tier to tier even if `warm` deferred every capture to a second pass
        // after all syncing, so the assertion below would pass on exactly the
        // implementation it exists to reject. Do not "simplify" that byte into a
        // call counter: it silently disarms this test.
        // Tier 0's stored byte must therefore be strictly less than tier 1's,
        // proving the capture happened before the next tier was synced.
        let (store, dir) = spy_store("boundary");
        let tiers = tiers_for("SYSTEM", "agents", "");
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };

        warm(&mut e, Some(&store), &tiers, &mut |_| {}).unwrap();
        let t0 = store.kv_load(tiers[0].key.as_ref().unwrap()).unwrap();
        let t1 = store.kv_load(tiers[1].key.as_ref().unwrap()).unwrap();
        assert!(
            t0.kv()[0] < t1.kv()[0],
            "each tier must be captured at its own boundary, before the next sync"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
