// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Volatility-tiered KV cache planning (issues #60, #64).
//!
//! The prompt prefix is a hierarchy of tiers ordered **most stable first**, each
//! an extension checkpoint of the one above it:
//!
//! | Tier | Content | Key | Storage |
//! |------|---------|-----|---------|
//! | 1 | system prompt (+ global MCP tool defs, incl. cached ones) | `fp1 = sha(model ‖ system)` | `sysprompt-<fp1>.kv_raw` (model-global) |
//! | 2 | project-stable context: `AGENTS.md`/`CLAUDE.md` set, memory, **local** MCP tool defs | `fp2 = tier(fp1, stable-hash ‖ local tool defs)` | `<project-key>/project-<fp2>.kv_raw` |
//! | 3 | session-volatile context: git status, date, hook output | — | never cached |
//! | 4 | conversation turns | `tier(fp2, transcript)` | `<session>.kv_raw` |
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

use crate::engine::{Engine, EngineError, EngineEvent, ThinkMode};
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

impl TierKind {
    /// What a front end should say while *this* tier is being prefilled, so the
    /// progress note names the stage actually running instead of always
    /// claiming the system prompt is being rebuilt. No trailing ellipsis: the
    /// caller appends the one its output style uses.
    #[must_use]
    pub fn warm_label(self) -> &'static str {
        match self {
            Self::System => "Updating system prompt cache",
            Self::ProjectStable => "Updating project context cache",
            Self::SessionVolatile => "Updating session context",
        }
    }
}

/// One tier of the prefix: the text it contributes, its chained fingerprint,
/// and where (if anywhere) its KV checkpoint lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierSpec {
    /// Which tier this is.
    pub kind: TierKind,
    /// Chained fingerprint `sha(parent ‖ NUL ‖ material)`.
    pub fingerprint: String,
    /// Fingerprint of the tier this one extends, or `None` for Tier 1.
    ///
    /// Redundant with `fingerprint` in principle — a chained fingerprint
    /// already embeds its parent's — but a hash cannot be inverted, so the
    /// lineage a `/kvcache` tree draws has to be recorded explicitly.
    pub parent: Option<String>,
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

/// Tier 1's fingerprint: `sha1(model ‖ NUL ‖ think ‖ NUL ‖ trusted_len ‖ NUL ‖ system)`.
///
/// The single definition of the system-prompt checkpoint key, shared by the
/// engine (which writes `sysprompt-*.kv_raw` with it) and by the tier planner
/// (which chains `fp2` off it). Keeping one implementation is what guarantees
/// the two never disagree and silently invalidate the whole chain.
///
/// The reasoning level is key material because [`ThinkMode::Max`] prepends the
/// reasoning-effort preamble *ahead of* the system prompt: two levels give the
/// same system text a different token prefix, and a checkpoint keyed on the
/// text alone would be restored under the wrong one.
///
/// `trusted_len` is key material for the same reason, one step subtler: it does
/// not change the prompt's *text* at all, only where the tokenizer stops
/// treating it as control text (`sysprompt::SplitSystemPrompt::trusted_len`).
/// Two builds that split differently render byte-identical prompts to
/// byte-*different* token streams, so without this a checkpoint from one would
/// be restored under the other and prefill from a KV that does not match.
#[must_use]
pub fn system_fingerprint(
    model: &str,
    system: &str,
    think: ThinkMode,
    trusted_len: usize,
) -> String {
    let mut data = model.as_bytes().to_vec();
    data.push(0);
    data.extend_from_slice(think.name().as_bytes());
    data.push(0);
    data.extend_from_slice(trusted_len.to_string().as_bytes());
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

/// Display material for the metadata sidecars [`warm`] writes.
///
/// Purely cosmetic — none of it is key material, and changing any field must
/// never change a fingerprint. It exists because `plan` derives tiers from an
/// already-hashed `fp1` and so cannot name the model, reasoning level, or MCP
/// servers behind the prefix it is planning.
#[derive(Debug, Clone, Default)]
pub struct TierLabels {
    /// Reasoning level the system prompt was fingerprinted under.
    pub think_mode: String,
    /// Byte offset where trusted control text stops in the system prompt.
    pub trusted_len: usize,
    /// Global MCP servers whose tool defs entered Tier 1.
    pub global_mcp: Vec<String>,
    /// Absolute path of the project Tier 2 belongs to.
    pub project_path: String,
    /// `AGENTS.md`/`CLAUDE.md` files that fed Tier 2's stable context.
    pub agents_files: Vec<String>,
    /// Project-local MCP servers whose tool defs entered Tier 2.
    pub local_mcp: Vec<String>,
}

/// The `/kvcache` display label for a tier.
///
/// The MCP names are split by tier on purpose: a global server's tool defs are
/// Tier 1 material and a project-local server's are Tier 2, so a local name
/// appearing on a system label would mean the segregation this module enforces
/// had broken.
#[must_use]
fn tier_label(kind: TierKind, labels: &TierLabels) -> crate::kvmeta::KvLabel {
    match kind {
        TierKind::System => crate::kvmeta::KvLabel::System {
            think_mode: labels.think_mode.clone(),
            trusted_len: labels.trusted_len,
            global_mcp: labels.global_mcp.clone(),
        },
        TierKind::ProjectStable => crate::kvmeta::KvLabel::Project {
            project_path: labels.project_path.clone(),
            agents_files: labels.agents_files.clone(),
            local_mcp: labels.local_mcp.clone(),
        },
        // Tier 3 is never cacheable, so this arm is unreachable through `warm`'s
        // store path; it exists so the match is total.
        TierKind::SessionVolatile => crate::kvmeta::KvLabel::Unknown,
    }
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
        parent: None,
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
            parent: Some(fp1.to_owned()),
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
            // `fp2` has already collapsed to `fp1` when there is no Tier 2, so
            // this names Tier 2 when one exists and Tier 1 when it does not —
            // never a tier absent from the list.
            parent: Some(fp2.clone()),
            text: volatile.to_owned(),
            key: None,
        });
    }
    tiers
}

/// Why a [`restore`] did or did not put a tier's KV into an engine.
///
/// A miss has three distinct causes with three different fixes, and "it
/// prefilled again" cannot tell them apart from the outside — hence an enum
/// rather than a bool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restored {
    /// The checkpoint loaded and the engine now holds its KV.
    Yes,
    /// The tier is not cacheable, or there is no session store.
    NoKey,
    /// Nothing on disk under this tier's key. Either nothing ever warmed this
    /// exact fingerprint, or something about the prompt changed since.
    NoCheckpoint,
    /// The file is there and keyed correctly but would not load — a stale
    /// format or a signature mismatch. Rebuilt, never trusted.
    Unreadable,
    /// The bytes loaded but the engine refused them.
    EngineRefused,
}

impl Restored {
    /// One line for a diagnostic, in the imperative of what happened.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Self::Yes => "restored from checkpoint",
            Self::NoKey => "tier is not cacheable",
            Self::NoCheckpoint => "no checkpoint on disk for this fingerprint",
            Self::Unreadable => "checkpoint present but would not load",
            Self::EngineRefused => "engine refused the checkpoint",
        }
    }
}

/// Instruments one tier's restore decision, gated behind `PLANK_DEBUG_SYSPROMPT`.
///
/// The investigation aid for cache churn: two launches leave two diffable
/// prompt dumps plus a running record of which tier resumed and why. It was
/// lost when the tier walk replaced `warm_system_prompt` — the one path it hung
/// off — and its absence is felt every time a "why did it re-prefill?" report
/// arrives, because [`Restored`] only says what happened to the tier that was
/// asked about, and only in memory.
///
/// The system tier's *text* is dumped, keyed by the first 8 of its fingerprint,
/// so two launches that disagree can be diffed byte-for-byte. Best-effort
/// throughout: this must never fail a warm-up.
fn debug_trace_tier(store: Option<&SessionStore>, tier: &TierSpec, outcome: &str) {
    if std::env::var_os("PLANK_DEBUG_SYSPROMPT").is_none() {
        return;
    }
    let fp = tier.key.as_ref().map_or("", KvKey::signature);
    let fp8 = &fp[..fp.len().min(8)];
    let dump = store.map(|s| {
        let kind = format!("{:?}", tier.kind).to_ascii_lowercase();
        let path = s.dir().join(format!("sysprompt-debug-{kind}-{fp8}.txt"));
        let _ = std::fs::write(&path, &tier.text);
        path
    });
    let line = format!(
        "pid={} tier={:?} outcome={outcome} fp={fp} text_len={} dump={}\n",
        std::process::id(),
        tier.kind,
        tier.text.len(),
        dump.as_ref()
            .map_or_else(String::new, |p| p.display().to_string()),
    );
    if let Some(store) = store {
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(store.dir().join("sysprompt-debug.log"))
            .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
    }
    eprint!("[sysprompt-debug] {line}");
}

/// Restores `tier`'s checkpoint into `engine`, or leaves the engine cold.
///
/// The restore leg of [`warm`] without its prefill leg, for callers that must
/// not pay a cold prefill *here*. `warm_sync` prefills uninterruptibly
/// (`interrupt: &|| false`) and reports progress only through the sink it is
/// given, so a caller with no progress sink — or one holding a thread the UI
/// draws on — turns a cold cache into a freeze with no output and no way to
/// cancel. Skipping it leaves the tokens to whichever pass needs them first,
/// where the ordinary prefill bar and the interrupt already work.
///
/// The engine is left untouched on a miss: `warm_reset` runs only once a
/// checkpoint is in hand, so a cold engine keeps an empty warm buffer rather
/// than one describing a prefix its KV does not hold.
///
/// # Errors
/// Returns [`EngineError`] only when the engine itself fails. A missing or
/// unloadable checkpoint is a miss, not an error.
pub fn restore(
    engine: &mut dyn Engine,
    store: Option<&SessionStore>,
    tier: &TierSpec,
) -> Result<Restored, EngineError> {
    let Some((key, store)) = tier.key.as_ref().zip(store) else {
        debug_trace_tier(None, tier, Restored::NoKey.reason());
        return Ok(Restored::NoKey);
    };
    let Some(cache) = store.kv_load(key) else {
        let outcome = if store.kv_path(key).exists() {
            Restored::Unreadable
        } else {
            Restored::NoCheckpoint
        };
        debug_trace_tier(Some(store), tier, outcome.reason());
        return Ok(outcome);
    };
    engine.warm_reset(&tier.text)?;
    if engine.set_kv(&cache).is_err() {
        debug_trace_tier(Some(store), tier, Restored::EngineRefused.reason());
        return Ok(Restored::EngineRefused);
    }
    debug_trace_tier(Some(store), tier, Restored::Yes.reason());
    // Same reason as the `i < resume` branch in `warm`: the engine's cumulative
    // token buffer must describe the whole restored prefix, or the next sync
    // sees a common prefix shorter than the buffer and throws the restored KV
    // away. The system tier's tokens are already there from `warm_reset`.
    engine.warm_append(None)?;
    Ok(Restored::Yes)
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
/// `on_stage` is called with each tier's kind immediately before that tier is
/// prefilled (never for a restored tier), so a front end can name the stage its
/// progress bar is actually covering — see [`TierKind::warm_label`].
///
/// `labels` supplies the human-readable display material recorded in each
/// persisted tier's metadata sidecar (see [`TierLabels`]). It is cosmetic: no
/// part of it is key material, so a wrong or empty field can never invalidate a
/// checkpoint.
///
/// # Errors
/// Returns [`EngineError`] only when the engine itself fails to restore or
/// prefill. Cache IO failures are best-effort and never propagate.
pub fn warm(
    engine: &mut dyn Engine,
    store: Option<&SessionStore>,
    tiers: &[TierSpec],
    on_event: &mut dyn FnMut(EngineEvent),
    on_stage: &mut dyn FnMut(TierKind),
    labels: &TierLabels,
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
            debug_trace_tier(store, t, "no checkpoint loaded for this fingerprint");
            continue;
        };
        if engine.set_kv(&cache).is_ok() {
            debug_trace_tier(store, t, "resumed here");
            resume = i + 1;
            break;
        }
        debug_trace_tier(store, t, "engine refused the checkpoint");
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
        on_stage(t.kind);
        // Warming is where a cold session does its heaviest prefill — the
        // system prompt and project context — so most of a session's prefill
        // time is spent here rather than in a turn's own short suffix.
        // Recorded at the one chokepoint every front-end shares.
        let model = engine.model_name();
        // Buffered rather than recorded live: a tier served from a KV
        // checkpoint still reports its token count against an elapsed of
        // almost nothing, which reads as thousands of tok/s — a number no
        // model produces, and one that would skew the average. `warm_sync`
        // only says whether it really prefilled once it returns.
        let mut samples: Vec<(i32, f64)> = Vec::new();
        let did_prefill = engine.warm_sync(&mut |ev| {
            if let EngineEvent::Prefill(p) = &ev {
                samples.push((p.done, p.tps));
            }
            on_event(ev);
        })?;
        if did_prefill {
            for (done, tps) in samples {
                crate::speeds::note_prefill_progress(&model, done, tps);
            }
        }
        prefilled |= did_prefill;
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
            let _ = store.kv_store_labeled(
                key,
                &cache,
                t.parent.as_deref(),
                &model,
                &tier_label(t.kind, labels),
            );
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
        let fp1 = system_fingerprint("model-a", "SYSTEM", crate::engine::ThinkMode::default(), 0);
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
        let fp1 = system_fingerprint("m", "SYSTEM", crate::engine::ThinkMode::default(), 0);
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
            agents_content: None,
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
    fn system_fingerprint_is_model_prompt_level_and_split_keyed() {
        use ThinkMode::{Max, Medium, Off};
        let a = system_fingerprint("model-a", "sys", Medium, 0);
        assert_eq!(a, system_fingerprint("model-a", "sys", Medium, 0));
        assert_ne!(a, system_fingerprint("model-b", "sys", Medium, 0));
        assert_ne!(a, system_fingerprint("model-a", "sys2", Medium, 0));
        // The reasoning level keys the tier: `max` prepends the effort preamble
        // ahead of the same system text, so its checkpoint must not be reused
        // for another level.
        assert_ne!(a, system_fingerprint("model-a", "sys", Max, 0));
        assert_ne!(a, system_fingerprint("model-a", "sys", Off, 0));
        assert_ne!(
            system_fingerprint("model-a", "sys", Max, 0),
            system_fingerprint("model-a", "sys", Off, 0)
        );
        // The NUL separators keep the three fields from bleeding together.
        assert_ne!(
            system_fingerprint("ab", "c", Medium, 0),
            system_fingerprint("a", "bc", Medium, 0)
        );
        // The tokenization split keys it too: identical text, different tokens.
        assert_ne!(a, system_fingerprint("model-a", "sys", Medium, 3));

        let mut expect = b"model-a".to_vec();
        expect.push(0);
        expect.extend_from_slice(b"medium");
        expect.push(0);
        expect.extend_from_slice(b"0");
        expect.push(0);
        expect.extend_from_slice(b"sys");
        assert_eq!(a, crate::session::sha1_hex(&expect));
    }

    #[test]
    fn each_tier_records_its_parent_fingerprint() {
        let tiers = plan(
            "fp1-system",
            "system text",
            "stable context",
            "volatile context",
            "local defs",
            Some(std::path::Path::new("/tmp/plank-lineage")),
        );
        let sys = tiers
            .iter()
            .find(|t| t.kind == TierKind::System)
            .expect("plan always emits Tier 1 first");
        assert!(sys.parent.is_none(), "the system tier is the root");
        let proj = tiers
            .iter()
            .find(|t| t.kind == TierKind::ProjectStable)
            .expect("a project tier exists when there is stable text");
        assert_eq!(proj.parent.as_deref(), Some("fp1-system"));
        let vol = tiers
            .iter()
            .find(|t| t.kind == TierKind::SessionVolatile)
            .expect("a volatile tier exists when there is volatile text");
        assert_eq!(
            vol.parent.as_deref(),
            Some(proj.fingerprint.as_str()),
            "Tier 3 chains off Tier 2, not off Tier 1"
        );
    }

    #[test]
    fn the_volatile_tier_chains_off_tier_1_when_there_is_no_stable_text() {
        // `plan` collapses fp2 to fp1 when `stable` is empty and emits no Tier 2,
        // so Tier 3's recorded parent must follow that collapse rather than naming
        // a tier that does not exist.
        let tiers = plan("fp1-system", "system text", "", "volatile", "", None);
        assert!(
            !tiers.iter().any(|t| t.kind == TierKind::ProjectStable),
            "no stable text means no Tier 2"
        );
        let vol = tiers
            .iter()
            .find(|t| t.kind == TierKind::SessionVolatile)
            .expect("volatile tier present");
        assert_eq!(vol.parent.as_deref(), Some("fp1-system"));
    }

    #[test]
    fn a_local_mcp_name_never_reaches_a_system_label() {
        // The audit surface for MCP segregation: global tool defs are Tier 1
        // material, project-local ones are Tier 2. A local name on a system label
        // would mean that split had broken.
        let labels = TierLabels {
            think_mode: "max".into(),
            trusted_len: 12,
            global_mcp: vec!["global-srv".into()],
            project_path: "/Users/x/Code/plank".into(),
            agents_files: vec!["AGENTS.md".into()],
            local_mcp: vec!["local-srv".into()],
        };
        match tier_label(TierKind::System, &labels) {
            crate::kvmeta::KvLabel::System {
                think_mode,
                trusted_len,
                global_mcp,
            } => {
                assert_eq!(think_mode, "max");
                assert_eq!(trusted_len, 12);
                assert_eq!(global_mcp, vec!["global-srv".to_owned()]);
                assert!(
                    !global_mcp.iter().any(|n| n == "local-srv"),
                    "a project-local server must never appear on a system label"
                );
            }
            other => panic!("expected a System label, got {other:?}"),
        }
        match tier_label(TierKind::ProjectStable, &labels) {
            crate::kvmeta::KvLabel::Project {
                project_path,
                agents_files,
                local_mcp,
            } => {
                assert_eq!(project_path, "/Users/x/Code/plank");
                assert_eq!(agents_files, vec!["AGENTS.md".to_owned()]);
                assert_eq!(local_mcp, vec!["local-srv".to_owned()]);
                assert!(
                    !local_mcp.iter().any(|n| n == "global-srv"),
                    "a global server must never appear on a project label"
                );
            }
            other => panic!("expected a Project label, got {other:?}"),
        }
        assert!(matches!(
            tier_label(TierKind::SessionVolatile, &labels),
            crate::kvmeta::KvLabel::Unknown
        ));
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
    /// When true `set_kv` fails, standing in for a checkpoint this build cannot
    /// load — keyed correctly, but not restorable.
    refuse_kv: bool,
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
        if self.refuse_kv {
            return Err(crate::engine::EngineError::new("refused"));
        }
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
    use super::{
        Restored, TierKind, TierLabels, TierSpec, plan, restore, system_fingerprint, warm,
    };
    use std::path::Path;

    use super::SpyEngine;

    fn spy_store(name: &str) -> (crate::session::SessionStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("plank-warm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (crate::session::SessionStore::open(&dir).unwrap(), dir)
    }

    fn tiers_for(system: &str, stable: &str, volatile: &str) -> Vec<TierSpec> {
        let fp1 = system_fingerprint("m", system, crate::engine::ThinkMode::default(), 0);
        plan(&fp1, system, stable, volatile, "", Some(Path::new("/p")))
    }

    /// `restore` is the restore leg alone. On a miss it must leave the engine
    /// completely untouched — no `warm_reset` — so a cold engine keeps an empty
    /// warm buffer instead of one describing a prefix its KV does not hold, and
    /// the pass that needs those tokens prefills them itself.
    #[test]
    fn restore_leaves_a_cold_engine_untouched_and_never_prefills() {
        let (store, dir) = spy_store("restore-miss");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };

        assert_eq!(
            restore(&mut e, Some(&store), &tiers[0]).unwrap(),
            Restored::NoCheckpoint,
            "a miss, and it says which kind"
        );
        assert_eq!(e.reset_to, None, "the engine was not touched at all");
        assert!(e.appended.is_empty());
        assert!(e.synced.is_empty(), "restore never prefills");

        // Seed the checkpoint the way a local-main session would, then re-run:
        // now it restores, still without prefilling.
        warm(
            &mut e,
            Some(&store),
            &tiers[..1],
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        assert_eq!(
            restore(&mut e, Some(&store), &tiers[0]).unwrap(),
            Restored::Yes,
            "a hit"
        );
        assert_eq!(e.reset_to.as_deref(), Some("SYSTEM"));
        assert_eq!(e.restored.len(), 1);
        // The buffer describes the restored prefix, or the next sync would
        // throw the restored KV away.
        assert_eq!(e.appended, vec![None]);
        assert!(e.synced.is_empty(), "restore never prefills");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `PLANK_DEBUG_SYSPROMPT` trace hung off `warm_system_prompt` and was
    /// lost when the tier walk replaced that path, so the one tool for
    /// diagnosing cache churn silently did nothing. It now covers every tier
    /// decision on both legs of the walk.
    #[test]
    fn the_sysprompt_debug_trace_records_every_tier_decision() {
        let (store, dir) = spy_store("sysprompt-debug");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        // SAFETY: the trace is read through `var_os` only, and no other test
        // touches this variable.
        unsafe { std::env::set_var("PLANK_DEBUG_SYSPROMPT", "1") };
        // Cold: every tier misses. Then warm again, so the second walk resumes
        // from a checkpoint the first one wrote.
        for _ in 0..2 {
            warm(
                &mut e,
                Some(&store),
                &tiers,
                &mut |_| {},
                &mut |_| {},
                &TierLabels::default(),
            )
            .unwrap();
        }
        unsafe { std::env::remove_var("PLANK_DEBUG_SYSPROMPT") };

        let log = std::fs::read_to_string(dir.join("sysprompt-debug.log")).expect("no debug log");
        assert!(
            log.contains("outcome=no checkpoint loaded for this fingerprint"),
            "the cold walk's misses are not recorded: {log}"
        );
        assert!(
            log.contains("outcome=resumed here"),
            "the second walk's resume point is not recorded: {log}"
        );
        assert!(log.contains("tier=System"), "{log}");
        // The prompt text itself is dumped, keyed by fingerprint, so two
        // launches that disagree can be diffed.
        let dumps: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("sysprompt-debug-system-"))
            .collect();
        assert_eq!(dumps.len(), 1, "{dumps:?}");
        assert_eq!(
            std::fs::read_to_string(dir.join(&dumps[0])).unwrap(),
            "SYSTEM"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The system checkpoint is what a sub-agent sidechain restores, and a
    /// one-tier walk is the only thing that writes it — so assert the *file*,
    /// not just that a prefill happened. Nothing else in the suite would notice
    /// if `warm` stopped persisting Tier 1.
    #[test]
    fn a_one_tier_warm_persists_the_system_checkpoint() {
        let (store, dir) = spy_store("tier1-persist");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };

        assert!(
            warm(
                &mut e,
                Some(&store),
                &tiers[..1],
                &mut |_| {},
                &mut |_| {},
                &TierLabels::default()
            )
            .unwrap()
        );

        let key = tiers[0].key.as_ref().expect("system tier is cacheable");
        assert!(
            store.kv_path(key).exists(),
            "the checkpoint a sidechain will look for: {}",
            store.kv_path(key).display()
        );
        assert!(store.kv_load(key).is_some(), "and it loads back");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The toll this whole area exists to avoid: a second launch on the same
    /// fingerprint must restore, not prefill. Two separate engines, because a
    /// launch is a fresh process — carrying state over in one engine would pass
    /// while the real thing still re-prefilled.
    #[test]
    fn a_second_launch_restores_the_system_tier_instead_of_prefilling() {
        let (store, dir) = spy_store("tier1-second-launch");
        let tiers = tiers_for("SYSTEM", "agents", "git");

        let mut first = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(
            &mut first,
            Some(&store),
            &tiers[..1],
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();
        assert_eq!(first.synced, vec![None], "the first launch pays for it");

        let mut second = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        let prefilled = warm(
            &mut second,
            Some(&store),
            &tiers[..1],
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();

        assert!(!prefilled, "the second launch prefills nothing");
        assert!(second.synced.is_empty(), "{:?}", second.synced);
        assert_eq!(second.restored.len(), 1, "it restored instead");
        // And the buffer still describes the restored prefix, or the first sync
        // of the turn would throw the restored KV away.
        assert_eq!(second.appended, vec![None]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Known gap, pinned deliberately: a valid deep tier means the tier above
    /// it is never prefilled and so never written. Correct for the main engine,
    /// which restores the superset — but it means Tier 1 can stay absent
    /// forever, which is why the sub-agent engine warms a one-tier list of its
    /// own. If this ever starts passing a checkpoint for Tier 1, that is a
    /// deliberate improvement and this test should be rewritten, not deleted.
    #[test]
    fn a_valid_deep_tier_leaves_the_system_checkpoint_unwritten() {
        let (store, dir) = spy_store("tier1-skipped");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        // Seed Tier 2 the way a previous launch would have, without Tier 1.
        let mut seed = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(
            &mut seed,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();
        let sys_key = tiers[0].key.as_ref().unwrap();
        let _ = std::fs::remove_file(store.kv_path(sys_key));
        assert!(!store.kv_path(sys_key).exists());

        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(
            &mut e,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();

        assert!(
            !store.kv_path(sys_key).exists(),
            "Tier 2 restored, so Tier 1 was skipped and never written"
        );
        assert!(e.synced.iter().all(|t| t.as_deref() != Some("SYSTEM")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A checkpoint that is present but will not load is a distinct outcome
    /// from one that is absent: the first says "something is wrong with this
    /// file", the second "nothing ever wrote one". Conflating them sent this
    /// investigation down the wrong path once already.
    #[test]
    fn restore_distinguishes_unreadable_from_absent_and_refused() {
        let (store, dir) = spy_store("restore-outcomes");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        let key = tiers[0].key.as_ref().unwrap();
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };

        assert_eq!(
            restore(&mut e, Some(&store), &tiers[0]).unwrap(),
            Restored::NoCheckpoint
        );
        // No store at all is the same class of miss, reported separately.
        assert_eq!(restore(&mut e, None, &tiers[0]).unwrap(), Restored::NoKey);
        // Tier 3 has no key even with a store.
        assert_eq!(
            restore(&mut e, Some(&store), &tiers[2]).unwrap(),
            Restored::NoKey
        );

        // Present, correctly named, and garbage inside.
        std::fs::write(store.kv_path(key), b"not a checkpoint").unwrap();
        assert_eq!(
            restore(&mut e, Some(&store), &tiers[0]).unwrap(),
            Restored::Unreadable
        );
        assert_eq!(e.reset_to, None, "and the engine is still untouched");

        // Readable, but this engine will not take it.
        warm(
            &mut e,
            Some(&store),
            &tiers[..1],
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();
        let mut fussy = SpyEngine {
            supports_kv: true,
            refuse_kv: true,
            ..Default::default()
        };
        assert_eq!(
            restore(&mut fussy, Some(&store), &tiers[0]).unwrap(),
            Restored::EngineRefused
        );
        assert!(
            fussy.synced.is_empty(),
            "a refusal never falls back to prefill"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cold_warm_prefills_every_tier_and_persists_the_cacheable_ones() {
        let (store, dir) = spy_store("cold");
        let tiers = tiers_for("SYSTEM", "agents", "git");
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };

        assert!(
            warm(
                &mut e,
                Some(&store),
                &tiers,
                &mut |_| {},
                &mut |_| {},
                &TierLabels::default()
            )
            .unwrap()
        );
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

    /// The warm progress note must name the tier actually prefilling: a launch
    /// that only has to redo the cheap session context should not claim it is
    /// rebuilding the system prompt cache. Restored tiers report nothing,
    /// because nothing is being rebuilt for them.
    #[test]
    fn stages_are_reported_only_for_the_tiers_that_prefill() {
        let (store, dir) = spy_store("stages");
        let tiers = tiers_for("SYSTEM", "agents", "git");

        let mut cold = Vec::new();
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(
            &mut e,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |k| cold.push(k),
            &TierLabels::default(),
        )
        .unwrap();
        assert_eq!(
            cold,
            vec![
                TierKind::System,
                TierKind::ProjectStable,
                TierKind::SessionVolatile
            ]
        );
        assert_eq!(
            TierKind::System.warm_label(),
            "Updating system prompt cache",
            "the label the C-era note used, kept for the Tier 1 case"
        );

        // Second launch: the cold run persisted both cacheable tiers, so only
        // the never-cached volatile tier is a stage now.
        let mut warm_stages = Vec::new();
        let mut e = SpyEngine {
            supports_kv: true,
            ..Default::default()
        };
        warm(
            &mut e,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |k| warm_stages.push(k),
            &TierLabels::default(),
        )
        .unwrap();
        assert_eq!(warm_stages, vec![TierKind::SessionVolatile]);
        assert_ne!(
            TierKind::SessionVolatile.warm_label(),
            TierKind::System.warm_label()
        );
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
            &mut |_| {},
            &TierLabels::default(),
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
            &mut |_| {},
            &TierLabels::default(),
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
            &mut |_| {},
            &TierLabels::default(),
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
        warm(
            &mut e,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();

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
        warm(
            &mut e,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();

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
        warm(
            &mut e,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();

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

        warm(
            &mut e,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();
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

        warm(
            &mut e,
            Some(&store),
            &tiers,
            &mut |_| {},
            &mut |_| {},
            &TierLabels::default(),
        )
        .unwrap();
        let t0 = store.kv_load(tiers[0].key.as_ref().unwrap()).unwrap();
        let t1 = store.kv_load(tiers[1].key.as_ref().unwrap()).unwrap();
        assert!(
            t0.kv()[0] < t1.kv()[0],
            "each tier must be captured at its own boundary, before the next sync"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
