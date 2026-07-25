// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Volatility-tiered KV cache planning (issues #60, #64).
//!
//! The prompt prefix is a hierarchy of tiers ordered **most stable first**, each
//! an extension checkpoint of the one above it:
//!
//! | Tier | Content | Key | Storage |
//! |------|---------|-----|---------|
//! | 1 | system prompt (+ global MCP tool defs) | `fp1 = sha(model ‖ system)` | `sysprompt-*.kv` (model-global) |
//! | 2 | project-stable context: `AGENTS.md`/`CLAUDE.md` set, memory, **local** MCP tool defs | `fp2 = tier(fp1, stable-hash ‖ local tool defs)` | `<project-key>/project-<fp2>.kv` |
//! | 3 | session-volatile context: git status, date, hook output | — | never cached |
//! | 4 | conversation turns | `tier(fp2, transcript)` | `<session>.payload` |
//!
//! This module owns the **pure** part of that machinery: building the tier list
//! below Tier 1, canonicalizing the local MCP tool definitions that key Tier 2,
//! and deciding — given a "is this checkpoint usable?" probe — which tier to
//! restore from and where prefill must resume. It is always compiled and unit
//! tested; the `#[cfg(ds4_engine)]` engine merely executes the plan it returns
//! (`Engine::warm_tiers`).
//!
//! The walk rule is the one already used for `sysprompt.kv`, generalized:
//! *reuse KV until the first fingerprint mismatch, then prefill only from
//! there*. Because each tier's key embeds its parent's fingerprint, a valid
//! tier implies every ancestor is valid too — so "deepest valid" and "last of
//! the leading valid run" coincide, and a stale checkpoint is rebuilt, never
//! trusted.

use std::path::Path;

use crate::session::KvKey;
use crate::session::tier_fingerprint;

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

/// Where a tier walk lands: which checkpoint to restore, and the first tier
/// that must actually be prefilled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResumePlan {
    /// Index into the tier list of the checkpoint to restore before prefilling,
    /// or `None` when nothing above is reusable and prefill starts at tier 0.
    pub restore: Option<usize>,
    /// Index of the first tier to prefill. Equals `tiers.len()` when every tier
    /// was reusable (nothing to do).
    pub prefill_from: usize,
}

/// Walks `tiers` most-stable-first and decides where to resume.
///
/// `usable` is the caller's probe: "does this tier's checkpoint exist on disk
/// with a matching stored fingerprint?" The walk stops at the first tier that
/// is not cacheable or whose probe fails; the last cacheable tier before that
/// point is the one to restore, and prefill resumes immediately after it.
/// Non-cacheable tiers (Tier 3) therefore terminate the reusable run without
/// invalidating what precedes them.
pub fn resume_point(tiers: &[TierSpec], mut usable: impl FnMut(&TierSpec) -> bool) -> ResumePlan {
    let mut plan = ResumePlan {
        restore: None,
        prefill_from: 0,
    };
    for (i, tier) in tiers.iter().enumerate() {
        if !tier.cacheable() || !usable(tier) {
            break;
        }
        plan.restore = Some(i);
        plan.prefill_from = i + 1;
    }
    plan
}

/// Fingerprint line prefixed to a tier checkpoint file, matching the
/// `sysprompt.kv` layout: `<fingerprint>\n<raw snapshot bytes>`. Returns the
/// stored fingerprint and the snapshot payload, or `None` when `bytes` is not a
/// well-formed checkpoint.
#[must_use]
pub fn split_checkpoint(bytes: &[u8]) -> Option<(&str, &[u8])> {
    let nl = bytes.iter().position(|&b| b == b'\n')?;
    let fp = std::str::from_utf8(&bytes[..nl]).ok()?;
    Some((fp, &bytes[nl + 1..]))
}

/// Whether the checkpoint file at `path` stores exactly `fingerprint`.
/// Best-effort: an unreadable or malformed file is simply "not usable".
#[must_use]
pub fn checkpoint_matches(path: &std::path::Path, fingerprint: &str) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| split_checkpoint(&bytes).map(|(fp, _)| fp == fingerprint))
        .unwrap_or(false)
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
    /// (`warm_tiers`, verbatim) and once per turn, where it has been through
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
    fn resume_point_restores_the_deepest_valid_tier() {
        let tiers = plan(
            "fp1",
            "SYSTEM",
            "agents\n",
            "git\n",
            "",
            Some(Path::new("/p")),
        );
        // Tier 2 hits: restore it, prefill only Tier 3.
        let p = resume_point(&tiers, |t| {
            t.kind == TierKind::System || t.kind == TierKind::ProjectStable
        });
        assert_eq!(
            p,
            ResumePlan {
                restore: Some(1),
                prefill_from: 2
            }
        );
    }

    #[test]
    fn resume_point_falls_back_to_full_prefill_on_the_first_mismatch() {
        let tiers = plan(
            "fp1",
            "SYSTEM",
            "agents\n",
            "git\n",
            "",
            Some(Path::new("/p")),
        );
        let p = resume_point(&tiers, |_| false);
        assert_eq!(
            p,
            ResumePlan {
                restore: None,
                prefill_from: 0
            },
            "a stale Tier 2 rebuilds itself and everything below it"
        );
    }

    #[test]
    fn resume_point_stops_at_the_first_uncacheable_tier() {
        // Tier 3 is never cacheable, so even an always-true probe cannot make
        // the walk claim it as reusable.
        let tiers = plan(
            "fp1",
            "SYSTEM",
            "agents\n",
            "git\n",
            "",
            Some(Path::new("/p")),
        );
        let p = resume_point(&tiers, |_| true);
        assert_eq!(p.restore, Some(1));
        assert_eq!(p.prefill_from, 2, "the volatile tier always prefills");
    }

    #[test]
    fn resume_point_on_an_empty_plan_prefills_nothing() {
        let p = resume_point(&[], |_| true);
        assert_eq!(
            p,
            ResumePlan {
                restore: None,
                prefill_from: 0
            }
        );
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

    #[test]
    fn split_checkpoint_reads_the_fingerprint_line() {
        assert_eq!(split_checkpoint(b"abc\nRAW"), Some(("abc", &b"RAW"[..])));
        assert_eq!(split_checkpoint(b"abc\n"), Some(("abc", &b""[..])));
        assert_eq!(split_checkpoint(b"no-newline"), None);
        assert_eq!(split_checkpoint(b""), None);
    }

    #[test]
    fn checkpoint_matches_only_on_an_exact_stored_fingerprint() {
        let dir = std::env::temp_dir().join(format!("plank-kvtier-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("project-abc.kv");
        std::fs::write(&path, b"abc\nsnapshot-bytes").unwrap();
        assert!(checkpoint_matches(&path, "abc"));
        assert!(!checkpoint_matches(&path, "abd"));
        assert!(!checkpoint_matches(&dir.join("missing.kv"), "abc"));
        // Malformed (no fingerprint line) is never trusted.
        std::fs::write(&path, b"garbage").unwrap();
        assert!(!checkpoint_matches(&path, "garbage"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
