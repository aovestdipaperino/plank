// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `/repro` — dump the current session to disk for bug diagnosis.
//!
//! A repro file captures the exact input the engine would see (the rendered
//! `[system]`/`[user]`/`[assistant]` prompt) plus the runtime knobs that shape
//! generation (model, backend, context size, sampling, think mode, engine
//! tuning). It is a self-contained artifact: hand it to a maintainer and they
//! can reproduce the state that triggered a bug without the live session.
//!
//! Files land in `~/.plank/repro/` (or the working dir when `HOME` is unset),
//! named `repro-<unix-seconds>[-<n>].md`. When the repetition guard stops a
//! looping pass the agent writes one automatically as
//! `repro-loop-<unix-seconds>[-<n>].md`, so a stall is captured without
//! anyone having to notice it. Nothing here touches the live session — it is
//! a read-only snapshot.
//!
//! Sub-agent sidechains are folded out of the transcript the moment they end,
//! so the main dump never shows what a sub-agent did. The agent keeps the last
//! [`SIDECHAIN_DUMPS_KEPT`] finished sidechains ([`SidechainDump`]) and `/repro`
//! writes each one beside the main file as `repro-<secs>.sub-<n>.md`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::config::AgentConfig;

/// Runtime facts worth recording alongside the transcript, gathered from the
/// live `Agent` by the caller (which owns the engine and config).
#[derive(Debug)]
pub struct Meta<'a> {
    /// plank version string.
    pub version: &'a str,
    /// Local ISO date/time the repro was taken.
    pub date: &'a str,
    /// Engine context window size (tokens).
    pub ctx_size: i32,
    /// Tokens the rendered transcript occupies, per the engine tokenizer.
    pub transcript_tokens: i32,
    /// KV position reported after the last generation (0 if none yet).
    pub last_ctx_used: i32,
    /// GPU power cap percent in effect.
    pub power_percent: i32,
    /// Reasoning level in effect. Carried on the meta rather than read from
    /// the config because `/think` can change it after launch.
    pub think: crate::engine::ThinkMode,
    /// Session identity SHA (empty when never saved).
    pub session_id: &'a str,
    /// Session tag (empty when unset).
    pub session_tag: &'a str,
    /// Optional user note describing the bug.
    pub note: &'a str,
}

/// How many finished sub-agent sidechains the agent remembers for `/repro`.
/// Oldest are dropped first; a fan-out of N slots contributes N entries.
pub const SIDECHAIN_DUMPS_KEPT: usize = 8;

/// One finished sub-agent sidechain, kept for the `/repro` sidecar. Images are
/// stripped from the messages: the embeddings are large and cannot be shown in
/// a text dump anyway.
#[derive(Debug, Clone)]
pub struct SidechainDump {
    /// Roster label: the agent name, or `sub-agent`.
    pub label: String,
    /// The plain delegated task (not the framed envelope).
    pub task: String,
    /// How it ended: `report`, `no report`, or `failed: <error>`.
    pub outcome: String,
    /// Transcript index in the parent where the sidechain was forked, so the
    /// sidecar can be lined up against the main dump.
    pub fork_at: usize,
    /// The sidechain's messages, framed task first, in order.
    pub messages: Vec<crate::session::Message>,
}

impl SidechainDump {
    /// Builds a dump from the sidechain's messages, dropping image payloads.
    #[must_use]
    pub fn new(
        label: &str,
        task: &str,
        fork_at: usize,
        messages: &[crate::session::Message],
    ) -> Self {
        Self {
            label: label.to_owned(),
            task: task.to_owned(),
            outcome: "ended".to_owned(),
            fork_at,
            messages: messages
                .iter()
                .map(|m| crate::session::Message {
                    role: m.role,
                    text: m.text.clone(),
                    at: m.at,
                    images: Vec::new(),
                })
                .collect(),
        }
    }
}

/// Builds a sidecar report for one sidechain: a header naming the main dump it
/// belongs to, the sub-agent's label, task and outcome, and the sidechain's
/// rendered messages between the same fences the main report uses. The system
/// prompt is the parent's and is not repeated; the header says so.
#[must_use]
pub fn build_sidecar_report(
    version: &str,
    main_file: &str,
    ordinal: usize,
    dump: &SidechainDump,
    rendered_messages: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# plank repro {version} — sub-agent sidecar {ordinal}");
    let _ = writeln!(out);
    let _ = writeln!(out, "- main repro: {main_file}");
    let _ = writeln!(out, "- label: {}", dump.label);
    let _ = writeln!(out, "- outcome: {}", dump.outcome);
    let _ = writeln!(out, "- forked at parent message: {}", dump.fork_at);
    let _ = writeln!(out, "- messages: {}", dump.messages.len());
    let _ = writeln!(
        out,
        "- system prompt: identical to the main repro (not repeated)"
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "## Task");
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", dump.task);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "## Sidechain transcript (after the shared system prompt)"
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "----- BEGIN TRANSCRIPT -----");
    out.push_str(rendered_messages);
    if !rendered_messages.ends_with('\n') {
        out.push('\n');
    }
    let _ = writeln!(out, "----- END TRANSCRIPT -----");
    out
}

/// The path of sidecar `ordinal` for the main dump at `main`: the main file's
/// stem plus `.sub-<ordinal>.md`, in the same directory.
#[must_use]
pub fn sidecar_path(main: &Path, ordinal: usize) -> PathBuf {
    let stem = main
        .file_stem()
        .map_or_else(|| "repro".to_owned(), |s| s.to_string_lossy().into_owned());
    main.with_file_name(format!("{stem}.sub-{ordinal}.md"))
}

/// Writes sidecar `ordinal` beside `main`.
///
/// # Errors
/// Returns the OS error message when the file cannot be written.
pub fn save_sidecar(main: &Path, ordinal: usize, report: &str) -> Result<PathBuf, String> {
    let path = sidecar_path(main, ordinal);
    std::fs::write(&path, report).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Directory repro files are written to (`~/.plank/repro`, or `<cwd>/.plank/
/// repro` when `HOME` is unset).
#[must_use]
pub fn repro_dir(cwd: &Path) -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || cwd.join(".plank").join("repro"),
        |h| PathBuf::from(h).join(".plank").join("repro"),
    )
}

/// Builds the repro report text: a metadata header, the config that shapes
/// generation, and the verbatim rendered transcript (the exact engine input).
///
/// The transcript is emitted between explicit `BEGIN`/`END` fences rather than
/// a markdown code block, because it can itself contain triple-backtick code
/// and must survive round-tripping byte-for-byte.
#[must_use]
pub fn build_report(meta: &Meta, cfg: &AgentConfig, rendered_transcript: &str) -> String {
    let g = &cfg.generation;
    let mut out = String::new();
    let _ = writeln!(out, "# plank repro {}", meta.version);
    let _ = writeln!(out);
    let _ = writeln!(out, "- date: {}", meta.date);
    let note = if meta.note.is_empty() {
        "(none)"
    } else {
        meta.note
    };
    let _ = writeln!(out, "- note: {note}");
    if !meta.session_id.is_empty() {
        let _ = writeln!(out, "- session: {}", meta.session_id);
    }
    if !meta.session_tag.is_empty() {
        let _ = writeln!(out, "- tag: {}", meta.session_tag);
    }
    let _ = writeln!(out, "- context size: {}", meta.ctx_size);
    let _ = writeln!(out, "- transcript tokens: {}", meta.transcript_tokens);
    let _ = writeln!(out, "- last ctx used: {}", meta.last_ctx_used);
    let _ = writeln!(out, "- power: {}%", meta.power_percent);
    let _ = writeln!(out);

    let _ = writeln!(out, "## Generation");
    let _ = writeln!(out);
    let _ = writeln!(out, "- think mode: {}", meta.think.name());
    let _ = writeln!(out, "- n_predict: {}", g.n_predict);
    let _ = writeln!(out, "- temperature: {}", g.temperature);
    let _ = writeln!(out, "- top_p: {}", g.top_p);
    let _ = writeln!(out, "- min_p: {}", g.min_p);
    let _ = writeln!(out, "- seed: {}", g.seed);
    if let Some(model) = &cfg.model_path {
        let _ = writeln!(out, "- model: {}", model.display());
    }
    if let Some(backend) = &cfg.backend {
        let _ = writeln!(out, "- backend: {backend:?}");
    }
    if cfg.engine != crate::config::EngineTuning::default() {
        let _ = writeln!(out, "- engine tuning: {:?}", cfg.engine);
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Rendered transcript (exact engine input)");
    let _ = writeln!(out);
    let _ = writeln!(out, "----- BEGIN TRANSCRIPT -----");
    out.push_str(rendered_transcript);
    if !rendered_transcript.ends_with('\n') {
        out.push('\n');
    }
    let _ = writeln!(out, "----- END TRANSCRIPT -----");
    out
}

/// Writes `report` to a fresh file in [`repro_dir`], returning its path. The
/// name is `repro-<secs>.md`, disambiguated with a `-<n>` suffix if that name
/// is already taken (two dumps within one second).
///
/// # Errors
///
/// Returns a message when the directory or file cannot be created.
pub fn save(cwd: &Path, secs: u64, report: &str) -> Result<PathBuf, String> {
    save_in(&repro_dir(cwd), "repro", secs, report)
}

/// Like [`save`], but named `repro-loop-<secs>.md`: the dump written
/// automatically the moment the repetition guard stops a looping pass, so a
/// stall is captured without the user having to notice it and type `/repro`.
///
/// # Errors
///
/// Returns a message when the directory or file cannot be created.
pub fn save_loop(cwd: &Path, secs: u64, report: &str) -> Result<PathBuf, String> {
    save_in(&repro_dir(cwd), "repro-loop", secs, report)
}

/// Writes `report` into an explicit directory as `<prefix>-<secs>.md`,
/// disambiguating same-second filenames with a `-N` suffix.
///
/// Split out from [`save`] so tests can exercise the naming rule against a
/// scratch directory without reaching for the process-global `HOME`.
///
/// # Errors
/// Returns the OS error message when the directory cannot be created or the
/// file cannot be written.
pub fn save_in(dir: &Path, prefix: &str, secs: u64, report: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut path = dir.join(format!("{prefix}-{secs}.md"));
    let mut n = 1;
    while path.exists() {
        path = dir.join(format!("{prefix}-{secs}-{n}.md"));
        n += 1;
    }
    std::fs::write(&path, report).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> Meta<'static> {
        Meta {
            version: "9.9.9",
            date: "2026-07-19T10:00:00",
            ctx_size: 1_000_000,
            transcript_tokens: 42,
            last_ctx_used: 40,
            power_percent: 100,
            think: crate::engine::ThinkMode::Medium,
            session_id: "abc123",
            session_tag: "",
            note: "model looped on edit",
        }
    }

    #[test]
    fn report_has_metadata_and_verbatim_transcript() {
        let cfg = AgentConfig::default();
        let transcript = "[system]\nsys\n[user]\nhi\n[assistant]\nyo\n";
        let report = build_report(&meta(), &cfg, transcript);
        assert!(report.starts_with("# plank repro 9.9.9\n"));
        assert!(report.contains("note: model looped on edit"));
        assert!(report.contains("session: abc123"));
        assert!(report.contains("think mode: medium"));
        // The transcript is embedded verbatim between the fences.
        let body = report
            .split_once("----- BEGIN TRANSCRIPT -----\n")
            .unwrap()
            .1
            .split_once("----- END TRANSCRIPT -----")
            .unwrap()
            .0;
        assert_eq!(body, transcript);
    }

    #[test]
    fn sidecar_sits_beside_the_main_dump_and_names_it() {
        let dir = std::env::temp_dir().join(format!("plank-repro-side-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let main = save_in(&dir, "repro", 2000, "main").unwrap();
        assert_eq!(sidecar_path(&main, 1), dir.join("repro-2000.sub-1.md"));
        let msgs = vec![
            crate::session::Message::user("task"),
            crate::session::Message::assistant("done"),
        ];
        let mut dump = SidechainDump::new("reviewer", "look at x", 7, &msgs);
        dump.outcome = "failed: interrupted".to_owned();
        let report = build_sidecar_report("9.9.9", "repro-2000.md", 1, &dump, "[user]\ntask\n");
        let path = save_sidecar(&main, 1, &report).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# plank repro 9.9.9 — sub-agent sidecar 1\n"));
        assert!(text.contains("- main repro: repro-2000.md\n"));
        assert!(text.contains("- label: reviewer\n"));
        assert!(text.contains("- outcome: failed: interrupted\n"));
        assert!(text.contains("- forked at parent message: 7\n"));
        assert!(text.contains("look at x\n"));
        assert!(
            text.contains(
                "----- BEGIN TRANSCRIPT -----\n[user]\ntask\n----- END TRANSCRIPT -----\n"
            )
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_dump_strips_images_but_keeps_text_and_timestamps() {
        let mut m = crate::session::Message::user("<tool_result>img</tool_result>");
        m.at = 42;
        m.images.push(crate::engine::VisionImage {
            path: "img.png".to_string(),
            embedding: crate::engine::VisionEmbedding {
                data: vec![0.0; 4],
                token_count: 1,
                layout: 0,
                grid_width: 1,
                grid_height: 1,
                width: 8,
                height: 8,
                content_width: 8,
                content_height: 8,
                fingerprint: [1; 32],
            },
        });
        let dump = SidechainDump::new("sub-agent", "t", 0, &[m]);
        assert_eq!(dump.messages.len(), 1);
        assert!(dump.messages[0].images.is_empty());
        assert_eq!(dump.messages[0].at, 42);
        assert_eq!(dump.messages[0].text, "<tool_result>img</tool_result>");
        assert_eq!(dump.outcome, "ended");
    }

    #[test]
    fn save_disambiguates_same_second() {
        let dir = std::env::temp_dir().join(format!("plank-repro-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        // Never mutate HOME here: `cargo test` runs tests on parallel threads
        // in one process, so a process-global env write races every other test
        // (and every `git` subprocess they spawn) — see issue #43.
        let a = save_in(&dir, "repro", 1000, "first").unwrap();
        let b = save_in(&dir, "repro", 1000, "second").unwrap();
        assert_ne!(a, b);
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "first");
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "second");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn loop_dumps_carry_their_own_prefix() {
        let dir = std::env::temp_dir().join(format!("plank-repro-loop-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let a = save_in(&dir, "repro-loop", 1000, "looped").unwrap();
        assert_eq!(
            a.file_name().unwrap().to_str().unwrap(),
            "repro-loop-1000.md"
        );
        let b = save_in(&dir, "repro-loop", 1000, "again").unwrap();
        assert_eq!(
            b.file_name().unwrap().to_str().unwrap(),
            "repro-loop-1000-1.md"
        );
        // A manual `/repro` in the same second does not collide with the loop dump.
        let c = save_in(&dir, "repro", 1000, "manual").unwrap();
        assert_eq!(c.file_name().unwrap().to_str().unwrap(), "repro-1000.md");
        std::fs::remove_dir_all(&dir).ok();
    }
}
