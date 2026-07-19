//! `/repro` — dump the current session to disk for bug diagnosis.
//!
//! A repro file captures the exact input the engine would see (the rendered
//! `[system]`/`[user]`/`[assistant]` prompt) plus the runtime knobs that shape
//! generation (model, backend, context size, sampling, think mode, engine
//! tuning). It is a self-contained artifact: hand it to a maintainer and they
//! can reproduce the state that triggered a bug without the live session.
//!
//! Files land in `~/.plank/repro/` (or the working dir when `HOME` is unset),
//! named `repro-<unix-seconds>[-<n>].md`. Nothing here touches the live
//! session — it is a read-only snapshot.

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
    /// Session identity SHA (empty when never saved).
    pub session_id: &'a str,
    /// Session tag (empty when unset).
    pub session_tag: &'a str,
    /// Optional user note describing the bug.
    pub note: &'a str,
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
    let _ = writeln!(out, "- think mode: {:?}", g.think_mode);
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
    let dir = repro_dir(cwd);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut path = dir.join(format!("repro-{secs}.md"));
    let mut n = 1;
    while path.exists() {
        path = dir.join(format!("repro-{secs}-{n}.md"));
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
        assert!(report.contains("think mode: On"));
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
    fn save_disambiguates_same_second() {
        let dir = std::env::temp_dir().join(format!("plank-repro-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        // Force HOME to the scratch dir so repro_dir is isolated.
        // SAFETY: single-threaded test; restored immediately after.
        let saved_home = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", &dir) };
        let a = save(&dir, 1000, "first").unwrap();
        let b = save(&dir, 1000, "second").unwrap();
        match saved_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        assert_ne!(a, b);
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "first");
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "second");
        std::fs::remove_dir_all(&dir).ok();
    }
}
