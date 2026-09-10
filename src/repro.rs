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

/// Which model produced the transcript, as far as anything can say.
///
/// Four different answers, because no one of them identifies a model on its
/// own. The configured *path* is often a symlink the reporter repointed; the
/// engine-reported *name* is the shape the C matched and is what selects the
/// dialect; the *family* decides the companion slot and the transcript
/// extension; and the artifact set *version* names the weights, which is the
/// one thing a maintainer cannot recover from the other three. A report that
/// carried only the path has repeatedly not been enough to tell which build of
/// which model was actually loaded.
#[derive(Debug, Default)]
pub struct ModelMeta<'a> {
    /// Shape name the engine reports after opening, e.g. `DeepSeek V4 Flash
    /// Vision Experimental`. Empty when no engine is loaded (the echo stub).
    pub name: &'a str,
    /// Family the loaded model belongs to, spelled as the CLI spells it
    /// (`ds4` / `qwen`).
    pub family: &'a str,
    /// Tool-call dialect in force. Derived from `name`, recorded separately
    /// because a mismatch between the two is itself a bug worth seeing.
    pub syntax: &'a str,
    /// `version` of the installed `ds4.manifest`, i.e. which artifact set is
    /// on disk. `None` when no manifest is installed.
    pub artifact_version: Option<u32>,
    /// The companion GGUF in effect: the `DSpark` draft checkpoint for
    /// `DeepSeek`, the PLE n-gram sidecar for Qwen. Empty when none is configured.
    pub companion: &'a str,
    /// File name of the main artifact the installed manifest declares — the
    /// weights' real name, which a symlinked `path` hides. Empty when no
    /// manifest is installed.
    pub weights_file: &'a str,
    /// Hugging Face *repository* page for the weights (see
    /// [`crate::manifest::hf_repo_url`]), so a report links somewhere a human
    /// can read rather than somewhere a click starts an 87 GB download. Empty
    /// when no manifest is installed or its URL is not a Hugging Face link.
    pub hf_url: &'a str,
}

/// Runtime facts worth recording alongside the transcript, gathered from the
/// live `Agent` by the caller (which owns the engine and config).
#[derive(Debug)]
pub struct Meta<'a> {
    /// Which model produced this transcript. See [`ModelMeta`].
    pub model: ModelMeta<'a>,
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
    /// Rendering settings in effect (`ui.showThinking`, `ui.showToolCalls`).
    /// What the user was shown decides which bugs they can have noticed, and
    /// a sub-agent pane honours the same switches as the main log.
    pub render: crate::session::RenderState,
    /// Session identity SHA (empty when never saved).
    pub session_id: &'a str,
    /// Session tag (empty when unset).
    pub session_tag: &'a str,
    /// Where the session transcript lives on disk (`<kvcache>/<id>.kv`), so
    /// a bug report names the file to attach or `/resume`. Empty when the
    /// session has no id yet.
    pub session_path: &'a str,
    /// Optional user note describing the bug.
    pub note: &'a str,
    /// Whether `tools.loopGuards` was armed when the dump was taken.
    pub guards_armed: bool,
    /// One entry per generation pass this session, oldest first; see
    /// [`PassNote`]. Empty for dumps taken before any pass ran.
    pub passes: &'a [PassNote],
}

/// One generation pass as the agent saw it end, for the `## Passes` table.
///
/// The transcript alone cannot answer the questions a loop dump raises —
/// `repro-loop-1789060243` left it ambiguous whether a 30 KB reasoning pass
/// with no `</think>` was stopped by the user or a guard, and how much of it
/// the guard had counted. This records the answer at the moment it is known.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PassNote {
    /// Wall-clock second the pass ended.
    pub at: u64,
    /// Which agent ran it: empty for the main turn, the sub-agent's label
    /// otherwise.
    pub label: String,
    /// Tokens the pass generated, and its rate.
    pub generated: i32,
    pub tps: f64,
    /// What the reasoning guard counted.
    pub guard: crate::insights::GuardSnapshot,
    /// Why the pass ended: `tool calls: N`, `answer`, `interrupted by user`,
    /// `guard: cycle` / `guard: draft` / `guard: budget`, `tool error`.
    pub stop: String,
}

/// Passes remembered for the table; older ones fall off the front.
pub const PASS_NOTES_CAP: usize = 256;

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
    /// The `subagent-<ordinal>` of the console window this sidechain streamed
    /// to, so a console attaching later can reopen the same window.
    pub ordinal: usize,
    /// Whether that window was connected when the sidechain ended. A dump
    /// that was never mirrored is what a late console gets backfilled with;
    /// set once the backfill has been written so it is sent only once.
    pub mirrored: bool,
}

impl SidechainDump {
    /// Builds a dump from the sidechain's messages, dropping image payloads.
    #[must_use]
    pub fn new(
        label: &str,
        task: &str,
        fork_at: usize,
        messages: &[crate::session::Message],
        ordinal: usize,
        mirrored: bool,
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
            ordinal,
            mirrored,
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
/// The `## Passes` table: one row per generation pass, oldest first; nothing
/// when no pass has run.
fn write_passes(out: &mut String, passes: &[PassNote]) {
    if !passes.is_empty() {
        let _ = writeln!(out, "## Passes");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "One row per generation pass, oldest first. `reasoning` is the bytes the guard saw inside `<think>`; `cycle` is a latched period × copies; `headings`/`fenced` are the draft rung's counts."
        );
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| # | ended | Δ | agent | tokens | tok/s | reasoning | cycle | headings | fenced | stop |"
        );
        let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|");
        let mut prev: Option<u64> = None;
        for (i, p) in passes.iter().enumerate() {
            let delta = prev.map_or_else(String::new, |q| {
                crate::ui::format_elapsed(p.at.saturating_sub(q))
            });
            prev = Some(p.at);
            let cycle = p.guard.cycle.map_or_else(
                || "-".to_owned(),
                |(period, copies)| format!("{period} B × {copies}"),
            );
            let label = if p.label.is_empty() {
                "main"
            } else {
                p.label.as_str()
            };
            let _ = writeln!(
                out,
                "| {} | {} | {delta} | {label} | {} | {:.1} | {} | {cycle} | {} | {} | {} |",
                i + 1,
                crate::context::format_local_time(p.at),
                p.generated,
                p.tps,
                p.guard.fed,
                p.guard.headings,
                p.guard.fenced_bytes,
                p.stop
            );
        }
        let _ = writeln!(out);
    }
}

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
    if !meta.session_path.is_empty() {
        let _ = writeln!(out, "- session file: {}", meta.session_path);
    }
    if !meta.session_tag.is_empty() {
        let _ = writeln!(out, "- tag: {}", meta.session_tag);
    }
    let _ = writeln!(out, "- context size: {}", meta.ctx_size);
    let _ = writeln!(out, "- transcript tokens: {}", meta.transcript_tokens);
    let _ = writeln!(out, "- last ctx used: {}", meta.last_ctx_used);
    let _ = writeln!(out, "- power: {}%", meta.power_percent);
    let _ = writeln!(out);

    // Its own section rather than more lines on the header list: this is the
    // block a maintainer reads first, and burying the weights' identity under
    // sampling knobs is what made "which model was this?" a question worth
    // asking of a report that already answered it.
    let m = &meta.model;
    let _ = writeln!(out, "## Model");
    let _ = writeln!(out);
    if m.name.is_empty() {
        // The echo stub, or a dump taken before the engine opened. Said out
        // loud, because a blank line reads as "the field was not filled in".
        let _ = writeln!(out, "- name: (no engine loaded)");
    } else {
        let _ = writeln!(out, "- name: {}", m.name);
    }
    if !m.family.is_empty() {
        let _ = writeln!(out, "- family: {}", m.family);
    }
    if !m.syntax.is_empty() {
        let _ = writeln!(out, "- tool dialect: {}", m.syntax);
    }
    if let Some(model) = &cfg.model_path {
        let _ = writeln!(out, "- path: {}", model.display());
    }
    if !m.companion.is_empty() {
        let _ = writeln!(out, "- companion: {}", m.companion);
    }
    match m.artifact_version {
        Some(v) => {
            let _ = writeln!(out, "- artifact set: version {v}");
        }
        None => {
            let _ = writeln!(out, "- artifact set: (no manifest installed)");
        }
    }
    if !m.weights_file.is_empty() {
        let _ = writeln!(out, "- weights file: {}", m.weights_file);
    }
    if !m.hf_url.is_empty() {
        let _ = writeln!(out, "- hugging face: {}", m.hf_url);
    }
    if let Some(backend) = &cfg.backend {
        let _ = writeln!(out, "- backend: {backend:?}");
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Generation");
    let _ = writeln!(out);
    let _ = writeln!(out, "- think mode: {}", meta.think.name());
    let _ = writeln!(out, "- show thinking: {}", meta.render.show_thinking);
    let _ = writeln!(out, "- show tool calls: {}", meta.render.show_tool_calls);
    let _ = writeln!(out, "- n_predict: {}", g.n_predict);
    let _ = writeln!(out, "- temperature: {}", g.temperature);
    let _ = writeln!(out, "- top_p: {}", g.top_p);
    let _ = writeln!(out, "- min_p: {}", g.min_p);
    let _ = writeln!(out, "- seed: {}", g.seed);
    // Model path and backend used to be listed here; they moved to `## Model`.
    if cfg.engine != crate::config::EngineTuning::default() {
        let _ = writeln!(out, "- engine tuning: {:?}", cfg.engine);
    }
    let _ = writeln!(
        out,
        "- loop guards: {}",
        if meta.guards_armed {
            "armed"
        } else {
            "off (/loopguard off)"
        }
    );
    if let Some(budget) = meta.passes.iter().rev().find_map(|p| p.guard.budget) {
        let _ = writeln!(out, "- think budget: {budget} bytes of reasoning per pass");
    }
    let _ = writeln!(out);

    write_passes(&mut out, meta.passes);

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

/// The last rendered transcript, kept for the panic dump.
///
/// Only written while the debug mirror is on ([`stash_transcript`]), because
/// that is the only case in which a dump is taken at all and the copy is not
/// free. A panic hook cannot borrow the agent — it runs with the stack already
/// unwinding, from whatever thread failed — so the one thing it can do is
/// write bytes somebody else prepared.
static PANIC_TRANSCRIPT: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Records the rendered transcript for a possible panic dump. Cheap to call
/// unconditionally: it returns immediately unless the debug mirror is on.
pub fn stash_transcript(rendered: &str) {
    if !crate::debugmirror::enabled() {
        return;
    }
    if let Ok(mut slot) = PANIC_TRANSCRIPT.lock() {
        *slot = Some(rendered.to_owned());
    }
}

/// The stashed transcript, or `None` if nothing was ever stashed.
#[must_use]
pub fn stashed_transcript() -> Option<String> {
    PANIC_TRANSCRIPT.lock().ok().and_then(|s| s.clone())
}

/// The panic dump's text: what [`build_report`] can still say once the agent
/// is gone — the version, the time, the panic itself, and the last transcript
/// the session rendered.
///
/// A separate builder rather than a [`Meta`] with holes in it: everything
/// [`build_report`] reports beyond this comes from the engine or the config,
/// and a hook has neither. Same transcript fences, so the two dumps read
/// alike and the same tooling parses both.
#[must_use]
pub fn build_panic_report(
    version: &str,
    date: &str,
    panic: &str,
    transcript: Option<&str>,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# plank repro {version} (panic)");
    let _ = writeln!(out);
    let _ = writeln!(out, "- date: {date}");
    let _ = writeln!(out, "- note: plank panicked");
    let _ = writeln!(out, "- panic: {}", panic.trim());
    let _ = writeln!(out);
    let _ = writeln!(out, "## Rendered transcript (exact engine input)");
    let _ = writeln!(out);
    let _ = writeln!(out, "----- BEGIN TRANSCRIPT -----");
    match transcript {
        Some(t) => {
            out.push_str(t);
            if !t.ends_with('\n') {
                out.push('\n');
            }
        }
        // A panic before the first pass, or with the mirror off until now.
        None => {
            let _ = writeln!(out, "(no transcript was captured before the panic)");
        }
    }
    let _ = writeln!(out, "----- END TRANSCRIPT -----");
    out
}

/// The panic payload as text: the two payload types `panic!` produces, and a
/// placeholder for anything else.
#[must_use]
pub fn panic_text(info: &std::panic::PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    let msg = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "(non-string panic payload)".to_owned());
    match info.location() {
        Some(loc) => format!("{msg} (at {}:{})", loc.file(), loc.line()),
        None => msg,
    }
}

/// Installs the panic dump hook, chaining to whatever hook is already set.
///
/// Armed for the whole run but *decides at panic time*: it dumps only when the
/// debug mirror is on, so `/debug on` mid-session arms it and a normal run
/// pays nothing. Failures are swallowed — a panic that also cannot write its
/// dump must still reach the default hook and print its message.
pub fn install_panic_hook(dir: PathBuf) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if crate::debugmirror::enabled() {
            let report = build_panic_report(
                &crate::logo::version_label(),
                &crate::context::current_local_iso_date(),
                &panic_text(info),
                stashed_transcript().as_deref(),
            );
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            if let Ok(path) = save_in(&dir, "repro-panic", secs, &report) {
                eprintln!("[panic repro written to {}]", path.display());
            }
        }
        previous(info);
    }));
}

/// Writes `report` into an explicit directory as `<prefix>-<secs>.md`,
/// disambiguating same-second filenames with a `-N` suffix.
///
/// The agent resolves the directory once ([`repro_dir`]) and passes it in, so
/// a test agent can aim its dumps at a scratch directory instead of `$HOME`.
/// `prefix` is `repro` for `/repro` and `repro-loop` for the automatic dump.
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
            model: ModelMeta {
                name: "DeepSeek V4 Flash Vision Experimental",
                family: "ds4",
                syntax: "dsml",
                artifact_version: Some(7),
                companion: "/home/u/.plank/ds4flash.dspark.gguf",
                weights_file: "DeepSeek-V4-Flash-Vision-Exp-IQ2XXS.gguf",
                hf_url: "https://huggingface.co/antirez/deepseek-v4-gguf",
            },
            version: "9.9.9",
            date: "2026-07-19T10:00:00",
            ctx_size: 1_000_000,
            transcript_tokens: 42,
            last_ctx_used: 40,
            power_percent: 100,
            think: crate::engine::ThinkMode::Medium,
            render: crate::session::RenderState {
                show_thinking: true,
                show_tool_calls: false,
            },
            guards_armed: true,
            passes: &[],
            session_id: "abc123",
            session_tag: "",
            session_path: "/home/u/.plank/kvcache/abc123.kv",
            note: "model looped on edit",
        }
    }

    #[test]
    fn a_panic_report_carries_the_panic_and_the_stashed_transcript() {
        let text = build_panic_report(
            "9.9.9",
            "2026-09-08T10:00:00",
            "index out of bounds (at src/x.rs:12)",
            Some("[system]\nsys\n[user]\nhi"),
        );
        assert!(text.starts_with("# plank repro 9.9.9 (panic)"), "{text}");
        assert!(text.contains("- panic: index out of bounds (at src/x.rs:12)"));
        // Same fences as `build_report`, so one parser reads both, and the
        // transcript is newline-terminated even when the input was not.
        assert!(text.contains(
            "----- BEGIN TRANSCRIPT -----\n[system]\nsys\n[user]\nhi\n----- END TRANSCRIPT -----"
        ));
    }

    /// A panic before the first pass has nothing stashed; the dump still has
    /// to be written, because the panic is the news.
    #[test]
    fn a_panic_report_with_no_transcript_says_so() {
        let text = build_panic_report("9.9.9", "d", "boom", None);
        assert!(text.contains("(no transcript was captured before the panic)"));
    }

    /// The weights' identity is the thing a maintainer cannot reconstruct from
    /// anything else in the file, so all four answers are pinned: the engine's
    /// own name, the family, the dialect, and the installed artifact version.
    #[test]
    fn the_model_section_records_which_model_produced_the_transcript() {
        let cfg = AgentConfig::default();
        let report = build_report(&meta(), &cfg, "[user]\nhi\n");
        let section = report
            .split_once("## Model\n")
            .expect("a Model section")
            .1
            .split_once("## Generation")
            .expect("followed by Generation")
            .0;
        assert!(
            section.contains("- name: DeepSeek V4 Flash Vision Experimental"),
            "{section}"
        );
        assert!(section.contains("- family: ds4"), "{section}");
        assert!(section.contains("- tool dialect: dsml"), "{section}");
        assert!(section.contains("- artifact set: version 7"), "{section}");
        assert!(
            section.contains("- weights file: DeepSeek-V4-Flash-Vision-Exp-IQ2XXS.gguf"),
            "{section}"
        );
        // The repo page, never the `/resolve/` download URL.
        assert!(
            section.contains("- hugging face: https://huggingface.co/antirez/deepseek-v4-gguf"),
            "{section}"
        );
        assert!(!section.contains("/resolve/"), "no download URL: {section}");
        assert!(
            section.contains("- companion: /home/u/.plank/ds4flash.dspark.gguf"),
            "{section}"
        );
        // It moved out of `## Generation`, so it must not be in both places.
        let generation = report.split_once("## Generation").unwrap().1;
        assert!(
            !generation.contains("- name:") && !generation.contains("- model:"),
            "the model is recorded once, in its own section: {generation}"
        );
    }

    /// An absent engine and an absent manifest are stated, not left blank: a
    /// missing line reads as a field nobody filled in, which is a different
    /// bug report from "there was no engine".
    #[test]
    fn an_unknown_model_says_so_rather_than_leaving_the_field_empty() {
        let cfg = AgentConfig::default();
        let mut m = meta();
        m.model = ModelMeta::default();
        let report = build_report(&m, &cfg, "[user]\nhi\n");
        assert!(report.contains("- name: (no engine loaded)"), "{report}");
        assert!(
            report.contains("- artifact set: (no manifest installed)"),
            "{report}"
        );
        // The optional lines are simply absent rather than printed empty.
        assert!(!report.contains("- family: \n"), "{report}");
        assert!(!report.contains("- companion: \n"), "{report}");
    }

    #[test]
    fn report_has_metadata_and_verbatim_transcript() {
        let cfg = AgentConfig::default();
        let transcript = "[system]\nsys\n[user]\nhi\n[assistant]\nyo\n";
        let report = build_report(&meta(), &cfg, transcript);
        assert!(report.starts_with("# plank repro 9.9.9\n"));
        assert!(report.contains("note: model looped on edit"));
        assert!(report.contains("session: abc123"));
        assert!(report.contains("session file: /home/u/.plank/kvcache/abc123.kv"));
        assert!(report.contains("think mode: medium"));
        // The rendering switches sit with the other generation-time knobs.
        assert!(report.contains("show thinking: true"));
        assert!(report.contains("show tool calls: false"));
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
        let mut dump = SidechainDump::new("reviewer", "look at x", 7, &msgs, 1, true);
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
        let dump = SidechainDump::new("sub-agent", "t", 0, &[m], 1, true);
        assert_eq!(dump.messages.len(), 1);
        assert!(dump.messages[0].images.is_empty());
        assert_eq!(dump.messages[0].at, 42);
        assert_eq!(dump.messages[0].text, "<tool_result>img</tool_result>");
        assert_eq!(dump.outcome, "ended");
    }

    #[test]
    fn a_dump_remembers_its_window_ordinal_and_whether_it_was_mirrored() {
        let m = crate::session::Message::assistant("hi");
        let dump = SidechainDump::new("sub-agent", "t", 0, &[m], 3, false);
        assert_eq!(dump.ordinal, 3);
        assert!(!dump.mirrored);
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

    #[test]
    fn the_passes_table_names_each_stop() {
        let passes = vec![
            PassNote {
                at: 1_700_000_000,
                label: String::new(),
                generated: 900,
                tps: 16.7,
                guard: crate::insights::GuardSnapshot {
                    fed: 30_467,
                    budget: Some(104_857),
                    cycle: None,
                    headings: 31,
                    fenced_bytes: 0,
                },
                stop: "interrupted by user".to_owned(),
            },
            PassNote {
                at: 1_700_000_311,
                label: "reviewer".to_owned(),
                generated: 500,
                tps: 15.0,
                guard: crate::insights::GuardSnapshot {
                    fed: 17_038,
                    budget: Some(104_857),
                    cycle: Some((631, 5)),
                    headings: 6,
                    fenced_bytes: 0,
                },
                stop: "guard: cycle".to_owned(),
            },
        ];
        let mut m = meta();
        m.passes = &passes;
        let report = build_report(&m, &AgentConfig::default(), "");
        assert!(report.contains("## Passes"), "{report}");
        assert!(
            report.contains("| main | 900 | 16.7 | 30467 | - | 31 | 0 | interrupted by user |"),
            "{report}"
        );
        assert!(
            report.contains(
                "| +5m11s | reviewer | 500 | 15.0 | 17038 | 631 B × 5 | 6 | 0 | guard: cycle |"
            ),
            "{report}"
        );
        assert!(report.contains("- loop guards: armed"), "{report}");
        assert!(report.contains("- think budget: 104857 bytes"), "{report}");
        // No passes: no table, and no budget line to invent one from.
        let report = build_report(&meta(), &AgentConfig::default(), "");
        assert!(!report.contains("## Passes"));
        assert!(!report.contains("think budget"));
    }
}
