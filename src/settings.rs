// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Persistent user preferences (`settings.json`).
//!
//! Holds the settings that are *stable preferences* rather than per-run
//! choices: engine defaults, UI tuning, safety defaults, and the MCP handshake
//! timeout. Operational flags (`--prompt`, `--non-interactive`, `--ui-remote`,
//! `--trace`, `--chdir`, `--seed`, and the serve/control options) describe one
//! invocation and deliberately have no settings key.
//!
//! Files are read from `~/.plank/settings.json` then `<cwd>/.plank/settings.json`,
//! the later file winning key by key. The full precedence chain is:
//!
//! ```text
//! built-in defaults < ~/.plank/settings.json < ./.plank/settings.json < env < CLI flags
//! ```
//!
//! A missing file, unreadable file, malformed JSON, or a value of the wrong
//! type all fall back to the default for that key: a broken settings file
//! degrades plank's preferences, never its ability to start.
//!
//! Secrets are excluded by design. `./.plank/settings.json` lives inside the
//! working tree and is easy to commit by accident, so the provider API key
//! stays on the environment and the command line.
//!
//! ```json
//! {
//!   "engine": { "model": "~/models/ds4.gguf", "threads": 8, "backend": "metal",
//!               "power": 80, "ctx": 262144 },
//!   "ui":     { "respectGitignore": true, "popupRows": 15, "indexRefreshSecs": 5,
//!               "historySize": 512, "showToolCalls": false, "showToolResults": false,
//!               "showThinking": true },
//!   "safety": { "sandbox": true, "btwSuspend": false },
//!   "mcp":    { "timeoutSecs": 30 },
//!   "ask":    { "maxOptions": 7 }
//! }
//! ```

use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use crate::tools::mcp::{Json, json_escape, json_parse, json_write};

/// Popup rows offered by `@` completion when unset.
pub const DEFAULT_POPUP_ROWS: usize = 15;
/// Seconds a built file index is trusted before a refresh is allowed.
pub const DEFAULT_INDEX_REFRESH_SECS: u64 = 5;
/// Prompt history entries retained when unset.
pub const DEFAULT_HISTORY_SIZE: usize = 512;
/// Seconds an MCP server has to answer a request when unset.
pub const DEFAULT_MCP_TIMEOUT_SECS: u64 = 30;
/// Most options the `ask` tool accepts when unset.
pub const DEFAULT_ASK_MAX_OPTIONS: usize = 7;
/// Fewest options the `ask` tool ever accepts; a choice needs two arms. Not
/// configurable — a one-option "choice" is a degenerate question.
pub const ASK_MIN_OPTIONS: usize = 2;

/// Engine defaults: the same knobs as `-m`, `-t`, `--backend`, `--power`, `-c`.
///
/// `model` replaces what used to be a hardcoded convention — plank falls back
/// to `~/.plank/ds4flash.gguf` only when neither this key nor `-m` is given.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineSettings {
    /// Model file to load; overridden by `-m`/`--model`.
    pub model: Option<PathBuf>,
    /// Worker thread count; overridden by `-t`/`--threads`.
    pub threads: Option<i32>,
    /// Backend name (`metal`, `cuda`, `cpu`); overridden by `--backend`.
    pub backend: Option<String>,
    /// GPU power cap percent; overridden by `--power`.
    pub power: Option<i32>,
    /// Context window in tokens; overridden by `-c`/`--ctx`.
    pub ctx: Option<i32>,
}

/// UI behaviour that used to be magic numbers in the source.
// The display toggles (showToolCalls/showToolResults/showThinking) are
// genuinely independent on/off knobs, not a state machine to model as an enum.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiSettings {
    /// Whether `@` completion honours `.gitignore` for untracked files.
    pub respect_gitignore: bool,
    /// Rows the `@` completion popup offers at most.
    pub popup_rows: usize,
    /// Seconds before the file index may be rebuilt.
    pub index_refresh_secs: u64,
    /// Prompt history entries retained.
    pub history_size: usize,
    /// Show the model's tool-call banners (`🛠️ …`). Off by default so the UI
    /// stays uncluttered; the DSML is always parsed regardless.
    pub show_tool_calls: bool,
    /// Echo tool result text (observations) into the scrollback. Off by
    /// default; the model always receives the results either way.
    pub show_tool_results: bool,
    /// Render the model's thinking text (dimmed) in the scrollback. On by
    /// default; when off, thinking is hidden from the display but the model
    /// still produces it.
    pub show_thinking: bool,
    /// Fire native macOS desktop notifications at turn lifecycle points
    /// (turn complete past the threshold, and awaiting input). On by default.
    pub notifications: bool,
    /// Minimum turn duration, in seconds, before a completed turn notifies.
    /// Awaiting-input notifications ignore this. Default 10.
    pub notify_after_secs: u64,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            respect_gitignore: true,
            popup_rows: DEFAULT_POPUP_ROWS,
            index_refresh_secs: DEFAULT_INDEX_REFRESH_SECS,
            history_size: DEFAULT_HISTORY_SIZE,
            show_tool_calls: false,
            show_tool_results: false,
            show_thinking: true,
            notifications: true,
            notify_after_secs: 10,
        }
    }
}

/// Persisted defaults for the two-sided safety flags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SafetySettings {
    /// Default for the bash write sandbox (on where sandbox-exec exists);
    /// overridden by `--sandbox`/`--no-sandbox`.
    pub sandbox: Option<bool>,
    /// Default for `/btw` mid-generation suspend; overridden by
    /// `--btw-suspend`/`--disable-btw-suspend`.
    pub btw_suspend: Option<bool>,
}

/// MCP client tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpSettings {
    /// Seconds an MCP server has to answer before it is considered dead.
    pub timeout_secs: u64,
}

impl Default for McpSettings {
    fn default() -> Self {
        Self {
            timeout_secs: DEFAULT_MCP_TIMEOUT_SECS,
        }
    }
}

/// `ask` tool tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskSettings {
    /// Most options the `ask` tool accepts (the minimum is fixed at 2).
    pub max_options: usize,
}

impl Default for AskSettings {
    fn default() -> Self {
        Self {
            max_options: DEFAULT_ASK_MAX_OPTIONS,
        }
    }
}

/// Opt-in switches for tools the `DeepSeek` model was not trained on (they have
/// no counterpart in the C `ds4_agent`). Off by default so the base model sees
/// roughly its trained tool surface; a small model tends to hallucinate a
/// pseudo-syntax (e.g. bare `<task>` blocks) for unfamiliar tools rather than
/// emit valid DSML, so these stay hidden unless deliberately enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolSettings {
    /// The `task` todo-list tool (issue #35).
    pub task: bool,
    /// The `agent` sub-agent delegation tool (issue #50).
    pub agent: bool,
    /// Plan mode (`EnterPlanMode`/`ExitPlanMode`, issue #50).
    pub plan_mode: bool,
}

/// The whole of `settings.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Settings {
    /// Engine defaults.
    pub engine: EngineSettings,
    /// UI behaviour.
    pub ui: UiSettings,
    /// Safety defaults.
    pub safety: SafetySettings,
    /// MCP client tuning.
    pub mcp: McpSettings,
    /// `ask` tool tuning.
    pub ask: AskSettings,
    /// Opt-in non-trained tools.
    pub tools: ToolSettings,
}

/// Reads a positive integer member, ignoring absent, non-numeric, and
/// out-of-range values so one bad key cannot discard the rest of the file.
fn num<T: TryFrom<i64>>(obj: Option<&Json>, key: &str) -> Option<T> {
    let Some(Json::Num(n)) = obj?.get(key) else {
        return None;
    };
    // `as` on a non-finite or huge f64 saturates rather than failing, so the
    // range has to be checked before the cast.
    if !n.is_finite() || *n < MIN_SAFE || *n > MAX_SAFE {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)] // range-checked directly above
    T::try_from(*n as i64).ok()
}

/// Bounds outside which an `f64 -> i64` cast is not exact.
const MAX_SAFE: f64 = 9_007_199_254_740_992.0;
const MIN_SAFE: f64 = -MAX_SAFE;

fn boolean(obj: Option<&Json>, key: &str) -> Option<bool> {
    match obj?.get(key) {
        Some(Json::Bool(b)) => Some(*b),
        _ => None,
    }
}

fn string(obj: Option<&Json>, key: &str) -> Option<String> {
    match obj?.get(key) {
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

impl Settings {
    /// Parses one `settings.json`, overlaying `self` key by key.
    ///
    /// Unknown keys are ignored, so a newer plank's file stays loadable by an
    /// older one.
    fn overlay(&mut self, text: &str) {
        let Some(root) = json_parse(text) else { return };
        let engine = root.get("engine");
        if let Some(v) = string(engine, "model") {
            self.engine.model = Some(expand_tilde(&v));
        }
        if let Some(v) = num(engine, "threads") {
            self.engine.threads = Some(v);
        }
        if let Some(v) = string(engine, "backend") {
            self.engine.backend = Some(v);
        }
        if let Some(v) = num(engine, "power") {
            self.engine.power = Some(v);
        }
        if let Some(v) = num(engine, "ctx") {
            self.engine.ctx = Some(v);
        }

        let ui = root.get("ui");
        if let Some(v) = boolean(ui, "respectGitignore") {
            self.ui.respect_gitignore = v;
        }
        // A zero-row popup or zero-entry history would silently disable the
        // feature rather than tune it; treat those as unset.
        if let Some(v) = num::<usize>(ui, "popupRows").filter(|v| *v > 0) {
            self.ui.popup_rows = v;
        }
        if let Some(v) = num(ui, "indexRefreshSecs") {
            self.ui.index_refresh_secs = v;
        }
        if let Some(v) = num::<usize>(ui, "historySize").filter(|v| *v > 0) {
            self.ui.history_size = v;
        }
        if let Some(v) = boolean(ui, "showToolCalls") {
            self.ui.show_tool_calls = v;
        }
        if let Some(v) = boolean(ui, "showToolResults") {
            self.ui.show_tool_results = v;
        }
        if let Some(v) = boolean(ui, "showThinking") {
            self.ui.show_thinking = v;
        }
        if let Some(v) = boolean(ui, "notifications") {
            self.ui.notifications = v;
        }
        if let Some(v) = num(ui, "notifyAfterSecs") {
            self.ui.notify_after_secs = v;
        }

        let safety = root.get("safety");
        if let Some(v) = boolean(safety, "sandbox") {
            self.safety.sandbox = Some(v);
        }
        if let Some(v) = boolean(safety, "btwSuspend") {
            self.safety.btw_suspend = Some(v);
        }

        if let Some(v) = num::<u64>(root.get("mcp"), "timeoutSecs").filter(|v| *v > 0) {
            self.mcp.timeout_secs = v;
        }

        // A max below the fixed minimum would make every `ask` call impossible;
        // clamp it up rather than silently breaking the tool.
        if let Some(v) = num::<usize>(root.get("ask"), "maxOptions") {
            self.ask.max_options = v.max(ASK_MIN_OPTIONS);
        }

        let tools = root.get("tools");
        if let Some(v) = boolean(tools, "task") {
            self.tools.task = v;
        }
        if let Some(v) = boolean(tools, "agent") {
            self.tools.agent = v;
        }
        if let Some(v) = boolean(tools, "planMode") {
            self.tools.plan_mode = v;
        }
    }

    /// Loads `~/.plank/settings.json` then `<cwd>/.plank/settings.json`.
    #[must_use]
    pub fn load() -> Self {
        let mut s = Self::default();
        for p in Self::paths() {
            if let Ok(text) = std::fs::read_to_string(&p) {
                s.overlay(&text);
            }
        }
        s
    }

    /// The files [`load`](Self::load) consults, in increasing precedence.
    #[must_use]
    pub fn paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(home) = std::env::var_os("HOME") {
            paths.push(PathBuf::from(home).join(".plank").join("settings.json"));
        }
        if let Ok(cwd) = std::env::current_dir() {
            paths.push(cwd.join(".plank").join("settings.json"));
        }
        paths
    }

    /// The settings files that actually exist, for the startup note.
    #[must_use]
    pub fn existing_paths() -> Vec<PathBuf> {
        Self::paths().into_iter().filter(|p| p.is_file()).collect()
    }
}

/// One line naming every setting that is actually in effect, or `None` when
/// the files changed nothing.
///
/// A settings file can move you off Metal onto the CPU or shrink the context,
/// and both are invisible once the UI is up — you just notice plank is slow.
/// This makes the cause self-diagnosing.
///
/// `cfg` is consulted so a setting a CLI flag overrode is *not* reported: the
/// note lists what is in force, never what a file merely asked for.
#[must_use]
pub fn startup_note(s: &Settings, cfg: &crate::config::AgentConfig) -> Option<String> {
    let d = Settings::default();
    let mut parts: Vec<String> = Vec::new();

    // Engine and safety keys: reported only when the parsed config still
    // carries the file's value, i.e. no flag overrode it.
    if let Some(m) = &s.engine.model
        && cfg.model_path.as_ref() == Some(m)
    {
        parts.push(format!("model={}", m.display()));
    }
    if let Some(t) = s.engine.threads
        && cfg.n_threads == t
    {
        parts.push(format!("threads={t}"));
    }
    if let Some(b) = s.engine.backend.as_deref()
        && cfg.backend == crate::config::parse_backend(b)
    {
        parts.push(format!("backend={b}"));
    }
    if let Some(p) = s.engine.power
        && cfg.power_percent == p
    {
        parts.push(format!("power={p}"));
    }
    if let Some(c) = s.engine.ctx
        && cfg.generation.ctx_size == c
    {
        parts.push(format!("ctx={c}"));
    }
    if let Some(v) = s.safety.sandbox
        && cfg.sandbox_override == Some(v)
    {
        parts.push(format!("sandbox={v}"));
    }
    if let Some(v) = s.safety.btw_suspend
        && cfg.btw.suspend == v
    {
        parts.push(format!("btwSuspend={v}"));
    }

    // UI and MCP keys have no flag, so any non-default value is in force.
    if s.ui.respect_gitignore != d.ui.respect_gitignore {
        parts.push(format!("respectGitignore={}", s.ui.respect_gitignore));
    }
    if s.ui.popup_rows != d.ui.popup_rows {
        parts.push(format!("popupRows={}", s.ui.popup_rows));
    }
    if s.ui.index_refresh_secs != d.ui.index_refresh_secs {
        parts.push(format!("indexRefreshSecs={}", s.ui.index_refresh_secs));
    }
    if s.ui.history_size != d.ui.history_size {
        parts.push(format!("historySize={}", s.ui.history_size));
    }
    if s.ui.show_tool_calls != d.ui.show_tool_calls {
        parts.push(format!("showToolCalls={}", s.ui.show_tool_calls));
    }
    if s.ui.show_tool_results != d.ui.show_tool_results {
        parts.push(format!("showToolResults={}", s.ui.show_tool_results));
    }
    if s.ui.show_thinking != d.ui.show_thinking {
        parts.push(format!("showThinking={}", s.ui.show_thinking));
    }
    if s.ui.notifications != d.ui.notifications {
        parts.push(format!("notifications={}", s.ui.notifications));
    }
    if s.ui.notify_after_secs != d.ui.notify_after_secs {
        parts.push(format!("notifyAfterSecs={}", s.ui.notify_after_secs));
    }
    if s.mcp.timeout_secs != d.mcp.timeout_secs {
        parts.push(format!("timeoutSecs={}", s.mcp.timeout_secs));
    }
    if s.ask.max_options != d.ask.max_options {
        parts.push(format!("maxOptions={}", s.ask.max_options));
    }
    if s.tools.task != d.tools.task {
        parts.push(format!("tools.task={}", s.tools.task));
    }
    if s.tools.agent != d.tools.agent {
        parts.push(format!("tools.agent={}", s.tools.agent));
    }
    if s.tools.plan_mode != d.tools.plan_mode {
        parts.push(format!("tools.planMode={}", s.tools.plan_mode));
    }

    if parts.is_empty() {
        return None;
    }
    let from = Settings::existing_paths()
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let from = if from.is_empty() {
        "settings".to_string()
    } else {
        from
    };
    Some(format!(
        "plank: settings in effect ({from}): {}",
        parts.join(", ")
    ))
}

/// Expands a leading `~/` against `$HOME`, leaving other paths untouched.
fn expand_tilde(s: &str) -> PathBuf {
    match (s.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(s),
    }
}

/// The project-scoped settings file, `<cwd>/.plank/settings.json` — the
/// highest-precedence file and where `/config` writes.
#[must_use]
pub fn project_path() -> Option<PathBuf> {
    std::env::current_dir()
        .ok()
        .map(|cwd| cwd.join(".plank").join("settings.json"))
}

fn upsert(obj: &mut Vec<(String, Json)>, key: &str, val: Json) {
    if let Some(slot) = obj.iter_mut().find(|(k, _)| k == key) {
        slot.1 = val;
    } else {
        obj.push((key.to_string(), val));
    }
}

/// Upserts an optional value, or removes the key when `val` is `None` so that
/// an unset optional is reflected as absence (its built-in default on reload).
fn upsert_opt(obj: &mut Vec<(String, Json)>, key: &str, val: Option<Json>) {
    match val {
        Some(v) => upsert(obj, key, v),
        None => obj.retain(|(k, _)| k != key),
    }
}

/// Returns the named section object, creating it if absent or non-object.
fn section<'a>(root: &'a mut Vec<(String, Json)>, name: &str) -> &'a mut Vec<(String, Json)> {
    let idx = if let Some(i) = root.iter().position(|(k, _)| k == name) {
        if !matches!(root[i].1, Json::Obj(_)) {
            root[i].1 = Json::Obj(Vec::new());
        }
        i
    } else {
        root.push((name.to_string(), Json::Obj(Vec::new())));
        root.len() - 1
    };
    match &mut root[idx].1 {
        Json::Obj(o) => o,
        _ => unreachable!("just ensured an object"),
    }
}

#[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
fn inum(v: i32) -> Json {
    Json::Num(v as f64)
}

#[allow(clippy::cast_precision_loss)]
fn unum(v: u64) -> Json {
    Json::Num(v as f64)
}

/// Pretty-prints a JSON value with two-space indentation (objects only get
/// multi-line treatment; scalars and arrays stay compact via [`json_write`]).
fn write_pretty(out: &mut String, v: &Json, indent: usize) {
    match v {
        Json::Obj(members) if !members.is_empty() => {
            out.push_str("{\n");
            for (i, (k, val)) in members.iter().enumerate() {
                for _ in 0..=indent {
                    out.push_str("  ");
                }
                json_escape(out, k);
                out.push_str(": ");
                write_pretty(out, val, indent + 1);
                if i + 1 < members.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            for _ in 0..indent {
                out.push_str("  ");
            }
            out.push('}');
        }
        other => json_write(out, other),
    }
}

impl Settings {
    /// Serializes these settings to `path`, preserving any unknown keys already
    /// present (so a newer plank's file survives an older binary's write).
    ///
    /// # Errors
    /// Returns `Err` if the parent directory cannot be created or the write fails.
    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        let mut root: Vec<(String, Json)> = match std::fs::read_to_string(path) {
            Ok(t) => match json_parse(&t) {
                Some(Json::Obj(o)) => o,
                _ => Vec::new(),
            },
            Err(_) => Vec::new(),
        };

        {
            let e = section(&mut root, "engine");
            upsert_opt(
                e,
                "model",
                self.engine
                    .model
                    .as_ref()
                    .map(|p| Json::Str(p.display().to_string())),
            );
            upsert_opt(e, "threads", self.engine.threads.map(inum));
            upsert_opt(e, "backend", self.engine.backend.clone().map(Json::Str));
            upsert_opt(e, "power", self.engine.power.map(inum));
            upsert_opt(e, "ctx", self.engine.ctx.map(inum));
        }
        {
            let u = section(&mut root, "ui");
            upsert(u, "respectGitignore", Json::Bool(self.ui.respect_gitignore));
            upsert(u, "popupRows", unum(self.ui.popup_rows as u64));
            upsert(u, "indexRefreshSecs", unum(self.ui.index_refresh_secs));
            upsert(u, "historySize", unum(self.ui.history_size as u64));
            upsert(u, "showToolCalls", Json::Bool(self.ui.show_tool_calls));
            upsert(u, "showToolResults", Json::Bool(self.ui.show_tool_results));
            upsert(u, "showThinking", Json::Bool(self.ui.show_thinking));
            upsert(u, "notifications", Json::Bool(self.ui.notifications));
            upsert(u, "notifyAfterSecs", unum(self.ui.notify_after_secs));
        }
        {
            let s = section(&mut root, "safety");
            upsert_opt(s, "sandbox", self.safety.sandbox.map(Json::Bool));
            upsert_opt(s, "btwSuspend", self.safety.btw_suspend.map(Json::Bool));
        }
        upsert(
            section(&mut root, "mcp"),
            "timeoutSecs",
            unum(self.mcp.timeout_secs),
        );
        upsert(
            section(&mut root, "ask"),
            "maxOptions",
            unum(self.ask.max_options as u64),
        );
        {
            let t = section(&mut root, "tools");
            upsert(t, "task", Json::Bool(self.tools.task));
            upsert(t, "agent", Json::Bool(self.tools.agent));
            upsert(t, "planMode", Json::Bool(self.tools.plan_mode));
        }

        let mut out = String::new();
        write_pretty(&mut out, &Json::Obj(root), 0);
        out.push('\n');
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, out).map_err(|e| e.to_string())
    }
}

/// Process-wide settings. A swappable `&'static` (the payload is `Box::leak`ed)
/// so [`reinstall`] can update it live from `/config` without changing the
/// zero-cost `&'static` contract that [`active`]'s many call sites rely on. The
/// per-swap leak is bounded — swaps happen only on explicit user action.
static ACTIVE: RwLock<Option<&'static Settings>> = RwLock::new(None);

/// Installs the process-wide settings. Later calls are ignored.
///
/// Call once from `main` before the UI starts. Code that reads settings via
/// [`active`] sees built-in defaults until this runs, which is what tests and
/// library consumers get.
pub fn install(settings: Settings) {
    // Recover rather than panic on a poisoned lock: settings are advisory and a
    // stale guard is harmless here.
    let mut slot = ACTIVE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot.is_none() {
        *slot = Some(Box::leak(Box::new(settings)));
    }
}

/// Replaces the process-wide settings (used by `/config` after a save), so the
/// current session picks up the change on its next [`active`] read.
pub fn reinstall(settings: Settings) {
    *ACTIVE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Box::leak(Box::new(settings)));
}

/// The process-wide settings, or the built-in defaults before [`install`].
#[must_use]
pub fn active() -> &'static Settings {
    static FALLBACK: OnceLock<Settings> = OnceLock::new();
    // References are `Copy`, so the `&'static` escapes the read guard cleanly.
    if let Some(s) = *ACTIVE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        s
    } else {
        FALLBACK.get_or_init(Settings::default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_json(text: &str) -> Settings {
        let mut s = Settings::default();
        s.overlay(text);
        s
    }

    #[test]
    fn defaults_match_the_previously_hardcoded_constants() {
        let s = Settings::default();
        assert!(s.ui.respect_gitignore);
        assert_eq!(s.ui.popup_rows, 15);
        assert_eq!(s.ui.index_refresh_secs, 5);
        assert_eq!(s.ui.history_size, 512);
        assert_eq!(s.mcp.timeout_secs, 30);
        assert_eq!(s.engine.model, None);
        assert_eq!(s.safety.sandbox, None);
    }

    #[test]
    fn reads_every_group() {
        let s = from_json(
            r#"{"engine":{"threads":8,"backend":"cpu","power":80,"ctx":262144},
                "ui":{"respectGitignore":false,"popupRows":25,
                      "indexRefreshSecs":30,"historySize":4096},
                "safety":{"sandbox":true,"btwSuspend":false},
                "mcp":{"timeoutSecs":90},
                "ask":{"maxOptions":10}}"#,
        );
        assert_eq!(s.ask.max_options, 10);
        assert_eq!(s.engine.threads, Some(8));
        assert_eq!(s.engine.backend.as_deref(), Some("cpu"));
        assert_eq!(s.engine.power, Some(80));
        assert_eq!(s.engine.ctx, Some(262_144));
        assert!(!s.ui.respect_gitignore);
        assert_eq!(s.ui.popup_rows, 25);
        assert_eq!(s.ui.index_refresh_secs, 30);
        assert_eq!(s.ui.history_size, 4096);
        assert_eq!(s.safety.sandbox, Some(true));
        assert_eq!(s.safety.btw_suspend, Some(false));
        assert_eq!(s.mcp.timeout_secs, 90);
    }

    #[test]
    fn tool_display_is_off_by_default_and_opt_in() {
        let d = Settings::default();
        assert!(!d.ui.show_tool_calls, "tool calls hidden by default");
        assert!(!d.ui.show_tool_results, "tool results hidden by default");
        let s = from_json(r#"{"ui":{"showToolCalls":true,"showToolResults":true}}"#);
        assert!(s.ui.show_tool_calls);
        assert!(s.ui.show_tool_results);
        // Surfaced in the startup note only when turned on.
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("showToolCalls=true"), "{note}");
        assert!(note.contains("showToolResults=true"), "{note}");
        assert_eq!(note_for(&Settings::default(), &[]), None);
    }

    #[test]
    fn non_trained_tools_default_off_and_opt_in() {
        let d = Settings::default();
        assert!(!d.tools.task && !d.tools.agent && !d.tools.plan_mode);
        let s = from_json(r#"{"tools":{"task":true,"agent":true,"planMode":true}}"#);
        assert!(s.tools.task && s.tools.agent && s.tools.plan_mode);
        // Only the enabled (non-default) flags surface in the startup note.
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("tools.task=true"), "{note}");
        assert!(note.contains("tools.agent=true"), "{note}");
        assert!(note.contains("tools.planMode=true"), "{note}");
    }

    #[test]
    fn show_thinking_defaults_on_and_can_be_turned_off() {
        assert!(
            Settings::default().ui.show_thinking,
            "thinking shown by default"
        );
        let s = from_json(r#"{"ui":{"showThinking":false}}"#);
        assert!(!s.ui.show_thinking);
        // Only the non-default (off) value is surfaced in the startup note.
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("showThinking=false"), "{note}");
    }

    #[test]
    fn ask_max_options_defaults_to_seven_and_clamps_up_to_the_minimum() {
        assert_eq!(Settings::default().ask.max_options, 7);
        // A max below the fixed minimum of 2 would make every ask impossible;
        // it clamps up rather than breaking the tool.
        assert_eq!(from_json(r#"{"ask":{"maxOptions":1}}"#).ask.max_options, 2);
        assert_eq!(from_json(r#"{"ask":{"maxOptions":0}}"#).ask.max_options, 2);
        assert_eq!(
            from_json(r#"{"ask":{"maxOptions":12}}"#).ask.max_options,
            12
        );
    }

    #[test]
    fn a_later_file_overlays_only_the_keys_it_sets() {
        let mut s = from_json(r#"{"ui":{"popupRows":25,"historySize":4096}}"#);
        s.overlay(r#"{"ui":{"popupRows":5}}"#);
        assert_eq!(s.ui.popup_rows, 5, "later file wins");
        assert_eq!(s.ui.history_size, 4096, "untouched key survives");
    }

    #[test]
    fn malformed_json_leaves_the_defaults_intact() {
        // A broken settings file must not stop plank from starting.
        for bad in ["", "{", "not json at all", "[]", "null"] {
            assert_eq!(from_json(bad), Settings::default(), "input {bad:?}");
        }
    }

    #[test]
    fn a_wrongly_typed_value_falls_back_to_its_default() {
        let s = from_json(r#"{"ui":{"popupRows":"lots","respectGitignore":"yes"},"mcp":{}}"#);
        assert_eq!(s.ui.popup_rows, 15);
        assert!(s.ui.respect_gitignore);
        assert_eq!(s.mcp.timeout_secs, 30);
    }

    #[test]
    fn zero_and_negative_sizes_are_rejected_rather_than_disabling_the_feature() {
        let s = from_json(r#"{"ui":{"popupRows":0,"historySize":-3},"mcp":{"timeoutSecs":0}}"#);
        assert_eq!(s.ui.popup_rows, 15);
        assert_eq!(s.ui.history_size, 512);
        assert_eq!(s.mcp.timeout_secs, 30);
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let s = from_json(r#"{"ui":{"popupRows":7,"futureKey":1},"newGroup":{"x":2}}"#);
        assert_eq!(s.ui.popup_rows, 7);
    }

    #[test]
    fn a_non_finite_number_does_not_saturate_into_a_value() {
        // `as i64` on a huge f64 saturates rather than failing, so the guard
        // has to reject it before the cast.
        let s = from_json(r#"{"ui":{"popupRows":1e309}}"#);
        assert_eq!(s.ui.popup_rows, 15);
    }

    fn note_for(s: &Settings, args: &[&str]) -> Option<String> {
        let flags: Vec<String> = args.iter().map(ToString::to_string).collect();
        let cfg = crate::config::parse_options_with(s, &flags).unwrap();
        startup_note(s, &cfg)
    }

    #[test]
    fn no_note_when_settings_change_nothing() {
        assert_eq!(note_for(&Settings::default(), &[]), None);
    }

    #[test]
    fn the_note_names_the_slow_settings() {
        // The exact situation that made plank mysteriously slow: a settings
        // file quietly moved it off Metal onto the CPU.
        let s = from_json(r#"{"engine":{"backend":"cpu","threads":3,"ctx":65536}}"#);
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("backend=cpu"), "{note}");
        assert!(note.contains("threads=3"), "{note}");
        assert!(note.contains("ctx=65536"), "{note}");
    }

    #[test]
    fn a_setting_a_flag_overrode_is_not_reported() {
        // The note must describe what is in force, never what a file asked
        // for: reporting `backend=cpu` while running on Metal would send
        // someone chasing the wrong cause.
        let s = from_json(r#"{"engine":{"backend":"cpu","threads":3}}"#);
        let note = note_for(&s, &["--metal"]).expect("threads still applies");
        assert!(!note.contains("backend"), "{note}");
        assert!(note.contains("threads=3"), "{note}");
        assert_eq!(note_for(&s, &["--metal", "-t", "16"]), None);
    }

    #[test]
    fn ui_and_mcp_keys_are_reported_since_no_flag_can_override_them() {
        let s = from_json(r#"{"ui":{"popupRows":4,"historySize":7},"mcp":{"timeoutSecs":45}}"#);
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("popupRows=4"), "{note}");
        assert!(note.contains("historySize=7"), "{note}");
        assert!(note.contains("timeoutSecs=45"), "{note}");
    }

    #[test]
    fn save_to_round_trips_and_preserves_unknown_keys() {
        let dir = std::env::temp_dir().join(format!("plank-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        // Seed a file with a key this binary does not know about.
        std::fs::write(&path, "{\"future\":{\"nope\":1},\"ui\":{\"popupRows\":3}}").unwrap();

        let mut s = Settings::default();
        s.ui.show_thinking = false;
        s.ui.popup_rows = 9;
        s.mcp.timeout_secs = 45;
        s.engine.ctx = Some(8192);
        s.engine.backend = None; // unset -> absent
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("\"future\""),
            "unknown section preserved:\n{text}"
        );
        assert!(!text.contains("backend"), "unset optional omitted");

        let mut reloaded = Settings::default();
        reloaded.overlay(&text);
        assert!(!reloaded.ui.show_thinking);
        assert_eq!(reloaded.ui.popup_rows, 9);
        assert_eq!(reloaded.mcp.timeout_secs, 45);
        assert_eq!(reloaded.engine.ctx, Some(8192));
        assert_eq!(reloaded.engine.backend, None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn active_is_the_defaults_until_installed() {
        // Tests never call `install`, so every consumer sees the defaults.
        assert_eq!(active().ui.popup_rows, 15);
    }

    #[test]
    fn notification_defaults_and_overlay() {
        let s = Settings::default();
        assert!(s.ui.notifications);
        assert_eq!(s.ui.notify_after_secs, 10);

        let mut s = Settings::default();
        s.overlay(r#"{ "ui": { "notifications": false, "notifyAfterSecs": 30 } }"#);
        assert!(!s.ui.notifications);
        assert_eq!(s.ui.notify_after_secs, 30);

        // Bad value ignored, default retained.
        let mut s = Settings::default();
        s.overlay(r#"{ "ui": { "notifyAfterSecs": "nope" } }"#);
        assert_eq!(s.ui.notify_after_secs, 10);
    }
}
