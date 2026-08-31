// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Interactive editor for the persisted user settings (`/config`).
//!
//! Front-end-agnostic core: a [`ConfigForm`] holds a working copy of
//! [`Settings`], a cursor over the editable [`FIELDS`], and an optional inline
//! edit buffer. [`ConfigForm::handle_key`] drives it from key events and
//! [`ConfigForm::rows`] yields render-ready rows; the TUI (`tui::draw_config`)
//! and the plain REPL both build on these without any terminal logic here.
//!
//! The same [`FIELDS`] table backs the textual `/config <key> <value>` setter
//! (see [`set_from_path`]) so both paths address settings identically, e.g.
//! `ui.showThinking` or `engine.ctx`.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::settings::Settings;

/// Which settings field a [`Field`] targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldId {
    EngineModel,
    EngineThreads,
    EngineBackend,
    EnginePower,
    EngineCtx,
    EngineThinkingToolCalls,
    UiRespectGitignore,
    UiPopupRows,
    UiIndexRefreshSecs,
    UiHistorySize,
    UiShowToolCalls,
    UiShowToolResults,
    UiShowThinking,
    UiNotifications,
    UiNotifyAfterSecs,
    UiCrtOff,
    UiScreensaver,
    UiScreensaverFace,
    UiReducedMotion,
    UiEasterEggs,
    UiBuiltinEditor,
    SafetySandbox,
    SafetyBtwSuspend,
    McpTimeoutSecs,
    AskMaxOptions,
    AgentsAutoRoute,
    AgentsMaxParallel,
    GitSignCommits,
    ToolsRepeatAdvisory,
    ToolsCallTimeoutSec,
    ToolsSpillMaxBytes,
    ToolsSpillPreviewBytes,
    ToolsRecall,
    ToolsFanout,
    ToolsRunCode,
    ContextMicrocompact,
}

/// The editing shape of a field, which decides how a key press mutates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A plain on/off flag (Enter/Space toggles).
    Bool,
    /// An `Option<bool>` cycled default → on → off → default.
    Tri,
    /// A positive integer (Enter opens an inline edit).
    Count,
    /// An optional signed integer; empty clears it to unset.
    OptInt,
    /// Optional free text; empty clears it to unset.
    OptText,
}

/// One editable settings field: how to address, label, and edit it.
#[derive(Debug)]
pub struct Field {
    /// Which concrete setting this row maps to.
    pub id: FieldId,
    /// `settings.json` section (also the group header).
    pub section: &'static str,
    /// `settings.json` camelCase key, used by the textual setter.
    pub key: &'static str,
    /// Human-facing label.
    pub label: &'static str,
    /// Editing shape.
    pub kind: Kind,
}

/// The full set of editable fields, in display order (grouped by section).
pub static FIELDS: &[Field] = &[
    f(
        FieldId::EngineModel,
        "engine",
        "model",
        "model file",
        Kind::OptText,
    ),
    f(
        FieldId::EngineThreads,
        "engine",
        "threads",
        "worker threads",
        Kind::OptInt,
    ),
    f(
        FieldId::EngineBackend,
        "engine",
        "backend",
        "backend (metal/cuda/cpu)",
        Kind::OptText,
    ),
    f(
        FieldId::EnginePower,
        "engine",
        "power",
        "GPU power cap %",
        Kind::OptInt,
    ),
    f(
        FieldId::EngineCtx,
        "engine",
        "ctx",
        "context window (tokens)",
        Kind::OptInt,
    ),
    f(
        FieldId::EngineThinkingToolCalls,
        "engine",
        "thinkingToolCalls",
        "dispatch tool calls made inside thinking",
        Kind::Bool,
    ),
    f(
        FieldId::UiRespectGitignore,
        "ui",
        "respectGitignore",
        "@ honours .gitignore",
        Kind::Bool,
    ),
    f(
        FieldId::UiPopupRows,
        "ui",
        "popupRows",
        "@ popup rows",
        Kind::Count,
    ),
    f(
        FieldId::UiIndexRefreshSecs,
        "ui",
        "indexRefreshSecs",
        "file index refresh (s)",
        Kind::Count,
    ),
    f(
        FieldId::UiHistorySize,
        "ui",
        "historySize",
        "prompt history entries",
        Kind::Count,
    ),
    f(
        FieldId::UiShowToolCalls,
        "ui",
        "showToolCalls",
        "show tool-call banners",
        Kind::Bool,
    ),
    f(
        FieldId::UiShowToolResults,
        "ui",
        "showToolResults",
        "echo tool results",
        Kind::Bool,
    ),
    f(
        FieldId::UiShowThinking,
        "ui",
        "showThinking",
        "show thinking text",
        Kind::Bool,
    ),
    f(
        FieldId::UiNotifications,
        "ui",
        "notifications",
        "macOS desktop notifications: always, unfocused, never",
        Kind::Bool,
    ),
    f(
        FieldId::UiNotifyAfterSecs,
        "ui",
        "notifyAfterSecs",
        "min turn seconds before a complete notification",
        Kind::Count,
    ),
    f(
        FieldId::UiCrtOff,
        "ui",
        "crtOff",
        "CRT power-off animation on TUI exit",
        Kind::Bool,
    ),
    f(
        FieldId::UiScreensaver,
        "ui",
        "screensaver",
        "idle screensaver after: 1m, 2m, 5m, never",
        Kind::Bool,
    ),
    f(
        FieldId::UiScreensaverFace,
        "ui",
        "screensaverFace",
        "which screensaver (built-in or plugin face)",
        Kind::Bool,
    ),
    f(
        FieldId::UiReducedMotion,
        "ui",
        "reducedMotion",
        "reduce motion: freeze TUI animations",
        Kind::Bool,
    ),
    f(
        FieldId::UiEasterEggs,
        "ui",
        "easterEggs",
        "arcade easter eggs exist as commands",
        Kind::Bool,
    ),
    f(
        FieldId::UiBuiltinEditor,
        "ui",
        "builtinEditor",
        "Ctrl-G opens the built-in editor",
        Kind::Bool,
    ),
    f(
        FieldId::SafetySandbox,
        "safety",
        "sandbox",
        "bash write sandbox",
        Kind::Tri,
    ),
    f(
        FieldId::SafetyBtwSuspend,
        "safety",
        "btwSuspend",
        "/btw mid-gen suspend",
        Kind::Tri,
    ),
    f(
        FieldId::McpTimeoutSecs,
        "mcp",
        "timeoutSecs",
        "MCP server timeout (s)",
        Kind::Count,
    ),
    f(
        FieldId::AskMaxOptions,
        "ask",
        "maxOptions",
        "ask tool max options",
        Kind::Count,
    ),
    f(
        FieldId::AgentsAutoRoute,
        "agents",
        "autoRoute",
        "let the model pick configured agents",
        Kind::Bool,
    ),
    f(
        FieldId::AgentsMaxParallel,
        "agents",
        "maxParallel",
        "max concurrent sub-agents",
        Kind::Count,
    ),
    f(
        FieldId::GitSignCommits,
        "git",
        "signCommits",
        "sign commits with plank's trailer",
        Kind::Bool,
    ),
    f(
        FieldId::ToolsRepeatAdvisory,
        "tools",
        "repeatAdvisory",
        "nudge the model when it repeats a tool call",
        Kind::Bool,
    ),
    f(
        FieldId::ToolsCallTimeoutSec,
        "tools",
        "callTimeoutSec",
        "per-call deadline (s); 0 = off",
        Kind::Count,
    ),
    f(
        FieldId::ToolsSpillMaxBytes,
        "tools",
        "spillMaxBytes",
        "spill a tool result larger than this (bytes)",
        Kind::Count,
    ),
    f(
        FieldId::ToolsSpillPreviewBytes,
        "tools",
        "spillPreviewBytes",
        "inline preview bytes of a spilled result",
        Kind::Count,
    ),
    f(
        FieldId::ToolsRecall,
        "tools",
        "recall",
        "offer the recall tool to the model",
        Kind::Bool,
    ),
    f(
        FieldId::ToolsFanout,
        "tools",
        "fanout",
        "offer the fanout tool to the model",
        Kind::Bool,
    ),
    f(
        FieldId::ToolsRunCode,
        "tools",
        "runCode",
        "offer the run_code tool to the model",
        Kind::Bool,
    ),
    f(
        FieldId::ContextMicrocompact,
        "context",
        "microcompact",
        "clear old tool-result bodies to reclaim context",
        Kind::Bool,
    ),
];

const fn f(
    id: FieldId,
    section: &'static str,
    key: &'static str,
    label: &'static str,
    kind: Kind,
) -> Field {
    Field {
        id,
        section,
        key,
        label,
        kind,
    }
}

/// Minimum for the `ask` tool, mirrored from [`crate::settings`].
const ASK_MIN_OPTIONS: usize = 2;

/// Current display value of a field, e.g. `true`, `30`, `(unset)`.
#[must_use]
pub fn display(s: &Settings, id: FieldId) -> String {
    fn optnum<T: ToString>(v: Option<T>) -> String {
        v.map_or_else(|| "(unset)".to_string(), |x| x.to_string())
    }
    match id {
        FieldId::EngineModel => s
            .engine
            .model
            .as_ref()
            .map_or_else(|| "(unset)".to_string(), |p| p.display().to_string()),
        FieldId::EngineThreads => optnum(s.engine.threads),
        FieldId::EngineBackend => s
            .engine
            .backend
            .clone()
            .unwrap_or_else(|| "(unset)".to_string()),
        FieldId::EnginePower => optnum(s.engine.power),
        FieldId::EngineCtx => optnum(s.engine.ctx),
        FieldId::EngineThinkingToolCalls => s.engine.thinking_tool_calls.to_string(),
        FieldId::UiRespectGitignore => s.ui.respect_gitignore.to_string(),
        FieldId::UiPopupRows => s.ui.popup_rows.to_string(),
        FieldId::UiIndexRefreshSecs => s.ui.index_refresh_secs.to_string(),
        FieldId::UiHistorySize => s.ui.history_size.to_string(),
        FieldId::UiShowToolCalls => s.ui.show_tool_calls.to_string(),
        FieldId::UiShowToolResults => s.ui.show_tool_results.to_string(),
        FieldId::UiShowThinking => s.ui.show_thinking.to_string(),
        FieldId::UiNotifications => s.ui.notifications.as_str().to_string(),
        FieldId::UiNotifyAfterSecs => s.ui.notify_after_secs.to_string(),
        FieldId::UiCrtOff => s.ui.crt_off.to_string(),
        FieldId::UiScreensaver => s.ui.screensaver.as_str().to_string(),
        // A contributed face is shown as the user wrote it: `<plugin>:<face>`.
        FieldId::UiScreensaverFace => {
            s.ui.screensaver_face_plugin
                .clone()
                .unwrap_or_else(|| s.ui.screensaver_face.as_str().to_string())
        }
        FieldId::UiReducedMotion => s.ui.reduced_motion.to_string(),
        FieldId::UiEasterEggs => s.ui.easter_eggs.to_string(),
        FieldId::UiBuiltinEditor => s.ui.builtin_editor.to_string(),
        FieldId::SafetySandbox => tri_str(s.safety.sandbox),
        FieldId::SafetyBtwSuspend => tri_str(s.safety.btw_suspend),
        FieldId::McpTimeoutSecs => s.mcp.timeout_secs.to_string(),
        FieldId::AskMaxOptions => s.ask.max_options.to_string(),
        FieldId::AgentsAutoRoute => s.agents.auto_route.to_string(),
        FieldId::AgentsMaxParallel => s.agents.max_parallel.to_string(),
        FieldId::GitSignCommits => s.git.sign_commits.to_string(),
        FieldId::ToolsRepeatAdvisory => s.tools.repeat_advisory.to_string(),
        FieldId::ToolsCallTimeoutSec => s.tools.call_timeout_sec.to_string(),
        FieldId::ToolsSpillMaxBytes => s.tools.spill_max_bytes.to_string(),
        FieldId::ToolsSpillPreviewBytes => s.tools.spill_preview_bytes.to_string(),
        FieldId::ToolsRecall => s.tools.recall.to_string(),
        FieldId::ToolsFanout => s.tools.fanout.to_string(),
        FieldId::ToolsRunCode => s.tools.run_code.to_string(),
        FieldId::ContextMicrocompact => s.context.microcompact.to_string(),
    }
}

fn tri_str(v: Option<bool>) -> String {
    match v {
        None => "(default)".to_string(),
        Some(true) => "true".to_string(),
        Some(false) => "false".to_string(),
    }
}

/// Toggles a `Bool`/`Tri` field in place; a no-op for other kinds.
fn toggle(s: &mut Settings, id: FieldId) {
    match id {
        FieldId::EngineThinkingToolCalls => {
            s.engine.thinking_tool_calls = !s.engine.thinking_tool_calls;
        }
        FieldId::UiRespectGitignore => s.ui.respect_gitignore = !s.ui.respect_gitignore,
        FieldId::UiShowToolCalls => s.ui.show_tool_calls = !s.ui.show_tool_calls,
        FieldId::UiShowToolResults => s.ui.show_tool_results = !s.ui.show_tool_results,
        FieldId::UiShowThinking => s.ui.show_thinking = !s.ui.show_thinking,
        // Cycles the three notification modes like Tri fields cycle.
        FieldId::UiNotifications => {
            use crate::notify::NotifyMode;
            s.ui.notifications = match s.ui.notifications {
                NotifyMode::Always => NotifyMode::Unfocused,
                NotifyMode::Unfocused => NotifyMode::Never,
                NotifyMode::Never => NotifyMode::Always,
            };
        }
        // Cycles 1m -> 2m -> 5m -> never, like the notification modes.
        FieldId::UiScreensaver => s.ui.screensaver = s.ui.screensaver.cycle(),
        FieldId::UiCrtOff => s.ui.crt_off = !s.ui.crt_off,
        FieldId::UiReducedMotion => s.ui.reduced_motion = !s.ui.reduced_motion,
        FieldId::UiEasterEggs => s.ui.easter_eggs = !s.ui.easter_eggs,
        FieldId::UiBuiltinEditor => s.ui.builtin_editor = !s.ui.builtin_editor,
        FieldId::ContextMicrocompact => s.context.microcompact = !s.context.microcompact,
        FieldId::SafetySandbox => s.safety.sandbox = cycle_tri(s.safety.sandbox),
        FieldId::SafetyBtwSuspend => s.safety.btw_suspend = cycle_tri(s.safety.btw_suspend),
        // `UiScreensaverFace` lands here deliberately: its cycle spans the
        // contributed faces too, which this value-only function cannot see, so
        // `ConfigForm` handles that field itself.
        _ => {}
    }
}

fn cycle_tri(v: Option<bool>) -> Option<bool> {
    match v {
        None => Some(true),
        Some(true) => Some(false),
        Some(false) => None,
    }
}

/// Applies a raw string to a `Count`/`OptInt`/`OptText` field, validating it.
///
/// An empty string clears an optional field to unset. Returns a human error on
/// a bad number or an out-of-range value.
///
/// # Errors
/// Returns `Err` when the field expects a number and `raw` is not one (or is
/// zero/negative where the field requires a positive value).
pub fn set_value(s: &mut Settings, id: FieldId, raw: &str) -> Result<(), String> {
    let raw = raw.trim();
    let empty = raw.is_empty();
    let parse_i32 = || {
        raw.parse::<i32>()
            .map_err(|_| format!("not a number: {raw}"))
    };
    let parse_pos = |min: u64| {
        raw.parse::<u64>()
            .map_err(|_| format!("not a number: {raw}"))
            .and_then(|v| {
                if v < min {
                    Err(format!("must be at least {min}"))
                } else {
                    Ok(v)
                }
            })
    };
    match id {
        FieldId::EngineModel => s.engine.model = if empty { None } else { Some(raw.into()) },
        FieldId::EngineBackend => {
            if empty {
                s.engine.backend = None;
            } else if matches!(raw, "metal" | "cuda" | "cpu") {
                s.engine.backend = Some(raw.to_string());
            } else {
                return Err(format!("backend must be metal, cuda, or cpu (got {raw})"));
            }
        }
        FieldId::EngineThreads => s.engine.threads = if empty { None } else { Some(parse_i32()?) },
        FieldId::EnginePower => s.engine.power = if empty { None } else { Some(parse_i32()?) },
        FieldId::EngineCtx => s.engine.ctx = if empty { None } else { Some(parse_i32()?) },
        FieldId::UiPopupRows => {
            s.ui.popup_rows = usize::try_from(parse_pos(1)?).unwrap_or(usize::MAX);
        }
        FieldId::UiIndexRefreshSecs => s.ui.index_refresh_secs = parse_pos(0)?,
        FieldId::UiNotifyAfterSecs => s.ui.notify_after_secs = parse_pos(0)?,
        FieldId::UiHistorySize => {
            s.ui.history_size = usize::try_from(parse_pos(1)?).unwrap_or(usize::MAX);
        }
        FieldId::McpTimeoutSecs => s.mcp.timeout_secs = parse_pos(1)?,
        FieldId::AskMaxOptions => {
            s.ask.max_options =
                usize::try_from(parse_pos(ASK_MIN_OPTIONS as u64)?).unwrap_or(usize::MAX);
        }
        FieldId::AgentsMaxParallel => {
            // Clamped rather than rejected, matching how the settings-file path
            // treats an out-of-range value.
            let n = usize::try_from(parse_pos(1)?).unwrap_or(usize::MAX);
            s.agents.max_parallel = n.min(crate::settings::AGENT_MAX_PARALLEL);
        }
        FieldId::ToolsCallTimeoutSec => s.tools.call_timeout_sec = parse_pos(0)?,
        FieldId::ToolsSpillMaxBytes => {
            s.tools.spill_max_bytes = usize::try_from(parse_pos(1)?).unwrap_or(usize::MAX);
        }
        FieldId::ToolsSpillPreviewBytes => {
            s.tools.spill_preview_bytes = usize::try_from(parse_pos(1)?).unwrap_or(usize::MAX);
        }
        // Bool/Tri fields accept an explicit textual value from the REPL path.
        // Accepts always/unfocused/never, plus the legacy true/false.
        FieldId::UiNotifications => {
            s.ui.notifications = crate::notify::NotifyMode::parse(raw).ok_or_else(|| {
                format!("notifications must be always, unfocused, or never (got {raw})")
            })?;
        }
        FieldId::UiScreensaver => {
            s.ui.screensaver = crate::arcade::ScreensaverDelay::parse(raw)
                .ok_or_else(|| format!("screensaver must be 1m, 2m, 5m, or never (got {raw})"))?;
        }
        FieldId::UiScreensaverFace => {
            // A typed value is either a built-in name or a `<plugin>:<face>`
            // address. The address is not validated against the loaded
            // components on purpose: a user editing settings for a plugin they
            // are about to install should not be told it does not exist, and an
            // address that never resolves costs a fallback to a built-in face,
            // not a broken screensaver.
            if let Some(built_in) = crate::arcade::ScreensaverFace::parse(raw) {
                s.ui.screensaver_face = built_in;
                s.ui.screensaver_face_plugin = None;
            } else if raw.contains(':') {
                s.ui.screensaver_face_plugin = Some(raw.to_string());
            } else {
                return Err(format!(
                    "screensaverFace must be matrix, starfield, minions, random, \
                     or a plugin face like screensavers:starfield (got {raw})"
                ));
            }
        }
        FieldId::EngineThinkingToolCalls
        | FieldId::UiRespectGitignore
        | FieldId::UiShowToolCalls
        | FieldId::UiShowToolResults
        | FieldId::UiShowThinking
        | FieldId::UiCrtOff
        | FieldId::UiReducedMotion
        | FieldId::UiEasterEggs
        | FieldId::UiBuiltinEditor
        | FieldId::AgentsAutoRoute
        | FieldId::GitSignCommits
        | FieldId::ToolsRepeatAdvisory
        | FieldId::ToolsRecall
        | FieldId::ToolsFanout
        | FieldId::ToolsRunCode
        | FieldId::ContextMicrocompact => {
            let b = parse_bool(raw)?;
            set_bool(s, id, b);
        }
        FieldId::SafetySandbox => s.safety.sandbox = parse_tri(raw)?,
        FieldId::SafetyBtwSuspend => s.safety.btw_suspend = parse_tri(raw)?,
    }
    Ok(())
}

fn set_bool(s: &mut Settings, id: FieldId, b: bool) {
    match id {
        FieldId::EngineThinkingToolCalls => s.engine.thinking_tool_calls = b,
        FieldId::UiRespectGitignore => s.ui.respect_gitignore = b,
        FieldId::UiShowToolCalls => s.ui.show_tool_calls = b,
        FieldId::UiShowToolResults => s.ui.show_tool_results = b,
        FieldId::UiShowThinking => s.ui.show_thinking = b,
        FieldId::UiCrtOff => s.ui.crt_off = b,
        FieldId::UiReducedMotion => s.ui.reduced_motion = b,
        FieldId::UiEasterEggs => s.ui.easter_eggs = b,
        FieldId::UiBuiltinEditor => s.ui.builtin_editor = b,
        FieldId::AgentsAutoRoute => s.agents.auto_route = b,
        FieldId::GitSignCommits => s.git.sign_commits = b,
        FieldId::ToolsRepeatAdvisory => s.tools.repeat_advisory = b,
        FieldId::ToolsRecall => s.tools.recall = b,
        FieldId::ToolsFanout => s.tools.fanout = b,
        FieldId::ToolsRunCode => s.tools.run_code = b,
        FieldId::ContextMicrocompact => s.context.microcompact = b,
        _ => {}
    }
}

fn parse_bool(raw: &str) -> Result<bool, String> {
    match raw {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(format!("expected true or false, got {raw}")),
    }
}

fn parse_tri(raw: &str) -> Result<Option<bool>, String> {
    if raw.is_empty() || raw == "default" {
        return Ok(None);
    }
    parse_bool(raw).map(Some)
}

/// Looks up a field by its `section.key` address (e.g. `ui.showThinking`).
#[must_use]
pub fn find(path: &str) -> Option<&'static Field> {
    FIELDS
        .iter()
        .find(|f| path == format!("{}.{}", f.section, f.key))
}

/// Applies a textual `section.key value` change to `s` for the REPL path.
///
/// # Errors
/// Returns `Err` on an unknown key or a value that fails [`set_value`].
pub fn set_from_path(s: &mut Settings, path: &str, value: &str) -> Result<&'static Field, String> {
    let field = find(path).ok_or_else(|| format!("unknown setting: {path}"))?;
    set_value(s, field.id, value)?;
    Ok(field)
}

/// Sets a plugin-declared option from `/config pluginConfig.<id>.<option> <v>`.
///
/// Separate from [`set_from_path`] because the validation lives in the
/// component's declaration rather than in a `FieldId`, so the caller has to
/// supply the declarations. Returns the key that was written.
///
/// # Errors
/// Returns a message when the path names no declared option, or when the value
/// is not one that option accepts — checked against the declaration so the
/// answer is the same one the form would give.
pub fn set_plugin_option(
    s: &mut Settings,
    path: &str,
    value: &str,
    declared: &[(String, crate::wasmreg::ConfigOption)],
) -> Result<String, String> {
    let key = path.strip_prefix("pluginConfig.").ok_or_else(|| {
        format!("not a plugin option: {path} (expected pluginConfig.<id>.<option>)")
    })?;
    // Split from the right: a component id contains dots, the option name does
    // not, so the last segment is the option and everything before it is the id.
    let (id, option) = key
        .rsplit_once('.')
        .ok_or_else(|| format!("not a plugin option: {path}"))?;
    let found = declared
        .iter()
        .find(|(cid, o)| cid == id && o.name == option)
        .ok_or_else(|| format!("no loaded component declares '{option}' on '{id}'"))?;
    if !found.1.accepts(value) {
        return Err(format!(
            "'{value}' is not a valid {option}: {}",
            describe_kind(&found.1.kind)
        ));
    }
    s.plugin_config.insert(key.to_string(), value.to_string());
    Ok(key.to_string())
}

/// One line describing what an option accepts, for an error a user can act on.
fn describe_kind(kind: &crate::wasmreg::ConfigKind) -> String {
    match kind {
        crate::wasmreg::ConfigKind::Enum(values) => format!("one of {}", values.join(", ")),
        crate::wasmreg::ConfigKind::Bool => "true or false".to_string(),
        crate::wasmreg::ConfigKind::Int { min, max } => format!("a number from {min} to {max}"),
        crate::wasmreg::ConfigKind::Text => "any text".to_string(),
    }
}

/// A render-ready row: either a section header or one editable field.
#[derive(Debug)]
pub struct Row {
    /// True for a section header line (no value, no selection).
    pub header: bool,
    /// Left label (section name or field label).
    pub label: String,
    /// Right value column (empty for headers).
    pub value: String,
    /// The cursor is on this field.
    pub selected: bool,
    /// This field is being edited inline (value shows the live buffer).
    pub editing: bool,
}

/// What a key press asks the caller to do with the form.
#[derive(Debug)]
pub enum Outcome {
    /// Stay open; the form absorbed the key.
    Stay,
    /// Close without saving.
    Cancel,
    /// Close and persist these settings. Boxed because the payload dwarfs the
    /// other variants, which would otherwise pay for it on every key press.
    Save(Box<Settings>),
}

/// One plugin-declared option, as a form row.
#[derive(Debug)]
pub struct PluginField {
    /// Component that declared it.
    pub id: String,
    /// The declared option.
    pub option: crate::wasmreg::ConfigOption,
}

impl PluginField {
    /// The settings key this row writes, matching what `render_config` reads.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}.{}", self.id, self.option.name)
    }
}

/// Interactive editor state over a working copy of [`Settings`].
#[derive(Debug)]
pub struct ConfigForm {
    working: Settings,
    cursor: usize,
    edit: Option<String>,
    status: Option<String>,
    /// Options plugin components declared, appended after the static fields.
    ///
    /// Passed in for the same reason `wasm_faces` is: the form is a pure editor
    /// over a `Settings` value and has no session to interrogate. Empty is the
    /// ordinary case and leaves the form exactly as it was.
    plugin_fields: Vec<PluginField>,
    /// Screensaver faces contributed by plugins, as `<plugin>:<face>`.
    ///
    /// Passed in rather than looked up: the form is a pure editor over a
    /// `Settings` value and has no session to ask. Empty is the ordinary case
    /// and makes the face field behave exactly as it did before contributed
    /// faces existed.
    wasm_faces: Vec<String>,
}

impl ConfigForm {
    /// Opens the form seeded from the given (current) settings.
    #[must_use]
    pub fn new(current: Settings) -> Self {
        Self::with_faces(current, Vec::new())
    }

    /// [`new`](Self::new) with the contributed screensaver faces the session
    /// has loaded, so the face field can cycle through them.
    #[must_use]
    pub fn with_faces(current: Settings, wasm_faces: Vec<String>) -> Self {
        Self::with_contributions(current, wasm_faces, Vec::new())
    }

    /// [`with_faces`](Self::with_faces) plus the plugin-declared options this
    /// session loaded, which become rows under a `plugins` section.
    #[must_use]
    pub fn with_contributions(
        current: Settings,
        wasm_faces: Vec<String>,
        plugin_fields: Vec<PluginField>,
    ) -> Self {
        Self {
            working: current,
            cursor: 0,
            edit: None,
            status: None,
            wasm_faces,
            plugin_fields,
        }
    }

    /// Total rows the cursor moves over: the static fields then the plugin ones.
    fn field_count(&self) -> usize {
        FIELDS.len() + self.plugin_fields.len()
    }

    /// The plugin field under the cursor, when the cursor is past the static
    /// fields.
    fn plugin_field(&self) -> Option<&PluginField> {
        self.cursor
            .checked_sub(FIELDS.len())
            .and_then(|i| self.plugin_fields.get(i))
    }

    /// The stored value for a plugin row, or its declared default.
    ///
    /// Falls back when the stored value has stopped being acceptable, matching
    /// what a component is actually handed at open time — a form showing a
    /// value the component would reject would be lying about what is in effect.
    fn plugin_value(&self, field: &PluginField) -> String {
        self.working
            .plugin_config
            .get(&field.key())
            .filter(|v| field.option.accepts(v))
            .cloned()
            .unwrap_or_else(|| field.option.default.clone())
    }

    /// The face field's full cycle: the built-in names, then every contributed
    /// face, then back to the start.
    ///
    /// One list rather than two fields, because "which screensaver" is one
    /// question — a user choosing between the rain and a plugin's starfield is
    /// not thinking about where each came from.
    fn face_cycle(&self) -> Vec<String> {
        let mut out: Vec<String> = crate::arcade::ScreensaverFace::NAMES
            .iter()
            .map(|n| (*n).to_string())
            .collect();
        out.extend(self.wasm_faces.iter().cloned());
        out
    }

    /// The face currently selected, as it is written in settings.
    fn current_face(&self) -> String {
        self.working
            .ui
            .screensaver_face_plugin
            .clone()
            .unwrap_or_else(|| self.working.ui.screensaver_face.as_str().to_string())
    }

    /// Selects `face`, routing it to whichever field can hold it.
    fn set_face(&mut self, face: &str) {
        if let Some(built_in) = crate::arcade::ScreensaverFace::parse(face) {
            self.working.ui.screensaver_face = built_in;
            self.working.ui.screensaver_face_plugin = None;
        } else {
            self.working.ui.screensaver_face_plugin = Some(face.to_string());
        }
    }

    /// The transient status/error line, if any.
    #[must_use]
    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    /// Whether an inline edit is in progress (the caller shows a cursor).
    #[must_use]
    pub fn editing(&self) -> bool {
        self.edit.is_some()
    }

    fn field(&self) -> &'static Field {
        &FIELDS[self.cursor]
    }

    /// Drives the form from one key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        if self.edit.is_some() {
            return self.handle_edit_key(key);
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => return Outcome::Cancel,
            // Escape means cancel everywhere else in plank; it must not be the
            // one place that silently commits a half-typed value. Saving is an
            // explicit act (Ctrl-S) so a reflexive "back out" keystroke never
            // writes to disk.
            KeyCode::Char('q' | 'Q') | KeyCode::Esc => return Outcome::Cancel,
            KeyCode::Char('s') if ctrl => return Outcome::Save(Box::new(self.working.clone())),
            KeyCode::Up => {
                self.status = None;
                self.cursor = self.cursor.checked_sub(1).unwrap_or(self.field_count() - 1);
            }
            KeyCode::Down => {
                self.status = None;
                self.cursor = (self.cursor + 1) % self.field_count();
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.status = None;
                // A plugin row before the static dispatch: its behaviour comes
                // from the component's declaration, not from a `FieldId`.
                if let Some(pf) = self.plugin_field() {
                    let (key, kind, current) =
                        (pf.key(), pf.option.kind.clone(), self.plugin_value(pf));
                    match kind {
                        crate::wasmreg::ConfigKind::Enum(values) => {
                            let next = values
                                .iter()
                                .position(|v| *v == current)
                                .map_or(0, |i| (i + 1) % values.len());
                            if let Some(v) = values.get(next) {
                                self.working.plugin_config.insert(key, v.clone());
                            }
                        }
                        crate::wasmreg::ConfigKind::Bool => {
                            let flipped = if current == "true" { "false" } else { "true" };
                            self.working.plugin_config.insert(key, flipped.to_string());
                        }
                        // Numbers and text open the same inline editor the
                        // static fields use, so one editing gesture covers both.
                        crate::wasmreg::ConfigKind::Int { .. }
                        | crate::wasmreg::ConfigKind::Text => {
                            self.edit = Some(current);
                        }
                    }
                    return Outcome::Stay;
                }
                let field = self.field();
                match field.kind {
                    // The face field cycles through a list the form owns
                    // (built-ins plus whatever this session loaded), so it
                    // cannot go through the value-only `toggle`.
                    Kind::Bool | Kind::Tri if field.id == FieldId::UiScreensaverFace => {
                        let cycle = self.face_cycle();
                        let current = self.current_face();
                        let next = cycle
                            .iter()
                            .position(|f| *f == current)
                            .map_or(0, |i| (i + 1) % cycle.len());
                        if let Some(face) = cycle.get(next).cloned() {
                            self.set_face(&face);
                        }
                    }
                    Kind::Bool | Kind::Tri => toggle(&mut self.working, field.id),
                    Kind::Count | Kind::OptInt | Kind::OptText => {
                        // Seed the buffer from the current value, minus the
                        // "(unset)"/"(default)" placeholders which aren't text.
                        let cur = display(&self.working, field.id);
                        self.edit = Some(if cur.starts_with('(') {
                            String::new()
                        } else {
                            cur
                        });
                    }
                }
            }
            _ => {}
        }
        Outcome::Stay
    }

    fn handle_edit_key(&mut self, key: KeyEvent) -> Outcome {
        let Some(buf) = self.edit.as_mut() else {
            return Outcome::Stay;
        };
        match key.code {
            KeyCode::Esc => {
                self.edit = None;
                self.status = None;
            }
            KeyCode::Backspace => {
                buf.pop();
            }
            KeyCode::Char(c) => buf.push(c),
            KeyCode::Enter => {
                let raw = buf.clone();
                // A plugin row validates against the component's own
                // declaration, which is the only thing that knows the bounds.
                // Rejecting here rather than at open time means the user finds
                // out while they are looking at the field.
                if let Some(pf) = self.plugin_field() {
                    let (key, name) = (pf.key(), pf.option.name.clone());
                    if pf.option.accepts(&raw) {
                        self.working.plugin_config.insert(key, raw);
                        self.edit = None;
                        self.status = None;
                    } else {
                        self.status = Some(format!("'{raw}' is not a valid {name}"));
                    }
                    return Outcome::Stay;
                }
                let id = self.field().id;
                match set_value(&mut self.working, id, &raw) {
                    Ok(()) => {
                        self.edit = None;
                        self.status = None;
                    }
                    Err(e) => self.status = Some(e),
                }
            }
            _ => {}
        }
        Outcome::Stay
    }

    /// Builds the render rows (section headers interleaved with fields).
    #[must_use]
    pub fn rows(&self) -> Vec<Row> {
        let mut rows = Vec::new();
        let mut section = "";
        for (i, field) in FIELDS.iter().enumerate() {
            if field.section != section {
                section = field.section;
                rows.push(Row {
                    header: true,
                    label: section.to_string(),
                    value: String::new(),
                    selected: false,
                    editing: false,
                });
            }
            let selected = i == self.cursor;
            let editing = selected && self.edit.is_some();
            let value = if editing {
                self.edit.clone().unwrap_or_default()
            } else {
                display(&self.working, field.id)
            };
            rows.push(Row {
                header: false,
                label: field.label.to_string(),
                value,
                selected,
                editing,
            });
        }
        // Plugin options last, under one header: they are the only rows whose
        // existence depends on what is installed, so keeping them together
        // means the rest of the form does not move when a plugin is added.
        if !self.plugin_fields.is_empty() {
            rows.push(Row {
                header: true,
                label: "plugins".to_string(),
                value: String::new(),
                selected: false,
                editing: false,
            });
            for (i, pf) in self.plugin_fields.iter().enumerate() {
                let selected = FIELDS.len() + i == self.cursor;
                let editing = selected && self.edit.is_some();
                let value = if editing {
                    self.edit.clone().unwrap_or_default()
                } else {
                    self.plugin_value(pf)
                };
                rows.push(Row {
                    header: false,
                    // Qualified, because two plugins may both declare
                    // "difficulty" and a bare label would be ambiguous.
                    label: pf.key(),
                    value,
                    selected,
                    editing,
                });
            }
        }
        rows
    }
}

/// Appends the plugin-declared options to a `/config` listing.
///
/// Separate function because the caller has the declarations and this module
/// deliberately does not: the form is a pure editor over a `Settings` value.
/// Without this the options are settable but undiscoverable, which is the same
/// as not having them.
#[must_use]
pub fn render_plugin_list(
    s: &Settings,
    declared: &[(String, crate::wasmreg::ConfigOption)],
) -> String {
    use std::fmt::Write as _;
    if declared.is_empty() {
        return String::new();
    }
    let mut out = String::from("  [pluginConfig]\n");
    for (id, option) in declared {
        let key = format!("{id}.{}", option.name);
        let value = s
            .plugin_config
            .get(&key)
            .filter(|v| option.accepts(v))
            .cloned()
            .unwrap_or_else(|| option.default.clone());
        let _ = writeln!(
            out,
            "    pluginConfig.{key:<28} = {value:<12} {}",
            describe_kind(&option.kind)
        );
    }
    out
}

/// Renders the current settings as a plain text table for the REPL `/config`.
#[must_use]
pub fn render_text_list(s: &Settings) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("settings (edit with: /config <key> <value>)\n");
    let mut section = "";
    for field in FIELDS {
        if field.section != section {
            section = field.section;
            let _ = writeln!(out, "  [{section}]");
        }
        let _ = writeln!(
            out,
            "    {}.{:<18} = {:<12} {}",
            field.section,
            field.key,
            display(s, field.id),
            field.label
        );
    }
    out
}

#[cfg(test)]
mod tests {

    /// The textual path both front-ends share: `/config pluginConfig.<id>.<opt>`.
    #[test]
    fn set_plugin_option_validates_and_writes() {
        use crate::wasmreg::{ConfigKind, ConfigOption};
        let declared = vec![
            (
                "dev.plank.demo".to_string(),
                ConfigOption {
                    name: "difficulty".to_string(),
                    kind: ConfigKind::Enum(vec!["easy".into(), "hard".into()]),
                    default: "easy".to_string(),
                },
            ),
            (
                "dev.plank.demo".to_string(),
                ConfigOption {
                    name: "speed".to_string(),
                    kind: ConfigKind::Int { min: 1, max: 9 },
                    default: "4".to_string(),
                },
            ),
        ];
        let mut s = Settings::default();

        let written = set_plugin_option(
            &mut s,
            "pluginConfig.dev.plank.demo.difficulty",
            "hard",
            &declared,
        )
        .expect("accepted");
        // The key written must be exactly what `render_config` looks up, or the
        // setting is stored somewhere nothing reads.
        assert_eq!(written, "dev.plank.demo.difficulty");
        assert_eq!(
            s.plugin_config.get("dev.plank.demo.difficulty"),
            Some(&"hard".to_string())
        );

        // An invalid value is refused, and the message says what is acceptable.
        let err = set_plugin_option(
            &mut s,
            "pluginConfig.dev.plank.demo.difficulty",
            "nightmare",
            &declared,
        )
        .expect_err("refused");
        assert!(err.contains("one of easy, hard"), "{err}");
        // ...and the previous value stands.
        assert_eq!(
            s.plugin_config.get("dev.plank.demo.difficulty"),
            Some(&"hard".to_string())
        );

        let err = set_plugin_option(&mut s, "pluginConfig.dev.plank.demo.speed", "0", &declared)
            .expect_err("out of range");
        assert!(err.contains("from 1 to 9"), "{err}");

        // An option nothing declares is named rather than silently stored: a
        // typo'd key that "works" is a setting the user believes is in effect.
        let err = set_plugin_option(&mut s, "pluginConfig.dev.plank.demo.nope", "x", &declared)
            .expect_err("undeclared");
        assert!(err.contains("no loaded component declares"), "{err}");
        assert_eq!(s.plugin_config.len(), 1);

        // A component id containing dots still resolves: the split is from the
        // right, because ids are reverse-DNS and option names are not.
        assert!(
            set_plugin_option(&mut s, "pluginConfig.dev.plank.demo.speed", "9", &declared).is_ok()
        );
    }

    /// A plugin option becomes a row, cycles on Enter, and lands under the key
    /// the host reads at open time.
    #[test]
    fn plugin_options_become_rows_and_cycle() {
        use crate::wasmreg::{ConfigKind, ConfigOption};
        let field = |name: &str, kind: ConfigKind, default: &str| PluginField {
            id: "dev.plank.demo".to_string(),
            option: ConfigOption {
                name: name.to_string(),
                kind,
                default: default.to_string(),
            },
        };
        let fields = vec![
            field(
                "difficulty",
                ConfigKind::Enum(vec!["easy".into(), "hard".into()]),
                "easy",
            ),
            field("sound", ConfigKind::Bool, "true"),
        ];
        let mut form = ConfigForm::with_contributions(Settings::default(), Vec::new(), fields);

        // Rows: the static fields, then one `plugins` header, then two options
        // labelled by their full key so two plugins declaring "difficulty" stay
        // distinguishable.
        let rows = form.rows();
        assert!(rows.iter().any(|r| r.header && r.label == "plugins"));
        let labels: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        assert!(labels.contains(&"dev.plank.demo.difficulty"), "{labels:?}");
        // Shown value is the declared default until the user sets anything.
        let row = rows
            .iter()
            .find(|r| r.label == "dev.plank.demo.difficulty")
            .unwrap();
        assert_eq!(row.value, "easy");

        // Move to the first plugin row and cycle it.
        for _ in 0..FIELDS.len() {
            form.handle_key(k(KeyCode::Down));
        }
        form.handle_key(k(KeyCode::Enter));
        let saved = &form.working;
        assert_eq!(
            saved.plugin_config.get("dev.plank.demo.difficulty"),
            Some(&"hard".to_string()),
            "Enter should advance the enum"
        );
        // ...and wrap.
        form.handle_key(k(KeyCode::Enter));
        assert_eq!(
            form.working.plugin_config.get("dev.plank.demo.difficulty"),
            Some(&"easy".to_string())
        );

        // The bool row toggles.
        form.handle_key(k(KeyCode::Down));
        form.handle_key(k(KeyCode::Enter));
        assert_eq!(
            form.working.plugin_config.get("dev.plank.demo.sound"),
            Some(&"false".to_string())
        );
    }

    /// An int option edits inline and refuses a value outside its declared
    /// bounds, while the user is still looking at the field.
    #[test]
    fn a_plugin_int_option_validates_against_its_declaration() {
        use crate::wasmreg::{ConfigKind, ConfigOption};
        let fields = vec![PluginField {
            id: "dev.plank.demo".to_string(),
            option: ConfigOption {
                name: "speed".to_string(),
                kind: ConfigKind::Int { min: 1, max: 9 },
                default: "4".to_string(),
            },
        }];
        let mut form = ConfigForm::with_contributions(Settings::default(), Vec::new(), fields);
        for _ in 0..FIELDS.len() {
            form.handle_key(k(KeyCode::Down));
        }
        // Enter opens the editor seeded with the current value.
        form.handle_key(k(KeyCode::Enter));
        assert!(form.editing());
        for _ in 0..3 {
            form.handle_key(k(KeyCode::Backspace));
        }
        for c in "99".chars() {
            form.handle_key(k(KeyCode::Char(c)));
        }
        form.handle_key(k(KeyCode::Enter));
        // Refused: still editing, with a reason, and nothing written.
        assert!(form.editing(), "an out-of-range value must not commit");
        assert!(form.working.plugin_config.is_empty());

        for _ in 0..2 {
            form.handle_key(k(KeyCode::Backspace));
        }
        form.handle_key(k(KeyCode::Char('7')));
        form.handle_key(k(KeyCode::Enter));
        assert!(!form.editing());
        assert_eq!(
            form.working.plugin_config.get("dev.plank.demo.speed"),
            Some(&"7".to_string())
        );
    }

    /// A stored value that has stopped being acceptable displays as the
    /// default, because that is what the component will actually be handed.
    #[test]
    fn a_stale_plugin_value_displays_as_the_default() {
        use crate::wasmreg::{ConfigKind, ConfigOption};
        let mut settings = Settings::default();
        settings.plugin_config.insert(
            "dev.plank.demo.difficulty".to_string(),
            "nightmare".to_string(),
        );
        let fields = vec![PluginField {
            id: "dev.plank.demo".to_string(),
            option: ConfigOption {
                name: "difficulty".to_string(),
                kind: ConfigKind::Enum(vec!["easy".into(), "hard".into()]),
                default: "hard".to_string(),
            },
        }];
        let form = ConfigForm::with_contributions(settings, Vec::new(), fields);
        let row = form
            .rows()
            .into_iter()
            .find(|r| r.label == "dev.plank.demo.difficulty")
            .unwrap();
        assert_eq!(
            row.value, "hard",
            "the form must show what is in effect, not a value the component would reject"
        );
    }

    /// With no plugins the form is exactly what it was: no header, no rows.
    #[test]
    fn no_plugins_means_no_plugin_section() {
        let form = ConfigForm::new(Settings::default());
        let rows = form.rows();
        assert!(!rows.iter().any(|r| r.header && r.label == "plugins"));
        assert_eq!(rows.iter().filter(|r| !r.header).count(), FIELDS.len());
    }
    use super::*;
    use ratatui::crossterm::event::{KeyEvent, KeyModifiers};

    fn k(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    /// Puts the cursor on a field by id, so a test does not depend on how many
    /// fields happen to sit above it.
    fn focus(form: &mut ConfigForm, id: FieldId) {
        form.cursor = FIELDS.iter().position(|f| f.id == id).expect("a field");
    }

    /// The face field cycles the built-ins and then every contributed face,
    /// and comes back round. Without the contributed ones it must behave
    /// exactly as it did before they existed.
    #[test]
    fn the_face_field_cycles_built_ins_then_contributed_faces() {
        let faces = vec![
            "screensavers:starfield".to_string(),
            "screensavers:minions".to_string(),
        ];
        let mut form = ConfigForm::with_faces(Settings::default(), faces);
        focus(&mut form, FieldId::UiScreensaverFace);

        let seen: Vec<String> = (0..6)
            .map(|_| {
                form.handle_key(k(KeyCode::Enter));
                display(&form.working, FieldId::UiScreensaverFace)
            })
            .collect();
        assert_eq!(
            seen,
            vec![
                // One built-in face left, then the contributed ones, then
                // round to the start.
                "random",
                "screensavers:starfield",
                "screensavers:minions",
                "matrix",
                "random",
                "screensavers:starfield",
            ]
        );
    }

    #[test]
    fn the_face_field_is_unchanged_without_contributed_faces() {
        let mut form = ConfigForm::new(Settings::default());
        focus(&mut form, FieldId::UiScreensaverFace);
        let seen: Vec<String> = (0..4)
            .map(|_| {
                form.handle_key(k(KeyCode::Enter));
                display(&form.working, FieldId::UiScreensaverFace)
            })
            .collect();
        assert_eq!(seen, vec!["random", "matrix", "random", "matrix"]);
    }

    /// Selecting a contributed face must route to the field that can hold it,
    /// and selecting a built-in must clear it again — otherwise a user who
    /// tried a plugin face and went back to the rain would keep getting the
    /// plugin, since the pinned address wins.
    #[test]
    fn choosing_a_built_in_face_clears_a_previously_pinned_plugin_face() {
        let mut form = ConfigForm::with_faces(
            Settings::default(),
            vec!["screensavers:starfield".to_string()],
        );
        form.set_face("screensavers:starfield");
        assert_eq!(
            form.working.ui.screensaver_face_plugin.as_deref(),
            Some("screensavers:starfield")
        );

        form.set_face("matrix");
        assert_eq!(
            form.working.ui.screensaver_face_plugin, None,
            "a pinned plugin face outlived the switch back to a built-in"
        );
        assert_eq!(
            form.working.ui.screensaver_face,
            crate::arcade::ScreensaverFace::Matrix
        );
    }

    #[test]
    fn every_field_round_trips_display_and_set() {
        // Each field is addressable and its displayed value is accepted back.
        let mut s = Settings::default();
        for field in FIELDS {
            let path = format!("{}.{}", field.section, field.key);
            assert!(find(&path).is_some(), "{path} not found");
            let shown = display(&s, field.id);
            if !shown.starts_with('(') {
                set_value(&mut s, field.id, &shown).unwrap_or_else(|e| panic!("{path}: {e}"));
            }
        }
    }

    #[test]
    fn toggling_a_bool_flips_it() {
        let mut form = ConfigForm::new(Settings::default());
        // Cursor starts at engine.model; walk to ui.showThinking.
        let idx = FIELDS
            .iter()
            .position(|f| f.id == FieldId::UiShowThinking)
            .unwrap();
        for _ in 0..idx {
            form.handle_key(k(KeyCode::Down));
        }
        assert!(!form.working.ui.show_thinking, "default is off");
        form.handle_key(k(KeyCode::Enter));
        assert!(form.working.ui.show_thinking, "toggled on");
    }

    #[test]
    fn tri_field_cycles_default_on_off() {
        assert_eq!(cycle_tri(None), Some(true));
        assert_eq!(cycle_tri(Some(true)), Some(false));
        assert_eq!(cycle_tri(Some(false)), None);
    }

    #[test]
    fn editing_a_count_parses_and_commits() {
        let mut form = ConfigForm::new(Settings::default());
        let idx = FIELDS
            .iter()
            .position(|f| f.id == FieldId::McpTimeoutSecs)
            .unwrap();
        for _ in 0..idx {
            form.handle_key(k(KeyCode::Down));
        }
        form.handle_key(k(KeyCode::Enter)); // open edit, buffer seeded with "30"
        assert!(form.editing());
        for _ in 0..8 {
            form.handle_key(k(KeyCode::Backspace)); // clear the seed
        }
        for c in "45".chars() {
            form.handle_key(k(KeyCode::Char(c)));
        }
        form.handle_key(k(KeyCode::Enter)); // commit
        assert!(!form.editing());
        assert_eq!(form.working.mcp.timeout_secs, 45);
    }

    #[test]
    fn editing_rejects_a_bad_number_and_stays() {
        let mut form = ConfigForm::new(Settings::default());
        let idx = FIELDS
            .iter()
            .position(|f| f.id == FieldId::McpTimeoutSecs)
            .unwrap();
        for _ in 0..idx {
            form.handle_key(k(KeyCode::Down));
        }
        form.handle_key(k(KeyCode::Enter));
        // Clear the seeded value, then type garbage.
        for _ in 0..8 {
            form.handle_key(k(KeyCode::Backspace));
        }
        for c in "abc".chars() {
            form.handle_key(k(KeyCode::Char(c)));
        }
        form.handle_key(k(KeyCode::Enter));
        assert!(form.editing(), "stays in edit on parse error");
        assert!(form.status().is_some());
        assert_eq!(form.working.mcp.timeout_secs, 30, "unchanged");
    }

    #[test]
    fn esc_and_q_cancel_ctrl_s_saves() {
        let mut form = ConfigForm::new(Settings::default());
        assert!(matches!(form.handle_key(k(KeyCode::Esc)), Outcome::Cancel));
        let mut form2 = ConfigForm::new(Settings::default());
        assert!(matches!(
            form2.handle_key(k(KeyCode::Char('q'))),
            Outcome::Cancel
        ));
        let mut form3 = ConfigForm::new(Settings::default());
        let ctrl_s = KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL);
        assert!(matches!(form3.handle_key(ctrl_s), Outcome::Save(_)));
    }

    /// Escape must discard an edited field rather than commit it: this is the
    /// exact scenario that used to write a half-typed value to disk (a stray
    /// sentence landing in `engine.model` because Escape saved instead of
    /// cancelling).
    #[test]
    fn esc_discards_working_copy() {
        let mut form = ConfigForm::new(Settings::default());
        focus(&mut form, FieldId::EngineModel);
        form.handle_key(k(KeyCode::Enter)); // open the inline editor
        for c in "oops".chars() {
            form.handle_key(k(KeyCode::Char(c)));
        }
        // Still mid-edit: Escape here only closes the inline editor (existing
        // behaviour), so commit the typed value into the working copy first to
        // simulate a value that made it in, then cancel the whole form.
        form.handle_key(k(KeyCode::Enter));
        assert_eq!(
            form.working.engine.model.as_deref(),
            Some(std::path::Path::new("oops"))
        );
        assert!(matches!(form.handle_key(k(KeyCode::Esc)), Outcome::Cancel));
        // Outcome::Cancel carries no settings — the caller must drop
        // `form.working` and leave `active()`/disk untouched, which is exactly
        // what the ui.rs Cancel arm does (no write, no `settings::install`).
    }

    #[test]
    fn set_from_path_validates_backend_and_rejects_unknown() {
        let mut s = Settings::default();
        assert!(set_from_path(&mut s, "engine.backend", "metal").is_ok());
        assert_eq!(s.engine.backend.as_deref(), Some("metal"));
        assert!(set_from_path(&mut s, "engine.backend", "wgpu").is_err());
        assert!(set_from_path(&mut s, "engine.nope", "1").is_err());
        // Empty clears the optional.
        set_from_path(&mut s, "engine.backend", "").unwrap();
        assert_eq!(s.engine.backend, None);
    }

    #[test]
    fn context_microcompact_is_a_known_config_key() {
        let mut s = Settings::default();
        assert!(s.context.microcompact, "on by default");
        let field = set_from_path(&mut s, "context.microcompact", "false").unwrap();
        assert_eq!(field.id, FieldId::ContextMicrocompact);
        assert!(!s.context.microcompact);
    }

    #[test]
    fn crt_off_field_toggles() {
        let mut form = ConfigForm::new(Settings::default());
        // Cursor starts at engine.model; walk to ui.crtOff.
        let idx = FIELDS
            .iter()
            .position(|f| f.id == FieldId::UiCrtOff)
            .unwrap();
        for _ in 0..idx {
            form.handle_key(k(KeyCode::Down));
        }
        assert!(form.working.ui.crt_off, "default on");
        form.handle_key(k(KeyCode::Enter));
        assert!(!form.working.ui.crt_off, "toggled off");
    }

    #[test]
    fn rows_include_section_headers() {
        let form = ConfigForm::new(Settings::default());
        let rows = form.rows();
        let headers: Vec<&str> = rows
            .iter()
            .filter(|r| r.header)
            .map(|r| r.label.as_str())
            .collect();
        assert_eq!(
            headers,
            [
                "engine", "ui", "safety", "mcp", "ask", "agents", "git", "tools", "context"
            ]
        );
        assert_eq!(rows.iter().filter(|r| !r.header).count(), FIELDS.len());
    }
}
