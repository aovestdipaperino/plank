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
        | FieldId::AgentsAutoRoute => {
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

/// Interactive editor state over a working copy of [`Settings`].
#[derive(Debug)]
pub struct ConfigForm {
    working: Settings,
    cursor: usize,
    edit: Option<String>,
    status: Option<String>,
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
        Self {
            working: current,
            cursor: 0,
            edit: None,
            status: None,
            wasm_faces,
        }
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
            KeyCode::Char('q' | 'Q') => return Outcome::Cancel,
            KeyCode::Esc => return Outcome::Save(Box::new(self.working.clone())),
            KeyCode::Up => {
                self.status = None;
                self.cursor = self.cursor.checked_sub(1).unwrap_or(FIELDS.len() - 1);
            }
            KeyCode::Down => {
                self.status = None;
                self.cursor = (self.cursor + 1) % FIELDS.len();
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.status = None;
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
        rows
    }
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
        assert!(form.working.ui.show_thinking, "default is on");
        form.handle_key(k(KeyCode::Enter));
        assert!(!form.working.ui.show_thinking, "toggled off");
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
    fn esc_saves_and_q_cancels() {
        let mut form = ConfigForm::new(Settings::default());
        assert!(matches!(form.handle_key(k(KeyCode::Esc)), Outcome::Save(_)));
        let mut form2 = ConfigForm::new(Settings::default());
        assert!(matches!(
            form2.handle_key(k(KeyCode::Char('q'))),
            Outcome::Cancel
        ));
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
        assert_eq!(headers, ["engine", "ui", "safety", "mcp", "ask", "agents"]);
        assert_eq!(rows.iter().filter(|r| !r.header).count(), FIELDS.len());
    }
}
