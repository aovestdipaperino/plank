// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Profile specs: the `profile` block of a plugin manifest.

use std::path::{Path, PathBuf};

use crate::tools::mcp::{Json, json_parse, json_write};

/// The accent color a profile paints the UI with.
///
/// Kept as plank's own enum rather than a `ratatui::style::Color` so this
/// module stays a leaf: `src/tui.rs` converts at the point of use, and the
/// plain-stdout path emits its own ANSI from the same value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accent {
    /// An ANSI 256-color index.
    Indexed(u8),
    /// A 24-bit color.
    Rgb(u8, u8, u8),
}

/// The `profile` block of a plugin manifest, with paths already resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSpec {
    /// Name shown in the banner, window title and status bar. Falls back to
    /// the plugin name when absent.
    pub display_name: Option<String>,
    /// PNG rendered as the startup banner; the plank logo when absent.
    pub logo: Option<PathBuf>,
    /// UI accent; the built-in green when absent.
    pub accent: Option<Accent>,
    /// The file holding the whole system prompt. The one required field.
    pub system_prompt: PathBuf,
    /// Builtin tools the profile allows. `None` means all of them.
    pub builtin_tools: Option<Vec<String>>,
    /// The `settings` object, re-serialized so `crate::settings` can overlay
    /// it with the same text-based path it uses for every settings file.
    pub settings_json: Option<String>,
    /// Non-fatal complaints raised while parsing.
    pub warnings: Vec<String>,
    /// `folderContext`: whether the session starts with context about the
    /// launch folder (its git status and `<folder>/.plank/MEMORY.md`).
    /// `false` unless the manifest says `true`.
    pub folder_context: bool,
    /// `agentsMd`: whether `AGENTS.md` files are read, and offered or linked
    /// at launch. `false` unless the manifest says `true`.
    pub agents_md: bool,
}

impl ProfileSpec {
    /// Whether builtin tool `name` is offered under this profile.
    ///
    /// The single source of truth: both `crate::tools::dispatch` and the
    /// system-prompt schema block call this, so the model can never be told
    /// about a tool dispatch would reject.
    #[must_use]
    pub fn builtin_enabled(&self, name: &str) -> bool {
        match &self.builtin_tools {
            None => true,
            Some(allowed) => allowed.iter().any(|t| t == name),
        }
    }
}

/// Parses an ANSI index (`"160"`) or a hex triple (`"#c04040"`).
#[must_use]
pub fn parse_accent(s: &str) -> Option<Accent> {
    if let Some(hex) = s.strip_prefix('#') {
        // Count and index by chars, not bytes: a non-ASCII manifest value
        // (e.g. a euro sign) can be 6 bytes without being 6 hex digits, and
        // slicing by byte offset into such a string panics on a non-char
        // boundary. Collecting to a Vec<char> keeps every index valid.
        let chars: Vec<char> = hex.chars().collect();
        if chars.len() != 6 {
            return None;
        }
        let byte = |i: usize| -> Option<u8> {
            let pair: String = chars[i..i + 2].iter().collect();
            u8::from_str_radix(&pair, 16).ok()
        };
        return Some(Accent::Rgb(byte(0)?, byte(2)?, byte(4)?));
    }
    s.parse::<u8>().ok().map(Accent::Indexed)
}

/// Parses the `profile` block out of a plugin manifest.
///
/// `None` when the manifest does not parse, has no `profile` object, or that
/// object has no `systemPrompt` — a block without a prompt is a skin, and
/// activating a skin over plank's own prompt would misrepresent what is
/// running. Every other malformed field yields a warning and a default.
#[must_use]
pub fn parse(manifest_text: &str, root: &Path) -> Option<ProfileSpec> {
    let root_json = json_parse(manifest_text)?;
    let block = root_json.get("profile")?;
    if !matches!(block, Json::Obj(_)) {
        return None;
    }
    let system_prompt = str_field(block, "systemPrompt")?;
    let mut warnings = Vec::new();

    let accent = match str_field(block, "accent") {
        None => None,
        Some(raw) => {
            let parsed = parse_accent(&raw);
            if parsed.is_none() {
                warnings.push(format!(
                    "profile: unrecognized accent {raw:?}; using the default"
                ));
            }
            parsed
        }
    };

    let builtin_tools = match block.get("tools").and_then(|t| t.get("builtin")) {
        None => None,
        Some(Json::Arr(items)) => {
            let mut bad = false;
            let names: Vec<String> = items
                .iter()
                .filter_map(|v| match v {
                    Json::Str(s) if !s.is_empty() => Some(s.clone()),
                    _ => {
                        bad = true;
                        None
                    }
                })
                .collect();
            if bad {
                warnings.push(
                    "profile: tools.builtin has a non-string or empty entry; skipping it"
                        .to_string(),
                );
            }
            warn_unknown_builtin_names(&names, &mut warnings);
            Some(names)
        }
        Some(_) => {
            // Fails closed, not open: `None` means "every builtin allowed",
            // so a malformed restriction must not silently widen access to
            // everything — an empty allow-list is the safe default here.
            warnings.push(
                "profile: tools.builtin is not an array of strings; allowing no builtins"
                    .to_string(),
            );
            Some(Vec::new())
        }
    };

    let display_name = match block.get("displayName") {
        None => None,
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        Some(_) => {
            warnings.push(
                "profile: displayName is not a non-empty string; using the default".to_string(),
            );
            None
        }
    };

    let logo = match block.get("logo") {
        None => None,
        Some(Json::Str(s)) if !s.is_empty() => Some(resolve(root, s)),
        Some(_) => {
            warnings.push("profile: logo is not a non-empty string; using the default".to_string());
            None
        }
    };

    let settings_json = match block.get("settings") {
        None => None,
        Some(s @ Json::Obj(members)) => {
            warn_refused_engine_settings(members, &mut warnings);
            let mut out = String::new();
            json_write(&mut out, s);
            Some(out)
        }
        Some(_) => {
            warnings.push("profile: settings is not an object; using the default".to_string());
            None
        }
    };

    let folder_context = bool_field(block, "folderContext", &mut warnings);
    let agents_md = bool_field(block, "agentsMd", &mut warnings);

    Some(ProfileSpec {
        display_name,
        logo,
        accent,
        system_prompt: resolve(root, &system_prompt),
        builtin_tools,
        settings_json,
        warnings,
        folder_context,
        agents_md,
    })
}

/// Pushes a warning for each `names` entry that matches no known builtin.
///
/// An entry matching nothing is a warning, not an error: a profile written
/// against a newer plank (one that added a builtin this binary does not have
/// yet) must still run, just with that entry silently inert.
fn warn_unknown_builtin_names(names: &[String], warnings: &mut Vec<String>) {
    let known = crate::sysprompt::known_builtin_names();
    for name in names {
        if !known.contains(name) {
            warnings.push(format!(
                "profile: tools.builtin names {name:?}, which matches no known builtin tool"
            ));
        }
    }
}

/// Pushes a warning for each `engine.*` key a profile's `settings` block
/// tries to set.
///
/// `crate::settings::Settings::overlay_from` drops `engine.*` from a
/// profile's settings layer the same way it drops it from a plugin's, but
/// silently — the caller there has no per-key warning channel to surface it
/// on. This is that surfacing: one warning per dropped key, at parse time,
/// the same shape `plugins.rs` emits for its own refused sections.
fn warn_refused_engine_settings(members: &[(String, Json)], warnings: &mut Vec<String>) {
    let Some((_, engine_value)) = members.iter().find(|(k, _)| k == "engine") else {
        return;
    };
    match engine_value {
        Json::Obj(engine_keys) if !engine_keys.is_empty() => {
            for (key, _) in engine_keys {
                warnings.push(format!(
                    "profile: settings.engine.{key} is refused (a profile may not set engine.*); set it yourself in ~/.plank/settings.json if you want it"
                ));
            }
        }
        _ => warnings.push(
            "profile: settings.engine is refused (a profile may not set engine.*); set it yourself in ~/.plank/settings.json if you want it"
                .to_string(),
        ),
    }
}

/// A non-empty string member, or `None`.
/// A boolean profile field: `false` when absent, and `false` with a warning
/// when it is not a boolean, so a typo never turns a context source on.
fn bool_field(obj: &Json, key: &str, warnings: &mut Vec<String>) -> bool {
    match obj.get(key) {
        None => false,
        Some(Json::Bool(b)) => *b,
        Some(_) => {
            warnings.push(format!("profile: {key} is not true or false; using false"));
            false
        }
    }
}

fn str_field(obj: &Json, key: &str) -> Option<String> {
    match obj.get(key) {
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Resolves a manifest path against the plugin root, leaving absolute paths be.
fn resolve(root: &Path, p: &str) -> PathBuf {
    let p = Path::new(p);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

/// The profile a run is operating under: the plugin name that selected it and
/// its parsed spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveProfile {
    /// The plugin name `--profile` named.
    pub name: String,
    /// The parsed block.
    pub spec: ProfileSpec,
    /// The contents of `spec.system_prompt`, read exactly once at
    /// activation (`resolve_and_activate_profile` in `main.rs`).
    ///
    /// Composition (`sysprompt.rs`) then uses this text and never touches
    /// the filesystem itself, so a prompt file that is deleted or edited
    /// mid-session cannot make composition fail or drift: the process reads
    /// it once, before there is a TUI to corrupt.
    pub prompt: String,
}

/// The active profile, set once at startup.
static ACTIVE: std::sync::OnceLock<ActiveProfile> = std::sync::OnceLock::new();

/// Installs the active profile. The first call wins; later calls are ignored,
/// which is what makes every reader below infallible.
pub fn install(active: ActiveProfile) {
    let _ = ACTIVE.set(active);
}

/// The active profile, or `None` for a plain plank run.
#[must_use]
pub fn active() -> Option<&'static ActiveProfile> {
    ACTIVE.get()
}

/// The active profile's plugin name, or `None`.
#[must_use]
pub fn active_name() -> Option<&'static str> {
    ACTIVE.get().map(|a| a.name.as_str())
}

/// What to call the agent in the banner, window title and status bar:
/// the profile's `displayName`, else its plugin name, else `plank`.
#[must_use]
pub fn display_name() -> &'static str {
    match ACTIVE.get() {
        Some(a) => a.spec.display_name.as_deref().unwrap_or(&a.name),
        None => "plank",
    }
}

// Test-only override for the active profile, consulted before the process
// global `ACTIVE` `OnceLock`.
//
// `ACTIVE` can only be set once per process, which makes it useless for
// testing the `dispatch` guard's wiring across multiple cases in the same
// test binary: whichever test runs first would poison every test after it.
// This thread-local lets a test install (and, on drop, remove) its own
// `ProfileSpec` without touching `ACTIVE` at all. Thread-local rather than
// process-global so tests running on different threads never see each
// other's override.
#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::RefCell<Option<ProfileSpec>> = const { std::cell::RefCell::new(None) };
}

/// RAII guard that installs a `ProfileSpec` into [`TEST_OVERRIDE`] for the
/// life of a test and clears it on drop — including an unwinding drop from a
/// panicking test, so a failing test can never leak its override into the
/// next test on the same thread.
#[cfg(test)]
pub(crate) struct TestProfileGuard;

#[cfg(test)]
impl TestProfileGuard {
    /// Installs `spec` as the active profile for the current thread until
    /// the returned guard is dropped.
    pub(crate) fn install(spec: ProfileSpec) -> Self {
        TEST_OVERRIDE.with(|cell| *cell.borrow_mut() = Some(spec));
        Self
    }
}

#[cfg(test)]
impl Drop for TestProfileGuard {
    fn drop(&mut self) {
        TEST_OVERRIDE.with(|cell| *cell.borrow_mut() = None);
    }
}

/// Whether the session starts with launch-folder context (git status and
/// project memory). Always true without a profile; a profile opts in with
/// `folderContext: true`.
#[must_use]
pub fn folder_context_enabled() -> bool {
    #[cfg(test)]
    {
        if let Some(on) =
            TEST_OVERRIDE.with(|cell| cell.borrow().as_ref().map(|s| s.folder_context))
        {
            return on;
        }
    }
    ACTIVE.get().is_none_or(|a| a.spec.folder_context)
}

/// Whether `AGENTS.md` is read, offered and linked. Always true without a
/// profile; a profile opts in with `agentsMd: true`.
#[must_use]
pub fn agents_md_enabled() -> bool {
    #[cfg(test)]
    {
        if let Some(on) = TEST_OVERRIDE.with(|cell| cell.borrow().as_ref().map(|s| s.agents_md)) {
            return on;
        }
    }
    ACTIVE.get().is_none_or(|a| a.spec.agents_md)
}

/// Whether builtin tool `name` is offered under the active profile.
/// True for every tool when no profile is active.
#[must_use]
pub fn builtin_enabled(name: &str) -> bool {
    #[cfg(test)]
    {
        if let Some(enabled) = TEST_OVERRIDE.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|spec| spec.builtin_enabled(name))
        }) {
            return enabled;
        }
    }
    ACTIVE.get().is_none_or(|a| a.spec.builtin_enabled(name))
}

/// What `--profile NAME` came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// No `--profile` was given; run as plain plank.
    None,
    /// Activate this profile.
    Activate(ActiveProfile),
    /// A bare `--profile`: list these names and exit successfully.
    List(Vec<String>),
    /// The name matched no plugin. Carries the available profile names so the
    /// error can show them.
    NoSuchPlugin(String, Vec<String>),
    /// The name matched a plugin that declares no `profile` block.
    NotAProfile(String),
    /// `--profile ""`: an explicit empty name, distinct from the bare-flag
    /// listing request. Fatal — see finding 4 in the profile cleanup notes.
    EmptyName,
}

/// Resolves `--profile`'s argument against the loaded plugins.
///
/// `explicit_empty` distinguishes `--profile ""` (a fatal error: the user
/// supplied an empty name) from a bare `--profile` (the listing request) —
/// both otherwise parse to `requested == Some("")`.
///
/// Failure is deliberately not silent: running as plain plank when the user
/// asked for HAL is worse than not running, so every miss is a distinct
/// variant the caller turns into a fatal message.
#[must_use]
pub fn resolve_profile(
    requested: Option<&str>,
    explicit_empty: bool,
    set: &crate::plugins::PluginSet,
) -> Resolution {
    let Some(name) = requested else {
        return Resolution::None;
    };
    if name.is_empty() {
        if explicit_empty {
            return Resolution::EmptyName;
        }
        return Resolution::List(crate::plugins::profile_names(set));
    }
    let Some(plugin) = set.plugins.iter().find(|p| p.name == name) else {
        return Resolution::NoSuchPlugin(name.to_string(), crate::plugins::profile_names(set));
    };
    match &plugin.profile {
        Some(spec) => Resolution::Activate(ActiveProfile {
            name: plugin.name.clone(),
            spec: spec.clone(),
            // Filled in by `resolve_and_activate_profile` (main.rs) once it
            // has confirmed the file is readable; this function has no
            // filesystem access and stays pure.
            prompt: String::new(),
        }),
        None => Resolution::NotAProfile(name.to_string()),
    }
}

/// The message shown when `--profile` is combined with a Qwen model.
///
/// Qwen's prompt is a different document with a different schema fence, built
/// whole by `sysprompt::build_qwen_tools_prompt_parts`; a profile's
/// replacement prose does not reach it. Everything else about the profile —
/// name, logo, accent, builtin allow-list — *would* apply, so the failure mode
/// without this refusal is plank's own prose wearing the profile's identity,
/// which is worse than not starting.
#[must_use]
pub fn refuse_under_qwen(name: &str) -> String {
    format!(
        concat!(
            "plank: profile {name:?} cannot run on a Qwen model: a profile replaces the system ",
            "prompt, and the Qwen prompt is built separately\n",
            "plank: drop --profile, or run it on a DeepSeek model"
        ),
        name = name
    )
}

/// The load-time warning for a profile prompt that never asks for the tool
/// protocol, or `None` when it does.
///
/// Without [`crate::sysprompt::TOOL_PROTOCOL_TOKEN`] the model is handed the
/// schema block but not the DSML call syntax, so it can see its tools and
/// has no way to call them. That can be a deliberate chat-only profile, so
/// it warns rather than refusing.
#[must_use]
pub fn missing_protocol_warning(name: &str, prompt: &str) -> Option<String> {
    (!prompt.contains(crate::sysprompt::TOOL_PROTOCOL_TOKEN)).then(|| {
        format!(
            "plank: profile {name:?}: the prompt has no {} token, so the model is not told how to call its tools",
            crate::sysprompt::TOOL_PROTOCOL_TOKEN
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_context_and_agents_md_default_to_false() {
        let root = Path::new("/p");
        let spec = parse(r#"{"profile":{"systemPrompt":"p.md"}}"#, root).expect("parses");
        assert!(!spec.folder_context && !spec.agents_md);
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","folderContext":true,"agentsMd":true}}"#,
            root,
        )
        .expect("parses");
        assert!(spec.folder_context && spec.agents_md);
        assert!(spec.warnings.is_empty(), "{:?}", spec.warnings);
    }

    #[test]
    fn a_non_boolean_context_flag_warns_and_stays_off() {
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","folderContext":"yes","agentsMd":1}}"#,
            Path::new("/p"),
        )
        .expect("parses");
        assert!(!spec.folder_context && !spec.agents_md);
        assert_eq!(spec.warnings.len(), 2, "{:?}", spec.warnings);
        assert!(spec.warnings[0].contains("folderContext"));
    }

    #[test]
    fn the_context_switches_follow_the_profile_and_stay_on_without_one() {
        assert!(
            folder_context_enabled() && agents_md_enabled(),
            "no profile: unchanged"
        );
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","agentsMd":true}}"#,
            Path::new("/p"),
        )
        .expect("parses");
        let _guard = TestProfileGuard::install(spec);
        assert!(!folder_context_enabled());
        assert!(agents_md_enabled());
    }

    #[test]
    fn a_prompt_without_the_protocol_token_warns_once_by_name() {
        let w = missing_protocol_warning("chatbgt", "You are ChatBGT.\n")
            .expect("no token, so a warning");
        assert!(w.contains("\"chatbgt\""), "{w}");
        assert!(w.contains(crate::sysprompt::TOOL_PROTOCOL_TOKEN), "{w}");
        let with = format!(
            "You are ChatBGT.\n\n{}\n",
            crate::sysprompt::TOOL_PROTOCOL_TOKEN
        );
        assert_eq!(missing_protocol_warning("chatbgt", &with), None);
    }

    #[test]
    fn the_qwen_refusal_names_the_profile_and_the_reason() {
        let msg = refuse_under_qwen("hal");
        assert!(msg.contains("hal"), "{msg}");
        assert!(msg.contains("Qwen"), "{msg}");
        // The user needs to know what to do, not merely that it failed.
        assert!(msg.contains("--profile"), "{msg}");
    }

    const FULL: &str = r#"{
      "name": "hal",
      "profile": {
        "displayName": "HAL",
        "logo": "assets/hal.png",
        "accent": "160",
        "systemPrompt": "prompt.md",
        "tools": { "builtin": ["bash", "ask"] },
        "settings": { "ui": { "showThinking": false } }
      }
    }"#;

    #[test]
    fn parses_every_field_and_resolves_paths_against_root() {
        let root = Path::new("/plugins/hal");
        let spec = parse(FULL, root).expect("has a profile block");
        assert_eq!(spec.display_name.as_deref(), Some("HAL"));
        assert_eq!(
            spec.logo.as_deref(),
            Some(Path::new("/plugins/hal/assets/hal.png"))
        );
        assert_eq!(spec.accent, Some(Accent::Indexed(160)));
        assert_eq!(spec.system_prompt, Path::new("/plugins/hal/prompt.md"));
        assert_eq!(
            spec.builtin_tools.as_deref(),
            Some(&["bash".to_string(), "ask".to_string()][..])
        );
        assert!(spec.settings_json.is_some());
        assert!(spec.warnings.is_empty());
    }

    #[test]
    fn a_profiles_engine_settings_are_dropped_loudly() {
        // Finding 5: `Settings::overlay_from` drops `engine.*` from a
        // profile's settings layer the same way it drops a plugin's, but
        // silently. This is where that becomes audible: one warning per
        // dropped key, mirroring `settings_audit_warnings`'s per-key shape
        // for a plugin's refused sections.
        let text = r#"{
          "profile": {
            "systemPrompt": "prompt.md",
            "settings": { "engine": { "model": "evil.gguf", "threads": 99 }, "ui": { "showThinking": false } }
          }
        }"#;
        let spec = parse(text, Path::new("/p")).expect("still parses as a profile");
        assert!(
            spec.warnings
                .iter()
                .any(|w| w.contains("engine.model") && w.contains("refused")),
            "expected a warning naming engine.model, got: {:?}",
            spec.warnings
        );
        assert!(
            spec.warnings
                .iter()
                .any(|w| w.contains("engine.threads") && w.contains("refused")),
            "expected a warning naming engine.threads, got: {:?}",
            spec.warnings
        );
        // The refusal itself (dropping the key) is `settings.rs`'s job, not
        // this parser's — it still serializes the block as given.
        assert!(spec.settings_json.unwrap().contains("\"engine\""));
    }

    #[test]
    fn settings_without_an_engine_block_warns_about_nothing() {
        let text = r#"{
          "profile": {
            "systemPrompt": "prompt.md",
            "settings": { "ui": { "showThinking": false } }
          }
        }"#;
        let spec = parse(text, Path::new("/p")).expect("still parses as a profile");
        assert!(
            spec.warnings.is_empty(),
            "no engine block, no warnings expected: {:?}",
            spec.warnings
        );
    }

    #[test]
    fn a_manifest_without_a_profile_block_is_not_a_profile() {
        let text = r#"{ "name": "plain", "description": "no profile" }"#;
        assert!(parse(text, Path::new("/p")).is_none());
    }

    #[test]
    fn a_profile_block_without_a_system_prompt_is_not_a_profile() {
        // systemPrompt is the identity; without it the block is a skin, and
        // silently activating a skin over the plank prompt would surprise.
        let text = r#"{ "profile": { "displayName": "X" } }"#;
        assert!(parse(text, Path::new("/p")).is_none());
    }

    #[test]
    fn unparseable_json_is_not_a_profile() {
        assert!(parse("{ not json", Path::new("/p")).is_none());
    }

    #[test]
    fn an_allow_list_entry_matching_no_builtin_warns_but_still_loads() {
        // Finding 3: a name a newer plank added (or a plain typo) must not
        // fail the whole profile — the design spec requires it to run, just
        // with that entry inert, and to say so through the warning channel.
        let text = r#"{
          "profile": {
            "systemPrompt": "prompt.md",
            "tools": { "builtin": ["bash", "not_a_real_tool"] }
          }
        }"#;
        let spec = parse(text, Path::new("/p")).expect("still parses as a profile");
        assert_eq!(
            spec.builtin_tools.as_deref(),
            Some(&["bash".to_string(), "not_a_real_tool".to_string()][..]),
            "the unknown entry stays in the allow-list, it is not dropped"
        );
        assert!(
            spec.warnings.iter().any(|w| w.contains("not_a_real_tool")),
            "expected a warning naming the unknown entry, got: {:?}",
            spec.warnings
        );
        assert!(spec.builtin_enabled("bash"));
    }

    #[test]
    fn a_recognized_native_extra_in_the_allow_list_warns_about_nothing() {
        // Companion: `glob` is a real builtin (a native extra, not one of
        // the twelve C-parsed schemas) so it must not trip the finding-3
        // warning the way a genuinely unknown name does.
        let text = r#"{
          "profile": {
            "systemPrompt": "prompt.md",
            "tools": { "builtin": ["glob", "ask"] }
          }
        }"#;
        let spec = parse(text, Path::new("/p")).expect("still parses as a profile");
        assert!(
            spec.warnings.is_empty(),
            "glob and ask are real builtins; got warnings: {:?}",
            spec.warnings
        );
    }

    #[test]
    fn a_bad_accent_warns_and_leaves_the_default() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "accent": "chartreuse" } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.accent, None);
        assert_eq!(spec.warnings.len(), 1);
        assert!(spec.warnings[0].contains("accent"));
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "futureThing": 3 } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert!(spec.warnings.is_empty());
    }

    #[test]
    fn accent_parses_an_ansi_index_and_a_hex_triple() {
        assert_eq!(parse_accent("0"), Some(Accent::Indexed(0)));
        assert_eq!(parse_accent("255"), Some(Accent::Indexed(255)));
        assert_eq!(parse_accent("#c04040"), Some(Accent::Rgb(0xc0, 0x40, 0x40)));
        assert_eq!(parse_accent("#C04040"), Some(Accent::Rgb(0xc0, 0x40, 0x40)));
        assert_eq!(parse_accent("256"), None);
        assert_eq!(parse_accent("#c040"), None);
        assert_eq!(parse_accent(""), None);
    }

    #[test]
    fn an_absent_allow_list_enables_every_builtin() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert!(spec.builtin_enabled("read"));
        assert!(spec.builtin_enabled("bash"));
    }

    #[test]
    fn an_allow_list_enables_only_what_it_names() {
        let spec = parse(FULL, Path::new("/p")).expect("a profile");
        assert!(spec.builtin_enabled("bash"));
        assert!(spec.builtin_enabled("ask"));
        assert!(!spec.builtin_enabled("read"));
    }

    #[test]
    fn an_absolute_path_in_the_manifest_is_used_as_is() {
        let text = r#"{ "profile": { "systemPrompt": "/etc/p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert_eq!(spec.system_prompt, Path::new("/etc/p.md"));
    }

    // FINDING 1: parse_accent must never panic, no matter the byte/char shape
    // of its input.

    #[test]
    fn accent_does_not_panic_on_a_six_byte_non_ascii_string() {
        // "#\u{20ac}000" is a '#' followed by 6 bytes (the euro sign is 3
        // bytes, "000" is 3 more), so the old byte-length check let a
        // char-boundary slice through.
        assert_eq!(parse_accent("#\u{20ac}000"), None);
    }

    #[test]
    fn accent_does_not_panic_on_a_multi_byte_char_of_non_six_byte_length() {
        assert_eq!(parse_accent("#\u{1f600}"), None);
        assert_eq!(parse_accent("#\u{1f600}\u{1f600}"), None);
    }

    #[test]
    fn accent_does_not_panic_on_an_empty_string() {
        assert_eq!(parse_accent(""), None);
        assert_eq!(parse_accent("#"), None);
    }

    // FINDING 2: malformed-but-present optional fields warn instead of
    // silently defaulting; absent fields stay silent.

    #[test]
    fn a_wrong_type_display_name_warns_and_defaults() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "displayName": 3 } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.display_name, None);
        assert!(spec.warnings.iter().any(|w| w.contains("displayName")));
    }

    #[test]
    fn an_absent_display_name_is_silent() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert_eq!(spec.display_name, None);
        assert!(spec.warnings.is_empty());
    }

    #[test]
    fn a_wrong_type_logo_warns_and_defaults() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "logo": 3 } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.logo, None);
        assert!(spec.warnings.iter().any(|w| w.contains("logo")));
    }

    #[test]
    fn an_absent_logo_is_silent() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert!(spec.warnings.is_empty());
    }

    #[test]
    fn a_wrong_type_settings_warns_and_defaults() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "settings": "nope" } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.settings_json, None);
        assert!(spec.warnings.iter().any(|w| w.contains("settings")));
    }

    #[test]
    fn an_absent_settings_is_silent() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert!(spec.warnings.is_empty());
    }

    #[test]
    fn a_wrong_type_tools_builtin_warns_and_fails_closed() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "tools": { "builtin": "bash" } } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        // Fails closed: an empty allow-list, not None (which would mean
        // "everything allowed") — a malformed restriction must restrict.
        assert_eq!(spec.builtin_tools.as_deref(), Some(&[][..]));
        assert!(!spec.builtin_enabled("bash"));
        assert!(!spec.builtin_enabled("read"));
        assert!(spec.warnings.iter().any(|w| w.contains("tools.builtin")));
    }

    #[test]
    fn an_absent_tools_builtin_is_silent() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert_eq!(spec.builtin_tools, None);
        assert!(spec.warnings.is_empty());
    }

    #[test]
    fn tools_builtin_array_with_bad_elements_warns_and_skips_them() {
        let text = r#"{ "profile": { "systemPrompt": "p.md",
            "tools": { "builtin": ["bash", 3, "", "read"] } } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(
            spec.builtin_tools.as_deref(),
            Some(&["bash".to_string(), "read".to_string()][..])
        );
        assert!(spec.warnings.iter().any(|w| w.contains("tools.builtin")));
    }

    // FINDING 3: warnings from independent malformed fields all accumulate.

    #[test]
    fn multiple_malformed_fields_accumulate_warnings_and_parse_still_succeeds() {
        let text = r#"{ "profile": {
            "systemPrompt": "p.md",
            "accent": "chartreuse",
            "displayName": 3,
            "tools": { "builtin": "bash" }
        } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert!(spec.warnings.iter().any(|w| w.contains("accent")));
        assert!(spec.warnings.iter().any(|w| w.contains("displayName")));
        assert!(spec.warnings.iter().any(|w| w.contains("tools.builtin")));
        assert_eq!(spec.warnings.len(), 3);
    }

    fn spec_named(tools: Option<Vec<String>>) -> ProfileSpec {
        ProfileSpec {
            display_name: Some("HAL".to_string()),
            logo: None,
            accent: None,
            system_prompt: PathBuf::from("/p/prompt.md"),
            builtin_tools: tools,
            settings_json: None,
            warnings: Vec::new(),
            folder_context: false,
            agents_md: false,
        }
    }

    #[test]
    fn the_predicate_follows_the_installed_spec() {
        // Exercised directly on the spec rather than through the global, so it
        // does not depend on test ordering within the process.
        let spec = spec_named(Some(vec!["bash".to_string()]));
        assert!(spec.builtin_enabled("bash"));
        assert!(!spec.builtin_enabled("read"));
    }

    #[test]
    fn no_flag_resolves_to_no_profile() {
        let set = crate::plugins::PluginSet::default();
        assert!(matches!(
            resolve_profile(None, false, &set),
            Resolution::None
        ));
    }

    #[test]
    fn a_bare_flag_lists_the_available_profiles() {
        let set = crate::plugins::PluginSet::default();
        match resolve_profile(Some(""), false, &set) {
            Resolution::List(names) => assert!(names.is_empty()),
            other => panic!("expected a listing, got {other:?}"),
        }
    }

    #[test]
    fn an_explicit_empty_name_is_fatal_not_a_listing() {
        // Finding 4: `--profile ""` must not collide with the bare-flag
        // listing sentinel — both parse `requested` to `Some("")`, so only
        // `explicit_empty` tells them apart.
        let set = crate::plugins::PluginSet::default();
        assert!(matches!(
            resolve_profile(Some(""), true, &set),
            Resolution::EmptyName
        ));
    }

    #[test]
    fn an_unknown_name_reports_what_is_available() {
        let set = crate::plugins::PluginSet::default();
        match resolve_profile(Some("nope"), false, &set) {
            Resolution::NoSuchPlugin(name, available) => {
                assert_eq!(name, "nope");
                assert!(available.is_empty());
            }
            other => panic!("expected NoSuchPlugin, got {other:?}"),
        }
    }

    #[test]
    fn a_plugin_without_a_profile_block_is_refused_by_name() {
        let (dir, set) = set_with_plugin("plain", r#"{"name":"plain"}"#);
        let _ = &dir;
        match resolve_profile(Some("plain"), false, &set) {
            Resolution::NotAProfile(name) => assert_eq!(name, "plain"),
            other => panic!("expected NotAProfile, got {other:?}"),
        }
    }

    #[test]
    fn a_profile_bearing_plugin_activates() {
        let (dir, set) = set_with_plugin(
            "hal",
            r#"{"name":"hal","profile":{"systemPrompt":"prompt.md","displayName":"HAL"}}"#,
        );
        let _ = &dir;
        match resolve_profile(Some("hal"), false, &set) {
            Resolution::Activate(a) => {
                assert_eq!(a.name, "hal");
                assert_eq!(a.spec.display_name.as_deref(), Some("HAL"));
            }
            other => panic!("expected Activate, got {other:?}"),
        }
    }

    /// A one-plugin `PluginSet` on disk, in a unique scratch directory under
    /// the OS temp dir (mirroring `src/plugins.rs`'s own test helper, since
    /// this project has no `tempfile` dev-dependency). The directory is
    /// returned so the caller can keep it alive for the duration of the
    /// assertion, though nothing here deletes it early.
    fn set_with_plugin(name: &str, manifest: &str) -> (PathBuf, crate::plugins::PluginSet) {
        let base = std::env::temp_dir().join(format!(
            "plank-profile-resolve-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join(name);
        std::fs::create_dir_all(dir.join(".plank-plugin")).expect("mkdir");
        std::fs::write(dir.join(".plank-plugin").join("plugin.json"), manifest).expect("write");
        std::fs::write(dir.join("prompt.md"), "You are a test agent.\n").expect("write prompt");
        let set = crate::plugins::load_in_with(None, &base, std::slice::from_ref(&dir));
        (base, set)
    }

    /// The shipped `chatbgt` profile must stay parseable and warning-free. A
    /// typo in a builtin name or an `engine.*` key in its settings would
    /// otherwise only be noticed by someone running it.
    #[test]
    fn the_shipped_chatbgt_profile_parses_without_warnings() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/profiles/chatbgt");
        let manifest = root.join(".plank-plugin").join("plugin.json");
        let text = std::fs::read_to_string(&manifest).expect("chatbgt manifest is readable");
        let spec = parse(&text, &root).expect("chatbgt declares a profile");
        assert_eq!(spec.display_name.as_deref(), Some("ChatBGT"));
        assert!(spec.system_prompt.is_file(), "prompt file is missing");
        assert!(spec.logo.is_none(), "chatbgt exercises the logo fallback");
        assert!(spec.warnings.is_empty(), "{:?}", spec.warnings);
    }
}
