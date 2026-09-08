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
        Some(s @ Json::Obj(_)) => {
            let mut out = String::new();
            json_write(&mut out, s);
            Some(out)
        }
        Some(_) => {
            warnings.push("profile: settings is not an object; using the default".to_string());
            None
        }
    };

    Some(ProfileSpec {
        display_name,
        logo,
        accent,
        system_prompt: resolve(root, &system_prompt),
        builtin_tools,
        settings_json,
        warnings,
    })
}

/// A non-empty string member, or `None`.
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

/// Whether builtin tool `name` is offered under the active profile.
/// True for every tool when no profile is active.
#[must_use]
pub fn builtin_enabled(name: &str) -> bool {
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
}

/// Resolves `--profile`'s argument against the loaded plugins.
///
/// Failure is deliberately not silent: running as plain plank when the user
/// asked for HAL is worse than not running, so every miss is a distinct
/// variant the caller turns into a fatal message.
#[must_use]
pub fn resolve_profile(requested: Option<&str>, set: &crate::plugins::PluginSet) -> Resolution {
    let Some(name) = requested else {
        return Resolution::None;
    };
    if name.is_empty() {
        return Resolution::List(crate::plugins::profile_names(set));
    }
    let Some(plugin) = set.plugins.iter().find(|p| p.name == name) else {
        return Resolution::NoSuchPlugin(name.to_string(), crate::plugins::profile_names(set));
    };
    match &plugin.profile {
        Some(spec) => Resolution::Activate(ActiveProfile {
            name: plugin.name.clone(),
            spec: spec.clone(),
        }),
        None => Resolution::NotAProfile(name.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(matches!(resolve_profile(None, &set), Resolution::None));
    }

    #[test]
    fn a_bare_flag_lists_the_available_profiles() {
        let set = crate::plugins::PluginSet::default();
        match resolve_profile(Some(""), &set) {
            Resolution::List(names) => assert!(names.is_empty()),
            other => panic!("expected a listing, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_name_reports_what_is_available() {
        let set = crate::plugins::PluginSet::default();
        match resolve_profile(Some("nope"), &set) {
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
        match resolve_profile(Some("plain"), &set) {
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
        match resolve_profile(Some("hal"), &set) {
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
        let set = crate::plugins::load_in(None, &base, std::slice::from_ref(&dir));
        (base, set)
    }
}
