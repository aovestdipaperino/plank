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
        if hex.len() != 6 {
            return None;
        }
        let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
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

    let builtin_tools = block
        .get("tools")
        .and_then(|t| t.get("builtin"))
        .and_then(|b| match b {
            Json::Arr(items) => Some(
                items
                    .iter()
                    .filter_map(|v| match v {
                        Json::Str(s) if !s.is_empty() => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        });

    let settings_json = block
        .get("settings")
        .filter(|s| matches!(s, Json::Obj(_)))
        .map(|s| {
            let mut out = String::new();
            json_write(&mut out, s);
            out
        });

    Some(ProfileSpec {
        display_name: str_field(block, "displayName"),
        logo: str_field(block, "logo").map(|p| resolve(root, &p)),
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
}
