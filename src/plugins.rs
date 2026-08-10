// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Plugins: directories bundling skills, agents, templates, hooks, MCP
//! servers and settings, contributed to a session as one unit.

use std::path::{Path, PathBuf};

/// Where a plugin was activated from, in increasing precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Auto-scanned from `~/.plank/plugins/dev/`.
    UserScan,
    /// Auto-scanned from `<cwd>/.plank/plugins/`.
    ProjectScan,
    /// Named explicitly by `--plugin-dir`.
    CliDir,
}

impl Origin {
    /// Short label used in listings and warnings.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Origin::UserScan => "user",
            Origin::ProjectScan => "project",
            Origin::CliDir => "--plugin-dir",
        }
    }
}

/// One loaded plugin.
#[derive(Debug, Clone)]
pub struct Plugin {
    /// Plugin name; the `<plugin>:` prefix of a namespaced contribution.
    pub name: String,
    /// One-line description shown by `/plugins`.
    pub description: String,
    /// Free-form version string; empty when the manifest omits it.
    pub version: String,
    /// Free-form author string; empty when the manifest omits it.
    pub author: String,
    /// Canonicalized plugin directory.
    pub root: PathBuf,
    /// Activation source.
    pub origin: Origin,
    /// Non-fatal complaints raised while loading this plugin.
    pub warnings: Vec<String>,
}

/// Component subdirectory spellings, plank name first, Claude Code name
/// second. Used by the scan to decide whether a manifest-less directory is a
/// plugin at all, and by the per-contribution accessors in later tasks.
const COMPONENT_PATHS: [(&str, &str); 6] = [
    ("skills", "skills"),
    ("agents", "agents"),
    ("templates", "commands"),
    ("hooks.json", "hooks/hooks.json"),
    (".mcp.json", ".mcp.json"),
    ("settings.json", "settings.json"),
];

/// The manifest plank will read for `dir`, preferring the plank flavor.
/// `None` when the directory has neither manifest.
#[must_use]
pub fn manifest_path(dir: &Path) -> Option<PathBuf> {
    let plank = dir.join(".plank-plugin").join("plugin.json");
    if plank.is_file() {
        return Some(plank);
    }
    let claude = dir.join(".claude-plugin").join("plugin.json");
    if claude.is_file() {
        return Some(claude);
    }
    None
}

/// Whether `dir` holds at least one recognizable component, which is what
/// makes a manifest-less directory still a plugin.
fn has_components(dir: &Path) -> bool {
    COMPONENT_PATHS
        .iter()
        .any(|(plank, cc)| dir.join(plank).exists() || dir.join(cc).exists())
}

/// Reads one top-level string field out of a flat JSON object without pulling
/// in a parser: the manifest fields plank needs are all plain strings, and
/// anything richer belongs to a later sub-project.
fn json_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut rest = text.get(text.find(&needle)? + needle.len()..)?.trim_start();
    rest = rest.strip_prefix(':')?.trim_start();
    let body = rest.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => out.push(chars.next()?),
            _ => out.push(c),
        }
    }
    None
}

/// Whether `name` is usable as a namespace prefix.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains(':')
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains("__")
}

/// Loads one plugin directory. `None` when `dir` is not a plugin at all —
/// neither a manifest nor any recognizable component.
#[must_use]
pub fn load_plugin(dir: &Path, origin: Origin) -> Option<Plugin> {
    let manifest = manifest_path(dir);
    if manifest.is_none() && !has_components(dir) {
        return None;
    }
    let dir_name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let root = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let mut plugin = Plugin {
        name: dir_name.clone(),
        description: String::new(),
        version: String::new(),
        author: String::new(),
        root,
        origin,
        warnings: Vec::new(),
    };
    let Some(manifest) = manifest else {
        plugin.warnings.push(format!(
            "{dir_name}: no plugin.json; named after its directory"
        ));
        return Some(plugin);
    };
    if manifest.ends_with(Path::new(".plank-plugin/plugin.json"))
        && dir.join(".claude-plugin").join("plugin.json").is_file()
    {
        plugin.warnings.push(format!(
            "{dir_name}: .claude-plugin/plugin.json shadowed by .plank-plugin/plugin.json"
        ));
    }
    let Ok(text) = std::fs::read_to_string(&manifest) else {
        plugin
            .warnings
            .push(format!("{dir_name}: unreadable {}", manifest.display()));
        return Some(plugin);
    };
    match json_string_field(&text, "name") {
        Some(name) if valid_name(&name) => plugin.name = name,
        Some(name) => plugin.warnings.push(format!(
            "{dir_name}: invalid name {name:?} in plugin.json; using the directory name"
        )),
        None => plugin
            .warnings
            .push(format!("{dir_name}: unreadable or nameless plugin.json")),
    }
    plugin.description = json_string_field(&text, "description").unwrap_or_default();
    plugin.version = json_string_field(&text, "version").unwrap_or_default();
    plugin.author = json_string_field(&text, "author").unwrap_or_default();
    Some(plugin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Writes `text` to `dir/rel`, creating parent directories.
    fn write(dir: &Path, rel: &str, text: &str) -> PathBuf {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().expect("has parent")).expect("mkdir");
        std::fs::write(&path, text).expect("write");
        path
    }

    /// A unique empty scratch directory under the OS temp dir.
    fn scratch(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("plank-plugins-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mkdir");
        base
    }

    #[test]
    fn a_claude_flavored_manifest_names_the_plugin() {
        let dir = scratch("claude-manifest").join("demo");
        write(
            &dir,
            ".claude-plugin/plugin.json",
            r#"{"name":"demo","description":"A demo","version":"1.2.3","author":"me","dependencies":["other"]}"#,
        );
        let p = load_plugin(&dir, Origin::UserScan).expect("loads");
        assert_eq!(p.name, "demo");
        assert_eq!(p.description, "A demo");
        assert_eq!(p.version, "1.2.3");
        assert_eq!(p.author, "me");
        assert!(p.warnings.is_empty(), "unknown fields must not warn");
    }

    #[test]
    fn the_plank_flavored_manifest_wins_when_both_exist() {
        let dir = scratch("both-manifests").join("demo");
        write(
            &dir,
            ".claude-plugin/plugin.json",
            r#"{"name":"claude-name"}"#,
        );
        write(
            &dir,
            ".plank-plugin/plugin.json",
            r#"{"name":"plank-name"}"#,
        );
        let p = load_plugin(&dir, Origin::UserScan).expect("loads");
        assert_eq!(p.name, "plank-name");
        assert!(p.warnings.iter().any(|w| w.contains(".claude-plugin")));
    }

    #[test]
    fn a_manifestless_directory_with_components_loads_under_its_dir_name() {
        let dir = scratch("manifestless").join("toolbox");
        write(&dir, "skills/greet/SKILL.md", "hi\n");
        let p = load_plugin(&dir, Origin::ProjectScan).expect("loads");
        assert_eq!(p.name, "toolbox");
        assert!(p.warnings.iter().any(|w| w.contains("no plugin.json")));
    }

    #[test]
    fn a_directory_with_neither_manifest_nor_components_is_not_a_plugin() {
        let dir = scratch("empty-dir").join("nothing");
        std::fs::create_dir_all(&dir).expect("mkdir");
        assert!(load_plugin(&dir, Origin::UserScan).is_none());
    }

    #[test]
    fn malformed_manifest_json_falls_back_to_the_directory_name_with_a_warning() {
        let dir = scratch("malformed").join("broken");
        write(&dir, ".plank-plugin/plugin.json", "{ not json");
        let p = load_plugin(&dir, Origin::UserScan).expect("loads");
        assert_eq!(p.name, "broken");
        assert!(p.warnings.iter().any(|w| w.contains("unreadable")));
    }

    #[test]
    fn a_name_with_a_separator_is_rejected_in_favor_of_the_directory_name() {
        let dir = scratch("bad-name").join("safe");
        write(&dir, ".plank-plugin/plugin.json", r#"{"name":"a:b"}"#);
        let p = load_plugin(&dir, Origin::UserScan).expect("loads");
        assert_eq!(p.name, "safe");
        assert!(p.warnings.iter().any(|w| w.contains("invalid name")));
    }
}
