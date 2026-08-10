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

/// Every plugin activated for a session, plus set-level load complaints.
#[derive(Debug, Clone, Default)]
pub struct PluginSet {
    /// Loaded plugins, in increasing precedence.
    pub plugins: Vec<Plugin>,
    /// Non-fatal complaints not attributable to a single loaded plugin.
    pub warnings: Vec<String>,
}

impl PluginSet {
    /// Every warning worth showing: set-level first, then per-plugin.
    #[must_use]
    pub fn all_warnings(&self) -> Vec<String> {
        let mut out = self.warnings.clone();
        for p in &self.plugins {
            out.extend(p.warnings.iter().cloned());
        }
        out
    }
}

/// Immediate subdirectories of `root`, sorted by name for a stable load order.
fn subdirs(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// Loads every activated plugin given an explicit home directory: scans
/// `<home>/.plank/plugins/dev/*`, then `<cwd>/.plank/plugins/*`, then each
/// `--plugin-dir` in the order given. A later source replaces an earlier
/// plugin of the same name. `home` is `None` when there is no home directory
/// to scan, in which case the user-scan source contributes nothing.
#[must_use]
pub fn load_in(home: Option<&Path>, cwd: &Path, cli_dirs: &[PathBuf]) -> PluginSet {
    let mut candidates: Vec<(PathBuf, Origin)> = Vec::new();
    if let Some(home) = home {
        let root = home.join(".plank").join("plugins").join("dev");
        candidates.extend(subdirs(&root).into_iter().map(|d| (d, Origin::UserScan)));
    }
    let project = cwd.join(".plank").join("plugins");
    candidates.extend(
        subdirs(&project)
            .into_iter()
            .map(|d| (d, Origin::ProjectScan)),
    );
    candidates.extend(cli_dirs.iter().map(|d| (d.clone(), Origin::CliDir)));

    let mut set = PluginSet::default();
    let mut seen_roots: Vec<PathBuf> = Vec::new();
    for (dir, origin) in candidates {
        if origin == Origin::CliDir && !dir.is_dir() {
            set.warnings
                .push(format!("--plugin-dir {}: not a directory", dir.display()));
            continue;
        }
        let Some(plugin) = load_plugin(&dir, origin) else {
            if origin == Origin::CliDir {
                set.warnings.push(format!(
                    "--plugin-dir {}: no plugin.json and no components",
                    dir.display()
                ));
            }
            continue;
        };
        if seen_roots.contains(&plugin.root) {
            continue;
        }
        seen_roots.push(plugin.root.clone());
        if let Some(slot) = set.plugins.iter_mut().find(|p| p.name == plugin.name) {
            set.warnings.push(format!(
                "plugin '{}' from {} shadows the one from {}",
                plugin.name,
                plugin.origin.label(),
                slot.origin.label()
            ));
            *slot = plugin;
        } else {
            set.plugins.push(plugin);
        }
    }
    set
}

/// Loads every activated plugin: `~/.plank/plugins/dev/*`, then
/// `<cwd>/.plank/plugins/*`, then each `--plugin-dir` in the order given.
/// A later source replaces an earlier plugin of the same name.
#[must_use]
pub fn load_default(cwd: &Path, cli_dirs: &[PathBuf]) -> PluginSet {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    load_in(home.as_deref(), cwd, cli_dirs)
}

/// A contribution addressed by name — skills, agents and templates all are.
/// Implemented here rather than in each module so the namespacing rule lives
/// in exactly one place.
pub trait Named {
    /// The name the entry is currently addressed by.
    fn name(&self) -> &str;
    /// Renames the entry, used to apply the `<plugin>:<name>` alias.
    fn set_name(&mut self, name: String);
}

impl Named for crate::skills::Skill {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

impl Named for crate::agents::AgentDef {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

impl Named for crate::templates::Template {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

/// The directory (or file) a plugin contributes for one component, preferring
/// the plank spelling over the Claude Code one. `None` when neither exists.
#[must_use]
pub fn component_root(plugin: &Plugin, plank: &str, cc: &str) -> Option<PathBuf> {
    let preferred = plugin.root.join(plank);
    if preferred.exists() {
        return Some(preferred);
    }
    let fallback = plugin.root.join(cc);
    if fallback.exists() {
        return Some(fallback);
    }
    None
}

/// Merges plugin-contributed entries into the user+project entries.
///
/// Every plugin entry is registered as `<plugin>:<name>`. It is registered a
/// second time under the bare `<name>` only when no non-plugin entry and no
/// other plugin claims that name. Returns the merged list and the collision
/// warnings.
#[must_use]
pub fn reconcile<T: Named + Clone>(
    local: Vec<T>,
    plugin_entries: Vec<(String, T)>,
) -> (Vec<T>, Vec<String>) {
    let mut warnings = Vec::new();
    let mut merged = local;

    // A bare name is contested when a non-plugin entry holds it, or when two
    // plugins both offer it.
    let mut claims: Vec<(String, Vec<String>)> = Vec::new();
    for (plugin, entry) in &plugin_entries {
        let key = entry.name().to_string();
        match claims.iter_mut().find(|(n, _)| *n == key) {
            Some((_, owners)) => owners.push(plugin.clone()),
            None => claims.push((key, vec![plugin.clone()])),
        }
    }

    for (plugin, entry) in plugin_entries {
        let bare = entry.name().to_string();
        let alias = format!("{plugin}:{bare}");
        let owners = claims
            .iter()
            .find(|(n, _)| *n == bare)
            .map(|(_, o)| o.clone())
            .unwrap_or_default();
        let taken_by_local = merged.iter().any(|e| e.name() == bare);

        let mut aliased = entry.clone();
        aliased.set_name(alias.clone());
        merged.push(aliased);

        if taken_by_local {
            warnings.push(format!(
                "'{bare}' is already defined outside plugins; plugin '{plugin}' contributes it as '{alias}'"
            ));
            continue;
        }
        if owners.len() > 1 {
            warnings.push(format!(
                "'{bare}' is contributed by plugins {}; use the namespaced names",
                owners.join(", ")
            ));
            continue;
        }
        merged.push(entry);
    }
    (merged, warnings)
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

    #[test]
    fn project_plugins_load_and_carry_their_origin() {
        let base = scratch("project-scan");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        write(
            &cwd,
            ".plank/plugins/alpha/.plank-plugin/plugin.json",
            r#"{"name":"alpha"}"#,
        );
        let set = load_in(Some(&home), &cwd, &[]);
        assert_eq!(set.plugins.len(), 1);
        assert_eq!(set.plugins[0].name, "alpha");
        assert_eq!(set.plugins[0].origin, Origin::ProjectScan);
    }

    #[test]
    fn a_cli_dir_shadows_an_auto_scanned_plugin_of_the_same_name() {
        let base = scratch("cli-shadow");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        write(
            &cwd,
            ".plank/plugins/alpha/.plank-plugin/plugin.json",
            r#"{"name":"alpha","version":"scanned"}"#,
        );
        let cli = base.join("elsewhere").join("alpha");
        write(
            &cli,
            ".plank-plugin/plugin.json",
            r#"{"name":"alpha","version":"cli"}"#,
        );
        let set = load_in(Some(&home), &cwd, &[cli]);
        assert_eq!(set.plugins.len(), 1);
        assert_eq!(set.plugins[0].version, "cli");
        assert_eq!(set.plugins[0].origin, Origin::CliDir);
        assert!(set.warnings.iter().any(|w| w.contains("shadow")));
    }

    #[test]
    fn the_same_directory_named_twice_loads_once() {
        let base = scratch("dedup");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        let cli = base.join("alpha");
        write(&cli, ".plank-plugin/plugin.json", r#"{"name":"alpha"}"#);
        let set = load_in(Some(&home), &cwd, &[cli.clone(), cli]);
        assert_eq!(set.plugins.len(), 1);
    }

    #[test]
    fn user_scanned_plugins_load_first_with_the_right_origin() {
        let base = scratch("user-scan");
        let home = base.join("home");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        write(
            &home,
            ".plank/plugins/dev/alpha/.plank-plugin/plugin.json",
            r#"{"name":"alpha"}"#,
        );
        let set = load_in(Some(&home), &cwd, &[]);
        assert_eq!(set.plugins.len(), 1);
        assert_eq!(set.plugins[0].name, "alpha");
        assert_eq!(set.plugins[0].origin, Origin::UserScan);
    }

    #[test]
    fn a_missing_cli_dir_warns_without_failing_the_set() {
        let base = scratch("missing-cli");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        let set = load_in(Some(&home), &cwd, &[base.join("nope")]);
        assert!(set.plugins.is_empty());
        assert!(set.warnings.iter().any(|w| w.contains("nope")));
    }

    /// A minimal `Named` stand-in so reconciliation is tested without building
    /// real skills or agents.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Item {
        name: String,
        from: &'static str,
    }

    impl Named for Item {
        fn name(&self) -> &str {
            &self.name
        }
        fn set_name(&mut self, name: String) {
            self.name = name;
        }
    }

    fn item(name: &str, from: &'static str) -> Item {
        Item {
            name: name.to_string(),
            from,
        }
    }

    fn names(items: &[Item]) -> Vec<&str> {
        items.iter().map(|i| i.name.as_str()).collect()
    }

    #[test]
    fn an_uncontested_plugin_entry_gets_both_the_bare_name_and_the_alias() {
        let (merged, warnings) =
            reconcile(vec![], vec![("demo".to_string(), item("greet", "plugin"))]);
        assert_eq!(names(&merged), vec!["demo:greet", "greet"]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn a_user_entry_keeps_the_bare_name_and_the_plugin_keeps_only_the_alias() {
        let (merged, warnings) = reconcile(
            vec![item("greet", "user")],
            vec![("demo".to_string(), item("greet", "plugin"))],
        );
        assert_eq!(names(&merged), vec!["greet", "demo:greet"]);
        assert_eq!(
            merged
                .iter()
                .find(|i| i.name == "greet")
                .expect("present")
                .from,
            "user"
        );
        assert!(warnings.iter().any(|w| w.contains("demo:greet")));
    }

    #[test]
    fn two_colliding_plugins_both_lose_the_bare_name() {
        let (merged, warnings) = reconcile(
            vec![],
            vec![
                ("alpha".to_string(), item("greet", "alpha")),
                ("beta".to_string(), item("greet", "beta")),
            ],
        );
        assert_eq!(names(&merged), vec!["alpha:greet", "beta:greet"]);
        assert!(!merged.iter().any(|i| i.name == "greet"));
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("alpha") && w.contains("beta"))
        );
    }

    #[test]
    fn component_roots_prefer_the_plank_spelling() {
        let dir = scratch("component-root").join("demo");
        write(&dir, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(&dir, "commands/a.md", "x\n");
        write(&dir, "templates/a.md", "x\n");
        let p = load_plugin(&dir, Origin::UserScan).expect("loads");
        let root = component_root(&p, "templates", "commands").expect("some");
        assert!(root.ends_with("templates"));
        assert!(
            p.warnings.is_empty(),
            "shadowing is warned by the caller, not here"
        );
    }

    #[test]
    fn a_component_root_absent_in_both_spellings_is_none() {
        let dir = scratch("no-component").join("demo");
        write(&dir, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        let p = load_plugin(&dir, Origin::UserScan).expect("loads");
        assert!(component_root(&p, "skills", "skills").is_none());
    }
}
