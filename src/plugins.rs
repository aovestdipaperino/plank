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
const COMPONENT_PATHS: [(&str, &str); 7] = [
    ("skills", "skills"),
    ("agents", "agents"),
    ("templates", "commands"),
    ("hooks.json", "hooks/hooks.json"),
    (".mcp.json", ".mcp.json"),
    ("settings.json", "settings.json"),
    // WASM components (docs/WASM-PLUGINS.md). No Claude Code spelling exists,
    // so both entries are the plank one — the pair is positional, and leaving
    // the second blank would make `component_root` test `plugin.root` itself,
    // which every plugin has.
    ("wasm", "wasm"),
];

/// Human-readable labels for the listing, in the same order as
/// [`COMPONENT_PATHS`]. Kept beside it and length-checked below so adding or
/// reordering a component can't silently truncate or mislabel `contributions`.
const COMPONENT_LABELS: [&str; 7] = [
    "skills",
    "agents",
    "templates",
    "hooks",
    "mcp",
    "settings",
    "wasm",
];

const _: () = assert!(COMPONENT_LABELS.len() == COMPONENT_PATHS.len());

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

/// Warnings for every `safety.*` key a plugin's `settings.json` sets.
///
/// Plugin settings are applied strictly below `~/.plank/settings.json`, so a
/// plugin can never override a key the user set. But `safety.sandbox` is an
/// `Option<bool>` that defaults to `None`, and almost nobody sets it
/// explicitly — so a plugin writing `{"safety":{"sandbox":false}}` wins by
/// default and turns the bash sandbox off silently. That is no more power than
/// `./.plank/settings.json` already has over the same key, so the setting is
/// honored rather than blocked; it is only made visible.
fn settings_safety_warnings(name: &str, root: &Path) -> Vec<String> {
    let path = root.join("settings.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Some(crate::tools::mcp::Json::Obj(top)) = crate::tools::mcp::json_parse(&text) else {
        return Vec::new();
    };
    let Some((_, crate::tools::mcp::Json::Obj(safety))) = top.iter().find(|(k, _)| k == "safety")
    else {
        return Vec::new();
    };
    safety
        .iter()
        .map(|(key, _)| {
            format!(
                "plugin '{name}': settings.json sets safety.{key}; it applies unless you set it yourself in ~/.plank/settings.json or ./.plank/settings.json"
            )
        })
        .collect()
}

/// Loads one plugin directory. `None` when `dir` is not a plugin at all —
/// neither a manifest nor any recognizable component.
#[must_use]
pub fn load_plugin(dir: &Path, origin: Origin) -> Option<Plugin> {
    let mut plugin = load_plugin_fields(dir, origin)?;
    // Appended after the manifest is read, so the warning uses the plugin's
    // final name rather than the directory it happened to sit in.
    let safety = settings_safety_warnings(&plugin.name, &plugin.root);
    plugin.warnings.extend(safety);
    Some(plugin)
}

/// The manifest-reading half of [`load_plugin`], split out so the safety-key
/// audit runs once on every exit path.
fn load_plugin_fields(dir: &Path, origin: Origin) -> Option<Plugin> {
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
    ///
    /// Holds both the load-time complaints and, once the session has merged
    /// the contributions, the collision warnings [`reconcile`] and
    /// [`mcp_servers`] raise — those span two contributors by construction, so
    /// they belong to the set rather than to one plugin, and putting them here
    /// is what makes `/plugins` show them.
    pub warnings: Vec<String>,
}

impl PluginSet {
    /// Folds post-load contribution warnings into the set so [`render_list`]
    /// shows them.
    pub fn add_warnings(&mut self, warnings: impl IntoIterator<Item = String>) {
        self.warnings.extend(warnings);
    }

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
        // The manifest `name` is gated by `valid_name` in `load_plugin`, but
        // the directory-name fallback is not — and it becomes the namespace
        // prefix just the same. A directory called `foo:bar` would mint the
        // ambiguous alias `foo:bar:greet`, and one called `a__b` the MCP server
        // name `a__b-weather`, unroutable because `mcp__<server>__<tool>` is
        // split at the first `__` (and rejected by `config_load_reporting`
        // everywhere else). Drop rather than sanitize: a sanitized name could
        // itself collide with a real plugin's.
        if !valid_name(&plugin.name) {
            set.warnings.push(format!(
                "plugin directory {}: unusable plugin name {:?}; skipped (give it a plugin.json with a valid \"name\")",
                dir.display(),
                plugin.name
            ));
            continue;
        }
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
    /// The file or directory the entry was loaded from. [`reconcile`] clones
    /// an entry to register it twice, so the two copies share this — which is
    /// how [`listing`] tells a plugin's own bare copy apart from a local entry
    /// that merely has the same name.
    fn source(&self) -> &Path;
}

impl Named for crate::skills::Skill {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    fn source(&self) -> &Path {
        &self.dir
    }
}

impl Named for crate::agents::AgentDef {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    fn source(&self) -> &Path {
        &self.path
    }
}

impl Named for crate::templates::Template {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    fn source(&self) -> &Path {
        &self.path
    }
}

/// One entry as a listing should show it: once, under the name worth typing,
/// plus the plugin that contributed it.
#[derive(Debug)]
pub struct Listed<'a, T> {
    /// The entry to render.
    pub entry: &'a T,
    /// The name to show. The bare name when the plugin won it, the
    /// `<plugin>:<name>` alias when something else holds the bare one.
    pub name: &'a str,
    /// The contributing plugin, or `None` for a user or project entry.
    pub plugin: Option<&'a str>,
}

/// Collapses a [`reconcile`]d list for display: an uncontested plugin entry is
/// registered twice, bare and aliased, so every listing showed it twice and the
/// model could read the two rows as two different skills. Both names stay
/// resolvable; only the rendering is deduplicated.
///
/// A local entry whose own name happens to contain `:` is reported as though it
/// came from a plugin. Names come from file stems and directory names, so that
/// is pathological rather than reachable in practice, and the alternative would
/// be to carry provenance on every entry.
#[must_use]
pub fn listing<T: Named>(entries: &[T]) -> Vec<Listed<'_, T>> {
    let mut out: Vec<Listed<'_, T>> = Vec::new();
    let mut suppressed: Vec<usize> = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        if suppressed.contains(&i) {
            continue;
        }
        let Some((plugin, bare)) = entry.name().split_once(':') else {
            out.push(Listed {
                entry,
                name: entry.name(),
                plugin: None,
            });
            continue;
        };
        // `reconcile` pushes the alias first and the bare copy right after, so
        // the twin is always later in the list; matching on the source path is
        // what distinguishes it from a same-named local entry.
        let twin = entries
            .iter()
            .enumerate()
            .find(|(j, o)| *j != i && o.name() == bare && o.source() == entry.source());
        match twin {
            Some((j, o)) => {
                suppressed.push(j);
                out.push(Listed {
                    entry: o,
                    name: o.name(),
                    plugin: Some(plugin),
                });
            }
            None => out.push(Listed {
                entry,
                name: entry.name(),
                plugin: Some(plugin),
            }),
        }
    }
    out
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

/// Settings files contributed by plugins, in load order. They are applied
/// below `~/.plank/settings.json`, so a plugin can never override the user.
#[must_use]
pub fn settings_paths(set: &PluginSet) -> Vec<PathBuf> {
    set.plugins
        .iter()
        .filter_map(|p| component_root(p, "settings.json", "settings.json"))
        .collect()
}

/// Which components a plugin actually contributes, for the listing.
fn contributions(plugin: &Plugin) -> Vec<&'static str> {
    let mut out = Vec::new();
    for (label, (plank, cc)) in COMPONENT_LABELS.into_iter().zip(COMPONENT_PATHS) {
        if component_root(plugin, plank, cc).is_some() {
            out.push(label);
        }
    }
    out
}

/// Renders the `/plugins` listing: one block per plugin, then every warning.
#[must_use]
pub fn render_list(set: &PluginSet) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    if set.plugins.is_empty() {
        out.push_str(
            "no plugins loaded (checked ~/.plank/plugins/dev, ./.plank/plugins, and --plugin-dir)\n",
        );
    }
    for plugin in &set.plugins {
        let _ = write!(out, "{} ({})", plugin.name, plugin.origin.label());
        if !plugin.version.is_empty() {
            let _ = write!(out, " v{}", plugin.version);
        }
        out.push('\n');
        if !plugin.description.is_empty() {
            let _ = writeln!(out, "  {}", plugin.description);
        }
        let _ = writeln!(out, "  {}", plugin.root.display());
        let parts = contributions(plugin);
        if parts.is_empty() {
            out.push_str("  contributes nothing\n");
        } else {
            let _ = writeln!(out, "  contributes: {}", parts.join(", "));
        }
    }
    // WASM components live inside these same plugins, so they are listed here
    // rather than under a command of their own — one place to answer "what is
    // active in this session, and what can it reach".
    out.push_str(&crate::wasmreg::render_components(
        &crate::wasmreg::discover(set),
    ));
    let warnings = set.all_warnings();
    if !warnings.is_empty() {
        out.push_str("\nwarnings:\n");
        for w in warnings {
            let _ = writeln!(out, "  {w}");
        }
    }
    out
}

/// The directory `/plugins install` copies into, and one of the two the loader
/// auto-scans.
#[must_use]
pub fn user_plugin_dir(home: &Path) -> PathBuf {
    home.join(".plank").join("plugins").join("dev")
}

/// Installs the plugin directory at `src` for the current user.
///
/// A copy rather than a symlink: a plugin that changed under a running session
/// would be a plugin whose approved SHA-256 no longer describes what is
/// loaded, and the trust store's whole premise is that the hash *is* the
/// identity.
///
/// Refuses to overwrite. Replacing an installed plugin means new bytes under
/// an approved name, which is exactly the case trust exists to catch, so it is
/// a deliberate act (remove, then install) rather than a silent one.
///
/// # Errors
/// Returns a message when `src` is not a plugin, when the destination exists,
/// or when the copy fails.
pub fn install(src: &Path, home: &Path) -> Result<PathBuf, String> {
    let plugin = load_plugin(src, Origin::CliDir)
        .ok_or_else(|| format!("{} is not a plugin directory", src.display()))?;
    let dest = user_plugin_dir(home).join(&plugin.name);
    if dest.exists() {
        return Err(format!(
            "'{}' is already installed at {}; remove it first",
            plugin.name,
            dest.display()
        ));
    }
    copy_tree(src, &dest).map_err(|e| format!("cannot install into {}: {e}", dest.display()))?;
    Ok(dest)
}

/// Cap on a downloaded plugin archive.
///
/// A plugin is a manifest and a few WASM modules; the two plank ships are well
/// under a megabyte each. The cap exists so a wrong URL — or a hostile one —
/// cannot fill the user's disk before anything has been validated.
const MAX_ARCHIVE_BYTES: u64 = 64 * 1024 * 1024;

/// Downloads the `.tar.gz` at `url` and installs the plugin inside it.
///
/// Split from [`install`] rather than folded into it: everything here is about
/// getting bytes onto disk safely, and once there is a directory the local path
/// is the same code, with the same refusal to overwrite and the same trust
/// prompt on first load.
///
/// The archive is extracted into a fresh temporary directory and *validated*
/// before anything is copied into the user's plugin directory:
///
/// - TLS is required except to loopback, reusing the `--remote` policy.
/// - The response is capped at [`MAX_ARCHIVE_BYTES`].
/// - Extraction goes to a directory this function created and owns, so a
///   traversal entry cannot land beside an existing plugin.
/// - Any symlink in the extracted tree is a refusal. A plugin has no use for
///   one, and `copy_tree` follows links: a `wasm -> /Users/x/.ssh` entry would
///   otherwise copy a private key into the plugin directory and, worse, make it
///   readable by anything that reads plugins.
///
/// # Errors
/// Returns a message when the URL is rejected, the download fails, the archive
/// will not extract, it contains no plugin, or it contains a symlink.
pub fn install_from_url(url: &str, home: &Path) -> Result<PathBuf, String> {
    crate::remote::validate_remote_url(url, false)?;
    let staging = staging_dir(home)?;
    // The staging directory is removed on every exit path, including the error
    // ones: a failed install that leaves a half-extracted tree behind would be
    // found by the next `install` as a plugin nobody asked for.
    let result = fetch_and_stage(url, &staging).and_then(|root| install(&root, home));
    let _ = std::fs::remove_dir_all(&staging);
    result
}

/// A private, empty directory to extract into. Removed by the caller.
fn staging_dir(home: &Path) -> Result<PathBuf, String> {
    let base = user_plugin_dir(home).join(".staging");
    // Fresh every time: reusing a directory would mix an old failed download
    // with a new one, and the plugin root is found by searching the tree.
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).map_err(|e| format!("cannot create {}: {e}", base.display()))?;
    Ok(base)
}

/// Downloads, extracts and validates, returning the plugin root inside `dir`.
fn fetch_and_stage(url: &str, dir: &Path) -> Result<PathBuf, String> {
    let archive = dir.join("plugin.tar.gz");
    download_to(url, &archive)?;
    let status = std::process::Command::new("tar")
        // No `-P`: tar then refuses absolute paths and `..` components itself,
        // which is the first of the two lines of defence. The second is the
        // symlink scan below, which does not depend on tar's behaviour.
        .arg("-xzf")
        .arg(&archive)
        .arg("-C")
        .arg(dir)
        .status()
        .map_err(|e| format!("cannot run tar: {e}"))?;
    if !status.success() {
        return Err(format!("{url} did not extract as a .tar.gz"));
    }
    let _ = std::fs::remove_file(&archive);
    reject_symlinks(dir)?;
    find_plugin_root(dir)
        .ok_or_else(|| format!("{url} contains no plugin (no .plank-plugin/plugin.json)"))
}

/// Streams `url` into `path`, refusing anything over [`MAX_ARCHIVE_BYTES`].
fn download_to(url: &str, path: &Path) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .timeout_global(Some(std::time::Duration::from_mins(5)))
        .build()
        .into();
    let mut resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("cannot fetch {url}: {e}"))?;
    // `take` rather than a Content-Length check: the header is advisory and a
    // server can send more than it promised.
    let mut body = std::io::Read::take(resp.body_mut().as_reader(), MAX_ARCHIVE_BYTES + 1);
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut body, &mut bytes)
        .map_err(|e| format!("cannot read {url}: {e}"))?;
    if bytes.len() as u64 > MAX_ARCHIVE_BYTES {
        return Err(format!(
            "{url} is larger than the {} MiB plugin limit",
            MAX_ARCHIVE_BYTES / (1024 * 1024)
        ));
    }
    std::fs::write(path, &bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Fails when any entry under `dir` is a symlink.
fn reject_symlinks(dir: &Path) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for entry in entries.flatten() {
        // `symlink_metadata`, not `metadata`: the latter follows the link and
        // would report the target's type, which is the thing being checked.
        let meta = entry
            .path()
            .symlink_metadata()
            .map_err(|e| format!("cannot inspect {}: {e}", entry.path().display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!(
                "refusing archive: {} is a symlink, which a plugin never needs",
                entry.file_name().to_string_lossy()
            ));
        }
        if meta.is_dir() {
            reject_symlinks(&entry.path())?;
        }
    }
    Ok(())
}

/// Finds the directory holding a plugin manifest, at `dir` or one level in.
///
/// A release tarball unpacks to `<name>/...`, so the plugin is usually one
/// level down; a hand-rolled archive may put the manifest at the top.
fn find_plugin_root(dir: &Path) -> Option<PathBuf> {
    if manifest_path(dir).is_some() {
        return Some(dir.to_path_buf());
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && manifest_path(p).is_some())
        .collect();
    // Sorted so an archive with two plugins in it resolves the same way twice
    // rather than depending on directory order.
    candidates.sort();
    candidates.into_iter().next()
}

/// Uninstalls the plugin named `name` from the user's plugin directory.
///
/// Only ever deletes inside that directory, and only a single path segment:
/// the name comes from a command line, and a name containing a separator would
/// otherwise reach anywhere on disk from a `/plugins remove`.
///
/// # Errors
/// Returns a message when the name is not a bare name, or nothing is installed
/// under it, or the delete fails.
pub fn uninstall(name: &str, home: &Path) -> Result<PathBuf, String> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(format!("'{name}' is not a plugin name"));
    }
    let dir = user_plugin_dir(home).join(name);
    if !dir.is_dir() {
        return Err(format!("'{name}' is not installed"));
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("cannot remove {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Recursively copies `src` to `dest`.
///
/// Hand-rolled rather than shelling out to `cp`: a plugin install that depends
/// on a shell is one that behaves differently under a sandbox, and this runs
/// on the one path where the user is handing plank code to run.
fn copy_tree(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            // A build tree is not part of a plugin: `target/` under a guest
            // crate is gigabytes of object files that would be copied into the
            // user's home and never read.
            if entry.file_name() == "target" {
                continue;
            }
            copy_tree(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
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
        // Invariant: an alias always contains `:` and a bare name never does —
        // `valid_name` rejects `:` in both plugin names and entry names, so the
        // only `:` in `merged` is the one this function inserts. That is why
        // `taken_by_local` stays correct even though the aliased copy is pushed
        // into `merged` on the very next lines and later iterations rescan the
        // whole vector: an aliased entry can never compare equal to a bare
        // `name`, so previously-aliased plugin entries are invisible to this
        // test and only genuine non-plugin entries can contest a bare name.
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

/// Loads one component out of every plugin in the set, tagging each entry
/// with the plugin that supplied it.
fn gather<T>(
    set: &PluginSet,
    plank: &str,
    cc: &str,
    load: impl Fn(&[PathBuf]) -> Vec<T>,
) -> Vec<(String, T)> {
    let mut out = Vec::new();
    for plugin in &set.plugins {
        let Some(root) = component_root(plugin, plank, cc) else {
            continue;
        };
        for entry in load(&[root]) {
            out.push((plugin.name.clone(), entry));
        }
    }
    out
}

/// Skills from `<home>/.plank` and `<cwd>/.plank`, plus every plugin's,
/// namespaced per the collision rule, given an explicit home directory.
/// `home` is `None` when there is no home directory to overlay. Returns the
/// merged list and the collision warnings.
#[must_use]
pub fn skills_in(
    home: Option<&Path>,
    cwd: &Path,
    set: &PluginSet,
) -> (Vec<crate::skills::Skill>, Vec<String>) {
    let mut roots = Vec::new();
    if let Some(home) = home {
        roots.push(home.join(".plank").join("skills"));
    }
    roots.push(cwd.join(".plank").join("skills"));
    let local = crate::skills::load_from(&roots);
    let plugin = gather(set, "skills", "skills", crate::skills::load_from);
    reconcile(local, plugin)
}

/// Skills from `~/.plank` and `./.plank`, plus every plugin's, namespaced per
/// the collision rule. Returns the merged list and the collision warnings.
#[must_use]
pub fn skills_with_plugins(
    cwd: &Path,
    set: &PluginSet,
) -> (Vec<crate::skills::Skill>, Vec<String>) {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    skills_in(home.as_deref(), cwd, set)
}

/// Agent definitions from `<home>/.plank` and `<cwd>/.plank`, plus every
/// plugin's, given an explicit home directory. `home` is `None` when there is
/// no home directory to overlay.
#[must_use]
pub fn agents_in(
    home: Option<&Path>,
    cwd: &Path,
    set: &PluginSet,
) -> (Vec<crate::agents::AgentDef>, Vec<String>) {
    let mut roots = Vec::new();
    if let Some(home) = home {
        roots.push(home.join(".plank").join("agents"));
    }
    roots.push(cwd.join(".plank").join("agents"));
    let local = crate::agents::load_from(&roots);
    let plugin = gather(set, "agents", "agents", crate::agents::load_from);
    reconcile(local, plugin)
}

/// Agent definitions from `~/.plank` and `./.plank`, plus every plugin's.
#[must_use]
pub fn agents_with_plugins(
    cwd: &Path,
    set: &PluginSet,
) -> (Vec<crate::agents::AgentDef>, Vec<String>) {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    agents_in(home.as_deref(), cwd, set)
}

/// Prompt templates from `<home>/.plank` and `<cwd>/.plank`, plus every
/// plugin's, given an explicit home directory. `home` is `None` when there is
/// no home directory to overlay. A plugin may spell the directory
/// `templates/` (plank) or `commands/` (Claude Code); the local half is
/// always `templates/`.
#[must_use]
pub fn templates_in(
    home: Option<&Path>,
    cwd: &Path,
    set: &PluginSet,
) -> (Vec<crate::templates::Template>, Vec<String>) {
    let mut roots = Vec::new();
    if let Some(home) = home {
        roots.push(home.join(".plank").join("templates"));
    }
    roots.push(cwd.join(".plank").join("templates"));
    let local = crate::templates::load_from(&roots);
    let plugin = gather(set, "templates", "commands", crate::templates::load_from);
    reconcile(local, plugin)
}

/// Prompt templates from `~/.plank` and `./.plank`, plus every plugin's.
/// A plugin may spell the directory `templates/` (plank) or `commands/`
/// (Claude Code).
#[must_use]
pub fn templates_with_plugins(
    cwd: &Path,
    set: &PluginSet,
) -> (Vec<crate::templates::Template>, Vec<String>) {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    templates_in(home.as_deref(), cwd, set)
}

/// Hook definitions from `<home>/.plank/hooks.json`, then every plugin's hook
/// file in load order, then `<cwd>/.plank/hooks.json`, given an explicit home
/// directory. `home` is `None` when there is no home directory to overlay.
/// Hooks are additive, so all of them run and there is no name to
/// disambiguate.
#[must_use]
pub fn hooks_in(home: Option<&Path>, cwd: &Path, set: &PluginSet) -> crate::hooks::Hooks {
    let mut paths = Vec::new();
    if let Some(home) = home {
        paths.push(home.join(".plank").join("hooks.json"));
    }
    for plugin in &set.plugins {
        if let Some(path) = component_root(plugin, "hooks.json", "hooks/hooks.json") {
            paths.push(path);
        }
    }
    paths.push(cwd.join(".plank").join("hooks.json"));
    crate::hooks::load_from(&paths)
}

/// Hook definitions from `~/.plank/hooks.json`, then every plugin's hook file
/// in load order, then `<cwd>/.plank/hooks.json`. Hooks are additive, so all
/// of them run and there is no name to disambiguate.
#[must_use]
pub fn hooks_with_plugins(cwd: &Path, set: &PluginSet) -> crate::hooks::Hooks {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    hooks_in(home.as_deref(), cwd, set)
}

/// MCP servers contributed by plugins, with names disambiguated.
///
/// Uncontested names stay bare. When two plugins offer the same server name,
/// both are renamed `<plugin>-<server>`. The separator is `-` rather than `:`
/// because a server name is embedded in `mcp__<server>__<tool>` and split at
/// the first `__`, so a plugin server name containing `__` is rejected.
///
/// The rename pass can itself introduce a collision — e.g. plugins `alpha`
/// and `beta` both offering `weather` become `alpha-weather` and
/// `beta-weather`, but a third plugin offering a server literally named
/// `alpha-weather` then collides with the renamed one. That collision is not
/// undone (the caller's `merge_configs` keeps its existing last-write-wins
/// precedence), but it is reported as a warning naming the colliding servers
/// and the plugins that contributed them.
///
/// Note the deliberate asymmetry against [`reconcile`]. For skills, agents and
/// templates a non-plugin entry *takes the bare name away* from the plugin,
/// which is then reachable only as `<plugin>:<name>`. A plugin MCP server does
/// the opposite: it keeps the bare name here, and the local `.mcp.json` shadows
/// it afterwards, in the caller's `merge_configs`. The asymmetry is structural,
/// not an oversight — `reconcile` sees plugin and non-plugin entries together
/// and can arbitrate between them in one pass, whereas this function only ever
/// sees plugin servers. Local and global servers are merged in a later, separate
/// stage that already implements local-wins precedence by name, so pre-renaming
/// here would defeat that merge: a plugin server renamed out of the way could no
/// longer be overridden by the local file entry the user wrote to override it.
#[must_use]
pub fn mcp_servers(set: &PluginSet) -> (Vec<crate::tools::mcp::McpServerConfig>, Vec<String>) {
    let mut warnings = Vec::new();
    let mut gathered: Vec<(String, crate::tools::mcp::McpServerConfig)> = Vec::new();
    for plugin in &set.plugins {
        let Some(path) = component_root(plugin, ".mcp.json", ".mcp.json") else {
            continue;
        };
        let Some((configs, rejected)) = crate::tools::mcp::config_load_reporting(&path) else {
            warnings.push(format!(
                "plugin '{}': unreadable {}",
                plugin.name,
                path.display()
            ));
            continue;
        };
        for reason in rejected {
            warnings.push(format!("plugin '{}': {reason}", plugin.name));
        }
        for config in configs {
            gathered.push((plugin.name.clone(), config));
        }
    }

    let mut out: Vec<(String, crate::tools::mcp::McpServerConfig)> = Vec::new();
    for (plugin, config) in &gathered {
        let contested = gathered
            .iter()
            .filter(|(_, other)| other.name == config.name)
            .count()
            > 1;
        let mut config = config.clone();
        if contested {
            warnings.push(format!(
                "MCP server '{}' is contributed by more than one plugin; renamed '{}-{}'",
                config.name, plugin, config.name
            ));
            config.name = format!("{plugin}-{}", config.name);
        }
        out.push((plugin.clone(), config));
    }

    for (name, plugin_contributors) in duplicate_names_after_rename(&out) {
        warnings.push(format!(
            "MCP server name '{name}' collides across plugins {} after disambiguation; the last one wins",
            plugin_contributors.join(", ")
        ));
    }

    (out.into_iter().map(|(_, c)| c).collect(), warnings)
}

/// Groups `(plugin, config)` pairs by final server name, returning only the
/// names contributed more than once, alongside the contributing plugin names
/// in encounter order.
fn duplicate_names_after_rename(
    entries: &[(String, crate::tools::mcp::McpServerConfig)],
) -> Vec<(String, Vec<String>)> {
    let mut by_name: Vec<(String, Vec<String>)> = Vec::new();
    for (plugin, config) in entries {
        match by_name.iter_mut().find(|(name, _)| *name == config.name) {
            Some((_, plugins)) => plugins.push(plugin.clone()),
            None => by_name.push((config.name.clone(), vec![plugin.clone()])),
        }
    }
    by_name.retain(|(_, plugins)| plugins.len() > 1);
    by_name
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Installing copies the tree, skips build output, and refuses to
    /// overwrite — replacing an installed plugin is new bytes under an
    /// approved name, which is the case trust exists to catch.
    /// The URL route is served over loopback HTTP by a throwaway server, so the
    /// whole path — fetch, extract, find the plugin, install — is exercised
    /// without a network.
    #[test]
    fn install_from_url_fetches_extracts_and_installs() {
        use std::io::{Read as _, Write as _};

        let root = std::env::temp_dir().join(format!("plank-url-install-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Build the tarball a release would publish: the plugin sits one level
        // down, under its own directory name.
        let staged = root.join("stage").join("demo");
        std::fs::create_dir_all(staged.join(".plank-plugin")).unwrap();
        std::fs::create_dir_all(staged.join("wasm")).unwrap();
        std::fs::write(
            staged.join(".plank-plugin").join("plugin.json"),
            r#"{"name": "demo", "wasm": [{"id": "dev.plank.demo", "module": "demo.wasm"}]}"#,
        )
        .unwrap();
        std::fs::write(staged.join("wasm").join("demo.wasm"), b"\0asm").unwrap();
        let tarball = root.join("demo.tar.gz");
        let ok = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&tarball)
            .arg("-C")
            .arg(root.join("stage"))
            .arg("demo")
            .status()
            .unwrap()
            .success();
        assert!(ok, "could not build the fixture tarball");
        let bytes = std::fs::read(&tarball).unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/gzip\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                )
                .as_bytes(),
            );
            let _ = sock.write_all(&bytes);
            let _ = sock.flush();
        });

        let home = root.join("home");
        // http to loopback is allowed without --insecure, same rule as --remote.
        let dest = install_from_url(&format!("http://127.0.0.1:{port}/demo.tar.gz"), &home)
            .expect("installed from url");
        server.join().unwrap();

        assert!(dest.ends_with("demo"), "{}", dest.display());
        assert!(dest.join("wasm").join("demo.wasm").is_file());
        assert!(dest.starts_with(user_plugin_dir(&home)));
        // The staging directory must not survive: the next install searches the
        // tree for a plugin root, and a leftover would be found as one.
        assert!(
            !user_plugin_dir(&home).join(".staging").exists(),
            "staging directory left behind"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The same path against the artifact the release actually publishes.
    ///
    /// The fixture test above builds its own tarball, which proves the logic
    /// but not that `guests/package.sh` produces something installable. This
    /// one consumes the real file. It skips when packaging has not been run —
    /// the normal state of a checkout — and is required in the CI job that
    /// packages, via the same `PLANK_REQUIRE_GUESTS` switch the WASM suite uses.
    #[test]
    fn the_real_packaged_artifact_installs() {
        use std::io::{Read as _, Write as _};

        let tarball = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("guests")
            .join("dist")
            .join("plank-arcades.tar.gz");
        let Ok(bytes) = std::fs::read(&tarball) else {
            assert!(
                std::env::var_os("PLANK_REQUIRE_GUESTS").is_none_or(|v| v == "0"),
                "PLANK_REQUIRE_GUESTS is set but {} is missing: run guests/package.sh",
                tarball.display()
            );
            eprintln!("skipping: run guests/package.sh first");
            return;
        };

        let root = std::env::temp_dir().join(format!("plank-real-pkg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let served = bytes.clone();
        let feeder = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    served.len()
                )
                .as_bytes(),
            );
            let _ = sock.write_all(&served);
            let _ = sock.flush();
        });

        let dest = install_from_url(
            &format!("http://127.0.0.1:{port}/plank-arcades.tar.gz"),
            &root,
        )
        .expect("the released artifact must install");
        feeder.join().unwrap();

        // The name comes from the manifest, and the module from its `wasm`
        // entry: if either drifted from what package.sh writes, the component
        // would load and then fail to find its own code.
        assert!(dest.ends_with("arcades"), "{}", dest.display());
        assert!(
            dest.join("wasm").join("arcades.wasm").is_file(),
            "module missing under {}",
            dest.display()
        );
        assert!(dest.join(".plank-plugin").join("plugin.json").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Plaintext to a non-loopback host is refused before any bytes move, and
    /// nothing is created on disk by the refusal.
    #[test]
    fn install_from_url_requires_tls_off_loopback() {
        let root = std::env::temp_dir().join(format!("plank-url-tls-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let err = install_from_url("http://plugins.example.com/x.tar.gz", &root)
            .expect_err("plaintext must be refused");
        assert!(err.contains("plaintext"), "{err}");
        assert!(
            !user_plugin_dir(&root).join(".staging").exists(),
            "a refused URL still created a staging directory"
        );
    }

    /// A symlink in the archive is refused. `copy_tree` follows links, so an
    /// entry pointing at `~/.ssh` would otherwise be copied into the plugin
    /// directory and made readable by anything that reads plugins.
    #[test]
    fn a_symlink_in_the_extracted_tree_is_refused() {
        let root = std::env::temp_dir().join(format!("plank-url-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let tree = root.join("tree");
        std::fs::create_dir_all(tree.join(".plank-plugin")).unwrap();
        std::fs::write(
            tree.join(".plank-plugin").join("plugin.json"),
            r#"{"name": "demo"}"#,
        )
        .unwrap();
        let secret = root.join("secret");
        std::fs::write(&secret, b"private key").unwrap();
        std::os::unix::fs::symlink(&secret, tree.join("stolen")).unwrap();

        let err = reject_symlinks(&tree).expect_err("a symlink must be refused");
        assert!(err.contains("symlink"), "{err}");
        assert!(
            err.contains("stolen"),
            "the offending entry must be named: {err}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A tarball whose manifest sits at the top level installs too: only the
    /// release layout puts it one level down.
    #[test]
    fn a_plugin_root_is_found_at_the_top_or_one_level_in() {
        let root = std::env::temp_dir().join(format!("plank-url-root-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let flat = root.join("flat");
        std::fs::create_dir_all(flat.join(".plank-plugin")).unwrap();
        std::fs::write(
            flat.join(".plank-plugin").join("plugin.json"),
            r#"{"name": "flat"}"#,
        )
        .unwrap();
        assert_eq!(find_plugin_root(&flat).as_deref(), Some(flat.as_path()));

        let nested_parent = root.join("nested");
        let inner = nested_parent.join("demo");
        std::fs::create_dir_all(inner.join(".plank-plugin")).unwrap();
        std::fs::write(
            inner.join(".plank-plugin").join("plugin.json"),
            r#"{"name": "demo"}"#,
        )
        .unwrap();
        assert_eq!(
            find_plugin_root(&nested_parent).as_deref(),
            Some(inner.as_path())
        );

        // Neither: an archive of something that is not a plugin.
        let empty = root.join("empty");
        std::fs::create_dir_all(empty.join("docs")).unwrap();
        assert!(find_plugin_root(&empty).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn install_copies_a_plugin_and_refuses_to_overwrite() {
        let root = std::env::temp_dir().join(format!("plank-install-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let src = root.join("src-plugin");
        std::fs::create_dir_all(src.join(".plank-plugin")).unwrap();
        std::fs::create_dir_all(src.join("wasm")).unwrap();
        std::fs::create_dir_all(src.join("target").join("junk")).unwrap();
        std::fs::write(
            src.join(".plank-plugin").join("plugin.json"),
            r#"{"name": "demo"}"#,
        )
        .unwrap();
        std::fs::write(src.join("wasm").join("demo.wasm"), b"\0asm").unwrap();
        std::fs::write(src.join("target").join("junk").join("big.o"), b"xxxx").unwrap();

        let home = root.join("home");
        let dest = install(&src, &home).expect("installed");
        assert!(dest.ends_with("demo"), "{}", dest.display());
        assert!(dest.join("wasm").join("demo.wasm").is_file());
        assert!(
            !dest.join("target").exists(),
            "a build tree was copied into the user's home"
        );
        // And it lands where the loader looks.
        assert!(dest.starts_with(user_plugin_dir(&home)));

        let err = install(&src, &home).unwrap_err();
        assert!(err.contains("already installed"), "{err}");
        // ...and the message's advice works: remove, then install again.
        uninstall("demo", &home).expect("removed");
        assert!(!dest.exists());
        install(&src, &home).expect("reinstalled after removal");

        // A name is a single path segment. Anything else could reach out of
        // the plugin directory from a `/plugins remove`.
        for bad in ["../evil", "a/b", ""] {
            assert!(
                uninstall(bad, &home).is_err(),
                "'{bad}' was accepted as a plugin name"
            );
        }
        assert!(
            uninstall("absent", &home)
                .unwrap_err()
                .contains("not installed")
        );

        // Not a plugin at all.
        let bare = root.join("not-a-plugin");
        std::fs::create_dir_all(&bare).unwrap();
        assert!(install(&bare, &home).unwrap_err().contains("not a plugin"));
        let _ = std::fs::remove_dir_all(&root);
    }
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
        path: PathBuf,
    }

    impl Named for Item {
        fn name(&self) -> &str {
            &self.name
        }
        fn set_name(&mut self, name: String) {
            self.name = name;
        }
        fn source(&self) -> &Path {
            &self.path
        }
    }

    fn item(name: &str, from: &'static str) -> Item {
        Item {
            name: name.to_string(),
            from,
            path: PathBuf::from(from).join(name),
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
    fn an_uncontested_plugin_entry_is_listed_once_under_its_bare_name() {
        let (merged, _) = reconcile(vec![], vec![("demo".to_string(), item("greet", "plugin"))]);
        assert_eq!(names(&merged), vec!["demo:greet", "greet"]);
        let listed = listing(&merged);
        assert_eq!(listed.len(), 1, "rendered once");
        assert_eq!(listed[0].name, "greet");
        assert_eq!(listed[0].plugin, Some("demo"));
    }

    #[test]
    fn a_contested_plugin_entry_is_listed_under_its_alias_beside_the_local_one() {
        let (merged, _) = reconcile(
            vec![item("greet", "user")],
            vec![("demo".to_string(), item("greet", "plugin"))],
        );
        let listed = listing(&merged);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].name, "greet");
        assert_eq!(listed[0].plugin, None, "the local entry has no plugin");
        assert_eq!(listed[1].name, "demo:greet");
        assert_eq!(listed[1].plugin, Some("demo"));
    }

    #[test]
    fn the_skill_listings_show_a_plugin_skill_once_with_its_origin() {
        let base = scratch("listing-skills");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            "skills/greet/SKILL.md",
            "---\nname: greet\ndescription: says hi\n---\nHello\n",
        );
        let set = load_in(Some(&home), &cwd, &[plugin]);
        let (skills, _) = skills_in(Some(&home), &cwd, &set);
        assert_eq!(skills.len(), 2, "both names stay resolvable");
        for out in [
            crate::skills::render_list(&skills),
            crate::skills::render_names(&skills),
        ] {
            assert_eq!(
                out.matches("greet").count(),
                1,
                "the skill is named once: {out}"
            );
            assert!(out.contains("[plugin demo]"), "origin missing: {out}");
        }
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

    #[test]
    fn a_plugin_skill_is_reachable_bare_and_namespaced() {
        let base = scratch("plugin-skills");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            "skills/greet/SKILL.md",
            "---\nname: greet\ndescription: says hi\n---\nHello\n",
        );
        let set = load_in(Some(&home), &cwd, &[plugin]);
        let (skills, warnings) = skills_in(Some(&home), &cwd, &set);
        assert!(skills.iter().any(|s| s.name == "greet"));
        assert!(skills.iter().any(|s| s.name == "demo:greet"));
        assert!(warnings.is_empty());
    }

    #[test]
    fn a_project_skill_beats_a_plugin_skill_of_the_same_name() {
        let base = scratch("skill-collision");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        write(
            &cwd,
            ".plank/skills/greet/SKILL.md",
            "---\nname: greet\ndescription: project version\n---\nProject\n",
        );
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            "skills/greet/SKILL.md",
            "---\nname: greet\ndescription: plugin version\n---\nPlugin\n",
        );
        let set = load_in(Some(&home), &cwd, &[plugin]);
        let (skills, warnings) = skills_in(Some(&home), &cwd, &set);
        let unqualified = skills
            .iter()
            .find(|s| s.name == "greet")
            .expect("bare exists");
        assert_eq!(unqualified.description, "project version");
        assert!(skills.iter().any(|s| s.name == "demo:greet"));
        assert!(warnings.iter().any(|w| w.contains("demo:greet")));
    }

    #[test]
    fn a_plugin_template_under_the_claude_code_commands_dir_loads() {
        let base = scratch("plugin-commands");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".claude-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            "commands/note.md",
            "---\ndescription: a note\n---\nBody\n",
        );
        let set = load_in(Some(&home), &cwd, &[plugin]);
        let (templates, _) = templates_in(Some(&home), &cwd, &set);
        assert!(templates.iter().any(|t| t.name == "note"));
        assert!(templates.iter().any(|t| t.name == "demo:note"));
    }

    #[test]
    fn a_plugin_agent_is_reachable_namespaced() {
        let base = scratch("plugin-agents");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            "agents/scout.md",
            "---\ndescription: scouts\n---\nBe a scout.\n",
        );
        let set = load_in(Some(&home), &cwd, &[plugin]);
        let (agents, _) = agents_in(Some(&home), &cwd, &set);
        assert!(agents.iter().any(|a| a.name == "demo:scout"));
    }

    #[test]
    fn plugin_hooks_are_added_between_the_user_and_project_files() {
        let base = scratch("plugin-hooks");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        write(
            &cwd,
            ".plank/hooks.json",
            r#"{"hooks":{"PreToolUse":[{"hooks":[{"type":"command","command":"project-hook"}]}]}}"#,
        );
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            "hooks/hooks.json",
            r#"{"hooks":{"PreToolUse":[{"hooks":[{"type":"command","command":"plugin-hook"}]}]}}"#,
        );
        let set = load_in(Some(&home), &cwd, &[plugin]);
        let hooks = hooks_in(Some(&home), &cwd, &set);
        let commands: Vec<&str> = hooks
            .pre_tool_use
            .iter()
            .flat_map(|m| m.hooks.iter())
            .map(|h| h.command.as_str())
            .collect();
        assert!(commands.contains(&"plugin-hook"));
        assert!(commands.contains(&"project-hook"));
        let plugin_at = commands
            .iter()
            .position(|c| *c == "plugin-hook")
            .expect("present");
        let project_at = commands
            .iter()
            .position(|c| *c == "project-hook")
            .expect("present");
        assert!(
            plugin_at < project_at,
            "plugin hooks run before project hooks"
        );
    }

    #[test]
    fn a_plugin_mcp_server_keeps_its_bare_name_when_uncontested() {
        let base = scratch("plugin-mcp");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            ".mcp.json",
            r#"{"mcpServers":{"weather":{"command":"weather-server"}}}"#,
        );
        let set = load_in(Some(&home), &cwd, &[plugin]);
        let (servers, warnings) = mcp_servers(&set);
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "weather");
        assert!(warnings.is_empty());
    }

    #[test]
    fn two_plugins_offering_the_same_server_name_are_both_prefixed() {
        let base = scratch("mcp-collision");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        for name in ["alpha", "beta"] {
            let plugin = base.join(name);
            write(
                &plugin,
                ".plank-plugin/plugin.json",
                &format!(r#"{{"name":"{name}"}}"#),
            );
            write(
                &plugin,
                ".mcp.json",
                r#"{"mcpServers":{"weather":{"command":"weather-server"}}}"#,
            );
        }
        let set = load_in(Some(&home), &cwd, &[base.join("alpha"), base.join("beta")]);
        let (servers, warnings) = mcp_servers(&set);
        let names: Vec<&str> = servers.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha-weather", "beta-weather"]);
        assert!(warnings.iter().any(|w| w.contains("weather")));
    }

    #[test]
    fn a_rename_collision_with_a_third_plugins_bare_name_is_warned_about() {
        let base = scratch("mcp-rename-collision");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        for name in ["alpha", "beta"] {
            let plugin = base.join(name);
            write(
                &plugin,
                ".plank-plugin/plugin.json",
                &format!(r#"{{"name":"{name}"}}"#),
            );
            write(
                &plugin,
                ".mcp.json",
                r#"{"mcpServers":{"weather":{"command":"weather-server"}}}"#,
            );
        }
        let third = base.join("gamma");
        write(&third, ".plank-plugin/plugin.json", r#"{"name":"gamma"}"#);
        write(
            &third,
            ".mcp.json",
            r#"{"mcpServers":{"alpha-weather":{"command":"x"}}}"#,
        );
        let set = load_in(
            Some(&home),
            &cwd,
            &[base.join("alpha"), base.join("beta"), third],
        );
        let (servers, warnings) = mcp_servers(&set);
        let names: Vec<&str> = servers.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["alpha-weather", "beta-weather", "alpha-weather"]
        );
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("alpha-weather") && w.contains("gamma") && w.contains("alpha")),
            "expected a warning naming the colliding servers and plugins, got: {warnings:?}"
        );
    }

    #[test]
    fn a_plugin_settings_file_is_discovered() {
        let base = scratch("plugin-settings");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(&plugin, "settings.json", r#"{"kvcache":{"maxBytes":7}}"#);
        let set = load_in(Some(&base.join("home")), &cwd, &[plugin]);
        let paths = settings_paths(&set);
        assert_eq!(paths.len(), 1);
        assert!(paths[0].ends_with("settings.json"));
    }

    #[test]
    fn a_plugin_server_name_containing_a_double_underscore_is_rejected() {
        let base = scratch("mcp-bad-name");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            ".mcp.json",
            r#"{"mcpServers":{"we__ather":{"command":"x"}}}"#,
        );
        let set = load_in(Some(&home), &cwd, &[plugin]);
        let (servers, warnings) = mcp_servers(&set);
        assert!(servers.is_empty());
        assert!(warnings.iter().any(|w| w.contains("we__ather")));
    }

    #[test]
    fn the_listing_names_each_plugin_its_origin_and_its_warnings() {
        let base = scratch("render-list");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(
            &plugin,
            ".plank-plugin/plugin.json",
            r#"{"name":"demo","description":"A demo","version":"1.0"}"#,
        );
        write(&plugin, "skills/greet/SKILL.md", "hi\n");
        let set = load_in(Some(&base.join("home")), &cwd, &[plugin]);
        let out = render_list(&set);
        assert!(out.contains("demo"));
        assert!(out.contains("A demo"));
        assert!(out.contains("1.0"));
        assert!(out.contains("--plugin-dir"));
        assert!(out.contains("skills"));
    }

    /// The directory-name fallback becomes the namespace prefix, so it has to
    /// clear the same bar the manifest `name` does. `foo:bar` would mint the
    /// ambiguous alias `foo:bar:greet`.
    #[test]
    fn a_directory_named_with_a_colon_is_dropped() {
        let base = scratch("dirname-colon");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("foo:bar");
        write(&plugin, "skills/greet/SKILL.md", "hi\n");
        let set = load_in(Some(&home), &cwd, &[plugin]);
        assert!(set.plugins.is_empty(), "the plugin was dropped");
        assert!(
            set.warnings.iter().any(|w| w.contains("foo:bar")),
            "the directory is named in a warning: {:?}",
            set.warnings
        );
    }

    /// `a__b` would mint the MCP server name `a__b-weather`, which
    /// `mcp__<server>__<tool>` cannot round-trip.
    #[test]
    fn a_directory_named_with_a_double_underscore_is_dropped() {
        let base = scratch("dirname-dunder");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("a__b");
        write(&plugin, ".mcp.json", r#"{"mcpServers":{"weather":{}}}"#);
        let set = load_in(Some(&home), &cwd, &[plugin]);
        assert!(set.plugins.is_empty(), "the plugin was dropped");
        assert!(
            set.warnings.iter().any(|w| w.contains("a__b")),
            "the directory is named in a warning: {:?}",
            set.warnings
        );
        assert!(mcp_servers(&set).0.is_empty());
    }

    /// A manifest with a valid `name` rescues an otherwise unusable directory
    /// name, since the fallback never applies.
    #[test]
    fn a_manifest_name_rescues_an_unusable_directory_name() {
        let base = scratch("dirname-rescued");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("foo:bar");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        let set = load_in(Some(&home), &cwd, &[plugin]);
        assert_eq!(set.plugins.len(), 1);
        assert_eq!(set.plugins[0].name, "demo");
    }

    /// The collision warnings used to be computed and dropped on the floor —
    /// `let (skills, _) = …` — so a plugin skill that silently lost its bare
    /// name, or a plugin MCP server rejected for containing `__`, showed up
    /// nowhere. Folding them onto the set is what puts them in `/plugins`.
    #[test]
    fn contribution_collision_warnings_reach_the_listing() {
        let base = scratch("warnings-surface");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        write(
            &cwd,
            ".plank/skills/greet/SKILL.md",
            "---\nname: greet\ndescription: project version\n---\nProject\n",
        );
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(
            &plugin,
            "skills/greet/SKILL.md",
            "---\nname: greet\ndescription: plugin version\n---\nPlugin\n",
        );
        write(
            &plugin,
            ".mcp.json",
            r#"{"mcpServers":{"we__ather":{"command":"true"}}}"#,
        );
        let mut set = load_in(Some(&home), &cwd, &[plugin]);
        assert!(
            !render_list(&set).contains("demo:greet"),
            "precondition: the collision is not in the load-time warnings"
        );

        let (_, skill_warnings) = skills_in(Some(&home), &cwd, &set);
        let (_, mcp_warnings) = mcp_servers(&set);
        assert!(!skill_warnings.is_empty(), "the collision was reported");
        assert!(!mcp_warnings.is_empty(), "the `__` server was rejected");
        set.add_warnings(skill_warnings);
        set.add_warnings(mcp_warnings);

        let out = render_list(&set);
        assert!(
            out.contains("demo:greet"),
            "shadowed skill warning missing: {out}"
        );
        assert!(
            out.contains("we__ather"),
            "rejected MCP server warning missing: {out}"
        );
    }

    /// `safety.sandbox` defaults to `None`, so a plugin setting it wins unless
    /// the user set it explicitly — which almost nobody does. The setting is
    /// still honored; it just cannot be silent.
    #[test]
    fn a_plugin_settings_file_touching_a_safety_key_warns() {
        let base = scratch("plugin-safety");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(&plugin, "settings.json", r#"{"safety":{"sandbox":false}}"#);
        let set = load_in(Some(&home), &cwd, &[plugin]);
        assert_eq!(set.plugins.len(), 1, "the plugin still loads");
        let out = render_list(&set);
        assert!(
            out.contains("safety.sandbox") && out.contains("demo"),
            "expected a warning naming the plugin and the key, got: {out}"
        );
    }

    #[test]
    fn a_plugin_settings_file_without_safety_keys_is_quiet() {
        let base = scratch("plugin-safety-quiet");
        let home = base.join("home");
        std::fs::create_dir_all(&home).expect("mkdir");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let plugin = base.join("demo");
        write(&plugin, ".plank-plugin/plugin.json", r#"{"name":"demo"}"#);
        write(&plugin, "settings.json", r#"{"kvcache":{"maxBytes":7}}"#);
        let set = load_in(Some(&home), &cwd, &[plugin]);
        assert!(
            !set.all_warnings().iter().any(|w| w.contains("safety")),
            "{:?}",
            set.all_warnings()
        );
    }

    #[test]
    fn the_listing_says_so_when_no_plugins_are_loaded() {
        let base = scratch("render-empty");
        let cwd = base.join("proj");
        std::fs::create_dir_all(&cwd).expect("mkdir");
        let set = load_in(Some(&base.join("home")), &cwd, &[]);
        let out = render_list(&set);
        assert!(out.contains("no plugins"));
        assert!(out.contains("--plugin-dir"));
    }
}
