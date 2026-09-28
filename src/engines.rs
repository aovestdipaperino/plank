//! The engine catalog: `engines.json`, layered compiled-in → fetched cache →
//! `~/.plank/engines.local.json`, and resolution of the user's model choice.
//!
//! An *engine* is a named main model plus optional `mtp` and `vision`
//! companions. The published layers may only describe downloadable files
//! (`url`/`bytes`/`sha256`); only the local layer may point a role at a
//! `path`, because a remote catalog must never choose where plank writes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::manifest::{EngineId, FileEntry};

/// The catalog shipped in this build, and the offline first-launch fallback.
pub const COMPILED_IN: &str = include_str!("../engines.json");

/// The default engine when no layer names a valid one: derived from the
/// compiled-in catalog's own `default` field.
fn compiled_in_default() -> &'static str {
    static CACHED: OnceLock<String> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let raw: serde_json::Value = serde_json::from_str(COMPILED_IN)
                .expect("compiled-in engines catalog must be valid JSON");
            raw.get("default")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .expect("compiled-in engines catalog must declare a default")
        })
        .as_str()
}

/// Roles this build knows how to load. Others are carried in `raw` only.
pub const ROLES: [&str; 3] = ["main", "mtp", "vision"];

/// Whether `s` is a valid engine name: `[a-z0-9-]+`.
#[must_use]
pub fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Which layer a catalog text comes from; decides whether `path` is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Compiled-in or fetched: download entries only.
    Published,
    /// `~/.plank/engines.local.json`: may use `path`, may omit versions.
    Local,
}

/// One engine after parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineEntry {
    /// The engine's name, the key it was listed under.
    pub name: String,
    /// Monotonic per-engine version; 0 for a local path-only engine.
    pub version: u32,
    /// Downloadable roles.
    pub files: BTreeMap<String, FileEntry>,
    /// Local-only roles pointing at an existing file.
    pub paths: BTreeMap<String, PathBuf>,
    /// The engine's JSON object, verbatim, for the installed record.
    pub raw: String,
}

impl EngineEntry {
    /// The entry as a standalone manifest for the downloader, or `None` when
    /// it has nothing to download (a local path-only engine).
    ///
    /// The manifest's `files` are the engine's role entries, so its raw text
    /// is rebuilt as `{"version":…,"released":…,"notes":…,"files":{…}}` from
    /// the verbatim engine object, keeping any unknown top-level field.
    #[must_use]
    pub fn to_manifest(&self) -> Option<crate::manifest::Manifest> {
        if self.files.is_empty() || self.version == 0 {
            return None;
        }
        let mut obj: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&self.raw).ok()?;
        let mut files = serde_json::Map::new();
        for role in ROLES {
            if let Some(v) = obj.remove(role) {
                files.insert(role.to_string(), v);
            }
        }
        obj.insert("files".to_string(), serde_json::Value::Object(files));
        let raw = serde_json::to_string_pretty(&serde_json::Value::Object(obj)).ok()?;
        crate::manifest::parse(&raw).ok()
    }
}

/// A parsed, possibly layered catalog.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Catalog {
    /// Monotonic catalog version; 0 for a local layer.
    pub version: u32,
    /// The `default` key as written, validated lazily by [`Catalog::default_name`].
    pub default: Option<String>,
    /// Engines by name.
    pub engines: BTreeMap<String, EngineEntry>,
}

impl Catalog {
    /// The engine called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&EngineEntry> {
        self.engines.get(name)
    }

    /// The default engine's name: the recorded one when it exists, otherwise
    /// the compiled-in default.
    #[must_use]
    pub fn default_name(&self) -> &str {
        match &self.default {
            Some(d) if self.engines.contains_key(d) => d,
            _ => compiled_in_default(),
        }
    }

    /// Known engine names, sorted, comma-separated, for error messages.
    #[must_use]
    pub fn names(&self) -> String {
        self.engines.keys().cloned().collect::<Vec<_>>().join(", ")
    }
}

/// Parses catalog `text` from `layer`, pushing a warning for each entry it drops.
///
/// # Errors
/// Only when the text is not a JSON object or a published layer lacks a
/// nonzero top-level `version`; a bad *entry* is a warning, not an error.
pub fn parse(text: &str, layer: Layer, warn: &mut Vec<String>) -> Result<Catalog, String> {
    let root: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("bad engines catalog: {e}"))?;
    let obj = root
        .as_object()
        .ok_or("bad engines catalog: not a JSON object")?;
    let version = obj
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0);
    if layer == Layer::Published && version == 0 {
        return Err("bad engines catalog: missing or zero version".to_string());
    }
    let default = obj
        .get("default")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let mut engines = BTreeMap::new();
    if let Some(list) = obj.get("engines").and_then(serde_json::Value::as_object) {
        for (name, value) in list {
            match parse_entry(name, value, layer) {
                Ok(e) => {
                    engines.insert(name.clone(), e);
                }
                Err(e) => warn.push(format!("engines: skipping `{name}`: {e}")),
            }
        }
    }
    Ok(Catalog {
        version,
        default,
        engines,
    })
}

fn parse_entry(name: &str, value: &serde_json::Value, layer: Layer) -> Result<EngineEntry, String> {
    if !valid_name(name) {
        return Err("names may use only a-z, 0-9 and -".to_string());
    }
    let obj = value.as_object().ok_or("not a JSON object")?;
    let version = obj
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0);
    let mut files = BTreeMap::new();
    let mut paths = BTreeMap::new();
    for role in ROLES {
        let Some(r) = obj.get(role) else { continue };
        if let Some(p) = r.get("path").and_then(serde_json::Value::as_str) {
            if layer == Layer::Published {
                return Err(format!("{role}: a published entry may not name a path"));
            }
            paths.insert(role.to_string(), crate::settings::expand_tilde(p));
            continue;
        }
        let entry: FileEntry =
            serde_json::from_value(r.clone()).map_err(|e| format!("{role}: {e}"))?;
        files.insert(role.to_string(), entry);
    }
    if !files.contains_key("main") && !paths.contains_key("main") {
        return Err("no main model".to_string());
    }
    if !files.is_empty() && version == 0 {
        return Err("downloadable entries need a nonzero version".to_string());
    }
    let e = EngineEntry {
        name: name.to_string(),
        version,
        files,
        paths,
        raw: serde_json::to_string(value).map_err(|e| e.to_string())?,
    };
    // Reuse the manifest validator (sha256 shape, https, nonzero bytes).
    if !e.files.is_empty() && e.to_manifest().is_none() {
        return Err("an artifact entry is malformed (sha256, url or bytes)".to_string());
    }
    Ok(e)
}

/// `over` stacked on `base`: whole-entry replacement per name; `over`'s
/// default wins when set; the higher version is kept.
#[must_use]
pub fn layer(mut base: Catalog, over: Catalog) -> Catalog {
    base.engines.extend(over.engines);
    if over.default.is_some() {
        base.default = over.default;
    }
    base.version = base.version.max(over.version);
    base
}

/// The cached fetched catalog under `root`.
fn cache_path_in(root: &Path) -> PathBuf {
    root.join("engines.remote.json")
}

/// The user's local layer under `root`.
fn local_path_in(root: &Path) -> PathBuf {
    root.join("engines.local.json")
}

/// The layered catalog under `root`: compiled-in, then the cache when it is
/// strictly newer, then the local file. Never fails; problems become warnings.
///
/// # Panics
/// Never in practice: the compiled-in catalog is validated by a unit test.
#[must_use]
pub fn load_in(root: &Path, warn: &mut Vec<String>) -> Catalog {
    let mut cat = parse(COMPILED_IN, Layer::Published, warn)
        .expect("the compiled-in catalog is validated by a unit test");
    if let Ok(text) = std::fs::read_to_string(cache_path_in(root)) {
        match parse(&text, Layer::Published, &mut Vec::new()) {
            Ok(c) if c.version > cat.version => cat = layer(cat, c),
            Ok(_) => {}
            Err(e) => warn.push(format!("engines: ignoring the cached catalog: {e}")),
        }
    }
    if let Ok(text) = std::fs::read_to_string(local_path_in(root)) {
        match parse(&text, Layer::Local, warn) {
            Ok(c) => cat = layer(cat, c),
            Err(e) => warn.push(format!(
                "engines: ignoring {}: {e}",
                local_path_in(root).display()
            )),
        }
    }
    if let Some(d) = cat.default.as_deref()
        && !cat.engines.contains_key(d)
    {
        warn.push(format!(
            "engines: the default `{d}` is not a known engine; using `{}`",
            cat.default_name()
        ));
    }
    cat
}

/// [`load_in`] rooted at `~/.plank`.
#[must_use]
pub fn load(warn: &mut Vec<String>) -> Catalog {
    load_in(&crate::manifest::plank_dir(), warn)
}

/// Records a freshly fetched catalog as the cache when it is newer than both
/// the compiled-in catalog and the current cache. Returns it when it parses.
#[must_use]
pub fn update_cache_in(root: &Path, fetched: &str) -> Option<Catalog> {
    let new = parse(fetched, Layer::Published, &mut Vec::new()).ok()?;
    let compiled = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).map_or(0, |c| c.version);
    let cached = std::fs::read_to_string(cache_path_in(root))
        .ok()
        .and_then(|t| parse(&t, Layer::Published, &mut Vec::new()).ok())
        .map_or(0, |c| c.version);
    if new.version > compiled.max(cached) {
        // Written beside and renamed over, so a crash or a second plank never
        // leaves a truncated cache that would silently read as absent.
        let _ = std::fs::create_dir_all(root);
        let path = cache_path_in(root);
        let tmp = path.with_extension(format!("json.tmp-{}", std::process::id()));
        if std::fs::write(&tmp, fetched)
            .and_then(|()| std::fs::rename(&tmp, &path))
            .is_err()
        {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Some(new)
}

/// What the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice<'a> {
    /// Nothing: the catalog default.
    Default,
    /// `--model X` / `engine.model`: an engine name, else a path.
    Spec(&'a str),
    /// `--model:X`: must be an engine name.
    Named(&'a str),
}

/// The model files a run will load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// The engine, or `None` for a bare path that matches no managed engine.
    pub id: Option<EngineId>,
    /// Main model file.
    pub main: PathBuf,
    /// The engine's mtp companion, if it declares one.
    pub mtp: Option<PathBuf>,
    /// The engine's vision encoder, if it declares one.
    pub vision: Option<PathBuf>,
    /// Whether `main` is a catalog download plank may fetch and upgrade.
    pub managed_main: bool,
}

fn select(root: &Path, entry: &EngineEntry, id: EngineId) -> Selection {
    let role = |r: &str| -> Option<PathBuf> {
        entry.paths.get(r).cloned().or_else(|| {
            entry
                .files
                .contains_key(r)
                .then(|| crate::manifest::local_path_for_in(root, id, r))
                .flatten()
        })
    };
    Selection {
        id: Some(id),
        main: role("main").expect("parse guarantees a main role"),
        mtp: role("mtp"),
        vision: role("vision"),
        managed_main: entry.files.contains_key("main"),
    }
}

/// Pre-catalog artifact names under `~/.plank`, and the engine role each one
/// became. `enginemigrate` renames them; [`resolve_with_note_in`] still
/// accepts the old main paths so a config written before the move keeps
/// working.
pub const LEGACY_ARTIFACTS: [(&str, EngineId, &str); 5] = [
    ("ds4flash.gguf", EngineId::DS4VISION, "main"),
    ("ds4flash.vision.gguf", EngineId::DS4VISION, "vision"),
    ("ds4flash.dspark.gguf", EngineId::DS4VISION, "mtp"),
    ("ds41flash.gguf", EngineId::DS41, "main"),
    ("ds41flash.vision.gguf", EngineId::DS41, "vision"),
];

/// The engine and role a pre-catalog artifact `path` under `root` now
/// belongs to, plus its old file name. The file itself is usually gone by
/// now (the migration moved it), so the directory is compared, not the file.
#[must_use]
pub fn legacy_artifact(root: &Path, path: &Path) -> Option<(EngineId, &'static str, &'static str)> {
    let leaf = path.file_name()?.to_str()?;
    let (old, id, role) = LEGACY_ARTIFACTS
        .into_iter()
        .find(|(old, _, _)| *old == leaf)?;
    let parent = path.parent()?;
    let same_dir = parent == root
        || matches!(
            (parent.canonicalize(), root.canonicalize()),
            (Ok(a), Ok(b)) if a == b
        );
    same_dir.then_some((id, role, old))
}

/// The one line a legacy artifact path gets when it resolves to its engine.
fn legacy_note(old: &str, id: EngineId) -> String {
    format!("~/.plank/{old} is now the {id} engine; use --model {id}")
}

/// Resolves `choice` against `catalog`, with managed files under `root`.
///
/// # Errors
/// As [`resolve_with_note_in`].
pub fn resolve_in(root: &Path, catalog: &Catalog, choice: Choice<'_>) -> Result<Selection, String> {
    resolve_with_note_in(root, catalog, choice).map(|(sel, _)| sel)
}

/// Resolves `choice` against `catalog`, with managed files under `root`, and
/// returns a deprecation note to print once when the choice named a
/// pre-catalog path (`~/.plank/ds4flash.gguf`, `~/.plank/ds41flash.gguf`)
/// that now belongs to an engine.
///
/// # Errors
/// An unknown name (for [`Choice::Named`]), or a bare word that is neither an
/// engine nor an existing file (for [`Choice::Spec`]).
pub fn resolve_with_note_in(
    root: &Path,
    catalog: &Catalog,
    choice: Choice<'_>,
) -> Result<(Selection, Option<String>), String> {
    let by_name = |name: &str| {
        let entry = catalog.get(name)?;
        let id = EngineId::new(name)?;
        Some(select(root, entry, id))
    };
    match choice {
        Choice::Default => by_name(catalog.default_name())
            .map(|sel| (sel, None))
            .ok_or_else(|| {
                format!(
                    "the default engine `{}` is not in the catalog",
                    catalog.default_name()
                )
            }),
        Choice::Named(n) => by_name(n)
            .map(|sel| (sel, None))
            .ok_or_else(|| format!("unknown engine `{n}`; known engines: {}", catalog.names())),
        Choice::Spec(s) => {
            if let Some(sel) = by_name(s) {
                return Ok((sel, None));
            }
            let path = crate::settings::expand_tilde(s);
            // A config written before the engine catalog names the old main
            // file, which the migration has since renamed.
            if let Some((id, "main", old)) = legacy_artifact(root, &path)
                && let Some(sel) = by_name(id.as_str())
            {
                return Ok((sel, Some(legacy_note(old, id))));
            }
            let looks_like_path = s.contains('/') || path.extension().is_some() || path.exists();
            if !looks_like_path {
                return Err(format!(
                    "no engine or file named `{s}`; known engines: {}",
                    catalog.names()
                ));
            }
            // A path that is exactly a managed engine's main selects the engine.
            for (name, entry) in &catalog.engines {
                if let Some(id) = EngineId::new(name) {
                    let sel = select(root, entry, id);
                    if sel.managed_main && sel.main == path {
                        return Ok((sel, None));
                    }
                }
            }
            Ok((
                Selection {
                    id: None,
                    main: path,
                    mtp: None,
                    vision: None,
                    managed_main: false,
                },
                None,
            ))
        }
    }
}

static ACTIVE: OnceLock<Selection> = OnceLock::new();

/// Records this process's selection once, at startup.
pub fn set_active(sel: Selection) {
    let _ = ACTIVE.set(sel);
}

/// This process's selection, when startup recorded one.
#[must_use]
pub fn active() -> Option<&'static Selection> {
    ACTIVE.get()
}

/// This process's engine, when it runs one from the catalog.
#[must_use]
pub fn active_id() -> Option<EngineId> {
    active().and_then(|s| s.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(c: char) -> String {
        std::iter::repeat_n(c, 64).collect()
    }

    fn entry_json(name: &str) -> String {
        format!(
            r#"{{"name":"{name}","url":"https://h/{name}","bytes":10,"sha256":"{}"}}"#,
            sha('a')
        )
    }

    fn one_engine(cat_version: u32, default: &str, engine: &str, ver: u32) -> String {
        format!(
            r#"{{"version":{cat_version},"default":"{default}","engines":{{"{engine}":{{"version":{ver},"main":{}}}}}}}"#,
            entry_json("m.gguf")
        )
    }

    #[test]
    fn the_compiled_in_catalog_parses_and_defaults_to_ds4vision() {
        let mut w = Vec::new();
        let c = parse(COMPILED_IN, Layer::Published, &mut w).expect("compiled-in parses");
        assert!(w.is_empty(), "warnings: {w:?}");
        assert_eq!(c.default_name(), "ds4vision");
        for n in ["ds4vision", "ds41", "qwen"] {
            assert!(c.get(n).is_some(), "{n} missing");
        }
        assert!(c.get("ds4vision").unwrap().files.contains_key("mtp"));
        assert!(!c.get("qwen").unwrap().files.contains_key("mtp"));
    }

    #[test]
    fn the_fallback_default_is_the_compiled_in_catalogs_own() {
        let raw: serde_json::Value = serde_json::from_str(COMPILED_IN).unwrap();
        assert_eq!(compiled_in_default(), raw["default"].as_str().unwrap());
    }

    #[test]
    fn names_are_lowercase_alnum_and_dash_only() {
        assert!(valid_name("ds4vision"));
        assert!(valid_name("my-engine-2"));
        for bad in ["", "DS4", "a b", "a/b", "a.b", "a_b", "ä"] {
            assert!(!valid_name(bad), "{bad:?} accepted");
        }
    }

    #[test]
    fn an_invalid_engine_name_drops_that_entry_with_a_warning() {
        let text = format!(
            r#"{{"version":1,"default":"ok","engines":{{"ok":{{"version":1,"main":{e}}},"Bad Name":{{"version":1,"main":{e}}}}}}}"#,
            e = entry_json("m.gguf")
        );
        let mut w = Vec::new();
        let c = parse(&text, Layer::Published, &mut w).unwrap();
        assert!(c.get("ok").is_some());
        assert!(c.get("Bad Name").is_none());
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn an_engine_without_main_is_dropped() {
        let text = format!(
            r#"{{"version":1,"default":"x","engines":{{"x":{{"version":1,"vision":{}}}}}}}"#,
            entry_json("v.gguf")
        );
        let mut w = Vec::new();
        let c = parse(&text, Layer::Published, &mut w).unwrap();
        assert!(c.get("x").is_none());
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn a_published_entry_may_not_name_a_path() {
        let text = r#"{"version":1,"default":"x","engines":{"x":{"version":1,"main":{"path":"/evil.gguf"}}}}"#;
        let mut w = Vec::new();
        let c = parse(text, Layer::Published, &mut w).unwrap();
        assert!(c.get("x").is_none());
        assert!(w[0].contains("path"), "{w:?}");
    }

    #[test]
    fn a_local_entry_may_name_a_path_and_is_then_unmanaged() {
        let text = r#"{"engines":{"mine":{"main":{"path":"/models/mine.gguf"}}}}"#;
        let mut w = Vec::new();
        let c = parse(text, Layer::Local, &mut w).unwrap();
        let e = c.get("mine").unwrap();
        assert_eq!(
            e.paths.get("main"),
            Some(&PathBuf::from("/models/mine.gguf"))
        );
        assert!(e.files.is_empty());
        assert!(
            e.to_manifest().is_none(),
            "path-only engines are never downloaded"
        );
    }

    #[test]
    fn unknown_roles_are_ignored_but_kept_in_raw() {
        let text = format!(
            r#"{{"version":1,"default":"x","engines":{{"x":{{"version":1,"main":{m},"future":{m}}}}}}}"#,
            m = entry_json("m.gguf")
        );
        let mut w = Vec::new();
        let c = parse(&text, Layer::Published, &mut w).unwrap();
        let e = c.get("x").unwrap();
        assert!(!e.files.contains_key("future"));
        assert!(e.raw.contains("future"));
    }

    #[test]
    fn to_manifest_carries_version_and_role_keyed_files() {
        let mut w = Vec::new();
        let c = parse(COMPILED_IN, Layer::Published, &mut w).unwrap();
        let m = c.get("ds4vision").unwrap().to_manifest().unwrap();
        assert_eq!(m.version, 1);
        assert!(m.files.contains_key("main") && m.files.contains_key("mtp"));
        assert!(!m.files.contains_key("dspark"));
        assert!(
            crate::manifest::parse(&m.raw).is_ok(),
            "raw must round-trip through manifest::parse"
        );
    }

    #[test]
    fn a_later_layer_replaces_whole_entries_and_the_default() {
        let mut w = Vec::new();
        let base = parse(&one_engine(1, "a", "a", 1), Layer::Published, &mut w).unwrap();
        let over = parse(
            r#"{"default":"b","engines":{"a":{"main":{"path":"/x.gguf"}},"b":{"main":{"path":"/b.gguf"}}}}"#,
            Layer::Local,
            &mut w,
        )
        .unwrap();
        let c = layer(base, over);
        assert_eq!(c.default_name(), "b");
        let a = c.get("a").unwrap();
        assert!(a.files.is_empty(), "replaced wholesale, not merged");
        assert_eq!(a.paths.get("main"), Some(&PathBuf::from("/x.gguf")));
    }

    #[test]
    fn a_default_naming_a_missing_engine_falls_back_to_the_compiled_in_one() {
        let mut w = Vec::new();
        let c = parse(&one_engine(1, "ghost", "a", 1), Layer::Published, &mut w).unwrap();
        assert_eq!(c.default_name(), "ds4vision");
    }

    #[test]
    fn names_lists_sorted_comma_separated() {
        let mut w = Vec::new();
        let c = parse(COMPILED_IN, Layer::Published, &mut w).unwrap();
        assert_eq!(c.names(), "ds41, ds4vision, qwen");
    }

    fn root(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("plank-engines-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn a_local_default_naming_a_missing_engine_warns_and_falls_back() {
        let r = root("ghost-default");
        std::fs::write(r.join("engines.local.json"), r#"{"default":"ghost"}"#).unwrap();
        let mut w = Vec::new();
        let c = load_in(&r, &mut w);
        assert_eq!(c.default_name(), "ds4vision");
        assert_eq!(
            w,
            vec![
                "engines: the default `ghost` is not a known engine; using `ds4vision`".to_string()
            ]
        );
        let s = resolve_in(&r, &c, Choice::Default).unwrap();
        assert_eq!(s.id, Some(crate::manifest::EngineId::DS4VISION));
    }

    #[test]
    fn a_valid_default_does_not_warn() {
        let r = root("good-default");
        std::fs::write(r.join("engines.local.json"), r#"{"default":"qwen"}"#).unwrap();
        let mut w = Vec::new();
        let _ = load_in(&r, &mut w);
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn the_cache_write_leaves_no_temporary_behind() {
        let r = root("cache-atomic");
        let fetched = one_engine(99, "a", "a", 1);
        assert!(update_cache_in(&r, &fetched).is_some());
        assert_eq!(
            std::fs::read_to_string(r.join("engines.remote.json")).unwrap(),
            fetched
        );
        let leftovers: Vec<_> = std::fs::read_dir(&r)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn load_layers_cache_then_local() {
        let r = root("load");
        std::fs::write(r.join("engines.remote.json"), one_engine(99, "a", "a", 1)).unwrap();
        std::fs::write(r.join("engines.local.json"), r#"{"default":"qwen"}"#).unwrap();
        let mut w = Vec::new();
        let c = load_in(&r, &mut w);
        assert!(c.get("a").is_some(), "fetched cache layered");
        assert!(c.get("ds4vision").is_some(), "compiled-in kept");
        assert_eq!(c.default_name(), "qwen");
    }

    #[test]
    fn a_cache_not_newer_than_the_compiled_in_catalog_is_ignored() {
        let r = root("stale");
        let compiled = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        // Same version as the compiled-in catalog: not strictly newer.
        std::fs::write(
            r.join("engines.remote.json"),
            one_engine(compiled.version, "a", "a", 1),
        )
        .unwrap();
        let c = load_in(&r, &mut Vec::new());
        assert!(
            c.get("a").is_none(),
            "an equal-version cache must not be layered"
        );
    }

    #[test]
    fn a_broken_local_file_warns_and_is_ignored() {
        let r = root("badlocal");
        std::fs::write(r.join("engines.local.json"), "{nope").unwrap();
        let mut w = Vec::new();
        let c = load_in(&r, &mut w);
        assert_eq!(c.default_name(), "ds4vision");
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn the_cache_is_replaced_only_by_a_higher_version() {
        let r = root("cache");
        assert!(update_cache_in(&r, &one_engine(1000, "a", "a", 1)).is_some());
        assert!(r.join("engines.remote.json").exists());
        let before = std::fs::read_to_string(r.join("engines.remote.json")).unwrap();
        assert!(
            update_cache_in(&r, &one_engine(999, "b", "b", 1)).is_some(),
            "still parses"
        );
        assert_eq!(
            std::fs::read_to_string(r.join("engines.remote.json")).unwrap(),
            before
        );
        assert!(update_cache_in(&r, "{garbage").is_none());
    }

    #[test]
    fn default_choice_resolves_to_the_default_engines_derived_paths() {
        let r = root("sel-default");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let s = resolve_in(&r, &c, Choice::Default).unwrap();
        assert_eq!(s.id, Some(crate::manifest::EngineId::DS4VISION));
        assert_eq!(s.main, r.join("ds4vision.gguf"));
        assert_eq!(s.mtp, Some(r.join("ds4vision.mtp.gguf")));
        assert_eq!(s.vision, Some(r.join("ds4vision.vision.gguf")));
        assert!(s.managed_main);
    }

    #[test]
    fn a_spec_naming_an_engine_selects_it() {
        let r = root("sel-name");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let s = resolve_in(&r, &c, Choice::Spec("qwen")).unwrap();
        assert_eq!(s.id, Some(crate::manifest::EngineId::QWEN));
        assert_eq!(s.mtp, None);
    }

    #[test]
    fn a_spec_that_is_a_path_gets_no_companions() {
        let r = root("sel-path");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let s = resolve_in(&r, &c, Choice::Spec("/models/x.gguf")).unwrap();
        assert_eq!(s.id, None);
        assert_eq!(s.main, PathBuf::from("/models/x.gguf"));
        assert!(s.mtp.is_none() && s.vision.is_none() && !s.managed_main);
    }

    #[test]
    fn a_path_equal_to_a_managed_main_selects_that_engine() {
        let r = root("sel-managed");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let p = r.join("qwen.gguf");
        let s = resolve_in(&r, &c, Choice::Spec(p.to_str().unwrap())).unwrap();
        assert_eq!(s.id, Some(crate::manifest::EngineId::QWEN));
        assert!(s.managed_main);
    }

    #[test]
    fn an_unknown_bare_word_that_is_not_a_file_errors_with_the_known_list() {
        let r = root("sel-unknown");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let e = resolve_in(&r, &c, Choice::Spec("nope")).unwrap_err();
        assert_eq!(
            e,
            "no engine or file named `nope`; known engines: ds41, ds4vision, qwen"
        );
    }

    #[test]
    fn a_named_choice_must_be_an_engine() {
        let r = root("sel-named");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        assert!(resolve_in(&r, &c, Choice::Named("ds41")).is_ok());
        assert_eq!(
            resolve_in(&r, &c, Choice::Named("x.gguf")).unwrap_err(),
            "unknown engine `x.gguf`; known engines: ds41, ds4vision, qwen"
        );
    }

    #[test]
    fn the_old_ds4_main_path_selects_ds4vision_with_a_note() {
        let r = root("sel-legacy-ds4");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let p = r.join("ds4flash.gguf");
        let (s, note) = resolve_with_note_in(&r, &c, Choice::Spec(p.to_str().unwrap())).unwrap();
        assert_eq!(s.id, Some(crate::manifest::EngineId::DS4VISION));
        assert_eq!(s.main, r.join("ds4vision.gguf"));
        assert_eq!(s.mtp, Some(r.join("ds4vision.mtp.gguf")));
        assert!(s.managed_main);
        assert_eq!(
            note.as_deref(),
            Some("~/.plank/ds4flash.gguf is now the ds4vision engine; use --model ds4vision")
        );
    }

    #[test]
    fn the_old_ds41_main_path_selects_ds41_with_a_note() {
        let r = root("sel-legacy-ds41");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let p = r.join("ds41flash.gguf");
        let (s, note) = resolve_with_note_in(&r, &c, Choice::Spec(p.to_str().unwrap())).unwrap();
        assert_eq!(s.id, Some(crate::manifest::EngineId::DS41));
        assert_eq!(s.main, r.join("ds41.gguf"));
        assert_eq!(
            note.as_deref(),
            Some("~/.plank/ds41flash.gguf is now the ds41 engine; use --model ds41")
        );
    }

    #[test]
    fn an_old_file_name_outside_the_plank_dir_is_just_a_path() {
        let r = root("sel-legacy-elsewhere");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let (s, note) =
            resolve_with_note_in(&r, &c, Choice::Spec("/models/ds4flash.gguf")).unwrap();
        assert_eq!(s.id, None);
        assert!(note.is_none());
    }

    #[test]
    fn plain_choices_carry_no_note() {
        let r = root("sel-no-note");
        let c = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        for choice in [Choice::Default, Choice::Spec("qwen"), Choice::Named("ds41")] {
            assert!(resolve_with_note_in(&r, &c, choice).unwrap().1.is_none());
        }
    }

    #[test]
    fn a_local_path_role_overrides_the_derived_path() {
        let r = root("sel-localpath");
        let base = parse(COMPILED_IN, Layer::Published, &mut Vec::new()).unwrap();
        let over = parse(r#"{"engines":{"mine":{"main":{"path":"/m/mine.gguf"},"vision":{"path":"/m/v.gguf"}}}}"#, Layer::Local, &mut Vec::new()).unwrap();
        let c = layer(base, over);
        let s = resolve_in(&r, &c, Choice::Spec("mine")).unwrap();
        assert_eq!(s.main, PathBuf::from("/m/mine.gguf"));
        assert_eq!(s.vision, Some(PathBuf::from("/m/v.gguf")));
        assert!(!s.managed_main, "path roles are never downloaded");
    }
}
