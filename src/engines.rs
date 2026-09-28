//! The engine catalog: `engines.json`, layered compiled-in → fetched cache →
//! `~/.plank/engines.local.json`, and resolution of the user's model choice.
//!
//! An *engine* is a named main model plus optional `mtp` and `vision`
//! companions. The published layers may only describe downloadable files
//! (`url`/`bytes`/`sha256`); only the local layer may point a role at a
//! `path`, because a remote catalog must never choose where plank writes.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::manifest::FileEntry;

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
}
