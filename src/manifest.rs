//! The model manifest: what the current `DeepSeek` V4 Flash artifact set is.
//!
//! plank used to answer "is there a newer model?" by scanning the Hugging Face
//! tree API for the newest commit within a quant *family*, where the family was
//! inferred by stripping a `-MMDD` tag off a filename. That made a filename
//! convention into a wire contract, it only ever tracked the main model — the
//! vision encoder and the `DSpark` drafter were pinned to compiled-in constants
//! and never upgraded at all — and it offered no integrity check beyond
//! comparing `Content-Length`.
//!
//! The manifest replaces all of it with one monotonic integer. `version` is the
//! entire comparison; ordering by commit dates and reasoning about year
//! boundaries stop being anyone's problem.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The artifact kinds plank knows how to install.
///
/// A manifest may name others (see [`parse`]); those are carried through and
/// ignored, so a future release can add a fourth artifact without breaking
/// every client that predates it.
pub const KINDS: [&str; 3] = ["main", "vision", "dspark"];

/// One artifact in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct FileEntry {
    /// Filename as published, for display only. The install location is
    /// decided locally by [`local_path_for`], never by the manifest: a
    /// manifest must not be able to name a path plank then writes to.
    pub name: String,
    /// Absolute URL the bytes come from.
    pub url: String,
    /// Expected length. Used to size the progress bar and, at swap time, as a
    /// cheap presence check on files already verified by hash.
    pub bytes: u64,
    /// Lowercase hex SHA-256 of the complete file.
    pub sha256: String,
}

/// A parsed `ds4.manifest`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Monotonic release counter. The whole upgrade comparison.
    pub version: u32,
    /// Release date, for display.
    pub released: String,
    /// One-line human summary, for display.
    pub notes: String,
    /// Artifacts, keyed by kind.
    pub files: BTreeMap<String, FileEntry>,
    /// The exact bytes this was parsed from.
    ///
    /// The installed manifest is written by copying this, never by
    /// re-serializing: a round trip through the struct above would silently
    /// drop any field a future manifest gains, and the installed copy would
    /// then disagree with what was actually fetched.
    pub raw: String,
}

/// Shape serde sees. Kept private so `Manifest::raw` cannot be forged.
#[derive(serde::Deserialize)]
struct Wire {
    version: u32,
    #[serde(default)]
    released: String,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    files: BTreeMap<String, FileEntry>,
}

/// Parses manifest `text`.
///
/// Unknown keys under `files` are kept rather than rejected. Version 0 is
/// refused because it is the sentinel for "nothing installed" everywhere else
/// in this module, so a manifest that claimed it would read as absent.
///
/// # Errors
/// Returns a message when the text is not JSON, is missing `version`, or
/// declares version 0.
pub fn parse(text: &str) -> Result<Manifest, String> {
    let wire: Wire = serde_json::from_str(text).map_err(|e| format!("bad manifest: {e}"))?;
    if wire.version == 0 {
        return Err("bad manifest: version 0 is reserved".to_string());
    }
    Ok(Manifest {
        version: wire.version,
        released: wire.released,
        notes: wire.notes,
        files: wire.files,
        raw: text.to_string(),
    })
}

/// `~/.plank`, or `./.plank` when `HOME` is unset.
///
/// Mirrors `download::default_model_path`'s fallback so the two never disagree
/// about where the model lives.
#[must_use]
pub fn plank_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".plank")
}

/// The installed manifest: what the files currently in `~/.plank` are.
///
/// Written only by a successful swap, and written *last*, so its presence is
/// proof the whole set landed.
#[must_use]
pub fn installed_path() -> PathBuf {
    plank_dir().join("ds4.manifest")
}

/// Where in-flight and verified-but-not-yet-installed artifacts live.
#[must_use]
pub fn staging_dir() -> PathBuf {
    plank_dir().join("staging")
}

/// Helper-process bookkeeping: lock, job, state, cancel flag, log.
#[must_use]
pub fn downloads_dir() -> PathBuf {
    plank_dir().join("downloads")
}

/// Where an artifact of `kind` is installed.
///
/// Deliberately local knowledge rather than a manifest field: a manifest that
/// could name its own destination path would be a manifest that could write
/// anywhere on disk.
#[must_use]
pub fn local_path_for(kind: &str) -> Option<PathBuf> {
    match kind {
        "main" => Some(crate::download::default_model_path()),
        "vision" => Some(crate::download::default_vision_path()),
        "dspark" => Some(crate::download::default_dspark_path()),
        _ => None,
    }
}

/// Reads and parses the manifest at `path`, if it is there and valid.
///
/// A corrupt installed manifest reads as absent rather than fatal: the worst
/// case is one offered upgrade, which is recoverable, where a hard error at
/// startup is not.
#[must_use]
pub fn read_at(path: &Path) -> Option<Manifest> {
    parse(&std::fs::read_to_string(path).ok()?).ok()
}

/// What startup should do about the remote manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Nothing to do: the installed set is current, or newer.
    UpToDate,
    /// The files on disk already match this manifest, but nothing recorded
    /// that. Write it as installed and say nothing.
    Adopt(Manifest),
    /// Offer to download this manifest's set. `from` is the installed version,
    /// or 0 when there was none.
    Offer { manifest: Manifest, from: u32 },
}

/// Decides what to do about `remote`, given what is installed and what is on
/// disk.
///
/// `size_of` reports the byte length of the locally installed artifact of a
/// given kind, or `None` when it is absent. It is injected rather than read
/// from the filesystem because a wrong answer here costs the user an 87 GB
/// download, which is worth testing exhaustively without a disk.
///
/// Only kinds in both [`KINDS`] and the manifest are considered: a manifest
/// entry this build does not know how to install cannot block adoption, and a
/// kind the manifest omits cannot be demanded on disk.
#[must_use]
pub fn decide(
    remote: Manifest,
    installed: Option<&Manifest>,
    size_of: &dyn Fn(&str) -> Option<u64>,
) -> Decision {
    if let Some(installed) = installed {
        return if remote.version > installed.version {
            let from = installed.version;
            Decision::Offer {
                manifest: remote,
                from,
            }
        } else {
            // Equal, or a rolled-back remote. Never downgrade.
            Decision::UpToDate
        };
    }

    // Adopt-on-first-sight: no installed manifest, but if every artifact this
    // build installs is present at the manifest's own size, the set on disk is
    // almost certainly already this release — recorded by a plank that predates
    // manifests. Silently adopt rather than offering a re-download of bytes the
    // user already has.
    let intersection: Vec<_> = KINDS
        .iter()
        .filter_map(|kind| remote.files.get(*kind).map(|e| (*kind, e)))
        .collect();
    // `all()` is vacuously true on an empty iterator, so an empty manifest
    // would otherwise adopt itself into the installed record while having
    // downloaded nothing, suppressing any genuine release at that version.
    let matches = !intersection.is_empty()
        && intersection
            .iter()
            .all(|(kind, entry)| size_of(kind) == Some(entry.bytes));
    if matches {
        Decision::Adopt(remote)
    } else {
        Decision::Offer {
            manifest: remote,
            from: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A complete, well-formed manifest, used by most tests here.
    fn sample() -> &'static str {
        r#"{
          "version": 3,
          "released": "2026-09-04",
          "notes": "Vision-Exp refresh",
          "files": {
            "main":   { "name": "m.gguf", "url": "https://example.invalid/m", "bytes": 100, "sha256": "aa" },
            "vision": { "name": "v.gguf", "url": "https://example.invalid/v", "bytes": 200, "sha256": "bb" },
            "dspark": { "name": "d.gguf", "url": "https://example.invalid/d", "bytes": 300, "sha256": "cc" }
          }
        }"#
    }

    #[test]
    fn parses_every_field() {
        let m = parse(sample()).expect("sample parses");
        assert_eq!(m.version, 3);
        assert_eq!(m.released, "2026-09-04");
        assert_eq!(m.notes, "Vision-Exp refresh");
        assert_eq!(m.files.len(), 3);
        let main = m.files.get("main").expect("main entry");
        assert_eq!(main.name, "m.gguf");
        assert_eq!(main.url, "https://example.invalid/m");
        assert_eq!(main.bytes, 100);
        assert_eq!(main.sha256, "aa");
    }

    #[test]
    fn keeps_the_raw_bytes_verbatim() {
        // The installed manifest is written by copying `raw`, never by
        // re-serializing: a round trip through serde would silently drop any
        // field this build does not know about.
        let m = parse(sample()).expect("sample parses");
        assert_eq!(m.raw, sample());
    }

    #[test]
    fn an_unknown_file_kind_is_ignored_not_an_error() {
        // The manifest must be able to grow a fourth artifact before the
        // client reading it knows what that artifact is.
        let text = sample().replace(r#""dspark":"#, r#""futureproof": { "name": "f.gguf", "url": "u", "bytes": 1, "sha256": "dd" }, "dspark":"#);
        let m = parse(&text).expect("unknown kind parses");
        assert_eq!(m.files.len(), 4);
        assert!(m.files.contains_key("futureproof"));
    }

    #[test]
    fn a_missing_dspark_entry_parses() {
        // Not every release has to ship all three. Absence is a fact for
        // `decide` (Task 2) to act on, not a parse failure.
        let text = r#"{"version":1,"released":"x","notes":"","files":{
            "main": { "name": "m.gguf", "url": "u", "bytes": 1, "sha256": "aa" }
        }}"#;
        let m = parse(text).expect("partial manifest parses");
        assert!(m.files.contains_key("main"));
        assert!(!m.files.contains_key("dspark"));
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(parse("not json at all").is_err());
        assert!(parse("").is_err());
        assert!(parse(r#"{"version":"three"}"#).is_err());
    }

    #[test]
    fn version_zero_is_rejected() {
        // Zero is the sentinel for "nothing installed" in `decide`, so a
        // manifest may not claim it.
        let text = sample().replace(r#""version": 3"#, r#""version": 0"#);
        assert!(parse(&text).is_err());
    }

    #[test]
    fn kinds_are_the_three_artifacts() {
        assert_eq!(KINDS, ["main", "vision", "dspark"]);
    }

    #[test]
    fn paths_nest_under_the_plank_directory() {
        let root = plank_dir();
        assert_eq!(installed_path(), root.join("ds4.manifest"));
        assert_eq!(staging_dir(), root.join("staging"));
        assert_eq!(downloads_dir(), root.join("downloads"));
    }

    /// `sample()` at a given version, so tests can build a "newer" manifest.
    fn sample_at(version: u32) -> String {
        sample().replace(r#""version": 3"#, &format!(r#""version": {version}"#))
    }

    /// A size lookup that reports every artifact present at its manifest size.
    fn all_present(kind: &str) -> Option<u64> {
        match kind {
            "main" => Some(100),
            "vision" => Some(200),
            "dspark" => Some(300),
            _ => None,
        }
    }

    #[test]
    fn same_version_is_up_to_date() {
        let remote = parse(sample()).expect("parses");
        let installed = parse(sample()).expect("parses");
        assert!(matches!(
            decide(remote, Some(&installed), &all_present),
            Decision::UpToDate
        ));
    }

    #[test]
    fn a_newer_remote_is_offered() {
        let remote = parse(&sample_at(4)).expect("parses");
        let installed = parse(sample()).expect("parses");
        match decide(remote, Some(&installed), &all_present) {
            Decision::Offer { manifest, from } => {
                assert_eq!(manifest.version, 4);
                assert_eq!(from, 3);
            }
            other => panic!("expected an offer, got {other:?}"),
        }
    }

    #[test]
    fn an_older_remote_is_ignored() {
        // A rolled-back manifest must never trigger a downgrade download.
        let remote = parse(&sample_at(2)).expect("parses");
        let installed = parse(sample()).expect("parses");
        assert!(matches!(
            decide(remote, Some(&installed), &all_present),
            Decision::UpToDate
        ));
    }

    #[test]
    fn no_installed_manifest_but_matching_sizes_adopts_silently() {
        // Adopt-on-first-sight. Without this rule, every existing user is offered
        // an 87 GB re-download the day this ships.
        let remote = parse(sample()).expect("parses");
        match decide(remote, None, &all_present) {
            Decision::Adopt(m) => assert_eq!(m.version, 3),
            other => panic!("expected adoption, got {other:?}"),
        }
    }

    #[test]
    fn no_installed_manifest_and_a_wrong_size_offers() {
        let remote = parse(sample()).expect("parses");
        let sizes = |kind: &str| {
            if kind == "vision" {
                Some(999)
            } else {
                all_present(kind)
            }
        };
        match decide(remote, None, &sizes) {
            Decision::Offer { from, .. } => assert_eq!(from, 0),
            other => panic!("expected an offer, got {other:?}"),
        }
    }

    #[test]
    fn no_installed_manifest_and_a_missing_file_offers() {
        let remote = parse(sample()).expect("parses");
        let sizes = |kind: &str| {
            if kind == "dspark" {
                None
            } else {
                all_present(kind)
            }
        };
        match decide(remote, None, &sizes) {
            Decision::Offer { from, .. } => assert_eq!(from, 0),
            other => panic!("expected an offer, got {other:?}"),
        }
    }

    #[test]
    fn adoption_only_considers_kinds_the_manifest_actually_lists() {
        // A manifest with no dspark entry must not demand a dspark file on disk.
        let text = r#"{"version":1,"released":"x","notes":"","files":{
            "main": { "name": "m.gguf", "url": "u", "bytes": 100, "sha256": "aa" }
        }}"#;
        let remote = parse(text).expect("parses");
        let sizes = |kind: &str| (kind == "main").then_some(100);
        assert!(matches!(decide(remote, None, &sizes), Decision::Adopt(_)));
    }

    #[test]
    fn an_unknown_kind_is_not_required_on_disk() {
        // `futureproof` is in the manifest but this build cannot install it, so it
        // must not block adoption or the size check.
        let text = sample().replace(
            r#""dspark":"#,
            r#""futureproof": { "name": "f.gguf", "url": "u", "bytes": 7, "sha256": "dd" }, "dspark":"#,
        );
        let remote = parse(&text).expect("parses");
        assert!(matches!(
            decide(remote, None, &all_present),
            Decision::Adopt(_)
        ));
    }

    #[test]
    fn a_manifest_naming_no_known_artifact_is_never_adopted() {
        // `all()` is vacuously true on an empty iterator, so without an explicit
        // emptiness check this manifest would be recorded as installed while
        // nothing had been downloaded — and would then suppress a genuine
        // release at the same version as "up to date".
        let text = r#"{"version":5,"released":"x","notes":"","files":{}}"#;
        let remote = parse(text).expect("parses");
        match decide(remote, None, &|_| None) {
            Decision::Offer { from, .. } => assert_eq!(from, 0),
            other => panic!("expected an offer, got {other:?}"),
        }
    }
}
