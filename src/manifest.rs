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

/// An engine's name, validated and interned so it stays `Copy`.
///
/// Every artifact path in this module is scoped by one of these. Engines are
/// kept wholly separate on disk — separate installed records, separate staging
/// directories — because the invariant that makes a swap safe is per-engine:
/// the installed record moves *last*, so its presence proves that engine landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EngineId(&'static str);

impl EngineId {
    /// `DeepSeek` V4 Flash Vision-Experimental, the shipped default.
    pub const DS4VISION: Self = Self("ds4vision");
    /// `DeepSeek` V4.1 Flash.
    pub const DS41: Self = Self("ds41");
    /// Qwen3.8-Flash-Next.
    pub const QWEN: Self = Self("qwen");

    /// A validated, interned id, or `None` for an invalid name.
    ///
    /// Interning leaks one small string per distinct name for the process
    /// lifetime; the catalog holds a handful, so that is the price of `Copy`.
    #[must_use]
    pub fn new(name: &str) -> Option<Self> {
        use std::sync::{Mutex, OnceLock};
        static POOL: OnceLock<Mutex<std::collections::BTreeSet<&'static str>>> = OnceLock::new();
        if !crate::engines::valid_name(name) {
            return None;
        }
        let mut pool = POOL
            .get_or_init(|| Mutex::new(std::collections::BTreeSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(s) = pool.get(name) {
            return Some(Self(s));
        }
        let s: &'static str = Box::leak(name.to_string().into_boxed_str());
        pool.insert(s);
        Some(Self(s))
    }

    /// The name, as the CLI and the on-disk layout spell it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        self.0
    }

    /// Parses the detached helper's argv. A helper spawned by a plank that
    /// predates engines passes `ds4` or nothing, which is today's `ds4vision`.
    #[must_use]
    pub fn from_legacy_arg(s: &str) -> Option<Self> {
        match s {
            "" | "ds4" => Some(Self::DS4VISION),
            other => Self::new(other),
        }
    }
}

impl std::fmt::Display for EngineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

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

/// The Hugging Face *repository page* behind an artifact URL, or `None` when
/// the URL is not a Hugging Face `resolve` link.
///
/// Deliberately not the artifact URL itself. A manifest entry points at
/// `…/resolve/main/<file>`, which is the download: putting that in a bug
/// report invites a maintainer to click it and start fetching ~87 GB. The repo
/// page is the thing a human actually wants to open, and the file name is
/// recorded separately beside it.
#[must_use]
pub fn hf_repo_url(artifact_url: &str) -> Option<String> {
    let rest = artifact_url.strip_prefix("https://huggingface.co/")?;
    let (repo, _) = rest.split_once("/resolve/")?;
    // `<owner>/<name>` exactly — anything else is a shape this does not know.
    let mut parts = repo.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return None;
    };
    (!owner.is_empty() && !name.is_empty())
        .then(|| format!("https://huggingface.co/{owner}/{name}"))
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
    for (kind, entry) in &wire.files {
        if !is_sha256_hex(&entry.sha256) {
            return Err(format!(
                "bad manifest: {kind}.sha256 is not 64 lowercase hex characters"
            ));
        }
        if entry.bytes == 0 {
            return Err(format!("bad manifest: {kind}.bytes must not be 0"));
        }
        if !entry.url.starts_with("https://") {
            return Err(format!("bad manifest: {kind}.url must start with https://"));
        }
    }
    Ok(Manifest {
        version: wire.version,
        released: wire.released,
        notes: wire.notes,
        files: wire.files,
        raw: text.to_string(),
    })
}

/// Whether `s` is exactly 64 lowercase hex characters — a well-formed
/// SHA-256 digest.
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// `~/.plank`, or `./.plank` when `HOME` is unset.
///
/// Mirrors `download::default_model_path`'s fallback so the two never disagree
/// about where the model lives.
#[must_use]
pub fn plank_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    crate::home::plank_home_in(home)
}

/// The installed record: what the files of engine `id` under `root` are.
#[must_use]
pub fn installed_path_in(root: &Path, id: EngineId) -> PathBuf {
    root.join("engines").join(format!("{id}.installed.json"))
}

/// The installed record: what the files of engine `id` in `~/.plank` are.
///
/// Written only by a successful swap, and written *last*, so its presence is
/// proof the whole engine landed.
#[must_use]
pub fn installed_path(id: EngineId) -> PathBuf {
    installed_path_in(&plank_dir(), id)
}

/// Where in-flight and verified-but-not-yet-installed artifacts of engine
/// `id` live, under `root`.
#[must_use]
pub fn staging_dir_in(root: &Path, id: EngineId) -> PathBuf {
    root.join("staging").join(id.as_str())
}

/// Where in-flight and verified-but-not-yet-installed artifacts live.
#[must_use]
pub fn staging_dir(id: EngineId) -> PathBuf {
    staging_dir_in(&plank_dir(), id)
}

/// The staged installed record: moved to [`installed_path_in`] last.
#[must_use]
pub fn staged_manifest_path_in(root: &Path, id: EngineId) -> PathBuf {
    staging_dir_in(root, id).join(format!("{id}.installed.json"))
}

/// Helper-process bookkeeping under `root`: lock, job, state, cancel flag, log.
#[must_use]
pub fn downloads_dir_in(root: &Path) -> PathBuf {
    root.join("downloads")
}

/// Helper-process bookkeeping: lock, job, state, cancel flag, log.
#[must_use]
pub fn downloads_dir() -> PathBuf {
    downloads_dir_in(&plank_dir())
}

/// Where engine `id`'s artifact of `role` is installed under `root`, or
/// `None` for a role this build does not know.
///
/// Deliberately local knowledge rather than a manifest field: a manifest that
/// could name its own destination path would be a manifest that could write
/// anywhere on disk. The filename is derived from the engine name alone, so
/// two engines can never land on the same file.
#[must_use]
pub fn local_path_for_in(root: &Path, id: EngineId, role: &str) -> Option<PathBuf> {
    match role {
        "main" => Some(root.join(format!("{id}.gguf"))),
        "mtp" | "vision" => Some(root.join(format!("{id}.{role}.gguf"))),
        _ => None,
    }
}

/// Where engine `id`'s artifact of `role` is installed under `~/.plank`.
#[must_use]
pub fn local_path_for(id: EngineId, role: &str) -> Option<PathBuf> {
    local_path_for_in(&plank_dir(), id, role)
}

/// Engines with an installed record under `root`, sorted.
#[must_use]
pub fn installed_ids_in(root: &Path) -> Vec<EngineId> {
    let Ok(dir) = std::fs::read_dir(root.join("engines")) else {
        return Vec::new();
    };
    let mut ids: Vec<EngineId> = dir
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            EngineId::new(name.strip_suffix(".installed.json")?)
        })
        .collect();
    ids.sort();
    ids
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
/// Only roles in both `kinds` and the manifest are considered: a manifest
/// entry this build does not know how to install cannot block adoption, and a
/// role the manifest omits cannot be demanded on disk.
#[must_use]
pub fn decide(
    remote: Manifest,
    installed: Option<&Manifest>,
    kinds: &[&str],
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
    let intersection: Vec<_> = kinds
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

    /// The repo page, not the artifact URL: a `/resolve/` link in a bug report
    /// is an invitation to start an 87 GB download by clicking it.
    #[test]
    fn an_artifact_url_yields_its_hugging_face_repo_page() {
        assert_eq!(
            super::hf_repo_url(
                "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/Model-Q2.gguf"
            )
            .as_deref(),
            Some("https://huggingface.co/antirez/deepseek-v4-gguf")
        );
        // Anything that is not a Hugging Face resolve link has no repo page,
        // and a mirror or a self-hosted manifest is a perfectly ordinary case.
        for other in [
            "https://example.com/models/main.gguf",
            "https://huggingface.co/antirez/deepseek-v4-gguf",
            "https://huggingface.co/too/many/segments/resolve/main/f.gguf",
            "https://huggingface.co//resolve/main/f.gguf",
        ] {
            assert_eq!(super::hf_repo_url(other), None, "{other}");
        }
    }

    use super::*;

    /// 64 lowercase hex characters, distinguishable by their leading digit so
    /// tests can tell entries apart at a glance.
    const SHA_MAIN: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const SHA_VISION: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const SHA_MTP: &str = "3333333333333333333333333333333333333333333333333333333333333333";
    const SHA_OTHER: &str = "4444444444444444444444444444444444444444444444444444444444444444";

    /// A complete, well-formed manifest, used by most tests here.
    fn sample() -> String {
        format!(
            r#"{{
          "version": 3,
          "released": "2026-09-04",
          "notes": "Vision-Exp refresh",
          "files": {{
            "main":   {{ "name": "m.gguf", "url": "https://example.invalid/m", "bytes": 100, "sha256": "{SHA_MAIN}" }},
            "vision": {{ "name": "v.gguf", "url": "https://example.invalid/v", "bytes": 200, "sha256": "{SHA_VISION}" }},
            "mtp":    {{ "name": "d.gguf", "url": "https://example.invalid/d", "bytes": 300, "sha256": "{SHA_MTP}" }}
          }}
        }}"#
        )
    }

    #[test]
    fn parses_every_field() {
        let m = parse(&sample()).expect("sample parses");
        assert_eq!(m.version, 3);
        assert_eq!(m.released, "2026-09-04");
        assert_eq!(m.notes, "Vision-Exp refresh");
        assert_eq!(m.files.len(), 3);
        let main = m.files.get("main").expect("main entry");
        assert_eq!(main.name, "m.gguf");
        assert_eq!(main.url, "https://example.invalid/m");
        assert_eq!(main.bytes, 100);
        assert_eq!(main.sha256, SHA_MAIN);
    }

    #[test]
    fn keeps_the_raw_bytes_verbatim() {
        // The installed manifest is written by copying `raw`, never by
        // re-serializing: a round trip through serde would silently drop any
        // field this build does not know about.
        let m = parse(&sample()).expect("sample parses");
        assert_eq!(m.raw, sample());
    }

    #[test]
    fn an_unknown_file_kind_is_ignored_not_an_error() {
        // The manifest must be able to grow a fourth artifact before the
        // client reading it knows what that artifact is.
        let text = sample().replace(
            r#""mtp":"#,
            &format!(
                r#""futureproof": {{ "name": "f.gguf", "url": "https://example.invalid/f", "bytes": 1, "sha256": "{SHA_OTHER}" }}, "mtp":"#
            ),
        );
        let m = parse(&text).expect("unknown kind parses");
        assert_eq!(m.files.len(), 4);
        assert!(m.files.contains_key("futureproof"));
    }

    #[test]
    fn a_missing_mtp_entry_parses() {
        // Not every release has to ship all three. Absence is a fact for
        // `decide` (Task 2) to act on, not a parse failure.
        let text = format!(
            r#"{{"version":1,"released":"x","notes":"","files":{{
            "main": {{ "name": "m.gguf", "url": "https://example.invalid/m", "bytes": 1, "sha256": "{SHA_MAIN}" }}
        }}}}"#
        );
        let m = parse(&text).expect("partial manifest parses");
        assert!(m.files.contains_key("main"));
        assert!(!m.files.contains_key("mtp"));
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
    fn in_variants_nest_under_the_given_root() {
        let root = Path::new("/tmp/some-root");
        assert_eq!(
            installed_path_in(root, EngineId::DS4VISION),
            root.join("engines/ds4vision.installed.json")
        );
        assert_eq!(
            staging_dir_in(root, EngineId::DS4VISION),
            root.join("staging/ds4vision")
        );
        assert_eq!(downloads_dir_in(root), root.join("downloads"));
        assert_eq!(
            local_path_for_in(root, EngineId::DS4VISION, "main"),
            Some(root.join("ds4vision.gguf"))
        );
        assert_eq!(
            local_path_for_in(root, EngineId::DS4VISION, "vision"),
            Some(root.join("ds4vision.vision.gguf"))
        );
        assert_eq!(
            local_path_for_in(root, EngineId::DS4VISION, "mtp"),
            Some(root.join("ds4vision.mtp.gguf"))
        );
        assert_eq!(local_path_for_in(root, EngineId::DS4VISION, "bogus"), None);
    }

    #[test]
    fn paths_nest_under_the_plank_directory() {
        let root = plank_dir();
        assert_eq!(
            installed_path(EngineId::DS4VISION),
            root.join("engines/ds4vision.installed.json")
        );
        assert_eq!(
            staging_dir(EngineId::DS4VISION),
            root.join("staging/ds4vision")
        );
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
            "mtp" => Some(300),
            _ => None,
        }
    }

    #[test]
    fn same_version_is_up_to_date() {
        let remote = parse(&sample()).expect("parses");
        let installed = parse(&sample()).expect("parses");
        assert!(matches!(
            decide(
                remote,
                Some(&installed),
                &crate::engines::ROLES,
                &all_present
            ),
            Decision::UpToDate
        ));
    }

    #[test]
    fn a_newer_remote_is_offered() {
        let remote = parse(&sample_at(4)).expect("parses");
        let installed = parse(&sample()).expect("parses");
        match decide(
            remote,
            Some(&installed),
            &crate::engines::ROLES,
            &all_present,
        ) {
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
        let installed = parse(&sample()).expect("parses");
        assert!(matches!(
            decide(
                remote,
                Some(&installed),
                &crate::engines::ROLES,
                &all_present
            ),
            Decision::UpToDate
        ));
    }

    /// The two sets must not share a single byte of disk state. A swap is
    /// only safe because the manifest moves last within its own staging area;
    /// one shared area would let a half-staged Qwen download read as proof
    /// about the `DeepSeek` set.
    #[test]
    fn the_two_sets_never_share_a_path() {
        let root = Path::new("/tmp/plank-set-test");
        for (a, b) in [
            (
                installed_path_in(root, EngineId::DS4VISION),
                installed_path_in(root, EngineId::QWEN),
            ),
            (
                staging_dir_in(root, EngineId::DS4VISION),
                staging_dir_in(root, EngineId::QWEN),
            ),
        ] {
            assert_ne!(a, b);
        }
        assert_ne!(
            local_path_for_in(root, EngineId::DS4VISION, "main"),
            local_path_for_in(root, EngineId::QWEN, "main"),
        );
    }

    /// Two engines must never collide on any path they write.
    ///
    /// Load-bearing, not cosmetic: the swap's guarantee is that the installed
    /// record moves *last*, so its presence proves that engine landed whole.
    /// Share a staging directory or a job file between two engines and a
    /// half-staged download of one could be read as proof about the other.
    #[test]
    fn the_sets_never_share_a_path() {
        let root = Path::new("/tmp/plank-set-disjoint");
        let ids = [EngineId::DS4VISION, EngineId::DS41, EngineId::QWEN];
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                let (a, b) = (*a, *b);
                assert_ne!(a.as_str(), b.as_str(), "{a} vs {b}");
                assert_ne!(
                    staging_dir_in(root, a),
                    staging_dir_in(root, b),
                    "{a} vs {b}"
                );
                assert_ne!(
                    installed_path_in(root, a),
                    installed_path_in(root, b),
                    "{a} vs {b}"
                );
                assert_ne!(
                    staged_manifest_path_in(root, a),
                    staged_manifest_path_in(root, b),
                    "{a} vs {b}"
                );
                // The job file too: one engine's pending job must never be
                // read as the other's, or a helper would download one engine
                // against the other's manifest.
                assert_ne!(
                    crate::downloader::job_path_in(root, a),
                    crate::downloader::job_path_in(root, b),
                    "{a} vs {b}"
                );
                // Every install path of one engine, against every install path
                // of the other: two engines sharing a role name must still
                // land on different files.
                for ka in crate::engines::ROLES {
                    for kb in crate::engines::ROLES {
                        let (pa, pb) = (
                            local_path_for_in(root, a, ka),
                            local_path_for_in(root, b, kb),
                        );
                        assert!(pa.is_some() && pb.is_some(), "{a}/{ka} {b}/{kb}");
                        assert_ne!(pa, pb, "{a}/{ka} collides with {b}/{kb}");
                        assert_ne!(
                            local_path_for(a, ka),
                            local_path_for(b, kb),
                            "{a}/{ka} collides with {b}/{kb}"
                        );
                    }
                }
            }
        }
    }

    /// The helper is handed its engine in argv, and one spawned by a plank
    /// that predates engines passes nothing.
    #[test]
    fn the_set_round_trips_through_argv() {
        for id in [EngineId::DS4VISION, EngineId::DS41, EngineId::QWEN] {
            assert_eq!(EngineId::from_legacy_arg(id.as_str()), Some(id), "{id}");
        }
        assert_eq!(EngineId::from_legacy_arg("ds4"), Some(EngineId::DS4VISION));
        // An older helper passes nothing at all; that path must keep working.
        assert_eq!(EngineId::from_legacy_arg(""), Some(EngineId::DS4VISION));
    }

    /// Adoption considers only the roles a manifest lists, so a Qwen manifest
    /// naming `main` and `mtp` is adopted on those two files without a vision
    /// encoder in sight.
    #[test]
    fn a_qwen_manifest_adopts_on_its_own_two_kinds() {
        let remote = parse(
            r#"{"version":1,"released":"t","notes":"","files":{
                "main": {"name":"q.gguf","url":"https://example.invalid/q","bytes":10,
                         "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
                "mtp":  {"name":"p.gguf","url":"https://example.invalid/p","bytes":20,
                         "sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}}"#,
        )
        .expect("parse");
        let size_of = |kind: &str| match kind {
            "main" => Some(10),
            "mtp" => Some(20),
            _ => None,
        };
        match decide(remote, None, &crate::engines::ROLES, &size_of) {
            Decision::Adopt(m) => assert_eq!(m.version, 1),
            other => panic!("expected adoption, got {other:?}"),
        }
    }

    #[test]
    fn no_installed_manifest_but_matching_sizes_adopts_silently() {
        // Adopt-on-first-sight. Without this rule, every existing user is offered
        // an 87 GB re-download the day this ships.
        let remote = parse(&sample()).expect("parses");
        match decide(remote, None, &crate::engines::ROLES, &all_present) {
            Decision::Adopt(m) => assert_eq!(m.version, 3),
            other => panic!("expected adoption, got {other:?}"),
        }
    }

    #[test]
    fn no_installed_manifest_and_a_wrong_size_offers() {
        let remote = parse(&sample()).expect("parses");
        let sizes = |kind: &str| {
            if kind == "vision" {
                Some(999)
            } else {
                all_present(kind)
            }
        };
        match decide(remote, None, &crate::engines::ROLES, &sizes) {
            Decision::Offer { from, .. } => assert_eq!(from, 0),
            other => panic!("expected an offer, got {other:?}"),
        }
    }

    #[test]
    fn no_installed_manifest_and_a_missing_file_offers() {
        let remote = parse(&sample()).expect("parses");
        let sizes = |kind: &str| {
            if kind == "mtp" {
                None
            } else {
                all_present(kind)
            }
        };
        match decide(remote, None, &crate::engines::ROLES, &sizes) {
            Decision::Offer { from, .. } => assert_eq!(from, 0),
            other => panic!("expected an offer, got {other:?}"),
        }
    }

    #[test]
    fn adoption_only_considers_kinds_the_manifest_actually_lists() {
        // A manifest with no mtp entry must not demand an mtp file on disk.
        let text = format!(
            r#"{{"version":1,"released":"x","notes":"","files":{{
            "main": {{ "name": "m.gguf", "url": "https://example.invalid/m", "bytes": 100, "sha256": "{SHA_MAIN}" }}
        }}}}"#
        );
        let remote = parse(&text).expect("parses");
        let sizes = |kind: &str| (kind == "main").then_some(100);
        assert!(matches!(
            decide(remote, None, &crate::engines::ROLES, &sizes),
            Decision::Adopt(_)
        ));
    }

    #[test]
    fn an_unknown_kind_is_not_required_on_disk() {
        // `futureproof` is in the manifest but this build cannot install it, so it
        // must not block adoption or the size check.
        let text = sample().replace(
            r#""mtp":"#,
            &format!(
                r#""futureproof": {{ "name": "f.gguf", "url": "https://example.invalid/f", "bytes": 7, "sha256": "{SHA_OTHER}" }}, "mtp":"#
            ),
        );
        let remote = parse(&text).expect("parses");
        assert!(matches!(
            decide(remote, None, &crate::engines::ROLES, &all_present),
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
        match decide(remote, None, &crate::engines::ROLES, &|_| None) {
            Decision::Offer { from, .. } => assert_eq!(from, 0),
            other => panic!("expected an offer, got {other:?}"),
        }
    }

    #[test]
    fn a_sha256_that_is_not_64_hex_characters_is_rejected() {
        for bad in ["aa", "", &"a".repeat(63), &"a".repeat(65), &"g".repeat(64)] {
            let text = sample().replace(SHA_MAIN, bad);
            assert!(parse(&text).is_err(), "{bad:?} should not pass as a sha256");
        }
    }

    #[test]
    fn an_uppercase_sha256_is_rejected() {
        // Only lowercase hex — a typo'd or mixed-case digest must not slip
        // through and cost a full download before the mismatch is caught.
        let mixed_case = "aB".repeat(32);
        let text = sample().replace(SHA_MAIN, &mixed_case);
        assert!(parse(&text).is_err());
    }

    #[test]
    fn zero_bytes_is_rejected() {
        let text = sample().replace(r#""bytes": 100"#, r#""bytes": 0"#);
        assert!(parse(&text).is_err());
    }

    #[test]
    fn a_non_https_url_is_rejected() {
        for scheme in ["http://example.invalid/m", "file:///etc/passwd", "ftp://x"] {
            let text = sample().replace("https://example.invalid/m", scheme);
            assert!(
                parse(&text).is_err(),
                "{scheme:?} must not be accepted as a manifest url"
            );
        }
    }

    #[test]
    fn engine_ids_validate_and_intern() {
        let a = EngineId::new("my-engine").expect("valid");
        let b = EngineId::new(&String::from("my-engine")).expect("valid");
        assert_eq!(a, b);
        assert_eq!(a.as_str(), "my-engine");
        assert!(EngineId::new("Bad").is_none());
        assert_eq!(EngineId::new("ds4vision"), Some(EngineId::DS4VISION));
    }

    #[test]
    fn legacy_helper_args_map_onto_engines() {
        assert_eq!(EngineId::from_legacy_arg(""), Some(EngineId::DS4VISION));
        assert_eq!(EngineId::from_legacy_arg("ds4"), Some(EngineId::DS4VISION));
        assert_eq!(EngineId::from_legacy_arg("ds41"), Some(EngineId::DS41));
        assert_eq!(EngineId::from_legacy_arg("qwen"), Some(EngineId::QWEN));
        assert_eq!(EngineId::from_legacy_arg("NOPE"), None);
    }

    #[test]
    fn install_paths_are_derived_from_the_engine_name() {
        let root = Path::new("/r");
        let id = EngineId::new("foo").unwrap();
        assert_eq!(
            local_path_for_in(root, id, "main"),
            Some(root.join("foo.gguf"))
        );
        assert_eq!(
            local_path_for_in(root, id, "mtp"),
            Some(root.join("foo.mtp.gguf"))
        );
        assert_eq!(
            local_path_for_in(root, id, "vision"),
            Some(root.join("foo.vision.gguf"))
        );
        assert_eq!(local_path_for_in(root, id, "future"), None);
        assert_eq!(
            installed_path_in(root, id),
            root.join("engines/foo.installed.json")
        );
        assert_eq!(staging_dir_in(root, id), root.join("staging/foo"));
        assert_eq!(
            staged_manifest_path_in(root, id),
            root.join("staging/foo/foo.installed.json")
        );
    }

    #[test]
    fn engines_never_share_a_path() {
        let root = Path::new("/r");
        let ids = [EngineId::DS4VISION, EngineId::DS41, EngineId::QWEN];
        let mut seen = std::collections::HashSet::new();
        for id in ids {
            for p in [installed_path_in(root, id), staging_dir_in(root, id)]
                .into_iter()
                .chain(
                    crate::engines::ROLES
                        .iter()
                        .filter_map(|r| local_path_for_in(root, id, r)),
                )
            {
                assert!(seen.insert(p.clone()), "{} shared", p.display());
            }
        }
    }

    #[test]
    fn installed_ids_are_found_by_scanning_the_engines_dir() {
        let root = std::env::temp_dir().join(format!("plank-ids-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("engines")).unwrap();
        std::fs::write(root.join("engines/qwen.installed.json"), "{}").unwrap();
        std::fs::write(root.join("engines/Bad.installed.json"), "{}").unwrap();
        std::fs::write(root.join("engines/notes.txt"), "").unwrap();
        assert_eq!(installed_ids_in(&root), vec![EngineId::QWEN]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_valid_entry_still_parses() {
        // The validation above must not be so strict it rejects the sample
        // fixture itself.
        assert!(parse(&sample()).is_ok());
    }
}
