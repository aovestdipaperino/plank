// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Advisory metadata beside every persisted KV blob.
//!
//! Each `<stem>.kv_raw` body may have a `<stem>.json` sidecar recording what
//! the blob is, which blob it extends, and how it has been used. The sidecar is
//! **never** a trust input: the signature inside the body
//! ([`crate::kvcache::KVCache::decode`]) remains the only thing that decides
//! whether cached bytes may be restored. A missing or corrupt sidecar costs a
//! nicer `/kvcache` row and some counters, nothing more.
//!
//! [`META_VERSION`] is this schema's own counter and is unrelated to
//! [`crate::kvcache::FORMAT_VERSION`], which versions the body. The two move
//! independently so a schema change cannot invalidate blobs.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Sidecar schema version. Independent of [`crate::kvcache::FORMAT_VERSION`].
pub const META_VERSION: u32 = 1;

/// Extension of a persisted KV body. Distinct from `.kv`, which is a session
/// transcript — before this split one glob could match both.
pub const BLOB_EXT: &str = ".kv_raw";

/// Extension of the metadata sidecar written beside a [`BLOB_EXT`] body.
pub const META_EXT: &str = "json";

/// Which tier a blob belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvRole {
    /// Tier 1: model + system prompt, shared across projects.
    System,
    /// Tier 2: project-stable context, shared by every session of a project.
    Project,
    /// Tier 4: one conversation's KV payload.
    Session,
}

impl KvRole {
    /// The word `/kvcache` prints for this role.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Project => "project",
            Self::Session => "session",
        }
    }
}

/// Role-specific detail, shown under a `/kvcache` row.
///
/// The split across variants is the audit surface for MCP segregation: global
/// server names may appear only under [`KvLabel::System`] and local ones only
/// under [`KvLabel::Project`], mirroring the tier split that `kvtier` enforces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum KvLabel {
    /// Tier 1 detail.
    System {
        /// Reasoning level the prompt was fingerprinted under.
        think_mode: String,
        /// Byte offset where trusted control text stops.
        trusted_len: usize,
        /// Global MCP servers whose tool defs entered this prefix.
        global_mcp: Vec<String>,
    },
    /// Tier 2 detail.
    Project {
        /// Absolute path of the project this tier belongs to.
        project_path: String,
        /// `AGENTS.md`/`CLAUDE.md` files that fed the stable context.
        agents_files: Vec<String>,
        /// Project-local MCP servers whose tool defs entered this prefix.
        local_mcp: Vec<String>,
    },
    /// Tier 4 detail.
    Session {
        /// Human-readable session name, e.g. `cheeky-bell`.
        name: String,
        /// Session title, as `/sessions` shows it.
        title: String,
    },
    /// No detail recorded — a synthesized or partially written sidecar.
    Unknown,
}

/// Advisory metadata for one persisted KV blob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvMeta {
    /// Schema version; a sidecar whose version is not [`META_VERSION`] is
    /// ignored.
    pub version: u32,
    /// Which tier this blob is.
    pub role: KvRole,
    /// The blob's own fingerprint — the signature its body carries.
    pub fingerprint: String,
    /// The fingerprint of the blob this one extends; `None` for a system tier.
    pub parent: Option<String>,
    /// Model name the blob was captured under; empty when unknown.
    pub model: String,
    /// Unix seconds when the blob was first written.
    pub created: u64,
    /// Unix seconds of the most recent successful load.
    pub last_used: u64,
    /// Successful loads since creation.
    pub hits: u64,
    /// Size of the body in bytes, cached so a tree render need not stat.
    pub bytes: u64,
    /// Whether the sweep must keep this blob regardless of age.
    pub pinned: bool,
    /// Role-specific detail for display.
    pub label: KvLabel,
}

impl KvMeta {
    /// Metadata for a blob found on disk with no readable sidecar: real size
    /// and mtime, no lineage, no history.
    #[must_use]
    pub fn synthesized(role: KvRole, fingerprint: &str, bytes: u64, mtime: u64) -> Self {
        Self {
            version: META_VERSION,
            role,
            fingerprint: fingerprint.to_owned(),
            parent: None,
            model: String::new(),
            created: mtime,
            last_used: mtime,
            hits: 0,
            bytes,
            pinned: false,
            label: KvLabel::Unknown,
        }
    }
}

/// Current wall clock in Unix seconds; `0` if the clock is before the epoch.
#[must_use]
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The sidecar that belongs to `blob`: same stem, `.json` extension.
#[must_use]
pub fn sidecar_path(blob: &Path) -> PathBuf {
    blob.with_extension(META_EXT)
}

/// Reads the sidecar beside `blob`.
///
/// `None` for absent, unreadable, malformed, or wrong-version sidecars. Callers
/// substitute [`KvMeta::synthesized`]; none of them treat `None` as a reason to
/// distrust the body.
#[must_use]
pub fn load(blob: &Path) -> Option<KvMeta> {
    let text = std::fs::read_to_string(sidecar_path(blob)).ok()?;
    let meta: KvMeta = serde_json::from_str(&text).ok()?;
    (meta.version == META_VERSION).then_some(meta)
}

/// Writes the sidecar beside `blob`, via a pid-suffixed temp file and a rename.
///
/// The tmp-then-rename shape matches [`crate::kvcache::KVCache::persist`]: two
/// plank processes may write the same sidecar, and a half-written one that
/// still parses would report nonsense counters.
///
/// # Errors
/// Returns the underlying [`std::io::Error`] when the write or rename fails.
/// Every caller treats a failure as best-effort.
pub fn store(blob: &Path, meta: &KvMeta) -> std::io::Result<()> {
    let path = sidecar_path(blob);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let name = path.file_name().unwrap_or_default();
    let mut tmp_name = name.to_os_string();
    tmp_name.push(format!(".tmp.{}", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    let body = serde_json::to_vec_pretty(meta)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plank-kvmeta-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn sample() -> KvMeta {
        KvMeta {
            version: META_VERSION,
            role: KvRole::Project,
            fingerprint: "7c02".into(),
            parent: Some("a19f".into()),
            model: "test-model".into(),
            created: 100,
            last_used: 200,
            hits: 41,
            bytes: 4096,
            pinned: false,
            label: KvLabel::Project {
                project_path: "/Users/x/Code/plank".into(),
                agents_files: vec!["AGENTS.md".into()],
                local_mcp: vec!["tokensave".into()],
            },
        }
    }

    #[test]
    fn sidecar_path_swaps_the_blob_extension() {
        assert_eq!(
            sidecar_path(Path::new("/c/sysprompt-a19f.kv_raw")),
            PathBuf::from("/c/sysprompt-a19f.json")
        );
        // A session blob and its transcript differ only by extension; the
        // sidecar must land on the blob, never on the transcript.
        assert_eq!(
            sidecar_path(Path::new("/c/cheeky-bell.kv_raw")),
            PathBuf::from("/c/cheeky-bell.json")
        );
    }

    #[test]
    fn roundtrips_through_store_and_load() {
        let dir = tmpdir("roundtrip");
        let blob = dir.join("project-7c02.kv_raw");
        store(&blob, &sample()).unwrap();
        let back = load(&blob).expect("sidecar loads");
        assert_eq!(back.fingerprint, "7c02");
        assert_eq!(back.parent.as_deref(), Some("a19f"));
        assert_eq!(back.hits, 41);
        assert!(matches!(back.role, KvRole::Project));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn absent_and_corrupt_sidecars_are_none_not_panics() {
        let dir = tmpdir("corrupt");
        let blob = dir.join("project-7c02.kv_raw");
        assert!(load(&blob).is_none(), "absent sidecar");
        std::fs::write(sidecar_path(&blob), b"{not json").unwrap();
        assert!(load(&blob).is_none(), "corrupt sidecar");
        std::fs::write(sidecar_path(&blob), b"{}").unwrap();
        assert!(load(&blob).is_none(), "sidecar missing required fields");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_future_schema_version_is_ignored() {
        // Forward compatibility runs one way only: an older plank must not
        // interpret a newer sidecar's fields, but must still leave the blob
        // usable. Rejecting here means counters reset, not that data is lost.
        let dir = tmpdir("future");
        let blob = dir.join("x.kv_raw");
        let mut m = sample();
        m.version = META_VERSION + 1;
        store(&blob, &m).unwrap();
        assert!(load(&blob).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthesized_metadata_is_usable_but_lineage_free() {
        let m = KvMeta::synthesized(KvRole::System, "a19f", 4096, 1_700_000_000);
        assert_eq!(m.fingerprint, "a19f");
        assert_eq!(m.bytes, 4096);
        assert_eq!(m.last_used, 1_700_000_000);
        assert_eq!(m.created, 1_700_000_000);
        assert_eq!(m.hits, 0);
        assert!(!m.pinned);
        assert!(m.parent.is_none());
        assert!(matches!(m.label, KvLabel::Unknown));
    }

    #[test]
    fn store_is_atomic_and_leaves_no_temp_behind() {
        let dir = tmpdir("atomic");
        let blob = dir.join("x.kv_raw");
        store(&blob, &sample()).unwrap();
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["x.json".to_owned()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
