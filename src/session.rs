// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Agent session persistence, listing, and history rendering.
//!
//! Port of the ds4 agent sections "Agent KV Store And Session Persistence"
//! and "Session Listing, History Rendering, And Completion". The C agent
//! persists engine KV state plus a rendered token transcript in one file;
//! plank persists the text transcript as the session file and keeps the
//! engine KV state in a fingerprinted `<name>.kv_raw` sidecar (a
//! [`crate::kvcache::KVCache`] keyed by [`KvKey::Session`]), while keeping the
//! same user-visible behavior: sessions
//! live under `~/.plank/kvcache` as `<name>.kv` files. A session id is a
//! memorable `adjective-celebrity` name minted on first save (e.g.
//! `deadly-einstein`), disambiguated with a short guid on the rare filename
//! collision; legacy `<40-hex>.kv` files from earlier versions still load and
//! list, and older payload-less sessions load fine (the sidecar is a pure
//! cache). Titles derive from the first user prompt, listings sort most recent
//! first, and history replay uses the exact `User:` / `Assistant:` /
//! `Tool result:` headers of the C agent.
//!
//! # On-disk format
//!
//! A session file is line-oriented UTF-8 with length-prefixed bodies:
//!
//! ```text
//! plank-session 1
//! created <unix-seconds>
//! used <unix-seconds>
//! title <byte-len>
//! <title bytes>
//! msg <user|assistant> <byte-len> [<unix-seconds>]
//! <message bytes>
//! ...
//! node <id> <parent-id|-> <user|assistant> <byte-len> [<unix-seconds>]
//! <message bytes>
//! ...
//! cwd <byte-len>
//! <project directory bytes>
//! meta <tag-byte-len> <last-prompt-byte-len>
//! <tag bytes>
//! <last prompt bytes>
//! ```
//!
//! Length prefixes make message bodies unambiguous regardless of content.
//! The trailing `meta` record duplicates listing metadata (tag, clipped last
//! user prompt) at the end of the file so `list()` can read it with a
//! bounded tail read instead of parsing the whole transcript; files written
//! before it existed simply lack the record.
//!
//! The trailing timestamp on a `msg`/`node` record is the wall-clock second
//! the message was appended, added for `/insights` (response times, activity
//! by hour, overlapping-session detection). It is optional in both directions:
//! records written before it existed parse with a zero timestamp, and a
//! message with a zero timestamp is written without the field, so a session
//! that never carried stamps stays byte-identical. The `cwd` record is the
//! same kind of optional addition, recording the project directory the session
//! ran in (which the transcript itself does not state).
//!
//! The `msg` records are the session tree's *active branch*; the optional
//! `node` records describe the off-path branches left by `/fork` and `/clone`
//! (issue #65), numbering the active branch `0..n` and continuing from there.
//! A linear session writes no `node` records at all, and a file without them
//! loads as a single-branch tree — that is what keeps every pre-branching
//! session readable.

use std::error::Error;
use std::fmt::{self, Write as _};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default number of user turns replayed by history rendering.
pub const HISTORY_DEFAULT_TURNS: usize = 3;
/// Maximum user turns history rendering will replay.
pub const HISTORY_MAX_TURNS: usize = 200;

const HISTORY_USER_MAX_LINES: usize = 24;
const HISTORY_USER_MAX_BYTES: usize = 6000;
const HISTORY_ASSISTANT_MAX_LINES: usize = 80;
const HISTORY_ASSISTANT_MAX_BYTES: usize = 12000;
const HISTORY_TOOL_MAX_LINES: usize = 12;
const HISTORY_TOOL_MAX_BYTES: usize = 3000;

const MAGIC: &str = "plank-session 1";
const FILE_EXT: &str = ".kv";
/// Extension of the engine KV payload written beside a transcript.
///
/// The C agent stores the engine payload inside the same `.kv` file as the
/// rendered text; plank's v1 transcript format predates payloads and must
/// keep loading, so the payload lives in a sidecar instead. The sidecar is a
/// rebuildable cache: its signature is a fingerprint tying it to the exact
/// model, system prompt, and rendered transcript, and a mismatch means the
/// payload is ignored and rebuilt by prefill — never trusted (see
/// [`payload_fingerprint`] and [`SessionStore::kv_load`]).
///
/// It shares one extension with the Tier 1 and Tier 2 checkpoints, distinct
/// from the transcripts' [`FILE_EXT`]: every KV body in the cache is a
/// `.kv_raw`, so one scan finds them all and no glob confuses a checkpoint for
/// a transcript.
const PAYLOAD_EXT: &str = crate::kvmeta::BLOB_EXT;

/// Extension the engine KV payload used before the `.kv_raw` unification.
/// Referenced only by [`SessionStore::migrate_legacy_blobs`].
const LEGACY_PAYLOAD_EXT: &str = ".payload";

/// Marker recording that [`SessionStore::migrate_legacy_blobs`] has run. Its
/// presence, not its contents, is the signal.
const MIGRATION_MARKER: &str = ".kvformat-2";

/// Error raised by session store operations.
///
/// Wraps both I/O failures and user-level failures (bad prefix, ambiguous
/// prefix, corrupt file) with a human-readable message matching the C
/// agent's wording where one exists.
pub struct SessionError {
    message: String,
    source: Option<io::Error>,
}

impl SessionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl fmt::Debug for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionError")
            .field("message", &self.message)
            .field("source", &self.source)
            .finish()
    }
}

impl Error for SessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|e| e as &(dyn Error + 'static))
    }
}

impl From<io::Error> for SessionError {
    fn from(e: io::Error) -> Self {
        Self {
            message: e.to_string(),
            source: Some(e),
        }
    }
}

/// Result alias for session store operations.
pub type Result<T> = std::result::Result<T, SessionError>;

/// Speaker of one transcript message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// A human (or tool-result pseudo-user) turn.
    User,
    /// A model turn.
    Assistant,
}

impl Role {
    fn tag(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One role-tagged message in a session transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Who produced the message.
    pub role: Role,
    /// Raw message text.
    pub text: String,
    /// Wall-clock second the message was appended, or 0 when unknown (every
    /// message loaded from a file written before timestamps existed, and every
    /// message built outside [`Session::push`]). Consumed by
    /// [`crate::insights`]; a zero is written to disk as an absent field.
    pub at: u64,
}

impl Message {
    /// Creates a user message with no timestamp.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            text: text.into(),
            at: 0,
        }
    }

    /// Creates an assistant message with no timestamp.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            text: text.into(),
            at: 0,
        }
    }

    /// Stamps the message with a wall-clock second, replacing any existing one.
    #[must_use]
    pub fn at(mut self, at: u64) -> Self {
        self.at = at;
        self
    }

    /// Tool results are stored as user turns; detect them like the C agent.
    pub(crate) fn is_tool_user(&self) -> bool {
        let t = self.text.trim();
        t.starts_with("<tool_result>") || t.starts_with("Tool:") || t.starts_with("Tool result")
    }

    /// Strips a `<tool_result>` wrapper, returning the inner payload.
    pub(crate) fn tool_result_payload(&self) -> &str {
        let t = self.text.trim();
        if let Some(inner) = t.strip_prefix("<tool_result>") {
            inner.strip_suffix("</tool_result>").unwrap_or(inner)
        } else {
            t
        }
    }
}

/// A resumable conversation session.
///
/// The id is a memorable `adjective-celebrity` name minted on first save;
/// resaving keeps the same file name while the transcript evolves.
#[derive(Debug, Clone)]
pub struct Session {
    /// Memorable `adjective-celebrity` id (or a legacy 40-hex id); empty
    /// until the first save.
    pub id: String,
    /// Title derived from the first user prompt (or set explicitly).
    pub title: String,
    /// Creation time in unix seconds; 0 until the first save.
    pub created_at: u64,
    /// User-assigned tag shown in listings (`/tag`); empty when unset.
    pub tag: String,
    /// Project directory the session ran in; empty when unknown (sessions
    /// saved before the field existed). Recorded for `/insights`, which groups
    /// work by project and cannot recover the path from the transcript.
    pub cwd: String,
    /// Alternating role-tagged messages: the *active branch* of the session
    /// tree (issue #65). Every existing caller keeps treating this as the
    /// whole conversation.
    pub transcript: Vec<Message>,
    /// Off-path nodes of the session tree — the branches `/fork` and `/clone`
    /// left behind ([`crate::branch`]). Empty for a linear session, in which
    /// case nothing extra is written and the file stays byte-identical to one
    /// written before branching existed.
    pub branches: Vec<crate::branch::OffNode>,
    /// Model-visible task list (issue #35), persisted next to the transcript so
    /// it survives compaction, `/resume`, and checkpoint rollback.
    pub tasks: crate::tasks::TaskList,
    /// True when the transcript has unsaved changes.
    pub dirty: bool,
}

impl Session {
    /// Creates an empty, unsaved session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: String::new(),
            title: String::new(),
            created_at: 0,
            tag: String::new(),
            cwd: String::new(),
            transcript: Vec::new(),
            branches: Vec::new(),
            tasks: crate::tasks::TaskList::new(),
            dirty: false,
        }
    }

    /// Appends a message and marks the session dirty.
    ///
    /// An unstamped message is stamped with the current second here: `push` is
    /// the one path every live turn takes, so stamping at the door keeps the
    /// timing data `/insights` needs without every call site knowing about it.
    /// Messages built by compaction or branch surgery keep whatever stamp they
    /// carry.
    pub fn push(&mut self, mut message: Message) {
        if message.at == 0 {
            message.at = unix_now();
        }
        self.transcript.push(message);
        self.dirty = true;
    }

    /// The session as a tree: the active branch plus the off-path nodes.
    #[must_use]
    pub fn tree(&self) -> crate::branch::BranchTree {
        crate::branch::BranchTree::from_parts(&self.transcript, &self.branches)
    }

    /// Replaces the transcript and off-path nodes from a tree, marking the
    /// session dirty. The active branch becomes the live transcript.
    pub fn set_tree(&mut self, tree: &crate::branch::BranchTree) {
        let (path, off) = tree.into_parts();
        self.transcript = path;
        self.branches = off;
        self.dirty = true;
    }

    /// Drops every off-path branch.
    ///
    /// Branch parents are transcript indices, so a wholesale transcript
    /// rewrite (compaction) invalidates them; the branches are dropped rather
    /// than left pointing at the wrong messages.
    pub fn clear_branches(&mut self) {
        self.branches.clear();
    }

    /// Total transcript size in bytes (listing shows this instead of tokens).
    #[must_use]
    pub fn transcript_bytes(&self) -> u64 {
        self.transcript.iter().map(|m| m.text.len() as u64).sum()
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Lightweight listing record for one saved session.
#[derive(Debug, Clone)]
pub struct SessionEntry {
    /// Session id (memorable name, or a legacy 40-hex id).
    pub id: String,
    /// Session title (possibly clipped by the caller for display).
    pub title: String,
    /// Creation time in unix seconds.
    pub created_at: u64,
    /// Last save time in unix seconds.
    pub last_used: u64,
    /// Size of the session file in bytes.
    pub file_size: u64,
    /// User-assigned tag; empty when unset.
    pub tag: String,
    /// Clipped last user prompt; empty for pre-meta files.
    pub last_prompt: String,
    /// Size of the engine KV payload sidecar in bytes; 0 when stripped or
    /// never saved (the C lists `payload_bytes == 0` as "stripped").
    pub payload_bytes: u64,
    /// Full path of the session file.
    pub path: PathBuf,
}

/// Identifies one persisted KV cache: its signature plus whatever scope the
/// signature alone cannot supply.
///
/// The project variant carries its own directory because two projects can hold
/// the same `fp2` only by coincidence of identical context — they must still
/// not share a file, and GC sweeps per project directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvKey {
    /// Tier 1: model + system prompt. `sysprompt-<fp>.kv_raw`, shared across projects.
    System { fp: String },
    /// Tier 2: project-stable context. `<project-key>/project-<fp>.kv_raw`.
    Project { dir: PathBuf, fp: String },
    /// A saved conversation's KV payload. `<id>.kv_raw`.
    ///
    /// Path and signature differ here: the file is named after the session id
    /// (stable across resaves), but it is only trusted when its stored
    /// signature equals `fp` — the [`payload_fingerprint`] over model, system
    /// prompt, and rendered transcript. Keying on the id alone would make a
    /// payload captured under a different model or system prompt a hit.
    Session { id: String, fp: String },
}

impl KvKey {
    /// The signature this key is stored under — the value a loaded file must
    /// carry for the load to be trusted.
    #[must_use]
    pub fn signature(&self) -> &str {
        match self {
            Self::System { fp } | Self::Project { fp, .. } | Self::Session { fp, .. } => fp,
        }
    }
}

/// Directory-backed store of saved sessions.
#[derive(Debug, Clone)]
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    /// Opens (creating if needed) a session store at `dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)
            .map_err(|_| SessionError::new(format!("failed to create {}", dir.display())))?;
        Ok(Self { dir })
    }

    /// Default cache directory: `$HOME/.plank/kvcache` (`.` if HOME unset).
    #[must_use]
    pub fn default_dir() -> PathBuf {
        let home = std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .map_or_else(|| PathBuf::from("."), PathBuf::from);
        home.join(".plank").join("kvcache")
    }

    /// Directory this store persists sessions in.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_for_id(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}{FILE_EXT}"))
    }

    /// Path of the engine KV payload sidecar for a session id.
    #[must_use]
    pub fn payload_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}{PAYLOAD_EXT}"))
    }

    /// Size in bytes of a session's KV payload sidecar; 0 when absent.
    #[must_use]
    pub fn payload_bytes(&self, id: &str) -> u64 {
        fs::metadata(self.payload_path(id)).map_or(0, |m| m.len())
    }

    /// Every KV blob in the cache, with its metadata.
    ///
    /// Scans the cache root and one level of per-project subdirectories for
    /// `*.kv_raw` bodies. A body with no readable sidecar yields
    /// [`crate::kvmeta::KvMeta::synthesized`] from its size and mtime, so an
    /// externally dropped or partially migrated file still shows in `/kvcache`
    /// and can still be deleted from there.
    #[must_use]
    pub fn kv_nodes(&self) -> Vec<crate::kvmeta::KvMeta> {
        self.kv_blob_nodes()
            .into_iter()
            .map(|(_, meta)| meta)
            .collect()
    }

    /// Every KV blob body in the cache paired with its metadata, in one walk.
    ///
    /// The single scan every caller shares, so index *i* of the returned vector
    /// always names path *i*. That pairing is the only sound way to identify a
    /// blob: a fingerprint does not identify a file. Two bodies may carry the
    /// same fingerprint (a root `sysprompt-X.kv_raw` beside a
    /// `<proj>/project-X.kv_raw`), and a [`KvKey::Session`] sidecar records the
    /// *payload* fingerprint, which never equals the `<id>` in its file name.
    /// Round-tripping a node through its fingerprint to find its path therefore
    /// either fails or, worse, finds a different body.
    #[must_use]
    pub fn kv_blob_nodes(&self) -> Vec<(PathBuf, crate::kvmeta::KvMeta)> {
        self.kv_blob_paths()
            .into_iter()
            .map(|(path, fp)| {
                let meta = Self::kv_node_at(&path, &fp);
                (path, meta)
            })
            .collect()
    }

    /// The node for one blob body: its sidecar when readable, else one
    /// synthesized from the file name, size and mtime.
    fn kv_node_at(path: &Path, fp: &str) -> crate::kvmeta::KvMeta {
        crate::kvmeta::load(path).unwrap_or_else(|| {
            let md = fs::metadata(path).ok();
            let bytes = md.as_ref().map_or(0, std::fs::Metadata::len);
            let mtime = md
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            let name = path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or_default();
            let role = if name.starts_with(SYSPROMPT_PREFIX) {
                crate::kvmeta::KvRole::System
            } else if name.starts_with(&format!("{PROJECT_STEM}-")) {
                crate::kvmeta::KvRole::Project
            } else {
                crate::kvmeta::KvRole::Session
            };
            crate::kvmeta::KvMeta::synthesized(role, fp, bytes, mtime)
        })
    }

    /// Every KV blob body in the cache, as `(path, fingerprint)`.
    ///
    /// The scan behind [`kv_nodes`](Self::kv_nodes), kept separate because a
    /// sweep needs the path it must unlink alongside the fingerprint the plan
    /// names. Covers the cache root and one level of per-project
    /// subdirectories — never deeper.
    ///
    /// **Sorted by path.** `read_dir` yields filesystem-hash order, which made
    /// `/kvcache`'s row order and the phase-2 budget tie-break reproducible only
    /// by accident; sorting makes both deterministic. Callers still must not
    /// treat an index as a durable handle across two scans — a concurrent unlink
    /// shifts every later position, which is why a mutation carries the expected
    /// fingerprint and re-checks it.
    #[must_use]
    pub fn kv_blob_paths(&self) -> Vec<(PathBuf, String)> {
        fn scan(dir: &Path, out: &mut Vec<(PathBuf, String)>, recurse: bool) {
            let Ok(entries) = fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if entry.file_type().is_ok_and(|t| t.is_dir()) {
                    if recurse {
                        scan(&path, out, false);
                    }
                    continue;
                }
                let name = entry.file_name();
                let Some(name) = name.to_str() else { continue };
                let Some(stem) = name.strip_suffix(PAYLOAD_EXT) else {
                    continue;
                };
                let fp = stem
                    .strip_prefix(SYSPROMPT_PREFIX)
                    .or_else(|| stem.strip_prefix(&format!("{PROJECT_STEM}-")))
                    .unwrap_or(stem);
                out.push((path, fp.to_owned()));
            }
        }
        let mut out = Vec::new();
        scan(&self.dir, &mut out, true);
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Project-scope subdirectory for Tier 2+ checkpoints: `<dir>/<project-key>/`
    /// (issue #60). Callers create it lazily before writing a project checkpoint.
    #[must_use]
    pub fn project_dir(&self, project: &Path) -> PathBuf {
        self.dir.join(project_key(project))
    }

    /// Path of the Tier 2 project-stable KV checkpoint for `project` at the
    /// chained fingerprint `fp2`: `<dir>/<project-key>/project-<fp2>.kv_raw`. Shared
    /// by all sessions of the project and read by every new session (issue #60).
    #[must_use]
    pub fn project_checkpoint_path(&self, project: &Path, fp2: &str) -> PathBuf {
        self.project_dir(project).join(project_checkpoint_name(fp2))
    }

    /// Deletes every blob [`crate::kvgc::plan_sweep`] condemns, returning the
    /// bytes reclaimed.
    ///
    /// Best-effort: a failed unlink is skipped and its bytes are not counted,
    /// and the next launch retries. Both the body and its sidecar go.
    ///
    /// Each verdict is re-checked against the disk immediately before the
    /// unlink, so a body a sibling process rewrote while this sweep was
    /// planning is left alone rather than deleted under a sidecar that has
    /// since been replaced.
    ///
    /// `active` is the set of fingerprints this launch is using, across *every*
    /// engine it holds: a provider main agent beside a `provider: local`
    /// sub-agent has two Tier 1 fingerprints, and passing only one of them is
    /// what used to delete the other's checkpoint on every launch.
    #[must_use]
    pub fn sweep(&self, active: &[&str], policy: &crate::kvgc::SweepPolicy, now: u64) -> u64 {
        // One paired walk: node `i` is the metadata of path `i`, so a verdict
        // names exactly one file. Matching on fingerprints instead would delete
        // a kept body that merely shares a doomed one's fingerprint — a root
        // `sysprompt-X` and a `<proj>/project-X`, say.
        let scan = self.kv_blob_nodes();
        let nodes: Vec<crate::kvmeta::KvMeta> = scan.iter().map(|(_, m)| m.clone()).collect();
        let plan = crate::kvgc::plan_sweep(&nodes, active, policy, now);
        let mut freed = 0u64;
        for i in plan.doomed {
            let Some((path, seen)) = scan.get(i) else {
                continue;
            };
            // A sibling process persisting a multi-hundred-megabyte body spans
            // the whole window between the scan above and this unlink, so the
            // file under an expired sidecar may already have been replaced by a
            // fresh one. Re-read the node and skip anything that moved: the
            // verdict was computed about bytes that are no longer there.
            let fp = seen.fingerprint.as_str();
            let fresh = Self::kv_node_at(path, fp);
            if fresh.last_used != seen.last_used
                || fresh.bytes != seen.bytes
                || fresh.pinned != seen.pinned
            {
                continue;
            }
            let size = fs::metadata(path).map_or(0, |m| m.len());
            if fs::remove_file(path).is_ok() {
                freed = freed.saturating_add(size);
                // The sidecar must go with the body, or the next scan reports a
                // phantom node.
                let _ = fs::remove_file(crate::kvmeta::sidecar_path(path));
            }
        }
        freed
    }

    /// The system-prompt text the last Tier 1 checkpoint was built from, or
    /// `None` when no sidecar has been written yet (first run, or a cleared
    /// cache). Used only to explain a Tier 1 miss — never to validate a cache.
    #[must_use]
    pub fn system_prompt_note(&self) -> Option<String> {
        fs::read_to_string(self.dir.join(SYSPROMPT_NOTE_NAME)).ok()
    }

    /// Records `system` as the prompt text behind the current Tier 1
    /// checkpoint.
    ///
    /// # Errors
    /// Returns the underlying [`io::Error`] when the write fails. Best-effort:
    /// losing the sidecar costs only the explanation on the next miss.
    pub fn store_system_prompt_note(&self, system: &str) -> io::Result<()> {
        fs::write(self.dir.join(SYSPROMPT_NOTE_NAME), system)
    }

    /// Filesystem location backing a [`KvKey`].
    ///
    /// Public so a diagnostic can name the exact file it looked for: "no
    /// checkpoint" is only actionable if you can go and see whether it is
    /// there.
    #[must_use]
    pub fn kv_path(&self, key: &KvKey) -> PathBuf {
        match key {
            KvKey::System { fp } => self.dir.join(sysprompt_checkpoint_name(fp)),
            KvKey::Project { dir, fp } => self.project_checkpoint_path(dir, fp),
            KvKey::Session { id, .. } => self.payload_path(id),
        }
    }

    /// Loads the cache stored under `key`, or `None` on any miss — absent,
    /// stale, corrupt, or written by an older format version.
    ///
    /// A hit bumps the sidecar's `hits` and `last_used`, which is what the TTL
    /// sweep and `/kvcache` read. That refresh is best-effort and strictly
    /// downstream of the load decision: the signature inside the body is still
    /// the only thing that decides whether these bytes may be used.
    #[must_use]
    pub fn kv_load(&self, key: &KvKey) -> Option<crate::kvcache::KVCache> {
        let path = self.kv_path(key);
        let cache = crate::kvcache::KVCache::from_file(&path, key.signature())?;
        if let Some(mut meta) = crate::kvmeta::load(&path) {
            meta.hits = meta.hits.saturating_add(1);
            meta.last_used = crate::kvmeta::now_secs();
            let _ = crate::kvmeta::store(&path, &meta);
        }
        Some(cache)
    }

    /// Persists `cache` under `key` with no lineage or label recorded.
    ///
    /// Prefer [`kv_store_labeled`](Self::kv_store_labeled): a blob stored this
    /// way shows in `/kvcache` as an unattributed root. This form exists for
    /// callers that genuinely have nothing to say about the blob.
    ///
    /// # Errors
    /// Returns the underlying [`io::Error`] when the write fails. Callers treat
    /// a failure as best-effort: the live session is correct either way and
    /// only the next launch pays.
    pub fn kv_store(&self, key: &KvKey, cache: &crate::kvcache::KVCache) -> io::Result<()> {
        self.kv_store_labeled(key, cache, None, "", &crate::kvmeta::KvLabel::Unknown)
    }

    /// Persists `cache` under `key` and records its lineage and display label.
    ///
    /// `parent` is the fingerprint of the blob this one extends — `None` only
    /// for a Tier 1 system checkpoint. An existing sidecar's `created` and
    /// `pinned` survive a rewrite: re-persisting the same fingerprint is a
    /// refresh, not a new blob, and silently unpinning one would be a nasty
    /// surprise.
    ///
    /// # Errors
    /// Returns the underlying [`io::Error`] when the body write fails. A failed
    /// *sidecar* write is swallowed: metadata is advisory and must never turn a
    /// good store into an error.
    pub fn kv_store_labeled(
        &self,
        key: &KvKey,
        cache: &crate::kvcache::KVCache,
        parent: Option<&str>,
        model: &str,
        label: &crate::kvmeta::KvLabel,
    ) -> io::Result<()> {
        let path = self.kv_path(key);
        cache.persist(&path, key.signature())?;
        let now = crate::kvmeta::now_secs();
        let prior = crate::kvmeta::load(&path);
        let meta = crate::kvmeta::KvMeta {
            version: crate::kvmeta::META_VERSION,
            role: kv_role(key),
            fingerprint: key.signature().to_owned(),
            parent: parent.map(ToOwned::to_owned),
            model: model.to_owned(),
            created: prior.as_ref().map_or(now, |p| p.created),
            last_used: now,
            hits: prior.as_ref().map_or(0, |p| p.hits),
            bytes: fs::metadata(&path).map_or(0, |m| m.len()),
            pinned: prior.as_ref().is_some_and(|p| p.pinned),
            label: label.clone(),
        };
        let _ = crate::kvmeta::store(&path, &meta);
        Ok(())
    }

    /// Deletes every pre-`.kv_raw` KV body once, returning the bytes reclaimed.
    ///
    /// Returns `None` when the marker file shows migration already ran, so the
    /// caller knows to stay silent. The old format carried no lineage and no
    /// usage history, and synthesizing either would produce a `/kvcache` tree
    /// that looks authoritative while being invented — so the blobs go and the
    /// tiers rebuild on demand.
    ///
    /// **Transcripts are never touched.** Only `<id>.payload`,
    /// `sysprompt-*.kv`, the legacy bare `sysprompt.kv`, `<proj>/project-*.kv`,
    /// and the diagnostic prompt note are removed; every `<id>.kv` survives, so
    /// resuming a session works and pays a single re-prefill.
    ///
    /// Best-effort throughout: an unlink that fails is skipped and its bytes
    /// are not counted, and the marker is still written — a second sweep would
    /// find the same undeletable file.
    ///
    /// The returned tally means "cache bytes reclaimed": the diagnostic
    /// prompt note is deleted too, but its bytes are deliberately excluded
    /// from the count.
    #[must_use]
    pub fn migrate_legacy_blobs(&self) -> Option<u64> {
        let marker = self.dir.join(MIGRATION_MARKER);
        if marker.exists() {
            return None;
        }
        let mut reclaimed = 0u64;
        let legacy_sysprompt = format!("{SYSPROMPT_STEM}{FILE_EXT}");
        let mut sweep = |dir: &Path, stems: &[&str]| {
            let Ok(entries) = fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                let Some(name) = name.to_str() else { continue };
                if !entry.file_type().is_ok_and(|t| t.is_file()) {
                    continue;
                }
                if name == SYSPROMPT_NOTE_NAME {
                    // Deleted but deliberately not counted: it's a tiny
                    // diagnostic sidecar, not cache, and the tally means
                    // cache bytes reclaimed.
                    let _ = fs::remove_file(entry.path());
                    continue;
                }
                let doomed = name == legacy_sysprompt
                    || name.ends_with(LEGACY_PAYLOAD_EXT)
                    || (name.ends_with(FILE_EXT) && stems.iter().any(|s| name.starts_with(s)));
                if !doomed {
                    continue;
                }
                let size = entry.metadata().map_or(0, |m| m.len());
                if fs::remove_file(entry.path()).is_ok() {
                    reclaimed = reclaimed.saturating_add(size);
                }
            }
        };
        sweep(&self.dir, &[SYSPROMPT_PREFIX]);
        // Project checkpoints live one level down, one directory per project.
        if let Ok(entries) = fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                if entry.file_type().is_ok_and(|t| t.is_dir()) {
                    sweep(&entry.path(), &[PROJECT_STEM]);
                }
            }
        }
        let _ = fs::write(&marker, b"2\n");
        Some(reclaimed)
    }

    /// Strips the engine KV payload from the session matching the hex
    /// prefix, preserving its transcript (`/strip`). Returns the full id and
    /// whether a payload sidecar actually existed; stripping an already
    /// stripped session succeeds, like the C's rewrite-with-zero-payload.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid, ambiguous, or unmatched prefix, or
    /// if the payload file cannot be removed.
    pub fn strip(&self, prefix: impl AsRef<str>) -> Result<(String, bool)> {
        let (id, _path) = self.find(prefix.as_ref())?;
        let payload = self.payload_path(&id);
        let had_payload = payload.exists();
        if had_payload {
            fs::remove_file(&payload)?;
            let _ = fs::remove_file(crate::kvmeta::sidecar_path(&payload));
        }
        Ok((id, had_payload))
    }

    /// Saves the session, assigning title, creation time, and id if missing.
    ///
    /// The file is written atomically (temp file + rename). On success the
    /// dirty flag clears and the stable 40-hex id is returned.
    ///
    /// # Errors
    ///
    /// Returns "nothing to save" for a transcript with no user turn, or an
    /// I/O error if the file cannot be written.
    pub fn save(&self, session: &mut Session) -> Result<String> {
        if !session.transcript.iter().any(|m| m.role == Role::User) {
            return Err(SessionError::new("nothing to save"));
        }
        fs::create_dir_all(&self.dir)
            .map_err(|_| SessionError::new(format!("failed to create {}", self.dir.display())))?;

        if session.title.is_empty() {
            session.title = title_from_transcript(&session.transcript, 0);
        }
        let now = unix_now();
        if session.created_at == 0 {
            session.created_at = now;
        }
        // First save: mint a memorable `adjective-celebrity` name (e.g.
        // `deadly-einstein`). On the rare filename collision, append a short
        // guid. Re-saving an already-named session keeps its name (overwrite).
        if session.id.is_empty() {
            let base = crate::names::session_slug();
            let mut id = base.clone();
            while self.path_for_id(&id).exists() {
                id = format!("{base}-{}", crate::names::guid8());
            }
            session.id = id;
        }
        let id = session.id.clone();

        let mut body = Vec::new();
        let _ = writeln!(body, "{MAGIC}");
        let _ = writeln!(body, "created {}", session.created_at);
        let _ = writeln!(body, "used {now}");
        let _ = writeln!(body, "title {}", session.title.len());
        body.extend_from_slice(session.title.as_bytes());
        body.push(b'\n');
        for m in &session.transcript {
            let _ = writeln!(body, "msg {} {}{}", m.role.tag(), m.text.len(), stamp(m.at));
            body.extend_from_slice(m.text.as_bytes());
            body.push(b'\n');
        }
        // Session-tree records (issue #65): the active branch is already
        // written above as plain `msg` records, so only the off-path nodes
        // need describing. Omitted entirely for a linear session, keeping
        // pre-branching files byte-identical.
        let branches = crate::branch::canonicalize(&session.transcript, &session.branches);
        for n in &branches {
            let parent = n.parent.map_or_else(|| "-".to_owned(), |p| p.to_string());
            let _ = writeln!(
                body,
                "node {} {parent} {} {}{}",
                n.id,
                n.message.role.tag(),
                n.message.text.len(),
                stamp(n.message.at)
            );
            body.extend_from_slice(n.message.text.as_bytes());
            body.push(b'\n');
        }
        session.branches = branches;
        // Task list records (issue #35): omitted entirely when the list is
        // empty so pre-feature files stay byte-identical.
        if !session.tasks.is_empty() {
            let _ = writeln!(body, "tasks {}", session.tasks.next_id());
            for t in session.tasks.tasks() {
                let af = t.active_form.as_deref().unwrap_or("");
                let _ = writeln!(
                    body,
                    "task {} {} {} {}",
                    t.id,
                    t.status.as_str(),
                    t.subject.len(),
                    af.len()
                );
                body.extend_from_slice(t.subject.as_bytes());
                body.push(b'\n');
                body.extend_from_slice(af.as_bytes());
                body.push(b'\n');
            }
        }
        // Project directory (issue: `/insights`): omitted when unset so a
        // session saved by an older build round-trips byte-identically.
        if !session.cwd.is_empty() {
            let _ = writeln!(body, "cwd {}", session.cwd.len());
            body.extend_from_slice(session.cwd.as_bytes());
            body.push(b'\n');
        }
        let last_prompt = last_prompt_of(&session.transcript);
        let _ = writeln!(body, "meta {} {}", session.tag.len(), last_prompt.len());
        body.extend_from_slice(session.tag.as_bytes());
        body.push(b'\n');
        body.extend_from_slice(last_prompt.as_bytes());
        body.push(b'\n');

        let path = self.path_for_id(&id);
        let tmp = self
            .dir
            .join(format!("{id}{FILE_EXT}.tmp.{}", std::process::id()));
        fs::write(&tmp, &body)?;
        if let Err(e) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        session.id.clone_from(&id);
        session.dirty = false;
        Ok(id)
    }

    /// Loads the session whose id matches the given hex prefix.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or ambiguous prefix, no match, or a
    /// corrupt session file.
    pub fn load(&self, prefix: impl AsRef<str>) -> Result<Session> {
        let (id, path) = self.find(prefix.as_ref())?;
        let mut session = read_session_file(&path)?;
        // The filename is the identity; the memorable name (or a legacy sha)
        // is authoritative, with no separate content cross-check.
        session.id = id;
        Ok(session)
    }

    /// Deletes the session matching the hex prefix, returning its full id.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid, ambiguous, or unmatched prefix, or
    /// if the file cannot be removed.
    pub fn delete(&self, prefix: impl AsRef<str>) -> Result<String> {
        let (id, path) = self.find(prefix.as_ref())?;
        fs::remove_file(&path)?;
        // The payload sidecar is a cache keyed to the transcript; it goes too.
        let payload = self.payload_path(&id);
        let _ = fs::remove_file(&payload);
        let _ = fs::remove_file(crate::kvmeta::sidecar_path(&payload));
        Ok(id)
    }

    /// Resolves a hex prefix to exactly one saved session id and path.
    ///
    /// # Errors
    ///
    /// Uses the C agent's wording: "invalid session SHA prefix",
    /// "no saved session matches ...", "session prefix ... is ambiguous".
    pub fn find(&self, prefix: &str) -> Result<(String, PathBuf)> {
        if !is_valid_id_prefix(prefix) {
            return Err(SessionError::new("invalid session name"));
        }
        let want = prefix.to_ascii_lowercase();
        let mut matched: Option<(String, PathBuf)> = None;
        for (id, path) in self.session_files()? {
            if !id.starts_with(&want) {
                continue;
            }
            if matched.is_some() {
                return Err(SessionError::new(format!(
                    "session prefix {prefix} is ambiguous"
                )));
            }
            matched = Some((id, path));
        }
        matched.ok_or_else(|| SessionError::new(format!("no saved session matches {prefix}")))
    }

    /// Lists saved sessions, most recently used first.
    ///
    /// Ties break on ascending id, matching the C agent. Unreadable files
    /// are listed with the title "(unreadable session)".
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be read.
    pub fn list(&self) -> Result<Vec<SessionEntry>> {
        let mut entries = Vec::new();
        for (id, path) in self.session_files()? {
            let file_size = fs::metadata(&path).map_or(0, |m| m.len());
            let payload_bytes = self.payload_bytes(&id);
            // Listing metadata comes from bounded head + tail reads; the
            // transcript itself is never parsed here.
            let entry = match read_head_meta(&path) {
                Some((created_at, last_used, title)) => {
                    let (tag, last_prompt) = read_meta_tail(&path).unwrap_or_default();
                    SessionEntry {
                        id,
                        title,
                        created_at,
                        last_used,
                        file_size,
                        tag,
                        last_prompt,
                        payload_bytes,
                        path,
                    }
                }
                None => SessionEntry {
                    id,
                    title: "(unreadable session)".to_owned(),
                    created_at: 0,
                    last_used: 0,
                    file_size,
                    tag: String::new(),
                    last_prompt: String::new(),
                    payload_bytes,
                    path,
                },
            };
            entries.push(entry);
        }
        entries.sort_by(|a, b| {
            let ta = if a.last_used != 0 {
                a.last_used
            } else {
                a.created_at
            };
            let tb = if b.last_used != 0 {
                b.last_used
            } else {
                b.created_at
            };
            tb.cmp(&ta).then_with(|| a.id.cmp(&b.id))
        });
        Ok(entries)
    }

    /// Session ids (sorted by recent use) whose id starts with `prefix`.
    ///
    /// Backs tab completion for `/switch`; an empty prefix matches all.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be read.
    pub fn complete(&self, prefix: &str) -> Result<Vec<String>> {
        if !prefix.is_empty() && !is_valid_id_prefix(prefix) {
            return Ok(Vec::new());
        }
        let want = prefix.to_ascii_lowercase();
        Ok(self
            .list()?
            .into_iter()
            .filter(|e| e.id.starts_with(&want))
            .map(|e| e.id)
            .collect())
    }

    /// Enumerates `<name>.kv` session files: memorable `adjective-celebrity`
    /// names, plus legacy `<40-hex>.kv` ids. Temp files (`*.kv.tmp.*`, which
    /// don't end in `.kv`) are skipped.
    fn session_files(&self) -> Result<Vec<(String, PathBuf)>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(stem) = name.strip_suffix(FILE_EXT) else {
                continue;
            };
            // The `sysprompt` guard is belt-and-braces against a legacy file:
            // no checkpoint ends in `.kv` any more (bodies are `.kv_raw`), and
            // `migrate_legacy_blobs` unlinks the old ones on first launch, so
            // this can only fire on a blob that survived the migration.
            if stem.starts_with(SYSPROMPT_STEM) || !is_valid_id_prefix(stem) {
                continue;
            }
            out.push((stem.to_ascii_lowercase(), entry.path()));
        }
        Ok(out)
    }
}

/// File-stem prefix of the system-prompt KV checkpoints, which share the
/// cache dir with session files but are not sessions. Names the legacy bare
/// `sysprompt.kv` that [`SessionStore::migrate_legacy_blobs`] removes.
const SYSPROMPT_STEM: &str = "sysprompt";

/// File-name prefix of a content-keyed system-prompt checkpoint,
/// `sysprompt-<fp1>.kv_raw`.
const SYSPROMPT_PREFIX: &str = "sysprompt-";

/// Sidecar holding the system-prompt text that the most recent Tier 1
/// checkpoint was built from. Deliberately *not* keyed by `fp1` — a changed
/// prompt yields a different key, so the point of this file is to be findable
/// after the fingerprint has already moved. Its name ends in `.prompt`, not
/// `.kv_raw`, so [`SessionStore::sweep`] never sees it as a blob.
const SYSPROMPT_NOTE_NAME: &str = "sysprompt-last.prompt";

/// File-stem prefix of the Tier 2 project-stable KV checkpoints (issue #60):
/// `project-<fp2>.kv_raw`, living under the per-project subdirectory.
const PROJECT_STEM: &str = "project";

/// Project-scope subdirectory name under `kvcache/`: 12 hex of `sha1(project
/// path)`. Tier 2+ checkpoints live under here so they are *project-scoped*,
/// while Tier 1 (`sysprompt-*.kv_raw`) stays model-global at the cache root and is
/// shared across every project (issue #60).
#[must_use]
pub fn project_key(project_dir: &Path) -> String {
    let hash = sha1_hex(project_dir.to_string_lossy().as_bytes());
    hash[..12].to_string()
}

/// Chains a child tier's fingerprint onto its parent's: `sha1(parent ‖ NUL ‖
/// material)`.
///
/// This generalizes the system-prompt checkpoint key (`model ‖ NUL ‖ system`,
/// [`crate::ds4engine`]) to an N-tier volatility hierarchy: each tier embeds its
/// parent's fingerprint, so validation walks the chain top-down and the first
/// mismatch at tier N rebuilds N and everything below it while tiers `1..N`
/// stay reusable verbatim. Pure and order-deterministic (issue #60).
///
/// Example chain: `fp1 = sha(model ‖ system)` (Tier 1),
/// `fp2 = tier_fingerprint(fp1, stable-context-hash ‖ local-tool-defs)` (Tier 2),
/// `fp4 = tier_fingerprint(fp2, transcript)` (Tier 4). Tier 3 is prefill-only
/// and never keyed.
#[must_use]
pub fn tier_fingerprint(parent_fp: &str, material: &[u8]) -> String {
    let mut data = parent_fp.as_bytes().to_vec();
    data.push(0);
    data.extend_from_slice(material);
    sha1_hex(&data)
}

/// File name of a Tier 2 project-stable checkpoint keyed by its chained
/// fingerprint `fp2`: `project-<fp2>.kv_raw`. One file per (project, AGENTS.md
/// revision); shared by every session of the project (issue #60).
#[must_use]
pub fn project_checkpoint_name(fp2: &str) -> String {
    format!("{PROJECT_STEM}-{fp2}{PAYLOAD_EXT}")
}

/// The metadata role a [`KvKey`] variant corresponds to.
#[must_use]
pub fn kv_role(key: &KvKey) -> crate::kvmeta::KvRole {
    match key {
        KvKey::System { .. } => crate::kvmeta::KvRole::System,
        KvKey::Project { .. } => crate::kvmeta::KvRole::Project,
        KvKey::Session { .. } => crate::kvmeta::KvRole::Session,
    }
}

/// File name of the Tier 1 system-prompt checkpoint keyed by its fingerprint
/// `fp1`: `sysprompt-<fp1>.kv_raw`. Model-global, at the cache root.
#[must_use]
pub fn sysprompt_checkpoint_name(fp1: &str) -> String {
    format!("{SYSPROMPT_PREFIX}{fp1}{PAYLOAD_EXT}")
}

/// Whether `s` is a valid session id (or lookup prefix): non-empty, at most 80
/// chars, ASCII alphanumeric or `-`. Covers both memorable names and legacy
/// 40-hex ids.
#[must_use]
pub fn is_valid_id_prefix(s: &str) -> bool {
    !s.is_empty() && s.len() <= 80 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

/// Display handle for a session id: the full memorable `adjective-celebrity`
/// name, or the first 8 chars of a legacy 40-hex id (which resume still
/// accepts as a prefix).
#[must_use]
pub fn display_id(id: &str) -> &str {
    if id.len() == 40 && id.bytes().all(|b| b.is_ascii_hexdigit()) {
        &id[..8]
    } else {
        id
    }
}

/// Renders the session list the way the C agent prints `/sessions`.
///
/// Each entry shows the 8-char id, title, then an indented age/size line;
/// a trailing help line explains `/switch` and `/del`. `now` is unix
/// seconds; `color` enables the C agent's ANSI styling.
#[must_use]
pub fn render_session_list(entries: &[SessionEntry], now: u64, color: bool) -> String {
    if entries.is_empty() {
        return "no saved sessions\n".to_owned();
    }
    let (sha_on, title_on, help_on, dim, reset) = if color {
        (
            "\x1b[1;96m",
            "\x1b[1;97m",
            "\x1b[97m",
            "\x1b[90m",
            "\x1b[0m",
        )
    } else {
        ("", "", "", "", "")
    };
    let mut out = String::new();
    for e in entries {
        let when = if e.last_used != 0 {
            e.last_used
        } else {
            e.created_at
        };
        let age = format_age(when, now);
        let short = display_id(&e.id);
        let tag = if e.tag.is_empty() {
            String::new()
        } else {
            format!(" {dim}[{}{reset}{dim}]{reset}", e.tag)
        };
        let _ = writeln!(
            out,
            "{sha_on}{short}{reset} {dim}>{reset} {title_on}{}{reset}{tag}",
            e.title
        );
        if !e.last_prompt.is_empty() {
            let _ = writeln!(out, "         {dim}> last: {}{reset}", e.last_prompt);
        }
        // Payload presence mirrors the C listing: a zero-byte payload reads
        // as ", stripped"; otherwise the sidecar's size is shown.
        let payload = if e.payload_bytes == 0 {
            ", stripped".to_owned()
        } else {
            format!(", KV {:.2} MB", to_mb(e.payload_bytes))
        };
        let _ = writeln!(
            out,
            "         {dim}> {age}, {:.2} MB{payload}{reset}\n",
            to_mb(e.file_size)
        );
    }
    let _ = writeln!(
        out,
        "{help_on}Use /switch <id> to select a session, /del <id> to remove, /strip <id> to strip KV cache.{reset}"
    );
    out
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Formats an age like the C agent: "42s ago", "3m ago", "5h ago", "2d ago".
#[must_use]
pub fn format_age(when: u64, now: u64) -> String {
    let age = if when != 0 && now > when {
        now - when
    } else {
        0
    };
    if age < 60 {
        format!("{age}s ago")
    } else if age < 3600 {
        format!("{}m ago", age / 60)
    } else if age < 86400 {
        format!("{}h ago", age / 3600)
    } else {
        format!("{}d ago", age / 86400)
    }
}

/// Derives a session title from the first user message of a transcript.
///
/// Whitespace is collapsed, the result is clipped to `max_bytes` with a
/// `...` suffix (0 means unlimited), and the placeholders
/// "(no user prompt)" / "(empty user prompt)" match the C agent.
#[must_use]
pub fn title_from_transcript(transcript: &[Message], max_bytes: usize) -> String {
    let Some(first) = transcript.iter().find(|m| m.role == Role::User) else {
        return "(no user prompt)".to_owned();
    };
    title_from_span(&first.text, max_bytes, "(empty user prompt)")
}

/// Normalizes arbitrary prompt text into a single-line title.
fn title_from_span(text: &str, max_bytes: usize, empty_title: &str) -> String {
    let limited = max_bytes != 0;
    let max_bytes = if limited { max_bytes.max(4) } else { 0 };
    let mut out = String::new();
    let mut pending_space = false;
    let mut truncated = false;
    for ch in text.trim().chars() {
        if ch.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space && (!limited || out.len() + 4 < max_bytes) {
            out.push(' ');
            pending_space = false;
        }
        if limited && out.len() + 4 > max_bytes {
            truncated = true;
            break;
        }
        out.push(ch);
    }
    if truncated {
        out.push_str("...");
    }
    if out.is_empty() {
        empty_title.to_owned()
    } else {
        out
    }
}

/// Clips an already-normalized title to a display budget with `...`.
#[must_use]
pub fn title_clip(title: &str, max_bytes: usize) -> String {
    if max_bytes == 0 || title.len() <= max_bytes {
        return title.to_owned();
    }
    let budget = max_bytes.max(4) - 3;
    let mut cut = budget.min(title.len());
    while cut > 0 && !title.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}...", &title[..cut])
}

/// Computes the stable session id: SHA-1 of `title || created_at` (LE u64).
#[must_use]
pub fn session_identity_sha(title: &str, created_at: u64) -> String {
    let mut bytes = Vec::with_capacity(title.len() + 8);
    bytes.extend_from_slice(title.as_bytes());
    bytes.extend_from_slice(&created_at.to_le_bytes());
    sha1_hex(&bytes)
}

/// Fingerprint tying an engine KV payload to the exact model, system prompt,
/// and rendered transcript it was snapshotted from.
///
/// This is the repo's KV discipline rule applied to per-session payloads:
/// a payload is only a cache of prefilling `transcript_render`, so any
/// difference in model, system prompt, or transcript text (including a
/// resave after more turns) makes it stale, and a stale payload is rebuilt
/// by prefill, never trusted. NUL separators keep the fields unambiguous.
///
/// `think` and `trusted_len` are here for the reason the text fields cannot
/// cover: both change the *tokens* the prompt prefills to while leaving every
/// byte of `system` and `transcript_render` identical. `ThinkMode::Max`
/// prepends the reasoning-effort preamble, and `trusted_len` decides how much
/// of the prompt is tokenized as rendered chat (native `｜DSML｜` versus
/// spelled-out BPE pieces). Without them a payload written before either
/// changed — including one written by a build with a different tokenization
/// split — passes the staleness gate and is restored over a KV it does not
/// match. That is the one failure mode this fingerprint exists to prevent.
#[must_use]
pub fn payload_fingerprint(
    model: &str,
    system: &str,
    transcript_render: &str,
    think: crate::engine::ThinkMode,
    trusted_len: usize,
) -> String {
    let mut data = Vec::with_capacity(model.len() + system.len() + transcript_render.len() + 4);
    data.extend_from_slice(model.as_bytes());
    data.push(0);
    data.extend_from_slice(think.name().as_bytes());
    data.push(0);
    data.extend_from_slice(trusted_len.to_string().as_bytes());
    data.push(0);
    data.extend_from_slice(system.as_bytes());
    data.push(0);
    data.extend_from_slice(transcript_render.as_bytes());
    sha1_hex(&data)
}

/// Selects the transcript index at which recent-history replay should begin,
/// showing the last `user_turns` human turns. Returns `(start, tool_only)`
/// where `tool_only` means no human turn was found and the window falls back to
/// raw user (tool-result) messages. `None` when there is nothing to replay.
#[must_use]
pub fn history_window(transcript: &[Message], user_turns: usize) -> Option<(usize, bool)> {
    if user_turns == 0 {
        return None;
    }
    let user_turns = user_turns.min(HISTORY_MAX_TURNS);

    let human_idx: Vec<usize> = transcript
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User && !m.is_tool_user())
        .map(|(i, _)| i)
        .collect();
    let all_user_idx: Vec<usize> = transcript
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User)
        .map(|(i, _)| i)
        .collect();

    if let Some(&i) = human_idx
        .len()
        .checked_sub(user_turns)
        .and_then(|k| human_idx.get(k))
        .or(human_idx.first())
    {
        Some((i, false))
    } else if let Some(&i) = all_user_idx
        .len()
        .checked_sub(user_turns)
        .and_then(|k| all_user_idx.get(k))
        .or(all_user_idx.first())
    {
        // Include the assistant turn that produced the leading tool result,
        // so replay shows the call and not just its output.
        let mut start = i;
        if let Some(j) = i.checked_sub(1)
            && transcript[j].role == Role::Assistant
        {
            start = j;
        }
        Some((start, true))
    } else {
        None
    }
}

/// Renders replayed history for a transcript like the C agent's `/history`.
///
/// Shows the last `user_turns` human turns (clamped to
/// [`HISTORY_MAX_TURNS`]) with `User:` / `Assistant:` / `Tool result:`
/// headers, per-section tail truncation, and the `--- session history ---`
/// framing. Tool-result pseudo-user turns are skipped when counting human
/// turns; a transcript tail of only tool turns falls back to recent
/// tool/assistant events. `color` enables the C agent's ANSI styling.
#[must_use]
pub fn render_history(transcript: &[Message], user_turns: usize, color: bool) -> String {
    if user_turns == 0 {
        return String::new();
    }

    let Some((start, tool_only)) = history_window(transcript, user_turns) else {
        return "\n(no user history)\n".to_owned();
    };

    let mut out = String::new();
    if color {
        out.push_str("\n\x1b[90m");
    } else {
        out.push('\n');
    }
    if tool_only {
        out.push_str("--- session history: recent tool/assistant events ---\n");
    } else {
        let _ = writeln!(
            out,
            "--- session history: last {user_turns} user turn{} ---",
            if user_turns == 1 { "" } else { "s" }
        );
    }
    if color {
        out.push_str("\x1b[0m");
    }

    for m in &transcript[start..] {
        let text = m.text.trim();
        match m.role {
            Role::User if m.is_tool_user() => {
                if color {
                    out.push_str("\x1b[90mTool result:\n");
                } else {
                    out.push_str("Tool result:\n");
                }
                push_limited(
                    &mut out,
                    m.tool_result_payload(),
                    HISTORY_TOOL_MAX_LINES,
                    HISTORY_TOOL_MAX_BYTES,
                );
                if color {
                    out.push_str("\x1b[0m");
                }
            }
            Role::User => {
                if color {
                    out.push_str("\x1b[1;32mUser:\x1b[0m\n");
                } else {
                    out.push_str("User:\n");
                }
                push_limited(
                    &mut out,
                    text,
                    HISTORY_USER_MAX_LINES,
                    HISTORY_USER_MAX_BYTES,
                );
            }
            Role::Assistant => {
                if text.is_empty() {
                    continue;
                }
                if color {
                    out.push_str("\x1b[1;37mAssistant:\x1b[0m\n");
                } else {
                    out.push_str("Assistant:\n");
                }
                push_limited(
                    &mut out,
                    text,
                    HISTORY_ASSISTANT_MAX_LINES,
                    HISTORY_ASSISTANT_MAX_BYTES,
                );
            }
        }
    }

    if color {
        out.push_str("\x1b[90m--- end history ---\x1b[0m\n");
    } else {
        out.push_str("--- end history ---\n");
    }
    out
}

/// Appends `text` to `out`, keeping only a bounded tail like the C agent.
fn push_limited(out: &mut String, text: &str, max_lines: usize, max_bytes: usize) {
    let (tail, truncated) = tail_start(text, max_lines, max_bytes);
    if truncated {
        out.push_str("\n... earlier history truncated; showing tail ...\n");
    }
    out.push_str(tail);
    if !tail.is_empty() && !tail.ends_with('\n') {
        out.push('\n');
    }
}

/// Finds the start of the last `max_lines` lines / `max_bytes` bytes.
fn tail_start(text: &str, max_lines: usize, max_bytes: usize) -> (&str, bool) {
    if text.is_empty() {
        return (text, false);
    }
    let bytes = text.as_bytes();
    let mut truncated = false;
    let mut start = 0usize;
    if max_bytes != 0 && bytes.len() > max_bytes {
        start = bytes.len() - max_bytes;
        truncated = true;
    }
    if max_lines > 0 {
        let mut scan = bytes.len();
        if scan > 0 && bytes[scan - 1] == b'\n' {
            scan -= 1;
        }
        let mut lines = 0usize;
        let mut line_start = 0usize;
        while scan > 0 {
            scan -= 1;
            if bytes[scan] == b'\n' {
                lines += 1;
                if lines == max_lines {
                    line_start = scan + 1;
                    break;
                }
            }
        }
        if line_start > 0 {
            truncated = true;
        }
        if line_start > start {
            start = line_start;
        }
    }
    while start < bytes.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    (&text[start..], truncated)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Reads (created, last-used, title) with a bounded head read; `None` when
/// the header is malformed. Titles longer than the read window are clipped —
/// acceptable for listings, which clip for display anyway.
fn read_head_meta(path: &Path) -> Option<(u64, u64, String)> {
    const HEAD_BYTES: usize = 8 * 1024;
    let mut buf = vec![0u8; HEAD_BYTES];
    let mut f = fs::File::open(path).ok()?;
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(read) => n += read,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    let buf = &buf[..n];
    let mut lines = buf.split(|&b| b == b'\n');
    if lines.next().map(String::from_utf8_lossy).as_deref() != Some(MAGIC) {
        return None;
    }
    let field = |lines: &mut std::slice::Split<'_, u8, _>, prefix: &str| -> Option<u64> {
        let line = String::from_utf8_lossy(lines.next()?).into_owned();
        line.strip_prefix(prefix)?.trim().parse().ok()
    };
    let created = field(&mut lines, "created ")?;
    let used = field(&mut lines, "used ")?;
    let title_len = usize::try_from(field(&mut lines, "title ")?).ok()?;
    // Offset of the title body: the four header lines plus their newlines.
    let offset = buf
        .split_inclusive(|&b| b == b'\n')
        .take(4)
        .map(<[u8]>::len)
        .sum::<usize>();
    let end = (offset + title_len).min(buf.len());
    let title = String::from_utf8_lossy(&buf[offset..end]).into_owned();
    Some((created, used, title))
}

/// Renders the optional trailing timestamp of a `msg`/`node` header: a
/// space-prefixed second, or nothing at all for the unstamped messages that
/// every pre-timestamp session file is made of.
fn stamp(at: u64) -> String {
    if at == 0 {
        String::new()
    } else {
        format!(" {at}")
    }
}

fn corrupt() -> SessionError {
    SessionError::new("invalid session file")
}

/// Reads a length-prefixed record body of `len` bytes plus its terminating
/// newline, advancing `pos`.
fn take_body(data: &[u8], pos: &mut usize, len: usize) -> Result<String> {
    if data.len() < *pos + len + 1 || data[*pos + len] != b'\n' {
        return Err(corrupt());
    }
    let s = String::from_utf8(data[*pos..*pos + len].to_vec()).map_err(|_| corrupt())?;
    *pos += len + 1;
    Ok(s)
}

/// Parses one `node <id> <parent|-> <role> <len>` record (the off-path nodes
/// of the session tree, issue #65) and its body.
fn parse_node_record(
    header_rest: &str,
    data: &[u8],
    pos: &mut usize,
) -> Result<crate::branch::OffNode> {
    let mut it = header_rest.split(' ');
    let id: usize = it.next().and_then(|v| v.parse().ok()).ok_or_else(corrupt)?;
    let parent = match it.next().ok_or_else(corrupt)? {
        "-" => None,
        v => Some(v.parse::<usize>().map_err(|_| corrupt())?),
    };
    let role = match it.next().ok_or_else(corrupt)? {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        _ => return Err(corrupt()),
    };
    let len: usize = it.next().and_then(|v| v.parse().ok()).ok_or_else(corrupt)?;
    // Optional trailing timestamp; absent in every pre-`/insights` file.
    let at: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let text = take_body(data, pos, len)?;
    Ok(crate::branch::OffNode {
        id,
        parent,
        message: Message { role, text, at },
    })
}

/// Parses the on-disk session format documented in the module docs.
fn read_session_file(path: &Path) -> Result<Session> {
    let data = fs::read(path)?;
    let mut pos = 0usize;

    let line = |data: &[u8], pos: &mut usize| -> Option<String> {
        let rest = &data[*pos..];
        let nl = rest.iter().position(|&b| b == b'\n')?;
        let s = String::from_utf8_lossy(&rest[..nl]).into_owned();
        *pos += nl + 1;
        Some(s)
    };

    if line(&data, &mut pos).ok_or_else(corrupt)? != MAGIC {
        return Err(corrupt());
    }
    let created_at: u64 = line(&data, &mut pos)
        .and_then(|l| l.strip_prefix("created ").map(str::to_owned))
        .and_then(|v| v.parse().ok())
        .ok_or_else(corrupt)?;
    let _used = line(&data, &mut pos)
        .filter(|l| l.starts_with("used "))
        .ok_or_else(corrupt)?;
    let title_len: usize = line(&data, &mut pos)
        .and_then(|l| l.strip_prefix("title ").map(str::to_owned))
        .and_then(|v| v.parse().ok())
        .ok_or_else(corrupt)?;
    let take = take_body;
    let title = take(&data, &mut pos, title_len)?;

    let mut transcript = Vec::new();
    let mut tag = String::new();
    let mut cwd = String::new();
    let mut tasks: Vec<crate::tasks::Task> = Vec::new();
    let mut tasks_next_id: u32 = 0;
    let mut branches: Vec<crate::branch::OffNode> = Vec::new();
    while pos < data.len() {
        let header = line(&data, &mut pos).ok_or_else(corrupt)?;
        // Session-tree off-path nodes (issue #65); absent in linear sessions
        // and in every file written before branching existed.
        if let Some(rest) = header.strip_prefix("node ") {
            branches.push(parse_node_record(rest, &data, &mut pos)?);
            continue;
        }
        // Task list records (issue #35): a `tasks <next_id>` marker followed by
        // one `task <id> <status> <subject-len> <active-form-len>` record per
        // task. Files written before the feature simply lack them.
        if let Some(rest) = header.strip_prefix("tasks ") {
            tasks_next_id = rest.trim().parse().map_err(|_| corrupt())?;
            continue;
        }
        if let Some(rest) = header.strip_prefix("task ") {
            let mut it = rest.split(' ');
            let id: u32 = it.next().and_then(|v| v.parse().ok()).ok_or_else(corrupt)?;
            let status = it.next().ok_or_else(corrupt)?;
            let status = crate::tasks::TaskStatus::parse(status).ok_or_else(corrupt)?;
            let subj_len: usize = it.next().and_then(|v| v.parse().ok()).ok_or_else(corrupt)?;
            let af_len: usize = it.next().and_then(|v| v.parse().ok()).ok_or_else(corrupt)?;
            let subject = take(&data, &mut pos, subj_len)?;
            let active = take(&data, &mut pos, af_len)?;
            tasks.push(crate::tasks::Task {
                id,
                subject,
                status,
                active_form: if active.is_empty() {
                    None
                } else {
                    Some(active)
                },
            });
            continue;
        }
        // Project directory the session ran in; absent in pre-`/insights`
        // files and whenever the agent had no path to record.
        if let Some(rest) = header.strip_prefix("cwd ") {
            let len: usize = rest.trim().parse().map_err(|_| corrupt())?;
            cwd = take(&data, &mut pos, len)?;
            continue;
        }
        // Trailing metadata record: tag + clipped last prompt (derived, so
        // only the tag is carried into the session).
        if let Some(rest) = header.strip_prefix("meta ") {
            let (tag_len, last_len) = rest.split_once(' ').ok_or_else(corrupt)?;
            let tag_len: usize = tag_len.parse().map_err(|_| corrupt())?;
            let last_len: usize = last_len.parse().map_err(|_| corrupt())?;
            tag = take(&data, &mut pos, tag_len)?;
            let _last = take(&data, &mut pos, last_len)?;
            continue;
        }
        let rest = header.strip_prefix("msg ").ok_or_else(corrupt)?;
        let mut it = rest.split(' ');
        let role = match it.next().ok_or_else(corrupt)? {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => return Err(corrupt()),
        };
        let len: usize = it.next().and_then(|v| v.parse().ok()).ok_or_else(corrupt)?;
        // Optional trailing timestamp; absent in every pre-`/insights` file.
        let at: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let text = take(&data, &mut pos, len)?;
        transcript.push(Message { role, text, at });
    }

    // Off-path nodes are a cache of structure, never trusted blindly: any
    // node whose parent did not survive is dropped with its descendants.
    let branches = crate::branch::canonicalize(&transcript, &branches);
    Ok(Session {
        id: String::new(),
        title,
        created_at,
        tag,
        cwd,
        transcript,
        branches,
        tasks: crate::tasks::TaskList::from_parts(tasks, tasks_next_id),
        dirty: false,
    })
}

/// Clipped, single-line form of the newest real (non-tool-result) user
/// prompt, for the listing metadata trailer.
fn last_prompt_of(transcript: &[Message]) -> String {
    let text = transcript
        .iter()
        .rev()
        .find(|m| m.role == Role::User && !m.is_tool_user())
        .map(|m| m.text.as_str())
        .unwrap_or_default();
    let first_line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    title_clip(first_line.trim(), 120)
}

/// Reads the trailing `meta` record with a bounded tail read; `None` for
/// files predating the record (or when validation fails).
///
/// Message bodies can contain `\nmeta ` lines, so candidates are scanned
/// from the end and accepted only when their declared lengths land exactly
/// on end-of-file.
fn read_meta_tail(path: &Path) -> Option<(String, String)> {
    const TAIL_BYTES: u64 = 8 * 1024;
    let mut f = fs::File::open(path).ok()?;
    let file_len = f.metadata().ok()?.len();
    let start = file_len.saturating_sub(TAIL_BYTES);
    let mut buf = Vec::new();
    {
        use std::io::Seek as _;
        f.seek(io::SeekFrom::Start(start)).ok()?;
        f.read_to_end(&mut buf).ok()?;
    }
    let mut search_end = buf.len();
    while let Some(at) = buf[..search_end].windows(5).rposition(|w| w == b"meta ") {
        // A candidate at buffer offset 0 is only a real line start when the
        // read began at the file start.
        let line_start = if at == 0 {
            start == 0
        } else {
            buf[at - 1] == b'\n'
        };
        if line_start && let Some(parsed) = parse_meta_at(&buf, at) {
            return Some(parsed);
        }
        if at == 0 {
            break;
        }
        search_end = at;
    }
    None
}

/// Parses a `meta` record starting at `at`, requiring it to end exactly at
/// the end of `buf` (which is the end of the file).
fn parse_meta_at(buf: &[u8], at: usize) -> Option<(String, String)> {
    let rest = &buf[at..];
    let nl = rest.iter().position(|&b| b == b'\n')?;
    let header = std::str::from_utf8(&rest[..nl]).ok()?;
    let (tag_len, last_len) = header.strip_prefix("meta ")?.split_once(' ')?;
    let (tag_len, last_len): (usize, usize) = (tag_len.parse().ok()?, last_len.parse().ok()?);
    let body = &rest[nl + 1..];
    if body.len() != tag_len + 1 + last_len + 1 {
        return None;
    }
    if body[tag_len] != b'\n' || body[tag_len + 1 + last_len] != b'\n' {
        return None;
    }
    let tag = std::str::from_utf8(&body[..tag_len]).ok()?.to_string();
    let last = std::str::from_utf8(&body[tag_len + 1..tag_len + 1 + last_len])
        .ok()?
        .to_string();
    Some((tag, last))
}

/// Renders the `/resume` picker: the most recent sessions, numbered, with
/// tag and last prompt. `entries` must already be sorted most recent first
/// (as [`SessionStore::list`] returns them).
#[must_use]
pub fn render_resume_list(entries: &[SessionEntry], now: u64, color: bool, limit: usize) -> String {
    if entries.is_empty() {
        return "no saved sessions to resume\n".to_owned();
    }
    let (num_on, title_on, help_on, dim, reset) = if color {
        (
            "\x1b[1;96m",
            "\x1b[1;97m",
            "\x1b[97m",
            "\x1b[90m",
            "\x1b[0m",
        )
    } else {
        ("", "", "", "", "")
    };
    let mut out = String::new();
    for (i, e) in entries.iter().take(limit).enumerate() {
        let when = if e.last_used != 0 {
            e.last_used
        } else {
            e.created_at
        };
        let tag = if e.tag.is_empty() {
            String::new()
        } else {
            format!(" {dim}[{}]{reset}", e.tag)
        };
        let _ = writeln!(
            out,
            "{num_on}{:>2}.{reset} {title_on}{}{reset}{tag} {dim}({}, {}){reset}",
            i + 1,
            e.title,
            format_age(when, now),
            display_id(&e.id)
        );
        if !e.last_prompt.is_empty() {
            let _ = writeln!(out, "     {dim}last: {}{reset}", e.last_prompt);
        }
    }
    let _ = writeln!(
        out,
        "{help_on}Use /resume <number> (or a sha prefix) to continue a session.{reset}"
    );
    out
}

/// Hex-encoded SHA-1 of `data` (std-only implementation).
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn sha1_hex(data: &[u8]) -> String {
    let mut h: [u32; 5] = [
        0x6745_2301,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }

    let mut out = String::with_capacity(40);
    for word in h {
        let _ = write!(out, "{word:08x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("plank-session-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn kv_nodes_finds_blobs_in_the_root_and_in_project_dirs() {
        let dir = std::env::temp_dir().join(format!("plank-kvnodes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SessionStore::open(&dir).unwrap();
        let proj = dir.join("abc123def456");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(dir.join("sysprompt-a19f.kv_raw"), vec![0u8; 100]).unwrap();
        std::fs::write(proj.join("project-7c02.kv_raw"), vec![0u8; 50]).unwrap();
        std::fs::write(dir.join("cheeky-bell.kv_raw"), vec![0u8; 200]).unwrap();
        // Neither a transcript nor an in-flight temp file is a node.
        std::fs::write(dir.join("cheeky-bell.kv"), b"plank-session 1\n").unwrap();
        std::fs::write(dir.join("cheeky-bell.kv_raw.tmp.99"), b"x").unwrap();

        let mut nodes = store.kv_nodes();
        nodes.sort_by(|a, b| a.fingerprint.cmp(&b.fingerprint));
        let got: Vec<(&str, &str, u64)> = nodes
            .iter()
            .map(|m| (m.role.as_str(), m.fingerprint.as_str(), m.bytes))
            .collect();
        assert_eq!(
            got,
            vec![
                ("project", "7c02", 50),
                ("system", "a19f", 100),
                ("session", "cheeky-bell", 200),
            ],
            "one node per .kv_raw body, role and fingerprint inferred from the name"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migration_wipes_old_blobs_and_keeps_every_transcript() {
        let dir = std::env::temp_dir().join(format!("plank-migrate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SessionStore::open(&dir).unwrap();
        let proj = dir.join("abc123def456");
        std::fs::create_dir_all(&proj).unwrap();

        // The four old blob species, the note, and one transcript.
        std::fs::write(dir.join("sysprompt-a19f.kv"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.join("sysprompt.kv"), vec![0u8; 10]).unwrap();
        std::fs::write(dir.join("cheeky-bell.payload"), vec![0u8; 200]).unwrap();
        std::fs::write(proj.join("project-7c02.kv"), vec![0u8; 50]).unwrap();
        std::fs::write(dir.join("sysprompt-last.prompt"), b"old prompt").unwrap();
        std::fs::write(dir.join("cheeky-bell.kv"), b"plank-session 1\n").unwrap();

        // Nested one level below the project directory: the sweep must not
        // recurse into it, so this legacy-named file must survive untouched.
        let nested = proj.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("project-dead.kv"), vec![0u8; 999]).unwrap();

        let reclaimed = store.migrate_legacy_blobs().expect("migration runs once");
        assert_eq!(reclaimed, 360, "100 + 10 + 200 + 50");

        assert!(!dir.join("sysprompt-a19f.kv").exists());
        assert!(!dir.join("sysprompt.kv").exists());
        assert!(!dir.join("cheeky-bell.payload").exists());
        assert!(!proj.join("project-7c02.kv").exists());
        assert!(!dir.join("sysprompt-last.prompt").exists());
        assert!(
            dir.join("cheeky-bell.kv").exists(),
            "transcripts must survive: resuming pays one re-prefill, it does not lose the conversation"
        );
        assert!(
            nested.join("project-dead.kv").exists(),
            "the sweep descends exactly one directory level, never arbitrarily deep"
        );

        assert!(
            store.migrate_legacy_blobs().is_none(),
            "second run is a no-op"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migration_does_not_touch_new_format_blobs() {
        let dir = std::env::temp_dir().join(format!("plank-migrate-new-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SessionStore::open(&dir).unwrap();
        std::fs::write(dir.join("sysprompt-a19f.kv_raw"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.join("sysprompt-a19f.json"), b"{}").unwrap();
        assert_eq!(store.migrate_legacy_blobs(), Some(0));
        assert!(dir.join("sysprompt-a19f.kv_raw").exists());
        assert!(dir.join("sysprompt-a19f.json").exists());
        assert!(
            store.migrate_legacy_blobs().is_none(),
            "marker is written even when nothing was deleted"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn blob_names_use_kv_raw_and_never_collide_with_transcripts() {
        assert_eq!(sysprompt_checkpoint_name("a19f"), "sysprompt-a19f.kv_raw");
        assert_eq!(project_checkpoint_name("7c02"), "project-7c02.kv_raw");
        // The whole point of the split: no blob name ends in the transcript
        // extension, so a `*.kv` glob can never sweep up a checkpoint.
        for n in [
            sysprompt_checkpoint_name("a19f"),
            project_checkpoint_name("7c02"),
        ] {
            assert!(
                !n.ends_with(FILE_EXT),
                "{n} must not look like a transcript"
            );
        }
        let store = SessionStore::open(
            std::env::temp_dir().join(format!("plank-blobname-{}", std::process::id())),
        )
        .unwrap();
        assert!(
            store
                .payload_path("cheeky-bell")
                .to_string_lossy()
                .ends_with("cheeky-bell.kv_raw")
        );
        let _ = std::fs::remove_dir_all(store.dir());
    }

    /// A session file exactly as plank wrote them before branching existed
    /// (issue #65): magic, header, `msg` records, `meta` trailer — and no
    /// `node` records at all.
    const LEGACY_SESSION: &[u8] = b"plank-session 1\n\
created 1700000000\n\
used 1700000100\n\
title 5\n\
hello\n\
msg user 5\n\
hello\n\
msg assistant 2\n\
hi\n\
meta 3 5\n\
wip\n\
hello\n";

    #[test]
    fn a_pre_branching_session_file_loads_as_a_single_branch_tree() {
        let dir = temp_dir("legacyload");
        let store = SessionStore::open(&dir).unwrap();
        fs::write(dir.join("old-session.kv"), LEGACY_SESSION).unwrap();

        let s = store.load("old-session").unwrap();
        assert_eq!(s.title, "hello");
        assert_eq!(s.created_at, 1_700_000_000);
        assert_eq!(s.tag, "wip");
        assert_eq!(s.transcript.len(), 2);
        assert_eq!(s.transcript[0], Message::user("hello"));
        assert_eq!(s.transcript[1], Message::assistant("hi"));
        // The hard backward-compat requirement: no branches on disk, and the
        // linear transcript reads back as a one-branch tree whose active path
        // is the whole transcript.
        assert!(s.branches.is_empty());
        let tree = s.tree();
        assert_eq!(tree.branch_count(), 1);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.active_messages(), s.transcript);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_linear_session_writes_no_node_records() {
        let dir = temp_dir("linearbytes");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("hello"));
        s.push(Message::assistant("hi"));
        let id = store.save(&mut s).unwrap();
        let bytes = fs::read(dir.join(format!("{id}.kv"))).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(!text.contains("\nnode "), "{text}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_forked_session_round_trips_its_branches() {
        let dir = temp_dir("forkroundtrip");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("first"));
        s.push(Message::assistant("a1"));
        s.push(Message::user("second"));
        s.push(Message::assistant("a2"));

        let mut tree = s.tree();
        tree.fork_at(2).unwrap();
        tree.append(Message::user("second-prime"));
        s.set_tree(&tree);
        assert_eq!(s.transcript.len(), 3);
        assert_eq!(s.branches.len(), 2);
        let id = store.save(&mut s).unwrap();

        let back = store.load(&id).unwrap();
        assert_eq!(back.transcript, s.transcript);
        assert_eq!(back.branches, s.branches);
        let tree = back.tree();
        assert_eq!(tree.branch_count(), 2);
        assert_eq!(tree.len(), 5);
        // The old branch is still reachable and still ends with "a2".
        let old_tip = tree
            .leaves()
            .into_iter()
            .find(|&n| tree.node(n).unwrap().message.text == "a2");
        assert!(old_tip.is_some(), "the forked-away branch must survive");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_node_record_is_rejected_like_any_other_record() {
        let dir = temp_dir("badnode");
        let store = SessionStore::open(&dir).unwrap();
        fs::write(
            dir.join("bad-session.kv"),
            b"plank-session 1\ncreated 1\nused 2\ntitle 2\nhi\nmsg user 2\nhi\nnode 2 - bogus 1\nx\n",
        )
        .unwrap();
        assert!(store.load("bad-session").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn node_records_with_dangling_parents_are_dropped_on_load() {
        let dir = temp_dir("danglingnode");
        let store = SessionStore::open(&dir).unwrap();
        fs::write(
            dir.join("dangle-session.kv"),
            b"plank-session 1\ncreated 1\nused 2\ntitle 2\nhi\nmsg user 2\nhi\nnode 9 7 user 4\nlost\n",
        )
        .unwrap();
        let s = store.load("dangle-session").unwrap();
        assert_eq!(s.transcript.len(), 1);
        assert!(s.branches.is_empty(), "{:?}", s.branches);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn per_project_sysprompt_checkpoints_are_not_listed_as_sessions() {
        let dir = temp_dir("syspromptskip");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("hi"));
        store.save(&mut s).unwrap();
        fs::write(dir.join("sysprompt.kv"), b"legacy").unwrap();
        fs::write(dir.join("sysprompt-0123456789ab.kv"), b"kv").unwrap();
        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 1, "checkpoints must be skipped: {entries:?}");
        assert_eq!(entries[0].id, s.id);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tier_fingerprint_chains_and_is_sensitive() {
        // fp1 stands in for the Tier 1 (sysprompt) key.
        let fp1 = sha1_hex(b"model\0system");
        let fp2 = tier_fingerprint(&fp1, b"agents-hash");
        // Deterministic.
        assert_eq!(fp2, tier_fingerprint(&fp1, b"agents-hash"));
        // A changed parent invalidates the child (chain property).
        let fp1b = sha1_hex(b"model\0other-system");
        assert_ne!(fp2, tier_fingerprint(&fp1b, b"agents-hash"));
        // A changed material invalidates the child.
        assert_ne!(fp2, tier_fingerprint(&fp1, b"agents-hash-2"));
        // The NUL separator prevents boundary collisions (parent+material
        // regrouping must not alias).
        assert_ne!(
            tier_fingerprint("ab", b"c"),
            tier_fingerprint("a", b"bc"),
            "NUL separator must make the boundary unambiguous"
        );
    }

    #[test]
    fn tier_which_change_invalidates() {
        // Simulate the volatility hierarchy: fp1 (Tier1) → fp2 (Tier2) → fp4
        // (Tier4). Assert exactly which tiers a given change rebuilds.
        let sys = "model\0system";
        let stable = b"agents-set".as_slice();
        let transcript = b"turns".as_slice();

        let build = |sys: &str, stable: &[u8], transcript: &[u8]| {
            let fp1 = sha1_hex(sys.as_bytes());
            let fp2 = tier_fingerprint(&fp1, stable);
            let fp4 = tier_fingerprint(&fp2, transcript);
            (fp1, fp2, fp4)
        };

        let (fp1, fp2, fp4) = build(sys, stable, transcript);

        // Volatile-only change (Tier 3, not keyed) leaves the whole chain valid:
        // nothing is rebuilt — this is the whole point of the reorder.
        assert_eq!(
            (fp1.clone(), fp2.clone(), fp4.clone()),
            build(sys, stable, transcript)
        );

        // Changing AGENTS.md (Tier 2 material) rebuilds Tier 2 and Tier 4,
        // but Tier 1 is reused verbatim.
        let (n1, n2, n4) = build(sys, b"agents-set-v2", transcript);
        assert_eq!(n1, fp1, "Tier 1 must survive an AGENTS.md change");
        assert_ne!(n2, fp2, "Tier 2 must rebuild on AGENTS.md change");
        assert_ne!(n4, fp4, "Tier 4 must rebuild below a rebuilt Tier 2");

        // Changing the system prompt (Tier 1) cascades to every tier below.
        let (m1, m2, m4) = build("model\0new-system", stable, transcript);
        assert_ne!(m1, fp1);
        assert_ne!(m2, fp2);
        assert_ne!(m4, fp4);
    }

    #[test]
    fn project_scoped_layout_paths() {
        let dir = temp_dir("proj-layout");
        let store = SessionStore::open(&dir).unwrap();
        let project = Path::new("/some/project");

        // Tier 2 lives under a per-project subdir; Tier 1 stays at the root.
        let pdir = store.project_dir(project);
        assert_eq!(pdir, dir.join(project_key(project)));
        assert_ne!(project_key(project), project_key(Path::new("/other")));

        let fp2 = "deadbeefcafe0000";
        let cp = store.project_checkpoint_path(project, fp2);
        assert!(cp.starts_with(&pdir));
        assert_eq!(
            cp.file_name().unwrap().to_string_lossy(),
            format!("project-{fp2}{PAYLOAD_EXT}")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn project_checkpoints_are_not_listed_as_sessions() {
        // Tier 2 files live in a subdir, so the flat session listing (which does
        // not recurse) never mistakes them for sessions.
        let dir = temp_dir("proj-skip");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("hi"));
        store.save(&mut s).unwrap();
        let project = Path::new("/proj/x");
        let pdir = store.project_dir(project);
        fs::create_dir_all(&pdir).unwrap();
        fs::write(store.project_checkpoint_path(project, "abc123"), b"kv").unwrap();
        let entries = store.list().unwrap();
        assert_eq!(
            entries.len(),
            1,
            "project checkpoints must be skipped: {entries:?}"
        );
        assert_eq!(entries[0].id, s.id);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Ported from `gc_keeps_only_the_current_system_checkpoint`: the sweep
    /// still collects a superseded Tier 1 checkpoint and still spares the live
    /// one. What changed is *why* a checkpoint dies — expiry rather than
    /// fingerprint inequality — so the clock is wound past the tier TTL to make
    /// the superseded ones collectable.
    #[test]
    fn sweep_collects_superseded_system_checkpoints_and_spares_the_live_ones() {
        let dir = temp_dir("sys-gc");
        let store = SessionStore::open(&dir).unwrap();

        for fp in ["aaa", "bbb", "ccc"] {
            fs::write(dir.join(sysprompt_checkpoint_name(fp)), b"fp\nkv").unwrap();
        }
        // Neighbours in the same root that must survive: a session transcript,
        // an in-flight temp write, and a non-blob file in a project subdir.
        let mut s = Session::new();
        s.push(Message::user("hi"));
        store.save(&mut s).unwrap();
        let payload = store.payload_path(&s.id);
        fs::write(&payload, b"payload").unwrap();
        // A *production-shaped* sidecar: a session node's fingerprint is its
        // payload fingerprint, which never equals the `<id>` in its file name.
        // Without this the node was synthesized from the path, its fingerprint
        // coincided with the id, and the "payload is active" assertion below
        // passed for the wrong reason.
        let payload_fp = payload_fingerprint(
            "model-a",
            "sys",
            "[user]\nhi\n",
            crate::engine::ThinkMode::Medium,
            0,
        );
        assert_ne!(payload_fp, s.id);
        crate::kvmeta::store(
            &payload,
            &crate::kvmeta::KvMeta::synthesized(crate::kvmeta::KvRole::Session, &payload_fp, 7, 0),
        )
        .unwrap();
        let tmp = dir.join(format!("{}.tmp.4242", sysprompt_checkpoint_name("ddd")));
        fs::write(&tmp, b"in flight").unwrap();
        let project = Path::new("/proj/sys");
        fs::create_dir_all(store.project_dir(project)).unwrap();
        // A Tier 2 checkpoint under its own fingerprint, deliberately *not*
        // "aaa": sharing a Tier 1 blob's fingerprint would leave its fate
        // ambiguous between policy and a fingerprint collision.
        let proj_cp = store.project_checkpoint_path(project, "p1");
        fs::write(&proj_cp, b"fp\nkv").unwrap();
        let stray = store.project_dir(project).join("notes.txt");
        fs::write(&stray, b"keep me").unwrap();

        let policy = crate::kvgc::SweepPolicy {
            ttl_session_secs: 14 * 86_400,
            ttl_tier_secs: 30 * 86_400,
            max_bytes: 0,
        };
        // Two Tier 1 fingerprints are live, not one: a session can hold two
        // engines — a provider main agent beside a `provider: local` sub-agent —
        // and each has its own. Keeping only one deleted the other's every
        // launch. The current session's own payload is live too.
        let future = crate::kvmeta::now_secs() + 400 * 86_400;
        let active = ["bbb", "ccc", payload_fp.as_str()];
        // Exactly the two expired, inactive bodies: `sysprompt-aaa` and the
        // Tier 2 `project-p1`, five bytes of `"fp\nkv"` each. An exact tally
        // also pins that nothing else was collected.
        assert_eq!(store.sweep(&active, &policy, future), 10);
        assert!(dir.join(sysprompt_checkpoint_name("bbb")).exists());
        assert!(dir.join(sysprompt_checkpoint_name("ccc")).exists());
        assert!(!dir.join(sysprompt_checkpoint_name("aaa")).exists());
        // New policy, stated explicitly: one sweep now covers project
        // directories too, and this checkpoint is decided by rule 4 — not
        // pinned, not active, no surviving child, and 400 days past the 30-day
        // tier TTL — so it is collected.
        assert!(
            !proj_cp.exists(),
            "an expired, inactive Tier 2 checkpoint is collected by rule 4"
        );
        assert!(payload.exists(), "the live session's payload is active");
        assert!(
            store.path_for_id(&s.id).exists(),
            "transcripts are not blobs"
        );
        assert!(tmp.exists(), "an in-flight temp write is not a blob");
        assert!(
            stray.exists(),
            "a non-blob file in a project dir is untouched"
        );

        // Idempotent, and harmless when there is nothing to collect.
        assert_eq!(store.sweep(&active, &policy, future), 0);
        let empty = temp_dir("sys-gc-empty");
        let empty_store = SessionStore::open(&empty).unwrap();
        assert_eq!(empty_store.sweep(&["bbb"], &policy, future), 0);
        // An empty active set is a full sweep of everything expired, not a
        // no-op: the caller decides what is live.
        // Everything left: the two formerly-active Tier 1 bodies at five bytes
        // each plus the session payload at seven.
        assert_eq!(store.sweep(&[], &policy, future), 17);
        assert!(!dir.join(sysprompt_checkpoint_name("bbb")).exists());
        assert!(!dir.join(sysprompt_checkpoint_name("ccc")).exists());
        assert!(!payload.exists());
        assert!(store.path_for_id(&s.id).exists());
        let _ = fs::remove_dir_all(&empty);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Ported from `gc_keeps_only_the_current_project_checkpoint`: superseded
    /// Tier 2 checkpoints are collected, the live one survives, and a
    /// neighbouring project's files are unaffected. The sweep is no longer
    /// scoped to one project directory, so the neighbour survives by being
    /// listed as active rather than by being out of scope.
    #[test]
    fn sweep_collects_superseded_project_checkpoints() {
        let dir = temp_dir("proj-gc");
        let store = SessionStore::open(&dir).unwrap();
        let project = Path::new("/proj/gc");
        let other = Path::new("/proj/other");
        fs::create_dir_all(store.project_dir(project)).unwrap();
        fs::create_dir_all(store.project_dir(other)).unwrap();

        for fp in ["aaa", "bbb", "ccc"] {
            fs::write(store.project_checkpoint_path(project, fp), b"fp\nkv").unwrap();
        }
        fs::write(store.project_checkpoint_path(other, "ddd"), b"fp\nkv").unwrap();
        let stray = store.project_dir(project).join("notes.txt");
        fs::write(&stray, b"keep me").unwrap();

        let policy = crate::kvgc::SweepPolicy {
            ttl_session_secs: 14 * 86_400,
            ttl_tier_secs: 30 * 86_400,
            max_bytes: 0,
        };
        let future = crate::kvmeta::now_secs() + 400 * 86_400;
        assert_eq!(
            store.sweep(&["bbb", "ddd"], &policy, future),
            10,
            "aaa + ccc"
        );
        assert!(store.project_checkpoint_path(project, "bbb").exists());
        assert!(!store.project_checkpoint_path(project, "aaa").exists());
        assert!(!store.project_checkpoint_path(project, "ccc").exists());
        assert!(store.project_checkpoint_path(other, "ddd").exists());
        assert!(stray.exists());

        // Idempotent, and harmless for a cache with nothing collectable.
        assert_eq!(store.sweep(&["bbb", "ddd"], &policy, future), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Two bodies can share one fingerprint: a root `sysprompt-<fp>.kv_raw` and
    /// a `<proj>/project-<fp>.kv_raw`. The verdict must follow the path, not the
    /// fingerprint — a fingerprint-keyed delete took the kept namesake with the
    /// doomed one.
    #[test]
    fn sweep_does_not_delete_a_kept_body_sharing_a_doomed_ones_fingerprint() {
        let dir = temp_dir("sweep-dup-fp");
        let store = SessionStore::open(&dir).unwrap();
        let project = Path::new("/proj/dup");
        fs::create_dir_all(store.project_dir(project)).unwrap();

        // Same fingerprint, two bodies. The root one is expired (sidecar with
        // `last_used = 0`); the project one is pinned, so only its own sidecar
        // can save it.
        let expired = dir.join(sysprompt_checkpoint_name("dup"));
        fs::write(&expired, b"fp\nkv").unwrap();
        crate::kvmeta::store(
            &expired,
            &crate::kvmeta::KvMeta::synthesized(crate::kvmeta::KvRole::System, "dup", 5, 0),
        )
        .unwrap();
        let pinned_path = store.project_checkpoint_path(project, "dup");
        fs::write(&pinned_path, b"fp\nkv").unwrap();
        let mut pinned =
            crate::kvmeta::KvMeta::synthesized(crate::kvmeta::KvRole::Project, "dup", 5, 0);
        pinned.pinned = true;
        crate::kvmeta::store(&pinned_path, &pinned).unwrap();

        let policy = crate::kvgc::SweepPolicy {
            ttl_session_secs: 14 * 86_400,
            ttl_tier_secs: 30 * 86_400,
            max_bytes: 0,
        };
        let future = crate::kvmeta::now_secs() + 400 * 86_400;
        assert_eq!(
            store.sweep(&[], &policy, future),
            5,
            "only the expired body"
        );
        assert!(!expired.exists(), "the expired body goes");
        assert!(pinned_path.exists(), "the pinned namesake survives");
        assert!(crate::kvmeta::sidecar_path(&pinned_path).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// A blob's sidecar goes with it; a phantom node otherwise survives every
    /// later scan, since `kv_nodes` reads sidecars as well as bodies.
    #[test]
    fn sweep_takes_the_sidecar_with_the_body() {
        let dir = temp_dir("sweep-sidecar");
        let store = SessionStore::open(&dir).unwrap();
        let blob = dir.join(sysprompt_checkpoint_name("aaa"));
        fs::write(&blob, b"fp\nkv").unwrap();
        let meta = crate::kvmeta::KvMeta::synthesized(crate::kvmeta::KvRole::System, "aaa", 5, 0);
        crate::kvmeta::store(&blob, &meta).unwrap();
        let policy = crate::kvgc::SweepPolicy {
            ttl_session_secs: 0,
            ttl_tier_secs: 0,
            max_bytes: 0,
        };
        assert_eq!(store.sweep(&[], &policy, crate::kvmeta::now_secs()), 5);
        assert!(!blob.exists());
        assert!(!crate::kvmeta::sidecar_path(&blob).exists());
        assert!(store.kv_nodes().is_empty(), "no phantom node remains");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The converse trust direction. Every other test corrupts the *sidecar* and
    /// checks the body still loads; this one leaves a perfectly agreeing sidecar
    /// in place and checks a bad body still does not. Metadata is advisory in
    /// both directions: the signature inside the body is the sole trust input.
    #[test]
    fn a_sidecar_that_agrees_with_a_bad_body_does_not_make_it_loadable() {
        let dir = temp_dir("sidecar-trust-converse");
        let store = SessionStore::open(&dir).unwrap();
        let key = KvKey::System { fp: "a".repeat(40) };
        let cache = crate::kvcache::KVCache::new(
            vec![1, 2, 3, 4],
            crate::ds4tokens::TokenTranscript::new(),
        );
        store
            .kv_store_labeled(&key, &cache, None, "m", &crate::kvmeta::KvLabel::Unknown)
            .unwrap();
        let path = store.kv_path(&key);
        assert!(store.kv_load(&key).is_some(), "the good body loads");

        // 1. Truncated body, sidecar untouched and still describing it.
        let good = fs::read(&path).unwrap();
        let nl = good.iter().position(|&b| b == b'\n').unwrap();
        fs::write(&path, &good[..=nl]).unwrap();
        assert_eq!(
            crate::kvmeta::load(&path).expect("a sidecar").fingerprint,
            *key.signature(),
            "the sidecar still agrees with the body's name"
        );
        assert!(
            store.kv_load(&key).is_none(),
            "a truncated body is not loadable however good its sidecar"
        );

        // 2. A body carrying a different signature, under a sidecar still naming
        // the right fingerprint.
        let other = crate::kvcache::KVCache::new(vec![9], crate::ds4tokens::TokenTranscript::new());
        fs::write(&path, other.encode("not-the-signature")).unwrap();
        assert_eq!(
            crate::kvmeta::load(&path).expect("a sidecar").fingerprint,
            *key.signature()
        );
        assert!(
            store.kv_load(&key).is_none(),
            "the body's own signature is the sole trust input"
        );
        // The blob still lists, because listing is advisory too: a blob you
        // cannot see is a blob you cannot delete.
        assert_eq!(store.kv_nodes().len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Two processes on one cache directory, modelled as two threads: one
    /// storing bodies while the other sweeps. Whatever the interleaving, a
    /// surviving body must never be left beside a sidecar describing a different
    /// blob — the sidecar's fingerprint must still be the signature the body
    /// itself carries.
    #[test]
    fn a_concurrent_store_and_sweep_never_pair_a_body_with_a_foreign_sidecar() {
        let dir = temp_dir("kv-interleave");
        let store = SessionStore::open(&dir).unwrap();
        // `KvKey::Session`, deliberately: a session body's signature is the
        // payload fingerprint, which never equals the `<id>` its file is named
        // after. With `KvKey::System` the two coincide, so a body left with *no*
        // sidecar still satisfied `from_file(path, synthesized_fp)` — the sweep
        // unlinking the sidecar of a body the other thread had just rewritten
        // passed silently, which is the only interesting outcome here.
        let ids: Vec<String> = (0..40u32).map(|i| format!("session-{i:02}")).collect();
        let fps: Vec<String> = (0..40u32).map(|i| format!("{:040x}", i + 1_000)).collect();
        for (id, fp) in ids.iter().zip(&fps) {
            assert_ne!(id, fp, "the signature must differ from the file stem");
        }
        std::thread::scope(|sc| {
            sc.spawn(|| {
                for (id, fp) in ids.iter().zip(&fps) {
                    let key = KvKey::Session {
                        id: id.clone(),
                        fp: fp.clone(),
                    };
                    let cache = crate::kvcache::KVCache::new(
                        vec![7u8; 512],
                        crate::ds4tokens::TokenTranscript::new(),
                    );
                    let _ = store.kv_store_labeled(
                        &key,
                        &cache,
                        None,
                        "m",
                        &crate::kvmeta::KvLabel::Unknown,
                    );
                }
            });
            sc.spawn(|| {
                // A zero TTL and a clock far in the future: every body the other
                // thread writes is immediately collectable, so the two threads
                // race over the same paths for the whole run.
                let policy = crate::kvgc::SweepPolicy {
                    ttl_session_secs: 0,
                    ttl_tier_secs: 0,
                    max_bytes: 0,
                };
                let future = crate::kvmeta::now_secs() + 400 * 86_400;
                for _ in 0..40 {
                    let _ = store.sweep(&[], &policy, future);
                }
            });
        });
        let survivors = store.kv_blob_nodes();
        for (path, meta) in &survivors {
            // A surviving body must still have its own sidecar. Each id is stored
            // exactly once, so a body the sweep unlinked never comes back — a
            // body present with its sidecar gone therefore means the sweep took
            // the sidecar of a body that outlived it.
            assert!(
                crate::kvmeta::sidecar_path(path).exists(),
                "{} outlived its sidecar",
                path.display()
            );
            assert_eq!(
                meta.fingerprint,
                crate::kvmeta::load(path).expect("a sidecar").fingerprint,
                "{} is nodes-scanned under a synthesized fingerprint",
                path.display()
            );
            assert!(
                crate::kvcache::KVCache::from_file(path, &meta.fingerprint).is_some(),
                "{} does not match the sidecar beside it",
                path.display()
            );
        }
        // How many bodies survive the race is timing-dependent, so nothing is
        // asserted about the count. That these bodies were genuinely sweepable —
        // i.e. the assertions above were not vacuously ranging over blobs no
        // sweep would ever have touched — is checked deterministically, single
        // threaded, after the joins.
        let policy = crate::kvgc::SweepPolicy {
            ttl_session_secs: 0,
            ttl_tier_secs: 0,
            max_bytes: 0,
        };
        let future = crate::kvmeta::now_secs() + 400 * 86_400;
        let _ = store.sweep(&[], &policy, future);
        assert!(
            store.kv_blob_nodes().is_empty(),
            "every body in this fixture is collectable"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sha1_known_vectors() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn save_load_round_trip() {
        let dir = temp_dir("roundtrip");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("Hello there,\nplease help me."));
        s.push(Message::assistant("Sure, I can help.\n"));
        assert!(s.dirty);
        let id = store.save(&mut s).unwrap();
        // A memorable `adjective-celebrity` name is minted on first save.
        assert!(is_valid_id_prefix(&id));
        assert!(id.contains('-'), "expected adjective-celebrity: {id}");
        assert!(!s.dirty);
        assert_eq!(s.title, "Hello there, please help me.");
        assert_eq!(s.id, id);

        // Loadable by full name and by prefix.
        let loaded = store.load(&id).unwrap();
        assert_eq!(loaded.id, id);
        let by_prefix = store.load(id.split_once('-').unwrap().0).unwrap();
        assert_eq!(by_prefix.id, id);
        assert_eq!(loaded.title, s.title);
        assert_eq!(loaded.created_at, s.created_at);
        assert_eq!(loaded.transcript, s.transcript);
        assert!(!loaded.dirty);

        // Resave keeps the same identity (overwrites the same file).
        s.push(Message::user("follow-up"));
        assert_eq!(store.save(&mut s).unwrap(), id);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn task_list_round_trips_through_save_and_load() {
        use crate::tasks::TaskStatus;
        let dir = temp_dir("tasks");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("plan the work"));
        s.push(Message::assistant("Planning.\n"));
        s.tasks.add("read the spec", None);
        s.tasks
            .add("write the parser", Some("Writing the parser".to_string()));
        // A subject with a newline and a fake record header, to prove the
        // length-prefixed encoding is robust.
        s.tasks
            .add("subject with\ntask 9 pending 1 2\nembedded", None);
        s.tasks
            .update(1, Some(TaskStatus::Completed), None, None)
            .unwrap();
        s.tasks
            .update(2, Some(TaskStatus::InProgress), None, None)
            .unwrap();
        let id = store.save(&mut s).unwrap();

        let loaded = store.load(&id).unwrap();
        assert_eq!(loaded.tasks, s.tasks);
        assert_eq!(loaded.tasks.tasks().len(), 3);
        assert_eq!(loaded.tasks.get(1).unwrap().status, TaskStatus::Completed);
        assert_eq!(
            loaded.tasks.get(2).unwrap().active_form.as_deref(),
            Some("Writing the parser")
        );
        assert_eq!(loaded.tasks.next_id(), s.tasks.next_id());
        // The transcript and tag survive alongside the task list.
        assert_eq!(loaded.transcript, s.transcript);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sessions_without_a_task_list_stay_byte_identical() {
        let dir = temp_dir("notasks");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("hi"));
        s.push(Message::assistant("hello"));
        let id = store.save(&mut s).unwrap();
        let raw = fs::read_to_string(store.path_for_id(&id)).unwrap();
        assert!(
            !raw.contains("\ntasks "),
            "empty list writes no task records"
        );
        let loaded = store.load(&id).unwrap();
        assert!(loaded.tasks.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn meta_trailer_round_trips_and_lists_without_full_parse() {
        let dir = temp_dir("meta");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("Fix the flaky test in ci.rs"));
        s.push(Message::assistant("Looking.\n"));
        // Adversarial body: contains a fake meta record that must not be
        // picked up by the tail reader (it does not end at EOF).
        s.push(Message::user(
            "<tool_result>\nmeta 3 4\nabc\nwxyz\n</tool_result>",
        ));
        s.push(Message::assistant("Done."));
        "wip".clone_into(&mut s.tag);
        let id = store.save(&mut s).unwrap();

        // Tail reader finds the real trailer.
        let path = store.path_for_id(&id);
        let (tag, last) = read_meta_tail(&path).unwrap();
        assert_eq!(tag, "wip");
        assert_eq!(last, "Fix the flaky test in ci.rs");

        // Head reader agrees with the full parse.
        let (created, _used, title) = read_head_meta(&path).unwrap();
        assert_eq!(created, s.created_at);
        assert_eq!(title, s.title);

        // Loading round-trips the tag and the exact transcript.
        let loaded = store.load(&id[..8]).unwrap();
        assert_eq!(loaded.tag, "wip");
        assert_eq!(loaded.transcript, s.transcript);

        // Listing carries the new metadata.
        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].tag, "wip");
        assert_eq!(entries[0].last_prompt, "Fix the flaky test in ci.rs");

        // Pre-meta files (no trailer) still list, with empty metadata.
        let raw = fs::read_to_string(&path).unwrap();
        let stripped = raw[..=raw.rfind("\nmeta ").unwrap()].to_string();
        fs::write(&path, stripped).unwrap();
        let entries = store.list().unwrap();
        assert_eq!(entries[0].tag, "");
        assert_eq!(entries[0].last_prompt, "");
        assert_eq!(entries[0].title, s.title);
        assert!(store.load(&id[..8]).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_list_renders_numbers_tags_and_last_prompt() {
        let entries = vec![SessionEntry {
            id: "a".repeat(40),
            title: "Fix CI".to_string(),
            created_at: 100,
            last_used: 100,
            file_size: 2048,
            tag: "wip".to_string(),
            last_prompt: "rerun the tests".to_string(),
            payload_bytes: 0,
            path: PathBuf::new(),
        }];
        let out = render_resume_list(&entries, 100, false, 10);
        assert!(out.contains(" 1. Fix CI [wip]"), "got: {out}");
        assert!(out.contains("last: rerun the tests"), "got: {out}");
        assert!(out.contains("Use /resume <number>"), "got: {out}");
        assert_eq!(
            render_resume_list(&[], 0, false, 10),
            "no saved sessions to resume\n"
        );
    }

    #[test]
    fn list_ordering_most_recent_first() {
        let dir = temp_dir("list");
        let store = SessionStore::open(&dir).unwrap();
        let mut ids = Vec::new();
        for (i, title) in ["first", "second", "third"].iter().enumerate() {
            let mut s = Session::new();
            s.push(Message::user(*title));
            s.created_at = 1000 + i as u64;
            ids.push(store.save(&mut s).unwrap());
        }
        // Force distinct last_used ordering by rewriting the "used" header.
        for (i, id) in ids.iter().enumerate() {
            let path = dir.join(format!("{id}.kv"));
            let text = fs::read_to_string(&path).unwrap();
            let text = text
                .lines()
                .map(|l| {
                    if l.starts_with("used ") {
                        format!("used {}", 2000 + i as u64)
                    } else {
                        l.to_owned()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            fs::write(&path, text).unwrap();
        }
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 3);
        assert_eq!(listed[0].title, "third");
        assert_eq!(listed[1].title, "second");
        assert_eq!(listed[2].title, "first");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_and_prefix_errors() {
        let dir = temp_dir("delete");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("delete me"));
        let id = store.save(&mut s).unwrap();

        // A prefix with illegal characters is rejected outright.
        assert_eq!(
            store.find("bad name").unwrap_err().to_string(),
            "invalid session name"
        );
        // A well-formed prefix that matches nothing.
        assert_eq!(
            store.find("nomatch").unwrap_err().to_string(),
            "no saved session matches nomatch"
        );
        assert_eq!(store.delete(&id).unwrap(), id);
        assert!(store.list().unwrap().is_empty());
        assert!(store.load(&id).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pre_timestamp_files_load_and_resave_byte_identically() {
        // The `msg` timestamp and the `cwd` record are optional in both
        // directions: a file written before `/insights` existed must load,
        // and re-saving it unchanged must not rewrite the format underneath
        // a user still running an older build.
        let dir = temp_dir("legacy-format");
        let store = SessionStore::open(&dir).unwrap();
        let legacy = concat!(
            "plank-session 1\n",
            "created 1000\n",
            "used 2000\n",
            "title 5\n",
            "hello\n",
            "msg user 2\n",
            "hi\n",
            "msg assistant 3\n",
            "yes\n",
            "meta 0 2\n",
            "\n",
            "hi\n",
        );
        let path = dir.join("legacy-one.kv");
        fs::write(&path, legacy).unwrap();

        let mut loaded = store.load("legacy-one").unwrap();
        assert_eq!(loaded.transcript.len(), 2);
        assert_eq!(loaded.title, "hello");
        // No stamps and no project: absent fields read as absent, not as bad
        // data.
        assert!(loaded.transcript.iter().all(|m| m.at == 0));
        assert!(loaded.cwd.is_empty());

        store.save(&mut loaded).unwrap();
        let rewritten = fs::read_to_string(&path).unwrap();
        // `used` is stamped fresh on every save; everything else must match.
        let strip_used = |s: &str| -> String {
            s.lines()
                .filter(|l| !l.starts_with("used "))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(strip_used(&rewritten), strip_used(legacy));

        // A stamped message and a recorded project do get written.
        loaded.transcript[0].at = 1_500;
        loaded.cwd = "/work".to_owned();
        store.save(&mut loaded).unwrap();
        let stamped = fs::read_to_string(&path).unwrap();
        assert!(stamped.contains("msg user 2 1500\n"), "{stamped}");
        assert!(stamped.contains("cwd 5\n/work\n"), "{stamped}");
        // And come back out again.
        let again = store.load("legacy-one").unwrap();
        assert_eq!(again.transcript[0].at, 1_500);
        assert_eq!(again.transcript[1].at, 0);
        assert_eq!(again.cwd, "/work");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn payload_round_trip_and_stale_rejection() {
        use crate::ds4tokens::TokenTranscript;
        use crate::kvcache::KVCache;

        let dir = temp_dir("payload");
        let store = SessionStore::open(&dir).unwrap();
        let med = crate::engine::ThinkMode::Medium;
        let fp = payload_fingerprint("model-a", "sys", "[user]\nhi\n", med, 0);
        let key = |fp: &str| KvKey::Session {
            id: "x".to_owned(),
            fp: fp.to_owned(),
        };
        let cache = KVCache::new(b"\x00\x01snapshot\nbytes".to_vec(), TokenTranscript::new());
        store.kv_store(&key(&fp), &cache).unwrap();
        assert_eq!(
            store.kv_load(&key(&fp)).map(|c| c.kv().to_vec()),
            Some(b"\x00\x01snapshot\nbytes".to_vec())
        );
        // Any drift in model, system prompt, or transcript is stale — and so is
        // drift in the reasoning level or the tokenization split, neither of
        // which shows up in any of the text fields.
        for stale in [
            payload_fingerprint("model-b", "sys", "[user]\nhi\n", med, 0),
            payload_fingerprint("model-a", "sys2", "[user]\nhi\n", med, 0),
            payload_fingerprint("model-a", "sys", "[user]\nhi\n[assistant]\nyo\n", med, 0),
            payload_fingerprint(
                "model-a",
                "sys",
                "[user]\nhi\n",
                crate::engine::ThinkMode::Max,
                0,
            ),
            payload_fingerprint("model-a", "sys", "[user]\nhi\n", med, 3),
        ] {
            assert_ne!(stale, fp);
            assert!(store.kv_load(&key(&stale)).is_none());
        }
        // Missing file and headerless garbage are also rejected.
        assert!(
            store
                .kv_load(&KvKey::Session {
                    id: "missing".to_owned(),
                    fp: fp.clone(),
                })
                .is_none()
        );
        fs::write(store.payload_path("x"), b"no newline header").unwrap();
        assert!(store.kv_load(&key(&fp)).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn strip_removes_payload_and_delete_cleans_up() {
        let dir = temp_dir("strip");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("strip me"));
        let id = store.save(&mut s).unwrap();

        // No payload yet: listing shows stripped, strip still succeeds.
        assert_eq!(store.payload_bytes(&id), 0);
        assert_eq!(store.strip(&id[..8]).unwrap(), (id.clone(), false));

        let put_payload = |bytes: &[u8]| {
            store
                .kv_store(
                    &KvKey::Session {
                        id: id.clone(),
                        fp: "fp".to_owned(),
                    },
                    &crate::kvcache::KVCache::new(
                        bytes.to_vec(),
                        crate::ds4tokens::TokenTranscript::new(),
                    ),
                )
                .unwrap();
        };

        put_payload(b"payload-bytes");
        assert!(store.payload_bytes(&id) > 0);
        assert!(store.list().unwrap()[0].payload_bytes > 0);
        // Payload sidecars must not be listed or matched as sessions.
        assert_eq!(store.list().unwrap().len(), 1);

        assert_eq!(store.strip(&id[..8]).unwrap(), (id.clone(), true));
        assert_eq!(store.payload_bytes(&id), 0);
        assert!(
            store.load(&id[..8]).is_ok(),
            "transcript must survive strip"
        );

        assert!(store.strip("ffffffff").is_err());

        put_payload(b"again");
        store.delete(&id[..8]).unwrap();
        assert!(!store.payload_path(&id).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn title_derivation() {
        assert_eq!(title_from_transcript(&[], 0), "(no user prompt)");
        assert_eq!(
            title_from_transcript(&[Message::user("   \n\t ")], 0),
            "(empty user prompt)"
        );
        assert_eq!(
            title_from_transcript(&[Message::user("  fix   the\nbug  now ")], 0),
            "fix the bug now"
        );
        let long = title_from_transcript(&[Message::user("abcdefghijklmnop")], 10);
        assert!(long.ends_with("..."));
        assert!(long.len() <= 10);
        assert_eq!(title_clip("short", 40), "short");
        assert_eq!(title_clip("abcdefghij", 8), "abcde...");
    }

    #[test]
    fn history_rendering_snapshot() {
        let transcript = vec![
            Message::user("first question"),
            Message::assistant("first answer"),
            Message::user("second question"),
            Message::user("<tool_result>tool output here</tool_result>"),
            Message::assistant("second answer"),
        ];
        let got = render_history(&transcript, 2, false);
        let want = "\n--- session history: last 2 user turns ---\n\
                    User:\nfirst question\n\
                    Assistant:\nfirst answer\n\
                    User:\nsecond question\n\
                    Tool result:\ntool output here\n\
                    Assistant:\nsecond answer\n\
                    --- end history ---\n";
        assert_eq!(got, want);

        assert_eq!(
            render_history(&transcript, 1, false),
            "\n--- session history: last 1 user turn ---\n\
             User:\nsecond question\n\
             Tool result:\ntool output here\n\
             Assistant:\nsecond answer\n\
             --- end history ---\n"
        );
        assert_eq!(render_history(&[], 3, false), "\n(no user history)\n");

        // Tool-only tail falls back to tool/assistant events and includes
        // the assistant turn that produced the tool result.
        let tool_only = vec![
            Message::assistant("calling tool"),
            Message::user("<tool_result>ok</tool_result>"),
        ];
        let got = render_history(&tool_only, 3, false);
        assert!(got.starts_with("\n--- session history: recent tool/assistant events ---\n"));
        assert!(got.contains("Assistant:\ncalling tool\n"));
        assert!(got.contains("Tool result:\nok\n"));
    }

    #[test]
    fn history_truncates_long_sections() {
        let long = (0..200).fold(String::new(), |mut s, i| {
            use std::fmt::Write as _;
            let _ = writeln!(s, "line {i}");
            s
        });
        let transcript = vec![Message::user(long)];
        let got = render_history(&transcript, 1, false);
        assert!(got.contains("... earlier history truncated; showing tail ..."));
        assert!(got.contains("line 199"));
        assert!(!got.contains("line 100\n"));
    }

    #[test]
    fn age_and_list_render() {
        assert_eq!(format_age(90, 100), "10s ago");
        assert_eq!(format_age(100, 400), "5m ago");
        assert_eq!(format_age(0, 100), "0s ago");
        assert_eq!(format_age(100, 100 + 7200), "2h ago");
        assert_eq!(format_age(100, 100 + 200_000), "2d ago");

        let entries = vec![SessionEntry {
            id: "aabbccddeeff00112233445566778899aabbccdd".to_owned(),
            title: "demo session".to_owned(),
            created_at: 100,
            last_used: 100,
            file_size: 2048,
            tag: String::new(),
            last_prompt: String::new(),
            payload_bytes: 0,
            path: PathBuf::from("/x"),
        }];
        let mut entries = entries;
        let out = render_session_list(&entries, 160, false);
        assert!(out.starts_with("aabbccdd > demo session\n"));
        assert!(out.contains("> 1m ago, 0.00 MB, stripped"));
        assert!(out.ends_with(
            "Use /switch <id> to select a session, /del <id> to remove, /strip <id> to strip KV cache.\n"
        ));
        entries[0].payload_bytes = 3 * 1024 * 1024;
        let out = render_session_list(&entries, 160, false);
        assert!(out.contains("> 1m ago, 0.00 MB, KV 3.00 MB"), "got: {out}");
        assert_eq!(render_session_list(&[], 0, false), "no saved sessions\n");
    }

    #[test]
    fn identity_is_stable_and_title_time_dependent() {
        let a = session_identity_sha("hello", 42);
        let b = session_identity_sha("hello", 42);
        let c = session_identity_sha("hello", 43);
        let d = session_identity_sha("hallo", 42);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn kv_store_and_load_roundtrip_per_key_kind() {
        use crate::ds4tokens::TokenTranscript;
        use crate::kvcache::KVCache;

        let dir = std::env::temp_dir().join(format!("plank-kvkey-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SessionStore::open(&dir).unwrap();
        let project = std::path::PathBuf::from("/tmp/some-project");

        let keys = [
            KvKey::System { fp: "fp1".into() },
            KvKey::Project {
                dir: project.clone(),
                fp: "fp2".into(),
            },
            KvKey::Session {
                id: "abc123".into(),
                fp: "fp3".into(),
            },
        ];
        for (i, key) in keys.iter().enumerate() {
            let i = u8::try_from(i).expect("index fits in u8");
            let cache = KVCache::new(vec![i; 4], TokenTranscript::new());
            store.kv_store(key, &cache).unwrap();
            let back = store.kv_load(key).expect("stored key loads back");
            assert_eq!(back.kv(), &[i; 4], "key {i} roundtrips");
        }

        // A key that was never written is a miss, not an error.
        assert!(
            store
                .kv_load(&KvKey::System { fp: "other".into() })
                .is_none()
        );
        // A different project dir is a different namespace, even at the same fp.
        assert!(
            store
                .kv_load(&KvKey::Project {
                    dir: "/tmp/elsewhere".into(),
                    fp: "fp2".into()
                })
                .is_none(),
            "project checkpoints must not collide across projects"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn storing_writes_a_sidecar_and_loading_counts_the_hit() {
        use crate::kvcache::KVCache;
        let dir = std::env::temp_dir().join(format!("plank-kvmeta-io-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SessionStore::open(&dir).unwrap();
        let key = KvKey::System { fp: "a19f".into() };
        let cache = KVCache::new(vec![1, 2, 3], crate::ds4tokens::TokenTranscript::new());

        store
            .kv_store_labeled(
                &key,
                &cache,
                None,
                "test-model",
                &crate::kvmeta::KvLabel::System {
                    think_mode: "max".into(),
                    trusted_len: 12,
                    global_mcp: vec!["tokensave".into()],
                },
            )
            .unwrap();

        let meta = crate::kvmeta::load(&store.kv_path(&key)).expect("sidecar written");
        assert_eq!(meta.fingerprint, "a19f");
        assert_eq!(meta.model, "test-model");
        assert_eq!(
            meta.bytes,
            std::fs::metadata(store.kv_path(&key)).unwrap().len()
        );
        assert_eq!(meta.hits, 0, "storing is not a hit");
        assert!(meta.parent.is_none());

        assert!(store.kv_load(&key).is_some());
        assert_eq!(
            crate::kvmeta::load(&store.kv_path(&key)).unwrap().hits,
            1,
            "a successful load is a hit"
        );

        // A miss must not invent a sidecar or touch counters.
        assert!(
            store
                .kv_load(&KvKey::System { fp: "beef".into() })
                .is_none()
        );
        assert_eq!(crate::kvmeta::load(&store.kv_path(&key)).unwrap().hits, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_sidecar_never_blocks_a_good_blob() {
        use crate::kvcache::KVCache;
        let dir = std::env::temp_dir().join(format!("plank-kvmeta-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SessionStore::open(&dir).unwrap();
        let key = KvKey::System { fp: "a19f".into() };
        store
            .kv_store(
                &key,
                &KVCache::new(vec![9], crate::ds4tokens::TokenTranscript::new()),
            )
            .unwrap();
        std::fs::write(
            crate::kvmeta::sidecar_path(&store.kv_path(&key)),
            b"garbage",
        )
        .unwrap();
        assert!(
            store.kv_load(&key).is_some(),
            "the signature in the body is the only trust input"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn kv_role_maps_each_key_variant() {
        use crate::kvmeta::KvRole;
        assert_eq!(kv_role(&KvKey::System { fp: "a".into() }), KvRole::System);
        assert_eq!(
            kv_role(&KvKey::Project {
                dir: PathBuf::from("/p"),
                fp: "b".into()
            }),
            KvRole::Project
        );
        assert_eq!(
            kv_role(&KvKey::Session {
                id: "i".into(),
                fp: "c".into()
            }),
            KvRole::Session
        );
    }
}
