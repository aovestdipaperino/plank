// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Per-turn file history: roll the working tree back to how it was before an
//! earlier turn (`/undo`, `/rollback --files`).
//!
//! It is not git. There is no index and no repository: a map of full-file
//! copies keyed by path and version, plus a restore that copies them back.
//!
//! * A **snapshot** is metadata, not content: for every tracked file it names
//!   which *version* was live when the turn began. The bytes live once per
//!   version in `<dir>/<hash(path)>@v<N>`.
//! * A backup of `None` is a first-class value: "this file did not exist
//!   then". Rewinding deletes a file the agent created rather than leaving it.
//! * **Only tracked files are covered.** A file becomes tracked the first
//!   time a tool is about to write it ([`FileHistory::track`]), so its
//!   pre-edit bytes are saved. Every later snapshot then covers it. The
//!   `bash` tool cannot say what it will write, so the caller tracks the
//!   paths a command *names* ([`FileHistory::track_command_paths`]); a shell
//!   command that writes files it never names is invisible here.
//! * [`FileHistory::snapshot`] mints a new version of a tracked file only when
//!   its content changed, so a snapshot over many untouched files costs one
//!   `stat` each and no copies.
//! * [`FileHistory::rewind`] takes a safety snapshot first and keeps every
//!   later snapshot, so a rewind can itself be rewound.
//!
//! State lives in `<dir>/state.txt`, rewritten atomically after each change.
//! Every operation degrades to "log and carry on": a file that cannot be
//! backed up or restored never fails the turn or the rest of the restore.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

/// Snapshots kept before the oldest is evicted (and its unreferenced backups
/// deleted).
pub const MAX_SNAPSHOTS: usize = 100;

/// Files larger than this are not tracked: a full copy per version of a large
/// binary is the cost this design cannot afford.
pub const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;

/// Most files [`FileHistory::track_command_paths`] will pull in from one
/// command that names a directory (`rm -rf src`).
pub const MAX_TREE_FILES: usize = 4000;

/// Most bytes one command's directory walk will copy.
pub const MAX_TREE_BYTES: u64 = 128 * 1024 * 1024;

/// Directory names never walked when a command names a directory.
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".build", "build", "dist"];

const STATE_FILE: &str = "state.txt";
const STATE_MAGIC: &str = "plank-file-history 1";

/// One version of one file as it stood when a snapshot was taken.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Backup {
    /// `None`: the file did not exist.
    name: Option<String>,
    version: u32,
    /// Size of the original when this version was copied, with `mtime`: the
    /// same cheap "has it changed" test git's index uses, so a snapshot over
    /// an untouched file reads nothing. `0` for "did not exist".
    size: u64,
    /// Modification time of the original then, in nanoseconds since the epoch.
    mtime: u128,
}

impl Backup {
    fn absent(version: u32) -> Self {
        Self {
            name: None,
            version,
            size: 0,
            mtime: 0,
        }
    }
}

/// Size and modification time (ns since the epoch) of `path` itself, not what a
/// symlink points at.
fn fingerprint(path: &Path) -> Option<(u64, u128)> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some((meta.len(), mtime))
}

/// The state of every tracked file at the start of one turn.
#[derive(Debug, Clone)]
struct Snapshot {
    id: u64,
    time: u64,
    kind: Kind,
    label: String,
    files: BTreeMap<PathBuf, Backup>,
}

/// Why a snapshot was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The start of a user turn: what `/undo` counts back through.
    Turn,
    /// A named `/checkpoint`.
    Checkpoint,
    /// Taken by a rewind just before it overwrote anything, so it can be undone.
    Safety,
}

impl Kind {
    fn tag(self) -> &'static str {
        match self {
            Self::Turn => "T",
            Self::Checkpoint => "C",
            Self::Safety => "S",
        }
    }

    fn from_tag(t: &str) -> Option<Self> {
        match t {
            "T" => Some(Self::Turn),
            "C" => Some(Self::Checkpoint),
            "S" => Some(Self::Safety),
            _ => None,
        }
    }
}

/// A snapshot as listed to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotInfo {
    /// Why it was taken.
    pub kind: Kind,
    /// Stable id, increasing for the life of the history.
    pub id: u64,
    /// Unix seconds when it was taken.
    pub time: u64,
    /// What it was taken for (a turn's prompt, "before undo", ...).
    pub label: String,
    /// Tracked files it covers.
    pub files: usize,
}

/// What a rewind would change, without changing it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiffStats {
    /// Files that differ from the snapshot (changed, deleted or created since).
    pub files: Vec<PathBuf>,
    /// Lines the rewind would bring back (in the snapshot, not on disk now).
    pub insertions: usize,
    /// Lines the rewind would remove (on disk now, not in the snapshot).
    pub deletions: usize,
}

/// What a rewind did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RewindReport {
    /// Files written back from a backup.
    pub restored: Vec<PathBuf>,
    /// Files deleted because they did not exist at the snapshot.
    pub removed: Vec<PathBuf>,
    /// Files left alone, each with why (outside the allowed roots, a symlink,
    /// a missing backup, an I/O error).
    pub skipped: Vec<(PathBuf, String)>,
    /// The safety snapshot taken first, to undo the rewind with.
    pub safety: Option<u64>,
}

/// The file history of one session.
#[derive(Debug)]
pub struct FileHistory {
    dir: PathBuf,
    snapshots: Vec<Snapshot>,
    tracked: BTreeSet<PathBuf>,
    sequence: u64,
    max_snapshots: usize,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Where every session's history lives: `~/.plank/file-history`. Under test it
/// is a per-process temp directory, so no test can touch the real home.
#[must_use]
pub fn root() -> PathBuf {
    #[cfg(test)]
    {
        std::env::temp_dir().join(format!("plank-file-history-test-{}", std::process::id()))
    }
    #[cfg(not(test))]
    {
        crate::home::plank_home().join("file-history")
    }
}

/// Removes the history directories under `root` last touched more than
/// `max_age_secs` ago, except `keep`. Returns how many went.
#[must_use]
pub fn sweep_in(root: &Path, max_age_secs: u64, keep: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let now = SystemTime::now();
    let mut removed = 0;
    for entry in entries.flatten() {
        if entry.file_name().to_str() == Some(keep) || !entry.path().is_dir() {
            continue;
        }
        let age = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .map_or(0, |d| d.as_secs());
        if age > max_age_secs && std::fs::remove_dir_all(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// `<root>/<session id>`: where one session's history lives.
#[must_use]
pub fn dir_in(root: &Path, session_id: &str) -> PathBuf {
    root.join(session_id)
}

impl FileHistory {
    /// An empty history stored in `dir` (created on first write).
    #[must_use]
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            snapshots: Vec::new(),
            tracked: BTreeSet::new(),
            sequence: 0,
            max_snapshots: MAX_SNAPSHOTS,
        }
    }

    /// The history saved in `dir`, or an empty one when there is none or it is
    /// unreadable (a damaged state file loses undo, never the session).
    #[must_use]
    pub fn open(dir: PathBuf) -> Self {
        let mut h = Self::new(dir);
        if let Ok(text) = std::fs::read_to_string(h.dir.join(STATE_FILE))
            && let Some((snapshots, sequence)) = parse_state(&text)
        {
            h.tracked = snapshots
                .iter()
                .flat_map(|s| s.files.keys().cloned())
                .collect();
            h.snapshots = snapshots;
            h.sequence = sequence;
        }
        h
    }

    /// Where this history keeps its backups.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The snapshots, oldest first.
    #[must_use]
    pub fn snapshots(&self) -> Vec<SnapshotInfo> {
        self.snapshots
            .iter()
            .map(|s| SnapshotInfo {
                kind: s.kind,
                id: s.id,
                time: s.time,
                label: s.label.clone(),
                files: s.files.len(),
            })
            .collect()
    }

    /// Whether snapshot `id` still exists (it may have been evicted).
    #[must_use]
    pub fn can_restore(&self, id: u64) -> bool {
        self.snapshots.iter().any(|s| s.id == id)
    }

    /// Number of tracked files.
    #[must_use]
    pub fn tracked_len(&self) -> usize {
        self.tracked.len()
    }

    /// The id of the `n`th most recent turn snapshot (`1` is the latest): the
    /// state before that turn began.
    #[must_use]
    pub fn turn_back(&self, n: usize) -> Option<u64> {
        self.snapshots
            .iter()
            .rev()
            .filter(|s| s.kind == Kind::Turn)
            .nth(n.checked_sub(1)?)
            .map(|s| s.id)
    }

    /// The id of the newest snapshot of `kind`.
    #[must_use]
    pub fn latest_of(&self, kind: Kind) -> Option<u64> {
        self.snapshots
            .iter()
            .rev()
            .find(|s| s.kind == kind)
            .map(|s| s.id)
    }

    /// The id of the newest snapshot.
    #[must_use]
    pub fn latest(&self) -> Option<u64> {
        self.snapshots.last().map(|s| s.id)
    }

    /// Caps the snapshots kept (tests, and a future setting).
    pub fn set_max_snapshots(&mut self, n: usize) {
        self.max_snapshots = n.max(1);
    }

    /// Takes a snapshot of every tracked file as it is now and returns its id.
    /// A file unchanged since its latest backup reuses that version; a changed
    /// one gets a new version; a vanished one records "did not exist".
    pub fn snapshot(&mut self, label: &str) -> u64 {
        self.snapshot_kind(Kind::Turn, label)
    }

    /// [`snapshot`](Self::snapshot) with an explicit [`Kind`].
    pub fn snapshot_kind(&mut self, kind: Kind, label: &str) -> u64 {
        let mut files = BTreeMap::new();
        for path in &self.tracked {
            let latest = self.snapshots.last().and_then(|s| s.files.get(path));
            files.insert(path.clone(), self.backup_now(path, latest));
        }
        self.sequence += 1;
        let id = self.sequence;
        self.snapshots.push(Snapshot {
            id,
            time: now_secs(),
            kind,
            label: label.chars().take(120).collect(),
            files,
        });
        self.evict();
        self.save();
        id
    }

    /// The backup entry for `path` as it is now, reusing `latest` when the
    /// file has not changed since.
    fn backup_now(&self, path: &Path, latest: Option<&Backup>) -> Backup {
        let missing = std::fs::symlink_metadata(path).is_err();
        if missing {
            return match latest {
                Some(b) if b.name.is_none() => b.clone(),
                _ => Backup::absent(latest.map_or(1, |b| b.version + 1)),
            };
        }
        if let Some(b) = latest
            && let Some(name) = &b.name
            && (fingerprint(path) == Some((b.size, b.mtime))
                || same_as_backup(path, &self.dir.join(name)))
        {
            return b.clone();
        }
        let version = self.next_version(path);
        match self.copy_backup(path, version) {
            Some(b) => b,
            // Could not copy it: keep the previous entry rather than lose the
            // file's history, or say it did not exist when it did.
            None => latest.cloned().unwrap_or_else(|| Backup::absent(version)),
        }
    }

    /// Starts tracking `path` (about to be written by a tool): saves its
    /// current bytes, or "did not exist", into the newest snapshot. A path
    /// already tracked is covered by every snapshot, so this is then a no-op,
    /// and calling it twice never overwrites the pre-edit copy with a
    /// post-edit one. Returns whether the path is (now) tracked.
    pub fn track(&mut self, path: &Path) -> bool {
        if self.tracked.contains(path) {
            return true;
        }
        if let Ok(meta) = std::fs::symlink_metadata(path)
            && (!meta.file_type().is_file() || meta.len() > MAX_FILE_BYTES)
        {
            return false;
        }
        if self.snapshots.is_empty() {
            self.snapshot("session start");
        }
        let backup = self.backup_now(path, None);
        self.tracked.insert(path.to_path_buf());
        if let Some(last) = self.snapshots.last_mut() {
            last.files.insert(path.to_path_buf(), backup);
        }
        self.save();
        true
    }

    /// Tracks what a shell command names, before it runs: every existing file
    /// it mentions (a directory contributes the files under it, bounded by
    /// [`MAX_TREE_FILES`], skipping `.git`, `target`, `node_modules` and the
    /// like), and its write destinations, existing or not (redirect targets,
    /// `touch`/`tee`/`cp`/`mv` operands). `allowed` is the write-containment
    /// test; a word outside it is ignored. Best effort: a command that writes
    /// something it never names is not covered. Returns files newly tracked.
    pub fn track_command_paths(
        &mut self,
        cwd: &Path,
        words: &[String],
        creates: &[String],
        allowed: &dyn Fn(&Path) -> bool,
    ) -> usize {
        let before = self.tracked.len();
        let mut budget = (MAX_TREE_FILES, MAX_TREE_BYTES);
        for w in words {
            let Some(path) = word_path(cwd, w) else {
                continue;
            };
            match std::fs::symlink_metadata(&path) {
                // `rm -rf target` names a regenerable directory: not worth
                // a copy of every build artifact.
                Ok(m) if m.is_dir() && is_skipped_dir(&path) => {}
                // Only a directory inside the project is walked: a command
                // naming `/usr` or `~` must not copy a machine's worth of files.
                Ok(m) if m.is_dir() && path.starts_with(cwd) => {
                    self.track_tree(&path, allowed, &mut budget);
                }
                Ok(m) if m.file_type().is_file() && allowed(&path) => {
                    self.track(&path);
                }
                _ => {}
            }
        }
        for w in creates {
            // An existing target is about to be overwritten, a missing one
            // created: either way its state now is what a rewind returns to.
            if let Some(path) = word_path(cwd, w)
                && allowed(&path)
            {
                self.track(&path);
            }
        }
        self.tracked.len() - before
    }

    fn track_tree(
        &mut self,
        dir: &Path,
        allowed: &dyn Fn(&Path) -> bool,
        budget: &mut (usize, u64),
    ) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            if budget.0 == 0 || budget.1 == 0 {
                return;
            }
            let path = entry.path();
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                if !is_skipped_dir(&path) {
                    self.track_tree(&path, allowed, budget);
                }
            } else if kind.is_file() && allowed(&path) {
                budget.0 -= 1;
                budget.1 = budget
                    .1
                    .saturating_sub(entry.metadata().map_or(0, |m| m.len()));
                self.track(&path);
            }
        }
    }

    /// What rewinding to snapshot `id` would change.
    #[must_use]
    pub fn diff_stats(&self, id: u64) -> Option<DiffStats> {
        let snap = self.snapshots.iter().find(|s| s.id == id)?;
        let mut stats = DiffStats::default();
        for path in &self.tracked {
            let Some(target) = self.target_for(snap, path) else {
                continue;
            };
            let now = read_text(path);
            let then = target
                .name
                .as_ref()
                .and_then(|n| read_text(&self.dir.join(n)));
            let exists_now = std::fs::symlink_metadata(path).is_ok();
            let differs = match (&target.name, exists_now) {
                (None, false) => false,
                (None, true) | (Some(_), false) => true,
                (Some(n), true) => !same_as_backup(path, &self.dir.join(n)),
            };
            if !differs {
                continue;
            }
            let (ins, del) =
                line_delta(now.as_deref().unwrap_or(""), then.as_deref().unwrap_or(""));
            stats.insertions += ins;
            stats.deletions += del;
            stats.files.push(path.clone());
        }
        Some(stats)
    }

    /// The backup a rewind to `snap` should restore for `path`: the one in the
    /// snapshot, or for a file first tracked later, its earliest version (the
    /// best answer to "as it was before the agent touched it").
    fn target_for(&self, snap: &Snapshot, path: &Path) -> Option<Backup> {
        if let Some(b) = snap.files.get(path) {
            return Some(b.clone());
        }
        self.snapshots
            .iter()
            .find_map(|s| s.files.get(path))
            .cloned()
    }

    /// Restores every tracked file to how snapshot `id` has it. Takes a safety
    /// snapshot first (reported in [`RewindReport::safety`]) and keeps all
    /// snapshots, so the rewind can itself be rewound. `allowed` is the
    /// write-containment test: a path outside it is skipped, never written.
    /// Per-file failures are collected, not fatal.
    pub fn rewind(&mut self, id: u64, allowed: &dyn Fn(&Path) -> bool) -> Option<RewindReport> {
        let snap = self.snapshots.iter().find(|s| s.id == id)?.clone();
        let mut report = RewindReport {
            safety: Some(self.snapshot_kind(Kind::Safety, "before rewind")),
            ..RewindReport::default()
        };
        let paths: Vec<PathBuf> = self.tracked.iter().cloned().collect();
        for path in paths {
            let Some(target) = self.target_for(&snap, &path) else {
                continue;
            };
            if !allowed(&path) {
                report
                    .skipped
                    .push((path, "outside the writable roots".into()));
                continue;
            }
            if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
                report.skipped.push((path, "is a symlink".into()));
                continue;
            }
            match &target.name {
                None => match std::fs::remove_file(&path) {
                    Ok(()) => report.removed.push(path),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => report.skipped.push((path, e.to_string())),
                },
                Some(name) => {
                    let backup = self.dir.join(name);
                    if !backup.is_file() {
                        report.skipped.push((path, "backup is missing".into()));
                    } else if std::fs::symlink_metadata(&path).is_ok()
                        && same_as_backup(&path, &backup)
                    {
                        // Already identical.
                    } else {
                        match restore_file(&backup, &path) {
                            Ok(()) => report.restored.push(path),
                            Err(e) => report.skipped.push((path, e.to_string())),
                        }
                    }
                }
            }
        }
        Some(report)
    }

    /// The next free version number for `path`.
    fn next_version(&self, path: &Path) -> u32 {
        let known = self
            .snapshots
            .iter()
            .filter_map(|s| s.files.get(path))
            .map(|b| b.version)
            .max()
            .unwrap_or(0);
        let mut v = known + 1;
        // Never overwrite an orphaned file left by an earlier run.
        while self.dir.join(backup_name(path, v)).exists() {
            v += 1;
        }
        v
    }

    fn copy_backup(&self, path: &Path, version: u32) -> Option<Backup> {
        let (size, mtime) = fingerprint(path)?;
        let name = backup_name(path, version);
        let dest = self.dir.join(&name);
        if std::fs::copy(path, &dest).is_err() {
            std::fs::create_dir_all(&self.dir).ok()?;
            std::fs::copy(path, &dest).ok()?;
        }
        Some(Backup {
            name: Some(name),
            version,
            size,
            mtime,
        })
    }

    /// Drops the oldest snapshots past the cap and deletes backups nothing
    /// refers to any more.
    fn evict(&mut self) {
        if self.snapshots.len() <= self.max_snapshots {
            return;
        }
        let excess = self.snapshots.len() - self.max_snapshots;
        self.snapshots.drain(..excess);
        let live: BTreeSet<&str> = self
            .snapshots
            .iter()
            .flat_map(|s| s.files.values())
            .filter_map(|b| b.name.as_deref())
            .collect();
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.contains('@') && !live.contains(name) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    /// Writes the state file atomically. Failures are ignored: history is a
    /// convenience, and a turn must never fail because it could not be saved.
    pub fn save(&self) {
        // A session that never wrote a file leaves nothing on disk.
        if self.tracked.is_empty() {
            return;
        }
        let _ = std::fs::create_dir_all(&self.dir);
        let tmp = self.dir.join(format!("{STATE_FILE}.tmp"));
        if std::fs::write(&tmp, render_state(&self.snapshots, self.sequence)).is_ok() {
            let _ = std::fs::rename(&tmp, self.dir.join(STATE_FILE));
        }
    }
}

fn is_skipped_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| SKIP_DIRS.contains(&n))
}

/// `<first 16 hex of sha256(path)>@v<version>`: the original name never leaks
/// into the store and odd paths cannot break its layout.
fn backup_name(path: &Path, version: u32) -> String {
    let digest = Sha256::digest(path.as_os_str().as_encoded_bytes());
    let mut hex = String::with_capacity(16);
    for b in &digest[..8] {
        let _ = write!(hex, "{b:02x}");
    }
    format!("{hex}@v{version}")
}

/// Whether `path` has the same size, permissions and bytes as `backup`. Reads
/// both files; modification times are not trusted here, because a copy can
/// carry its source's time (macOS does) and so say nothing about order.
fn same_as_backup(path: &Path, backup: &Path) -> bool {
    let (Ok(a), Ok(b)) = (std::fs::metadata(path), std::fs::metadata(backup)) else {
        return false;
    };
    if a.len() != b.len() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if a.permissions().mode() != b.permissions().mode() {
            return false;
        }
    }
    matches!((std::fs::read(path), std::fs::read(backup)), (Ok(x), Ok(y)) if x == y)
}

fn restore_file(backup: &Path, dest: &Path) -> std::io::Result<()> {
    if std::fs::copy(backup, dest).is_ok() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(backup, dest).map(|_| ())
}

fn read_text(path: &Path) -> Option<String> {
    String::from_utf8(std::fs::read(path).ok()?).ok()
}

/// Lines in `then` missing from `now` (a rewind brings them back) and lines in
/// `now` missing from `then` (it removes them), by multiset, so a moved line
/// counts as neither. A count, not a diff.
fn line_delta(now: &str, then: &str) -> (usize, usize) {
    let mut count: BTreeMap<&str, i64> = BTreeMap::new();
    for l in now.lines() {
        *count.entry(l).or_default() += 1;
    }
    for l in then.lines() {
        *count.entry(l).or_default() -= 1;
    }
    let (mut ins, mut del) = (0usize, 0usize);
    for n in count.values() {
        if *n < 0 {
            ins += usize::try_from(-n).unwrap_or(0);
        } else {
            del += usize::try_from(*n).unwrap_or(0);
        }
    }
    (ins, del)
}

/// `word` as an absolute path under `cwd`, or `None` for a flag, an empty word
/// or a URL.
fn word_path(cwd: &Path, word: &str) -> Option<PathBuf> {
    if word.is_empty() || word.starts_with('-') || word.contains("://") || word.contains('\0') {
        return None;
    }
    let expanded = match word.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var_os("HOME")?).join(rest),
        None => PathBuf::from(word),
    };
    Some(if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    })
}

/// Percent-escapes the characters the state file's line format cannot hold.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '%' => out.push_str("%25"),
            '\t' => out.push_str("%09"),
            '\n' => out.push_str("%0A"),
            '\r' => out.push_str("%0D"),
            c => out.push(c),
        }
    }
    out
}

fn unesc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let hex: String = it.by_ref().take(2).collect();
        if let Ok(b) = u8::from_str_radix(&hex, 16) {
            out.push(char::from(b));
        } else {
            out.push('%');
            out.push_str(&hex);
        }
    }
    out
}

fn render_state(snapshots: &[Snapshot], sequence: u64) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{STATE_MAGIC}");
    let _ = writeln!(out, "seq\t{sequence}");
    for s in snapshots {
        let _ = writeln!(
            out,
            "S\t{}\t{}\t{}\t{}",
            s.id,
            s.time,
            s.kind.tag(),
            esc(&s.label)
        );
        for (path, b) in &s.files {
            let _ = writeln!(
                out,
                "B\t{}\t{}\t{}\t{}\t{}",
                esc(&path.to_string_lossy()),
                b.version,
                b.name.as_deref().map_or_else(|| "-".to_string(), esc),
                b.size,
                b.mtime
            );
        }
    }
    out
}

fn parse_state(text: &str) -> Option<(Vec<Snapshot>, u64)> {
    let mut lines = text.lines();
    if lines.next()? != STATE_MAGIC {
        return None;
    }
    let mut sequence = 0;
    let mut snapshots: Vec<Snapshot> = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        match f.as_slice() {
            ["seq", n] => sequence = n.parse().ok()?,
            ["S", id, time, kind, label] => snapshots.push(Snapshot {
                id: id.parse().ok()?,
                time: time.parse().ok()?,
                kind: Kind::from_tag(kind)?,
                label: unesc(label),
                files: BTreeMap::new(),
            }),
            ["B", path, version, name, size, mtime] => {
                let backup = Backup {
                    name: (*name != "-").then(|| unesc(name)),
                    version: version.parse().ok()?,
                    size: size.parse().ok()?,
                    mtime: mtime.parse().ok()?,
                };
                snapshots
                    .last_mut()?
                    .files
                    .insert(PathBuf::from(unesc(path)), backup);
            }
            _ => return None,
        }
    }
    let seq = snapshots
        .iter()
        .map(|s| s.id)
        .max()
        .unwrap_or(0)
        .max(sequence);
    Some((snapshots, seq))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plank-fh-{tag}-{}-{}",
                std::process::id(),
                now_secs()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("work")).unwrap();
            Self(dir)
        }

        fn work(&self) -> PathBuf {
            self.0.join("work")
        }

        fn history(&self) -> FileHistory {
            FileHistory::new(self.0.join("history"))
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn any(_: &Path) -> bool {
        true
    }

    fn write(p: &Path, s: &str) {
        std::fs::write(p, s).unwrap();
    }

    fn read(p: &Path) -> String {
        std::fs::read_to_string(p).unwrap()
    }

    #[test]
    fn rewind_restores_the_bytes_from_before_a_turn() {
        let s = Scratch::new("restore");
        let f = s.work().join("a.txt");
        write(&f, "one");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f);
        write(&f, "two");
        let _turn2 = h.snapshot("turn 2");
        write(&f, "three");
        let report = h.rewind(turn1, &any).unwrap();
        assert_eq!(read(&f), "one");
        assert_eq!(report.restored, vec![f]);
    }

    #[test]
    fn rewinding_to_a_middle_turn_keeps_that_turns_edit() {
        let s = Scratch::new("middle");
        let f = s.work().join("a.txt");
        write(&f, "one");
        let mut h = s.history();
        h.snapshot("turn 1");
        h.track(&f);
        write(&f, "two");
        let turn2 = h.snapshot("turn 2");
        write(&f, "three");
        h.snapshot("turn 3");
        h.rewind(turn2, &any).unwrap();
        assert_eq!(read(&f), "two");
    }

    #[test]
    fn a_file_the_agent_created_is_deleted_by_a_rewind() {
        let s = Scratch::new("created");
        let f = s.work().join("new.txt");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f); // about to be created
        write(&f, "hello");
        let report = h.rewind(turn1, &any).unwrap();
        assert!(!f.exists());
        assert_eq!(report.removed, vec![f]);
    }

    #[test]
    fn a_deleted_file_comes_back() {
        let s = Scratch::new("deleted");
        let f = s.work().join("gone.txt");
        write(&f, "precious");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f);
        std::fs::remove_file(&f).unwrap();
        h.snapshot("turn 2");
        h.rewind(turn1, &any).unwrap();
        assert_eq!(read(&f), "precious");
    }

    #[test]
    fn tracking_twice_does_not_replace_the_pre_edit_copy() {
        let s = Scratch::new("twice");
        let f = s.work().join("a.txt");
        write(&f, "before");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f);
        write(&f, "after");
        h.track(&f); // a second edit tool call
        h.rewind(turn1, &any).unwrap();
        assert_eq!(read(&f), "before");
    }

    #[test]
    fn unchanged_files_cost_no_new_backup() {
        let s = Scratch::new("reuse");
        let f = s.work().join("a.txt");
        write(&f, "same");
        let mut h = s.history();
        h.snapshot("turn 1");
        h.track(&f);
        for i in 2..6 {
            h.snapshot(&format!("turn {i}"));
        }
        let backups = std::fs::read_dir(h.dir())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().contains('@'))
            .count();
        assert_eq!(backups, 1);
    }

    #[test]
    fn a_rewind_can_be_rewound() {
        let s = Scratch::new("redo");
        let f = s.work().join("a.txt");
        write(&f, "old");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f);
        write(&f, "new");
        h.snapshot("turn 2");
        let report = h.rewind(turn1, &any).unwrap();
        assert_eq!(read(&f), "old");
        h.rewind(report.safety.unwrap(), &any).unwrap();
        assert_eq!(read(&f), "new");
    }

    #[test]
    fn a_path_outside_the_allowed_roots_is_never_written() {
        let s = Scratch::new("roots");
        let f = s.work().join("a.txt");
        write(&f, "old");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f);
        write(&f, "new");
        let report = h.rewind(turn1, &|_| false).unwrap();
        assert_eq!(read(&f), "new");
        assert_eq!(report.skipped.len(), 1);
    }

    #[test]
    fn a_symlink_is_not_followed_on_restore() {
        let s = Scratch::new("symlink");
        let f = s.work().join("a.txt");
        let victim = s.work().join("victim.txt");
        write(&f, "old");
        write(&victim, "safe");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f);
        std::fs::remove_file(&f).unwrap();
        std::os::unix::fs::symlink(&victim, &f).unwrap();
        let report = h.rewind(turn1, &any).unwrap();
        assert_eq!(read(&victim), "safe");
        assert_eq!(report.skipped.len(), 1);
    }

    #[test]
    fn the_executable_bit_survives_a_round_trip() {
        use std::os::unix::fs::PermissionsExt as _;
        let s = Scratch::new("mode");
        let f = s.work().join("run.sh");
        write(&f, "#!/bin/sh\n");
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f);
        std::fs::remove_file(&f).unwrap();
        h.snapshot("turn 2");
        h.rewind(turn1, &any).unwrap();
        assert_eq!(
            std::fs::metadata(&f).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }

    #[test]
    fn diff_stats_preview_matches_what_a_rewind_changes() {
        let s = Scratch::new("stats");
        let f = s.work().join("a.txt");
        let g = s.work().join("b.txt");
        write(&f, "a\nb\nc\n");
        write(&g, "keep\n");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        h.track(&f);
        h.track(&g);
        write(&f, "a\nB\nc\nd\n");
        let stats = h.diff_stats(turn1).unwrap();
        assert_eq!(stats.files, vec![f]);
        assert_eq!((stats.insertions, stats.deletions), (1, 2));
        assert_eq!(h.diff_stats(999), None);
    }

    #[test]
    fn eviction_drops_old_snapshots_and_their_orphaned_backups() {
        let s = Scratch::new("evict");
        let f = s.work().join("a.txt");
        write(&f, "v0");
        let mut h = s.history();
        h.set_max_snapshots(2);
        let first = h.snapshot("turn 1");
        h.track(&f);
        for i in 1..5 {
            write(&f, &format!("v{i}"));
            h.snapshot(&format!("turn {}", i + 1));
        }
        assert_eq!(h.snapshots().len(), 2);
        assert!(!h.can_restore(first));
        let backups = std::fs::read_dir(h.dir())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().contains('@'))
            .count();
        assert_eq!(backups, 2, "only the two live versions remain");
    }

    #[test]
    fn state_survives_a_reopen() {
        let s = Scratch::new("reopen");
        let f = s.work().join("a.txt");
        write(&f, "old");
        let mut h = s.history();
        let turn1 = h.snapshot("turn\t1\nwith odd label");
        h.track(&f);
        write(&f, "new");
        h.snapshot("turn 2");
        let mut again = FileHistory::open(h.dir().to_path_buf());
        assert_eq!(again.snapshots().len(), h.snapshots().len());
        assert_eq!(again.snapshots()[0].label, "turn\t1\nwith odd label");
        again.rewind(turn1, &any).unwrap();
        assert_eq!(read(&f), "old");
        // Ids keep growing past a reopen.
        assert!(again.snapshot("later") > turn1);
    }

    #[test]
    fn a_damaged_state_file_gives_an_empty_history() {
        let s = Scratch::new("damaged");
        let dir = s.0.join("history");
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir.join(STATE_FILE), "not a state file\n\u{0}garbage");
        let h = FileHistory::open(dir);
        assert!(h.snapshots().is_empty());
    }

    #[test]
    fn a_command_that_names_files_and_a_directory_has_them_tracked() {
        let s = Scratch::new("cmd");
        let w = s.work();
        std::fs::create_dir_all(w.join("src/deep")).unwrap();
        std::fs::create_dir_all(w.join("target")).unwrap();
        write(&w.join("src/a.rs"), "a");
        write(&w.join("src/deep/b.rs"), "b");
        write(&w.join("target/big"), "x");
        write(&w.join("one.txt"), "1");
        let mut h = s.history();
        let turn1 = h.snapshot("turn 1");
        let words: Vec<String> = ["rm", "-rf", "src", "one.txt", "target", "--force"]
            .iter()
            .map(ToString::to_string)
            .collect();
        let creates = vec!["out.log".to_string()];
        let n = h.track_command_paths(&w, &words, &creates, &any);
        assert_eq!(
            n, 4,
            "src/a.rs, src/deep/b.rs, one.txt, out.log; not target/"
        );
        std::fs::remove_dir_all(w.join("src")).unwrap();
        std::fs::remove_file(w.join("one.txt")).unwrap();
        write(&w.join("out.log"), "noise");
        h.rewind(turn1, &any).unwrap();
        assert_eq!(read(&w.join("src/deep/b.rs")), "b");
        assert_eq!(read(&w.join("one.txt")), "1");
        assert!(!w.join("out.log").exists());
    }

    #[test]
    fn command_words_outside_the_allowed_roots_are_not_tracked() {
        let s = Scratch::new("cmdroots");
        let w = s.work();
        write(&w.join("a.txt"), "a");
        let mut h = s.history();
        h.snapshot("turn 1");
        let words = vec!["a.txt".to_string()];
        assert_eq!(h.track_command_paths(&w, &words, &[], &|_| false), 0);
    }

    #[test]
    fn a_huge_or_non_regular_file_is_not_tracked() {
        let s = Scratch::new("skip");
        let mut h = s.history();
        h.snapshot("turn 1");
        assert!(!h.track(&s.work()), "a directory is not a file");
    }

    #[test]
    fn a_session_that_tracked_nothing_leaves_nothing_on_disk() {
        let s = Scratch::new("quiet");
        let mut h = s.history();
        h.snapshot("turn 1");
        h.snapshot("turn 2");
        assert!(!h.dir().exists());
    }

    #[test]
    fn turn_back_counts_only_turn_snapshots() {
        let s = Scratch::new("turnback");
        let f = s.work().join("a.txt");
        write(&f, "x");
        let mut h = s.history();
        let t1 = h.snapshot("turn 1");
        h.track(&f);
        let t2 = h.snapshot("turn 2");
        h.snapshot_kind(Kind::Checkpoint, "cp");
        let t3 = h.snapshot("turn 3");
        h.rewind(t2, &any).unwrap();
        assert_eq!(h.turn_back(1), Some(t3));
        assert_eq!(h.turn_back(2), Some(t2));
        assert_eq!(h.turn_back(3), Some(t1));
        assert_eq!(h.turn_back(4), None);
        assert_eq!(h.turn_back(0), None);
        assert!(h.latest_of(Kind::Safety).is_some());
    }

    #[test]
    fn sweep_removes_only_old_directories_other_than_the_kept_one() {
        let s = Scratch::new("sweep");
        let root = s.0.join("root");
        for d in ["old", "keep"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        assert_eq!(sweep_in(&root, 3600, "keep"), 0, "nothing is old yet");
        assert_eq!(sweep_in(&root, 0, "keep"), 0, "age must exceed, not equal");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert_eq!(sweep_in(&root, 0, "keep"), 1);
        assert!(root.join("keep").exists());
        assert!(!root.join("old").exists());
    }

    #[test]
    fn escaping_round_trips() {
        for t in ["plain", "a\tb", "a\nb", "100%", "%41", "tab\t%09"] {
            assert_eq!(unesc(&esc(t)), t);
        }
    }

    #[test]
    fn backup_names_hide_the_path_and_separate_versions() {
        let a = backup_name(Path::new("/x/secret.txt"), 1);
        assert!(!a.contains("secret"));
        assert_ne!(a, backup_name(Path::new("/x/secret.txt"), 2));
        assert_ne!(a, backup_name(Path::new("/y/secret.txt"), 1));
    }
}
