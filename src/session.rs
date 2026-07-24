// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Agent session persistence, listing, and history rendering.
//!
//! Port of the ds4 agent sections "Agent KV Store And Session Persistence"
//! and "Session Listing, History Rendering, And Completion". The C agent
//! persists engine KV state plus a rendered token transcript in one file;
//! plank persists the text transcript as the session file and keeps the
//! engine KV state in a fingerprinted `<name>.payload` sidecar (see
//! [`write_payload`]), while keeping the same user-visible behavior: sessions
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
//! msg <user|assistant> <byte-len>
//! <message bytes>
//! ...
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
/// Extension of the engine KV payload sidecar written next to a transcript.
///
/// The C agent stores the engine payload inside the same `.kv` file as the
/// rendered text; plank's v1 transcript format predates payloads and must
/// keep loading, so the payload lives in a sidecar (`<sha>.payload`) instead.
/// The sidecar is a rebuildable cache: its first line is a fingerprint tying
/// it to the exact model, system prompt, and rendered transcript, and a
/// mismatch means the payload is ignored and rebuilt by prefill — never
/// trusted (see [`read_payload`]).
const PAYLOAD_EXT: &str = ".payload";

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
}

impl Message {
    /// Creates a user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            text: text.into(),
        }
    }

    /// Creates an assistant message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            text: text.into(),
        }
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
    /// Alternating role-tagged messages.
    pub transcript: Vec<Message>,
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
            transcript: Vec::new(),
            tasks: crate::tasks::TaskList::new(),
            dirty: false,
        }
    }

    /// Appends a message and marks the session dirty.
    pub fn push(&mut self, message: Message) {
        self.transcript.push(message);
        self.dirty = true;
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
            let _ = writeln!(body, "msg {} {}", m.role.tag(), m.text.len());
            body.extend_from_slice(m.text.as_bytes());
            body.push(b'\n');
        }
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
        let _ = fs::remove_file(self.payload_path(&id));
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
    /// names, plus legacy `<40-hex>.kv` ids. The system prompt checkpoints
    /// (`sysprompt-*.kv`, plus the legacy shared `sysprompt.kv`) and temp
    /// files (`*.kv.tmp.*`, which don't end in `.kv`) are skipped.
    fn session_files(&self) -> Result<Vec<(String, PathBuf)>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(stem) = name.strip_suffix(FILE_EXT) else {
                continue;
            };
            if stem.starts_with(SYSPROMPT_STEM) || !is_valid_id_prefix(stem) {
                continue;
            }
            out.push((stem.to_ascii_lowercase(), entry.path()));
        }
        Ok(out)
    }
}

/// File-stem prefix of the system-prompt KV checkpoints, which share the
/// cache dir with session files but are not sessions.
const SYSPROMPT_STEM: &str = "sysprompt";

/// File name of the per-project system-prompt KV checkpoint:
/// `sysprompt-<12 hex of sha1(project path)>.kv`.
///
/// The system prompt embeds per-project inputs (AGENTS.md, the local MCP
/// config, the session-start context), so a single shared checkpoint would be
/// invalidated and rebuilt on nearly every project switch — and two projects
/// used alternately would thrash it. Keying the file by project directory
/// gives each project its own stable snapshot.
#[must_use]
pub fn sysprompt_checkpoint_name(project_dir: &Path) -> String {
    let hash = sha1_hex(project_dir.to_string_lossy().as_bytes());
    format!("{SYSPROMPT_STEM}-{}{FILE_EXT}", &hash[..12])
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
#[must_use]
pub fn payload_fingerprint(model: &str, system: &str, transcript_render: &str) -> String {
    let mut data = Vec::with_capacity(model.len() + system.len() + transcript_render.len() + 2);
    data.extend_from_slice(model.as_bytes());
    data.push(0);
    data.extend_from_slice(system.as_bytes());
    data.push(0);
    data.extend_from_slice(transcript_render.as_bytes());
    sha1_hex(&data)
}

/// Writes an engine KV payload file: the fingerprint line, then raw bytes.
///
/// Same layout as the `sysprompt.kv` checkpoint, so both caches share one
/// staleness rule. Written atomically (temp file + rename) so a crash never
/// leaves a truncated payload that could be half-loaded.
///
/// # Errors
///
/// Returns the underlying I/O error; callers treat payload writes as
/// best-effort (a failure just means the next resume re-prefills).
pub fn write_payload(path: &Path, fingerprint: &str, bytes: &[u8]) -> io::Result<()> {
    let mut file = Vec::with_capacity(bytes.len() + fingerprint.len() + 1);
    file.extend_from_slice(fingerprint.as_bytes());
    file.push(b'\n');
    file.extend_from_slice(bytes);
    let tmp = path.with_extension(format!("payload.tmp.{}", std::process::id()));
    fs::write(&tmp, &file)?;
    fs::rename(&tmp, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })
}

/// Reads an engine KV payload written by [`write_payload`], returning its
/// bytes only when the stored fingerprint matches `fingerprint` exactly.
/// A missing file, malformed header, or fingerprint mismatch returns `None`:
/// the caller falls back to a full prefill (stale payloads are rebuilt,
/// never trusted).
#[must_use]
pub fn read_payload(path: &Path, fingerprint: &str) -> Option<Vec<u8>> {
    let mut bytes = fs::read(path).ok()?;
    let nl = bytes.iter().position(|&b| b == b'\n')?;
    if bytes[..nl] != *fingerprint.as_bytes() {
        return None;
    }
    Some(bytes.split_off(nl + 1))
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

fn corrupt() -> SessionError {
    SessionError::new("invalid session file")
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
    let take = |data: &[u8], pos: &mut usize, len: usize| -> Result<String> {
        if data.len() < *pos + len + 1 || data[*pos + len] != b'\n' {
            return Err(corrupt());
        }
        let s = String::from_utf8(data[*pos..*pos + len].to_vec()).map_err(|_| corrupt())?;
        *pos += len + 1;
        Ok(s)
    };
    let title = take(&data, &mut pos, title_len)?;

    let mut transcript = Vec::new();
    let mut tag = String::new();
    let mut tasks: Vec<crate::tasks::Task> = Vec::new();
    let mut tasks_next_id: u32 = 0;
    while pos < data.len() {
        let header = line(&data, &mut pos).ok_or_else(corrupt)?;
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
        let (role, len) = rest.split_once(' ').ok_or_else(corrupt)?;
        let role = match role {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => return Err(corrupt()),
        };
        let len: usize = len.parse().map_err(|_| corrupt())?;
        let text = take(&data, &mut pos, len)?;
        transcript.push(Message { role, text });
    }

    Ok(Session {
        id: String::new(),
        title,
        created_at,
        tag,
        transcript,
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
    fn sysprompt_checkpoint_name_is_stable_and_project_keyed() {
        let a = sysprompt_checkpoint_name(Path::new("/proj/a"));
        let b = sysprompt_checkpoint_name(Path::new("/proj/b"));
        assert_ne!(a, b, "different projects must not share a checkpoint");
        assert_eq!(a, sysprompt_checkpoint_name(Path::new("/proj/a")));
        assert!(a.starts_with("sysprompt-") && a.ends_with(FILE_EXT), "{a}");
    }

    #[test]
    fn per_project_sysprompt_checkpoints_are_not_listed_as_sessions() {
        let dir = temp_dir("syspromptskip");
        let store = SessionStore::open(&dir).unwrap();
        let mut s = Session::new();
        s.push(Message::user("hi"));
        store.save(&mut s).unwrap();
        fs::write(dir.join("sysprompt.kv"), b"legacy").unwrap();
        fs::write(
            dir.join(sysprompt_checkpoint_name(Path::new("/proj/a"))),
            b"kv",
        )
        .unwrap();
        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 1, "checkpoints must be skipped: {entries:?}");
        assert_eq!(entries[0].id, s.id);
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
    fn payload_round_trip_and_stale_rejection() {
        let dir = temp_dir("payload");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.payload");
        let fp = payload_fingerprint("model-a", "sys", "[user]\nhi\n");
        write_payload(&path, &fp, b"\x00\x01snapshot\nbytes").unwrap();
        assert_eq!(
            read_payload(&path, &fp).as_deref(),
            Some(b"\x00\x01snapshot\nbytes".as_slice())
        );
        // Any drift in model, system prompt, or transcript is stale.
        for stale in [
            payload_fingerprint("model-b", "sys", "[user]\nhi\n"),
            payload_fingerprint("model-a", "sys2", "[user]\nhi\n"),
            payload_fingerprint("model-a", "sys", "[user]\nhi\n[assistant]\nyo\n"),
        ] {
            assert_ne!(stale, fp);
            assert!(read_payload(&path, &stale).is_none());
        }
        // Missing file and headerless garbage are also rejected.
        assert!(read_payload(&dir.join("missing.payload"), &fp).is_none());
        fs::write(&path, b"no newline header").unwrap();
        assert!(read_payload(&path, &fp).is_none());
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

        write_payload(&store.payload_path(&id), "fp", b"payload-bytes").unwrap();
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

        write_payload(&store.payload_path(&id), "fp", b"again").unwrap();
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
}
