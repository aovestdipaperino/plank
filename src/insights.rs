// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `/insights`: a personal usage report computed from saved sessions.
//!
//! Adapted from Claude Code's builtin of the same name (reverse-engineered
//! design notes in `local/INSIGHTS.md`), with its central discipline kept
//! intact: **every number in the report is computed deterministically in
//! code, and the model is used only for prose it cannot replace**. The two
//! halves never mix — a failed or skipped model call costs the report its
//! narrative, never its statistics.
//!
//! The reference also asks the model to judge each session individually
//! ("facets": inferred goal, satisfaction, friction). plank does not: facet
//! extraction is one model call per session, and against a local engine over
//! a few hundred saved sessions that is minutes of compute for a paragraph of
//! prose. Instead the per-session pass is fully deterministic and the model
//! sees only the finished aggregate, which costs a fixed handful of calls no
//! matter how much history has accumulated.
//!
//! # Pipeline
//!
//! 1. **Scan** every session in the store (`SessionStore::list`), cheaply —
//!    listing metadata only, no transcript parsing.
//! 2. **Per-session stats** ([`SessionMeta`]): served from
//!    `~/.plank/usage-data/session-meta/<id>.json` when the cache entry still
//!    matches the file's size and mtime, otherwise computed by parsing the
//!    transcript and written back. Unlike the reference, whose transcripts are
//!    append-only per session, plank rewrites a session file in place on every
//!    save — so the cache is keyed by id *and validated against the file's
//!    size and mtime*, not by id alone.
//! 3. **Filter** sessions that are too slight to say anything about (fewer
//!    than [`MIN_HUMAN_MESSAGES`] human turns).
//! 4. **Aggregate** ([`Aggregated`]): totals, histograms, response-time
//!    percentiles, activity by hour, and overlapping-session detection.
//! 5. **Narrative** ([`Narrative`]): a fixed number of engine calls over the
//!    aggregate JSON, each demanding one JSON object, each independently
//!    droppable on failure.
//! 6. **Render**: a self-contained HTML report plus a condensed summary for
//!    the terminal.
//!
//! # What the transcript can and cannot say
//!
//! plank stores a session as flat role-tagged text, so the statistics are
//! recovered by re-parsing it: tool calls come from the DSML stanzas in
//! assistant turns (via [`crate::export::segments`], the same parse the HTML
//! export uses), and tool outcomes from the `Tool result N (name):` framing
//! [`crate::tools::dispatch_all`] writes, where a failure is any payload
//! beginning `Tool error:` and a failed shell is a non-zero `exit_status=`.
//!
//! Timestamps are the one thing no amount of parsing recovers: per-message
//! stamps were added to the session format for this feature, so response
//! times, activity by hour, and overlap detection exist only for sessions
//! recorded since. Sessions older than that still contribute every other
//! statistic; the report says how many of them there were rather than
//! quietly averaging over a smaller set.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::export::{Segment, html_escape};
use crate::session::{Message, Role, Session, SessionEntry, SessionStore};

/// Bumped whenever [`SessionMeta`] extraction changes meaning, so stale
/// cache entries are recomputed instead of silently mixing two definitions
/// of the same statistic into one report.
const META_VERSION: u32 = 1;

/// A session needs this many human turns before it says anything about how
/// the user works. One-shot questions and warm-ups are counted (the report
/// shows how many there were) but excluded from every other statistic.
pub const MIN_HUMAN_MESSAGES: u32 = 2;

/// Gap bounds, in seconds, for a user response time to be believable: below
/// this the "gap" is the user hitting enter on something already typed, and
/// the upper bound is in [`MAX_RESPONSE_SECS`].
const MIN_RESPONSE_SECS: u64 = 2;
/// Above this a gap is a break, a meeting, or overnight — not a response.
const MAX_RESPONSE_SECS: u64 = 3600;

/// Stands in for the project of a session saved before the directory was
/// recorded. Displayed in the chart, but never sent to the model as if it
/// were a project name.
const UNRECORDED_PROJECT: &str = "(unrecorded)";

/// Window within which two sessions trading messages count as concurrent use.
const OVERLAP_WINDOW_SECS: u64 = 30 * 60;

/// Buckets for the response-time histogram, as `(upper bound in seconds,
/// label)`; the final bucket is unbounded.
const RESPONSE_BUCKETS: &[(u64, &str)] = &[
    (10, "2-10s"),
    (30, "10-30s"),
    (60, "30s-1m"),
    (120, "1-2m"),
    (300, "2-5m"),
    (900, "5-15m"),
    (u64::MAX, ">15m"),
];

/// File extension to language name, for the "what you work in" histogram.
const EXTENSION_TO_LANGUAGE: &[(&str, &str)] = &[
    ("rs", "Rust"),
    ("ts", "TypeScript"),
    ("tsx", "TypeScript"),
    ("js", "JavaScript"),
    ("jsx", "JavaScript"),
    ("py", "Python"),
    ("go", "Go"),
    ("c", "C"),
    ("h", "C"),
    ("cc", "C++"),
    ("cpp", "C++"),
    ("hpp", "C++"),
    ("java", "Java"),
    ("kt", "Kotlin"),
    ("swift", "Swift"),
    ("rb", "Ruby"),
    ("php", "PHP"),
    ("cs", "C#"),
    ("sh", "Shell"),
    ("zsh", "Shell"),
    ("bash", "Shell"),
    ("sql", "SQL"),
    ("html", "HTML"),
    ("css", "CSS"),
    ("scss", "CSS"),
    ("md", "Markdown"),
    ("toml", "TOML"),
    ("yaml", "YAML"),
    ("yml", "YAML"),
    ("json", "JSON"),
];

/// Prefixes plank itself prepends to the first user turns of a session
/// (agent instructions, git status, hook and system context). They are
/// machinery, not something the user typed, and must not be counted as human
/// messages or read as prompts.
const CONTEXT_PREFIXES: &[&str] = &[
    "Agent instructions:",
    "This is the git status at the start",
    "<hook_context>",
    "<system-reminder",
    "<local-command",
    "Caveat:",
    "<command-name>",
];

// ---------------------------------------------------------------------------
// Per-session statistics
// ---------------------------------------------------------------------------

/// Everything one session contributes to the report, computed once and
/// cached on disk.
///
/// Serialized as JSON under `usage-data/session-meta/<id>.json`. `version`,
/// `src_size`, and `src_mtime` exist purely to decide whether a cache entry
/// still describes the file on disk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMeta {
    /// Extraction version the entry was written by; a mismatch invalidates it.
    pub version: u32,
    /// Session id (the file's stem).
    pub id: String,
    /// Size of the session file when this entry was computed.
    pub src_size: u64,
    /// Mtime of the session file when this entry was computed.
    pub src_mtime: u64,

    /// Session title.
    pub title: String,
    /// Project directory, or empty for sessions saved before it was recorded.
    pub project: String,
    /// First thing the user actually typed, clipped.
    pub first_prompt: String,
    /// Creation time, unix seconds.
    pub created_at: u64,
    /// Last save time, unix seconds.
    pub last_used: u64,
    /// Stamp of the first stamped message, or 0 when the session predates
    /// per-message timestamps.
    pub first_ts: u64,
    /// Stamp of the last stamped message.
    pub last_ts: u64,

    /// Turns the user typed (excluding tool results and injected context).
    pub human_messages: u32,
    /// Model turns.
    pub assistant_messages: u32,
    /// Tool-result turns fed back to the model.
    pub tool_results: u32,
    /// Transcript size in bytes.
    pub bytes: u64,

    /// Tool name to invocation count.
    pub tools: BTreeMap<String, u32>,
    /// Language name to number of file touches.
    pub languages: BTreeMap<String, u32>,
    /// Distinct files written or edited.
    pub files: Vec<String>,
    /// Lines added by `edit`/`write`.
    pub lines_added: u32,
    /// Lines removed by `edit`.
    pub lines_removed: u32,
    /// `git commit` invocations.
    pub commits: u32,
    /// `git push` invocations.
    pub pushes: u32,

    /// Error category to count.
    pub errors: BTreeMap<String, u32>,
    /// Tool name to failure count.
    pub error_tools: BTreeMap<String, u32>,
    /// Turns where the user interrupted the model.
    pub interruptions: u32,

    /// Believable user response gaps, in seconds.
    pub response_secs: Vec<u64>,
    /// Timestamps of human turns, for activity and overlap analysis. Empty
    /// for sessions recorded before per-message stamps existed.
    pub human_ts: Vec<u64>,
}

impl SessionMeta {
    /// Whether the entry still describes the file it was computed from.
    fn matches(&self, entry: &SessionEntry, mtime: u64) -> bool {
        self.version == META_VERSION && self.src_size == entry.file_size && self.src_mtime == mtime
    }

    /// Whether the session has enough human input to describe how its owner
    /// works.
    #[must_use]
    pub fn is_substantive(&self) -> bool {
        self.human_messages >= MIN_HUMAN_MESSAGES
    }

    /// Wall-clock span of the session in seconds.
    ///
    /// Measured across the message stamps when there are any. The file's own
    /// `created`/`used` times cannot stand in for this: a session's `created`
    /// is stamped at its *first save*, so one saved only at exit has
    /// `created == used` and would read as instantaneous. The file times are
    /// still the fallback for sessions recorded before per-message stamps,
    /// where they are the only times there are.
    #[must_use]
    pub fn duration_secs(&self) -> u64 {
        if self.first_ts != 0 && self.last_ts > self.first_ts {
            return self.last_ts - self.first_ts;
        }
        self.last_used.saturating_sub(self.created_at)
    }
}

/// Whether a user turn is plank-injected context rather than something the
/// user typed.
fn is_context_preamble(text: &str) -> bool {
    let t = text.trim_start();
    CONTEXT_PREFIXES.iter().any(|p| t.starts_with(p))
}

/// Whether a user turn is a real prompt: not a tool result, not injected
/// context, not empty.
fn is_human_turn(m: &Message) -> bool {
    m.role == Role::User
        && !m.text.trim().is_empty()
        && !m.is_tool_user()
        && !is_context_preamble(&m.text)
}

/// Language for a path's extension, if it maps to one.
fn language_of(path: &str) -> Option<&'static str> {
    let ext = Path::new(path).extension()?.to_str()?.to_ascii_lowercase();
    EXTENSION_TO_LANGUAGE
        .iter()
        .find(|(e, _)| *e == ext)
        .map(|(_, lang)| *lang)
}

/// Buckets one tool failure by its message.
///
/// The categories are plank's own failure modes, read off the `Tool error:`
/// strings the tools actually emit, rather than the reference's categories
/// (which describe a different tool set).
fn classify_error(text: &str) -> &'static str {
    let t = text.to_ascii_lowercase();
    if t.starts_with("exit status") {
        "Command failed"
    } else if t.contains("old text anchor not found") || t.contains("old not found") {
        "Edit anchor missing"
    } else if t.contains("old text anchor is not unique") {
        "Edit anchor ambiguous"
    } else if t.contains("invalid dsml tool call") || t.contains("unexpected dsml tag") {
        "Malformed tool call"
    } else if t.contains("unknown tool") || t.contains("missing tool name") {
        "Unknown tool"
    } else if t.contains("mcp") {
        "MCP unavailable"
    } else if t.contains("not found") || t.contains("no such") || t.contains("opendir failed") {
        "File not found"
    } else if t.contains("escapes workspace") || t.contains("blocked by") || t.contains("plan mode")
    {
        "Blocked by policy"
    } else if t.contains("requires") || t.contains("invalid") || t.contains("malformed") {
        "Bad arguments"
    } else if t.contains("failed") {
        "Command failed"
    } else {
        "Other"
    }
}

/// Counts `git commit` / `git push` in a shell command, ignoring the `git`
/// subcommands that merely mention them (`git log --grep=commit`).
fn git_activity(command: &str, meta: &mut SessionMeta) {
    for part in command.split(&['\n', ';', '&', '|'][..]) {
        let p = part.trim();
        if p.starts_with("git commit") || p.contains(" git commit") {
            meta.commits += 1;
        }
        if p.starts_with("git push") || p.contains(" git push") {
            meta.pushes += 1;
        }
    }
}

/// Line delta of one `edit` call, as `(added, removed)`.
fn edit_delta(old: &str, new: &str) -> (u32, u32) {
    let diff = similar::TextDiff::from_lines(old, new);
    let mut added = 0u32;
    let mut removed = 0u32;
    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Insert => added += 1,
            similar::ChangeTag::Delete => removed += 1,
            similar::ChangeTag::Equal => {}
        }
    }
    (added, removed)
}

/// Attributes one `Tool result N (name):` block, returning the tool name and
/// whether it failed.
///
/// A block is a failure when its payload opens with plank's `Tool error:`
/// convention, or when a shell result reports a non-zero exit status.
fn result_outcome(block: &str) -> Option<(String, bool, String)> {
    let (header, body) = block.split_once('\n').unwrap_or((block, ""));
    let name = header
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(n, _)| n.to_owned())?;
    let body = body.trim_start();
    if let Some(msg) = body.strip_prefix("Tool error:") {
        return Some((name, true, msg.trim().to_owned()));
    }
    // A bash job that ran but exited non-zero is a failure the model had to
    // recover from, even though the tool itself succeeded.
    if let Some(at) = body.find("exit_status=") {
        let code = body[at + "exit_status=".len()..]
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap_or("0");
        if code != "0" && !code.is_empty() {
            return Some((name, true, format!("exit status {code}")));
        }
    }
    Some((name, false, String::new()))
}

/// Splits a tool-result turn into its per-call `Tool result N (name):`
/// blocks.
fn result_blocks(payload: &str) -> Vec<&str> {
    let mut starts: Vec<usize> = Vec::new();
    for (i, line) in payload.match_indices("Tool result ") {
        let _ = line;
        if i == 0 || payload.as_bytes()[i - 1] == b'\n' {
            starts.push(i);
        }
    }
    if starts.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(starts.len());
    for (n, &s) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(payload.len());
        out.push(&payload[s..end]);
    }
    out
}

/// Folds the tool calls of one assistant turn into `meta`, appending any
/// newly touched file to `files`.
///
/// The calls come back out of the message text through the same DSML parse
/// the HTML export uses, so a stanza the exporter would not render as a tool
/// call is not counted as one here either.
fn count_tool_calls(m: &Message, meta: &mut SessionMeta, files: &mut Vec<String>) {
    for seg in crate::export::segments(std::slice::from_ref(m)) {
        let Segment::ToolCall(call) = seg else {
            continue;
        };
        let name = if call.name.is_empty() {
            "unknown".to_owned()
        } else {
            call.name.clone()
        };
        *meta.tools.entry(name.clone()).or_insert(0) += 1;

        let path = call.arg_value("path").unwrap_or("").to_owned();
        if !path.is_empty()
            && let Some(lang) = language_of(&path)
        {
            *meta.languages.entry(lang.to_owned()).or_insert(0) += 1;
        }
        let mut touched = || {
            if !path.is_empty() && !files.contains(&path) {
                files.push(path.clone());
            }
        };
        match name.as_str() {
            "edit" => {
                let (a, r) = edit_delta(
                    call.arg_value("old").unwrap_or(""),
                    call.arg_value("new").unwrap_or(""),
                );
                meta.lines_added += a;
                meta.lines_removed += r;
                touched();
            }
            "write" => {
                let content = call.arg_value("content").unwrap_or("");
                meta.lines_added += u32::try_from(content.lines().count()).unwrap_or(0);
                touched();
            }
            "bash" => git_activity(call.arg_value("command").unwrap_or(""), meta),
            _ => {}
        }
    }
}

/// Computes the statistics one session contributes.
///
/// Pure: given the same transcript it always produces the same numbers, which
/// is what makes disk caching safe and the unit tests meaningful.
#[must_use]
pub fn extract_meta(entry: &SessionEntry, mtime: u64, session: &Session) -> SessionMeta {
    let mut meta = SessionMeta {
        version: META_VERSION,
        id: entry.id.clone(),
        src_size: entry.file_size,
        src_mtime: mtime,
        title: entry.title.clone(),
        project: session.cwd.clone(),
        created_at: entry.created_at,
        last_used: entry.last_used,
        ..SessionMeta::default()
    };

    let mut files: Vec<String> = Vec::new();
    // Timestamp of the most recent model turn, so a user gap is measured from
    // when the model stopped talking rather than from the previous prompt.
    let mut last_assistant_ts: Option<u64> = None;

    for m in &session.transcript {
        meta.bytes += m.text.len() as u64;
        if m.at != 0 {
            if meta.first_ts == 0 {
                meta.first_ts = m.at;
            }
            meta.last_ts = m.at;
        }
        match m.role {
            Role::User if m.is_tool_user() => {
                meta.tool_results += 1;
                for block in result_blocks(m.tool_result_payload()) {
                    if let Some((name, failed, msg)) = result_outcome(block)
                        && failed
                    {
                        *meta
                            .errors
                            .entry(classify_error(&msg).to_owned())
                            .or_insert(0) += 1;
                        *meta.error_tools.entry(name).or_insert(0) += 1;
                    }
                }
            }
            Role::User if is_human_turn(m) => {
                meta.human_messages += 1;
                if meta.first_prompt.is_empty() {
                    meta.first_prompt = clip(m.text.trim(), 200);
                }
                if m.text.contains("[Request interrupted") {
                    meta.interruptions += 1;
                }
                if m.at != 0 {
                    meta.human_ts.push(m.at);
                    if let Some(prev) = last_assistant_ts {
                        let gap = m.at.saturating_sub(prev);
                        if (MIN_RESPONSE_SECS..=MAX_RESPONSE_SECS).contains(&gap) {
                            meta.response_secs.push(gap);
                        }
                    }
                }
            }
            Role::User => {}
            Role::Assistant => {
                meta.assistant_messages += 1;
                if m.at != 0 {
                    last_assistant_ts = Some(m.at);
                }
                count_tool_calls(m, &mut meta, &mut files);
            }
        }
    }
    meta.files = files;
    meta
}

/// Clips a string to `max` bytes on a char boundary, collapsing whitespace so
/// a multi-line prompt reads as one line.
fn clip(s: &str, max: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.len() <= max {
        return flat;
    }
    let mut end = max;
    while end > 0 && !flat.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &flat[..end])
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// The local timezone's offset from UTC in seconds, or 0 if it cannot be
/// determined.
///
/// Every timestamp in a session file is UTC, but "when do you work" and "how
/// many days active" are questions about the user's own clock, so the whole
/// report is bucketed at this offset. Read once per run rather than per
/// timestamp: a report spanning a DST boundary buckets a handful of messages
/// into the neighbouring hour, which is not worth a calendar dependency.
#[must_use]
pub fn local_utc_offset() -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let Ok(t) = libc::time_t::try_from(now) else {
        return 0;
    };
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&raw const t, &raw mut tm) }.is_null() {
        return 0;
    }
    tm.tm_gmtoff
}

/// Root of the report's working files, `~/.plank/usage-data`.
#[must_use]
pub fn usage_dir() -> PathBuf {
    SessionStore::default_dir()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("usage-data")
}

/// Path of the cached statistics for one session.
fn meta_cache_path(root: &Path, id: &str) -> PathBuf {
    root.join("session-meta").join(format!("{id}.json"))
}

/// Mtime of a file in unix seconds, or 0 when unavailable.
fn mtime_secs(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs())
}

/// Writes a file readable only by its owner, creating parents as needed.
///
/// The report and its caches quote prompts, file paths, and shell commands
/// verbatim; none of it belongs to anyone but the user, so nothing this
/// module writes is group- or world-readable.
fn write_private(path: &Path, body: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, body)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// Outcome of a scan: the statistics, or the user's decision to stop.
///
/// Cancellation is not an error. Everything already computed is still written
/// to the cache on the way past, so a scan abandoned halfway makes the next
/// run that much shorter rather than being wasted.
#[derive(Debug)]
pub enum Scan {
    /// Every session was read.
    Done(Vec<SessionMeta>),
    /// The user interrupted partway through.
    Cancelled,
}

/// Loads every session's statistics, computing and caching what is missing.
///
/// `progress` is called with `(done, total)` after each session so a long
/// first run can show movement. Sessions that fail to load are skipped: one
/// corrupt file must not cost the user the whole report.
///
/// `cancel` is polled once per session so a long first run over a large
/// history can be abandoned; it returns [`Scan::Cancelled`] rather than an
/// error, since a user who asked to stop has not hit a failure.
///
/// # Errors
///
/// Returns an error only if the session store cannot be listed at all.
pub fn collect_metas(
    store: &SessionStore,
    root: &Path,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &dyn Fn() -> bool,
) -> Result<Scan, String> {
    let entries = store.list().map_err(|e| e.to_string())?;
    let total = entries.len();
    let mut out = Vec::with_capacity(total);
    for (i, entry) in entries.iter().enumerate() {
        if cancel() {
            return Ok(Scan::Cancelled);
        }
        progress(i, total);
        let mtime = mtime_secs(&entry.path);
        let cache = meta_cache_path(root, &entry.id);
        let cached = std::fs::read_to_string(&cache)
            .ok()
            .and_then(|s| serde_json::from_str::<SessionMeta>(&s).ok())
            .filter(|m| m.matches(entry, mtime));
        if let Some(meta) = cached {
            out.push(meta);
            continue;
        }
        let Ok(session) = store.load(&entry.id) else {
            continue;
        };
        let meta = extract_meta(entry, mtime, &session);
        if let Ok(json) = serde_json::to_string(&meta) {
            let _ = write_private(&cache, &json);
        }
        out.push(meta);
    }
    progress(total, total);
    Ok(Scan::Done(out))
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Two sessions caught trading messages inside one [`OVERLAP_WINDOW_SECS`]
/// window — the user working on two things at once.
#[derive(Debug, Clone, Serialize)]
pub struct Overlap {
    /// The two session ids involved.
    pub a: String,
    /// The second session id.
    pub b: String,
    /// When the interleaving happened, unix seconds.
    pub at: u64,
}

/// Everything the report and the narrative prompts are built from.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Aggregated {
    /// Sessions found in the store.
    pub sessions_scanned: usize,
    /// Sessions substantive enough to count.
    pub sessions_counted: usize,
    /// Counted sessions that predate per-message timestamps.
    pub sessions_without_timing: usize,
    /// Earliest session start, unix seconds.
    pub first_session: u64,
    /// Latest session end, unix seconds.
    pub last_session: u64,
    /// Distinct calendar days with activity.
    pub days_active: usize,

    /// Human turns across all counted sessions.
    pub human_messages: u32,
    /// Model turns.
    pub assistant_messages: u32,
    /// Total session wall-clock, in hours.
    pub hours: f64,
    /// Lines added by edits and writes.
    pub lines_added: u32,
    /// Lines removed by edits.
    pub lines_removed: u32,
    /// Distinct files touched.
    pub files_touched: usize,
    /// `git commit` invocations.
    pub commits: u32,
    /// `git push` invocations.
    pub pushes: u32,

    /// Tool name to invocation count, descending.
    pub tools: Vec<(String, u32)>,
    /// Language to file touches, descending.
    pub languages: Vec<(String, u32)>,
    /// Project directory to session count, descending.
    pub projects: Vec<(String, u32)>,
    /// Error category to count, descending.
    pub errors: Vec<(String, u32)>,
    /// Tool name to failure count, descending.
    pub error_tools: Vec<(String, u32)>,
    /// Per-tool failure rate as a percentage, for tools used at least 10
    /// times, descending.
    pub error_rates: Vec<(String, f64)>,

    /// Response-time histogram, in [`RESPONSE_BUCKETS`] order.
    pub response_histogram: Vec<(String, u32)>,
    /// Median user response, seconds; 0 when nothing was timed.
    pub response_median: u64,
    /// Mean user response, seconds.
    pub response_mean: u64,
    /// Human messages per local hour of the day, indexed 0..24.
    pub by_hour: [u32; 24],
    /// Sessions overlapping in time.
    pub overlaps: Vec<Overlap>,
    /// Share of human messages sent while another session was live.
    pub overlap_share: f64,

    /// One line per counted session, newest first, for the narrative prompt.
    pub highlights: Vec<String>,

    /// What this installation already extends plank with, one line per
    /// extension point ("skills: release, review", "hooks: none"). Filled in
    /// by the caller from the live rosters, since none of it is derivable from
    /// the transcripts, and left empty when it could not be determined. The
    /// `features` section reads it so it recommends what is *missing* rather
    /// than what is already installed.
    pub extensions: Vec<String>,

    /// Feedback ratings (M2): total, positive, negative. These count only
    /// ratings that still have a subject; see `ratings_orphaned`.
    pub ratings_total: u32,
    pub ratings_positive: u32,
    pub ratings_negative: u32,
    /// Ratings whose turn no longer exists: the transcript was compacted or
    /// rolled back and renumbered, or the session file is gone. They are
    /// reported but never counted into the satisfaction statistics, and never
    /// reattributed to whatever now sits at the same index.
    pub ratings_orphaned: u32,
    /// Satisfaction over time: `(day, positive share)` per local day with a
    /// rating, newest first.
    pub satisfaction_by_day: Vec<(String, f64)>,
    /// Worst-rated turns: `(session id, note)` for negative ratings, newest
    /// first. The note is the user's own words; no transcript text is quoted.
    pub worst_turns: Vec<(String, String)>,
}

impl Aggregated {
    /// Whether the timing statistics describe the history as a whole rather
    /// than the fraction of it that happens to carry timestamps.
    ///
    /// A near-zero hour count over a history whose sessions mostly predate
    /// per-message stamps is not a small number, it is an absent one, and
    /// presenting it as a small number invites exactly the wrong conclusion
    /// ("you barely use this") from reader and model alike.
    #[must_use]
    pub fn timing_is_representative(&self) -> bool {
        self.sessions_counted > 0 && self.sessions_without_timing * 2 <= self.sessions_counted
    }
}

/// Sorts a `name -> count` map into a descending vector.
fn ranked(map: BTreeMap<String, u32>) -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = map.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

/// Local calendar day index for a unix second.
///
/// Uses the process's local-time offset rather than a calendar library: the
/// report only ever needs "which day" and "which hour", and both fall out of
/// the offset arithmetic that [`crate::context`] already relies on.
fn local_day(ts: u64, offset: i64) -> i64 {
    (ts.cast_signed() + offset).div_euclid(86_400)
}

/// Local hour (0-23) for a unix second.
fn local_hour(ts: u64, offset: i64) -> usize {
    usize::try_from((ts.cast_signed() + offset).rem_euclid(86_400) / 3600).unwrap_or(0)
}

/// Folds per-session statistics into the report's aggregate.
///
/// `tz_offset` is the local timezone's offset from UTC in seconds, passed in
/// rather than read from the environment so the tests are not at the mercy of
/// where they run.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn aggregate(metas: &[SessionMeta], tz_offset: i64) -> Aggregated {
    let mut agg = Aggregated {
        sessions_scanned: metas.len(),
        ..Aggregated::default()
    };
    let counted: Vec<&SessionMeta> = metas.iter().filter(|m| m.is_substantive()).collect();
    agg.sessions_counted = counted.len();

    let mut tools: BTreeMap<String, u32> = BTreeMap::new();
    let mut languages: BTreeMap<String, u32> = BTreeMap::new();
    let mut projects: BTreeMap<String, u32> = BTreeMap::new();
    let mut errors: BTreeMap<String, u32> = BTreeMap::new();
    let mut error_tools: BTreeMap<String, u32> = BTreeMap::new();
    let mut files: Vec<&str> = Vec::new();
    let mut days: Vec<i64> = Vec::new();
    let mut responses: Vec<u64> = Vec::new();

    for m in &counted {
        if m.human_ts.is_empty() {
            agg.sessions_without_timing += 1;
        }
        agg.human_messages += m.human_messages;
        agg.assistant_messages += m.assistant_messages;
        agg.lines_added += m.lines_added;
        agg.lines_removed += m.lines_removed;
        agg.commits += m.commits;
        agg.pushes += m.pushes;
        #[allow(clippy::cast_precision_loss)] // hours, to one decimal place
        {
            agg.hours += m.duration_secs() as f64 / 3600.0;
        }

        if agg.first_session == 0 || m.created_at < agg.first_session {
            agg.first_session = m.created_at;
        }
        agg.last_session = agg.last_session.max(m.last_used);

        for (k, v) in &m.tools {
            *tools.entry(k.clone()).or_insert(0) += v;
        }
        for (k, v) in &m.languages {
            *languages.entry(k.clone()).or_insert(0) += v;
        }
        for (k, v) in &m.errors {
            *errors.entry(k.clone()).or_insert(0) += v;
        }
        for (k, v) in &m.error_tools {
            *error_tools.entry(k.clone()).or_insert(0) += v;
        }
        let project = if m.project.is_empty() {
            UNRECORDED_PROJECT
        } else {
            m.project.as_str()
        };
        *projects.entry(project.to_owned()).or_insert(0) += 1;
        for f in &m.files {
            if !files.contains(&f.as_str()) {
                files.push(f);
            }
        }
        responses.extend(m.response_secs.iter().copied());
        for &ts in &m.human_ts {
            agg.by_hour[local_hour(ts, tz_offset)] += 1;
            let day = local_day(ts, tz_offset);
            if !days.contains(&day) {
                days.push(day);
            }
        }
        // A session with no stamps still happened on a day; fall back to its
        // start so "days active" is not skewed by the format change.
        if m.human_ts.is_empty() && m.created_at != 0 {
            let day = local_day(m.created_at, tz_offset);
            if !days.contains(&day) {
                days.push(day);
            }
        }
    }

    agg.files_touched = files.len();
    agg.days_active = days.len();
    agg.error_rates = tools
        .iter()
        .filter(|&(_, &n)| n >= 10)
        .filter_map(|(name, &n)| {
            let failures = error_tools.get(name).copied()?;
            Some((name.clone(), f64::from(failures) * 100.0 / f64::from(n)))
        })
        .collect();
    agg.error_rates
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    agg.tools = ranked(tools);
    agg.languages = ranked(languages);
    agg.projects = ranked(projects);
    agg.errors = ranked(errors);
    agg.error_tools = ranked(error_tools);

    responses.sort_unstable();
    if !responses.is_empty() {
        agg.response_median = responses[responses.len() / 2];
        agg.response_mean = responses.iter().sum::<u64>() / responses.len() as u64;
    }
    agg.response_histogram = RESPONSE_BUCKETS
        .iter()
        .scan(MIN_RESPONSE_SECS, |low, &(high, label)| {
            let n = responses.iter().filter(|&&r| r >= *low && r < high).count();
            *low = high;
            Some((label.to_owned(), u32::try_from(n).unwrap_or(u32::MAX)))
        })
        .collect();

    let (overlaps, overlapped) = detect_overlaps(&counted);
    agg.overlaps = overlaps;
    agg.overlap_share = if agg.human_messages == 0 {
        0.0
    } else {
        f64::from(overlapped) * 100.0 / f64::from(agg.human_messages)
    };

    let mut by_recency: Vec<&&SessionMeta> = counted.iter().collect();
    by_recency.sort_by_key(|m| std::cmp::Reverse(m.last_used));
    agg.highlights = by_recency
        .iter()
        .take(50)
        .map(|m| {
            let prompt = if m.first_prompt.is_empty() {
                clip(&m.title, 120)
            } else {
                clip(&m.first_prompt, 120)
            };
            format!(
                "{} turns, {} tool calls, {} edits: {prompt}",
                m.human_messages,
                m.tools.values().sum::<u32>(),
                m.lines_added + m.lines_removed
            )
        })
        .collect();

    agg
}

/// Fills the feedback fields of an [`Aggregated`] from every session's rating
/// sidecar (M2). Ratings never enter the transcript, model context, or KV —
/// they are facts *about* sessions, consumed only here.
///
/// Satisfaction is the positive share per local day; worst-rated turns are the
/// user's own notes on negative ratings, never transcript text.
pub fn aggregate_feedback(
    agg: &mut Aggregated,
    feedback: &[(String, Vec<crate::feedback::Rating>)],
    tz_offset: i64,
    transcript_of: &dyn Fn(&str) -> Option<Vec<crate::session::Message>>,
) {
    let mut by_day: BTreeMap<i64, (u32, u32)> = BTreeMap::new();
    let mut worst: Vec<(String, String)> = Vec::new();
    for (session_id, ratings) in feedback {
        // One load per rated session, not per rating: `/insights` reads the
        // whole history and most sessions carry no ratings at all.
        let transcript = transcript_of(session_id);
        for r in ratings {
            // A rating whose turn is gone is reported as orphaned and kept out
            // of the statistics. A session whose file no longer exists has no
            // subject either, so its ratings are orphaned too.
            let orphaned = transcript
                .as_ref()
                .is_none_or(|t| crate::feedback::is_orphaned(t, r));
            if orphaned {
                agg.ratings_orphaned += 1;
                continue;
            }
            agg.ratings_total += 1;
            if r.positive {
                agg.ratings_positive += 1;
            } else {
                agg.ratings_negative += 1;
                worst.push((session_id.clone(), r.note.clone()));
            }
            let day = local_day(r.at, tz_offset);
            let (pos, neg) = by_day.entry(day).or_insert((0, 0));
            if r.positive {
                *pos += 1;
            } else {
                *neg += 1;
            }
        }
    }
    let mut days: Vec<(i64, (u32, u32))> = by_day.into_iter().collect();
    days.sort_by_key(|(d, _)| std::cmp::Reverse(*d));
    agg.satisfaction_by_day = days
        .into_iter()
        .map(|(day, (pos, neg))| {
            let total = pos + neg;
            let share = if total == 0 {
                0.0
            } else {
                f64::from(pos) / f64::from(total)
            };
            (
                date_str(u64::try_from(day).unwrap_or(0) * 86_400, tz_offset),
                share,
            )
        })
        .collect();
    worst.sort_by_key(|(_, note)| note.clone());
    agg.worst_turns = worst;
}

/// Finds sessions whose messages interleave, and counts how many messages
/// were sent while another session was live.
///
/// Every human message from every session is merged into one timeline; an
/// overlap is the reference's A → B → A pattern inside a
/// [`OVERLAP_WINDOW_SECS`] window, which is stricter than mere adjacency (two
/// sessions run back to back produce A → B, but only genuinely concurrent
/// work makes the user come *back* to A).
fn detect_overlaps(metas: &[&SessionMeta]) -> (Vec<Overlap>, u32) {
    let mut timeline: Vec<(u64, &str)> = Vec::new();
    for m in metas {
        for &ts in &m.human_ts {
            timeline.push((ts, m.id.as_str()));
        }
    }
    timeline.sort_by_key(|&(ts, _)| ts);

    let mut overlaps: Vec<Overlap> = Vec::new();
    let mut overlapped = 0u32;
    for w in timeline.windows(3) {
        let [(t0, a), (_, b), (t2, c)] = [w[0], w[1], w[2]];
        if a == b || a != c || t2.saturating_sub(t0) > OVERLAP_WINDOW_SECS {
            continue;
        }
        overlapped += 3;
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        // One pair, however often the user bounced between them, keeps the
        // list a list of *relationships* rather than of individual messages.
        if !overlaps.iter().any(|o| o.a == lo && o.b == hi) {
            overlaps.push(Overlap {
                a: lo.to_owned(),
                b: hi.to_owned(),
                at: t0,
            });
        }
    }
    (overlaps, overlapped)
}

// ---------------------------------------------------------------------------
// Narrative
// ---------------------------------------------------------------------------

/// One model-written section of the report.
#[derive(Debug)]
pub struct SectionSpec {
    /// Stable key, used as the field name in the narrative and the HTML id.
    pub key: &'static str,
    /// Heading shown in the report.
    pub heading: &'static str,
    /// Instructions appended to the shared data context.
    pub prompt: &'static str,
}

/// System prompt for the narrative calls.
///
/// The agent's own system prompt is wrong for this work: it describes a tool
/// user operating on a codebase, and under it a request for a bare JSON
/// object comes back as reasoning and tool calls. The reference solves the
/// same problem the same way — an empty system prompt and no tools — and this
/// is the smallest framing that still says what the job is.
pub const ANALYST_SYSTEM: &str = "You are a careful analyst. You answer with a single JSON \
object and nothing else: no preamble, no explanation, no markdown fence, no tool calls.";

/// plank's extension points, as the `features` section is told about them.
///
/// The model has no reliable knowledge of plank itself, so a recommendation to
/// "add a skill" is only actionable if the prompt says what a skill *is* and
/// where it goes. Kept in sync with `user-guide/09-extending.md`: paths that
/// drift here produce snippets that write files plank never reads.
///
/// A macro rather than a plain `const` because the `features` prompt splices it
/// with `concat!`, which takes literals only.
macro_rules! feature_catalogue {
    () => {
        "\
- Skills: a directory `~/.plank/skills/<name>/SKILL.md` (or `./.plank/skills/`) \
holding a named procedure with `name`/`description`/`argument-hint` frontmatter \
and `$ARGUMENTS` in the body. Runnable as `/<name>` and reachable by the model \
itself. Best for a multi-step routine repeated across sessions.
- Templates: one file `~/.plank/templates/<name>.md` whose body has `{{named}}` \
holes. Lighter than a skill: it saves retyping a prompt shape, and the model \
cannot invoke it.
- Subagents: `~/.plank/agents/<name>.md`, dispatched with `/subagent:<name> \
<task>`; only the final report enters the transcript. Frontmatter can add \
`isolation: worktree` for its own checkout, or `provider:` to run it on another \
engine. Best for delegated investigation and for fan-out over independent work.
- Hooks: `~/.plank/hooks.json` or `./.plank/hooks.json`, shell commands bound to \
events (`PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `SessionStart`, `Stop`) \
with a tool-name matcher. Best for a check that should run every time rather \
than when remembered — formatting, linting, a guard on a dangerous command.
- MCP servers: `~/.plank/.mcp.json` or `./.mcp.json`, stdio servers whose tools \
join the roster. Best for reaching a system plank has no built-in tool for.
- AGENTS.md: project instructions read at session start. Best for facts and \
conventions repeated to the model by hand.\n"
    };
}

/// The catalogue as a string, for tests and for anything that wants to show it.
pub const FEATURE_CATALOGUE: &str = feature_catalogue!();

/// The prose sections, each one independent model call.
///
/// Every prompt ends the same way — one JSON object, named keys, no prose
/// around it — because the reply is parsed by pulling the first `{...}` out
/// of the text. Failure of any one section costs that section and nothing
/// else.
pub const SECTIONS: &[SectionSpec] = &[
    SectionSpec {
        key: "how_you_work",
        heading: "How you work",
        prompt: "Describe how this person works with the agent: the rhythm of their \
sessions, how much they delegate versus steer, and what their tool mix says \
about their habits. Two short paragraphs, second person, concrete. Do not \
quote numbers back at them and do not use the raw snake_case names.\n\
RESPOND WITH ONLY A VALID JSON OBJECT: {\"paragraphs\": [\"...\", \"...\"], \
\"key_pattern\": \"one sentence naming their single most characteristic habit\"}",
    },
    SectionSpec {
        key: "what_works",
        heading: "What is working",
        prompt: "Name three things this person does that are working well, based on the \
session summaries and the tool and outcome mix. Be specific about the \
behaviour, not generic praise.\n\
RESPOND WITH ONLY A VALID JSON OBJECT: {\"items\": [{\"title\": \"short title\", \
\"detail\": \"two sentences\"}]}",
    },
    SectionSpec {
        key: "friction",
        heading: "Where you lose time",
        prompt: "Identify the three biggest sources of friction, grounded in the error \
categories, failure rates, and interruptions. For each, say what is going \
wrong and what would reduce it.\n\
RESPOND WITH ONLY A VALID JSON OBJECT: {\"items\": [{\"title\": \"short title\", \
\"detail\": \"what is going wrong\", \"fix\": \"what to change\"}]}",
    },
    SectionSpec {
        key: "suggestions",
        heading: "Try this next",
        prompt: "Suggest changes this person could make. `agents_md` should hold \
instructions they clearly repeat across sessions and should therefore write \
down in AGENTS.md once. `prompts` should hold ready-to-paste prompts that fit \
work they actually do.\n\
RESPOND WITH ONLY A VALID JSON OBJECT: {\"agents_md\": [\"instruction\"], \
\"prompts\": [{\"title\": \"short title\", \"prompt\": \"the prompt to paste\"}]}",
    },
    SectionSpec {
        key: "features",
        heading: "Features to try",
        prompt: concat!(
            "Recommend two or three of plank's own extension points this person is not \
using but clearly would benefit from, grounded in their sessions, tool mix and \
errors. Skip anything `extensions_in_use` says they already have unless you are \
recommending a specific new one of that kind. Here is the catalogue, with where \
each one lives:\n",
            feature_catalogue!(),
            "\n`why` must name the evidence in their data. `snippet` must be a complete, \
ready-to-run shell command or file body that sets the thing up for the work they \
actually do — no placeholders like <your project>.\n\
RESPOND WITH ONLY A VALID JSON OBJECT: {\"features\": [{\"name\": \"feature name\", \
\"one_liner\": \"what it is, one sentence\", \"why\": \"why it fits this person\", \
\"snippet\": \"the command or file body to set it up\"}]}",
        ),
    },
    SectionSpec {
        key: "at_a_glance",
        heading: "At a glance",
        prompt: "Write the report's opening summary for this person, in a coaching tone, \
two or three sentences per field. Do not cite numbers and do not use \
snake_case names.\n\
RESPOND WITH ONLY A VALID JSON OBJECT: {\"working\": \"what is going well\", \
\"hindering\": \"what is holding them back\", \"quick_win\": \"the one change \
worth making this week\"}",
    },
];

/// Model-written prose, keyed by [`SectionSpec::key`].
///
/// A missing key means that section's call failed or returned unusable text;
/// the renderer simply omits it.
pub type Narrative = BTreeMap<String, serde_json::Value>;

/// The data context shared by every section prompt.
///
/// Deliberately headline-only: the model is asked to interpret, and a full
/// dump of every session would both blow the context and invite it to
/// recompute statistics that are already exact.
///
/// A field that is not *meaningful* is left out rather than sent as a zero or
/// a placeholder. This matters more than it sounds: a history recorded before
/// per-message timestamps reports almost no hours, and one recorded before
/// project directories reports every session under `(unrecorded)`. Sent as-is,
/// the model spends its reasoning litigating whether the numbers can be right
/// — correctly, since they cannot — instead of answering. An absent field asks
/// nothing of it.
#[must_use]
pub fn narrative_context(agg: &Aggregated) -> String {
    let top = |v: &[(String, u32)], n: usize| -> Vec<serde_json::Value> {
        v.iter()
            .take(n)
            .map(|(k, c)| serde_json::json!({ "name": k, "count": c }))
            .collect()
    };
    let mut ctx = serde_json::Map::new();
    let mut put = |k: &str, v: serde_json::Value| {
        ctx.insert(k.to_owned(), v);
    };
    put("sessions", agg.sessions_counted.into());
    put("days_active", agg.days_active.into());
    put("human_messages", agg.human_messages.into());
    put("lines_added", agg.lines_added.into());
    put("lines_removed", agg.lines_removed.into());
    put("files_touched", agg.files_touched.into());
    put("commits", agg.commits.into());
    put("top_tools", top(&agg.tools, 12).into());
    put("languages", top(&agg.languages, 8).into());
    put("error_categories", top(&agg.errors, 8).into());
    put("failing_tools", top(&agg.error_tools, 8).into());
    put("recent_sessions", agg.highlights.clone().into());
    // Only when the rosters were actually read: an empty list would read as
    // "nothing installed" and invite recommending what is already there.
    if !agg.extensions.is_empty() {
        put("extensions_in_use", agg.extensions.clone().into());
    }

    // Timing exists only for sessions recorded since per-message stamps, and
    // is sent only when enough of them carry it to mean anything.
    if agg.timing_is_representative() {
        if agg.hours >= 0.05 {
            put("hours", ((agg.hours * 10.0).round() / 10.0).into());
        }
        if agg.response_median > 0 {
            put("median_response_secs", agg.response_median.into());
            put("concurrent_session_pairs", agg.overlaps.len().into());
        }
    }
    // Nothing to say about projects when none of them were recorded.
    if agg
        .projects
        .iter()
        .any(|(name, _)| name != UNRECORDED_PROJECT)
    {
        put("projects", top(&agg.projects, 8).into());
    }
    serde_json::to_string_pretty(&serde_json::Value::Object(ctx)).unwrap_or_default()
}

/// One `extensions_in_use` line: what is installed at an extension point, or
/// that nothing is.
///
/// "none" is said explicitly rather than by omission — an absent line reads as
/// "not known", and the `features` section's whole job is to tell those two
/// apart before it recommends anything.
#[must_use]
pub fn extension_line(kind: &str, names: &[String]) -> String {
    if names.is_empty() {
        return format!("{kind}: none");
    }
    let shown: Vec<&str> = names.iter().take(12).map(String::as_str).collect();
    let more = names.len().saturating_sub(shown.len());
    let mut line = format!("{kind}: {}", shown.join(", "));
    if more > 0 {
        let _ = write!(line, " (+{more} more)");
    }
    line
}

/// Builds the full prompt for one section.
#[must_use]
pub fn section_prompt(spec: &SectionSpec, context: &str) -> String {
    format!(
        "Here is a summary of how one developer used a coding agent, as JSON.\n\n\
{context}\n\n{}\n",
        spec.prompt
    )
}

/// Turns a section's streaming reply into displayable lines of the model's
/// thinking.
///
/// A section takes long enough that silence reads as a hang, so the reasoning
/// is shown while it happens, the way an ordinary turn shows it. Only the
/// thinking is displayed: what follows `</think>` is the JSON answer, which is
/// parsed rather than read.
///
/// Local engines open the think block implicitly in the prefill, so the stream
/// *starts* inside it and `</think>` is the only marker that ever arrives —
/// the same asymmetry [`extract_json`] relies on.
#[derive(Debug, Default)]
pub struct ThinkTicker {
    pending: String,
    done: bool,
}

/// Width at which unbroken reasoning is wrapped into a tick.
///
/// Not a maximum: a line the model ends itself is emitted whole, however long.
/// This threshold exists only so that a model reasoning in one long paragraph
/// — no newlines at all — still produces visible movement rather than one
/// enormous line at the very end.
const TICK_WIDTH: usize = 88;

impl ThinkTicker {
    /// A ticker positioned at the start of a reply, inside the think block.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one streamed chunk, returning whatever complete lines it
    /// finished. Returns nothing once the answer has begun.
    pub fn feed(&mut self, chunk: &str) -> Vec<String> {
        if self.done {
            return Vec::new();
        }
        let mut text = chunk;
        if let Some(end) = text.find("</think>") {
            text = &text[..end];
            self.done = true;
        }
        self.pending.push_str(text);

        let mut out = Vec::new();
        loop {
            // A newline ends a line; failing that, wrap at the last word
            // boundary before the width so long reasoning still ticks.
            let cut = self.pending.find('\n').or_else(|| {
                (self.pending.len() > TICK_WIDTH)
                    .then(|| {
                        // TICK_WIDTH is a byte index and may land inside a
                        // multi-byte char; step back to a char boundary
                        // before slicing.
                        let mut end = TICK_WIDTH.min(self.pending.len());
                        while end > 0 && !self.pending.is_char_boundary(end) {
                            end -= 1;
                        }
                        self.pending[..end]
                            .rfind(char::is_whitespace)
                            .filter(|&i| i > 0)
                    })
                    .flatten()
            });
            let Some(cut) = cut else { break };
            let line = self.pending[..cut].trim().to_owned();
            self.pending.drain(..=cut.min(self.pending.len() - 1));
            if !line.is_empty() {
                out.push(line);
            }
        }
        if self.done {
            let tail = std::mem::take(&mut self.pending).trim().to_owned();
            if !tail.is_empty() {
                out.push(tail);
            }
        }
        out
    }
}

/// Watches a section's stream for the model falling into a repetition loop.
///
/// The recommendation sections ask for a list of objects, and a small model
/// that runs out of things to say sometimes keeps emitting the same clause or
/// the same list entry until the token budget runs out. That is minutes of
/// visible nonsense per section on a real engine, so the stream is watched and
/// the pass stopped as soon as the tail is provably cyclic. A false positive
/// costs one section of prose; the statistics and every other section are
/// untouched.
#[derive(Debug, Default)]
pub struct RepeatGuard {
    tail: String,
    since_check: usize,
}

/// Bytes of trailing output examined for a cycle.
const REPEAT_WINDOW: usize = 1024;
/// Shortest cycle treated as a loop. Below this, ordinary prose repeats
/// itself legitimately — "the the", a run of list punctuation.
const REPEAT_MIN_PERIOD: usize = 12;
/// Consecutive identical blocks needed before the pass is stopped.
const REPEAT_CYCLES: usize = 4;
/// Bytes generated between checks. Scanning every chunk would be wasted work;
/// a loop is not urgent enough to catch within one token.
const REPEAT_CHECK_EVERY: usize = 64;

impl RepeatGuard {
    /// A guard positioned at the start of a reply.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one streamed chunk. Returns true once the output has been
    /// repeating itself for [`REPEAT_CYCLES`] cycles.
    pub fn feed(&mut self, chunk: &str) -> bool {
        self.tail.push_str(chunk);
        if self.tail.len() > REPEAT_WINDOW {
            // Trim from the front to a char boundary: the window is a byte
            // budget, and slicing mid-character would panic.
            let mut cut = self.tail.len() - REPEAT_WINDOW;
            while cut < self.tail.len() && !self.tail.is_char_boundary(cut) {
                cut += 1;
            }
            self.tail.drain(..cut);
        }
        self.since_check += chunk.len();
        if self.since_check < REPEAT_CHECK_EVERY {
            return false;
        }
        self.since_check = 0;
        self.is_cyclic()
    }

    /// Whether the tail ends in the same block repeated back to back.
    fn is_cyclic(&self) -> bool {
        let bytes = self.tail.as_bytes();
        let len = bytes.len();
        for period in REPEAT_MIN_PERIOD..=len / REPEAT_CYCLES {
            let block = &bytes[len - period..];
            if (1..REPEAT_CYCLES)
                .all(|back| &bytes[len - period * (back + 1)..len - period * back] == block)
            {
                return true;
            }
        }
        false
    }
}

/// Pulls the answer object out of a model reply.
///
/// A reply that never yields valid JSON is a dropped section rather than an
/// error, so this is deliberately forgiving about what surrounds the object —
/// but not about the object itself. Two things make the naive "first `{`"
/// rule wrong here:
///
/// - The engine opens a `<think>` block implicitly, so the reply usually
///   *starts* with reasoning, and reasoning about a JSON schema is full of
///   braces. Everything up to the last `</think>` is dropped first.
/// - The model may still open a brace it does not close, or write a snippet
///   before the real answer, so every remaining `{` is tried in turn and the
///   first one that parses wins.
#[must_use]
pub fn extract_json(text: &str) -> Option<serde_json::Value> {
    let after_think = text
        .rfind("</think>")
        .map_or(text, |i| &text[i + "</think>".len()..]);
    // A reply that is all thinking and no answer still deserves a look: the
    // model sometimes closes no think tag at all.
    for candidate in [after_think, text] {
        if let Some(v) = first_json_object(candidate) {
            return Some(v);
        }
    }
    None
}

/// The first balanced, parseable `{...}` object in `text`.
fn first_json_object(text: &str) -> Option<serde_json::Value> {
    let mut from = 0usize;
    while let Some(rel) = text[from..].find('{') {
        let start = from + rel;
        if let Some(v) = balanced_object_at(text, start) {
            return Some(v);
        }
        from = start + 1;
    }
    None
}

/// Parses the balanced object beginning at `start`, if it is both balanced
/// and valid JSON.
fn balanced_object_at(text: &str, start: usize) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&text[start..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Formats a count with thousands separators.
fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Formats a duration in seconds as a short human string.
fn dur(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// A `YYYY-MM-DD` date for a unix second at the given offset.
pub(crate) fn date_str(ts: u64, offset: i64) -> String {
    // Civil-from-days, Howard Hinnant's algorithm: the report needs calendar
    // dates and plank has no date dependency to lean on.
    let days = local_day(ts, offset);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// A `YYYY-MM-DD HH:MM` local wall-clock stamp for a unix second.
fn datetime_str(ts: u64, offset: i64) -> String {
    let secs = (ts.cast_signed() + offset).rem_euclid(86_400);
    format!(
        "{} {:02}:{:02}",
        date_str(ts, offset),
        secs / 3600,
        (secs % 3600) / 60
    )
}

/// The same instant as a filename-safe `YYYYMMDD-HHMM`.
fn file_stamp(ts: u64, offset: i64) -> String {
    datetime_str(ts, offset)
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == ' ')
        .collect::<String>()
        .replace(' ', "-")
}

/// The wall-clock second a report is being generated at.
#[must_use]
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Condensed report for the terminal: the numbers that fit on a screen, plus
/// whichever narrative made it back.
// A render function; the length is not complexity.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn render_summary(
    agg: &Aggregated,
    narrative: &Narrative,
    tz_offset: i64,
    generated_at: u64,
) -> Vec<String> {
    let mut out = vec![format!(
        "generated {}",
        datetime_str(generated_at, tz_offset)
    )];
    if agg.sessions_counted == 0 {
        out.push("No sessions substantial enough to report on yet.".to_owned());
        return out;
    }
    out.push(format!(
        "{} sessions ({} scanned) from {} to {}, {} days active",
        commas(agg.sessions_counted as u64),
        commas(agg.sessions_scanned as u64),
        date_str(agg.first_session, tz_offset),
        date_str(agg.last_session, tz_offset),
        agg.days_active
    ));
    // Dropped rather than reported small when most of the history predates
    // per-message stamps: an unrepresentative total misleads more than a
    // missing one.
    let hours = if agg.timing_is_representative() && agg.hours >= 0.05 {
        format!("{:.1}h in session, ", agg.hours)
    } else {
        String::new()
    };
    out.push(format!(
        "{} prompts, {hours}+{} / -{} lines across {} files, {} commits",
        commas(u64::from(agg.human_messages)),
        commas(u64::from(agg.lines_added)),
        commas(u64::from(agg.lines_removed)),
        commas(agg.files_touched as u64),
        agg.commits
    ));
    if !agg.tools.is_empty() {
        let tools = agg
            .tools
            .iter()
            .take(5)
            .map(|(n, c)| format!("{n} {c}"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!("top tools: {tools}"));
    }
    if agg.response_median > 0 {
        out.push(format!(
            "median reply time {}, mean {}",
            dur(agg.response_median),
            dur(agg.response_mean)
        ));
    }
    if !agg.errors.is_empty() {
        let errs = agg
            .errors
            .iter()
            .take(3)
            .map(|(n, c)| format!("{n} {c}"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!("top friction: {errs}"));
    }
    if !agg.overlaps.is_empty() {
        out.push(format!(
            "{} session pairs run concurrently ({:.0}% of prompts)",
            agg.overlaps.len(),
            agg.overlap_share
        ));
    }
    for field in ["working", "hindering", "quick_win"] {
        if let Some(text) = narrative
            .get("at_a_glance")
            .and_then(|v| v.get(field))
            .and_then(serde_json::Value::as_str)
        {
            out.push(format!("{field}: {text}"));
        }
    }
    // Feedback (M2): satisfaction over time and the worst-rated turns. The
    // ratings are sidecar facts, never transcript text.
    if agg.ratings_total > 0 {
        let pct = f64::from(agg.ratings_positive) * 100.0 / f64::from(agg.ratings_total);
        out.push(format!(
            "{} ratings: {} positive ({:.0}%), {} negative",
            agg.ratings_total, agg.ratings_positive, pct, agg.ratings_negative
        ));
        let days = agg
            .satisfaction_by_day
            .iter()
            .take(5)
            .map(|(d, s)| format!("{d} {:.0}%", s * 100.0))
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!("satisfaction by day: {days}"));
        if !agg.worst_turns.is_empty() {
            let worst = agg
                .worst_turns
                .iter()
                .take(5)
                .map(|(id, note)| {
                    if note.is_empty() {
                        id.clone()
                    } else {
                        format!("{id}: {note}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            out.push(format!("worst-rated turns: {worst}"));
        }
    }
    // Reported even when every rating is orphaned and the block above is
    // skipped: silently dropping them would look like the ratings vanished.
    if agg.ratings_orphaned > 0 {
        out.push(format!(
            "{} orphaned (the rated turn is gone; not counted above)",
            agg.ratings_orphaned
        ));
    }
    out
}

/// A horizontal bar chart as plain divs, widest bar full width.
fn bar_chart(rows: &[(String, u32)], limit: usize, accent: &str) -> String {
    let max = rows
        .iter()
        .take(limit)
        .map(|r| r.1)
        .max()
        .unwrap_or(1)
        .max(1);
    let mut out = String::from("<div class=\"chart\">");
    for (name, count) in rows.iter().take(limit) {
        let pct = f64::from(*count) * 100.0 / f64::from(max);
        let _ = write!(
            out,
            "<div class=\"row\"><div class=\"label\">{}</div>\
<div class=\"track\"><div class=\"bar\" style=\"width:{pct:.1}%;background:{accent}\"></div></div>\
<div class=\"value\">{}</div></div>",
            html_escape(name),
            commas(u64::from(*count))
        );
    }
    out.push_str("</div>");
    out
}

/// A section wrapper, emitted only when there is something to put in it.
fn section(out: &mut String, id: &str, heading: &str, body: &str) {
    if body.trim().is_empty() {
        return;
    }
    let _ = write!(
        out,
        "<section id=\"{id}\"><h2>{}</h2>{body}</section>",
        html_escape(heading)
    );
}

/// Writes one copyable block: the text itself, a Copy button, and — when
/// `pick` — a checkbox the section's "Copy all checked" button collects.
///
/// The report is a file, not a terminal, so a suggestion is only as useful as
/// it is easy to act on: everything the model proposes is something you paste
/// back into plank, and a paste starts with a copy. `tag` is the element the
/// text lives in (`pre` for a snippet, `p` for a sentence of prose).
fn copy_row(out: &mut String, tag: &str, class: &str, text: &str, pick: bool) {
    out.push_str("<div class=\"copyrow\">");
    if pick {
        out.push_str("<input type=\"checkbox\" class=\"pick\" checked>");
    }
    let _ = write!(
        out,
        "<{tag} class=\"{class}\" data-copy>{}</{tag}>\
<button type=\"button\" class=\"copy\">Copy</button></div>",
        html_escape(text)
    );
}

/// The "Copy all checked" control for a section of picked suggestions.
fn copy_all_row(out: &mut String) {
    out.push_str(
        "<div class=\"copyall\"><button type=\"button\" class=\"copy-all\">\
Copy all checked</button></div>",
    );
}

/// Renders the narrative sections that came back, in [`SECTIONS`] order and
/// skipping `at_a_glance` (which the header already shows).
fn narrative_html(narrative: &Narrative) -> String {
    let mut out = String::new();
    for spec in SECTIONS {
        if spec.key == "at_a_glance" {
            continue;
        }
        let Some(value) = narrative.get(spec.key) else {
            continue;
        };
        let mut body = String::new();
        if let Some(paras) = value
            .get("paragraphs")
            .and_then(serde_json::Value::as_array)
        {
            for p in paras.iter().filter_map(serde_json::Value::as_str) {
                let _ = write!(body, "<p>{}</p>", html_escape(p));
            }
        }
        if let Some(key) = value.get("key_pattern").and_then(serde_json::Value::as_str) {
            let _ = write!(body, "<p class=\"callout\">{}</p>", html_escape(key));
        }
        if let Some(items) = value.get("items").and_then(serde_json::Value::as_array) {
            body.push_str("<div class=\"cards\">");
            for item in items {
                let title = item.get("title").and_then(serde_json::Value::as_str);
                let detail = item.get("detail").and_then(serde_json::Value::as_str);
                let fix = item.get("fix").and_then(serde_json::Value::as_str);
                body.push_str("<div class=\"card\">");
                if let Some(t) = title {
                    let _ = write!(body, "<h3>{}</h3>", html_escape(t));
                }
                if let Some(d) = detail {
                    let _ = write!(body, "<p>{}</p>", html_escape(d));
                }
                if let Some(f) = fix {
                    let _ = write!(body, "<p class=\"fix\">{}</p>", html_escape(f));
                }
                body.push_str("</div>");
            }
            body.push_str("</div>");
        }
        if let Some(features) = value.get("features").and_then(serde_json::Value::as_array) {
            body.push_str(
                "<p class=\"note\">Paste a snippet into plank and it will set it up.</p>",
            );
            for f in features {
                let name = f.get("name").and_then(serde_json::Value::as_str);
                let one_liner = f.get("one_liner").and_then(serde_json::Value::as_str);
                let why = f.get("why").and_then(serde_json::Value::as_str);
                let snippet = f.get("snippet").and_then(serde_json::Value::as_str);
                body.push_str("<div class=\"card feature\">");
                if let Some(n) = name {
                    let _ = write!(body, "<h3>{}</h3>", html_escape(n));
                }
                if let Some(o) = one_liner {
                    let _ = write!(body, "<p>{}</p>", html_escape(o));
                }
                if let Some(w) = why {
                    let _ = write!(
                        body,
                        "<p class=\"fix\"><b>Why for you.</b> {}</p>",
                        html_escape(w)
                    );
                }
                if let Some(s) = snippet {
                    copy_row(&mut body, "pre", "snippet", s, false);
                }
                body.push_str("</div>");
            }
        }
        if let Some(lines) = value.get("agents_md").and_then(serde_json::Value::as_array) {
            // A checklist rather than a bullet list: these are instructions
            // meant to end up in a file, so the report hands them over ready
            // to paste — untick the ones you disagree with, copy the rest.
            body.push_str(
                "<h3>Worth putting in AGENTS.md</h3>\
<p class=\"note\">Untick anything you disagree with, then paste the rest into \
plank and ask it to add them to AGENTS.md.</p>",
            );
            copy_all_row(&mut body);
            for l in lines.iter().filter_map(serde_json::Value::as_str) {
                copy_row(&mut body, "p", "instruction", l, true);
            }
        }
        if let Some(prompts) = value.get("prompts").and_then(serde_json::Value::as_array) {
            body.push_str("<h3>Prompts to try</h3><div class=\"cards\">");
            for p in prompts {
                let title = p.get("title").and_then(serde_json::Value::as_str);
                let text = p.get("prompt").and_then(serde_json::Value::as_str);
                body.push_str("<div class=\"card\">");
                if let Some(t) = title {
                    let _ = write!(body, "<h3>{}</h3>", html_escape(t));
                }
                if let Some(t) = text {
                    copy_row(&mut body, "pre", "snippet", t, false);
                }
                body.push_str("</div>");
            }
            body.push_str("</div>");
        }
        section(&mut out, spec.key, spec.heading, &body);
    }
    out
}

/// The report's stylesheet, inlined so the file is self-contained (no fonts,
/// no scripts, no network: it is a private file about private work).
const REPORT_CSS: &str = "\
:root{color-scheme:light dark;--bg:#fbfaf8;--fg:#1c1b19;--dim:#6b6863;--line:#e5e1da;--card:#fff;--accent:#c96442}\
@media(prefers-color-scheme:dark){:root{--bg:#191817;--fg:#f0eee9;--dim:#9a958d;--line:#302e2b;--card:#211f1e}}\
*{box-sizing:border-box}\
body{margin:0;padding:2.5rem 1.25rem 5rem;background:var(--bg);color:var(--fg);\
font:16px/1.6 ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}\
main{max-width:56rem;margin:0 auto}\
h1{font-size:1.9rem;margin:0 0 .25rem}h2{font-size:1.25rem;margin:0 0 1rem}h3{font-size:1rem;margin:0 0 .35rem}\
.sub{color:var(--dim);margin:0 0 2.5rem}\
section{margin:0 0 2.75rem}\
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.75rem;margin:0 0 2.5rem}\
.stat{background:var(--card);border:1px solid var(--line);border-radius:.6rem;padding:.85rem 1rem}\
.stat b{display:block;font-size:1.5rem;font-weight:650;line-height:1.2}\
.stat span{color:var(--dim);font-size:.8rem}\
.glance{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);\
border-radius:.6rem;padding:1.25rem 1.4rem;margin:0 0 2.5rem}\
.glance p{margin:0 0 .9rem}.glance p:last-child{margin:0}.glance em{color:var(--dim);font-style:normal;font-weight:650}\
.chart{display:flex;flex-direction:column;gap:.35rem}\
.row{display:grid;grid-template-columns:minmax(6rem,11rem) 1fr auto;gap:.75rem;align-items:center}\
.label{color:var(--dim);font-size:.85rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\
.track{background:var(--line);border-radius:.25rem;height:.85rem;overflow:hidden}\
.bar{height:100%;border-radius:.25rem}\
.value{font-variant-numeric:tabular-nums;font-size:.85rem;color:var(--dim)}\
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:.75rem}\
.card{background:var(--card);border:1px solid var(--line);border-radius:.6rem;padding:1rem}\
.card p{margin:0 0 .5rem}.card p:last-child{margin:0}\
.fix{color:var(--accent)}\
.feature{margin:0 0 .75rem}.feature pre{margin:.7rem 0 0}\
.callout{border-left:3px solid var(--accent);padding-left:.9rem;color:var(--dim)}\
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);\
border-radius:.4rem;padding:.7rem;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0}\
ul{margin:0;padding-left:1.2rem}li{margin:0 0 .4rem}\
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(18rem,1fr));gap:2rem}\
.note{color:var(--dim);font-size:.85rem}\
.copyrow{display:flex;align-items:flex-start;gap:.55rem;margin:0 0 .55rem}\
.copyrow>[data-copy]{flex:1;margin:0}\
.copyrow p[data-copy]{background:var(--card);border:1px solid var(--line);border-radius:.4rem;padding:.55rem .7rem}\
.copyrow input.pick{margin:.75rem 0 0;accent-color:var(--accent)}\
.copy,.copy-all{flex:none;background:var(--card);color:var(--fg);border:1px solid var(--line);\
border-radius:.4rem;padding:.35rem .7rem;font:inherit;font-size:.8rem;cursor:pointer}\
.copy:hover,.copy-all:hover{border-color:var(--accent);color:var(--accent)}\
.copy.ok,.copy-all.ok{border-color:var(--accent);color:var(--accent)}\
.copyall{margin:0 0 .75rem}\
footer{color:var(--dim);font-size:.8rem;border-top:1px solid var(--line);padding-top:1rem}\
";

/// Clipboard behaviour for the copy controls, inlined for the same reason the
/// stylesheet is: the report is a private local file and must work with no
/// network and no assets beside it.
///
/// `navigator.clipboard` is unavailable on `file://` in most browsers (not a
/// secure context), which is exactly where this report is opened, so the
/// hidden-textarea `execCommand` path is the one that usually runs — it is the
/// fallback in name only.
const REPORT_JS: &str = "\
(function(){\
function flash(b){var t=b.textContent;b.textContent='Copied';b.classList.add('ok');\
setTimeout(function(){b.textContent=t;b.classList.remove('ok');},1200);}\
function legacy(text,b){var ta=document.createElement('textarea');ta.value=text;\
ta.setAttribute('readonly','');ta.style.position='fixed';ta.style.opacity='0';\
document.body.appendChild(ta);ta.select();\
try{if(document.execCommand('copy')){flash(b);}}catch(e){}\
document.body.removeChild(ta);}\
function copy(text,b){\
if(navigator.clipboard&&navigator.clipboard.writeText){\
navigator.clipboard.writeText(text).then(function(){flash(b);},function(){legacy(text,b);});}\
else{legacy(text,b);}}\
function textOf(el){return el?el.innerText:'';}\
document.addEventListener('click',function(ev){\
var b=ev.target.closest&&ev.target.closest('button.copy');\
if(b){var row=b.closest('.copyrow');copy(textOf(row&&row.querySelector('[data-copy]')),b);return;}\
b=ev.target.closest&&ev.target.closest('button.copy-all');\
if(!b){return;}\
var parts=[];\
b.closest('section').querySelectorAll('.copyrow').forEach(function(r){\
var pick=r.querySelector('input.pick');\
if(!pick||pick.checked){parts.push(textOf(r.querySelector('[data-copy]')));}});\
copy(parts.join('\\n\\n'),b);});\
})();\
";

/// Renders the full HTML report.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn render_html(
    agg: &Aggregated,
    narrative: &Narrative,
    tz_offset: i64,
    generated_at: u64,
) -> String {
    render_html_cancellable(agg, narrative, tz_offset, generated_at, &|| false).unwrap_or_default()
}

/// [`render_html`], abandoning the document if `cancel` goes true.
///
/// Rendering is normally milliseconds, but it is the last thing standing
/// between the user and their prompt, and a history large enough to make the
/// scan slow makes this slow too. Checked between sections rather than
/// mid-string, so a cancelled render yields no document at all — never a
/// truncated one, which paired with the rename in [`write_report`] is what
/// keeps the previous report intact.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn render_html_cancellable(
    agg: &Aggregated,
    narrative: &Narrative,
    tz_offset: i64,
    generated_at: u64,
    cancel: &dyn Fn() -> bool,
) -> Option<String> {
    let mut out = String::with_capacity(16 * 1024);
    out.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n");
    out.push_str("<title>plank insights</title>\n<style>");
    out.push_str(REPORT_CSS);
    out.push_str("</style>\n</head>\n<body>\n<main>\n");

    let _ = write!(
        out,
        "<h1>Your plank insights</h1><p class=\"sub\">generated {} · {} sessions ({} scanned) · {} → {} · {} days active</p>",
        datetime_str(generated_at, tz_offset),
        commas(agg.sessions_counted as u64),
        commas(agg.sessions_scanned as u64),
        date_str(agg.first_session, tz_offset),
        date_str(agg.last_session, tz_offset),
        agg.days_active
    );

    if let Some(glance) = narrative.get("at_a_glance") {
        let mut body = String::new();
        for (field, label) in [
            ("working", "What's working"),
            ("hindering", "What's holding you back"),
            ("quick_win", "Quick win"),
        ] {
            if let Some(text) = glance.get(field).and_then(serde_json::Value::as_str) {
                let _ = write!(body, "<p><em>{label}.</em> {}</p>", html_escape(text));
            }
        }
        if !body.is_empty() {
            let _ = write!(out, "<div class=\"glance\">{body}</div>");
        }
    }

    if cancel() {
        return None;
    }
    out.push_str("<div class=\"stats\">");
    let stats: [(String, &str); 6] = [
        (commas(u64::from(agg.human_messages)), "prompts"),
        (
            if agg.timing_is_representative() && agg.hours >= 0.05 {
                format!("{:.1}h", agg.hours)
            } else {
                "—".to_owned()
            },
            "in session",
        ),
        (
            format!(
                "+{} / −{}",
                commas(u64::from(agg.lines_added)),
                commas(u64::from(agg.lines_removed))
            ),
            "lines",
        ),
        (commas(agg.files_touched as u64), "files touched"),
        (commas(u64::from(agg.commits)), "commits"),
        (
            if agg.response_median > 0 {
                dur(agg.response_median)
            } else {
                "—".to_owned()
            },
            "median reply",
        ),
    ];
    for (value, label) in &stats {
        let _ = write!(
            out,
            "<div class=\"stat\"><b>{value}</b><span>{label}</span></div>"
        );
    }
    out.push_str("</div>");

    // Deterministic charts first: they are the part of the report that is
    // always true, narrative or no narrative.
    let mut charts = String::from("<div class=\"grid2\">");
    if !agg.tools.is_empty() {
        let _ = write!(
            charts,
            "<div><h3>Tools you use</h3>{}</div>",
            bar_chart(&agg.tools, 12, "var(--accent)")
        );
    }
    if !agg.languages.is_empty() {
        let _ = write!(
            charts,
            "<div><h3>Languages</h3>{}</div>",
            bar_chart(&agg.languages, 10, "#5b8c6e")
        );
    }
    charts.push_str("</div>");
    section(&mut out, "activity", "What you work on", &charts);
    if cancel() {
        return None;
    }

    if !agg.projects.is_empty() {
        section(
            &mut out,
            "projects",
            "Projects",
            &bar_chart(&agg.projects, 10, "#7a6ea8"),
        );
    }

    // Activity by hour, with an explicit note when part of the history
    // predates timestamps — an honest gap beats a chart that silently covers
    // a fraction of the sessions.
    if cancel() {
        return None;
    }
    if agg.by_hour.iter().any(|&n| n > 0) {
        let hours: Vec<(String, u32)> = agg
            .by_hour
            .iter()
            .enumerate()
            .map(|(h, &n)| (format!("{h:02}:00"), n))
            .collect();
        let mut body = bar_chart(&hours, 24, "#c98f42");
        if agg.sessions_without_timing > 0 {
            let _ = write!(
                body,
                "<p class=\"note\">{} of {} sessions predate per-message timestamps and are not \
in this chart.</p>",
                agg.sessions_without_timing, agg.sessions_counted
            );
        }
        section(&mut out, "hours", "When you work", &body);
    }

    if agg.response_median > 0 {
        let mut body = bar_chart(&agg.response_histogram, 8, "#4f7fa8");
        let _ = write!(
            body,
            "<p class=\"note\">Median {}, mean {} between the agent finishing and your next \
message.",
            dur(agg.response_median),
            dur(agg.response_mean)
        );
        // Same exposure as the hour chart: say what the sample is rather than
        // let a partial history read as the whole one.
        if agg.sessions_without_timing > 0 {
            let _ = write!(
                body,
                " Measured over {} of {} sessions; the rest predate per-message timestamps.",
                agg.sessions_counted - agg.sessions_without_timing,
                agg.sessions_counted
            );
        }
        body.push_str("</p>");
        section(&mut out, "response", "How fast you reply", &body);
    }

    if cancel() {
        return None;
    }
    if !agg.errors.is_empty() {
        let mut body = String::from("<div class=\"grid2\">");
        let _ = write!(
            body,
            "<div><h3>By kind</h3>{}</div>",
            bar_chart(&agg.errors, 10, "#b4523f")
        );
        let _ = write!(
            body,
            "<div><h3>By tool</h3>{}</div>",
            bar_chart(&agg.error_tools, 10, "#b4523f")
        );
        body.push_str("</div>");
        if !agg.error_rates.is_empty() {
            let rates = agg
                .error_rates
                .iter()
                .take(5)
                .map(|(n, r)| format!("{} {r:.0}%", html_escape(n)))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(body, "<p class=\"note\">Failure rate: {rates}.</p>");
        }
        section(&mut out, "errors", "Where things went wrong", &body);
    }

    if agg.overlaps.is_empty() {
        section(
            &mut out,
            "overlap",
            "One thing at a time",
            "<p class=\"note\">No two sessions were ever live at the same moment.</p>",
        );
    } else {
        let mut body = format!(
            "<p>{} pairs of sessions were live at the same time, covering about {:.0}% of your \
prompts.</p><ul>",
            agg.overlaps.len(),
            agg.overlap_share
        );
        for o in agg.overlaps.iter().take(10) {
            let _ = write!(
                body,
                "<li>{} ↔ {} <span class=\"note\">({})</span></li>",
                html_escape(&o.a),
                html_escape(&o.b),
                date_str(o.at, tz_offset)
            );
        }
        body.push_str("</ul>");
        section(&mut out, "overlap", "Running in parallel", &body);
    }

    if cancel() {
        return None;
    }
    out.push_str(&narrative_html(narrative));

    let _ = write!(
        out,
        "<footer>Generated locally by plank on {}. Nothing here left this machine.</footer>",
        datetime_str(generated_at, tz_offset)
    );
    out.push_str("\n</main>\n<script>");
    out.push_str(REPORT_JS);
    out.push_str("</script>\n</body>\n</html>\n");
    Some(out)
}

/// Writes the report and returns its path.
///
/// The name carries the generation time (`report-YYYYMMDD-HHMM.html`), so
/// successive runs sit side by side rather than overwriting each other.
/// Written to a matching `.tmp` and renamed into place, so a report already
/// under that name survives intact until the new one is complete. A run that
/// fails or is interrupted partway therefore costs the user nothing: the
/// report they already had is still there, whole, rather than replaced by a
/// truncated file. The rename is atomic on the same filesystem, so a reader never sees
/// a half-written report either.
///
/// # Errors
///
/// Returns the underlying I/O error's message if the file cannot be written
/// or the rename fails; the temporary file is cleaned up in both cases.
pub fn write_report(
    root: &Path,
    html: &str,
    tz_offset: i64,
    generated_at: u64,
) -> Result<PathBuf, String> {
    let stem = format!("report-{}", file_stamp(generated_at, tz_offset));
    let path = root.join(format!("{stem}.html"));
    let tmp = root.join(format!("{stem}.html.tmp"));
    write_private(&tmp, html).map_err(|e| format!("{}: {e}", tmp.display()))?;
    if let Err(e) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{}: {e}", path.display()));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Message;

    fn session_with(msgs: Vec<Message>) -> Session {
        let mut s = Session::new();
        s.transcript = msgs;
        s
    }

    fn entry(id: &str) -> SessionEntry {
        SessionEntry {
            id: id.to_owned(),
            title: "t".to_owned(),
            created_at: 1_000,
            last_used: 2_000,
            file_size: 10,
            tag: String::new(),
            last_prompt: String::new(),
            payload_bytes: 0,
            path: PathBuf::from("/tmp/x.kv"),
        }
    }

    fn tool_call(name: &str, args: &[(&str, &str)]) -> String {
        let mut s = String::from("<｜DSML｜tool_calls>\n");
        let _ = writeln!(s, "<｜DSML｜invoke name=\"{name}\">");
        for (k, v) in args {
            let _ = write!(s, "<｜DSML｜parameter name=\"{k}\" string=\"true\">{v}");
            s.push_str("</｜DSML｜parameter>\n");
        }
        s.push_str("</｜DSML｜invoke>\n</｜DSML｜tool_calls>");
        s
    }

    #[test]
    fn injected_context_is_not_a_human_message() {
        let s = session_with(vec![
            Message::user("Agent instructions:\n\nread CLAUDE.md"),
            Message::user("This is the git status at the start of the conversation."),
            Message::user("actually fix the parser"),
            Message::user("<tool_result>Tool result 1 (bash):\nok\n</tool_result>"),
        ]);
        let meta = extract_meta(&entry("a"), 0, &s);
        assert_eq!(meta.human_messages, 1);
        assert_eq!(meta.tool_results, 1);
        assert_eq!(meta.first_prompt, "actually fix the parser");
    }

    #[test]
    fn tool_calls_edits_and_languages_are_counted() {
        let s = session_with(vec![
            Message::user("go"),
            Message::assistant(tool_call(
                "edit",
                &[
                    ("path", "/p/src/main.rs"),
                    ("old", "a\nb\n"),
                    ("new", "a\nc\nd\n"),
                ],
            )),
            Message::assistant(tool_call(
                "bash",
                &[("command", "git commit -m x && git push")],
            )),
        ]);
        let meta = extract_meta(&entry("a"), 0, &s);
        assert_eq!(meta.tools.get("edit"), Some(&1));
        assert_eq!(meta.tools.get("bash"), Some(&1));
        assert_eq!(meta.languages.get("Rust"), Some(&1));
        assert_eq!(meta.files, vec!["/p/src/main.rs".to_owned()]);
        assert_eq!(meta.commits, 1);
        assert_eq!(meta.pushes, 1);
        // "b" became "c", and "d" was appended.
        assert_eq!((meta.lines_added, meta.lines_removed), (2, 1));
    }

    /// The transcript a rating points at, and a rating that matches it. Ratings
    /// are keyed by ordinal *and* digest, so a test fixture has to carry the
    /// real sha1 of the message text or every rating reads as orphaned.
    #[cfg(test)]
    fn rated(text: &str, ordinal: usize, positive: bool, note: &str) -> crate::feedback::Rating {
        crate::feedback::Rating {
            ordinal,
            digest: crate::session::sha1_hex(text.as_bytes()),
            positive,
            note: note.to_string(),
            at: 1_700_000_000,
        }
    }

    #[test]
    fn aggregate_feedback_counts_and_ranks() {
        let mut agg = Aggregated::default();
        let s1 = vec![Message::user("turn zero"), Message::assistant("turn one")];
        let s2 = vec![Message::user("only turn")];
        let feedback = vec![
            (
                "s1".to_string(),
                vec![
                    rated("turn zero", 0, true, "great"),
                    rated("turn one", 1, false, "confusing"),
                ],
            ),
            ("s2".to_string(), vec![rated("only turn", 0, true, "")]),
        ];
        let lookup = |id: &str| match id {
            "s1" => Some(s1.clone()),
            "s2" => Some(s2.clone()),
            _ => None,
        };
        aggregate_feedback(&mut agg, &feedback, 0, &lookup);
        assert_eq!(agg.ratings_orphaned, 0, "every rating still has its turn");
        assert_eq!(agg.ratings_total, 3);
        assert_eq!(agg.ratings_positive, 2);
        assert_eq!(agg.ratings_negative, 1);
        assert_eq!(
            agg.worst_turns,
            vec![("s1".to_string(), "confusing".to_string())]
        );
        assert_eq!(agg.satisfaction_by_day.len(), 1);
        assert!((agg.satisfaction_by_day[0].1 - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_rating_orphaned_by_compaction_is_reported_and_not_counted() {
        // The M2 assertion: compaction renumbers the transcript, so a rating
        // keyed to the old index no longer has a subject. It must be reported
        // as orphaned, kept out of the satisfaction statistics, and never
        // reattributed to whatever now sits at that index.
        let mut agg = Aggregated::default();
        let feedback = vec![(
            "s1".to_string(),
            vec![
                rated("kept turn", 0, true, ""),
                rated("rated turn", 1, false, "was wrong"),
            ],
        )];
        // Compaction rewrote index 1 into something else.
        let compacted = vec![Message::user("kept turn"), Message::assistant("a summary")];
        aggregate_feedback(&mut agg, &feedback, 0, &|_| Some(compacted.clone()));
        assert_eq!(agg.ratings_orphaned, 1, "the renumbered rating is orphaned");
        assert_eq!(agg.ratings_total, 1, "orphans stay out of the total");
        assert_eq!(agg.ratings_positive, 1);
        assert_eq!(
            agg.ratings_negative, 0,
            "the orphan must not be reattributed to the summary now at index 1"
        );
        assert!(
            agg.worst_turns.is_empty(),
            "an orphaned negative rating names no turn: {:?}",
            agg.worst_turns
        );
    }

    #[test]
    fn ratings_for_a_vanished_session_are_orphaned() {
        // The session file was swept; its ratings have no subject at all.
        let mut agg = Aggregated::default();
        let feedback = vec![("gone".to_string(), vec![rated("whatever", 0, true, "")])];
        aggregate_feedback(&mut agg, &feedback, 0, &|_| None);
        assert_eq!(agg.ratings_orphaned, 1);
        assert_eq!(agg.ratings_total, 0);
    }

    #[test]
    fn an_all_orphan_report_still_mentions_them() {
        // `sessions_counted` is non-zero or the summary short-circuits.
        let agg = Aggregated {
            sessions_counted: 1,
            ratings_orphaned: 2,
            ..Default::default()
        };
        let lines = render_summary(&agg, &Narrative::new(), 0, 1_700_000_000);
        assert!(
            lines.iter().any(|l| l.contains("2 orphaned")),
            "orphans must be reported even with no live ratings: {lines:?}"
        );
    }

    #[test]
    fn tool_errors_are_attributed_and_classified() {
        let s = session_with(vec![
            Message::user("go"),
            Message::user(
                "<tool_result>Tool result 1 (edit):\nTool error: old text anchor not found\n\
Tool result 2 (bash):\nexit_status=2\n<output>\nboom\n</output>\n\
Tool result 3 (read):\nfine\n</tool_result>",
            ),
        ]);
        let meta = extract_meta(&entry("a"), 0, &s);
        assert_eq!(meta.errors.get("Edit anchor missing"), Some(&1));
        assert_eq!(meta.errors.get("Command failed"), Some(&1));
        assert_eq!(meta.error_tools.get("edit"), Some(&1));
        assert_eq!(meta.error_tools.get("bash"), Some(&1));
        assert_eq!(meta.error_tools.get("read"), None);
    }

    #[test]
    fn response_gaps_ignore_the_implausible() {
        let s = session_with(vec![
            Message::user("one").at(1_000),
            Message::assistant("reply").at(1_010),
            Message::user("two").at(1_040), // 30s: counted
            Message::assistant("reply").at(1_050),
            Message::user("three").at(1_051), // 1s: too fast to be a response
            Message::assistant("reply").at(1_060),
            Message::user("four").at(99_999), // hours later: a break, not a reply
        ]);
        let meta = extract_meta(&entry("a"), 0, &s);
        assert_eq!(meta.response_secs, vec![30]);
        assert_eq!(meta.human_ts.len(), 4);
    }

    #[test]
    fn duration_comes_from_the_messages_not_the_file() {
        // A session saved only at exit has `created == used`, so the file
        // times say it took no time at all; the stamps know better.
        let mut e = entry("a");
        e.created_at = 5_000;
        e.last_used = 5_000;
        let stamped = extract_meta(
            &e,
            0,
            &session_with(vec![
                Message::user("one").at(1_000),
                Message::assistant("reply").at(1_900),
            ]),
        );
        assert_eq!(stamped.duration_secs(), 900);

        // Without stamps the file times are all there is.
        e.last_used = 5_600;
        let unstamped = extract_meta(&e, 0, &session_with(vec![Message::user("one")]));
        assert_eq!(unstamped.duration_secs(), 600);
    }

    #[test]
    fn slight_sessions_are_scanned_but_not_counted() {
        let one = extract_meta(&entry("a"), 0, &session_with(vec![Message::user("hi")]));
        let two = extract_meta(
            &entry("b"),
            0,
            &session_with(vec![Message::user("hi"), Message::user("more")]),
        );
        assert!(!one.is_substantive());
        assert!(two.is_substantive());
        let agg = aggregate(&[one, two], 0);
        assert_eq!(agg.sessions_scanned, 2);
        assert_eq!(agg.sessions_counted, 1);
    }

    #[test]
    fn aggregate_ranks_and_dedupes() {
        let mut a = extract_meta(
            &entry("a"),
            0,
            &session_with(vec![
                Message::user("x"),
                Message::user("y"),
                Message::assistant(tool_call("read", &[("path", "/p/a.rs")])),
            ]),
        );
        a.files = vec!["/p/a.rs".to_owned()];
        let mut b = a.clone();
        b.id = "b".to_owned();
        b.tools.insert("bash".to_owned(), 5);
        let agg = aggregate(&[a, b], 0);
        assert_eq!(agg.tools[0], ("bash".to_owned(), 5));
        // The same file touched in two sessions is one file touched.
        assert_eq!(agg.files_touched, 1);
    }

    #[test]
    fn overlap_needs_a_return_to_the_first_session() {
        let make = |id: &str, ts: &[u64]| SessionMeta {
            id: id.to_owned(),
            human_messages: 2,
            human_ts: ts.to_vec(),
            ..SessionMeta::default()
        };
        // a → b → a inside the window.
        let agg = aggregate(&[make("a", &[100, 300]), make("b", &[200, 400])], 0);
        assert_eq!(agg.overlaps.len(), 1);
        assert_eq!(
            (agg.overlaps[0].a.as_str(), agg.overlaps[0].b.as_str()),
            ("a", "b")
        );

        // Back-to-back sessions never interleave.
        let agg = aggregate(&[make("a", &[100, 200]), make("b", &[9_000, 9_100])], 0);
        assert!(agg.overlaps.is_empty());
    }

    #[test]
    fn response_histogram_covers_every_gap() {
        let meta = SessionMeta {
            id: "a".to_owned(),
            human_messages: 2,
            response_secs: vec![3, 45, 700, 5_000],
            ..SessionMeta::default()
        };
        let agg = aggregate(&[meta], 0);
        let total: u32 = agg.response_histogram.iter().map(|r| r.1).sum();
        assert_eq!(total, 4, "{:?}", agg.response_histogram);
        assert_eq!(agg.response_median, 700);
    }

    #[test]
    fn the_repeat_guard_stops_a_repeating_reply_but_not_ordinary_prose() {
        // The observed failure: a recommendation section that keeps emitting
        // the same list entry until the token budget runs out.
        let mut guard = RepeatGuard::new();
        let mut tripped = false;
        for _ in 0..20 {
            if guard.feed("{\"name\": \"Skills\", \"why\": \"you repeat steps\"}, ") {
                tripped = true;
                break;
            }
        }
        assert!(tripped, "a repeating list entry must stop the pass");

        // A run of blank lines is the other shape the same failure takes.
        let mut guard = RepeatGuard::new();
        assert!(
            (0..40).any(|_| guard.feed("\n\n\n\n\n\n\n\n")),
            "a newline run must stop the pass"
        );

        // Ordinary varied prose, well past the window, must not trip it.
        let mut guard = RepeatGuard::new();
        for i in 0..200 {
            assert!(
                !guard.feed(&format!(
                    "sentence {i} says something different about how they work, at length. "
                )),
                "varied prose tripped the guard at chunk {i}"
            );
        }

        // Neither must a reply full of multi-byte characters: the window is a
        // byte budget and trimming it must land on a char boundary.
        let mut guard = RepeatGuard::new();
        for i in 0..200 {
            let _ = guard.feed(&format!("\u{2014} r\u{e9}sum\u{e9} {i} \u{2014} "));
        }
    }

    #[test]
    fn extract_json_survives_surrounding_prose() {
        let v = extract_json("sure! here you go:\n```json\n{\"a\": {\"b\": \"}\"}}\n```\ndone")
            .expect("nested object with a brace in a string");
        assert_eq!(v["a"]["b"], "}");
        assert!(extract_json("no json here").is_none());
        assert!(extract_json("{\"unterminated\": ").is_none());
    }

    #[test]
    fn narrative_context_omits_fields_that_carry_no_information() {
        // Sending `hours: 0.0` and a project list that is entirely
        // "(unrecorded)" makes the model argue with the data instead of
        // reading it — observed doing exactly that on a real history.
        let legacy = SessionMeta {
            id: "a".to_owned(),
            human_messages: 4,
            tools: [("bash".to_owned(), 3)].into_iter().collect(),
            ..SessionMeta::default()
        };
        let ctx = narrative_context(&aggregate(std::slice::from_ref(&legacy), 0));
        for absent in [
            "hours",
            "projects",
            "median_response_secs",
            "concurrent_session_pairs",
        ] {
            assert!(!ctx.contains(absent), "{absent} should be omitted:\n{ctx}");
        }
        // What is real is still sent.
        assert!(ctx.contains("top_tools") && ctx.contains("human_messages"));

        // With real timing and a real project, the fields come back.
        let recorded = SessionMeta {
            project: "/work/thing".to_owned(),
            response_secs: vec![30],
            human_ts: vec![1_000, 8_200],
            first_ts: 1_000,
            last_ts: 8_200,
            ..legacy.clone()
        };
        let ctx = narrative_context(&aggregate(std::slice::from_ref(&recorded), 0));
        assert!(ctx.contains("hours"), "{ctx}");
        assert!(ctx.contains("/work/thing"), "{ctx}");
        assert!(ctx.contains("median_response_secs"), "{ctx}");

        // The case that actually bit: a couple of timed sessions among many
        // untimed ones sum to a small non-zero hour count. That is an absent
        // number wearing a small number's clothes — the model read it as
        // "barely uses this" and said so in the report.
        let mut mostly_untimed = vec![recorded];
        for i in 0..8 {
            mostly_untimed.push(SessionMeta {
                id: format!("u{i}"),
                ..legacy.clone()
            });
        }
        let agg = aggregate(&mostly_untimed, 0);
        assert!(agg.hours > 0.05, "the misleading total is really there");
        assert!(!agg.timing_is_representative());
        let ctx = narrative_context(&agg);
        assert!(
            !ctx.contains("hours"),
            "an unrepresentative total must not be sent:\n{ctx}"
        );
        // And it is not shown to the user as a number either.
        assert!(render_html(&agg, &Narrative::new(), 0, 0).contains("—"));
        assert!(
            !render_summary(&agg, &Narrative::new(), 0, 0)
                .iter()
                .any(|l| l.contains("in session"))
        );
    }

    #[test]
    fn think_ticker_emits_lines_while_reasoning_and_stops_at_the_answer() {
        let mut t = ThinkTicker::new();
        assert!(
            t.feed("first thought").is_empty(),
            "an unfinished line waits"
        );
        assert_eq!(
            t.feed(" continues\nsecond"),
            vec!["first thought continues"]
        );

        // The answer is parsed, never displayed — and the last scrap of
        // thinking is flushed when the block closes.
        let out = t.feed(" thought</think>\n{\"a\": 1}");
        assert_eq!(out, vec!["second thought"]);
        assert!(
            t.feed("{\"more\": 2}").is_empty(),
            "nothing after the answer starts"
        );
    }

    #[test]
    fn think_ticker_wraps_reasoning_that_has_no_line_breaks() {
        // A model that reasons in one long paragraph must still produce
        // visible movement rather than one line at the very end.
        let mut t = ThinkTicker::new();
        let words = "alpha ".repeat(40);
        let lines = t.feed(&words);
        assert!(lines.len() > 1, "expected several ticks, got {lines:?}");
        assert!(lines.iter().all(|l| l.len() <= TICK_WIDTH), "{lines:?}");

        // The width is a wrap threshold, not a maximum: a line the model ends
        // itself is passed through whole rather than chopped mid-sentence.
        let mut t = ThinkTicker::new();
        let long = format!("{}\n", "beta ".repeat(30));
        assert_eq!(t.feed(&long), vec![long.trim().to_owned()]);
    }

    #[test]
    fn think_ticker_wraps_without_panicking_on_a_multibyte_boundary() {
        // Regression: TICK_WIDTH is a byte index, and when it lands inside a
        // multi-byte char the old `self.pending[..TICK_WIDTH]` slice panicked
        // with "end byte index 88 is not a char boundary". Lay out 86 ASCII
        // bytes, a space, then an em-dash starting at byte 87 (bytes 87..90),
        // so byte 88 falls inside the dash; pad well past the width.
        let mut t = ThinkTicker::new();
        let line = format!("{} —{} ", "a".repeat(86), "b".repeat(30));
        let lines = t.feed(&line);
        assert!(!lines.is_empty(), "expected a tick, got {lines:?}");
        assert!(lines.iter().all(|l| l.len() <= TICK_WIDTH), "{lines:?}");
    }

    #[test]
    fn extract_json_skips_thinking_and_unparseable_braces() {
        // The engine opens `<think>` implicitly, so a reply that reasons
        // about the schema before answering must not have its reasoning
        // mistaken for the answer.
        let reply = "the schema wants {\"working\": ...} so\n</think>\n{\"working\": \"yes\"}";
        assert_eq!(
            extract_json(reply).expect("post-think object")["working"],
            "yes"
        );

        // A stray unbalanced brace before the answer is stepped over.
        assert_eq!(
            extract_json("note: { is a brace. {\"a\": 1}").expect("later object")["a"],
            1
        );

        // With no think tag at all the whole reply is still searched.
        assert_eq!(
            extract_json("here: {\"a\": 2}").expect("bare object")["a"],
            2
        );
    }

    #[test]
    fn html_report_is_self_contained_and_escapes_narrative() {
        let agg = aggregate(
            &[SessionMeta {
                id: "a".to_owned(),
                human_messages: 4,
                created_at: 1_700_000_000,
                last_used: 1_700_003_600,
                tools: [("bash".to_owned(), 3)].into_iter().collect(),
                ..SessionMeta::default()
            }],
            0,
        );
        let mut narrative = Narrative::new();
        narrative.insert(
            "at_a_glance".to_owned(),
            serde_json::json!({"working": "<script>alert(1)</script>"}),
        );
        let html = render_html(&agg, &narrative, 0, 0);
        // The report carries exactly one script — its own copy-button
        // handler — and narrative text can never become a second one.
        assert!(!html.contains("<script>alert"), "narrative must be escaped");
        assert_eq!(html.matches("<script>").count(), 1, "only the report's own");
        assert!(html.contains("&lt;script&gt;"));
        // Self-contained: no network references of any kind.
        assert!(!html.contains("http://") && !html.contains("https://"));
        assert!(html.contains("2023-11-14"));
    }

    #[test]
    fn agents_md_suggestions_are_a_copyable_checklist() {
        let agg = aggregate(
            &[SessionMeta {
                id: "a".to_owned(),
                human_messages: 4,
                tools: [("bash".to_owned(), 3)].into_iter().collect(),
                ..SessionMeta::default()
            }],
            0,
        );
        let mut narrative = Narrative::new();
        narrative.insert(
            "suggestions".to_owned(),
            serde_json::json!({
                "agents_md": ["Always run cargo fmt & clippy before committing."],
                "prompts": [{"title": "Release", "prompt": "cut a beta release"}],
            }),
        );
        let html = render_html(&agg, &narrative, 0, 0);
        // Each instruction is picked by default, copyable on its own, and
        // collected by the section's copy-all control.
        assert!(html.contains("class=\"pick\" checked"), "{html}");
        assert!(html.contains("class=\"copy-all\""), "{html}");
        assert_eq!(html.matches("class=\"copyrow\"").count(), 2, "{html}");
        // Ampersands in an instruction are escaped, not markup.
        assert!(html.contains("cargo fmt &amp; clippy"), "{html}");
        // The handler that makes the buttons work ships with the file, and
        // nothing it needs is fetched.
        assert!(
            html.contains("execCommand"),
            "clipboard fallback is inlined"
        );
        assert!(!html.contains("http://") && !html.contains("https://"));
    }

    #[test]
    fn the_features_section_renders_a_card_per_recommendation() {
        let agg = aggregate(
            &[SessionMeta {
                id: "a".to_owned(),
                human_messages: 4,
                tools: [("bash".to_owned(), 3)].into_iter().collect(),
                ..SessionMeta::default()
            }],
            0,
        );
        let mut narrative = Narrative::new();
        narrative.insert(
            "features".to_owned(),
            serde_json::json!({"features": [{
                "name": "Hooks",
                "one_liner": "Shell commands bound to events.",
                "why": "You reran cargo fmt & clippy by hand every session.",
                "snippet": "{\"PostToolUse\": []}",
            }]}),
        );
        let html = render_html(&agg, &narrative, 0, 0);
        assert!(html.contains("id=\"features\""), "section is emitted");
        assert!(html.contains("Features to try"));
        assert!(html.contains("Why for you."));
        // Snippets are code, not markup: braces and quotes survive escaped.
        assert!(html.contains("&quot;PostToolUse&quot;"));
        // And it is copyable with one click, like every other suggestion.
        assert!(html.contains("class=\"copy\""), "{html}");
    }

    #[test]
    fn an_empty_extension_point_is_reported_as_none_rather_than_omitted() {
        assert_eq!(extension_line("skills", &[]), "skills: none");
        assert_eq!(
            extension_line("skills", &["a".to_owned(), "b".to_owned()]),
            "skills: a, b"
        );
        let many: Vec<String> = (0..15).map(|i| format!("s{i}")).collect();
        assert!(extension_line("skills", &many).ends_with("(+3 more)"));
    }

    #[test]
    fn the_inventory_reaches_the_narrative_context_only_when_it_was_read() {
        let mut agg = aggregate(
            &[SessionMeta {
                id: "a".to_owned(),
                human_messages: 4,
                ..SessionMeta::default()
            }],
            0,
        );
        assert!(!narrative_context(&agg).contains("extensions_in_use"));
        agg.extensions = vec![extension_line("skills", &["release".to_owned()])];
        let ctx = narrative_context(&agg);
        assert!(ctx.contains("extensions_in_use") && ctx.contains("skills: release"));
    }

    #[test]
    fn a_cancelled_render_yields_nothing_rather_than_a_partial_document() {
        let agg = aggregate(
            &[SessionMeta {
                id: "a".to_owned(),
                human_messages: 4,
                tools: [("bash".to_owned(), 3)].into_iter().collect(),
                errors: [("Command failed".to_owned(), 2)].into_iter().collect(),
                ..SessionMeta::default()
            }],
            0,
        );
        // Cancelled from the very first check.
        assert!(render_html_cancellable(&agg, &Narrative::new(), 0, 0, &|| true).is_none());

        // Cancelled partway: still nothing, never a half-built document that
        // could be written over a good report.
        let seen = std::cell::Cell::new(0);
        let out = render_html_cancellable(&agg, &Narrative::new(), 0, 0, &|| {
            seen.set(seen.get() + 1);
            seen.get() > 2
        });
        assert!(out.is_none(), "a partial render must not be returned");
        assert!(seen.get() > 2, "cancellation is checked more than once");

        // Uncancelled, the document is whole.
        let whole = render_html_cancellable(&agg, &Narrative::new(), 0, 0, &|| false)
            .expect("an uncancelled render");
        assert!(whole.starts_with("<!DOCTYPE html>") && whole.ends_with("</html>\n"));
    }

    #[test]
    fn writing_the_report_replaces_it_only_once_the_new_one_is_whole() {
        let root = std::env::temp_dir().join(format!("plank-report-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let first = write_report(&root, "<html>one</html>", 0, 1_754_800_000).unwrap();
        assert_eq!(first.file_name().unwrap(), "report-20250810-0426.html");
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "<html>one</html>");
        // The scratch file does not survive a successful write.
        assert!(!root.join("report-20250810-0426.html.tmp").exists());

        // A later run is a new file; the earlier report is untouched.
        let later = write_report(&root, "<html>later</html>", 0, 1_754_803_600).unwrap();
        assert_eq!(later.file_name().unwrap(), "report-20250810-0526.html");
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "<html>one</html>");

        let second = write_report(&root, "<html>two</html>", 0, 1_754_800_000).unwrap();
        assert_eq!(second, first, "same stamp, replaced in place");
        assert_eq!(
            std::fs::read_to_string(&second).unwrap(),
            "<html>two</html>"
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&second).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "the report is the user's alone");
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn summary_reports_an_empty_history_plainly() {
        let agg = aggregate(&[], 0);
        let lines = render_summary(&agg, &Narrative::new(), 0, 1_754_800_000);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "generated 2025-08-10 04:26");
        assert!(lines[1].contains("No sessions"));
    }

    #[test]
    fn stale_cache_entries_are_rejected() {
        let e = entry("a");
        let fresh = SessionMeta {
            version: META_VERSION,
            src_size: e.file_size,
            src_mtime: 42,
            ..SessionMeta::default()
        };
        assert!(fresh.matches(&e, 42));
        assert!(
            !fresh.matches(&e, 43),
            "a rewritten session must be reparsed"
        );
        let old = SessionMeta {
            version: META_VERSION - 1,
            ..fresh.clone()
        };
        assert!(
            !old.matches(&e, 42),
            "an older extractor's numbers must not be reused"
        );
    }
}
