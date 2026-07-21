//! `@` typeahead completion for the Ratatui TUI prompt.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant, SystemTime};

use crate::editor::LineBuffer;
use nucleo_matcher::{Config, Matcher, Utf32Str};
use ratatui::crossterm::event::{KeyCode, KeyEvent};

/// A `@`-prefixed completion token found to the left of the cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtToken {
    /// Byte offset of the `@` within the inspected text.
    pub start: usize,
    /// Text typed after the `@` (and after the opening quote, if any).
    pub query: String,
    /// True when the token opened with a double quote.
    pub quoted: bool,
}

/// Finds the active `@` completion token in the text left of the cursor.
///
/// A token starts at a `@` that sits at the start of the input or directly
/// after whitespace. Returns `None` on a line whose first character is `!`
/// (a shell escape, not a prompt) or when no such `@` is present.
#[must_use]
pub fn detect_at_token(left: &str) -> Option<AtToken> {
    // Only the *current* line matters: with the multiline prompt, a first line
    // starting `!` must not suppress completion on every later line.
    let line = left.rsplit('\n').next().unwrap_or(left);
    if line.starts_with('!') {
        return None;
    }
    let at = left.rfind('@')?;
    // The `@` must open a word: start of input or preceded by whitespace.
    if at > 0 {
        let prev = left[..at].chars().next_back()?;
        if !prev.is_whitespace() {
            return None;
        }
    }
    let rest = &left[at + 1..];
    let (quoted, body) = rest.strip_prefix('"').map_or((false, rest), |b| (true, b));
    // Unquoted tokens end at whitespace; a quoted token may contain spaces.
    if !quoted && body.contains(char::is_whitespace) {
        return None;
    }
    Some(AtToken {
        start: at,
        query: body.to_string(),
        quoted,
    })
}

/// What a suggestion refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A regular file.
    File,
    /// A directory; rendered and inserted with a trailing `/`.
    Dir,
    /// An MCP resource, addressed `{server}:{uri}`.
    Resource,
}

/// One suggestion offered to the ranker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The text inserted when accepted.
    pub text: String,
    /// What the candidate refers to.
    pub kind: Kind,
    /// Set for paths inside a git submodule: vendored reference material that
    /// must never outrank the project's own files (see issue #45).
    pub demoted: bool,
}

/// A candidate that survived ranking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    /// The text inserted when accepted.
    pub text: String,
    /// What the candidate refers to.
    pub kind: Kind,
    /// Higher sorts first. Nucleo's match score; not comparable across queries.
    pub score: u16,
}

/// Scores `text` against `query` with nucleo, or `None` when it does not match.
///
/// `Config::DEFAULT.match_paths()` is nucleo's path-aware profile: it already
/// rewards matches at a path-segment boundary and inside the final segment,
/// which is what the hand-rolled scorer this replaced spent most of its code
/// on. An empty query matches everything at score 0.
fn score_one(matcher: &mut Matcher, query: &str, text: &str) -> Option<u16> {
    if query.is_empty() {
        return Some(0);
    }
    let mut hay = Vec::new();
    let mut needle = Vec::new();
    matcher.fuzzy_match(
        Utf32Str::new(text, &mut hay),
        Utf32Str::new(&query.to_lowercase(), &mut needle),
    )
}

/// Normalises a user-typed path query against the repo-relative index.
///
/// The index stores repo-relative paths, so an explicit `./` or `~/` prefix
/// would otherwise fail the subsequence test and empty the popup. `./` is
/// stripped. `~/` is expanded against `$HOME`; when the expansion cannot be
/// made relative to the current directory (the only frame the index shares),
/// it degrades to the repo-relative remainder rather than matching nothing.
#[must_use]
pub fn normalize_query(query: &str) -> String {
    if let Some(rest) = query.strip_prefix("./") {
        return rest.to_string();
    }
    if query == "." || query == "~" {
        return String::new();
    }
    let Some(rest) = query.strip_prefix("~/") else {
        return query.to_string();
    };
    if let Some(home) = std::env::var_os("HOME") {
        let abs = PathBuf::from(home).join(rest);
        if let Ok(cwd) = std::env::current_dir()
            && let Ok(rel) = abs.strip_prefix(&cwd)
        {
            return rel.to_string_lossy().into_owned();
        }
    }
    rest.to_string()
}

/// Ranks `cands` against `query`, best first, truncated to `limit`.
///
/// Demoted (submodule) candidates are considered only when nothing else
/// matches: leaving them in would spend the row budget and poison the Tab
/// common prefix, which is most of what issue #45 was. Partitioning rather
/// than penalising the score keeps this independent of the scorer's range.
#[must_use]
pub fn rank(query: &str, cands: &[Candidate], limit: usize) -> Vec<Match> {
    let query = &normalize_query(query);
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    let score = |matcher: &mut Matcher, only_demoted: bool| -> Vec<Match> {
        cands
            .iter()
            .filter(|c| c.demoted == only_demoted)
            .filter_map(|c| {
                Some(Match {
                    text: c.text.clone(),
                    kind: c.kind,
                    score: score_one(matcher, query, &c.text)?,
                })
            })
            .collect()
    };
    let mut out = score(&mut matcher, false);
    if out.is_empty() {
        out = score(&mut matcher, true);
    }
    // Score descending, then files before resources, then the shorter path,
    // then lexicographic — every tier deterministic.
    out.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| kind_rank(a.kind).cmp(&kind_rank(b.kind)))
            .then_with(|| a.text.chars().count().cmp(&b.text.chars().count()))
            .then_with(|| a.text.cmp(&b.text))
    });
    out.truncate(limit);
    out
}

/// Sort tier for a kind: files and directories outrank MCP resources when the
/// match quality is otherwise equal.
fn kind_rank(kind: Kind) -> u8 {
    match kind {
        Kind::File | Kind::Dir => 0,
        Kind::Resource => 1,
    }
}

/// The longest prefix shared by every match, compared by characters.
#[must_use]
pub fn longest_common_prefix(matches: &[Match]) -> String {
    let Some(first) = matches.first() else {
        return String::new();
    };
    let mut prefix: Vec<char> = first.text.chars().collect();
    for m in &matches[1..] {
        let n = m
            .text
            .chars()
            .zip(prefix.iter())
            .take_while(|(a, b)| a == *b)
            .count();
        prefix.truncate(n);
    }
    prefix.into_iter().collect()
}

/// How long a built index is trusted before a refresh is allowed.
///
/// From `ui.indexRefreshSecs`; see [`crate::settings`].
fn refresh_throttle() -> Duration {
    Duration::from_secs(crate::settings::active().ui.index_refresh_secs)
}
/// Every Nth path feeds the change-detection signature.
const SIGNATURE_STRIDE: usize = 16;

/// Runs a command in `root` and returns its stdout split on newlines.
///
/// Returns an empty vector when the command cannot be run or exits non-zero,
/// so a missing `git` or `rg` degrades to "no suggestions" rather than an error.
fn lines_from(root: &Path, program: &str, args: &[&str]) -> Vec<String> {
    let Ok(out) = Command::new(program).args(args).current_dir(root).output() else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// True when `root` is inside a git working tree.
fn is_git_repo(root: &Path) -> bool {
    !lines_from(root, "git", &["rev-parse", "--is-inside-work-tree"]).is_empty()
}

/// FNV-1a over the path count and every [`SIGNATURE_STRIDE`]th path.
fn signature_of(paths: &BTreeSet<String>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    };
    feed(&paths.len().to_le_bytes());
    for p in paths.iter().step_by(SIGNATURE_STRIDE) {
        feed(p.as_bytes());
    }
    h
}

/// Modification time of `root/.git/index`, when it exists.
#[must_use]
pub fn git_index_mtime(root: &Path) -> Option<SystemTime> {
    std::fs::metadata(root.join(".git").join("index"))
        .ok()?
        .modified()
        .ok()
}

/// An index of completable paths under one root.
#[derive(Debug)]
pub struct FileIndex {
    paths: BTreeSet<String>,
    cands: Vec<Candidate>,
    signature: u64,
    last_refresh: Option<Instant>,
    last_git_mtime: Option<SystemTime>,
    /// Cached `is_git_repo(root)`; probing it costs a subprocess per call.
    is_git: bool,
    /// Submodule roots, each with a trailing `/`; everything beneath one is
    /// demoted so vendored trees cannot bury the project's own files.
    submodules: Vec<String>,
}

/// Reads submodule paths from `.gitmodules`, each returned with a trailing `/`.
///
/// Parses the file directly rather than shelling out: `git config` would cost
/// another subprocess on every index build, and the `path =` lines are the
/// whole of what we need.
fn submodule_prefixes(root: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(root.join(".gitmodules")) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("path"))
        .filter_map(|r| r.trim_start().strip_prefix('='))
        .map(|p| format!("{}/", p.trim().trim_end_matches('/')))
        .collect()
}

impl FileIndex {
    /// Builds an index of tracked files under `root`.
    ///
    /// Uses `git ls-files --recurse-submodules` inside a git tree and
    /// `rg --files` outside one. Untracked files are folded in separately by
    /// [`FileIndex::fold_untracked`].
    #[must_use]
    pub fn build(root: &Path, respect_gitignore: bool) -> Self {
        let is_git = is_git_repo(root);
        let paths: BTreeSet<String> = if is_git {
            lines_from(root, "git", &["ls-files", "--recurse-submodules"])
                .into_iter()
                .collect()
        } else {
            let mut args = vec!["--files"];
            if !respect_gitignore {
                args.push("--no-ignore");
            }
            lines_from(root, "rg", &args)
                .into_iter()
                .map(|p| p.trim_start_matches("./").to_string())
                .collect()
        };
        let mut idx = Self {
            paths,
            cands: Vec::new(),
            signature: 0,
            last_refresh: None,
            last_git_mtime: None,
            is_git,
            submodules: submodule_prefixes(root),
        };
        idx.rebuild_candidates();
        idx
    }

    /// Folds untracked files into an already-built index.
    ///
    /// Honours `.gitignore` when `respect_gitignore` is set.
    pub fn fold_untracked(&mut self, root: &Path, respect_gitignore: bool) {
        if !self.is_git {
            return;
        }
        let mut args = vec!["ls-files", "--others"];
        if respect_gitignore {
            args.push("--exclude-standard");
        }
        for p in lines_from(root, "git", &args) {
            self.paths.insert(p);
        }
        self.rebuild_candidates();
    }

    /// Recomputes candidates and the signature from `self.paths`.
    ///
    /// Drops anything under `.git/` and synthesises every parent directory as
    /// its own entry with a trailing `/`.
    fn rebuild_candidates(&mut self) {
        self.paths
            .retain(|p| !p.starts_with(".git/") && p != ".git");
        let mut dirs: BTreeSet<String> = BTreeSet::new();
        for p in &self.paths {
            let mut cut = 0usize;
            while let Some(i) = p[cut..].find('/') {
                cut += i + 1;
                dirs.insert(p[..cut].to_string());
            }
        }
        let subs = &self.submodules;
        let demoted = |t: &str| subs.iter().any(|s| t.starts_with(s.as_str()));
        self.cands = dirs
            .into_iter()
            .map(|text| Candidate {
                demoted: demoted(&text),
                text,
                kind: Kind::Dir,
            })
            .chain(self.paths.iter().map(|p| Candidate {
                text: p.clone(),
                kind: Kind::File,
                demoted: demoted(p),
            }))
            .collect();
        self.signature = signature_of(&self.paths);
    }

    /// The candidates this index offers.
    #[must_use]
    pub fn candidates(&self) -> &[Candidate] {
        &self.cands
    }

    /// Change-detection hash; an equal signature means a rebuild is a no-op.
    #[must_use]
    pub fn signature(&self) -> u64 {
        self.signature
    }

    /// True when the index may be rebuilt: the throttle has expired, or
    /// `.git/index` moved since the last refresh.
    #[must_use]
    pub fn needs_refresh(&self, now: Instant, git_index_mtime: Option<SystemTime>) -> bool {
        if git_index_mtime.is_some() && git_index_mtime != self.last_git_mtime {
            return true;
        }
        self.last_refresh
            .is_none_or(|t| now.duration_since(t) >= refresh_throttle())
    }

    /// Records that a refresh just happened.
    pub fn mark_refreshed(&mut self, now: Instant, git_index_mtime: Option<SystemTime>) {
        self.last_refresh = Some(now);
        self.last_git_mtime = git_index_mtime;
    }
}

/// Maximum rows the worker returns for one query.
///
/// From `ui.popupRows`; see [`crate::settings`].
#[must_use]
pub fn max_rows() -> usize {
    crate::settings::active().ui.popup_rows
}

/// A message from the index worker to the UI.
#[derive(Debug)]
pub enum IndexMsg {
    /// Ranked rows for the query stamped `generation`.
    Results {
        /// The generation of the query these rows answer.
        generation: u64,
        /// Ranked rows, best first.
        rows: Vec<Match>,
    },
    /// The index changed (the untracked fold or a refresh completed).
    Refreshed,
}

/// A query sent to the index worker.
#[derive(Debug)]
struct QueryMsg {
    generation: u64,
    text: String,
}

/// A request to the index worker.
#[derive(Debug)]
enum Req {
    /// Rank `text` and reply with `IndexMsg::Results`.
    Query(QueryMsg),
    /// Replace the extra (MCP resource) candidates mixed into every ranking.
    SetExtra(Vec<Candidate>),
}

/// Owns the file index on its own thread and answers ranked queries.
///
/// Dropping the worker closes the request channel, which ends the thread.
#[derive(Debug)]
pub struct IndexWorker {
    tx: Sender<Req>,
    rx: Receiver<IndexMsg>,
}

impl IndexWorker {
    /// Starts the worker for `root`, mixing `extra` (MCP resources) into every
    /// ranking pass.
    #[must_use]
    pub fn spawn(root: PathBuf, extra: Vec<Candidate>, respect_gitignore: bool) -> Self {
        let (tx, qrx) = channel::<Req>();
        let (mrx_tx, rx) = channel::<IndexMsg>();
        std::thread::spawn(move || {
            let mut index = FileIndex::build(&root, respect_gitignore);
            index.mark_refreshed(Instant::now(), git_index_mtime(&root));
            // Untracked files are slower to enumerate; fold them in once the
            // tracked set is already answerable.
            index.fold_untracked(&root, respect_gitignore);
            if mrx_tx.send(IndexMsg::Refreshed).is_err() {
                return;
            }
            let mut extra = extra;
            while let Ok(req) = qrx.recv() {
                let q = match req {
                    Req::Query(q) => q,
                    Req::SetExtra(e) => {
                        extra = e;
                        continue;
                    }
                };
                let now = Instant::now();
                let mtime = git_index_mtime(&root);
                if index.needs_refresh(now, mtime) {
                    let fresh = FileIndex::build(&root, respect_gitignore);
                    // Equal signature means the rebuild is a no-op; keep the
                    // existing index (which already holds untracked files).
                    if fresh.signature() == index.signature() {
                        index.mark_refreshed(now, mtime);
                    } else {
                        index = fresh;
                        index.mark_refreshed(now, mtime);
                        index.fold_untracked(&root, respect_gitignore);
                        if mrx_tx.send(IndexMsg::Refreshed).is_err() {
                            return;
                        }
                    }
                }
                let mut pool: Vec<Candidate> = index.candidates().to_vec();
                pool.extend(extra.iter().cloned());
                let rows = rank(&q.text, &pool, max_rows());
                if mrx_tx
                    .send(IndexMsg::Results {
                        generation: q.generation,
                        rows,
                    })
                    .is_err()
                {
                    return;
                }
            }
        });
        Self { tx, rx }
    }

    /// Requests ranked rows for `text`, stamped with `generation`.
    ///
    /// A dead worker is ignored; the popup simply shows nothing.
    pub fn query(&self, generation: u64, text: &str) {
        let _ = self.tx.send(Req::Query(QueryMsg {
            generation,
            text: text.to_string(),
        }));
    }

    /// Replaces the extra (MCP resource) candidates for later queries.
    ///
    /// Without this the resource list would stay frozen at whatever was live
    /// when the worker started, so a server connecting later would never
    /// contribute completions (issue #41).
    pub fn set_extra(&self, extra: Vec<Candidate>) {
        let _ = self.tx.send(Req::SetExtra(extra));
    }

    /// Takes one pending message, if any.
    #[must_use]
    pub fn try_recv(&self) -> Option<IndexMsg> {
        self.rx.try_recv().ok()
    }
}

/// Wraps `text` in double quotes when it contains whitespace.
#[must_use]
pub fn quote_if_needed(text: &str) -> String {
    if text.contains(char::is_whitespace) {
        format!("\"{text}\"")
    } else {
        text.to_string()
    }
}

/// What the caller should do after [`Popup::handle_key`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopupAction {
    /// The popup handled the key and stays open.
    Consumed,
    /// The popup handled the key and should now be closed.
    Dismissed,
    /// Not a popup key; run the caller's normal binding.
    Passthrough,
}

/// Open suggestion popup state.
#[derive(Debug)]
pub struct Popup {
    token: AtToken,
    rows: Vec<Match>,
    selected: usize,
    generation: u64,
}

impl Popup {
    /// Opens a popup for `token` with no rows yet.
    #[must_use]
    pub fn new(token: AtToken) -> Self {
        Self {
            token,
            rows: Vec::new(),
            selected: 0,
            generation: 1,
        }
    }

    /// The generation of the query currently in flight.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Retargets the popup at `token` and returns the new generation to query
    /// with. Results stamped with any earlier generation are then discarded.
    pub fn bump_generation(&mut self, token: AtToken) -> u64 {
        self.token = token;
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    /// Applies a worker message. Returns false when the message was stale (or
    /// not results) and nothing changed.
    pub fn accept_msg(&mut self, msg: IndexMsg) -> bool {
        let IndexMsg::Results { generation, rows } = msg else {
            return false;
        };
        if generation != self.generation {
            return false;
        }
        self.rows = rows;
        self.selected = 0;
        true
    }

    /// The rows currently displayed, capped at [`max_rows`] by the worker.
    #[must_use]
    pub fn rows(&self) -> &[Match] {
        &self.rows
    }

    /// Index of the highlighted row.
    #[must_use]
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Replaces the `@` token in `buf` with `text`, plus `suffix`.
    fn replace_token(&self, buf: &mut LineBuffer, text: &str, suffix: &str) {
        let end = buf.cursor();
        buf.replace_range(self.token.start, end, format!("{text}{suffix}"));
    }

    /// Handles one key while the popup is open.
    ///
    /// Tab inserts the longest common prefix and keeps the popup open. Enter
    /// accepts the selection, replacing the whole token including the `@`; a
    /// directory keeps the popup open for drill-down, anything else closes it
    /// with exactly one trailing space. Esc dismisses without touching `buf`.
    pub fn handle_key(&mut self, key: KeyEvent, buf: &mut LineBuffer) -> PopupAction {
        match key.code {
            KeyCode::Esc => PopupAction::Dismissed,
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                PopupAction::Consumed
            }
            KeyCode::Down => {
                if self.selected + 1 < self.rows.len() {
                    self.selected += 1;
                }
                PopupAction::Consumed
            }
            KeyCode::Tab => {
                // Only rows that literally begin with the query can contribute
                // to a *prefix* completion. Matching is fuzzy, so the popup
                // legitimately holds rows that do not — `@Carg` matches
                // `docs/SHARED-ENGINE-DESIGN.md` — and letting those vote
                // empties the common prefix and makes Tab do nothing.
                let prefixed: Vec<Match> = self
                    .rows
                    .iter()
                    .filter(|m| m.text.starts_with(self.token.query.as_str()))
                    .cloned()
                    .collect();
                let lcp = longest_common_prefix(if prefixed.is_empty() {
                    &self.rows
                } else {
                    &prefixed
                });
                // Tab must only ever *extend* what was typed: inserting a
                // shorter prefix would shrink `@docs/ARCH` to `@d`, silently
                // eating the user's input.
                if lcp.len() <= self.token.query.len() || !lcp.starts_with(&self.token.query) {
                    return PopupAction::Consumed;
                }
                let quote = if self.token.quoted { "\"" } else { "" };
                self.replace_token(buf, &format!("@{quote}{lcp}"), "");
                PopupAction::Consumed
            }
            KeyCode::Enter => {
                let Some(sel) = self.rows.get(self.selected) else {
                    return PopupAction::Dismissed;
                };
                let (text, kind) = (sel.text.clone(), sel.kind);
                if kind == Kind::Dir {
                    // Keep the `@` and the popup so the user can drill down.
                    let quote = if self.token.quoted { "\"" } else { "" };
                    self.replace_token(buf, &format!("@{quote}{text}"), "");
                    return PopupAction::Consumed;
                }
                self.replace_token(buf, &quote_if_needed(&text), " ");
                PopupAction::Dismissed
            }
            _ => PopupAction::Passthrough,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::LineBuffer;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn popup_with(rows: &[(&str, Kind)], token_text: &str) -> Popup {
        let token = detect_at_token(token_text).expect("token");
        let mut p = Popup::new(token);
        let r#gen = p.generation();
        p.accept_msg(IndexMsg::Results {
            generation: r#gen,
            rows: rows
                .iter()
                .map(|(t, k)| Match {
                    text: (*t).to_string(),
                    kind: *k,
                    score: 0,
                })
                .collect(),
        });
        p
    }

    #[test]
    fn stale_generation_results_are_dropped() {
        let mut p = popup_with(&[("src/ui.rs", Kind::File)], "@ui");
        let stale = p.generation().wrapping_sub(1);
        let accepted = p.accept_msg(IndexMsg::Results {
            generation: stale,
            rows: vec![Match {
                text: "STALE".to_string(),
                kind: Kind::File,
                score: 0,
            }],
        });
        assert!(!accepted, "stale results must be rejected");
        assert_eq!(p.rows()[0].text, "src/ui.rs");
    }

    #[test]
    fn up_and_down_move_the_selection() {
        let mut p = popup_with(
            &[("a", Kind::File), ("b", Kind::File), ("c", Kind::File)],
            "@x",
        );
        let mut buf = LineBuffer::new();
        assert_eq!(p.selected(), 0);
        assert!(matches!(
            p.handle_key(key(KeyCode::Down), &mut buf),
            PopupAction::Consumed
        ));
        assert_eq!(p.selected(), 1);
        assert!(matches!(
            p.handle_key(key(KeyCode::Up), &mut buf),
            PopupAction::Consumed
        ));
        assert_eq!(p.selected(), 0);
        // Up at the top stays at the top.
        p.handle_key(key(KeyCode::Up), &mut buf);
        assert_eq!(p.selected(), 0);
    }

    #[test]
    fn tab_inserts_the_longest_common_prefix_and_keeps_the_popup_open() {
        let mut buf = LineBuffer::new();
        buf.set_text("@src/uti");
        buf.move_end();
        let mut p = popup_with(
            &[("src/utils", Kind::File), ("src/utilities", Kind::File)],
            "@src/uti",
        );
        assert!(matches!(
            p.handle_key(key(KeyCode::Tab), &mut buf),
            PopupAction::Consumed
        ));
        assert_eq!(buf.text(), "@src/util");
    }

    #[test]
    fn enter_replaces_the_token_including_the_at_and_adds_one_space() {
        let mut buf = LineBuffer::new();
        buf.set_text("look at @ui");
        buf.move_end();
        let mut p = popup_with(&[("src/ui.rs", Kind::File)], "look at @ui");
        assert!(matches!(
            p.handle_key(key(KeyCode::Enter), &mut buf),
            PopupAction::Dismissed
        ));
        assert_eq!(buf.text(), "look at src/ui.rs ");
    }

    #[test]
    fn enter_on_a_directory_keeps_the_popup_open_with_a_trailing_slash() {
        let mut buf = LineBuffer::new();
        buf.set_text("@src");
        buf.move_end();
        let mut p = popup_with(&[("src/", Kind::Dir)], "@src");
        assert!(matches!(
            p.handle_key(key(KeyCode::Enter), &mut buf),
            PopupAction::Consumed
        ));
        assert_eq!(buf.text(), "@src/");
    }

    #[test]
    fn enter_on_a_path_with_spaces_quotes_the_result() {
        let mut buf = LineBuffer::new();
        buf.set_text("@\"two wor");
        buf.move_end();
        let mut p = popup_with(&[("two words.txt", Kind::File)], "@\"two wor");
        p.handle_key(key(KeyCode::Enter), &mut buf);
        assert_eq!(buf.text(), "\"two words.txt\" ");
    }

    #[test]
    fn enter_inserts_an_mcp_resource_token_verbatim() {
        let mut buf = LineBuffer::new();
        buf.set_text("@tolaria");
        buf.move_end();
        let mut p = popup_with(&[("tolaria:note://b", Kind::Resource)], "@tolaria");
        p.handle_key(key(KeyCode::Enter), &mut buf);
        assert_eq!(buf.text(), "tolaria:note://b ");
    }

    #[test]
    fn esc_dismisses_and_leaves_the_text_untouched() {
        let mut buf = LineBuffer::new();
        buf.set_text("@src/ui");
        buf.move_end();
        let mut p = popup_with(&[("src/ui.rs", Kind::File)], "@src/ui");
        assert!(matches!(
            p.handle_key(key(KeyCode::Esc), &mut buf),
            PopupAction::Dismissed
        ));
        assert_eq!(buf.text(), "@src/ui");
    }

    #[test]
    fn other_keys_pass_through() {
        let mut buf = LineBuffer::new();
        let mut p = popup_with(&[("a", Kind::File)], "@a");
        assert!(matches!(
            p.handle_key(key(KeyCode::Char('x')), &mut buf),
            PopupAction::Passthrough
        ));
    }

    #[test]
    fn quotes_only_when_the_path_has_a_space() {
        assert_eq!(quote_if_needed("src/ui.rs"), "src/ui.rs");
        assert_eq!(quote_if_needed("two words.txt"), "\"two words.txt\"");
    }

    fn file(t: &str) -> Candidate {
        Candidate {
            text: t.to_string(),
            kind: Kind::File,
            demoted: false,
        }
    }

    #[test]
    fn empty_query_returns_everything_up_to_limit() {
        let c = vec![file("a"), file("b"), file("c")];
        assert_eq!(rank("", &c, 2).len(), 2);
    }

    #[test]
    fn requires_subsequence_match() {
        let c = vec![file("src/ui.rs"), file("Cargo.toml")];
        let m = rank("sui", &c, 15);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].text, "src/ui.rs");
    }

    #[test]
    fn matching_is_case_insensitive() {
        let c = vec![file("src/Cargo.toml")];
        assert_eq!(rank("cargo", &c, 15).len(), 1);
    }

    #[test]
    fn basename_hit_outranks_a_directory_hit() {
        let c = vec![file("viz/other.rs"), file("src/viz.rs")];
        let m = rank("viz", &c, 15);
        assert_eq!(m[0].text, "src/viz.rs");
    }

    #[test]
    fn shorter_path_wins_a_tie() {
        let c = vec![file("src/deep/nested/ui.rs"), file("src/ui.rs")];
        let m = rank("ui.rs", &c, 15);
        assert_eq!(m[0].text, "src/ui.rs");
    }

    #[test]
    fn files_outrank_resources_on_equal_score() {
        let c = vec![
            Candidate {
                text: "notes".to_string(),
                kind: Kind::Resource,
                demoted: false,
            },
            Candidate {
                text: "notes".to_string(),
                kind: Kind::File,
                demoted: false,
            },
        ];
        let m = rank("notes", &c, 15);
        assert_eq!(m[0].kind, Kind::File);
    }

    #[test]
    fn submodule_paths_sort_below_every_project_path() {
        // The submodule path is the *better* textual match: exact basename,
        // consecutive run. Demotion must still bury it (issue #45).
        let mut sub = file("refs/ds4/ui.rs");
        sub.demoted = true;
        let c = vec![sub, file("src/deep/nested/elsewhere/ui_helpers.rs")];
        let m = rank("ui.rs", &c, 15);
        assert_eq!(m.len(), 1, "a project match hides submodule rows: {m:?}");
        assert_eq!(m[0].text, "src/deep/nested/elsewhere/ui_helpers.rs");
    }

    #[test]
    fn tab_extends_past_a_fuzzy_row_that_shares_no_prefix() {
        // `@Carg` also fuzzy-matches docs/SHARED-ENGINE-DESIGN.md (c-a-r-g).
        // Letting that row vote on the common prefix empties it, so Tab did
        // nothing at all; only prefix matches may contribute.
        let mut p = Popup::new(AtToken {
            start: 0,
            query: "Carg".to_string(),
            quoted: false,
        });
        p.accept_msg(IndexMsg::Results {
            generation: p.generation(),
            rows: rank(
                "Carg",
                &[
                    file("Cargo.toml"),
                    file("Cargo.lock"),
                    file("docs/SHARED-ENGINE-DESIGN.md"),
                ],
                15,
            ),
        });
        let mut buf = LineBuffer::new();
        buf.insert("@Carg");
        p.handle_key(key(KeyCode::Tab), &mut buf);
        assert_eq!(buf.text(), "@Cargo.");
    }

    #[test]
    fn tab_never_shortens_what_was_typed() {
        // No row begins with the query, so the prefix subset is empty and the
        // common prefix of the fuzzy rows is shorter than what was typed.
        // Inserting it would eat the user's input, so Tab must do nothing.
        let mut p = Popup::new(AtToken {
            start: 0,
            query: "docs/ARCH".to_string(),
            quoted: false,
        });
        p.accept_msg(IndexMsg::Results {
            generation: p.generation(),
            rows: rank("docs/ARCH", &[file("d/o/c/s/A/R/C/H.rs")], 15),
        });
        let mut buf = LineBuffer::new();
        buf.insert("@docs/ARCH");
        p.handle_key(key(KeyCode::Tab), &mut buf);
        assert_eq!(buf.text(), "@docs/ARCH");
    }

    #[test]
    fn tab_completes_a_unique_prefix_match_in_full() {
        let mut p = Popup::new(AtToken {
            start: 0,
            query: "docs/ARCH".to_string(),
            quoted: false,
        });
        p.accept_msg(IndexMsg::Results {
            generation: p.generation(),
            rows: rank(
                "docs/ARCH",
                &[file("docs/ARCHITECTURE.md"), file("d/o/c/s/A/R/C/H.rs")],
                15,
            ),
        });
        let mut buf = LineBuffer::new();
        buf.insert("@docs/ARCH");
        p.handle_key(key(KeyCode::Tab), &mut buf);
        assert_eq!(buf.text(), "@docs/ARCHITECTURE.md");
    }

    #[test]
    fn submodule_rows_do_not_poison_the_tab_prefix() {
        // The regression behind issue #45's second symptom: submodule rows
        // shared no prefix with the project's, so Tab could never extend.
        let mut sub = file("refs/ds4/Cargo.toml");
        sub.demoted = true;
        let c = vec![sub, file("Cargo.toml"), file("Cargo.lock")];
        let m = rank("Cargo", &c, 15);
        assert_eq!(longest_common_prefix(&m), "Cargo.");
    }

    #[test]
    fn submodule_paths_remain_reachable() {
        let mut sub = file("refs/ds4/agent.c");
        sub.demoted = true;
        let m = rank("agent.c", &[sub], 15);
        assert_eq!(m.len(), 1, "demotion must not filter, only reorder");
    }

    #[test]
    fn gitmodules_paths_are_read_with_a_trailing_slash() {
        let dir = temp_repo(&["a.txt"]);
        std::fs::write(
            dir.join(".gitmodules"),
            "[submodule \"refs/ds4\"]\n\tpath = refs/ds4\n\turl = https://example.invalid\n",
        )
        .unwrap();
        assert_eq!(submodule_prefixes(&dir), vec!["refs/ds4/".to_string()]);
    }

    #[test]
    fn index_marks_submodule_files_demoted() {
        let dir = temp_repo(&["src/ui.rs", "refs/ds4/ui.rs"]);
        std::fs::write(
            dir.join(".gitmodules"),
            "[submodule \"refs/ds4\"]\n\tpath = refs/ds4\n",
        )
        .unwrap();
        let idx = FileIndex::build(&dir, true);
        let by = |t: &str| idx.candidates().iter().find(|c| c.text == t).cloned();
        assert!(
            by("refs/ds4/ui.rs")
                .expect("submodule file indexed")
                .demoted
        );
        assert!(!by("src/ui.rs").expect("project file indexed").demoted);
        assert!(by("refs/ds4/").expect("submodule dir indexed").demoted);
    }

    #[test]
    fn longest_common_prefix_of_candidates() {
        let m = vec![
            Match {
                text: "src/utils".to_string(),
                kind: Kind::File,
                score: 0,
            },
            Match {
                text: "src/utilities".to_string(),
                kind: Kind::File,
                score: 0,
            },
        ];
        assert_eq!(longest_common_prefix(&m), "src/util");
    }

    #[test]
    fn longest_common_prefix_of_one_is_the_whole_string() {
        let m = vec![Match {
            text: "src/ui.rs".to_string(),
            kind: Kind::File,
            score: 0,
        }];
        assert_eq!(longest_common_prefix(&m), "src/ui.rs");
    }

    #[test]
    fn longest_common_prefix_of_none_is_empty() {
        assert_eq!(longest_common_prefix(&[]), "");
    }

    #[test]
    fn detects_at_at_start_and_after_whitespace() {
        let t = detect_at_token("@src").expect("start of line");
        assert_eq!(t.start, 0);
        assert_eq!(t.query, "src");
        assert!(!t.quoted);

        let t = detect_at_token("look at @src/ui").expect("after whitespace");
        assert_eq!(t.start, 8);
        assert_eq!(t.query, "src/ui");
    }

    #[test]
    fn ignores_at_mid_word() {
        assert!(detect_at_token("user@host").is_none());
        assert!(detect_at_token("mail me at foo@bar.com").is_none());
    }

    #[test]
    fn ignores_shell_escape_line() {
        assert!(detect_at_token("!ls @src").is_none());
        assert!(detect_at_token("!@src").is_none());
    }

    #[test]
    fn detects_quoted_token() {
        let t = detect_at_token("open @\"two wor").expect("quoted");
        assert_eq!(t.start, 5);
        assert_eq!(t.query, "two wor");
        assert!(t.quoted);
    }

    #[test]
    fn shell_escape_only_suppresses_its_own_line() {
        // Multiline prompt: a first line starting `!` must not disable `@`
        // completion for every later line.
        let t = detect_at_token("!ls\n@src").expect("second line completes");
        assert_eq!(t.query, "src");
        assert!(detect_at_token("hello\n!ls @src").is_none());
    }

    #[test]
    fn a_dot_slash_query_matches_repo_relative_paths() {
        let c = vec![file("src/complete.rs"), file("Cargo.toml")];
        let m = rank("./src", &c, 15);
        assert!(
            m.iter().any(|r| r.text == "src/complete.rs"),
            "`@./src` must not return an empty popup: {m:?}"
        );
    }

    #[test]
    fn a_tilde_query_expands_against_home() {
        // Expansion lands outside the index root here, so it degrades to the
        // repo-relative remainder rather than matching nothing.
        let c = vec![file("src/complete.rs")];
        let m = rank("~/src/complete.rs", &c, 15);
        assert!(
            m.iter().any(|r| r.text == "src/complete.rs"),
            "`@~/` must not return an empty popup: {m:?}"
        );
        assert_eq!(normalize_query("~/"), "");
        assert_eq!(normalize_query("./a/b"), "a/b");
    }

    #[test]
    fn no_token_without_at() {
        assert!(detect_at_token("plain text").is_none());
        assert!(detect_at_token("").is_none());
    }

    /// Creates a git repo under a unique temp dir with the given files.
    fn temp_repo(files: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "plank-complete-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for f in files {
            let p = dir.join(f);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&p, b"x").unwrap();
        }
        for args in [
            vec!["init", "-q"],
            vec!["config", "user.email", "t@example.com"],
            vec!["config", "user.name", "t"],
            vec!["add", "-A"],
            vec!["commit", "-qm", "init"],
        ] {
            Command::new("git")
                .args(&args)
                .current_dir(&dir)
                .output()
                .unwrap();
        }
        dir
    }

    #[test]
    fn indexes_tracked_files_and_parent_dirs() {
        let dir = temp_repo(&["src/ui.rs", "Cargo.toml"]);
        let idx = FileIndex::build(&dir, true);
        let texts: Vec<&str> = idx.candidates().iter().map(|c| c.text.as_str()).collect();
        assert!(texts.contains(&"src/ui.rs"), "{texts:?}");
        assert!(texts.contains(&"Cargo.toml"), "{texts:?}");
        assert!(texts.contains(&"src/"), "parent dir indexed: {texts:?}");
        let dirs: Vec<&Candidate> = idx
            .candidates()
            .iter()
            .filter(|c| c.kind == Kind::Dir)
            .collect();
        assert!(dirs.iter().all(|d| d.text.ends_with('/')));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn never_indexes_dot_git() {
        let dir = temp_repo(&["src/ui.rs"]);
        let idx = FileIndex::build(&dir, true);
        assert!(
            !idx.candidates().iter().any(|c| c.text.starts_with(".git/")),
            "no .git entries"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn untracked_file_appears_after_the_background_fold() {
        let dir = temp_repo(&["src/ui.rs"]);
        std::fs::write(dir.join("scratch.txt"), b"x").unwrap();
        let mut idx = FileIndex::build(&dir, true);
        assert!(!idx.candidates().iter().any(|c| c.text == "scratch.txt"));
        idx.fold_untracked(&dir, true);
        assert!(idx.candidates().iter().any(|c| c.text == "scratch.txt"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn respect_gitignore_hides_ignored_files() {
        let dir = temp_repo(&["src/ui.rs"]);
        std::fs::write(dir.join(".gitignore"), b"ignored.txt\n").unwrap();
        std::fs::write(dir.join("ignored.txt"), b"x").unwrap();

        let mut on = FileIndex::build(&dir, true);
        on.fold_untracked(&dir, true);
        assert!(!on.candidates().iter().any(|c| c.text == "ignored.txt"));

        let mut off = FileIndex::build(&dir, false);
        off.fold_untracked(&dir, false);
        assert!(off.candidates().iter().any(|c| c.text == "ignored.txt"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn identical_file_lists_produce_identical_signatures() {
        let dir = temp_repo(&["src/ui.rs", "Cargo.toml"]);
        let a = FileIndex::build(&dir, true);
        let b = FileIndex::build(&dir, true);
        assert_eq!(a.signature(), b.signature());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn signature_changes_when_the_file_list_changes() {
        let dir = temp_repo(&["src/ui.rs"]);
        let a = FileIndex::build(&dir, true);
        std::fs::write(dir.join("new.rs"), b"x").unwrap();
        Command::new("git")
            .args(["add", "-A"])
            .current_dir(&dir)
            .output()
            .unwrap();
        let b = FileIndex::build(&dir, true);
        assert_ne!(a.signature(), b.signature());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn throttle_blocks_a_refresh_before_five_seconds() {
        let dir = temp_repo(&["src/ui.rs"]);
        let now = Instant::now();
        let mut idx = FileIndex::build(&dir, true);
        let mtime = git_index_mtime(&dir);
        idx.mark_refreshed(now, mtime);
        assert!(!idx.needs_refresh(now + Duration::from_secs(1), mtime));
        assert!(idx.needs_refresh(now + Duration::from_secs(6), mtime));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_changed_git_index_mtime_bypasses_the_throttle() {
        let dir = temp_repo(&["src/ui.rs"]);
        let now = Instant::now();
        let mut idx = FileIndex::build(&dir, true);
        idx.mark_refreshed(now, git_index_mtime(&dir));
        let moved = Some(std::time::SystemTime::now() + Duration::from_mins(1));
        assert!(idx.needs_refresh(now + Duration::from_millis(10), moved));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn falls_back_to_ripgrep_outside_a_git_repo() {
        // The fallback only ever fires when `rg` is on PATH, so on a machine
        // without it (CI runners, typically) there is nothing to assert.
        if Command::new("rg").arg("--version").output().is_err() {
            eprintln!("skipping: ripgrep not installed");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "plank-nogit-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/loose.txt"), b"x").unwrap();
        let idx = FileIndex::build(&dir, true);
        assert!(
            idx.candidates().iter().any(|c| c.text == "sub/loose.txt"),
            "{:?}",
            idx.candidates()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Blocks until the worker sends a message, or panics after 10s.
    fn recv_blocking(w: &IndexWorker) -> IndexMsg {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(m) = w.try_recv() {
                return m;
            }
            assert!(Instant::now() < deadline, "worker produced nothing");
            std::thread::yield_now();
        }
    }

    #[test]
    fn worker_answers_a_query_with_a_matching_generation() {
        let dir = temp_repo(&["src/ui.rs"]);
        let w = IndexWorker::spawn(dir.clone(), Vec::new(), true);
        w.query(7, "ui");
        loop {
            match recv_blocking(&w) {
                IndexMsg::Results { generation, rows } => {
                    assert_eq!(generation, 7);
                    assert!(rows.iter().any(|r| r.text == "src/ui.rs"), "{rows:?}");
                    break;
                }
                IndexMsg::Refreshed => {}
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn worker_reports_the_untracked_fold_with_refreshed() {
        let dir = temp_repo(&["src/ui.rs"]);
        std::fs::write(dir.join("scratch.txt"), b"x").unwrap();
        let w = IndexWorker::spawn(dir.clone(), Vec::new(), true);
        // Await the fold rather than sleeping.
        loop {
            if matches!(recv_blocking(&w), IndexMsg::Refreshed) {
                break;
            }
        }
        w.query(1, "scratch");
        loop {
            if let IndexMsg::Results { rows, .. } = recv_blocking(&w) {
                assert!(rows.iter().any(|r| r.text == "scratch.txt"), "{rows:?}");
                break;
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn set_extra_makes_a_late_server_completable() {
        // Issue #41: the worker starts with no resources, as it does when the
        // first `@` precedes an MCP server finishing its handshake.
        let dir = temp_repo(&["src/ui.rs"]);
        let w = IndexWorker::spawn(dir.clone(), Vec::new(), true);
        w.set_extra(vec![Candidate {
            text: "latecomer:note://z".to_string(),
            kind: Kind::Resource,
            demoted: false,
        }]);
        w.query(1, "latecomer");
        loop {
            if let IndexMsg::Results { rows, .. } = recv_blocking(&w) {
                assert!(
                    rows.iter().any(|r| r.text == "latecomer:note://z"),
                    "{rows:?}"
                );
                break;
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn worker_includes_extra_candidates() {
        let dir = temp_repo(&["src/ui.rs"]);
        let extra = vec![Candidate {
            text: "tolaria:note://b".to_string(),
            kind: Kind::Resource,
            demoted: false,
        }];
        let w = IndexWorker::spawn(dir.clone(), extra, true);
        w.query(1, "tolaria");
        loop {
            if let IndexMsg::Results { rows, .. } = recv_blocking(&w) {
                assert!(
                    rows.iter().any(|r| r.text == "tolaria:note://b"),
                    "{rows:?}"
                );
                break;
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
