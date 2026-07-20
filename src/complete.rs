//! `@` typeahead completion for the Ratatui TUI prompt.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

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
    if left.starts_with('!') {
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
}

/// A candidate that survived ranking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    /// The text inserted when accepted.
    pub text: String,
    /// What the candidate refers to.
    pub kind: Kind,
    /// Higher sorts first.
    pub score: i32,
}

/// Bonus for a match landing right after a path separator.
const BONUS_SEGMENT: i32 = 12;
/// Bonus for each additional consecutively matched character.
const BONUS_CONSECUTIVE: i32 = 8;
/// Bonus for the match lying entirely within the basename.
const BONUS_BASENAME: i32 = 20;
/// Files and directories outrank MCP resources at equal quality.
const BONUS_FILE_KIND: i32 = 5;

/// Scores `text` against `query` as a case-insensitive subsequence.
///
/// Returns `None` when `query` is not a subsequence of `text`. Consecutive
/// runs, matches at a path-segment boundary, and matches inside the basename
/// all raise the score; longer paths are penalised so shorter ones win ties.
fn score_one(query: &str, text: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let hay: Vec<char> = text.to_lowercase().chars().collect();
    let needle: Vec<char> = query.to_lowercase().chars().collect();
    let basename_start = text.rfind('/').map_or(0, |i| i + 1);
    let mut score = 0;
    let mut hi = 0usize;
    let mut last_hit: Option<usize> = None;
    let mut first_hit: Option<usize> = None;
    for &n in &needle {
        loop {
            let h = *hay.get(hi)?;
            hi += 1;
            if h == n {
                break;
            }
        }
        let pos = hi - 1;
        if first_hit.is_none() {
            first_hit = Some(pos);
        }
        if last_hit == Some(pos.wrapping_sub(1)) {
            score += BONUS_CONSECUTIVE;
        }
        if pos == 0 || hay.get(pos.wrapping_sub(1)) == Some(&'/') {
            score += BONUS_SEGMENT;
        }
        last_hit = Some(pos);
    }
    if first_hit.is_some_and(|f| f >= basename_start) {
        score += BONUS_BASENAME;
    }
    // Penalise length so a shorter path wins an otherwise equal contest.
    score -= i32::try_from(text.chars().count()).unwrap_or(i32::MAX);
    Some(score)
}

/// Ranks `cands` against `query`, best first, truncated to `limit`.
#[must_use]
pub fn rank(query: &str, cands: &[Candidate], limit: usize) -> Vec<Match> {
    let mut out: Vec<Match> = cands
        .iter()
        .filter_map(|c| {
            let base = score_one(query, &c.text)?;
            let kind_bonus = if c.kind == Kind::Resource {
                0
            } else {
                BONUS_FILE_KIND
            };
            Some(Match {
                text: c.text.clone(),
                kind: c.kind,
                score: base + kind_bonus,
            })
        })
        .collect();
    // Stable ordering: score descending, then lexicographic for determinism.
    out.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.text.cmp(&b.text)));
    out.truncate(limit);
    out
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
const REFRESH_THROTTLE: Duration = Duration::from_secs(5);
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

/// Reads `respectGitignore` from `~/.plank/settings.json` then
/// `<cwd>/.plank/settings.json`, later file winning. Defaults to `true`.
#[must_use]
pub fn respect_gitignore_setting() -> bool {
    use crate::tools::mcp::json_parse;
    let mut value = true;
    let mut paths = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        paths.push(PathBuf::from(home).join(".plank").join("settings.json"));
    }
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join(".plank").join("settings.json"));
    }
    for p in paths {
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        if let Some(root) = json_parse(&text)
            && let Some(crate::tools::mcp::Json::Bool(b)) = root.get("respectGitignore")
        {
            value = *b;
        }
    }
    value
}

/// An index of completable paths under one root.
#[derive(Debug)]
pub struct FileIndex {
    paths: BTreeSet<String>,
    cands: Vec<Candidate>,
    signature: u64,
    last_refresh: Option<Instant>,
    last_git_mtime: Option<SystemTime>,
}

impl FileIndex {
    /// Builds an index of tracked files under `root`.
    ///
    /// Uses `git ls-files --recurse-submodules` inside a git tree and
    /// `rg --files` outside one. Untracked files are folded in separately by
    /// [`FileIndex::fold_untracked`].
    #[must_use]
    pub fn build(root: &Path, respect_gitignore: bool) -> Self {
        let paths: BTreeSet<String> = if is_git_repo(root) {
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
        };
        idx.rebuild_candidates();
        idx
    }

    /// Folds untracked files into an already-built index.
    ///
    /// Honours `.gitignore` when `respect_gitignore` is set.
    pub fn fold_untracked(&mut self, root: &Path, respect_gitignore: bool) {
        if !is_git_repo(root) {
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
        self.cands = dirs
            .into_iter()
            .map(|text| Candidate {
                text,
                kind: Kind::Dir,
            })
            .chain(self.paths.iter().map(|p| Candidate {
                text: p.clone(),
                kind: Kind::File,
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
            .is_none_or(|t| now.duration_since(t) >= REFRESH_THROTTLE)
    }

    /// Records that a refresh just happened.
    pub fn mark_refreshed(&mut self, now: Instant, git_index_mtime: Option<SystemTime>) {
        self.last_refresh = Some(now);
        self.last_git_mtime = git_index_mtime;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(t: &str) -> Candidate {
        Candidate {
            text: t.to_string(),
            kind: Kind::File,
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
            },
            Candidate {
                text: "notes".to_string(),
                kind: Kind::File,
            },
        ];
        let m = rank("notes", &c, 15);
        assert_eq!(m[0].kind, Kind::File);
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
        let moved = Some(std::time::SystemTime::now() + Duration::from_secs(60));
        assert!(idx.needs_refresh(now + Duration::from_millis(10), moved));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn falls_back_to_ripgrep_outside_a_git_repo() {
        let dir = std::env::temp_dir().join(format!("plank-nogit-{}", std::process::id()));
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
}
