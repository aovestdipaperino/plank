//! `@` typeahead completion for the Ratatui TUI prompt.

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
}
