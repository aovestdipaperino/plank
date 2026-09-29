//! What changed between two CSV texts, in counts only: the model is told how
//! many rows changed, never what they hold.

/// LCS is only run on a middle whose cell count (`rows_before *
/// rows_after`) is at most this, roughly a 500x500 square. `lcs_pairs`
/// allocates and fills an `(n+1)x(m+1)` table, so an unbounded middle is a
/// blocking `frame_close` paying O(n*m) time and memory: at csvedit's
/// 10,000-row ceiling that is ~400 MB. A common prefix and suffix are
/// trimmed before this check, so the limit only bites a middle that is
/// itself large — the common case (a handful of edited rows in a big file)
/// trims down to a tiny middle and always gets the exact alignment.
const LCS_CELL_LIMIT: usize = 250_000;

/// Row counts between `before` and `after`. A common prefix and a common
/// suffix of identical rows are trimmed first (cheap, and the common case).
/// The remaining middle is aligned by longest common subsequence when small
/// enough (see [`LCS_CELL_LIMIT`]) — so a row that survives unmoved, even
/// sandwiched between edits elsewhere in the middle, is never miscounted as
/// a change — or, above that limit, paired off by position: `changed` is
/// the shorter side and the remainder is `added`/`deleted`, which slightly
/// over-counts `changed` on a large edit but never runs an unbounded table.
/// Header changes are reported as `columns changed`.
#[must_use]
pub fn summarize(before: &str, after: &str) -> String {
    // Counts lines, not CSV records: a quoted field with an embedded newline
    // counts as two rows (a known limitation; the summary is not a parser).
    let rows = |t: &str| t.lines().map(str::to_owned).collect::<Vec<_>>();
    let (b, a) = (rows(before), rows(after));
    let mut parts = Vec::new();
    if b.is_empty() && !a.is_empty() {
        parts.push("new file".to_string());
    } else if b.first() != a.first() {
        return "columns changed".to_string();
    }
    let (b, a) = (b.get(1..).unwrap_or(&[]), a.get(1..).unwrap_or(&[]));

    let prefix = b.iter().zip(a).take_while(|(x, y)| x == y).count();
    let (b, a) = (&b[prefix..], &a[prefix..]);
    let suffix = b
        .iter()
        .rev()
        .zip(a.iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let (b, a) = (&b[..b.len() - suffix], &a[..a.len() - suffix]);

    let (mut changed, mut added, mut deleted) = (0usize, 0usize, 0usize);
    let mut gap = |gb: usize, ga: usize| {
        changed += gb.min(ga);
        if ga > gb {
            added += ga - gb;
        } else {
            deleted += gb - ga;
        }
    };
    if b.len().saturating_mul(a.len()) <= LCS_CELL_LIMIT {
        let (mut pi, mut pj) = (0usize, 0usize);
        for (mi, mj) in lcs_pairs(b, a) {
            gap(mi - pi, mj - pj);
            (pi, pj) = (mi + 1, mj + 1);
        }
        gap(b.len() - pi, a.len() - pj);
    } else {
        // The middle is too big to align exactly without an unbounded
        // table: pair it off by position instead, same as a single LCS gap
        // spanning the whole middle would.
        gap(b.len(), a.len());
    }

    let count = |n: usize, what: &str| format!("{n} row{} {what}", if n == 1 { "" } else { "s" });
    if changed > 0 {
        parts.push(count(changed, "changed"));
    }
    if added > 0 {
        parts.push(count(added, "added"));
    }
    if deleted > 0 {
        parts.push(count(deleted, "deleted"));
    }
    if parts.is_empty() {
        "no changes".to_string()
    } else {
        parts.join(", ")
    }
}

/// Indices `(i, j)` of rows in `b`/`a` that form a longest common
/// subsequence, in increasing order.
fn lcs_pairs(b: &[String], a: &[String]) -> Vec<(usize, usize)> {
    let (n, m) = (b.len(), a.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if b[i - 1] == a[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        if b[i - 1] == a[j - 1] {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

/// The line `tool_resume` returns, from the row `summary` of the last
/// tool-invoked close (`None` when none was recorded), the host's `error`
/// (empty when none) and whether the host `rewrote` the target (its
/// `changed` or `written` flag).
///
/// A rewrite with no row changes is reported as such rather than as `no
/// changes`: [`summarize`] compares lines, which drops a `\r`, and the
/// writer always emits LF, so saving an untouched CRLF file rewrites it on
/// disk while every row compares equal. With no recorded summary the host's
/// own flag is all there is, so a rewrite reads as the host's fallback line.
#[must_use]
pub fn resume_line(path: &str, summary: Option<&str>, error: &str, rewrote: bool) -> String {
    let rows = summary.unwrap_or("no changes");
    if !error.is_empty() {
        format!("error: {rows} in the editor, but {error}")
    } else if rows != "no changes" {
        format!("{rows} in {path}")
    } else if !rewrote {
        format!("no changes to {path}")
    } else if summary.is_some() {
        format!("rewrote {path} (no row changes)")
    } else {
        format!("saved changes to {path}")
    }
}

/// Reads a JSON boolean field out of a flat payload; anything absent or not
/// literally `true` reads as `false`.
#[must_use]
pub fn flag(input: &str, key: &str) -> bool {
    input
        .split_once(&format!("\"{key}\":"))
        .is_some_and(|(_, rest)| rest.trim_start().starts_with("true"))
}

#[cfg(test)]
mod tests {
    use super::{flag, resume_line, summarize};

    #[test]
    fn counts_changed_added_and_deleted_rows_without_values() {
        let before = "name,qty\napple,1\npear,2\nplum,3\n";
        assert_eq!(summarize(before, before), "no changes");
        assert_eq!(
            summarize(before, "name,qty\napple,1\npear,5\nplum,3\n"),
            "1 row changed"
        );
        assert_eq!(
            summarize(before, "name,qty\napple,1\npear,2\nplum,3\nfig,4\nkiwi,5\n"),
            "2 rows added"
        );
        assert_eq!(
            summarize(before, "name,qty\napple,1\nplum,3\n"),
            "1 row deleted"
        );
        let out = summarize(before, "name,qty\napple,9\npear,2\n");
        assert_eq!(out, "1 row changed, 1 row deleted");
        assert!(!out.contains("apple") && !out.contains('9'));
    }

    #[test]
    fn a_new_file_and_a_changed_header_are_reported() {
        assert_eq!(summarize("", "a,b\n1,2\n"), "new file, 1 row added");
        assert_eq!(summarize("a,b\n1,2\n", "a,c\n1,2\n"), "columns changed");
    }

    /// A one-row change deep inside a large file trims down to a tiny
    /// middle, so it stays exact (not the fallback's overcount) and fast:
    /// this must run in well under a second even though it is 10,000 rows.
    #[test]
    fn a_single_change_in_ten_thousand_rows_is_exact_and_fast() {
        let mut before = String::from("name,qty\n");
        for i in 0..10_000 {
            before.push_str(&format!("row{i},1\n"));
        }
        let mut after = before.clone();
        after = after.replace("row5000,1\n", "row5000,9\n");
        assert_eq!(summarize(&before, &after), "1 row changed");
    }

    /// A middle too big for the LCS cell limit (3,000x3,000 = 9,000,000
    /// cells, well above [`super::LCS_CELL_LIMIT`]'s ~250,000) must fall
    /// back to positional pairing rather than building the full DP table —
    /// this asserts it returns promptly, which an unbounded table would not.
    #[test]
    fn a_large_middle_falls_back_without_building_a_big_table() {
        let mut before = String::from("name,qty\n");
        let mut after = String::from("name,qty\n");
        for i in 0..3_000 {
            before.push_str(&format!("before-row{i},1\n"));
            after.push_str(&format!("after-row{i},1\n"));
        }
        // Entirely distinct rows on both sides: the fallback pairs them
        // 1:1 as "changed" since both sides are the same length.
        assert_eq!(summarize(&before, &after), "3000 rows changed");
    }

    #[test]
    fn a_rewrite_with_no_row_changes_says_so() {
        let none = Some("no changes");
        assert_eq!(
            resume_line("a.csv", none, "", true),
            "rewrote a.csv (no row changes)"
        );
        assert_eq!(resume_line("a.csv", none, "", false), "no changes to a.csv");
        assert_eq!(
            resume_line("a.csv", Some("1 row changed"), "", true),
            "1 row changed in a.csv"
        );
        assert_eq!(
            resume_line("a.csv", none, "disk full", true),
            "error: no changes in the editor, but disk full"
        );
    }

    /// No recorded summary (the open was not tool-invoked): the host's flag
    /// decides, and nothing claims row counts it never computed.
    #[test]
    fn without_a_summary_the_hosts_flag_decides() {
        assert_eq!(resume_line("a.csv", None, "", false), "no changes to a.csv");
        assert_eq!(
            resume_line("a.csv", None, "", true),
            "saved changes to a.csv"
        );
    }

    #[test]
    fn flag_reads_only_a_literal_true() {
        let p = r#"{"path": "a.csv", "changed": true, "written": false, "error": null}"#;
        assert!(flag(p, "changed"));
        assert!(!flag(p, "written"));
        assert!(!flag(p, "missing"));
    }
}
