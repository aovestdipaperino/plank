//! What changed between two CSV texts, in counts only: the model is told how
//! many rows changed, never what they hold.

/// Row counts between `before` and `after`: rows are aligned by longest
/// common subsequence, so a row that survives unmoved (even sandwiched
/// between edits elsewhere) is never miscounted as a change. Each gap
/// between aligned rows is paired off by position (changed), with any
/// remainder added or deleted. Header changes are reported as `columns
/// changed`.
#[must_use]
pub fn summarize(before: &str, after: &str) -> String {
    let rows = |t: &str| t.lines().map(str::to_owned).collect::<Vec<_>>();
    let (b, a) = (rows(before), rows(after));
    let mut parts = Vec::new();
    if b.is_empty() && !a.is_empty() {
        parts.push("new file".to_string());
    } else if b.first() != a.first() {
        return "columns changed".to_string();
    }
    let (b, a) = (
        b.get(1..).unwrap_or(&[]).to_vec(),
        a.get(1..).unwrap_or(&[]).to_vec(),
    );

    let (mut changed, mut added, mut deleted) = (0usize, 0usize, 0usize);
    let mut gap = |gb: usize, ga: usize| {
        changed += gb.min(ga);
        if ga > gb {
            added += ga - gb;
        } else {
            deleted += gb - ga;
        }
    };
    let (mut pi, mut pj) = (0usize, 0usize);
    for (mi, mj) in lcs_pairs(&b, &a) {
        gap(mi - pi, mj - pj);
        (pi, pj) = (mi + 1, mj + 1);
    }
    gap(b.len() - pi, a.len() - pj);

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

#[cfg(test)]
mod tests {
    use super::summarize;

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
}
