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

#[cfg(test)]
mod tests {
    use super::*;

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
