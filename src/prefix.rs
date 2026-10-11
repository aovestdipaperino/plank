//! Conversation prefix conditioning: scripted `USER:` / `ASSISTANT:` turns
//! placed between the system prompt and the live conversation (upstream ds4
//! `b0a147a`, `ds4_prompt_prefix.c`).
//!
//! The turns read to the model as an exchange that already happened, so it
//! continues in the manner they demonstrate: few-shot conditioning, which a
//! local model follows more reliably than a rule stated in the system prompt.
//! They come from `--prefix-file` or a profile's `prefix`, are read once at
//! startup, and ride on the main session ([`crate::session::Session::prefix`])
//! without ever being saved or compacted.

use std::path::Path;
use std::sync::OnceLock;

use crate::session::Message;

/// The prefix the run conditions on, set once at startup.
static ACTIVE: OnceLock<Vec<Message>> = OnceLock::new();

/// Installs the run's prefix. The first call wins.
pub fn install(turns: Vec<Message>) {
    let _ = ACTIVE.set(turns);
}

/// The run's prefix; empty when none was given.
#[must_use]
pub fn active() -> &'static [Message] {
    ACTIVE.get().map_or(&[], Vec::as_slice)
}

/// The role a marker at `at` opens, and the marker's length.
fn marker_at(data: &str, at: usize) -> Option<(bool, usize)> {
    let rest = &data[at..];
    if rest.starts_with("USER:") {
        Some((false, "USER:".len()))
    } else if rest.starts_with("ASSISTANT:") {
        Some((true, "ASSISTANT:".len()))
    } else {
        None
    }
}

/// Parses a prefix file: complete `USER:` / `ASSISTANT:` pairs, a marker
/// starting a turn only at the beginning of a line, continuation lines
/// belonging to the turn above. A port of `ds4_prompt_prefix_parse`, error
/// messages included.
///
/// # Errors
///
/// The C's message for an empty file, a NUL byte, a line that opens no turn,
/// roles out of order, an empty turn, or a file that does not end with an
/// `ASSISTANT:` turn.
pub fn parse(data: &str) -> Result<Vec<Message>, String> {
    if data.is_empty() {
        return Err("prefix file is empty".to_owned());
    }
    if data.contains('\0') {
        return Err("prefix file contains a NUL byte".to_owned());
    }
    let mut turns: Vec<Message> = Vec::new();
    let mut pos = usize::from(data.starts_with('\u{feff}')) * '\u{feff}'.len_utf8();
    let mut line = 1usize;
    while pos < data.len() {
        let Some((assistant, marker_len)) = marker_at(data, pos) else {
            return Err(format!("line {line} must start with USER: or ASSISTANT:"));
        };
        let expected_assistant = turns.len() % 2 == 1;
        if assistant != expected_assistant {
            let want = if expected_assistant {
                "ASSISTANT"
            } else {
                "USER"
            };
            return Err(format!("line {line}: expected {want}:"));
        }
        let mut start = pos + marker_len;
        if data[start..].starts_with([' ', '\t']) {
            start += 1;
        }
        // The next turn starts at the first line that opens with a marker.
        let (mut next, mut next_line, mut scan_line) = (data.len(), line, line);
        for (i, _) in data[start..].match_indices('\n') {
            let i = start + i;
            scan_line += 1;
            if i + 1 < data.len() && marker_at(data, i + 1).is_some() {
                next = i + 1;
                next_line = scan_line;
                break;
            }
        }
        let mut end = next;
        if end > start && data.as_bytes()[end - 1] == b'\n' {
            end -= 1;
            if end > start && data.as_bytes()[end - 1] == b'\r' {
                end -= 1;
            }
        }
        if end == start {
            let role = if assistant { "ASSISTANT" } else { "USER" };
            return Err(format!("line {line} has an empty {role} turn"));
        }
        let text = data[start..end].to_owned();
        turns.push(if assistant {
            Message::assistant(text)
        } else {
            Message::user(text)
        });
        pos = next;
        line = next_line;
    }
    if turns.is_empty() {
        return Err("prefix file contains no turns".to_owned());
    }
    if turns.len() % 2 == 1 {
        return Err("prefix file must end with an ASSISTANT turn".to_owned());
    }
    Ok(turns)
}

/// Reads and parses a prefix file.
///
/// # Errors
///
/// The C's "cannot open prefix file" when it cannot be read, otherwise
/// whatever [`parse`] reports.
pub fn load(path: &Path) -> Result<Vec<Message>, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot open prefix file {}: {e}", path.display()))?;
    let text = String::from_utf8(bytes)
        .map_err(|_| format!("prefix file {} is not UTF-8", path.display()))?;
    parse(&text)
}

#[cfg(test)]
mod tests {
    use super::parse;
    use crate::session::Role;

    fn roles_and_texts(src: &str) -> Vec<(Role, String)> {
        parse(src)
            .unwrap()
            .into_iter()
            .map(|m| (m.role, m.text))
            .collect()
    }

    #[test]
    fn turns_alternate_and_continuation_lines_stay_in_their_turn() {
        let src = "USER: Write terse answers.\n\
                   This line is still part of the user turn.\n\
                   ASSISTANT: Understood.\n\
                   USER: Return code without an introduction.\n\
                   ASSISTANT: I will return only the code.\n";
        assert_eq!(
            roles_and_texts(src),
            vec![
                (
                    Role::User,
                    "Write terse answers.\nThis line is still part of the user turn.".to_owned()
                ),
                (Role::Assistant, "Understood.".to_owned()),
                (
                    Role::User,
                    "Return code without an introduction.".to_owned()
                ),
                (Role::Assistant, "I will return only the code.".to_owned()),
            ]
        );
    }

    #[test]
    fn a_marker_counts_only_at_the_start_of_a_line() {
        let src = "USER: say USER: twice\r\nASSISTANT:\tok\n";
        assert_eq!(
            roles_and_texts(src),
            vec![
                (Role::User, "say USER: twice".to_owned()),
                (Role::Assistant, "ok".to_owned()),
            ]
        );
        // A byte-order mark is skipped, and the last turn may lack a newline.
        assert_eq!(parse("\u{feff}USER: a\nASSISTANT: b").unwrap().len(), 2);
    }

    #[test]
    fn malformed_files_get_the_c_messages() {
        let cases = [
            ("", "prefix file is empty"),
            (
                "USER: a\0\nASSISTANT: b\n",
                "prefix file contains a NUL byte",
            ),
            ("hello\n", "line 1 must start with USER: or ASSISTANT:"),
            ("ASSISTANT: a\n", "line 1: expected USER:"),
            ("USER: a\nUSER: b\n", "line 2: expected ASSISTANT:"),
            (
                "USER: a\nASSISTANT:\nUSER: b\n",
                "line 2 has an empty ASSISTANT turn",
            ),
            ("USER: a\n", "prefix file must end with an ASSISTANT turn"),
        ];
        for (src, want) in cases {
            assert_eq!(parse(src).unwrap_err(), want, "{src:?}");
        }
    }
}
