// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The `HELLO` handshake.
//!
//! ```text
//! client -> control :  HELLO <session-name>\n
//! server ->         :  PORT <n>\n      or      ERR <reason>\n
//! ```
//!
//! A first line that is not a `HELLO` is not an error: the connection is
//! treated as a raw anonymous stream, so `nc host 7878 < capture.txt` works
//! with no ceremony.

/// Maximum session-name length, in bytes.
pub const NAME_MAX: usize = 64;

/// Why a line was not a usable `HELLO`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelloError {
    /// Not a handshake at all — treat the connection as a raw stream.
    NotHello,
    /// A handshake with an unusable name.
    BadName,
}

impl HelloError {
    /// The line to send back, without its newline.
    #[must_use]
    pub fn wire(&self) -> &'static str {
        match self {
            Self::NotHello => "ERR not a handshake",
            Self::BadName => "ERR bad name",
        }
    }
}

/// Parses one handshake line, returning the session name.
///
/// # Errors
/// [`HelloError::NotHello`] when the line has no `HELLO ` prefix;
/// [`HelloError::BadName`] when the name is empty, longer than [`NAME_MAX`],
/// or contains anything but printable non-space ASCII.
pub fn parse_hello(line: &str) -> Result<String, HelloError> {
    let line = line.trim_end_matches(['\r', '\n']);
    let name = line.strip_prefix("HELLO ").ok_or(HelloError::NotHello)?;
    if name.is_empty() || name.len() > NAME_MAX {
        return Err(HelloError::BadName);
    }
    if !name.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err(HelloError::BadName);
    }
    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_well_formed_hello() {
        assert_eq!(parse_hello("HELLO build-agent").unwrap(), "build-agent");
    }

    #[test]
    fn trailing_cr_is_tolerated() {
        assert_eq!(parse_hello("HELLO x\r").unwrap(), "x");
    }

    #[test]
    fn a_non_hello_line_is_not_an_error_but_a_raw_stream() {
        assert!(matches!(
            parse_hello("hello there"),
            Err(HelloError::NotHello)
        ));
        assert!(matches!(
            parse_hello("{\"tok\":1}"),
            Err(HelloError::NotHello)
        ));
    }

    #[test]
    fn empty_oversized_and_whitespace_names_are_rejected() {
        assert!(matches!(parse_hello("HELLO "), Err(HelloError::BadName)));
        assert!(matches!(parse_hello("HELLO a b"), Err(HelloError::BadName)));
        let long = "x".repeat(65);
        assert!(matches!(
            parse_hello(&format!("HELLO {long}")),
            Err(HelloError::BadName)
        ));
        assert!(parse_hello(&format!("HELLO {}", "x".repeat(64))).is_ok());
    }

    #[test]
    fn non_printable_names_are_rejected() {
        assert!(matches!(
            parse_hello("HELLO na\u{7}me"),
            Err(HelloError::BadName)
        ));
        assert!(matches!(
            parse_hello("HELLO café"),
            Err(HelloError::BadName)
        ));
    }
}
