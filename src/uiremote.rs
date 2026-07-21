//! `--ui-remote`: remote control of the TUI for testing.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::tools::mcp::{Json, json_escape, json_parse, json_write};

/// A command received from a remote-control client.
#[derive(Debug)]
pub enum RemoteCmd {
    /// Inject these key events, in order.
    Keypress(Vec<KeyEvent>),
    /// Return the rendered screen as ANSI text.
    Snapshot,
    /// Return the frame's structure as JSON.
    Uitree,
}

/// Parses one key description into a key event.
///
/// Accepts a single literal character, a named key (`enter`, `esc`, `tab`,
/// `backspace`, `delete`, `space`, `up`, `down`, `left`, `right`, `home`,
/// `end`), or a modifier form (`ctrl+`, `alt+`, `shift+`) wrapping either.
///
/// # Errors
/// Returns a message when the name is empty or unrecognised. Unknown names are
/// rejected rather than silently ignored, so a typo in a test fails loudly.
pub fn parse_key(s: &str) -> Result<KeyEvent, String> {
    let mut modifiers = KeyModifiers::NONE;
    let mut rest = s;
    loop {
        let lower = rest.to_ascii_lowercase();
        if let Some(r) = lower.strip_prefix("ctrl+") {
            modifiers |= KeyModifiers::CONTROL;
            rest = &rest[rest.len() - r.len()..];
        } else if let Some(r) = lower.strip_prefix("alt+") {
            modifiers |= KeyModifiers::ALT;
            rest = &rest[rest.len() - r.len()..];
        } else if let Some(r) = lower.strip_prefix("shift+") {
            modifiers |= KeyModifiers::SHIFT;
            rest = &rest[rest.len() - r.len()..];
        } else {
            break;
        }
    }
    let code = match rest.to_ascii_lowercase().as_str() {
        "enter" => KeyCode::Enter,
        "esc" | "escape" => KeyCode::Esc,
        "tab" => KeyCode::Tab,
        "backspace" => KeyCode::Backspace,
        "delete" => KeyCode::Delete,
        "space" => KeyCode::Char(' '),
        "up" => KeyCode::Up,
        "down" => KeyCode::Down,
        "left" => KeyCode::Left,
        "right" => KeyCode::Right,
        "home" => KeyCode::Home,
        "end" => KeyCode::End,
        _ => {
            let mut chars = rest.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => KeyCode::Char(c),
                _ => return Err(format!("unknown key: {rest:?}")),
            }
        }
    };
    Ok(KeyEvent::new(code, modifiers))
}

/// Parses one request line.
///
/// # Errors
/// Returns a message when the line is not a JSON object, carries no known
/// `cmd`, or a `keypress` has a missing or unparseable `keys` array.
pub fn parse_command(line: &str) -> Result<RemoteCmd, String> {
    let root = json_parse(line).ok_or_else(|| "malformed JSON".to_string())?;
    let cmd = root.str_or("cmd", "");
    match cmd {
        "snapshot" => Ok(RemoteCmd::Snapshot),
        "uitree" => Ok(RemoteCmd::Uitree),
        "keypress" => {
            let Some(Json::Arr(list)) = root.get("keys") else {
                return Err("keypress needs a keys array".to_string());
            };
            let mut keys = Vec::with_capacity(list.len());
            for k in list {
                let Json::Str(s) = k else {
                    return Err("keys must be strings".to_string());
                };
                keys.push(parse_key(s)?);
            }
            Ok(RemoteCmd::Keypress(keys))
        }
        "" => Err("missing cmd".to_string()),
        other => Err(format!("unknown cmd: {other}")),
    }
}

/// Renders a failure reply as one JSON line.
#[must_use]
pub fn error_reply(msg: &str) -> String {
    let mut out = String::from(r#"{"ok":false,"error":"#);
    json_escape(&mut out, msg);
    out.push('}');
    out
}

/// Renders a success reply carrying `fields`, as one JSON line.
#[must_use]
pub fn ok_reply(fields: &[(&str, Json)]) -> String {
    let mut out = String::from(r#"{"ok":true"#);
    for (k, v) in fields {
        out.push(',');
        json_escape(&mut out, k);
        out.push(':');
        json_write(&mut out, v);
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};

    #[test]
    fn parses_literal_character_keys() {
        let k = parse_key("@").expect("literal");
        assert_eq!(k.code, KeyCode::Char('@'));
        assert_eq!(k.modifiers, KeyModifiers::NONE);
    }

    #[test]
    fn parses_named_keys() {
        assert_eq!(parse_key("enter").unwrap().code, KeyCode::Enter);
        assert_eq!(parse_key("esc").unwrap().code, KeyCode::Esc);
        assert_eq!(parse_key("tab").unwrap().code, KeyCode::Tab);
        assert_eq!(parse_key("up").unwrap().code, KeyCode::Up);
        assert_eq!(parse_key("down").unwrap().code, KeyCode::Down);
        assert_eq!(parse_key("space").unwrap().code, KeyCode::Char(' '));
    }

    #[test]
    fn parses_modifier_forms() {
        let k = parse_key("ctrl+u").expect("ctrl");
        assert_eq!(k.code, KeyCode::Char('u'));
        assert!(k.modifiers.contains(KeyModifiers::CONTROL));

        let k = parse_key("shift+enter").expect("shift+named");
        assert_eq!(k.code, KeyCode::Enter);
        assert!(k.modifiers.contains(KeyModifiers::SHIFT));
    }

    #[test]
    fn rejects_unknown_key_names() {
        assert!(parse_key("wibble").is_err());
        assert!(parse_key("ctrl+wibble").is_err());
        assert!(parse_key("").is_err());
    }

    #[test]
    fn parses_the_three_commands() {
        assert!(matches!(
            parse_command(r#"{"cmd":"snapshot"}"#),
            Ok(RemoteCmd::Snapshot)
        ));
        assert!(matches!(
            parse_command(r#"{"cmd":"uitree"}"#),
            Ok(RemoteCmd::Uitree)
        ));
        let Ok(RemoteCmd::Keypress(keys)) =
            parse_command(r#"{"cmd":"keypress","keys":["@","down","enter"]}"#)
        else {
            panic!("expected keypress");
        };
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0].code, KeyCode::Char('@'));
        assert_eq!(keys[1].code, KeyCode::Down);
        assert_eq!(keys[2].code, KeyCode::Enter);
    }

    #[test]
    fn rejects_malformed_commands() {
        assert!(parse_command("not json").is_err());
        assert!(parse_command("{}").is_err());
        assert!(parse_command(r#"{"cmd":"nope"}"#).is_err());
        assert!(parse_command(r#"{"cmd":"keypress"}"#).is_err());
        assert!(parse_command(r#"{"cmd":"keypress","keys":["wibble"]}"#).is_err());
    }

    #[test]
    fn replies_are_single_line_json() {
        let e = error_reply("busy");
        assert_eq!(e, r#"{"ok":false,"error":"busy"}"#);
        assert!(!e.contains('\n'));

        let o = ok_reply(&[("cols", Json::Num(100.0))]);
        assert!(o.starts_with(r#"{"ok":true,"#), "{o}");
        assert!(o.contains(r#""cols":100"#), "{o}");
        assert!(!o.contains('\n'));
    }

    #[test]
    fn error_replies_escape_their_message() {
        assert!(error_reply(r#"bad "quote""#).contains(r#"\"quote\""#));
    }
}
