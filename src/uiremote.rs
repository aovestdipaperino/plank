//! `--ui-remote`: remote control of the TUI for testing.

use std::fmt::Write as _;

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use unicode_width::UnicodeWidthStr;

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

/// Maps a colour to its SGR foreground parameter, or `None` when it is the
/// default (unstyled) colour.
fn sgr_fg(c: Color) -> Option<String> {
    Some(match c {
        Color::Reset => return None,
        Color::Black => "30".into(),
        Color::Red => "31".into(),
        Color::Green => "32".into(),
        Color::Yellow => "33".into(),
        Color::Blue => "34".into(),
        Color::Magenta => "35".into(),
        Color::Cyan => "36".into(),
        Color::Gray => "37".into(),
        Color::DarkGray => "90".into(),
        Color::LightRed => "91".into(),
        Color::LightGreen => "92".into(),
        Color::LightYellow => "93".into(),
        Color::LightBlue => "94".into(),
        Color::LightMagenta => "95".into(),
        Color::LightCyan => "96".into(),
        Color::White => "97".into(),
        Color::Indexed(i) => format!("38;5;{i}"),
        Color::Rgb(r, g, b) => format!("38;2;{r};{g};{b}"),
    })
}

/// Same as [`sgr_fg`] but for background colours.
fn sgr_bg(c: Color) -> Option<String> {
    let fg = sgr_fg(c)?;
    // Background codes are the foreground codes shifted by 10 (bright
    // 90-97 -> 100-107), and the extended forms swap their `38` introducer
    // for `48` rather than being numerically shifted.
    Some(if let Some(rest) = fg.strip_prefix("38;") {
        format!("48;{rest}")
    } else {
        let n: u32 = fg.parse().unwrap_or(30);
        (n + 10).to_string()
    })
}

/// Normalises a style so `Color::Reset` and `None` compare equal.
///
/// `Cell::style()` always reports `fg`/`bg` as `Some(Color::Reset)` rather
/// than `None`, so comparing the raw style against `Style::default()` (or
/// another cell's raw style) would treat every unstyled cell as styled, and
/// two unstyled cells as different. This collapses that distinction before
/// comparing.
fn normalize(mut style: Style) -> Style {
    if matches!(style.fg, Some(Color::Reset)) {
        style.fg = None;
    }
    if matches!(style.bg, Some(Color::Reset)) {
        style.bg = None;
    }
    if matches!(style.underline_color, Some(Color::Reset)) {
        style.underline_color = None;
    }
    style
}

/// Whether `style` renders identically to an unstyled cell.
fn is_plain(style: Style) -> bool {
    normalize(style) == Style::default()
}

/// Serialises `buf` as ANSI text: one line per row, with SGR codes emitted
/// only where the style changes from the previous cell (starting from
/// [`Style::default`] at the start of each row), and a `\x1b[0m` reset
/// closing any row that ends mid-style.
///
/// Trailing blanks are preserved so rows stay column-aligned and snapshots
/// diff cleanly.
#[must_use]
pub fn buffer_to_ansi(buf: &Buffer) -> String {
    let area = *buf.area();
    let mut out = String::new();
    for y in area.top()..area.bottom() {
        if y > area.top() {
            out.push('\n');
        }
        let mut active = Style::default();
        // Continuation cells after a double-width grapheme carry a literal
        // space (`Cell::reset()`), not the wide-char marker `cell.skip`
        // (an unrelated overlay/redraw hint) — so we step over them by
        // display width ourselves, mirroring `Buffer::diff`.
        let mut skip = 0usize;
        for x in area.left()..area.right() {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            let cell = &buf[(x, y)];
            let style = normalize(cell.style());
            if style != active {
                if !is_plain(active) {
                    // Clear whatever the previous style left active before
                    // applying the next one (or nothing, if plain).
                    out.push_str("\u{1b}[0m");
                }
                if !is_plain(style) {
                    let mut params: Vec<String> = Vec::new();
                    if let Some(p) = sgr_fg(style.fg.unwrap_or(Color::Reset)) {
                        params.push(p);
                    }
                    if let Some(p) = sgr_bg(style.bg.unwrap_or(Color::Reset)) {
                        params.push(p);
                    }
                    if style.add_modifier.contains(Modifier::BOLD) {
                        params.push("1".to_string());
                    }
                    if style.add_modifier.contains(Modifier::DIM) {
                        params.push("2".to_string());
                    }
                    if style.add_modifier.contains(Modifier::ITALIC) {
                        params.push("3".to_string());
                    }
                    if style.add_modifier.contains(Modifier::UNDERLINED) {
                        params.push("4".to_string());
                    }
                    if style.add_modifier.contains(Modifier::REVERSED) {
                        params.push("7".to_string());
                    }
                    let _ = write!(out, "\u{1b}[{}m", params.join(";"));
                }
                active = style;
            }
            out.push_str(cell.symbol());
            skip = cell.symbol().width().saturating_sub(1);
        }
        if !is_plain(active) {
            out.push_str("\u{1b}[0m");
        }
    }
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

    #[test]
    fn ctrl_modifier_composes_with_multi_byte_utf8_char() {
        let k = parse_key("ctrl+é").expect("ctrl+é");
        assert_eq!(k.code, KeyCode::Char('é'));
        assert!(k.modifiers.contains(KeyModifiers::CONTROL));
    }

    #[test]
    fn keypress_rejects_non_string_key_element() {
        assert!(parse_command(r#"{"cmd":"keypress","keys":[5]}"#).is_err());
    }

    #[test]
    fn stacked_modifiers_compose() {
        let k = parse_key("ctrl+shift+a").expect("ctrl+shift+a");
        assert_eq!(k.code, KeyCode::Char('a'));
        assert!(k.modifiers.contains(KeyModifiers::CONTROL));
        assert!(k.modifiers.contains(KeyModifiers::SHIFT));
    }

    #[test]
    fn dangling_modifier_prefix_is_an_error_but_bare_plus_is_a_char() {
        assert!(parse_key("ctrl+").is_err());
        assert_eq!(parse_key("+").unwrap().code, KeyCode::Char('+'));
    }

    use ratatui::layout::Rect;

    #[test]
    fn plain_buffer_serialises_to_plain_rows() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 3, 2));
        buf.set_string(0, 0, "abc", Style::default());
        buf.set_string(0, 1, "def", Style::default());
        assert_eq!(buffer_to_ansi(&buf), "abc\ndef");
    }

    #[test]
    fn styled_runs_emit_sgr_once_and_reset_at_end_of_row() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 1));
        buf.set_string(0, 0, "ab", Style::default().fg(Color::Green));
        buf.set_string(2, 0, "cd", Style::default());
        let out = buffer_to_ansi(&buf);
        // One green introducer, not one per cell.
        assert_eq!(out.matches("\u{1b}[32m").count(), 1, "{out:?}");
        assert!(out.contains("ab"), "{out:?}");
        assert!(out.ends_with("\u{1b}[0m") || out.ends_with("cd"), "{out:?}");
    }

    #[test]
    fn trailing_blanks_are_preserved_so_rows_align() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 5, 1));
        buf.set_string(0, 0, "hi", Style::default());
        let row = buffer_to_ansi(&buf);
        assert!(row.starts_with("hi"), "{row:?}");
        assert_eq!(row.chars().filter(|c| *c == ' ').count(), 3, "{row:?}");
    }

    #[test]
    fn output_is_deterministic_for_the_same_buffer() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 6, 2));
        buf.set_string(0, 0, "one", Style::default().fg(Color::Red));
        buf.set_string(0, 1, "two", Style::default());
        assert_eq!(buffer_to_ansi(&buf), buffer_to_ansi(&buf));
    }

    #[test]
    fn an_empty_buffer_is_an_empty_string() {
        let buf = Buffer::empty(Rect::new(0, 0, 0, 0));
        assert_eq!(buffer_to_ansi(&buf), "");
    }

    #[test]
    fn wide_glyphs_serialise_without_overshooting_declared_width() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 1));
        buf.set_string(0, 0, "世界", Style::default());
        // Two double-width glyphs should occupy exactly 4 terminal columns:
        // the glyph itself plus nothing at all for its continuation cell.
        assert_eq!(buffer_to_ansi(&buf), "世界");
    }

    #[test]
    fn ascii_and_wide_glyph_mix_stays_column_aligned() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 6, 1));
        buf.set_string(0, 0, "a", Style::default());
        buf.set_string(1, 0, "世", Style::default());
        buf.set_string(3, 0, "bc", Style::default());
        // a(1) + 世(2) + bc(2) = 5 declared columns, 1 trailing blank pad.
        assert_eq!(buffer_to_ansi(&buf), "a世bc ");
    }

    #[test]
    fn emoji_serialises_at_correct_display_width() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 1));
        buf.set_string(0, 0, "🎉", Style::default());
        assert_eq!(buffer_to_ansi(&buf), "🎉  ");
    }

    #[test]
    fn cell_with_skip_flag_and_real_content_is_not_dropped() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 3, 1));
        buf.set_string(0, 0, "abc", Style::default());
        buf[(1, 0)].set_skip(true);
        // `skip` is an overlay/redraw hint, not the wide-char continuation
        // marker; content marked skip=true must still be emitted.
        assert_eq!(buffer_to_ansi(&buf), "abc");
    }
}
