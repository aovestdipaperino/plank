//! plank's key payload, `{"code": "ctrl-s", "text": "S"}`, as a Turbo Vision
//! key event.

use turbo_vision::core::event::Event;
use turbo_vision::core::keys::{KeyCode, KeyEvent, KeyModifiers};

/// The key event for a payload, or `None` for a key Turbo Vision has no name
/// for. `text`, when present, is the typed character with its case; `code`
/// is lowercase by design and is used for everything else.
#[must_use]
pub fn translate(code: &str, text: Option<char>) -> Option<Event> {
    let mut mods = KeyModifiers::empty();
    let mut base = code;
    loop {
        if let Some(rest) = base.strip_prefix("ctrl-") {
            mods |= KeyModifiers::CONTROL;
            base = rest;
        } else if let Some(rest) = base.strip_prefix("alt-") {
            mods |= KeyModifiers::ALT;
            base = rest;
        } else if let Some(rest) = base.strip_prefix("shift-") {
            mods |= KeyModifiers::SHIFT;
            base = rest;
        } else {
            break;
        }
    }
    let key = match base {
        "enter" => KeyCode::Enter,
        "escape" => KeyCode::Esc,
        "backspace" => KeyCode::Backspace,
        "tab" => KeyCode::Tab,
        "backtab" => KeyCode::BackTab,
        "delete" => KeyCode::Delete,
        "insert" => KeyCode::Insert,
        "home" => KeyCode::Home,
        "end" => KeyCode::End,
        "pageup" => KeyCode::PageUp,
        "pagedown" => KeyCode::PageDown,
        "up" => KeyCode::Up,
        "down" => KeyCode::Down,
        "left" => KeyCode::Left,
        "right" => KeyCode::Right,
        "space" => KeyCode::Char(' '),
        f if f.len() > 1 && f.starts_with('f') && f[1..].parse::<u8>().is_ok() => {
            KeyCode::F(f[1..].parse().ok()?)
        }
        s => {
            let mut chars = s.chars();
            let c = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            // Only a plain key types; a chord keeps its lowercase code.
            let typed = if mods.contains(KeyModifiers::CONTROL) || mods.contains(KeyModifiers::ALT)
            {
                c
            } else {
                text.unwrap_or(c)
            };
            KeyCode::Char(typed)
        }
    };
    Some(Event::from_crossterm_key(KeyEvent::new(key, mods)))
}

/// Whether a translated event carries the Control modifier.
///
/// Ctrl+Ins and Ctrl+Del share key codes with plain Ins/Del in Turbo Vision
/// (`crossterm_to_keycode` maps both to `KB_INS`/`KB_DEL` regardless of
/// modifiers, and there is no `KB_CTRL_INS`/`KB_CTRL_DEL`), so callers that
/// need to tell them apart must check the event's modifiers, not its code.
#[must_use]
pub fn is_ctrl(ev: &Event) -> bool {
    ev.key_modifiers.contains(KeyModifiers::CONTROL)
}

/// The `text` field of a flat key payload, unescaped, when it is one char.
#[must_use]
pub fn payload_text(payload: &str) -> Option<char> {
    let start = payload.find("\"text\":")? + "\"text\":".len();
    let rest = payload[start..].trim_start().strip_prefix('"')?;
    let mut chars = rest.chars();
    let c = match chars.next()? {
        '\\' => match chars.next()? {
            'n' => '\n',
            't' => '\t',
            'u' => {
                let hex: String = chars.by_ref().take(4).collect();
                char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?
            }
            other => other,
        },
        c => c,
    };
    (chars.next() == Some('"')).then_some(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_vision::core::event::{
        KB_DEL, KB_ENTER, KB_ESC, KB_F10, KB_INS, KB_LEFT, KB_SHIFT_TAB,
    };

    fn code_of(code: &str, text: Option<char>) -> u16 {
        translate(code, text).expect("translates").key_code
    }

    #[test]
    fn named_keys() {
        assert_eq!(code_of("enter", None), KB_ENTER);
        assert_eq!(code_of("escape", None), KB_ESC);
        assert_eq!(code_of("left", None), KB_LEFT);
        assert_eq!(code_of("insert", None), KB_INS);
        assert_eq!(code_of("delete", None), KB_DEL);
        assert_eq!(code_of("f10", None), KB_F10);
        assert_eq!(code_of("backtab", None), KB_SHIFT_TAB);
    }

    #[test]
    fn text_wins_for_printable_keys_so_case_survives() {
        let e = translate("a", Some('A')).unwrap();
        assert_eq!(e.key_code, u16::from(b'A'));
        assert_eq!(translate(" ", Some(' ')).unwrap().key_code, u16::from(b' '));
    }

    #[test]
    fn chords_use_the_code() {
        let ctrl_s = translate("ctrl-s", None).unwrap();
        let expected =
            Event::from_crossterm_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL));
        assert_eq!(ctrl_s.key_code, expected.key_code);
        let alt_x = translate("alt-x", None).unwrap();
        let expected =
            Event::from_crossterm_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::ALT));
        assert_eq!(alt_x.key_code, expected.key_code);
    }

    #[test]
    fn ctrl_insert_and_delete_keep_their_modifier() {
        let ctrl_ins = translate("ctrl-insert", None).unwrap();
        assert_eq!(ctrl_ins.key_code, KB_INS);
        assert!(ctrl_ins.key_modifiers.contains(KeyModifiers::CONTROL));
        assert!(is_ctrl(&ctrl_ins));

        let ins = translate("insert", None).unwrap();
        assert_eq!(ins.key_code, KB_INS);
        assert!(!ins.key_modifiers.contains(KeyModifiers::CONTROL));
        assert!(!is_ctrl(&ins));

        let ctrl_del = translate("ctrl-delete", None).unwrap();
        assert_eq!(ctrl_del.key_code, KB_DEL);
        assert!(ctrl_del.key_modifiers.contains(KeyModifiers::CONTROL));
        assert!(is_ctrl(&ctrl_del));

        let del = translate("delete", None).unwrap();
        assert_eq!(del.key_code, KB_DEL);
        assert!(!del.key_modifiers.contains(KeyModifiers::CONTROL));
        assert!(!is_ctrl(&del));
    }

    #[test]
    fn unknown_keys_translate_to_nothing() {
        assert!(translate("capslock", None).is_none());
        assert!(translate("", None).is_none());
    }

    #[test]
    fn payload_text_reads_one_char_and_unescapes() {
        assert_eq!(payload_text(r#"{"code": "a", "text": "A"}"#), Some('A'));
        assert_eq!(payload_text(r#"{"code": "\"", "text": "\""}"#), Some('"'));
        assert_eq!(payload_text(r#"{"code": "enter"}"#), None);
    }

    #[test]
    fn the_mac_chords_are_distinct_control_codes() {
        use turbo_vision::core::event::{
            KB_CTRL_K, KB_CTRL_L, KB_CTRL_O, KB_CTRL_Q, KB_CTRL_R, KB_CTRL_Y, KB_TAB,
        };
        let chords = [
            ("ctrl-q", KB_CTRL_Q),
            ("ctrl-o", KB_CTRL_O),
            ("ctrl-r", KB_CTRL_R),
            ("ctrl-y", KB_CTRL_Y),
            ("ctrl-l", KB_CTRL_L),
            ("ctrl-k", KB_CTRL_K),
        ];
        for (code, expected) in chords {
            let got = code_of(code, None);
            assert_eq!(got, expected, "{code}");
            assert!(![KB_TAB, KB_ENTER, KB_ESC].contains(&got), "{code}");
        }
        let mut codes: Vec<u16> = chords.iter().map(|(c, _)| code_of(c, None)).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), chords.len());
    }
}
