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

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use ratatui::layout::Rect;

/// One instrumented draw region.
#[derive(Debug, Clone)]
pub struct Region {
    /// Stable identifier for the region, e.g. `"popup"`.
    pub name: String,
    /// Where it was drawn this frame.
    pub rect: Rect,
    /// Extra state the region chose to publish.
    pub state: Vec<(String, Json)>,
}

/// Set while `--ui-remote` is active. Keeps [`region`] free otherwise.
static RECORDING: AtomicBool = AtomicBool::new(false);

/// Serialises every test that touches the process-global `RECORDING` flag.
///
/// Lives at module level (not inside `mod tests`) so tests in other modules —
/// `crate::tui`'s draw-site instrumentation tests — take the *same* guard;
/// two separate mutexes would not serialise anything.
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

thread_local! {
    /// Regions recorded during the current frame. Drawing only happens on the
    /// UI thread, so a thread-local avoids threading a recorder through every
    /// draw signature. This is deliberate hidden state; see the design doc.
    static REGIONS: RefCell<Vec<Region>> = const { RefCell::new(Vec::new()) };
}

/// True while draw-time region recording is on.
#[must_use]
pub fn recording_enabled() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

/// Turns draw-time region recording on or off.
pub fn set_recording(on: bool) {
    RECORDING.store(on, Ordering::Relaxed);
}

/// Clears the previous frame's regions. Call once per draw pass.
///
/// This always clears, even while recording is off: recording can be toggled
/// on between draw passes, and if the clear were gated on the flag, the first
/// frame after re-enabling it would silently inherit whatever regions were
/// left over from the last time it was on. A `Vec::clear` is negligible next
/// to the many `region()` calls a real frame makes, so there is no need to
/// skip it — only `region()` itself needs to be free when recording is off.
pub fn begin_frame() {
    REGIONS.with(|r| r.borrow_mut().clear());
}

/// Records one drawn region. A no-op unless recording is enabled.
pub fn region(name: &str, rect: Rect, state: &[(&str, Json)]) {
    if !recording_enabled() {
        return;
    }
    let entry = Region {
        name: name.to_string(),
        rect,
        state: state
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    };
    REGIONS.with(|r| r.borrow_mut().push(entry));
}

/// True when `outer` fully contains `inner`.
///
/// Equal-sized (and equally-positioned) rects deliberately do NOT count as
/// containment: a full-screen root and a full-screen overlay have identical
/// rects, and treating one as "inside" the other would arbitrarily bury
/// whichever was drawn second inside the first with no way to tell them
/// apart as siblings. Both are instead left as top-level regions; see
/// `frame_tree`'s handling of multiple top-level regions.
fn contains(outer: Rect, inner: Rect) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && inner.right() <= outer.right()
        && inner.bottom() <= outer.bottom()
        && (outer.width, outer.height) != (inner.width, inner.height)
}

/// Renders one region and its children as a JSON object.
fn node_json(regions: &[Region], idx: usize, out: &mut String) {
    let r = &regions[idx];
    out.push('{');
    out.push_str(r#""name":"#);
    json_escape(out, &r.name);
    let _ = write!(
        out,
        r#","x":{},"y":{},"width":{},"height":{}"#,
        r.rect.x, r.rect.y, r.rect.width, r.rect.height
    );
    for (k, v) in &r.state {
        out.push(',');
        json_escape(out, k);
        out.push(':');
        json_write(out, v);
    }
    // Children are later regions contained by this one and not by any
    // intervening region — the nearest enclosing ancestor wins. This is
    // O(n^2) per node (O(n^3) overall) but frames hold a handful of regions,
    // so clarity wins over speed here.
    let kids: Vec<usize> = (idx + 1..regions.len())
        .filter(|&j| {
            contains(r.rect, regions[j].rect)
                && (idx + 1..j).all(|m| {
                    !(contains(r.rect, regions[m].rect)
                        && contains(regions[m].rect, regions[j].rect))
                })
        })
        .collect();
    if !kids.is_empty() {
        out.push_str(r#","children":["#);
        for (n, &j) in kids.iter().enumerate() {
            if n > 0 {
                out.push(',');
            }
            node_json(regions, j, out);
        }
        out.push(']');
    }
    out.push('}');
}

/// The current frame's regions as a JSON tree, nested by containment.
///
/// Returns `{}` when nothing was recorded. When every region nests under a
/// single top-level region (the common case: one full-screen root), that
/// region is rendered directly. When more than one region is NOT contained
/// by any other (e.g. two full-screen rects, or a popup drawn outside the
/// root's bounds), all of them are wrapped in a synthetic
/// `{"children":[...]}` object rather than silently dropping every top-level
/// region but the first.
#[must_use]
pub fn frame_tree() -> String {
    REGIONS.with(|r| {
        let regions = r.borrow();
        if regions.is_empty() {
            return "{}".to_string();
        }
        let tops: Vec<usize> = (0..regions.len())
            .filter(|&i| {
                !(0..regions.len()).any(|j| j != i && contains(regions[j].rect, regions[i].rect))
            })
            .collect();
        let mut out = String::new();
        if let [only] = tops[..] {
            node_json(&regions, only, &mut out);
        } else {
            out.push_str(r#"{"children":["#);
            for (n, &i) in tops.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                node_json(&regions, i, &mut out);
            }
            out.push_str("]}");
        }
        out
    })
}

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

/// How long `serve_conn` waits for the UI thread to answer a pending command
/// before giving up and replying with an error itself.
///
/// The UI thread is expected to drain [`RemoteHandle`] every tick (Task 5),
/// so a healthy round trip is well under a second. If the UI thread is wedged
/// (or the whole handle was dropped without the receiver's `Sender` half
/// being dropped first — see [`RemoteHandle`]'s `Drop` behaviour below) a
/// client blocked on `rrx.recv()` with no timeout would hang forever with no
/// way to notice, let alone recover. A generous bound turns that into a
/// bounded stall and a diagnosable error reply instead.
const REPLY_TIMEOUT: Duration = Duration::from_secs(10);

/// A command waiting for the UI thread, with the channel to answer on.
#[derive(Debug)]
pub struct Pending {
    /// What the client asked for.
    pub cmd: RemoteCmd,
    /// Send exactly one reply line here.
    pub reply: Sender<String>,
}

/// Handle held by the UI thread while remote control is active.
#[derive(Debug)]
pub struct RemoteHandle {
    /// The bound port; resolved even when 0 was requested.
    pub port: u16,
    rx: Receiver<Pending>,
}

impl RemoteHandle {
    /// Takes one pending command, if any.
    #[must_use]
    pub fn try_recv(&self) -> Option<Pending> {
        self.rx.try_recv().ok()
    }
}

/// Starts the remote-control listener on `127.0.0.1:port`.
///
/// Passing 0 binds an ephemeral port, reported in [`RemoteHandle::port`].
/// Binding is loopback-only by construction: the address is not
/// configurable, and only the port is caller-supplied. This is a
/// keystroke-injection port into an agent that can run shell commands, so it
/// must never be reachable from anything but the local machine.
///
/// Connections are served one at a time, sequentially, on the listener's own
/// thread: a second client that connects while the first is still being
/// served does not get an immediate error — it simply waits in the kernel's
/// accept backlog until the first connection closes, then is served in turn.
/// (An earlier sketch of this function tracked a `busy` flag to reject a
/// second concurrent client outright; because accept and serve both happen
/// on the same single thread, that flag could never be observed as `true` by
/// anyone — there is no second thread to race it — so it was dead code and
/// has been removed rather than kept as a misleading no-op.)
///
/// # Errors
/// Returns a message when the port cannot be bound.
pub fn start(port: u16) -> Result<RemoteHandle, String> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("ui-remote: bind 127.0.0.1:{port}: {e}"))?;
    let bound = listener
        .local_addr()
        .map_err(|e| format!("ui-remote: local_addr: {e}"))?
        .port();
    let (tx, rx) = channel::<Pending>();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            serve_conn(&mut stream, &tx);
        }
    });
    Ok(RemoteHandle { port: bound, rx })
}

/// Serves one client until it disconnects.
fn serve_conn(stream: &mut std::net::TcpStream, tx: &Sender<Pending>) {
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let reply = match parse_command(trimmed) {
            Err(e) => error_reply(&e),
            Ok(cmd) => {
                let (rtx, rrx) = channel::<String>();
                // If `tx.send` fails, the UI thread's `RemoteHandle` (and its
                // `Receiver`) has been dropped — nothing will ever answer, so
                // give up on this connection immediately rather than waiting
                // out the timeout for nothing.
                if tx.send(Pending { cmd, reply: rtx }).is_err() {
                    return;
                }
                // The UI thread answers on its next tick; block until it
                // does, but only up to REPLY_TIMEOUT so a wedged UI thread
                // stalls the client instead of hanging it forever.
                rrx.recv_timeout(REPLY_TIMEOUT)
                    .unwrap_or_else(|_| error_reply("ui thread timed out"))
            }
        };
        if writeln!(stream, "{reply}").is_err() {
            return;
        }
    }
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

    // `RECORDING` and `REGIONS` are process/thread-local globals shared by
    // every test in this binary, and cargo test runs tests in parallel
    // threads by default. Without serializing, one test's `set_recording`
    // races another's. `REGIONS` is itself a `thread_local!`, so it is safe
    // per-thread, but `RECORDING` is a plain `static` and must be guarded.
    use super::TEST_LOCK as UIREMOTE_TEST_LOCK;

    #[test]
    fn region_is_a_no_op_when_recording_is_off() {
        let _guard = UIREMOTE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set_recording(false);
        begin_frame();
        region("popup", Rect::new(0, 0, 10, 5), &[]);
        assert_eq!(frame_tree(), "{}");
    }

    #[test]
    fn records_a_flat_region_with_its_rect_and_state() {
        let _guard = UIREMOTE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set_recording(true);
        begin_frame();
        region(
            "popup",
            Rect::new(1, 2, 30, 15),
            &[("rows", Json::Num(15.0)), ("selected", Json::Num(3.0))],
        );
        let tree = frame_tree();
        assert!(tree.contains(r#""name":"popup""#), "{tree}");
        assert!(tree.contains(r#""x":1"#), "{tree}");
        assert!(tree.contains(r#""y":2"#), "{tree}");
        assert!(tree.contains(r#""width":30"#), "{tree}");
        assert!(tree.contains(r#""height":15"#), "{tree}");
        assert!(tree.contains(r#""selected":3"#), "{tree}");
        set_recording(false);
    }

    #[test]
    fn nests_a_contained_region_inside_its_container() {
        let _guard = UIREMOTE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set_recording(true);
        begin_frame();
        region("root", Rect::new(0, 0, 100, 30), &[]);
        region("input", Rect::new(0, 28, 100, 1), &[]);
        let tree = frame_tree();
        // `input` sits inside `root`, so it must appear in root's children,
        // not as a second top-level node.
        let root_at = tree.find("\"name\":\"root\"").expect("root present");
        let input_at = tree.find("\"name\":\"input\"").expect("input present");
        assert!(root_at < input_at, "child must follow its parent: {tree}");
        assert!(tree.contains(r#""children""#), "{tree}");
        set_recording(false);
    }

    #[test]
    fn begin_frame_discards_the_previous_frames_regions() {
        let _guard = UIREMOTE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set_recording(true);
        begin_frame();
        region("stale", Rect::new(0, 0, 5, 5), &[]);
        begin_frame();
        region("fresh", Rect::new(0, 0, 5, 5), &[]);
        let tree = frame_tree();
        assert!(!tree.contains("stale"), "{tree}");
        assert!(tree.contains("fresh"), "{tree}");
        set_recording(false);
    }

    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;

    /// Sends one line and returns the one-line reply.
    fn roundtrip(port: u16, line: &str) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        s.write_all(line.as_bytes()).unwrap();
        s.write_all(b"\n").unwrap();
        s.flush().unwrap();
        let mut reader = BufReader::new(s);
        let mut out = String::new();
        reader.read_line(&mut out).expect("reply");
        out.trim_end().to_string()
    }

    #[test]
    fn binds_an_ephemeral_port_and_serves_a_command() {
        let h = start(0).expect("start");
        assert_ne!(h.port, 0, "ephemeral port must be reported");
        let port = h.port;

        // The UI thread answers; emulate it on this thread. It returns as
        // soon as it has served the one command the test sends, so it never
        // outlives the test.
        let handle = std::thread::spawn(move || {
            loop {
                if let Some(p) = h.try_recv() {
                    assert!(matches!(p.cmd, RemoteCmd::Snapshot));
                    let _ = p.reply.send(ok_reply(&[("cols", Json::Num(80.0))]));
                    return;
                }
                std::thread::yield_now();
            }
        });

        let reply = roundtrip(port, r#"{"cmd":"snapshot"}"#);
        assert!(reply.contains(r#""cols":80"#), "{reply}");
        handle.join().expect("ui thread");
    }

    #[test]
    fn rejects_a_malformed_line_without_closing_the_connection() {
        let h = start(0).expect("start");
        let port = h.port;

        // The fake UI thread must not spin forever after the test is done:
        // it exits once `stop` is set below.
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stop_thread = std::sync::Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                if let Some(p) = h.try_recv() {
                    let _ = p.reply.send(ok_reply(&[]));
                }
                std::thread::yield_now();
            }
        });

        let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        s.write_all(b"not json\n").unwrap();
        s.flush().unwrap();
        let mut reader = BufReader::new(s.try_clone().unwrap());
        let mut first = String::new();
        reader.read_line(&mut first).unwrap();
        assert!(first.contains(r#""ok":false"#), "{first}");
        // Connection stays usable.
        s.write_all(b"{\"cmd\":\"snapshot\"}\n").unwrap();
        s.flush().unwrap();
        let mut second = String::new();
        reader.read_line(&mut second).unwrap();
        assert!(second.contains(r#""ok":true"#), "{second}");

        stop.store(true, Ordering::Relaxed);
        handle.join().expect("ui thread");
    }

    #[test]
    fn binds_loopback_only() {
        let h = start(0).expect("start");
        // Connecting via loopback works; the bound address must be 127.0.0.1.
        assert!(TcpStream::connect(("127.0.0.1", h.port)).is_ok());
    }
}
