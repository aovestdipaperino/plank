//! Trace logging, ported from the "Trace Logging" section of `ds4_agent.c`.
//!
//! Every trace line starts with a local-time timestamp in the exact format
//! produced by `agent_trace_time` (`YYYY-MM-DD HH:MM:SS.mmm`), and every line
//! is flushed to disk immediately, matching the per-line `fflush` in the C
//! reference.
//!
//! Adaptation notes (plank has no real inference engine yet):
//! - The C `agent_trace_token` receives a caller-supplied generation index.
//!   Here [`Trace::token`] keeps an internal running index so the output shape
//!   (`token index=.. id=.. bytes=.. text=".." hex=..`) is preserved.
//! - The C `agent_trace_tokens` walks a `ds4_tokens` list and detokenizes via
//!   the engine. [`Trace::tokens`] instead takes `(id, text)` pairs.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Trace log writer; all methods are no-ops when opened without a path.
pub struct Trace {
    out: Option<BufWriter<File>>,
    /// Running token index, mirroring the caller-supplied index in the C code.
    token_index: i32,
}

impl std::fmt::Debug for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trace")
            .field("enabled", &self.out.is_some())
            .field("token_index", &self.token_index)
            .finish()
    }
}

impl Trace {
    /// Opens a trace log at `path`, or returns a disabled trace when `None`.
    ///
    /// A disabled trace turns every method into a no-op, mirroring the
    /// `if (!w->trace) return;` guards in the C reference.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created.
    pub fn open(path: Option<&Path>) -> std::io::Result<Self> {
        let out = match path {
            Some(p) => Some(BufWriter::new(File::create(p)?)),
            None => None,
        };
        Ok(Self {
            out,
            token_index: 0,
        })
    }

    /// Returns whether tracing is enabled.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.out.is_some()
    }

    /// Writes a timestamped line, like `agent_trace`.
    ///
    /// Callers format the message themselves (Rust's `format!` replaces the C
    /// varargs interface).
    pub fn line(&mut self, message: &str) {
        let stamp = timestamp();
        if let Some(out) = self.out.as_mut() {
            let _ = writeln!(out, "{stamp} {message}");
            let _ = out.flush();
        }
    }

    /// Writes a timestamped `label="escaped text"` line, like `agent_trace_text`.
    pub fn text(&mut self, label: &str, s: &str) {
        if self.out.is_none() {
            return;
        }
        let label = if label.is_empty() { "text" } else { label };
        let msg = format!("{label}=\"{}\"", escape(s.as_bytes()));
        self.line(&msg);
    }

    /// Writes one token line, like `agent_trace_token`.
    ///
    /// The generation index is tracked internally (see module docs); the C
    /// code receives it from the caller.
    pub fn token(&mut self, id: i32, text: &str) {
        if self.out.is_none() {
            return;
        }
        let bytes = text.as_bytes();
        let mut hex = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            let _ = write!(hex, "{b:02x}");
        }
        let msg = format!(
            "token index={} id={id} bytes={} text=\"{}\" hex={hex}",
            self.token_index,
            bytes.len(),
            escape(bytes),
        );
        self.token_index += 1;
        self.line(&msg);
    }

    /// Writes a token-list header plus one line per token, like `agent_trace_tokens`.
    ///
    /// Takes `(id, text)` pairs in place of the engine-backed `ds4_tokens`.
    pub fn tokens(&mut self, label: &str, toks: &[(i32, &str)], start: usize) {
        if self.out.is_none() {
            return;
        }
        let start = start.min(toks.len());
        let msg = format!("tokens label={label} start={start} len={}", toks.len());
        self.line(&msg);
        self.token_index = i32::try_from(start).unwrap_or(i32::MAX);
        for &(id, text) in &toks[start..] {
            self.token(id, text);
        }
    }
}

/// Formats the current local time as `YYYY-MM-DD HH:MM:SS.mmm`.
///
/// Byte-for-byte the format of `agent_trace_time`: `clock_gettime` +
/// `localtime_r` + millisecond fraction.
fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = libc::time_t::try_from(now.as_secs()).unwrap_or_default();
    let millis = now.subsec_millis();
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: `secs` and `tm` are valid, distinct pointers for the call.
    unsafe {
        libc::localtime_r(&raw const secs, &raw mut tm);
    }
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{millis:03}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
    )
}

/// Escapes bytes exactly like `agent_trace_escaped`.
///
/// `\\`, `\n`, `\r`, `\t`, and `"` get two-character escapes; other bytes
/// below 32 and byte 127 become `\xNN`; everything else passes through.
fn escape(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            b'"' => out.push_str("\\\""),
            _ if b < 32 || b == 127 => {
                let _ = write!(out, "\\x{b:02x}");
            }
            _ => out.push(b as char),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("plank-trace-test-{}-{name}", std::process::id()));
        p
    }

    /// Asserts `stamp` matches `YYYY-MM-DD HH:MM:SS.mmm` structurally.
    fn assert_timestamp_shape(stamp: &str) {
        assert_eq!(stamp.len(), 23, "bad length: {stamp:?}");
        for (i, c) in stamp.chars().enumerate() {
            match i {
                4 | 7 => assert_eq!(c, '-', "at {i} in {stamp:?}"),
                10 => assert_eq!(c, ' ', "at {i} in {stamp:?}"),
                13 | 16 => assert_eq!(c, ':', "at {i} in {stamp:?}"),
                19 => assert_eq!(c, '.', "at {i} in {stamp:?}"),
                _ => assert!(c.is_ascii_digit(), "at {i} in {stamp:?}"),
            }
        }
    }

    #[test]
    fn line_has_timestamp_prefix() {
        let path = temp_path("line");
        let mut t = Trace::open(Some(&path)).unwrap();
        assert!(t.enabled());
        t.line("hello world");
        drop(t);
        let content = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let line = content.lines().next().unwrap();
        assert_timestamp_shape(&line[..23]);
        assert_eq!(&line[23..], " hello world");
    }

    #[test]
    fn escaping_matches_c_reference() {
        assert_eq!(escape(b"plain"), "plain");
        assert_eq!(escape(b"\\"), "\\\\");
        assert_eq!(escape(b"\n\r\t\""), "\\n\\r\\t\\\"");
        assert_eq!(escape(&[0x00, 0x1f, 0x7f]), "\\x00\\x1f\\x7f");
        assert_eq!(escape(&[0x1b]), "\\x1b");
        // Byte 32 (space) and 126 pass through; 31 and 127 do not.
        assert_eq!(escape(&[31, 32, 126, 127]), "\\x1f ~\\x7f");
        // High bytes (>= 128) pass through unescaped, as in the C.
        assert_eq!(escape("é".as_bytes()), "\u{c3}\u{a9}");
    }

    #[test]
    fn text_and_token_output_shape() {
        let path = temp_path("shapes");
        let mut t = Trace::open(Some(&path)).unwrap();
        t.text("user", "a\"b\nc");
        t.text("", "x");
        t.tokens("prefill", &[(5, "hi"), (9, "\n")], 1);
        drop(t);
        let content = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(&lines[0][24..], "user=\"a\\\"b\\nc\"");
        assert_eq!(&lines[1][24..], "text=\"x\"");
        assert_eq!(&lines[2][24..], "tokens label=prefill start=1 len=2");
        assert_eq!(
            &lines[3][24..],
            "token index=1 id=9 bytes=1 text=\"\\n\" hex=0a"
        );
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn disabled_trace_is_noop() {
        let mut t = Trace::open(None).unwrap();
        assert!(!t.enabled());
        t.line("nope");
        t.text("label", "text");
        t.token(1, "x");
        t.tokens("l", &[(1, "x")], 0);
    }
}
