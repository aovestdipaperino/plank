//! System prompt rendering: tool prompt text, reminders, datetime context.
//!
//! Port of the prompt-construction half of the "System Prompt Rendering And
//! Worker Output Queues" section of `ds4_agent.c` (roughly lines 703-1065).
//! The long tool-protocol strings are model-facing and replicated verbatim
//! from the C reference.

use std::time::{SystemTime, UNIX_EPOCH};

/// Introductory section of the tools prompt (verbatim from C).
const TOOLS_PROMPT_INTRO: &str = "You are a coding agent running in a local workspace. Use tools for local file and system work. \
Avoid printing large file contents or large code blocks as answers; create or edit files with tools, \
then summarize results briefly.\n\n\
## Tools\n\n\
You have access to native DSML tools. Invoke tools by writing exactly this shape:\n\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\n\
Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.\n\n\
String parameters use raw text and string=\"true\". Numbers and booleans use JSON text and string=\"false\".\n\n\
Read defaults to a bounded chunk: path alone returns the first 500 lines, not the whole file. \
If read says more lines are available, call more with count=<lines> to read the next chunk; \
more defaults to the next 500 lines. \
The read result also reports continue_offset=N, which is the next start_line if you need to jump manually. \
If the user explicitly asks you to read a complete file into context, call read with whole=true. \
A whole-file read may fail if the result would not fit the current context; then explain that and use chunks.\n\n";

/// Editing-instructions section of the tools prompt (verbatim from C).
const TOOLS_PROMPT_EDIT_LINE: &str = "## Editing files\n\n\
Use write for new files or deliberate whole-file replacement. Use edit with path, old, and new for changes. \
For edit, always put the edited file path as the first parameter. \
The old text must match exactly once in the current file; otherwise edit fails for safety.\n\
For large replacements, prefer anchored old text: write the first lines, then [upto], then the final lines. \
The tool replaces everything from the head through the tail. If the head or tail is ambiguous, the edit fails.\n\
After [upto], always write unique final lines before closing old; never close old immediately after [upto].\n\
Do not use a generic tail anchor like:\n\
- BigNum bignum_add(BigNum *a, BigNum *b) {\n\
- [upto]\n\
- }\n\
because the closing brace may match many functions. Instead include final lines that are unique near that function, \
for example its last calculation and return line before the brace.\n\
Example anchored edit:\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"edit\">\n\
<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/example.c</｜DSML｜parameter>\n\
<｜DSML｜parameter name=\"old\" string=\"true\">static int parse(void) {\n    int ok = 0;\n\
[upto]\n    return ok;\n\
}</｜DSML｜parameter>\n\
<｜DSML｜parameter name=\"new\" string=\"true\">static int parse(void) {\n    return parse_impl();\n\
}</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\
To insert text, use edit with old set to an exact unique anchor and new set to that anchor plus the added text.\n\
Use read raw=true only when you need plain file text without line numbers or read annotations.\n\n";

/// Trailing section of the tools prompt: web tools, schemas, rules.
///
/// Byte-identical to the C `agent_tools_prompt_after_edit`. Kept as a
/// resource file because a `\`-continued Rust string literal strips the
/// next line's leading whitespace, silently deleting the indentation the
/// JSON schemas carry (see FINDINGS.md); `tests/c_parity.rs` enforces the
/// byte identity.
const TOOLS_PROMPT_AFTER_EDIT: &str = include_str!("resources/tools_prompt_after_edit.txt");

/// Token-estimate distance after which the system prompt reminder is re-injected.
pub const SYSTEM_PROMPT_REMINDER_TOKENS: i32 = 50_000;

/// Builds the full tools prompt (intro, editing, schemas, rules, MCP tools).
///
/// Mirrors `agent_build_tools_prompt`: the three verbatim C string constants
/// followed by the schemas of any MCP tools loaded at startup.
#[must_use]
pub fn build_tools_prompt(mcp_servers: &[crate::tools::mcp::McpServer]) -> String {
    let mut out = String::with_capacity(
        TOOLS_PROMPT_INTRO.len() + TOOLS_PROMPT_EDIT_LINE.len() + TOOLS_PROMPT_AFTER_EDIT.len(),
    );
    out.push_str(TOOLS_PROMPT_INTRO);
    out.push_str(TOOLS_PROMPT_EDIT_LINE);
    out.push_str(TOOLS_PROMPT_AFTER_EDIT);
    crate::tools::mcp::append_tool_schemas(&mut out, mcp_servers);
    crate::tools::mcp::append_server_instructions(&mut out, mcp_servers);
    out
}

/// Returns the short DSML syntax reminder (verbatim from C).
#[must_use]
pub fn dsml_syntax_reminder() -> &'static str {
    "DSML syntax reminder:\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n"
}

/// Builds the full system prompt reminder block, framed like the C version.
///
/// Mirrors `agent_build_system_prompt_reminder`: the tools prompt wrapped in
/// start/end reminder markers.
#[must_use]
pub fn build_system_prompt_reminder(mcp_servers: &[crate::tools::mcp::McpServer]) -> String {
    let mut out = String::from("\n\n[System prompt reminder follows.]\n");
    out.push_str(&build_tools_prompt(mcp_servers));
    out.push_str("[End system prompt reminder.]\n\n");
    out
}

/// Composes the initial system prompt: tools prompt plus optional user text.
///
/// Mirrors `agent_append_system_prompt`: the built-in tools prompt comes
/// first, and non-empty user `-sys` text is appended after a blank line. In
/// the C agent the two parts are tokenized differently (the built-in prompt
/// as rendered chat so DSML markers become control tokens, user text as plain
/// content); here both are returned as one composed string.
#[must_use]
/// **Cache-boundary rule** (docs/SYSTEM-PROMPT.md): everything composed here
/// enters the fingerprinted `sysprompt.kv` KV prefix, so only inputs that are
/// stable across sessions are allowed — the verbatim tools prompt, MCP
/// schemas/instructions, and `-sys` text. Per-session data (date, git state,
/// AGENTS.md) belongs in [`crate::context::ContextContent`] instead; the
/// `fingerprinted_prompt_contains_no_volatile_bytes` test guards this.
pub fn build_system_prompt(
    user_system: &str,
    mcp_servers: &[crate::tools::mcp::McpServer],
) -> String {
    let mut out = build_tools_prompt(mcp_servers);
    if !user_system.is_empty() {
        out.push_str("\n\n");
        out.push_str(user_system);
    }
    out
}

/// Formats the session-start datetime context line for the given instant.
///
/// Mirrors `agent_worker_maybe_append_datetime_context`: the timestamp is the
/// local time formatted as `%Y-%m-%d %H:%M:%S %Z`, falling back to the raw
/// Unix seconds if formatting fails.
#[must_use]
pub fn datetime_context_line(now: SystemTime) -> String {
    let secs = match now.duration_since(UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        Err(e) => -i64::try_from(e.duration().as_secs()).unwrap_or(i64::MAX),
    };
    let when = format_local(secs).unwrap_or_else(|| secs.to_string());
    format!(
        "Current local date and time at session start: {when}. \
         Use this only when date or time matters."
    )
}

/// Formats Unix seconds as local time `%Y-%m-%d %H:%M:%S %Z`, or `None` on failure.
fn format_local(secs: i64) -> Option<String> {
    let t: libc::time_t = secs;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: `t` and `tm` are valid for reads/writes; localtime_r fills `tm`
    // or returns NULL on failure.
    if unsafe { libc::localtime_r(&raw const t, &raw mut tm) }.is_null() {
        return None;
    }
    let mut buf = [0u8; 128];
    let fmt = c"%Y-%m-%d %H:%M:%S %Z";
    // SAFETY: `buf` is a writable buffer of the given length, `fmt` and `tm`
    // are valid; strftime NUL-terminates on success and returns 0 on failure.
    let n = unsafe {
        libc::strftime(
            buf.as_mut_ptr().cast::<libc::c_char>(),
            buf.len(),
            fmt.as_ptr(),
            &raw const tm,
        )
    };
    if n == 0 {
        return None;
    }
    Some(String::from_utf8_lossy(&buf[..n]).into_owned())
}

/// Pressure-controlled policy for re-injecting the system prompt reminder.
///
/// Mirrors `agent_worker_maybe_append_system_prompt_reminder` together with
/// `agent_worker_note_system_prompt_seen`. Positions are token-estimate
/// offsets into the transcript (the C code uses `transcript.len`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SystemPromptReminder {
    last_reminder_at: i32,
}

impl SystemPromptReminder {
    /// Creates a policy that has not yet seen a system prompt.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that the system prompt was (re)seen at `current_pos`.
    pub fn note_seen(&mut self, current_pos: i32) {
        self.last_reminder_at = current_pos;
    }

    /// Decides whether to re-inject the reminder at `current_pos`.
    ///
    /// Returns `true` when at least [`SYSTEM_PROMPT_REMINDER_TOKENS`] have
    /// accumulated since the prompt was last seen; the caller must then
    /// inject [`build_system_prompt_reminder`]. As in the C code, a
    /// non-positive last-seen position only records the current position.
    pub fn should_remind(&mut self, current_pos: i32) -> bool {
        if self.last_reminder_at <= 0 {
            self.note_seen(current_pos);
            return false;
        }
        if current_pos - self.last_reminder_at < SYSTEM_PROMPT_REMINDER_TOKENS {
            return false;
        }
        self.note_seen(current_pos);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn tools_prompt_contains_verbatim_phrases() {
        let p = build_tools_prompt(&[]);
        assert!(p.starts_with("You are a coding agent running in a local workspace."));
        assert!(p.contains("<｜DSML｜tool_calls>"));
        assert!(p.contains("Tool calls are not allowed inside <think></think>"));
        assert!(p.contains("## Editing files"));
        assert!(p.contains("never close old immediately after [upto]"));
        assert!(p.contains("Use google_search to find web pages."));
        assert!(p.contains("### Available Tool Schemas"));
        assert!(p.contains("\"name\": \"bash_stop\""));
        assert!(p.contains("- Always use strict syntax for DSML tool stanzas.\n"));
        assert!(p.ends_with("unless explicitly asked otherwise by the user.\n"));
    }

    /// Guards the static/volatile boundary (docs/SYSTEM-PROMPT.md): the
    /// composed system prompt is what `sysprompt.kv` fingerprints, so any
    /// per-session bytes (date, git state, AGENTS.md) sneaking in would make
    /// the disk snapshot rebuild on every launch. Volatile context belongs in
    /// the first user turn (`context::ContextContent`), never here.
    #[test]
    fn fingerprinted_prompt_contains_no_volatile_bytes() {
        let a = build_system_prompt("user -sys text", &[]);
        let b = build_system_prompt("user -sys text", &[]);
        assert_eq!(a, b, "system prompt must be deterministic");
        let today = crate::context::current_local_iso_date();
        assert!(
            !a.contains(&today),
            "today's date leaked into the cached prefix"
        );
        for marker in [
            "Today's date",
            "This is the git status",
            "Current branch:",
            "Main branch",
            "Git user:",
            "Agent instructions:",
        ] {
            assert!(
                !a.contains(marker),
                "volatile marker {marker:?} leaked into the cached prefix"
            );
        }
    }

    #[test]
    fn dsml_reminder_shape() {
        let r = dsml_syntax_reminder();
        assert!(r.starts_with("DSML syntax reminder:\n"));
        assert!(r.contains("<｜DSML｜invoke name=\"$TOOL_NAME\">"));
        assert!(r.ends_with("</｜DSML｜tool_calls>\n"));
    }

    #[test]
    fn system_prompt_reminder_framing() {
        let r = build_system_prompt_reminder(&[]);
        assert!(r.starts_with("\n\n[System prompt reminder follows.]\n"));
        assert!(r.ends_with("[End system prompt reminder.]\n\n"));
        assert!(r.contains("## Tools"));
    }

    #[test]
    fn system_prompt_composition() {
        assert_eq!(build_system_prompt("", &[]), build_tools_prompt(&[]));
        let with_extra = build_system_prompt("Be terse.", &[]);
        assert!(with_extra.starts_with(&build_tools_prompt(&[])));
        assert!(with_extra.ends_with("\n\nBe terse."));
    }

    #[test]
    fn reminder_policy_thresholds() {
        let mut r = SystemPromptReminder::new();
        // First call only records the position.
        assert!(!r.should_remind(1000));
        // Below threshold: no reminder.
        assert!(!r.should_remind(1000 + SYSTEM_PROMPT_REMINDER_TOKENS - 1));
        // At threshold: reminder fires and position resets.
        assert!(r.should_remind(1000 + SYSTEM_PROMPT_REMINDER_TOKENS));
        // Immediately after, no reminder again.
        assert!(!r.should_remind(1000 + SYSTEM_PROMPT_REMINDER_TOKENS + 10));
    }

    #[test]
    fn datetime_line_shape() {
        let now = UNIX_EPOCH + Duration::from_secs(1_752_800_000);
        let line = datetime_context_line(now);
        assert!(line.starts_with("Current local date and time at session start: "));
        assert!(line.ends_with("Use this only when date or time matters."));
        // Local date portion: YYYY-MM-DD HH:MM:SS.
        let ts = line
            .strip_prefix("Current local date and time at session start: ")
            .unwrap();
        let bytes = ts.as_bytes();
        assert_eq!(&bytes[4..5], b"-");
        assert_eq!(&bytes[7..8], b"-");
        assert_eq!(&bytes[10..11], b" ");
        assert_eq!(&bytes[13..14], b":");
        assert_eq!(&bytes[16..17], b":");
    }
}
