//! In-session named rollback points (`/checkpoint`, `/rollback`).
//!
//! A checkpoint is a snapshot of the current conversation the user can return
//! to later without leaving the session: the full transcript at capture time
//! plus, when the engine supports it, the serialized engine KV state. Rolling
//! back restores the transcript verbatim and hands the KV bytes back to the
//! engine so the next turn resumes with (near-)zero re-prefill.
//!
//! Storing the *whole* transcript — rather than a truncation offset — is what
//! lets a rollback cross a compaction boundary: the pre-compaction transcript
//! is reconstructed exactly, no matter how the live session was rewritten in
//! between.
//!
//! Checkpoints are per-session and in-memory only; they are dropped when the
//! session is replaced (`/new`, `/switch`, `/resume`) and are not persisted to
//! disk. The engine KV payload is optional: on an engine without snapshot
//! support (the `EchoEngine`), a checkpoint records only the transcript and a
//! rollback restores the text, re-prefilling on the next turn.

use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::session::{Message, Role, Session, format_age, title_clip};

/// One named rollback point captured during a session.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// User-chosen (or auto-generated) name; unique within a store.
    pub name: String,
    /// Capture time in unix seconds.
    pub created_at: u64,
    /// One-line summary of the conversation at capture time.
    pub summary: String,
    /// Full transcript snapshot, restored verbatim on rollback.
    pub transcript: Vec<Message>,
    /// Serialized engine KV state; `None` when the engine has no snapshot
    /// support, in which case rollback re-prefills.
    pub kv: Option<Vec<u8>>,
}

/// In-memory, per-session set of named checkpoints (insertion order).
#[derive(Debug, Default)]
pub struct CheckpointStore {
    entries: Vec<Checkpoint>,
}

impl CheckpointStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// True when no checkpoints have been saved.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of checkpoints held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Checkpoints in insertion order.
    #[must_use]
    pub fn list(&self) -> &[Checkpoint] {
        &self.entries
    }

    /// Drops every checkpoint (called when the session is replaced).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Captures `session` (and optional engine `kv`) under `name`.
    ///
    /// An existing checkpoint with the same name is overwritten in place;
    /// returns `true` when it replaced an existing one.
    pub fn save(&mut self, name: &str, session: &Session, kv: Option<Vec<u8>>) -> bool {
        let cp = Checkpoint {
            name: name.to_owned(),
            created_at: unix_now(),
            summary: summarize(&session.transcript),
            transcript: session.transcript.clone(),
            kv,
        };
        if let Some(slot) = self.entries.iter_mut().find(|c| c.name == name) {
            *slot = cp;
            true
        } else {
            self.entries.push(cp);
            false
        }
    }

    /// Looks up a checkpoint by exact name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Checkpoint> {
        self.entries.iter().find(|c| c.name == name)
    }
}

/// Applies a checkpoint's transcript to `session`, marking it dirty.
///
/// The engine KV restore is the caller's responsibility (it owns the engine);
/// this only rewinds the text state so the two stay consistent.
pub fn restore_transcript(session: &mut Session, cp: &Checkpoint) {
    session.transcript.clone_from(&cp.transcript);
    session.dirty = true;
}

/// One-line summary of a transcript: the newest real (non-tool) user prompt,
/// falling back to the first user prompt, clipped for display.
fn summarize(transcript: &[Message]) -> String {
    let text = transcript
        .iter()
        .rev()
        .find(|m| m.role == Role::User && !is_tool_user(m))
        .or_else(|| transcript.iter().find(|m| m.role == Role::User))
        .map(|m| m.text.as_str())
        .unwrap_or_default();
    let first_line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let clipped = title_clip(first_line.trim(), 60);
    if clipped.is_empty() {
        "(no user prompt yet)".to_owned()
    } else {
        clipped
    }
}

/// Mirror of `session`'s tool-result detection (that helper is private).
fn is_tool_user(m: &Message) -> bool {
    let t = m.text.trim();
    m.role == Role::User
        && (t.starts_with("<tool_result>")
            || t.starts_with("Tool:")
            || t.starts_with("Tool result"))
}

/// Renders the checkpoint list the way `/checkpoint` (no arg) prints it.
///
/// `now` is unix seconds; `color` enables ANSI styling matching the session
/// listing.
#[must_use]
pub fn render_list(store: &CheckpointStore, now: u64, color: bool) -> String {
    if store.is_empty() {
        return "no checkpoints in this session; use /checkpoint <name> to add one\n".to_owned();
    }
    let (name_on, sum_on, help_on, dim, reset) = if color {
        (
            "\x1b[1;96m",
            "\x1b[1;97m",
            "\x1b[97m",
            "\x1b[90m",
            "\x1b[0m",
        )
    } else {
        ("", "", "", "", "")
    };
    let mut out = String::new();
    for c in store.list() {
        let kv = if c.kv.is_some() { " +kv" } else { "" };
        let _ = writeln!(
            out,
            "{name_on}{}{reset} {dim}>{reset} {sum_on}{}{reset}",
            c.name, c.summary
        );
        let _ = writeln!(
            out,
            "         {dim}> {}{kv}{reset}\n",
            format_age(c.created_at, now)
        );
    }
    let _ = writeln!(
        out,
        "{help_on}Use /rollback <name> to return to a checkpoint.{reset}"
    );
    out
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_with(msgs: &[Message]) -> Session {
        let mut s = Session::new();
        for m in msgs {
            s.push(m.clone());
        }
        s
    }

    #[test]
    fn create_and_list() {
        let mut store = CheckpointStore::new();
        assert!(store.is_empty());
        let s = session_with(&[Message::user("fix the parser"), Message::assistant("on it")]);
        assert!(!store.save("before-fix", &s, None));
        assert_eq!(store.len(), 1);
        assert!(!store.save("second", &s, Some(vec![1, 2, 3])));
        assert_eq!(store.len(), 2);

        let cp = store.get("before-fix").unwrap();
        assert_eq!(cp.summary, "fix the parser");
        assert!(cp.kv.is_none());
        assert!(store.get("second").unwrap().kv.is_some());
        assert!(store.get("missing").is_none());

        let listed = render_list(&store, cp.created_at, false);
        assert!(listed.contains("before-fix > fix the parser"));
        assert!(listed.contains("second > fix the parser"));
        assert!(listed.contains("Use /rollback <name>"));
    }

    #[test]
    fn save_overwrites_same_name() {
        let mut store = CheckpointStore::new();
        let s1 = session_with(&[Message::user("first")]);
        assert!(!store.save("wip", &s1, None));
        let s2 = session_with(&[Message::user("first"), Message::user("second")]);
        assert!(store.save("wip", &s2, None), "same name should overwrite");
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("wip").unwrap().transcript.len(), 2);
    }

    #[test]
    fn rollback_restores_transcript() {
        let mut store = CheckpointStore::new();
        let mut session = session_with(&[
            Message::user("step one"),
            Message::assistant("did step one"),
        ]);
        store.save("cp", &session, None);

        // Session grows past the checkpoint.
        session.push(Message::user("step two"));
        session.push(Message::assistant("did step two"));
        assert_eq!(session.transcript.len(), 4);

        let cp = store.get("cp").unwrap().clone();
        restore_transcript(&mut session, &cp);
        assert_eq!(session.transcript.len(), 2);
        assert_eq!(session.transcript[0].text, "step one");
        assert_eq!(session.transcript[1].text, "did step one");
        assert!(session.dirty);
    }

    #[test]
    fn rollback_without_kv_does_not_panic() {
        // Mirrors the EchoEngine path: no KV payload is captured, and a
        // rollback must still restore the transcript cleanly.
        let mut store = CheckpointStore::new();
        let mut session = session_with(&[Message::user("only text")]);
        store.save("t", &session, None);
        session.push(Message::user("more"));
        let cp = store.get("t").unwrap().clone();
        assert!(cp.kv.is_none());
        restore_transcript(&mut session, &cp);
        assert_eq!(session.transcript.len(), 1);
    }

    #[test]
    fn empty_list_message() {
        let store = CheckpointStore::new();
        assert!(render_list(&store, 0, false).starts_with("no checkpoints in this session"));
    }

    #[test]
    fn summary_prefers_latest_user_prompt() {
        let t = vec![
            Message::user("old question"),
            Message::assistant("answer"),
            Message::user("<tool_result>ignored</tool_result>"),
            Message::user("newest question"),
        ];
        assert_eq!(summarize(&t), "newest question");
    }
}
