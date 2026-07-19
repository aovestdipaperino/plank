//! Worker/UI split for the TUI (issue #12), mirroring the C's "Model Worker
//! Thread" and "Worker/UI Synchronization Helpers" sections.
//!
//! During a turn the engine, session, and tool dispatch run on a scoped
//! worker thread while the UI thread keeps its real event loop: the next
//! prompt stays editable (and is queued for the worker to drain between tool
//! rounds, like the C's `queued_user_drain`), scrolling and interrupts work
//! directly, and redraws happen at the UI's own cadence instead of inside an
//! engine callback.
//!
//! The channel payload is [`UiEvent`]: the [`crate::viz::RenderSink`] calls
//! made by the worker's stream renderer, plus log lines and status snapshots.
//! [`RenderSink`] is the natural serialization boundary — the UI replays the
//! same calls against the real [`crate::tui::OutputLog`].

use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;

use crate::status::Status;
use crate::tui::OutputLog;
use crate::viz::RenderSink;

/// One worker→UI message during a turn.
#[derive(Debug)]
pub enum UiEvent {
    /// Rendered assistant text ([`RenderSink::visible_text`]).
    Visible(String),
    /// Thinking text ([`RenderSink::think_text`]).
    Think(String),
    /// Tool banner text ([`RenderSink::tool_text`]).
    Tool(String),
    /// Stream error text ([`RenderSink::error_text`]).
    Error(String),
    /// A dim log line (tool observations, notices, hook warnings).
    Dim(String),
    /// A plain log line.
    Plain(String),
    /// A user-echo line (queued prompts, `/btw` questions).
    UserEcho(String),
    /// Terminates the in-progress rendered line.
    EndLine,
    /// Worker progress snapshot for the status footer.
    Status(Status),
}

/// [`RenderSink`] forwarding render calls over the worker→UI channel.
///
/// A receiver that hangs up mid-turn just drops the text: the worker keeps
/// running and the transcript (owned by the worker) stays authoritative.
#[derive(Debug)]
pub struct ChannelSink(pub Sender<UiEvent>);

impl RenderSink for ChannelSink {
    fn visible_text(&mut self, text: &str) {
        let _ = self.0.send(UiEvent::Visible(text.to_owned()));
    }
    fn think_text(&mut self, text: &str) {
        let _ = self.0.send(UiEvent::Think(text.to_owned()));
    }
    fn tool_text(&mut self, text: &str) {
        let _ = self.0.send(UiEvent::Tool(text.to_owned()));
    }
    fn error_text(&mut self, text: &str) {
        let _ = self.0.send(UiEvent::Error(text.to_owned()));
    }
}

/// State shared by the UI and worker threads for one turn.
#[derive(Debug, Default)]
pub struct TurnShared {
    /// Set by the UI (Esc / Ctrl-C / SIGINT) to stop the worker at the next
    /// sampling or prefill checkpoint.
    pub interrupt: AtomicBool,
    /// User lines typed while the worker is busy. The worker drains them into
    /// the transcript between tool rounds (the C's `queued_user_drain`);
    /// lines still queued when the turn ends start a fresh turn.
    pub queued: Mutex<Vec<String>>,
    /// `/btw` side questions queued while the worker is busy, answered FIFO
    /// at generation boundaries (modeled on `OpenClaw`'s side-question queue).
    pub btw: Mutex<Vec<String>>,
}

/// Cap on queued `/btw` questions; a push beyond it drops the oldest entry
/// (`OpenClaw`'s bounded-buffer `drop-oldest` overflow policy).
pub const BTW_QUEUE_CAP: usize = 20;

impl TurnShared {
    /// Takes all queued user lines.
    pub fn take_queued(&self) -> Vec<String> {
        self.queued.lock().map_or_else(
            |e| std::mem::take(&mut *e.into_inner()),
            |mut q| std::mem::take(&mut *q),
        )
    }

    /// Queues one user line for the worker.
    pub fn push_queued(&self, line: String) {
        match self.queued.lock() {
            Ok(mut q) => q.push(line),
            Err(e) => e.into_inner().push(line),
        }
    }

    /// Queues one `/btw` question (FIFO, capped at [`BTW_QUEUE_CAP`]).
    /// Returns the oldest entry when the cap forced it out, so the caller can
    /// surface a visible drop notice instead of losing it silently.
    pub fn push_btw(&self, question: String) -> Option<String> {
        let mut q = match self.btw.lock() {
            Ok(q) => q,
            Err(e) => e.into_inner(),
        };
        q.push(question);
        if q.len() > BTW_QUEUE_CAP {
            Some(q.remove(0))
        } else {
            None
        }
    }

    /// Takes the oldest queued `/btw` question.
    pub fn pop_btw(&self) -> Option<String> {
        let mut q = match self.btw.lock() {
            Ok(q) => q,
            Err(e) => e.into_inner(),
        };
        if q.is_empty() {
            None
        } else {
            Some(q.remove(0))
        }
    }

    /// Drops all queued `/btw` questions, returning how many were cleared.
    pub fn clear_btw(&self) -> usize {
        let mut q = match self.btw.lock() {
            Ok(q) => q,
            Err(e) => e.into_inner(),
        };
        let n = q.len();
        q.clear();
        n
    }

    /// Takes all queued `/btw` questions.
    pub fn take_btw(&self) -> Vec<String> {
        match self.btw.lock() {
            Ok(mut q) => std::mem::take(&mut *q),
            Err(e) => std::mem::take(&mut *e.into_inner()),
        }
    }
}

/// Applies one non-status event to the TUI output log.
pub fn apply(log: &mut OutputLog, ev: UiEvent) {
    match ev {
        UiEvent::Visible(t) => log.visible_text(&t),
        UiEvent::Think(t) => log.think_text(&t),
        UiEvent::Tool(t) => log.tool_text(&t),
        UiEvent::Error(t) => log.error_text(&t),
        UiEvent::Dim(t) => log.push_dim(t),
        UiEvent::Plain(t) => log.push_plain(t),
        UiEvent::UserEcho(t) => log.push_spans(crate::tui::user_echo_spans(&t)),
        UiEvent::EndLine => log.end_line(),
        UiEvent::Status(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_sink_forwards_render_calls_in_order() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut sink = ChannelSink(tx);
        sink.visible_text("a");
        sink.think_text("b");
        sink.tool_text("c");
        sink.error_text("d");
        let got: Vec<UiEvent> = rx.try_iter().collect();
        assert!(matches!(&got[0], UiEvent::Visible(t) if t == "a"));
        assert!(matches!(&got[1], UiEvent::Think(t) if t == "b"));
        assert!(matches!(&got[2], UiEvent::Tool(t) if t == "c"));
        assert!(matches!(&got[3], UiEvent::Error(t) if t == "d"));
    }

    #[test]
    fn btw_queue_is_fifo_and_drops_oldest_beyond_cap() {
        let shared = TurnShared::default();
        for i in 0..(BTW_QUEUE_CAP + 2) {
            let dropped = shared.push_btw(format!("q{i}"));
            match i {
                i if i < BTW_QUEUE_CAP => assert!(dropped.is_none()),
                i if i == BTW_QUEUE_CAP => assert_eq!(dropped.as_deref(), Some("q0")),
                _ => assert_eq!(dropped.as_deref(), Some("q1")),
            }
        }
        assert_eq!(shared.pop_btw().as_deref(), Some("q2"));
        assert_eq!(shared.take_btw().len(), BTW_QUEUE_CAP - 1);
        shared.push_btw("late".to_owned());
        assert_eq!(shared.clear_btw(), 1);
        assert!(shared.pop_btw().is_none());
    }

    #[test]
    fn queued_lines_round_trip() {
        let shared = TurnShared::default();
        shared.push_queued("one".into());
        shared.push_queued("two".into());
        assert_eq!(shared.take_queued(), vec!["one", "two"]);
        assert!(shared.take_queued().is_empty());
    }
}
