// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

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
#[derive(Debug, Clone)]
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
    /// An agent system status line (`agent_publish_system_status`): carries the
    /// bare message, styled by the UI with [`crate::status::system_line`].
    SystemStatus(String),
    /// A git-style diff card for a file an `edit`/`write` call changed.
    EditCard(crate::tools::diff::EditPreview),
    /// A plain log line.
    Plain(String),
    /// A user-echo line (queued prompts, `/btw` questions).
    UserEcho(String),
    /// The transcript was replaced — `/clear`, `/new`, `/switch`, `/resume`.
    /// Everything above it belongs to a session that no longer exists, so a
    /// remote front-end clears its log, and [`BroadcastBus::broadcast`] drops
    /// the scrollback rather than replaying a cleared transcript to whoever
    /// attaches next. The local UI clears its own log at the call site.
    SessionReset,
    /// The turn finished: the same headline and body the local desktop
    /// notification carries, so an attached remote front-end can raise its own
    /// (a browser notification, a bell, a title flash) instead of the operator
    /// having to watch the page. Carries no on-screen text — the local UI
    /// already showed the turn — so the TUI ignores it.
    Notify {
        /// Notification headline (the prompt, or that the turn was interrupted).
        title: String,
        /// Notification body (the tail of the answer).
        body: String,
    },
    /// Terminates the in-progress rendered line.
    EndLine,
    /// Worker progress snapshot for the status footer.
    Status(Status),
    /// Task list snapshot for the status-bar counter and the contextual strip
    /// (issue #35), sent whenever a `task` tool call changes the list.
    Tasks(crate::tui::TaskView),
    /// A `/btw` side answer is starting: the UI opens the side panel (if not
    /// already visible) and routes subsequent render events into it.
    BtwBegin,
    /// The `/btw` answer finished (or was cancelled): the UI stops routing to
    /// the panel and resumes rendering into the main log, but leaves the panel
    /// on screen (frozen) so the answer stays readable while the main task
    /// continues. The panel is dismissed by the user with Esc, not here.
    BtwEnd,
    /// Marks the start of a main generation pass: the UI snapshots the main
    /// log length so a later [`UiEvent::MainRollback`] can discard a
    /// preempted pass's partial output.
    MainCheckpoint,
    /// A main pass was preempted by a priority `/btw`: the UI truncates the
    /// main log back to the last [`UiEvent::MainCheckpoint`] so the re-run
    /// does not duplicate the discarded partial output.
    MainRollback,
    /// A sub-agent run began; the payload is its display label (the configured
    /// agent name, or a generic one for an unnamed sub-agent). The UI clears
    /// the sub-agent buffer and starts filling it.
    SubStart(String),
    /// The sub-agent run finished, successfully or not. The buffer is kept so
    /// the user can still read it.
    SubEnd,
    /// One render payload destined for the sub-agent buffer rather than the
    /// main log. Boxed so the enum does not grow by a whole `UiEvent`.
    Sub(Box<UiEvent>),
    /// One render payload destined for the `/btw` panel, addressed per event
    /// rather than by the [`BtwBegin`](Self::BtwBegin)/[`BtwEnd`](Self::BtwEnd)
    /// mode.
    ///
    /// The mode cannot serve a *multiplexed* aside: the main task and the aside
    /// interleave, so their events arrive mixed and a mode flag would misroute
    /// whichever stream it is not currently set to. Boxed for the same reason
    /// as [`Sub`](Self::Sub).
    Btw(Box<UiEvent>),
}

impl UiEvent {
    /// Whether this event belongs to the local sub-agent pane only and must not
    /// go onto the [`BroadcastBus`]. A remote-side pane is out of scope, so
    /// `ServerMsg::from_event` maps these three to `None` — broadcasting them
    /// anyway would burn scrollback ring slots (one `Sub` per streamed chunk)
    /// and evict the real transcript a reconnecting client replays. The remote
    /// protocol does not require contiguous sequence ids, so simply not
    /// broadcasting is safe.
    #[must_use]
    pub fn is_local_pane_only(&self) -> bool {
        matches!(
            self,
            Self::SubStart(_) | Self::SubEnd | Self::Sub(_) | Self::Btw(_)
        )
    }
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

/// [`RenderSink`] forwarding a sub-agent's render calls over the worker→UI
/// channel, wrapped in [`UiEvent::Sub`] so the UI routes them to the sub-agent
/// buffer instead of the main log. Same hang-up tolerance as [`ChannelSink`].
#[derive(Debug)]
pub struct SubAgentSink(pub Sender<UiEvent>);

impl RenderSink for SubAgentSink {
    fn visible_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Sub(Box::new(UiEvent::Visible(text.to_owned()))));
    }
    fn think_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Sub(Box::new(UiEvent::Think(text.to_owned()))));
    }
    fn tool_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Sub(Box::new(UiEvent::Tool(text.to_owned()))));
    }
    fn error_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Sub(Box::new(UiEvent::Error(text.to_owned()))));
    }
}

/// [`RenderSink`] forwarding a `/btw` aside's render calls, wrapped in
/// [`UiEvent::Btw`] so they reach the side panel even while the main task is
/// streaming into the main log beside them. Same hang-up tolerance as
/// [`ChannelSink`].
///
/// A multiplexed aside needs its own renderer, not just its own sink: without
/// one its raw model output reaches the panel with the `<think>` tags still in
/// it.
#[derive(Debug)]
pub struct BtwSink(pub Sender<UiEvent>);

impl RenderSink for BtwSink {
    fn visible_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Btw(Box::new(UiEvent::Visible(text.to_owned()))));
    }
    fn think_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Btw(Box::new(UiEvent::Think(text.to_owned()))));
    }
    fn tool_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Btw(Box::new(UiEvent::Tool(text.to_owned()))));
    }
    fn error_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Btw(Box::new(UiEvent::Error(text.to_owned()))));
    }
}

/// A sequenced worker event: a monotonic `id` plus the [`UiEvent`]. The id lets
/// a reconnecting remote client resume with `resume_from` (see
/// `docs/REMOTE-CONTROL-DESIGN.md` §4.8) without re-seeing frames it already has.
#[derive(Debug, Clone)]
pub struct SeqEvent {
    /// Monotonic sequence id, unique and increasing per [`BroadcastBus`].
    pub id: u64,
    /// The event payload.
    pub event: UiEvent,
}

/// Default cap on the scrollback ring, in events. Bounds memory while giving a
/// late-joining remote client enough recent context to replay a `snapshot`.
pub const SCROLLBACK_CAP: usize = 4096;

/// Fan-out of [`UiEvent`]s to any number of consumers (the local TUI plus each
/// remote session), the single structural change remote control needs
/// (`docs/REMOTE-CONTROL-DESIGN.md` §4.2).
///
/// `broadcast` clones each event to every live subscriber and prunes hung-up
/// ones; a slow or vanished subscriber never blocks the worker, inheriting the
/// [`ChannelSink`] resilience contract. A bounded scrollback ring with
/// monotonic sequence ids backs late-join replay and `resume_from`.
#[derive(Debug, Default)]
pub struct BroadcastBus {
    inner: Mutex<BusInner>,
}

#[derive(Debug, Default)]
struct BusInner {
    subscribers: Vec<Sender<SeqEvent>>,
    scrollback: std::collections::VecDeque<SeqEvent>,
    next_id: u64,
}

impl BroadcastBus {
    /// A new, empty bus.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new consumer, returning the receiver end. The subscriber is
    /// pruned automatically on the first `broadcast` after it hangs up.
    pub fn subscribe(&self) -> std::sync::mpsc::Receiver<SeqEvent> {
        let (tx, rx) = std::sync::mpsc::channel();
        Self::lock(&self.inner).subscribers.push(tx);
        rx
    }

    /// Assigns the next sequence id, appends to the scrollback ring (evicting
    /// the oldest beyond [`SCROLLBACK_CAP`]), and fans the event out to every
    /// live subscriber, dropping any that have hung up.
    pub fn broadcast(&self, event: UiEvent) {
        let mut inner = Self::lock(&self.inner);
        // A reset invalidates everything before it: drop the retained tail so a
        // client attaching afterwards is not replayed the transcript the
        // operator just cleared. Ids keep climbing, so an already-connected
        // client's `resume_from` stays meaningful across the reset.
        if matches!(event, UiEvent::SessionReset) {
            inner.scrollback.clear();
        }
        let id = inner.next_id;
        inner.next_id += 1;
        let seq = SeqEvent { id, event };
        inner.scrollback.push_back(seq.clone());
        while inner.scrollback.len() > SCROLLBACK_CAP {
            inner.scrollback.pop_front();
        }
        inner.subscribers.retain(|tx| tx.send(seq.clone()).is_ok());
    }

    /// Returns scrollback events with `id > resume_from` for late-join replay.
    /// A `resume_from` older than the retained tail yields the whole tail
    /// (best-effort resume, §4.8). Also returns the highest id assigned so far,
    /// or `None` when nothing has been broadcast yet.
    #[must_use]
    pub fn scrollback_since(&self, resume_from: Option<u64>) -> (Vec<SeqEvent>, Option<u64>) {
        let inner = Self::lock(&self.inner);
        let highest = inner.next_id.checked_sub(1);
        let tail = inner
            .scrollback
            .iter()
            .filter(|s| resume_from.is_none_or(|from| s.id > from))
            .cloned()
            .collect();
        (tail, highest)
    }

    /// Number of currently registered subscribers (test/introspection helper).
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        Self::lock(&self.inner).subscribers.len()
    }

    fn lock(m: &Mutex<BusInner>) -> std::sync::MutexGuard<'_, BusInner> {
        m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A [`RenderSink`] that forwards render calls into a [`BroadcastBus`]. Mirrors
/// [`ChannelSink`] but fans out to every subscriber instead of one channel.
#[derive(Debug)]
pub struct BusSink<'a>(pub &'a BroadcastBus);

impl RenderSink for BusSink<'_> {
    fn visible_text(&mut self, text: &str) {
        self.0.broadcast(UiEvent::Visible(text.to_owned()));
    }
    fn think_text(&mut self, text: &str) {
        self.0.broadcast(UiEvent::Think(text.to_owned()));
    }
    fn tool_text(&mut self, text: &str) {
        self.0.broadcast(UiEvent::Tool(text.to_owned()));
    }
    fn error_text(&mut self, text: &str) {
        self.0.broadcast(UiEvent::Error(text.to_owned()));
    }
}

/// State shared by the UI and worker threads for one turn.
#[derive(Debug, Default)]
pub struct TurnShared {
    /// Set by the UI (Esc / Ctrl-C / SIGINT) to stop the worker at the next
    /// sampling or prefill checkpoint.
    pub interrupt: AtomicBool,
    /// Set by the UI when a `/btw` is submitted mid-turn: the current main
    /// generation pass stops immediately so the side question is answered
    /// with priority, then the interrupted pass is re-run (nothing is
    /// committed until a pass completes, so the restart is transcript-safe).
    pub preempt: AtomicBool,
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

/// Dim marker shown when a mid-generation `/btw` freezes the main task
/// (BTW-SUSPEND-DESIGN §4.4). Emitted into the main log before the aside's
/// side panel opens.
pub const BTW_SUSPEND_MARKER: &str = "[btw — main task paused]";

/// Dim marker shown when the frozen main task resumes after the aside(s)
/// (BTW-SUSPEND-DESIGN §4.4).
pub const BTW_RESUME_MARKER: &str = "[btw — resuming]";

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
        UiEvent::SystemStatus(t) => log.push_ansi(&crate::status::system_line(&t, true)),
        UiEvent::EditCard(p) => crate::tui::render_diff_card(log, &p),
        UiEvent::Plain(t) => log.push_plain(t),
        UiEvent::UserEcho(t) => log.push_spans(crate::tui::user_echo_spans(&t)),
        UiEvent::EndLine => log.end_line(),
        // Both are for remote front-ends only: locally the desktop notification
        // has already fired, and the log was cleared at the `/clear` call site.
        UiEvent::Notify { .. }
        | UiEvent::SessionReset
        | UiEvent::Status(_)
        | UiEvent::Tasks(_)
        | UiEvent::BtwBegin
        | UiEvent::BtwEnd
        | UiEvent::MainCheckpoint
        | UiEvent::MainRollback
        | UiEvent::SubStart(_)
        | UiEvent::SubEnd
        | UiEvent::Sub(_)
        | UiEvent::Btw(_) => {}
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
    fn sub_agent_sink_wraps_render_calls_in_sub_events() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut sink = SubAgentSink(tx);
        sink.visible_text("a");
        sink.think_text("b");
        sink.tool_text("c");
        sink.error_text("d");
        drop(sink);
        let got: Vec<UiEvent> = rx.iter().collect();
        assert!(
            matches!(got.as_slice(), [
                UiEvent::Sub(a),
                UiEvent::Sub(b),
                UiEvent::Sub(c),
                UiEvent::Sub(d),
            ] if matches!(**a, UiEvent::Visible(ref s) if s == "a")
                && matches!(**b, UiEvent::Think(ref s) if s == "b")
                && matches!(**c, UiEvent::Tool(ref s) if s == "c")
                && matches!(**d, UiEvent::Error(ref s) if s == "d")),
            "unexpected events: {got:?}"
        );
    }

    #[test]
    fn sub_agent_events_are_local_pane_only_and_never_reach_the_bus() {
        assert!(UiEvent::SubStart("agent".to_string()).is_local_pane_only());
        assert!(UiEvent::SubEnd.is_local_pane_only());
        assert!(UiEvent::Sub(Box::new(UiEvent::Visible("x".to_string()))).is_local_pane_only());
        assert!(!UiEvent::Visible("v".to_string()).is_local_pane_only());
        assert!(!UiEvent::Dim("d".to_string()).is_local_pane_only());

        // The mirror sites filter on that predicate, so a chatty sub-agent run
        // cannot evict the real transcript from the bounded scrollback ring.
        let bus = BroadcastBus::new();
        let rx = bus.subscribe();
        let stream = [
            UiEvent::Visible("before".to_string()),
            UiEvent::SubStart("agent".to_string()),
            UiEvent::Sub(Box::new(UiEvent::Visible("noise".to_string()))),
            UiEvent::SubEnd,
            UiEvent::Visible("after".to_string()),
        ];
        for ev in stream {
            if !ev.is_local_pane_only() {
                bus.broadcast(ev);
            }
        }
        let got: Vec<UiEvent> = rx.try_iter().map(|s| s.event).collect();
        assert!(
            matches!(got.as_slice(), [UiEvent::Visible(a), UiEvent::Visible(b)]
                if a == "before" && b == "after"),
            "unexpected events: {got:?}"
        );
        // And the ring a reconnecting client replays holds only those two.
        assert_eq!(bus.scrollback_since(None).0.len(), 2);
    }

    #[test]
    fn apply_ignores_sub_events_on_the_main_log() {
        // The main log must never absorb sub-agent output: the UI unwraps `Sub`
        // and applies the inner event to the sub-agent buffer instead.
        let mut log = OutputLog::new();
        let before = log.line_count();
        apply(&mut log, UiEvent::SubStart("research".to_string()));
        apply(
            &mut log,
            UiEvent::Sub(Box::new(UiEvent::Visible("x".to_string()))),
        );
        apply(&mut log, UiEvent::SubEnd);
        assert_eq!(log.line_count(), before);
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
    fn bus_fans_out_to_multiple_subscribers() {
        let bus = BroadcastBus::new();
        let a = bus.subscribe();
        let b = bus.subscribe();
        bus.broadcast(UiEvent::Visible("hi".into()));
        for rx in [&a, &b] {
            let got = rx.try_recv().expect("subscriber received");
            assert_eq!(got.id, 0);
            assert!(matches!(got.event, UiEvent::Visible(ref t) if t == "hi"));
        }
        // A dropped subscriber is pruned on the next broadcast and does not
        // stall the survivor.
        drop(b);
        bus.broadcast(UiEvent::Visible("again".into()));
        assert_eq!(bus.subscriber_count(), 1);
        let got = a.try_recv().expect("survivor still receives");
        assert_eq!(got.id, 1);
    }

    #[test]
    fn bus_scrollback_replays_on_late_join() {
        let bus = BroadcastBus::new();
        bus.broadcast(UiEvent::Visible("one".into()));
        bus.broadcast(UiEvent::Visible("two".into()));
        // Late joiner replays the full tail...
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 2);
        assert_eq!(highest, Some(1));
        // ...and a resume_from only sees newer events.
        let (since, _) = bus.scrollback_since(Some(0));
        assert_eq!(since.len(), 1);
        assert_eq!(since[0].id, 1);
        // Empty bus reports no highest id.
        assert_eq!(BroadcastBus::new().scrollback_since(None).1, None);
    }

    #[test]
    fn bus_scrollback_is_bounded() {
        let bus = BroadcastBus::new();
        for _ in 0..(SCROLLBACK_CAP + 10) {
            bus.broadcast(UiEvent::EndLine);
        }
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), SCROLLBACK_CAP);
        assert_eq!(highest, Some((SCROLLBACK_CAP + 10 - 1) as u64));
        // Oldest ids were evicted; resume from a rolled-past id yields the tail.
        assert_eq!(tail.first().unwrap().id, 10);
    }

    /// A session reset drops the scrollback with it. Otherwise a client that
    /// attaches after `/clear` is replayed the transcript the operator just
    /// cleared — the reset would only be visible to whoever was already
    /// connected to see the frame go past.
    #[test]
    fn session_reset_drops_the_scrollback_it_invalidates() {
        let bus = BroadcastBus::new();
        bus.broadcast(UiEvent::Visible("stale answer".into()));
        bus.broadcast(UiEvent::UserEcho("stale prompt".into()));
        let before = bus.scrollback_since(None).0.len();
        assert_eq!(before, 2);

        bus.broadcast(UiEvent::SessionReset);
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 1, "only the reset survives: {tail:?}");
        assert!(matches!(tail[0].event, UiEvent::SessionReset));
        // Ids stay monotonic across the reset, so an already-connected client's
        // `resume_from` is still meaningful and does not replay the reset twice.
        assert_eq!(highest, Some(2));
        assert_eq!(tail[0].id, 2);

        // Output after the reset accumulates normally again.
        bus.broadcast(UiEvent::Visible("fresh".into()));
        assert_eq!(bus.scrollback_since(None).0.len(), 2);
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
