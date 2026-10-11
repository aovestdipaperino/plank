//! Opt-in programming hints, `/hints on|off` (upstream ds4 `6546ae5`).
//!
//! With hints on, the model now and then adds a one- or two-sentence
//! `> **Hint:**` aside explaining the concept behind the work: a design
//! choice, a failure mechanism, a trade-off, a way to verify. The switch is
//! cache-friendly by construction: nothing already in the conversation is
//! rewritten. The full instruction joins the transcript once per context, a
//! later change of mind is a one-line note, and both land only at a turn
//! boundary ([`HintsState::note`]). Compaction may summarize the instruction
//! away, so it re-arms the state ([`HintsState::compacted`]).
//!
//! The wish itself ([`set_wanted`]) is the user's, for this run only: new and
//! resumed sessions start with hints off and nothing is saved, as in the C.

#[cfg(not(test))]
use std::sync::atomic::{AtomicBool, Ordering};

/// The full instruction, sent once per live context (verbatim from
/// `ds4_agent.c`, `agent_hints_prompt`).
pub const PROMPT: &str = "The user enabled programming hints. Occasionally explain a useful concept \
behind the current work: a design choice, failure mechanism, trade-off, \
or way to verify correctness. Base each hint on code or results you have \
actually examined. Write one or two sentences as a normal Markdown \
blockquote starting with > **Hint:**. Hints are prose, not tool calls. \
Put them after the relevant discovery, including between tool calls. \
Prefer no hint to routine narration, repetition, or generic advice. \
Do not invent personal experience or a learner profile. Keep working \
without waiting for an answer. Later system notes may enable or disable \
hints; follow the latest setting.\n";

/// The note for turning hints back on once the instruction is in context.
pub const ON_NOTE: &str = "The user enabled programming hints. Produce them from now on.\n";

/// The note for turning them off.
pub const OFF_NOTE: &str = "Programming hints are disabled. Do not produce teaching asides; \
continue the task normally.\n";

/// Whether the user wants hints, as last set by `/hints`. Process-wide, since
/// the TUI records a mid-turn `/hints` on its own thread for the worker.
#[cfg(not(test))]
static WANTED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
thread_local! {
    /// Per test thread instead, so one test turning hints on cannot leak a
    /// note into another test's turn running beside it.
    static WANTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Records the user's `/hints` choice; it reaches the model at the next turn
/// boundary.
pub fn set_wanted(on: bool) {
    #[cfg(not(test))]
    WANTED.store(on, Ordering::Relaxed);
    #[cfg(test)]
    WANTED.with(|w| w.set(on));
}

/// The user's current `/hints` choice.
#[must_use]
pub fn wanted() -> bool {
    #[cfg(not(test))]
    return WANTED.load(Ordering::Relaxed);
    #[cfg(test)]
    WANTED.with(std::cell::Cell::get)
}

/// Parses a `/hints` argument: exactly `on` or `off`.
#[must_use]
pub fn parse(arg: &str) -> Option<bool> {
    match arg {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    }
}

/// What the conversation last told the model about hints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Applied {
    /// Nothing: a fresh session, where silence already means off.
    #[default]
    Unset,
    Off,
    On,
    /// Unknowable: a resumed session, or a compaction whose retained tail may
    /// still hold an old note. The next boundary states the setting outright.
    Unknown,
}

/// The per-context half of the switch: whether the full instruction is in the
/// conversation, and which setting the conversation last stated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HintsState {
    instructed: bool,
    applied: Applied,
}

impl HintsState {
    /// The state for a session loaded from disk: whatever it said about hints
    /// is unknown, so the first boundary says "off" (or instructs, if the
    /// user has since turned them on) rather than trusting an old note.
    #[must_use]
    pub fn resumed() -> Self {
        Self {
            instructed: false,
            applied: Applied::Unknown,
        }
    }

    /// The note to append at a turn boundary for the wish `enabled`, if any
    /// (`agent_hints_note`). The full [`PROMPT`] the first time hints are on
    /// in this context, a one-line [`ON_NOTE`] / [`OFF_NOTE`] when the wish
    /// differs from what the conversation last said, else nothing. Several
    /// changes between boundaries coalesce to the latest.
    pub fn note(&mut self, enabled: bool) -> Option<&'static str> {
        let wanted = if enabled { Applied::On } else { Applied::Off };
        let note = if enabled && !self.instructed {
            self.instructed = true;
            PROMPT
        } else if self.applied != Applied::Unset && self.applied != wanted {
            if enabled { ON_NOTE } else { OFF_NOTE }
        } else {
            return None;
        };
        self.applied = wanted;
        Some(note)
    }

    /// After a compaction: the instruction may have been summarized away and
    /// the retained tail may still carry an old note (`agent_hints_compacted`).
    pub fn compacted(&mut self) {
        if self.applied != Applied::Unset {
            self.applied = Applied::Unknown;
        }
        self.instructed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::{HintsState, OFF_NOTE, ON_NOTE, PROMPT, parse};

    /// A port of the C's `test_agent_hints_state`.
    #[test]
    fn notes_follow_the_c_state_machine() {
        assert_eq!(parse("on"), Some(true));
        assert_eq!(parse("off"), Some(false));
        assert_eq!(parse(""), None);
        assert_eq!(parse("toggle"), None);
        assert_eq!(parse("on off"), None);

        let mut h = HintsState::default();
        assert_eq!(h.note(false), None);
        assert_eq!(h.note(true), Some(PROMPT));
        assert_eq!(h.note(true), None);
        assert_eq!(h.note(false), Some(OFF_NOTE));
        assert_eq!(h.note(false), None);
        assert_eq!(h.note(true), Some(ON_NOTE), "instructed once per context");

        h.compacted();
        assert_eq!(h.note(true), Some(PROMPT), "re-instructed after compaction");
        h.compacted();
        assert_eq!(h.note(false), Some(OFF_NOTE));
        h.compacted();
        assert_eq!(h.note(false), Some(OFF_NOTE), "an old on note may survive");

        // A resumed session overrides old on notes without restoring a setting.
        let mut h = HintsState::resumed();
        assert_eq!(h.note(false), Some(OFF_NOTE));
        assert_eq!(h.note(true), Some(PROMPT));

        // Changes before a boundary coalesce to the latest wish.
        let mut h = HintsState::default();
        assert_eq!(h.note(false), None);
        h.compacted();
        assert_eq!(h.note(false), None);
    }
}
