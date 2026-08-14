// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The event bus WASM components observe (`docs/WASM-PLUGINS.md`, "Events").
//!
//! A component subscribes in its manifest and plank calls it only for what it
//! subscribed to, so an unsubscribed event costs nothing — no call, no
//! serialization, no fuel.
//!
//! ## One export, not one per event
//!
//! The design sketches a handler per event. This implements a single
//! `on_event` export that receives the event name alongside its payload, for
//! two reasons. Extism cannot type-check an export signature, so N exports buy
//! no safety over one — the ABI handshake is what catches drift either way.
//! And a new event would otherwise mean every component that wants it grows an
//! export, where the payload-evolution rule says additions must not disturb
//! existing components. The cost is that a guest writes its own `match` on the
//! event name; the Rust PDK makes that four lines.
//!
//! ## Classes
//!
//! An event's *class* is what its return value means, and it is a property of
//! the event, not of the subscriber:
//!
//! - **Notify** — the reply is ignored. The component is being told.
//! - **Veto** — the component may block the action, and the reason goes to
//!   whoever the action belonged to (the model, for tool events).
//! - **Transform** — the component may replace the payload, and replacements
//!   chain through subscribers in load order, so the second subscriber sees
//!   what the first produced.
//!
//! A component that answers outside its event's class is ignored rather than
//! corrected: a `block` on a notify event is not a reason to fail a turn.

use crate::tools::mcp::{Json, json_parse};

/// Which event fired.
///
/// Only the events plank actually dispatches are here. The design lists more
/// (`idle`, `resize`, `job_start`, …) and adding one is additive — a new
/// variant, a new manifest spelling, no ABI bump — but an event nothing fires
/// is a promise, not a feature, so the enum stays honest about what exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventKind {
    /// A session began. Notify.
    SessionStart,
    /// The user submitted a prompt. Transform and veto: a component may
    /// rewrite the prompt or refuse it.
    UserPromptSubmit,
    /// A tool is about to run. Veto: blocking returns the reason to the model
    /// as the tool's error.
    PreToolUse,
    /// A tool finished. Transform: the replacement becomes the observation the
    /// model sees, which is how a redactor or a summarizer works.
    PostToolUse,
    /// A turn ended. Notify.
    TurnEnd,
}

/// What a return value means for a given event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// The reply is ignored.
    Notify,
    /// The component may block, with a reason.
    Veto,
    /// The component may block *or* replace the payload.
    Transform,
}

impl EventKind {
    /// The manifest spelling.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "session_start" => Self::SessionStart,
            "user_prompt_submit" => Self::UserPromptSubmit,
            "pre_tool_use" => Self::PreToolUse,
            "post_tool_use" => Self::PostToolUse,
            "turn_end" => Self::TurnEnd,
            _ => return None,
        })
    }

    /// The manifest spelling, and what the guest is handed as `event`.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::SessionStart => "session_start",
            Self::UserPromptSubmit => "user_prompt_submit",
            Self::PreToolUse => "pre_tool_use",
            Self::PostToolUse => "post_tool_use",
            Self::TurnEnd => "turn_end",
        }
    }

    /// What this event does with a reply.
    #[must_use]
    pub fn class(self) -> Class {
        match self {
            Self::SessionStart | Self::TurnEnd => Class::Notify,
            Self::PreToolUse => Class::Veto,
            Self::UserPromptSubmit | Self::PostToolUse => Class::Transform,
        }
    }
}

/// One event and its payload fields, as `(key, value)` pairs.
///
/// Payloads are flat string maps rather than a typed struct per event: the
/// evolution rule is "fields are only ever added", and a flat map makes adding
/// one a one-line change on both sides that no existing subscriber notices.
#[derive(Debug, Clone)]
pub struct Event<'a> {
    /// Which event.
    pub kind: EventKind,
    /// The payload, in a fixed order so the JSON is deterministic.
    pub fields: Vec<(&'a str, String)>,
}

impl<'a> Event<'a> {
    /// Builds an event.
    #[must_use]
    pub fn new(kind: EventKind, fields: Vec<(&'a str, String)>) -> Self {
        Self { kind, fields }
    }

    /// The JSON a guest receives: `{"event": "<name>", <fields>}`.
    #[must_use]
    pub fn to_json(&self) -> String {
        use std::fmt::Write as _;

        let mut out = format!("{{\"event\": {}", json_string(self.kind.label()));
        for (key, value) in &self.fields {
            let _ = write!(out, ", {}: {}", json_string(key), json_string(value));
        }
        out.push('}');
        out
    }

    /// The field a [`Class::Transform`] replacement substitutes.
    ///
    /// Each transform event has exactly one payload field that is *the* thing
    /// being transformed — the prompt, the tool's output — so a replacement is
    /// one string rather than a whole payload the guest has to echo back
    /// unchanged.
    #[must_use]
    pub fn transform_field(kind: EventKind) -> Option<&'static str> {
        match kind {
            EventKind::UserPromptSubmit => Some("prompt"),
            EventKind::PostToolUse => Some("output"),
            _ => None,
        }
    }
}

/// What a subscriber asked for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reply {
    /// Block the action, with this reason.
    pub block: Option<String>,
    /// Replace the transform field with this text.
    pub replace: Option<String>,
}

impl Reply {
    /// Parses a handler's reply. Anything unreadable is an empty reply: a
    /// component that answered nonsense still *ran*, and treating a bad reply
    /// as a veto would let a broken component halt a turn.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Self {
        let mut out = Self::default();
        let Ok(text) = std::str::from_utf8(bytes) else {
            return out;
        };
        let Some(Json::Obj(fields)) = json_parse(text) else {
            return out;
        };
        for (key, value) in fields {
            match (key.as_str(), value) {
                ("block", Json::Str(s)) => out.block = Some(s),
                ("replace", Json::Str(s)) => out.replace = Some(s),
                _ => {}
            }
        }
        out
    }

    /// Drops anything the event's class does not permit, so a component cannot
    /// veto a notify event or rewrite something that is not transformable.
    #[must_use]
    pub fn constrained_to(mut self, class: Class) -> Self {
        match class {
            Class::Notify => {
                self.block = None;
                self.replace = None;
            }
            Class::Veto => self.replace = None,
            Class::Transform => {}
        }
        self
    }

    /// Whether this reply changes anything.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.block.is_none() && self.replace.is_none()
    }
}

/// The result of dispatching one event to every subscriber.
#[derive(Debug, Clone, Default)]
pub struct Outcome {
    /// Set when a subscriber blocked, with the id that did it and its reason.
    /// The first block wins and the chain stops: once an action is refused,
    /// asking the remaining components about it is asking them to weigh in on
    /// something that will not happen.
    pub blocked: Option<(String, String)>,
    /// The transform field's final value, when any subscriber replaced it.
    pub replaced: Option<String>,
    /// Lines components printed while handling the event, already drained.
    pub printed: Vec<String>,
    /// Non-fatal complaints: a trap, a disabled component.
    pub warnings: Vec<String>,
}

impl Outcome {
    /// Whether anything happened worth acting on.
    #[must_use]
    pub fn is_quiet(&self) -> bool {
        self.blocked.is_none()
            && self.replaced.is_none()
            && self.printed.is_empty()
            && self.warnings.is_empty()
    }
}

/// Minimal JSON string escaping for payloads.
fn json_string(s: &str) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_payload_is_json_with_the_event_name_first() {
        let e = Event::new(
            EventKind::PreToolUse,
            vec![
                ("name", "bash".to_string()),
                ("args", "{\"x\": 1}".to_string()),
            ],
        );
        assert_eq!(
            e.to_json(),
            r#"{"event": "pre_tool_use", "name": "bash", "args": "{\"x\": 1}"}"#
        );
    }

    /// A payload carrying a newline or a quote must not produce JSON a guest
    /// cannot parse — prompts and tool output routinely contain both.
    #[test]
    fn payload_strings_are_escaped() {
        let e = Event::new(
            EventKind::UserPromptSubmit,
            vec![("prompt", "line\none\t\"quoted\"".to_string())],
        );
        let json = e.to_json();
        assert!(json.contains(r#""line\none\t\"quoted\"""#), "{json}");
        assert!(json_parse(&json).is_some(), "unparseable: {json}");
    }

    #[test]
    fn a_reply_parses_block_and_replace_and_tolerates_junk() {
        assert_eq!(Reply::parse(b"not json"), Reply::default());
        assert_eq!(Reply::parse(b"{}"), Reply::default());
        let r = Reply::parse(br#"{"block": "nope", "replace": "x", "other": 1}"#);
        assert_eq!(r.block.as_deref(), Some("nope"));
        assert_eq!(r.replace.as_deref(), Some("x"));
        assert!(!r.is_empty());
    }

    /// The class is the contract. A component answering outside it is ignored,
    /// never corrected — a `block` on a notify event must not fail a turn.
    #[test]
    fn a_reply_is_constrained_by_its_events_class() {
        let full = Reply {
            block: Some("no".to_string()),
            replace: Some("x".to_string()),
        };
        let notify = full.clone().constrained_to(Class::Notify);
        assert!(notify.is_empty(), "notify ignores everything");

        let veto = full.clone().constrained_to(Class::Veto);
        assert_eq!(veto.block.as_deref(), Some("no"));
        assert_eq!(veto.replace, None, "a veto event cannot rewrite");

        let transform = full.constrained_to(Class::Transform);
        assert_eq!(transform.block.as_deref(), Some("no"));
        assert_eq!(transform.replace.as_deref(), Some("x"));
    }

    #[test]
    fn every_event_kind_round_trips_its_spelling_and_knows_its_class() {
        for kind in [
            EventKind::SessionStart,
            EventKind::UserPromptSubmit,
            EventKind::PreToolUse,
            EventKind::PostToolUse,
            EventKind::TurnEnd,
        ] {
            assert_eq!(EventKind::parse(kind.label()), Some(kind));
            // Only transform events name a field, and every one that does must.
            assert_eq!(
                Event::transform_field(kind).is_some(),
                kind.class() == Class::Transform,
                "{}",
                kind.label()
            );
        }
        assert_eq!(EventKind::parse("token_batch"), None, "cut from v1");
    }
}
