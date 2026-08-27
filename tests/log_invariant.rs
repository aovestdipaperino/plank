//! The log-everything invariant (M3): anything that reaches a model request
//! must be reconstructible from the session log — either as a transcript
//! entry, or as the separately-fingerprinted system prompt.
//!
//! This test assembles a real request (`ui::render_transcript`) from a session
//! carrying every injection site the audit enumerated, then asserts the
//! rendered request is exactly `[system]` + one block per transcript message.
//! The assertion is structural — on the assembled request, not on a list of
//! call sites — so an unlogged span (text injected into the request but never
//! written to the transcript) fails the equality and is caught.

use std::fmt::Write as _;

use plank::session::{Message, Role, Session};

/// Builds a session carrying every injection site, then asserts the rendered
/// request is reconstructible from the transcript plus the system prompt.
#[test]
fn every_injection_site_is_reconstructible_from_the_transcript() {
    let mut session = Session::new();

    // Session-start context (stable + volatile): git status, AGENTS.md,
    // memory, the roster, and the date line. These are pushed as user
    // messages, exactly as `push_session_context` does.
    let content = plank::context::ContextContent {
        git_content: Some("This is the git status at the start of the conversation.".to_string()),
        agents_md_content: Some("Agent instructions:\n\nDo the work.\n".to_string()),
        agents_content: Some("Configured sub-agents, selectable by passing".to_string()),
        memory_content: Some("Persistent memory (durable notes from past sessions".to_string()),
        date_content: "Today's date is 2026-08-26.".to_string(),
    };
    let stable = content.stable_context();
    if !stable.is_empty() {
        session.push(Message::user(stable));
    }
    let volatile = content.volatile_context();
    if !volatile.is_empty() {
        session.push(Message::user(volatile));
    }

    // Task-list re-injection (post-compaction): the rendered block is pushed
    // into the transcript, never injected at request-assembly time.
    session.tasks.add("do the thing", None);
    if let Some(block) = session.tasks.inject_block(None) {
        session.push(Message::user(block));
    }

    // Compaction re-injection: current contents of recently read files.
    let dir = std::env::temp_dir().join(format!("plank-log-invariant-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let file = dir.join("recent.txt");
    std::fs::write(&file, "current contents").expect("write");
    let reinject = plank::compact::build_reinjection(std::slice::from_ref(&file), 4096, &mut |s| {
        i32::try_from(s.len()).unwrap_or(i32::MAX)
    });
    if let Some(block) = reinject {
        session.push(Message::user(block));
    }

    // Subagent report pushed back to the parent.
    session.push(Message::user(plank::agents::report_message(
        "task", "report",
    )));

    // Ordinary conversation.
    session.push(Message::user("user turn"));
    session.push(Message::assistant("assistant turn"));

    // The system prompt is the separately-fingerprinted exception; it carries
    // the MCP tool advertisements.
    let system = "system prompt with mcp__server__tool adverts";

    let rendered = plank::ui::render_transcript(&session, system);

    // Structural assertion: every span of the request is the system prompt or
    // one transcript message, in order. An unlogged span breaks the equality.
    let mut expected = format!("[system]\n{system}\n");
    for m in &session.transcript {
        let tag = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let _ = write!(expected, "[{tag}]\n{}\n", m.text);
    }
    assert_eq!(
        rendered, expected,
        "every span must be attributable to the transcript or the system prompt"
    );

    // Sanity: the injection sites actually reached the request.
    assert!(rendered.contains("This is the git status"));
    assert!(rendered.contains("Today's date is"));
    assert!(rendered.contains("# Task list"));
    assert!(rendered.contains("Post-compaction context re-injection"));
    assert!(rendered.contains("Subagent report"));
    assert!(rendered.contains("mcp__server__tool"));

    std::fs::remove_dir_all(&dir).ok();
}

/// The system prompt is the separately-fingerprinted exception: it is not a
/// transcript entry, but it is accounted for by `fp1`. The invariant allows it.
#[test]
fn the_system_prompt_is_the_fingerprinted_exception() {
    let mut session = Session::new();
    session.push(Message::user("hello"));
    let system = "fingerprinted system prompt";
    let rendered = plank::ui::render_transcript(&session, system);
    assert!(rendered.starts_with("[system]\nfingerprinted system prompt\n"));
    assert!(rendered.contains("[user]\nhello\n"));
}
