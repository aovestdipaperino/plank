// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Terminal window title, kept in sync with what plank is doing.
//!
//! A handful of states, so the window (and tab) names plank's phase at a
//! glance: `🚀 Plank loading...` before a front end is up, `🪵 Plank - READY.`
//! while idle at the prompt, `🚀 <prompt>` while a turn runs (the rocket
//! cycles through a few sparkling glyphs, one step per [`tick`], unless
//! reduced motion is on),
//! `❓ waiting for you...` while the `ask` tool holds the turn open for an
//! answer, and
//! `👀 introspecting...` while `/insights` reads back the user's own history. Set via the OSC 0
//! escape (`ESC ] 0 ; title BEL`), written to **stderr** in a single write:
//! stderr reaches the same tty as stdout but bypasses the Ratatui frame
//! buffer, so a title change can never tear a frame even when emitted from the
//! worker thread. No-op when stderr is not a terminal (piped runs, tests,
//! `--non-interactive` under a harness).

use std::io::{IsTerminal, Write};

/// What plank is doing, as reflected in the window title.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State<'a> {
    /// Starting up — no front end is accepting input yet.
    Loading,
    /// Sitting at the prompt, waiting for the user.
    Idle,
    /// Running a turn for the given user prompt.
    Busy(&'a str),
    /// Building the `/insights` report. Its own state rather than a
    /// [`State::Busy`] prompt because it is not a turn — no user prompt is
    /// being answered — and it runs long enough that the window should say
    /// what it is doing.
    Introspecting,
    /// Blocked on the user inside the `ask` tool. Its own state for the same
    /// reason as [`State::Compacting`]: it interrupts a running turn, and it is
    /// the one phase where a backgrounded window should say the turn is not
    /// stalled but waiting on *you*. Always set through [`Scoped`], so whatever
    /// the turn was showing — normally the [`State::Busy`] rocket — comes back
    /// however the question ends, including a declined or interrupted one.
    Asking,
    /// Summarizing the transcript to reclaim context. Like
    /// [`State::Introspecting`], its own state: it interrupts whatever the user
    /// asked for, takes long enough to be worth naming, and is the one phase
    /// where a background window should say "not your turn yet".
    Compacting,
}

/// Longest prompt (in characters) kept in a [`State::Busy`] title before it is
/// truncated with an ellipsis.
const TITLE_PROMPT_MAX: usize = 20;

/// Title shown while starting up — including the KV-cache prefill, which is the
/// slowest launch step and the one most likely to be looked at.
const LOADING: &str = "🚀 Plank loading...";

/// Title shown while `/insights` is reading back the user's own history.
const INTROSPECTING: &str = "👀 introspecting...";

/// Title shown while a compaction pass is summarizing the transcript.
const COMPACTING: &str = "🗑️ compacting...";

/// Title shown while the `ask` tool is waiting on the user's choice.
const ASKING: &str = "❓ waiting for you...";

/// Formats the window title for `state`. A [`State::Busy`] prompt is collapsed
/// to one line and truncated past [`TITLE_PROMPT_MAX`] characters; a
/// whitespace-only prompt degrades to the plain loading form.
#[must_use]
pub fn window_title(state: State<'_>) -> String {
    match state {
        State::Loading => LOADING.to_string(),
        State::Idle => "🪵 Plank - READY.".to_string(),
        State::Introspecting => INTROSPECTING.to_string(),
        State::Compacting => COMPACTING.to_string(),
        State::Asking => ASKING.to_string(),
        State::Busy(p) => match collapse_prompt(p) {
            Some(prompt) => busy_title(&prompt, 0),
            None => LOADING.to_string(),
        },
    }
}

/// Collapses a busy prompt to one whitespace-normalized line; `None` when
/// nothing is left, so the caller can fall back to the loading title.
fn collapse_prompt(prompt: &str) -> Option<String> {
    let collapsed = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    (!collapsed.is_empty()).then_some(collapsed)
}

/// The busy rocket's animation frames, cycled in place: the plain rocket, then
/// a sparkle, a dizzy star and a glowing star trailing it. Frame `0` is the
/// bare rocket and doubles as the static form (reduced motion); it is padded
/// to the width of the two-glyph frames so the rocket does not shift as the
/// trail appears.
const ROCKET_GLYPHS: [&str; 4] = ["  🚀", "✨🚀", "💫🚀", "🌟🚀"];

/// Frames in one cycle of the rocket glyphs.
const ROCKET_FRAMES: usize = ROCKET_GLYPHS.len();

/// Formats the [`State::Busy`] title for animation frame `frame`: that frame's
/// rocket glyph from [`ROCKET_GLYPHS`], then the already-collapsed `prompt`,
/// truncated past [`TITLE_PROMPT_MAX`] characters.
fn busy_title(prompt: &str, frame: usize) -> String {
    let glyph = ROCKET_GLYPHS[frame % ROCKET_FRAMES];
    match prompt.char_indices().nth(TITLE_PROMPT_MAX) {
        Some((i, _)) => format!("{glyph} {}…", prompt[..i].trim_end()),
        None => format!("{glyph} {prompt}"),
    }
}

/// The running busy animation: the collapsed prompt and the frame last shown.
/// `Some` only while the title is a [`State::Busy`] rocket; any other state
/// clears it, and [`Scoped`] parks and restores it with the title it displaces.
static BUSY: std::sync::Mutex<Option<(String, usize)>> = std::sync::Mutex::new(None);

fn busy_lock() -> std::sync::MutexGuard<'static, Option<(String, usize)>> {
    BUSY.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Advances the busy rocket one frame, if a [`State::Busy`] title is showing.
/// Called from the TUI's redraw loop; a no-op at any other title, and under
/// reduced motion (`ui.reducedMotion`), where the rocket stays at frame 0.
pub fn tick() {
    if crate::anim::reduced_motion() {
        return;
    }
    let next = {
        let mut busy = busy_lock();
        let Some((prompt, frame)) = busy.as_mut() else {
            return;
        };
        *frame = (*frame + 1) % ROCKET_FRAMES;
        busy_title(prompt, *frame)
    };
    set_text(&next);
}

/// The title last written, so a transient state ([`Scoped`]) can put back what
/// it displaced instead of every caller having to know what came before.
static LAST: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Sets the terminal window title to [`window_title`]`(state)`. Best-effort:
/// errors are ignored, and nothing is written when stderr is not a tty.
pub fn set(state: State<'_>) {
    // Record the prompt for `tick` only when the title is actually the busy
    // rocket — a blank prompt degrades to the loading form and must not animate.
    *busy_lock() = match state {
        State::Busy(p) => collapse_prompt(p).map(|prompt| (prompt, 0)),
        _ => None,
    };
    set_text(&window_title(state));
}

/// Writes an already-formatted title, recording it as the current one.
fn set_text(title: &str) {
    *LAST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(title.to_owned());
    let mut err = std::io::stderr();
    if !err.is_terminal() {
        return;
    }
    // OSC 0 (icon + window title), BEL-terminated — the most widely supported
    // form. One write so it cannot interleave with other stderr output.
    let seq = format!("\x1b]0;{title}\x07");
    let _ = err.write_all(seq.as_bytes());
    let _ = err.flush();
}

/// Shows a title for as long as the guard lives, then restores the one it
/// displaced.
///
/// For phases that interrupt something else and must hand the window back
/// afterwards. Compaction is the case in point: it runs both mid-turn (title
/// was [`State::Busy`]) and from `/compact` at the prompt (title was
/// [`State::Idle`]), so the phase itself cannot know what to restore — and
/// restoring on drop covers the interrupted and failed passes too.
#[derive(Debug)]
pub struct Scoped {
    title: Option<String>,
    /// The busy animation that was running, parked while the guard lives so
    /// `tick` does not fly the rocket over the displaced title.
    busy: Option<(String, usize)>,
}

impl Scoped {
    /// Displaces the current title with `state`'s.
    #[must_use]
    pub fn set(state: State<'_>) -> Self {
        let title = LAST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let busy = busy_lock().take();
        crate::title::set(state);
        Self { title, busy }
    }
}

impl Drop for Scoped {
    fn drop(&mut self) {
        *busy_lock() = self.busy.take();
        if let Some(previous) = self.title.take() {
            set_text(&previous);
        }
    }
}

/// Serializes every test that asserts on, or writes, the process-global
/// `LAST`. It lives at module scope rather than inside `mod tests` because the
/// compaction tests in [`crate::ui`] drive `Scoped` through the real
/// `do_compact_notify`, so they write this global too and must serialize
/// against the title tests — otherwise a compaction test landing between a
/// title test's `set_text` and its read flakes it.
#[cfg(test)]
pub(crate) static TITLE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_and_idle_are_fixed_strings() {
        assert_eq!(window_title(State::Loading), "🚀 Plank loading...");
        assert_eq!(window_title(State::Idle), "🪵 Plank - READY.");
        assert_eq!(window_title(State::Introspecting), "👀 introspecting...");
        assert_eq!(window_title(State::Compacting), "🗑️ compacting...");
        assert_eq!(window_title(State::Asking), "❓ waiting for you...");
    }

    /// The `ask` tool's contract with the window title: the question mark is up
    /// only while the user is being asked, and the rocket the turn was flying
    /// comes back afterwards — whichever way the question ended.
    #[test]
    fn asking_displaces_the_busy_rocket_and_gives_it_back() {
        let _serial = TITLE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let busy = window_title(State::Busy("port the arcade"));
        set_text(&busy);
        let guard = Scoped::set(State::Asking);
        let during = LAST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(during.as_deref(), Some("❓ waiting for you..."));
        drop(guard);
        let after = LAST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(after.as_deref(), Some(busy.as_str()));
    }

    /// Serialized against itself, and the only test that reads `LAST`'s exact
    /// contents: the compaction tests in [`crate::ui`] also write the global (via
    /// [`Scoped`]), so this takes `TITLE_TEST_LOCK` and keeps its critical
    /// section to two adjacent calls.
    #[test]
    fn scoped_captures_and_restores_the_title_it_displaced() {
        let _serial = TITLE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set_text("sentinel-title");
        let guard = Scoped::set(State::Compacting);
        assert_eq!(
            guard.title.as_deref(),
            Some("sentinel-title"),
            "the guard must capture the title it displaced"
        );
        drop(guard);
        let restored = LAST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(restored.as_deref(), Some("sentinel-title"));
    }

    #[test]
    fn blank_busy_prompt_falls_back_to_loading() {
        assert_eq!(window_title(State::Busy("   ")), "🚀 Plank loading...");
        assert_eq!(window_title(State::Busy("")), "🚀 Plank loading...");
    }

    #[test]
    fn busy_prompt_is_collapsed_and_truncated() {
        assert_eq!(
            window_title(State::Busy("fix  the\nbug")),
            "  🚀 fix the bug"
        );
        let long = "a".repeat(60);
        let t = window_title(State::Busy(&long));
        assert!(t.starts_with("  🚀 "));
        assert!(t.ends_with('…'));
        assert_eq!(
            t.chars().count(),
            "  🚀 ".chars().count() + TITLE_PROMPT_MAX + 1
        );
    }

    /// The glyphs cycle in place and wrap back to the bare rocket; a truncated
    /// prompt is cut the same way on every frame.
    #[test]
    fn rocket_glyphs_cycle_and_wrap() {
        let frames: Vec<String> = (0..=ROCKET_FRAMES).map(|f| busy_title("go", f)).collect();
        assert_eq!(
            frames,
            ["  🚀 go", "✨🚀 go", "💫🚀 go", "🌟🚀 go", "  🚀 go"]
        );
        let long = "b".repeat(40);
        assert!(busy_title(&long, 3).ends_with(&busy_title(&long, 0)[6..]));
    }

    /// `tick` advances only a busy title, is parked by a `Scoped` displacement
    /// and resumes where it left off, and stops once the title leaves Busy.
    #[test]
    fn tick_flies_the_rocket_only_while_busy() {
        let _serial = TITLE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let was_reduced = crate::anim::reduced_motion();
        crate::anim::set_reduced_motion(false);
        let last = || {
            LAST.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        };
        set(State::Busy("go"));
        tick();
        assert_eq!(last().as_deref(), Some("✨🚀 go"));
        let guard = Scoped::set(State::Compacting);
        tick();
        assert_eq!(
            last().as_deref(),
            Some(COMPACTING),
            "parked while displaced"
        );
        drop(guard);
        assert_eq!(
            last().as_deref(),
            Some("✨🚀 go"),
            "displaced frame restored"
        );
        tick();
        assert_eq!(
            last().as_deref(),
            Some("💫🚀 go"),
            "resumes from where it was"
        );
        set(State::Idle);
        tick();
        assert_eq!(last().as_deref(), Some("🪵 Plank - READY."));
        crate::anim::set_reduced_motion(true);
        set(State::Busy("go"));
        tick();
        assert_eq!(
            last().as_deref(),
            Some("  🚀 go"),
            "still under reduced motion"
        );
        crate::anim::set_reduced_motion(was_reduced);
    }
}
