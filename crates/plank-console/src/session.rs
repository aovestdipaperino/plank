// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! One window per stream session.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use plank_stream::render::RenderOptions;
use turbo_vision::core::event::Event;
use turbo_vision::core::geometry::Rect;
use turbo_vision::terminal::Terminal;
use turbo_vision::views::view::View;

use crate::pipeline::Pipeline;
use crate::registry::SessionId;
use crate::streamview::StreamView;

/// A `StreamView` addressable from both the desktop and the event pump.
pub type SharedView = Rc<RefCell<StreamView>>;

/// Forwards `View` calls into a shared `StreamView`.
#[derive(Debug)]
pub struct SharedStreamView(pub SharedView);

impl View for SharedStreamView {
    fn bounds(&self) -> Rect {
        self.0.borrow().bounds()
    }
    fn set_bounds(&mut self, bounds: Rect) {
        self.0.borrow_mut().set_bounds(bounds);
    }
    fn draw(&mut self, terminal: &mut Terminal) {
        self.0.borrow_mut().draw(terminal);
    }
    fn handle_event(&mut self, event: &mut Event) {
        self.0.borrow_mut().handle_event(event);
    }
    fn can_focus(&self) -> bool {
        true
    }
    fn get_palette(&self) -> Option<turbo_vision::core::palette::Palette> {
        None
    }
}

/// Per-session state owned by the main loop.
#[derive(Debug)]
pub struct SessionState {
    pub name: String,
    pub port: u16,
    pub view: SharedView,
    pub pipeline: Pipeline,
    pub connected: bool,
}

impl SessionState {
    /// Title text for this session's window.
    #[must_use]
    pub fn window_title(&self) -> String {
        if self.connected {
            format!("{} :{}", self.name, self.port)
        } else {
            format!("{} :{} [disconnected]", self.name, self.port)
        }
    }
}

/// All live sessions, keyed by id.
#[derive(Debug, Default)]
pub struct Sessions {
    inner: HashMap<SessionId, SessionState>,
}

impl Sessions {
    pub fn insert(
        &mut self,
        id: SessionId,
        name: String,
        port: u16,
        view: SharedView,
        opts: RenderOptions,
    ) {
        self.inner.insert(
            id,
            SessionState {
                name,
                port,
                view,
                pipeline: Pipeline::new(opts),
                connected: false,
            },
        );
    }

    pub fn get_mut(&mut self, id: SessionId) -> Option<&mut SessionState> {
        self.inner.get_mut(&id)
    }

    pub fn remove(&mut self, id: SessionId) -> Option<SessionState> {
        self.inner.remove(&id)
    }

    /// Feeds bytes into a session's pipeline and view.
    pub fn feed(&mut self, id: SessionId, data: &[u8]) {
        if let Some(s) = self.inner.get_mut(&id) {
            let mut view = s.view.borrow_mut();
            s.pipeline.feed(data, &mut view);
        }
    }

    /// Draws a horizontal rule announcing a reattached client.
    pub fn mark_reconnected(&mut self, id: SessionId) {
        if let Some(s) = self.inner.get_mut(&id) {
            s.connected = true;
            let mut view = s.view.borrow_mut();
            s.pipeline.feed(b"\n-- reconnected --\n", &mut view);
        }
    }

    pub fn mark_disconnected(&mut self, id: SessionId) {
        if let Some(s) = self.inner.get_mut(&id) {
            s.connected = false;
            let mut view = s.view.borrow_mut();
            s.pipeline.finish(&mut view);
        }
    }

    /// Applies new render options to every session. Text already on screen
    /// keeps the styling it was rendered with; only new bytes change.
    pub fn set_options(&mut self, opts: RenderOptions) {
        for s in self.inner.values_mut() {
            let mut view = s.view.borrow_mut();
            s.pipeline.set_options(opts, &mut view);
        }
    }

    /// Empties one session's scrollback.
    pub fn clear(&mut self, id: SessionId) {
        if let Some(s) = self.inner.get_mut(&id) {
            s.view.borrow_mut().clear();
        }
    }

    /// One session's scrollback as plain text, for File > Save As.
    #[must_use]
    pub fn plain_text(&self, id: SessionId) -> Option<String> {
        self.inner.get(&id).map(|s| s.view.borrow().plain_text())
    }

    /// The title a session's window should currently show, for reflecting
    /// connect/disconnect state after the fact (the window itself is not
    /// reachable from here — the caller owns the desktop).
    #[must_use]
    pub fn window_title(&self, id: SessionId) -> Option<String> {
        self.inner.get(&id).map(SessionState::window_title)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plank_stream::render::RenderOptions;

    fn opts() -> RenderOptions {
        RenderOptions {
            use_color: true,
            format_thinking: true,
            format_markdown: true,
        }
    }

    fn view() -> SharedView {
        Rc::new(RefCell::new(StreamView::new(Rect::new(0, 0, 80, 24))))
    }

    #[test]
    fn feed_reaches_the_session_view() {
        let mut sessions = Sessions::default();
        sessions.insert(1, "demo".into(), 4242, view(), opts());
        sessions.feed(1, b"hello\n");
        assert!(sessions.plain_text(1).unwrap().contains("hello"));
    }

    #[test]
    fn window_title_reflects_connection_state() {
        let mut sessions = Sessions::default();
        sessions.insert(1, "demo".into(), 4242, view(), opts());
        assert_eq!(
            sessions.window_title(1).unwrap(),
            "demo :4242 [disconnected]"
        );
        sessions.mark_reconnected(1);
        assert_eq!(sessions.window_title(1).unwrap(), "demo :4242");
        sessions.mark_disconnected(1);
        assert_eq!(
            sessions.window_title(1).unwrap(),
            "demo :4242 [disconnected]"
        );
    }

    #[test]
    fn mark_reconnected_draws_a_horizontal_rule() {
        let mut sessions = Sessions::default();
        sessions.insert(1, "demo".into(), 4242, view(), opts());
        sessions.mark_reconnected(1);
        assert!(sessions.plain_text(1).unwrap().contains("reconnected"));
    }

    #[test]
    fn clear_empties_the_scrollback() {
        let mut sessions = Sessions::default();
        sessions.insert(1, "demo".into(), 4242, view(), opts());
        sessions.feed(1, b"hello\n");
        sessions.clear(1);
        assert_eq!(sessions.plain_text(1).unwrap(), "");
    }

    #[test]
    fn set_options_applies_to_every_session() {
        let mut sessions = Sessions::default();
        sessions.insert(1, "demo".into(), 4242, view(), opts());
        let mut new_opts = opts();
        new_opts.format_markdown = false;
        sessions.set_options(new_opts);
        assert!(
            !sessions
                .get_mut(1)
                .unwrap()
                .pipeline
                .options()
                .format_markdown
        );
    }

    #[test]
    fn remove_drops_the_session() {
        let mut sessions = Sessions::default();
        sessions.insert(1, "demo".into(), 4242, view(), opts());
        assert!(sessions.remove(1).is_some());
        assert!(sessions.plain_text(1).is_none());
    }

    #[test]
    fn unknown_id_returns_none_everywhere() {
        let sessions = Sessions::default();
        assert!(sessions.plain_text(99).is_none());
        assert!(sessions.window_title(99).is_none());
    }
}
