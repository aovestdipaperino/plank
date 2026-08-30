// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! A scrollback view over styled cells, one `Vec<Cell>` per line.

use turbo_vision::core::draw::{Cell, DrawBuffer};
use turbo_vision::core::event::{
    Event, EventType, KB_DOWN, KB_END, KB_HOME, KB_PGDN, KB_PGUP, KB_UP,
};
use turbo_vision::core::geometry::Rect;
use turbo_vision::core::palette::{Attr, TvColor};
use turbo_vision::terminal::Terminal;
use turbo_vision::views::view::{View, write_line_to_terminal};

/// Default scrollback depth.
pub const DEFAULT_MAX_LINES: usize = 10_000;

/// A scrollback of styled lines, with autoscroll that releases when the user
/// scrolls back and re-arms at the bottom.
#[derive(Debug)]
pub struct StreamView {
    bounds: Rect,
    /// Completed lines, oldest first.
    lines: Vec<Vec<Cell>>,
    /// The line currently streaming in, not yet terminated by a newline.
    partial: Option<Vec<Cell>>,
    max_lines: usize,
    /// Index of the topmost displayed line.
    top: usize,
    /// Horizontal scroll offset, in cells.
    left: usize,
    /// True while the view follows the tail.
    follow: bool,
    fill: Attr,
}

impl StreamView {
    #[must_use]
    pub fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            lines: Vec::new(),
            partial: None,
            max_lines: DEFAULT_MAX_LINES,
            top: 0,
            left: 0,
            follow: true,
            fill: Attr::new(TvColor::LightGray, TvColor::Black),
        }
    }

    pub fn set_max_lines(&mut self, n: usize) {
        self.max_lines = n.max(1);
        self.trim();
    }

    /// Appends a completed line.
    pub fn push_line(&mut self, cells: Vec<Cell>) {
        self.lines.push(cells);
        self.trim();
        if self.follow {
            self.scroll_to_bottom();
        }
    }

    /// Replaces the in-progress line. Called on every repaint while a line is
    /// still streaming, so it must overwrite rather than append.
    pub fn set_partial(&mut self, cells: Vec<Cell>) {
        self.partial = if cells.is_empty() { None } else { Some(cells) };
        if self.follow {
            self.scroll_to_bottom();
        }
    }

    pub fn clear(&mut self) {
        self.lines.clear();
        self.partial = None;
        self.top = 0;
        self.left = 0;
        self.follow = true;
    }

    /// Total displayed lines, including the in-progress one.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len() + usize::from(self.partial.is_some())
    }

    /// Visible rows, i.e. the view height.
    fn page(&self) -> usize {
        usize::try_from(self.bounds.height()).unwrap_or(0).max(1)
    }

    fn max_top(&self) -> usize {
        self.line_count().saturating_sub(self.page())
    }

    pub fn scroll_to_bottom(&mut self) {
        self.top = self.max_top();
        self.follow = true;
    }

    pub fn scroll_to_top(&mut self) {
        self.top = 0;
        self.follow = false;
    }

    pub fn scroll_up(&mut self, n: usize) {
        self.top = self.top.saturating_sub(n);
        self.follow = false;
    }

    pub fn scroll_down(&mut self, n: usize) {
        self.top = (self.top + n).min(self.max_top());
        self.follow = self.top == self.max_top();
    }

    #[must_use]
    pub fn is_at_bottom(&self) -> bool {
        self.follow
    }

    /// The whole scrollback with attributes stripped, for File > Save As.
    #[must_use]
    pub fn plain_text(&self) -> String {
        let mut out = String::new();
        for (i, line) in self.iter_lines().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.extend(line.iter().map(|c| c.ch));
        }
        out
    }

    fn iter_lines(&self) -> impl Iterator<Item = &Vec<Cell>> {
        self.lines.iter().chain(self.partial.iter())
    }

    fn trim(&mut self) {
        if self.lines.len() > self.max_lines {
            let drop = self.lines.len() - self.max_lines;
            self.lines.drain(..drop);
            self.top = self.top.saturating_sub(drop);
        }
    }
}

impl View for StreamView {
    fn bounds(&self) -> Rect {
        self.bounds
    }

    fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
        if self.follow {
            self.scroll_to_bottom();
        }
    }

    fn draw(&mut self, terminal: &mut Terminal) {
        let width = usize::try_from(self.bounds.width()).unwrap_or(0);
        let page = self.page();
        let lines: Vec<&Vec<Cell>> = self.iter_lines().skip(self.top).take(page).collect();

        for row in 0..page {
            let mut buf = DrawBuffer::new(width);
            for i in 0..width {
                buf.put_char(i, ' ', self.fill);
            }
            if let Some(line) = lines.get(row) {
                for (i, cell) in line.iter().skip(self.left).take(width).enumerate() {
                    buf.put_char(i, cell.ch, cell.attr);
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let y = self.bounds.a.y + row as i16;
            write_line_to_terminal(terminal, self.bounds.a.x, y, &buf);
        }
    }

    fn handle_event(&mut self, event: &mut Event) {
        if event.what != EventType::Keyboard {
            return;
        }
        let page = self.page();
        match event.key_code {
            KB_UP => self.scroll_up(1),
            KB_DOWN => self.scroll_down(1),
            KB_PGUP => self.scroll_up(page),
            KB_PGDN => self.scroll_down(page),
            KB_HOME => self.scroll_to_top(),
            KB_END => self.scroll_to_bottom(),
            _ => return,
        }
        event.clear();
    }

    fn can_focus(&self) -> bool {
        true
    }

    fn get_palette(&self) -> Option<turbo_vision::core::palette::Palette> {
        // Cells already carry resolved `Attr`s (from `AnsiLineAssembler`), so
        // there is no logical-color index for a palette to remap.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_vision::core::palette::TvColor;

    fn line(s: &str) -> Vec<Cell> {
        s.chars()
            .map(|c| Cell::new(c, Attr::new(TvColor::LightGray, TvColor::Black)))
            .collect()
    }

    fn view() -> StreamView {
        StreamView::new(Rect::new(0, 0, 40, 10))
    }

    #[test]
    fn scrollback_cap_drops_oldest_lines() {
        let mut v = view();
        v.set_max_lines(3);
        for i in 0..5 {
            v.push_line(line(&i.to_string()));
        }
        assert_eq!(v.line_count(), 3);
        assert_eq!(v.plain_text(), "2\n3\n4");
    }

    #[test]
    fn autoscroll_holds_at_bottom_while_lines_arrive() {
        let mut v = view();
        for i in 0..50 {
            v.push_line(line(&i.to_string()));
        }
        assert!(v.is_at_bottom());
    }

    #[test]
    fn scrolling_up_releases_autoscroll_and_end_rearms_it() {
        let mut v = view();
        for i in 0..50 {
            v.push_line(line(&i.to_string()));
        }
        v.scroll_up(5);
        assert!(!v.is_at_bottom());
        v.push_line(line("new"));
        assert!(
            !v.is_at_bottom(),
            "a new line must not yank a scrolled-back reader to the bottom"
        );
        v.scroll_to_bottom();
        assert!(v.is_at_bottom());
    }

    #[test]
    fn partial_line_is_replaced_not_appended() {
        let mut v = view();
        v.set_partial(line("par"));
        v.set_partial(line("part"));
        assert_eq!(v.plain_text(), "part");
        assert_eq!(v.line_count(), 1);
    }

    #[test]
    fn plain_text_strips_attributes() {
        let mut v = view();
        v.push_line(vec![Cell::new(
            'x',
            Attr::new(TvColor::LightRed, TvColor::Blue),
        )]);
        assert_eq!(v.plain_text(), "x");
    }
}
