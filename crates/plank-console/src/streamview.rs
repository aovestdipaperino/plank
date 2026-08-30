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

    /// The whole scrollback with attributes intact, for tests and golden files.
    #[must_use]
    pub fn styled_lines(&self) -> Vec<Vec<Cell>> {
        self.iter_lines().cloned().collect()
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
        } else {
            self.top = self.top.min(self.max_top());
        }
    }

    fn draw(&mut self, terminal: &mut Terminal) {
        if self.bounds.height() <= 0 {
            return;
        }
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
            let y = self.bounds.a.y + i16::try_from(row).unwrap_or(i16::MAX);
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
    use std::io;
    use std::time::Duration;
    use turbo_vision::core::palette::TvColor;
    use turbo_vision::terminal::Backend;

    fn line(s: &str) -> Vec<Cell> {
        s.chars()
            .map(|c| Cell::new(c, Attr::new(TvColor::LightGray, TvColor::Black)))
            .collect()
    }

    fn view() -> StreamView {
        StreamView::new(Rect::new(0, 0, 40, 10))
    }

    /// An in-memory `Backend` for tests: no real TTY, fixed size, no I/O.
    /// `Terminal::write_line`/`write_cell` write straight into `Terminal`'s
    /// own in-memory buffer, so this stub only needs to satisfy
    /// initialization and size queries for `Terminal::with_backend`.
    struct FakeBackend {
        width: u16,
        height: u16,
    }

    impl Backend for FakeBackend {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn init(&mut self) -> io::Result<()> {
            Ok(())
        }

        fn cleanup(&mut self) -> io::Result<()> {
            Ok(())
        }

        fn size(&self) -> io::Result<(u16, u16)> {
            Ok((self.width, self.height))
        }

        fn poll_event(&mut self, _timeout: Duration) -> io::Result<Option<Event>> {
            Ok(None)
        }

        fn write_raw(&mut self, _data: &[u8]) -> io::Result<()> {
            Ok(())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }

        fn show_cursor(&mut self, _x: u16, _y: u16) -> io::Result<()> {
            Ok(())
        }

        fn hide_cursor(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn fake_terminal(width: u16, height: u16) -> Terminal {
        Terminal::with_backend(Box::new(FakeBackend { width, height }))
            .expect("fake backend never fails to init")
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

    #[test]
    fn resize_larger_while_scrolled_back_reclamps_top_to_show_a_full_page() {
        let mut v = StreamView::new(Rect::new(0, 0, 40, 5));
        for i in 0..50 {
            v.push_line(line(&i.to_string()));
        }
        // Scroll back so `top` sits well below the current max_top()
        // (line_count 50, page 5 -> max_top 45).
        v.scroll_to_top();
        v.scroll_down(40);
        assert!(!v.is_at_bottom());
        let old_top = v.top;
        assert!(old_top < v.max_top());

        // Grow the view a lot: max_top() shrinks to line_count - new_page
        // (50 - 48 = 2), which is now well below the old `top` (40). Left
        // unclamped, that would leave blank rows at the bottom of the
        // viewport even though unshown history sits above.
        v.set_bounds(Rect::new(0, 0, 40, 48));

        assert!(
            v.top <= v.max_top(),
            "top ({}) must not exceed max_top ({}) after growing",
            v.top,
            v.max_top()
        );
        let lines: Vec<&Vec<Cell>> = v.iter_lines().skip(v.top).take(v.page()).collect();
        assert_eq!(
            lines.len(),
            v.page().min(v.line_count()),
            "a full page of content should be visible after growing"
        );
    }

    #[test]
    fn draw_clips_to_bounds_and_applies_horizontal_offset() {
        let mut v = StreamView::new(Rect::new(2, 1, 8, 4));
        v.push_line(line("abcdefghij")); // longer than the 6-wide view
        v.push_line(line("short")); // shorter than the 6-wide view
        v.left = 2; // horizontal scroll offset

        let mut terminal = fake_terminal(20, 10);
        v.draw(&mut terminal);

        // Row 0 (bounds.a.y == 1): "abcdefghij" skipped by `left` 2, then
        // clipped to width 6, drawn starting at bounds.a.x == 2.
        for (i, expected) in "cdefgh".chars().enumerate() {
            let cell = terminal
                .read_cell(2 + i16::try_from(i).unwrap_or(i16::MAX), 1)
                .expect("cell within terminal bounds");
            assert_eq!(cell.ch, expected);
        }
        // Nothing is drawn past the view's width (x == 8 is out of bounds).
        assert_eq!(terminal.read_cell(8, 1).unwrap().ch, ' ');

        // Row 1 (bounds.a.y == 2): "short" skipped by `left` 2 -> "ort",
        // padded with the fill space for the remaining 3 columns.
        for (i, expected) in "ort   ".chars().enumerate() {
            let cell = terminal
                .read_cell(2 + i16::try_from(i).unwrap_or(i16::MAX), 2)
                .expect("cell within terminal bounds");
            assert_eq!(cell.ch, expected);
        }

        // Nothing above or below the view's rows was touched.
        assert_eq!(terminal.read_cell(2, 0).unwrap().ch, ' ');
    }

    #[test]
    fn draw_on_zero_height_view_writes_nothing() {
        let mut v = StreamView::new(Rect::new(0, 0, 10, 0));
        v.push_line(line("hello"));
        let mut terminal = fake_terminal(20, 10);
        v.draw(&mut terminal);
        for y in 0..10 {
            for x in 0..20 {
                assert_eq!(
                    terminal.read_cell(x, y).unwrap().ch,
                    ' ',
                    "zero-height view must not write any cell"
                );
            }
        }
    }
}
