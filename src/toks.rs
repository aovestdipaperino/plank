// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Enzo Lombardi

//! Generation throughput history for `/toks`.
//!
//! A time series, not a per-pass tally: while the model is decoding, a
//! [`Sampler`] measures the instantaneous rate once every
//! [`Sampler::INTERVAL`] and records it in one process-wide [`TokRing`]
//! ([`record`]/[`snapshot`]), so the x axis is decode time and the chart knows
//! nothing about passes, turns or tool rounds. [`render_chart`] draws the ring
//! as a braille line chart: two samples per column and four levels per row.
//! Text, not a widget, so the same chart serves the plain REPL and the TUI's
//! report panel, and the TUI can redraw it on every status tick.

use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::{Duration, Instant};

static RING: Mutex<TokRing> = Mutex::new(TokRing::new());

/// Records one throughput sample in the process-wide ring.
pub fn record(tps: f64) {
    if let Ok(mut r) = RING.lock() {
        r.push(tps);
    }
}

/// The recorded samples, oldest first.
#[must_use]
pub fn snapshot() -> Vec<f64> {
    RING.lock().map_or_else(|_| Vec::new(), |r| r.samples())
}

/// Measures the decode rate over fixed wall-clock windows from a running
/// token count. Feed it every generated token; it hands back a rate once per
/// [`Sampler::INTERVAL`], which the caller records.
#[derive(Debug, Default)]
pub struct Sampler {
    /// Start of the current window and the token count then.
    mark: Option<(Instant, i32)>,
    /// Tokens counted by [`Sampler::note_token`], for callers without their
    /// own running count.
    count: i32,
}

impl Sampler {
    /// Wall-clock width of one sample: a second, so a 64-cell chart shows
    /// about two minutes of decoding.
    pub const INTERVAL: Duration = Duration::from_secs(1);

    /// Notes that `generated` tokens have been produced so far, at `now`.
    /// Returns the tokens-per-second over the window just closed when one
    /// full interval has elapsed since the last sample.
    pub fn observe(&mut self, now: Instant, generated: i32) -> Option<f64> {
        let Some((at, count_at)) = self.mark else {
            self.mark = Some((now, generated));
            return None;
        };
        let elapsed = now.saturating_duration_since(at);
        if elapsed < Self::INTERVAL {
            return None;
        }
        self.mark = Some((now, generated));
        let tokens = f64::from(generated.saturating_sub(count_at));
        Some(tokens / elapsed.as_secs_f64())
    }

    /// Convenience: [`Sampler::observe`] at the current instant, recording
    /// any sample into the process-wide ring.
    pub fn tick(&mut self, generated: i32) {
        if let Some(tps) = self.observe(Instant::now(), generated) {
            record(tps);
        }
    }

    /// Counts one generated token and [`Sampler::tick`]s with the total.
    pub fn note_token(&mut self) {
        self.count = self.count.saturating_add(1);
        self.tick(self.count);
    }
}

/// Fixed-capacity ring of recent generation rates, oldest first when read.
#[derive(Debug, Clone)]
pub struct TokRing {
    buf: Vec<f64>,
    /// Index the next sample lands in once the ring is full.
    head: usize,
}

impl Default for TokRing {
    fn default() -> Self {
        Self::new()
    }
}

impl TokRing {
    /// Samples kept: with two samples per braille column this fills a
    /// 128-cell chart, wider than any panel plank draws.
    pub const CAPACITY: usize = 256;

    /// An empty ring. `const` so it can back a `static`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buf: Vec::new(),
            head: 0,
        }
    }

    /// Records one sample. Non-finite and non-positive values are dropped: a
    /// window with nothing generated reports zero, which would read as a
    /// stall on the chart.
    pub fn push(&mut self, tps: f64) {
        if !tps.is_finite() || tps <= 0.0 {
            return;
        }
        if self.buf.len() < Self::CAPACITY {
            self.buf.push(tps);
        } else {
            self.buf[self.head] = tps;
            self.head = (self.head + 1) % Self::CAPACITY;
        }
    }

    /// Number of samples held, at most [`Self::CAPACITY`].
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// True when nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The samples in arrival order, oldest first.
    #[must_use]
    pub fn samples(&self) -> Vec<f64> {
        let (newer, older) = self.buf.split_at(self.head);
        older.iter().chain(newer.iter()).copied().collect()
    }
}

/// Braille dot bit for column `x` (0 or 1) and row `y` (0 = top, 3 = bottom)
/// of one cell, per the Unicode braille block layout.
const fn dot(x: usize, y: usize) -> u32 {
    const BITS: [[u32; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
    BITS[x][y]
}

/// Draws `samples` (oldest first) as a braille line chart `width` cells wide
/// and `height` cells tall, each line prefixed by a right-aligned axis label
/// on the top and bottom rows. Only the newest `2 * width` samples fit; older
/// ones fall off the left. Adjacent samples are joined vertically so a jump
/// reads as a line rather than two dots.
#[must_use]
pub fn render_chart(samples: &[f64], width: usize, height: usize) -> String {
    let width = width.max(1);
    let height = height.max(1);
    let cols = width * 2;
    let rows = height * 4;
    let start = samples.len().saturating_sub(cols);
    let shown = &samples[start..];
    let (min, max) = shown
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    // A flat series still needs a span, or every point divides by zero.
    let (lo, span) = if shown.is_empty() {
        (0.0, 1.0)
    } else if (max - min).abs() < f64::EPSILON {
        (min - 1.0, 2.0)
    } else {
        (min, max - min)
    };
    // `rows` is a handful of dot rows, so both casts are exact.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let level = |v: f64| -> usize {
        // 0 is the bottom dot row, rows-1 the top.
        let t = ((v - lo) / span).clamp(0.0, 1.0);
        ((t * (rows - 1) as f64).round() as usize).min(rows - 1)
    };
    let mut grid = vec![vec![0u32; width]; height];
    let mut set = |col: usize, lvl: usize| {
        let y = rows - 1 - lvl;
        grid[y / 4][col / 2] |= dot(col % 2, y % 4);
    };
    let mut prev: Option<usize> = None;
    for (col, &v) in shown.iter().enumerate() {
        let l = level(v);
        // Fill the vertical gap to the previous point so the line is
        // continuous; the run is split between the two columns at midpoint.
        if let Some(p) = prev {
            let (a, b) = if p < l { (p, l) } else { (l, p) };
            let mid = usize::midpoint(a, b);
            for y in a..=b {
                let on_prev = if p < l { y <= mid } else { y >= mid };
                set(if on_prev { col - 1 } else { col }, y);
            }
        }
        set(col, l);
        prev = Some(l);
    }
    let label_w = 7;
    let mut out = String::new();
    for (row, cells) in grid.iter().enumerate() {
        let label = if shown.is_empty() {
            String::new()
        } else if row == 0 {
            format!("{max:.1}")
        } else if row == height - 1 {
            format!("{min:.1}")
        } else {
            String::new()
        };
        let _ = write!(out, "{label:>label_w$} ┤");
        for &bits in cells {
            let ch = char::from_u32(0x2800 + bits).unwrap_or(' ');
            out.push(ch);
        }
        out.push('\n');
    }
    out
}

/// The full `/toks` report over `samples` (oldest first): the chart plus a
/// one-line summary. `width` is the chart width in cells.
#[must_use]
pub fn render_report(samples: &[f64], width: usize, height: usize, color: bool) -> String {
    let dim = if color { "\x1b[2m" } else { "" };
    let bold = if color { "\x1b[1m" } else { "" };
    let reset = if color { "\x1b[0m" } else { "" };
    let mut out = format!(
        "{bold}Generation speed{reset} {dim}(tok/s, one sample per second of decoding){reset}\n"
    );
    if samples.is_empty() {
        let _ = writeln!(out, "{dim}nothing generated yet{reset}");
        return out;
    }
    let chart = render_chart(samples, width, height);
    if color {
        // The dots in the theme green, the axis labels left as they are: the
        // chart line is the accent, the scale is furniture.
        let green = format!("\x1b[38;5;{}m", crate::status::THEME_COLOR);
        for line in chart.lines() {
            match line.split_once('┤') {
                Some((axis, dots)) => {
                    let _ = writeln!(out, "{axis}┤{green}{dots}{reset}");
                }
                None => out.push_str(line),
            }
        }
    } else {
        out.push_str(&chart);
    }
    let n = samples.len();
    let last = samples[n - 1];
    // At most `TokRing::CAPACITY` samples, exact in an f64.
    #[allow(clippy::cast_precision_loss)]
    let avg = samples.iter().sum::<f64>() / n as f64;
    let min = samples.iter().copied().fold(f64::MAX, f64::min);
    let max = samples.iter().copied().fold(f64::MIN, f64::max);
    let shown = n.min(width * 2);
    let _ = writeln!(
        out,
        "{dim}now{reset} {bold}{last:.1}{reset} {dim}· avg{reset} {avg:.1} {dim}· min{reset} {min:.1} {dim}· max{reset} {max:.1} {dim}· last {shown} s of {n}{reset}"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_keeps_the_newest_samples_in_order() {
        let mut ring = TokRing::new();
        for i in 1..=(TokRing::CAPACITY + 3) {
            ring.push(f64::from(u16::try_from(i).unwrap()));
        }
        let s = ring.samples();
        assert_eq!(s.len(), TokRing::CAPACITY);
        assert!((s[0] - 4.0).abs() < f64::EPSILON);
        let newest = f64::from(u16::try_from(TokRing::CAPACITY + 3).unwrap());
        assert!((s[TokRing::CAPACITY - 1] - newest).abs() < f64::EPSILON);
        assert!(s.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn junk_rates_are_not_recorded() {
        let mut ring = TokRing::new();
        ring.push(0.0);
        ring.push(-3.0);
        ring.push(f64::NAN);
        ring.push(f64::INFINITY);
        assert!(ring.is_empty());
    }

    #[test]
    fn the_chart_puts_the_extremes_on_the_axis() {
        let chart = render_chart(&[10.0, 20.0, 30.0], 4, 2);
        let lines: Vec<&str> = chart.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("   30.0 ┤"), "{chart}");
        assert!(lines[1].starts_with("   10.0 ┤"), "{chart}");
        // Every column of dots is braille and the row is exactly `width` cells.
        for l in &lines {
            let cells: Vec<char> = l.chars().skip(label_prefix_len()).collect();
            assert_eq!(cells.len(), 4, "{l:?}");
            assert!(cells.iter().all(|c| ('\u{2800}'..='\u{28FF}').contains(c)));
        }
    }

    /// Characters before the first braille cell: the 7-wide label, a space
    /// and the axis tick.
    fn label_prefix_len() -> usize {
        "        ┤".chars().count()
    }

    #[test]
    fn a_rising_series_draws_a_continuous_line() {
        // Two samples, bottom-left to top-right, in one cell: the join fills
        // the rows between them so no dot row is left blank.
        let chart = render_chart(&[1.0, 2.0], 1, 1);
        let cell = chart.chars().nth(label_prefix_len()).unwrap();
        let bits = cell as u32 - 0x2800;
        let rows_lit = (0..4)
            .filter(|&y| bits & (dot(0, y) | dot(1, y)) != 0)
            .count();
        assert_eq!(rows_lit, 4, "{chart}");
    }

    #[test]
    fn a_flat_series_still_renders() {
        let chart = render_chart(&[5.0; 10], 3, 2);
        assert_eq!(chart.lines().count(), 2);
        assert!(
            chart.contains('\u{2800}')
                || chart
                    .chars()
                    .any(|c| ('\u{2801}'..='\u{28FF}').contains(&c))
        );
    }

    #[test]
    fn the_report_summarises_and_handles_empty() {
        assert!(render_report(&[], 20, 4, false).contains("nothing generated yet"));
        let r = render_report(&[20.0, 30.0], 20, 4, false);
        assert!(r.contains("now 30.0"), "{r}");
        assert!(r.contains("avg 25.0"), "{r}");
        assert!(r.contains("last 2 s of 2"), "{r}");
    }

    #[test]
    fn the_colored_chart_paints_the_dots_theme_green() {
        let r = render_report(&[20.0, 30.0], 4, 2, true);
        let green = format!("\x1b[38;5;{}m", crate::status::THEME_COLOR);
        // Every chart row: axis tick, then green, then dots, then reset.
        let rows: Vec<&str> = r.lines().filter(|l| l.contains('┤')).collect();
        assert_eq!(rows.len(), 2, "{r}");
        for row in rows {
            let (_, dots) = row.split_once('┤').unwrap();
            assert!(dots.starts_with(&green), "{row:?}");
            assert!(dots.ends_with("\x1b[0m"), "{row:?}");
        }
        assert!(!render_report(&[20.0], 4, 2, false).contains(&green));
    }

    #[test]
    fn the_sampler_reports_once_per_interval_over_the_window() {
        let mut s = Sampler::default();
        let t0 = Instant::now();
        assert_eq!(s.observe(t0, 0), None);
        // Half a window in: nothing yet, and the mark does not move.
        assert_eq!(s.observe(t0 + Duration::from_millis(500), 7), None);
        // Two seconds in, 40 tokens: 20 tok/s over the window.
        let r = s.observe(t0 + Duration::from_secs(2), 40).unwrap();
        assert!((r - 20.0).abs() < 1e-9, "{r}");
        // The next window starts where the last sample was taken.
        assert_eq!(s.observe(t0 + Duration::from_millis(2500), 50), None);
        let r = s.observe(t0 + Duration::from_secs(3), 55).unwrap();
        assert!((r - 15.0).abs() < 1e-9, "{r}");
    }
}
