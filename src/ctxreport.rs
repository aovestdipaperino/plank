// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Enzo Lombardi

//! Rendering for the `/context` usage report.
//!
//! Split from the agent because the panel has to stay current *during* a turn,
//! and the worker owns the agent for its whole duration. Counting tokens needs
//! the engine's tokenizer, so only the agent can do it ([`Breakdown`], gathered
//! at every point the transcript changes and published into
//! `crate::worker::TurnShared`); drawing the result needs nothing but numbers,
//! so the UI thread can redraw on every status tick with the live fill.
//!
//! That split is what makes the panel live at two granularities. The category
//! breakdown refines at each tool boundary, which is where the transcript
//! actually grows in bulk — a tool result is the big item. The total, the
//! percentage and the grid move with every generated token, because
//! [`render`] takes the live figure from the status snapshot rather than
//! reading a field the worker updates only at the end of a pass.

use std::fmt::Write as _;

/// Glyph for an unused context cell in the grid.
const FREE_CELL: char = '⛶';
/// Grid width in cells.
const GRID_COLS: usize = 20;
/// Maximum grid height in rows.
const MAX_GRID_ROWS: usize = 16;
/// Category colors matching Claude Code: violet, cyan, purple, gray.
const COL_SYSTEM: &str = "\x1b[38;5;105m";
const COL_MCP: &str = "\x1b[38;5;44m";
const COL_MSG: &str = "\x1b[38;5;134m";
const COL_CONTEXT: &str = "\x1b[38;5;208m";
const COL_MEMORY: &str = "\x1b[38;5;114m";
const COL_FREE: &str = "\x1b[38;5;240m";
/// ANSI style reset.
const ANSI_RESET: &str = "\x1b[0m";

/// The token counts behind the `/context` report, as the agent measured them.
///
/// Raw and unscaled: the scaling against what the engine reports actually
/// resident is [`render`]'s job, because that figure is the one that moves
/// while a pass generates and the counts here do not.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Breakdown {
    /// Context window size in tokens. Zero means "not known yet", which
    /// [`render`] shows as an empty report rather than dividing by it.
    pub ctx_size: i32,
    /// Loaded model, for the report's first line. Empty is allowed.
    pub model: String,
    /// System prompt, with the MCP schemas already subtracted out.
    pub system: i32,
    /// MCP tool schemas, which live inside the composed system prompt.
    pub mcp: i32,
    /// Transcript messages, less the `AGENTS.md` and memory they carry.
    pub messages: i32,
    /// `AGENTS.md` discovered at session start.
    pub agents_md: i32,
    /// Persistent memory files.
    pub memory: i32,
}

impl Breakdown {
    /// The sum of the categories, before any scaling.
    #[must_use]
    pub fn estimated(&self) -> i32 {
        self.system + self.mcp + self.messages + self.agents_md + self.memory
    }
}

/// Categories after scaling to `used`, in report order, each with its color.
///
/// The engine's resident count is authoritative and always at least the
/// estimate, since the rendered transcript carries framing the per-message
/// counts do not. The three measured-by-tokenizer categories absorb the
/// difference proportionally; `AGENTS.md` and memory are exact subsets of the
/// transcript, so scaling them would double-count the same tokens twice.
fn scaled_categories(b: &Breakdown, used: i32) -> Vec<(&'static str, i32, &'static str)> {
    let estimated = b.estimated();
    let scale = |t: i32| {
        if used > estimated && estimated > 0 {
            i32::try_from(i64::from(t) * i64::from(used) / i64::from(estimated)).unwrap_or(t)
        } else {
            t
        }
    };
    let mut categories = vec![
        ("System prompt", scale(b.system), COL_SYSTEM),
        ("MCP tools", scale(b.mcp), COL_MCP),
    ];
    if b.agents_md > 0 {
        categories.push(("AGENTS.md", b.agents_md, COL_CONTEXT));
    }
    if b.memory > 0 {
        categories.push(("Memory", b.memory, COL_MEMORY));
    }
    categories.push(("Messages", scale(b.messages), COL_MSG));
    categories
}

/// Glyph for a cell by its fill fraction: <25%, <50%, <75%, full.
fn fill_glyph(frac: f64) -> char {
    if frac < 0.25 {
        '⛀'
    } else if frac < 0.5 {
        '⛂'
    } else if frac < 0.75 {
        '⛁'
    } else {
        '⛃'
    }
}

/// The grid cells, and how many tokens each one stands for.
///
/// Adaptive density: 1k tokens per cell, coarsened in 1k steps so the grid
/// never exceeds half a typical 24-row screen.
fn grid(
    categories: &[(&'static str, i32, &'static str)],
    ctx_size: i32,
) -> (Vec<(char, &'static str)>, usize) {
    #[allow(clippy::cast_sign_loss)]
    let ctx = ctx_size.max(1) as usize;
    let tokens_per_cell = ctx
        .div_ceil(GRID_COLS * MAX_GRID_ROWS)
        .div_ceil(1000)
        .max(1)
        * 1000;
    let total_cells = ctx.div_ceil(tokens_per_cell);
    let mut cells: Vec<(char, &'static str)> = Vec::with_capacity(total_cells);
    for &(_, tokens, col) in categories {
        if tokens <= 0 || cells.len() == total_cells {
            continue;
        }
        // Whole cells render full; the trailing remainder renders with a
        // glyph matching its fill fraction.
        #[allow(clippy::cast_sign_loss)]
        let tokens = tokens as usize;
        let full = (tokens / tokens_per_cell).min(total_cells - cells.len());
        cells.extend(std::iter::repeat_n(('⛃', col), full));
        let rem = tokens % tokens_per_cell;
        if rem > 0 && cells.len() < total_cells {
            #[allow(clippy::cast_precision_loss)]
            cells.push((fill_glyph(rem as f64 / tokens_per_cell as f64), col));
        }
    }
    cells.truncate(total_cells);
    cells.resize(total_cells, (FREE_CELL, COL_FREE));
    (cells, tokens_per_cell)
}

/// The right-hand column: model, totals, then the category legend.
fn legend(
    b: &Breakdown,
    categories: &[(&'static str, i32, &'static str)],
    used: i32,
    tokens_per_cell: usize,
    color: bool,
) -> Vec<String> {
    let paint = |col: &'static str| if color { col } else { "" };
    let reset = if color { ANSI_RESET } else { "" };
    let ctx_size = b.ctx_size.max(1);
    let pct = |n: i32| f64::from(n) * 100.0 / f64::from(ctx_size);
    let mut right: Vec<String> = Vec::new();
    if !b.model.is_empty() {
        right.push(b.model.clone());
    }
    right.push(format!(
        "{}/{} tokens ({:.0}%)",
        crate::status::format_ctx_size(used),
        crate::status::format_ctx_size(ctx_size),
        pct(used)
    ));
    right.push(String::new());
    right.push("Estimated usage by category".to_owned());
    for &(label, tokens, col) in categories {
        right.push(format!(
            "{}⛃{reset} {label}: {} tokens ({:.1}%)",
            paint(col),
            crate::status::format_ctx_size(tokens),
            pct(tokens)
        ));
    }
    right.push(format!(
        "{}{FREE_CELL}{reset} Free space: {} ({:.1}%)",
        paint(COL_FREE),
        crate::status::format_ctx_size(ctx_size - used),
        pct(ctx_size - used)
    ));
    right.push(format!(
        "1 cell = {} tokens",
        crate::status::format_ctx_size(i32::try_from(tokens_per_cell).unwrap_or(i32::MAX))
    ));
    right
}

/// The `/context` report: a 20-column cell grid beside the model and totals,
/// then the estimated usage per category (Claude Code's layout).
///
/// `used` is what the engine reports resident right now — the figure the
/// status snapshot carries and updates on every generated token, so the panel
/// fills live. A non-positive `used` falls back to [`Breakdown::estimated`],
/// which is what an idle session before its first pass has to show.
#[must_use]
pub fn render(b: &Breakdown, used: i32, color: bool) -> String {
    let paint = |col: &'static str| if color { col } else { "" };
    let reset = if color { ANSI_RESET } else { "" };
    let ctx_size = b.ctx_size.max(1);
    let used = if used > 0 { used } else { b.estimated() };
    let categories = scaled_categories(b, used);
    // Re-derived from the scaled categories rather than taken from `used`:
    // the cells are drawn from these numbers, so the total has to be their
    // sum or the legend and the grid disagree with each other.
    let used = categories
        .iter()
        .map(|&(_, t, _)| t)
        .sum::<i32>()
        .min(ctx_size);
    let (cells, tokens_per_cell) = grid(&categories, ctx_size);
    let right = legend(b, &categories, used, tokens_per_cell, color);
    let grid_rows = cells.len().div_ceil(GRID_COLS);

    let mut out = String::from("Context Usage\n");
    let rows = right.len().max(grid_rows);
    for row in 0..rows {
        out.push_str("  ");
        if row < grid_rows {
            let start = row * GRID_COLS;
            let end = (start + GRID_COLS).min(cells.len());
            for &(glyph, col) in &cells[start..end] {
                out.push_str(paint(col));
                out.push(glyph);
                out.push_str(reset);
                out.push(' ');
            }
            out.push_str(&" ".repeat(2 * (start + GRID_COLS - end)));
        } else {
            out.push_str(&" ".repeat(2 * GRID_COLS));
        }
        if let Some(text) = right.get(row) {
            let _ = write!(out, "   {text}");
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breakdown() -> Breakdown {
        Breakdown {
            ctx_size: 100_000,
            model: "Test Model".to_owned(),
            system: 4_000,
            mcp: 1_000,
            messages: 5_000,
            agents_md: 0,
            memory: 0,
        }
    }

    #[test]
    fn the_report_shows_the_model_totals_and_every_category() {
        let out = render(&breakdown(), 10_000, false);
        assert!(out.starts_with("Context Usage\n"), "{out}");
        assert!(out.contains("Test Model"), "{out}");
        assert!(out.contains("10k/100k tokens (10%)"), "{out}");
        assert!(out.contains("System prompt: 4k tokens (4.0%)"), "{out}");
        assert!(out.contains("MCP tools: 1k tokens (1.0%)"), "{out}");
        assert!(out.contains("Messages: 5k tokens (5.0%)"), "{out}");
        assert!(out.contains("Free space: 90k (90.0%)"), "{out}");
        // Absent categories stay absent rather than showing a zero row.
        assert!(!out.contains("AGENTS.md"), "{out}");
        assert!(!out.contains("Memory"), "{out}");
    }

    #[test]
    fn agents_md_and_memory_get_their_own_rows_when_present() {
        let b = Breakdown {
            agents_md: 900,
            memory: 300,
            ..breakdown()
        };
        let out = render(&b, 0, false);
        assert!(out.contains("AGENTS.md: 900 tokens"), "{out}");
        assert!(out.contains("Memory: 300 tokens"), "{out}");
    }

    /// The live figure is the whole point of the split: the same breakdown
    /// drawn against a larger resident count reports a fuller window, without
    /// the tokenizer running again.
    #[test]
    fn a_larger_live_figure_fills_the_report_without_recounting() {
        let b = breakdown();
        let idle = render(&b, 0, false);
        assert!(idle.contains("10k/100k tokens (10%)"), "{idle}");
        let mid_pass = render(&b, 40_000, false);
        assert!(mid_pass.contains("40k/100k tokens (40%)"), "{mid_pass}");
        // The measured categories absorb the difference proportionally; the
        // transcript subsets do not, so they cannot be counted twice.
        assert!(mid_pass.contains("System prompt: 16k"), "{mid_pass}");
        assert!(mid_pass.contains("Messages: 20k"), "{mid_pass}");
        // More of the grid is filled: fewer free cells than at idle.
        let free_at_idle = idle.matches(FREE_CELL).count();
        let free_mid_pass = mid_pass.matches(FREE_CELL).count();
        assert!(
            free_mid_pass < free_at_idle,
            "idle {free_at_idle} vs {free_mid_pass}"
        );
    }

    #[test]
    fn a_breakdown_with_no_engine_yet_renders_rather_than_dividing_by_zero() {
        let out = render(&Breakdown::default(), 0, false);
        assert!(out.starts_with("Context Usage\n"), "{out}");
        assert!(out.contains("0/1 tokens (0%)"), "{out}");
    }

    #[test]
    fn color_paints_the_cells_and_plain_text_does_not() {
        let colored = render(&breakdown(), 10_000, true);
        assert!(colored.contains(COL_SYSTEM), "{colored}");
        assert!(colored.contains(ANSI_RESET));
        let plain = render(&breakdown(), 10_000, false);
        assert!(!plain.contains('\u{1b}'), "{plain}");
    }
}
