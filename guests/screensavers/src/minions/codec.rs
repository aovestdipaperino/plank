// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Sprite-sheet codec for the minions screensaver: one format, two users.
//!
//! `build.rs` compiles this file *as well* (through `#[path]`) so the encoder
//! and the decoder can never drift into two different formats — the usual
//! failure of a "generate an asset at build time" scheme. The encoder runs at
//! build time and never reaches the binary; the decoder runs once, the first
//! time the screensaver comes up.
//!
//! # Why compress six small grids at all
//!
//! Because the alternative is not "a few hundred bytes", it is the whole sheet
//! sitting in `.rodata` for the life of the process, in a binary that is
//! shipped as a Homebrew bottle. The art is 1 440 cells of a 30-ink alphabet
//! that repeats enormously, in three different directions, and the format has
//! exactly one op for each:
//!
//! - **[`COPY`] — the same cell as the pose before.** The poses are ordered so
//!   that neighbours are near-identical (idle → blink is two cells), which
//!   turns a whole 240-cell pose into a handful of bytes. One byte, no operand:
//!   the cheapest op, because it is the one that fires most.
//! - **[`RUN`] — the same ink again.** The margins, the strap, the overalls:
//!   any stretch of one ink costs two bytes whatever its length.
//! - **[`MATCH`] — something already drawn, up to 256 cells back.** A minion is
//!   built out of a handful of rows repeated down the body, and a row is 15
//!   cells: this is what buys the *first* pose, which has no pose behind it to
//!   copy from.
//!
//! Anything none of the three can produce goes out as a literal. That is the
//! whole format: no dictionary, no entropy coder, no dependency, and a decoder
//! of twenty lines that cannot fail on data this file produced. It packs the
//! sheet to about a fifth; the exact figures are measured at build time and
//! land in `SHEET_BYTES` and `BLOB_BYTES`.

/// The sheet's alphabet: the ink at index `i` is written as `LEGEND[i]` in
/// `src/resources/minions.txt`, and painted by `INKS[i]` in the module above.
///
/// Index 0 is the transparent cell, which matters twice over: it is the ink
/// the margins are made of, and it is what a decode buffer starts out as, so
/// the first pose delta-codes against emptiness for free.
pub const LEGEND: &[u8] = b".YystS[]{}|-Wec^<=>123456BpGFh";

/// Every pose is the same size, which is what lets one delta-code against the
/// next. The short minion is bottom-aligned inside the grid rather than given a
/// grid of its own.
pub const SHEET_W: usize = 15;
pub const SHEET_H: usize = 16;
/// Poses in the sheet: idle, blink and laugh for each of the two.
pub const POSES: usize = 6;
/// Cells in one pose.
pub const POSE_CELLS: usize = SHEET_W * SHEET_H;

/// Op kinds, in the top two bits of an op byte. The low six carry `len - 1`,
/// so one byte covers a span of 1..=64 cells — enough for the longest stretch
/// in a 15-wide grid and small enough that a two-cell match is still a win.
const COPY: u8 = 0x00;
const RUN: u8 = 0x40;
const LIT: u8 = 0x80;
const MATCH: u8 = 0xc0;
/// Widest span a single op can cover.
const MAX_SPAN: usize = 64;
/// How far back a [`MATCH`] can reach, carried in one operand byte as
/// `distance - 1`. 256 cells is a whole pose plus a row, which is every repeat
/// the art actually contains — a wider window would cost a second byte on
/// every match to buy nothing.
const MAX_BACK: usize = 256;
/// Shortest match worth a [`COPY`] op: one byte for two cells beats the two
/// bytes those cells would cost inside a literal.
const MIN_COPY: usize = 2;
/// Shortest repeat worth a two-byte op.
const MIN_RUN: usize = 3;

/// Bytes the header occupies: poses, width, height.
///
/// Written out rather than assumed so the decoder can reject a blob built from
/// a different sheet instead of drawing garbage — the two halves of this file
/// are compiled separately, and a stale `OUT_DIR` is exactly the kind of thing
/// that happens.
const HEADER: usize = 3;

/// Turns the authored sheet into one flat run of ink codes.
///
/// Comments (`#`), blank lines and the `@name` pose headers are the file's
/// scaffolding and carry nothing; every other line is a row and must be
/// exactly [`SHEET_W`] cells of legend bytes.
///
/// # Errors
///
/// Returns a message naming the offending line when a row is the wrong width
/// or holds a byte that is not in [`LEGEND`], and when the sheet does not come
/// to exactly [`POSES`] poses.
pub fn parse_sheet(text: &str) -> Result<Vec<u8>, String> {
    let mut cells = Vec::with_capacity(POSES * POSE_CELLS);
    for (n, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') || line.starts_with('@') {
            continue;
        }
        if line.len() != SHEET_W {
            return Err(format!(
                "line {}: row is {} cells wide, expected {SHEET_W}",
                n + 1,
                line.len()
            ));
        }
        for b in line.bytes() {
            let code = LEGEND
                .iter()
                .position(|l| *l == b)
                .ok_or_else(|| format!("line {}: {:?} is not in the legend", n + 1, b as char))?;
            // The legend is 30 entries; a code always fits a byte.
            cells.push(u8::try_from(code).unwrap_or(0));
        }
    }
    if cells.len() != POSES * POSE_CELLS {
        return Err(format!(
            "sheet holds {} cells, expected {} ({POSES} poses of {SHEET_W}x{SHEET_H})",
            cells.len(),
            POSES * POSE_CELLS
        ));
    }
    Ok(cells)
}

/// Packs the flat cell run into the blob the binary carries.
///
/// Greedy and single-pass: at every cell it takes the longest [`COPY`] against
/// the previous pose, else the longer of the best [`RUN`] and the best
/// [`MATCH`], else it lets a literal grow until one of those becomes worth
/// having. Optimal parsing would buy a handful of bytes on a sheet this
/// regular and cost the reader an hour, so it is not done.
///
/// Ops never straddle a pose: it keeps [`COPY`]'s reference unambiguous and
/// costs a byte or two of alignment at six boundaries.
#[must_use]
pub fn encode(cells: &[u8]) -> Vec<u8> {
    let mut out = vec![
        u8::try_from(POSES).unwrap_or(0),
        u8::try_from(SHEET_W).unwrap_or(0),
        u8::try_from(SHEET_H).unwrap_or(0),
    ];
    for pose in 0..POSES {
        let mut i = 0;
        let mut lit = Vec::new();
        while i < POSE_CELLS {
            let at = pose * POSE_CELLS + i;
            let left = POSE_CELLS - i;
            let copy = span(left, |k| behind(cells, at + k) == cells[at + k]);
            let run = span(left, |k| cells[at + k] == cells[at]);
            let (dist, matched) = longest_match(cells, at, left);
            let repeat = run.max(matched);
            if copy >= MIN_COPY && copy >= repeat {
                flush(&mut out, &mut lit);
                out.push(COPY | tally(copy));
                i += copy;
            } else if repeat >= MIN_RUN {
                flush(&mut out, &mut lit);
                // A tie goes to the run: same two bytes, and it is the op that
                // needs no window to be reconstructed from.
                if run >= matched {
                    out.push(RUN | tally(run));
                    out.push(cells[at]);
                    i += run;
                } else {
                    out.push(MATCH | tally(matched));
                    out.push(tally(dist));
                    i += matched;
                }
            } else {
                lit.push(cells[at]);
                i += 1;
                if lit.len() == MAX_SPAN {
                    flush(&mut out, &mut lit);
                }
            }
        }
        flush(&mut out, &mut lit);
    }
    out
}

/// The cell one pose back, which is what [`COPY`] reproduces. Before the first
/// pose that is transparent ink — an empty grid to delta against, which is
/// most of any pose's margin anyway.
fn behind(cells: &[u8], at: usize) -> u8 {
    at.checked_sub(POSE_CELLS).map_or(0, |back| cells[back])
}

/// The longest repeat of what is already drawn, and how far back it starts.
///
/// A match may reach past the cell it starts at — `dist` of 1 is a run — since
/// the decoder copies forward one cell at a time. That is not exploited on
/// purpose; it simply is not worth forbidding.
fn longest_match(cells: &[u8], at: usize, left: usize) -> (usize, usize) {
    let mut best = (1, 0);
    for dist in 1..=MAX_BACK.min(at) {
        let len = span(left, |k| cells[at + k - dist] == cells[at + k]);
        if len > best.1 {
            best = (dist, len);
        }
        if best.1 == MAX_SPAN {
            break;
        }
    }
    best
}

/// How far `same` holds from the current cell, capped at one op's reach.
fn span(left: usize, same: impl Fn(usize) -> bool) -> usize {
    (0..left.min(MAX_SPAN)).take_while(|k| same(*k)).count()
}

/// A span length as the low six bits of an op byte.
fn tally(len: usize) -> u8 {
    u8::try_from(len - 1).unwrap_or(0)
}

/// Emits the pending literal, if any, and clears it.
fn flush(out: &mut Vec<u8>, lit: &mut Vec<u8>) {
    if lit.is_empty() {
        return;
    }
    out.push(LIT | tally(lit.len()));
    out.append(lit);
}

/// Unpacks a blob produced by [`encode`] back into [`POSES`] grids of ink
/// codes, laid end to end.
///
/// # Panics
///
/// Panics when the blob was not built from this sheet — a dimension mismatch
/// or a truncated stream. Both mean the build wrote one format and the binary
/// reads another, which is a bug here rather than bad input: nothing but
/// `build.rs` ever feeds this.
#[must_use]
pub fn decode(blob: &[u8]) -> Vec<u8> {
    assert!(
        blob.len() > HEADER
            && usize::from(blob[0]) == POSES
            && usize::from(blob[1]) == SHEET_W
            && usize::from(blob[2]) == SHEET_H,
        "minions sprite blob was built for a different sheet"
    );
    let mut cells = vec![0u8; POSES * POSE_CELLS];
    let mut at = 0usize;
    let mut op = HEADER;
    while at < cells.len() {
        let byte = blob[op];
        op += 1;
        let len = usize::from(byte & 0x3f) + 1;
        assert!(at + len <= cells.len(), "minions sprite blob overruns");
        match byte & 0xc0 {
            COPY => {
                // Copy from the pose before this one — one whole pose back,
                // which for the first pose is off the front of the buffer and
                // therefore transparent, exactly as the encoder assumed.
                for k in 0..len {
                    cells[at + k] = at.checked_sub(POSE_CELLS).map_or(0, |back| cells[back + k]);
                }
            }
            RUN => {
                let ink = blob[op];
                op += 1;
                cells[at..at + len].fill(ink);
            }
            LIT => {
                cells[at..at + len].copy_from_slice(&blob[op..op + len]);
                op += len;
            }
            _ => {
                let dist = usize::from(blob[op]) + 1;
                op += 1;
                assert!(dist <= at, "minions sprite blob reaches before its start");
                // Forward, one cell at a time: a match is allowed to overlap
                // what it is still writing, which is how a short repeat covers
                // a long stretch.
                for k in 0..len {
                    cells[at + k] = cells[at + k - dist];
                }
            }
        }
        at += len;
    }
    cells
}
