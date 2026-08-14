// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Matrix rain: columns of glyphs falling down the screen.
//!
//! **Ported verbatim from plank's `src/arcade/matrix.rs`**, with one change:
//! the imports. Everything about how the rain behaves — the speeds, the trail
//! curve, the flicker rate, the ring of held glyphs — is the original, because
//! the point of the port is that the rain looks the same after it, and the
//! two existing arcade tests are the check.
//!
//! Weather rather than a game — like the starfield it has no score, no lives
//! and no end — but it keeps the same contract as the rest of
//! [`crate::arcade`]: state advances only through an injected `dt`, the only
//! randomness comes from a seeded [`Rng`], and rendering hands out [`Glyph`]s
//! for `tui.rs` to paint. So a given seed always rains the same way, which is
//! what makes it testable without a terminal.
//!
//! # What makes it read as *the* rain
//!
//! Four things, and all four matter:
//!
//! - **The head is nearly white.** The leading cell of every column is the
//!   brightest thing on screen; the trail behind it falls away on a curve, not
//!   linearly, so the glow stays concentrated just under the head.
//! - **Cells hold their glyph while the trail passes over them.** The
//!   characters do not fall — the *light* falls, over a column of glyphs that
//!   were already there. That is why each column carries a small ring of
//!   glyphs indexed by screen row ([`Column::cells`]) instead of a queue that
//!   scrolls.
//! - **Glyphs flicker.** A steady trickle of cells is re-rolled every second,
//!   so the field shimmers without the whole screen reshuffling at once.
//! - **Columns differ.** Speed, trail length and brightness are dealt per
//!   column and re-dealt on every pass, and a dim column reads as a layer
//!   further back. Uniform columns look like a progress bar falling over.
//!
//! # Geometry
//!
//! Everything vertical is stored as a fraction of the screen height and every
//! column is independent of the terminal's width, so a resize re-lays the rain
//! instead of stranding it. [`Rain`] therefore needs no size to be built, and
//! [`Rain::glyphs`] simply uses the first `w` columns of the [`MAX_COLS`] it
//! keeps — enough for any terminal anyone is going to open.

// The port is verbatim, so it carries accessors the *host* used for its
// footer (`speed`, `charset`, `Charset::label`) that this guest has no
// consumer for yet. Kept rather than trimmed: the value of a verbatim port is
// that it can be diffed against the original, and a HUD is a face away.
#![allow(dead_code)]

use crate::support::{Glyph, MAX_STEP_MS, Rgb, Rng, lerp_rgb};

/// Columns kept, whatever the terminal's width.
///
/// [`Rain::step`] has no size to work from (the arcade's step signature is a
/// bare `dt`), so the field cannot grow on a resize — it has to be wide enough
/// up front. 512 columns covers a maximized terminal on any display sold this
/// decade, and costs about 40 KB while it is on screen.
const MAX_COLS: usize = 512;

/// Glyphs remembered per column, indexed by screen row modulo this.
///
/// Prime, and comfortably above the height of a normal terminal, so the same
/// glyph does not visibly recur down a column on a tall screen.
const RING: usize = 61;

/// Fall speed dealt to a column, in screen heights per second, before the
/// user's own multiplier.
const SPEED_MIN: f32 = 0.14;
const SPEED_MAX: f32 = 0.62;

/// Trail length dealt to a column, as a fraction of the screen height. The
/// longest reach the full height: at any moment a few columns are lit from the
/// top edge to the bottom, which is what keeps the screen feeling covered.
const LEN_MIN: f32 = 0.35;
const LEN_MAX: f32 = 1.0;

/// Brightness dealt to an ordinary column. Below 1.0 across the board: the
/// full-brightness columns are the exception ([`HERO_CHANCE`]), which is what
/// gives the field its depth.
const GAIN_MIN: f32 = 0.35;
const GAIN_MAX: f32 = 0.8;
/// How often a column is dealt full brightness instead.
const HERO_CHANCE: f32 = 0.18;

/// How far above the top edge a column is parked when it is re-dealt, in
/// screen heights. This gap *is* the pause between passes: nothing is drawn
/// while the head is above the screen, so a random gap staggers the columns
/// apart and keeps the rain from falling in ranks.
const GAP_MIN: f32 = 0.02;
const GAP_MAX: f32 = 0.9;

/// The user's speed multiplier, and how far it can be pushed either way.
const SPEED_SCALE_MIN: f32 = 0.15;
const SPEED_SCALE_MAX: f32 = 4.0;
const SPEED_SCALE_DEFAULT: f32 = 1.0;

/// Fraction of the field's glyphs re-rolled per second, so a cell holds its
/// character for two seconds on average. Tuned by eye: enough that the field
/// is never still, slow enough that a glyph you are looking at stays put long
/// enough to read. Expressed as a fraction rather than a count because only
/// the columns that fit the terminal are ever seen — a flat count would make
/// the flicker depend on the window's width.
const CHURN_PER_SEC: f32 = 0.5;

/// How sharply the trail falls away behind the head. Above 1.0 concentrates
/// the light just under the head and lets the rest fade into the dark.
const TAIL_CURVE: f32 = 1.7;

/// The leading cell: white with just enough green in it to belong.
const RAIN_HEAD: Rgb = (208, 255, 216);
/// The brightest the trail itself gets, immediately behind the head.
const RAIN_BODY: Rgb = (56, 232, 92);
/// The far end of the trail, a green barely out of the black.
const RAIN_TAIL: Rgb = (0, 40, 18);
/// What a column is dimmed towards. Black, so a dim column really does recede
/// rather than turning grey.
const BLACK: Rgb = (0, 0, 0);

/// Dimmest green still worth painting.
///
/// Not an optimization — over live output (`t`, the translucent layer) every
/// glyph painted *replaces* the character underneath, so a cell too dark to
/// see would silently punch a hole in the model's text. Below this the cell is
/// left alone instead, which also gives the dimmest columns a soft end
/// instead of a hard stop.
const MIN_VISIBLE: u8 = 18;

/// Which alphabet the rain falls in, cycled in place with `c`.
///
/// The katakana are the point — that is what the film's rain is made of — but
/// they are also the one thing here that a terminal font can fail to draw. The
/// cycle is the escape hatch: [`Charset::Binary`] is pure ASCII and cannot
/// fail anywhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Charset {
    /// Half-width katakana, digits and a few marks. The default.
    #[default]
    Katakana,
    /// Ones and zeros. The safe fallback for a font without katakana.
    Binary,
    /// Punctuation-heavy ASCII: source code falling down the screen.
    Code,
}

impl Charset {
    /// The name shown in the footer.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Katakana => "katakana",
            Self::Binary => "binary",
            Self::Code => "code",
        }
    }

    /// The next alphabet in the `c` cycle.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Katakana => Self::Binary,
            Self::Binary => Self::Code,
            Self::Code => Self::Katakana,
        }
    }

    /// The glyphs themselves.
    ///
    /// Every one of these must be exactly one terminal cell wide, or a column
    /// would smear into its neighbour — half-width katakana are, full-width
    /// ones are not. A test holds the line.
    #[must_use]
    pub const fn glyphs(self) -> &'static [char] {
        match self {
            Self::Katakana => KATAKANA,
            Self::Binary => BINARY,
            Self::Code => CODE,
        }
    }
}

/// Half-width katakana (U+FF66..U+FF9D) with digits and a few marks mixed in,
/// which is roughly what the film's rain is: kana carrying the texture,
/// numerals breaking it up.
const KATAKANA: &[char] = &[
    'ｱ', 'ｲ', 'ｳ', 'ｴ', 'ｵ', 'ｶ', 'ｷ', 'ｸ', 'ｹ', 'ｺ', 'ｻ', 'ｼ', 'ｽ', 'ｾ', 'ｿ', 'ﾀ', 'ﾁ', 'ﾂ', 'ﾃ',
    'ﾄ', 'ﾅ', 'ﾆ', 'ﾇ', 'ﾈ', 'ﾉ', 'ﾊ', 'ﾋ', 'ﾌ', 'ﾍ', 'ﾎ', 'ﾏ', 'ﾐ', 'ﾑ', 'ﾒ', 'ﾓ', 'ﾔ', 'ﾕ', 'ﾖ',
    'ﾗ', 'ﾘ', 'ﾙ', 'ﾚ', 'ﾛ', 'ﾜ', 'ﾝ', 'ｦ', 'ｯ', 'ｰ', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', ':', '=', '*', '+',
];

/// Ones and zeros, and nothing a font can miss.
const BINARY: &[char] = &['0', '1'];

/// Source code falling down the screen: brackets, operators and identifiers'
/// worth of letters.
const CODE: &[char] = &[
    '{', '}', '[', ']', '(', ')', '<', '>', '/', '\\', '|', ';', ':', '=', '+', '-', '*', '&', '^',
    '%', '$', '#', '@', '!', '?', '~', '_', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'C', 'E', 'F', 'H', 'K', 'L', 'N', 'R', 'T', 'X', 'Z',
];

/// One falling column.
#[derive(Debug, Clone)]
struct Column {
    /// Where the leading cell is, in screen heights: 0.0 is the top row, 1.0
    /// the bottom. Negative means the head has not reached the screen yet.
    head: f32,
    /// Fall speed, in screen heights per second, before the user's multiplier.
    speed: f32,
    /// How far the trail reaches behind the head, as a fraction of the height.
    len: f32,
    /// Overall brightness, `0.0`..=`1.0`.
    gain: f32,
    /// The glyph shown at each screen row, indexed by `row % RING`.
    ///
    /// Indices into the current [`Charset`], reduced modulo its length at
    /// paint time — which is what lets `c` re-letter the whole field without
    /// disturbing a single cell's identity.
    cells: [u8; RING],
}

impl Column {
    /// A column mid-fall, for the very first frame.
    fn new(rng: &mut Rng) -> Self {
        let mut cells = [0u8; RING];
        for cell in &mut cells {
            *cell = glyph_index(rng);
        }
        let mut col = Self {
            head: 0.0,
            speed: 0.0,
            len: 0.0,
            gain: 0.0,
            cells,
        };
        col.deal(rng);
        // Opening on an empty screen that then fills from the top reads as a
        // loading bar, not as rain. Scattering the heads over (and above) the
        // screen means the first frame is already weather.
        col.head = rng.range(-GAP_MAX, 1.0);
        col
    }

    /// Deals this column its next pass: new speed, length and brightness, and
    /// a head parked a random distance above the top edge.
    fn deal(&mut self, rng: &mut Rng) {
        self.speed = rng.range(SPEED_MIN, SPEED_MAX);
        self.len = rng.range(LEN_MIN, LEN_MAX);
        self.gain = if rng.next_f32() < HERO_CHANCE {
            1.0
        } else {
            rng.range(GAIN_MIN, GAIN_MAX)
        };
        self.head = -rng.range(GAP_MIN, GAP_MAX);
    }
}

/// A random glyph slot, kept as a `u8` so a column's memory is 61 bytes rather
/// than 61 pointers. Reduced modulo the live alphabet at paint time.
fn glyph_index(rng: &mut Rng) -> u8 {
    u8::try_from(rng.next_u64() & 0xff).unwrap_or(0)
}

/// A random index below `n`, without a cast that clippy has to be told about.
fn pick(rng: &mut Rng, n: usize) -> usize {
    let n = u64::try_from(n).unwrap_or(1).max(1);
    usize::try_from(rng.next_u64() % n).unwrap_or(0)
}

/// The rain itself: [`MAX_COLS`] independent columns and a shared speed knob.
#[derive(Debug)]
pub struct Rain {
    cols: Vec<Column>,
    rng: Rng,
    /// Multiplies every column's own speed (`↑`/`↓`, wheel).
    speed: f32,
    charset: Charset,
    /// Glyph re-rolls owed from the last step, carried across frames.
    ///
    /// At 20 Hz a frame earns four-and-a-bit re-rolls; dropping the fraction
    /// every frame would quietly cost a fifth of the flicker.
    churn: f32,
}

impl Rain {
    /// Deals a field already mid-fall.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let cols = (0..MAX_COLS).map(|_| Column::new(&mut rng)).collect();
        Self {
            cols,
            rng,
            speed: SPEED_SCALE_DEFAULT,
            charset: Charset::default(),
            churn: 0.0,
        }
    }

    /// The current speed multiplier.
    #[must_use]
    pub const fn speed(&self) -> f32 {
        self.speed
    }

    /// Scales the fall speed, clamped to the usable range.
    pub const fn scale_speed(&mut self, factor: f32) {
        self.speed = (self.speed * factor).clamp(SPEED_SCALE_MIN, SPEED_SCALE_MAX);
    }

    /// Which alphabet is falling.
    #[must_use]
    pub const fn charset(&self) -> Charset {
        self.charset
    }

    /// Moves to the next alphabet, leaving every cell where it is.
    pub const fn cycle_charset(&mut self) {
        self.charset = self.charset.next();
    }

    /// Advances every column by `dt_ms`, re-dealing the ones that have fallen
    /// clear of the bottom, and re-rolls a few glyphs so the field shimmers.
    #[allow(clippy::cast_precision_loss)] // dt is bounded by MAX_STEP_MS
    pub fn step(&mut self, dt_ms: u64) {
        let dt = dt_ms.min(MAX_STEP_MS) as f32 / 1000.0;
        // Split borrow: the columns are walked mutably while the shared Rng
        // deals the ones that have finished their pass.
        let Self {
            cols, rng, speed, ..
        } = self;
        let travel = *speed * dt;
        for col in cols.iter_mut() {
            col.head += col.speed * travel;
            // Re-deal only once even the tail is past the bottom edge, or the
            // column would blink out with light still on screen.
            if col.head - col.len > 1.0 {
                col.deal(rng);
            }
        }
        self.flicker(dt);
    }

    /// Re-rolls the glyphs owed for a `dt`-second step.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // bounded below
    fn flicker(&mut self, dt: f32) {
        // Bounded by construction: dt is at most MAX_STEP_MS and the field is
        // MAX_COLS x RING, so the count fits comfortably and the truncation is
        // the intended floor.
        let field = (self.cols.len() * RING) as f32;
        self.churn += CHURN_PER_SEC * field * dt;
        let owed = self.churn.floor();
        self.churn -= owed;
        for _ in 0..(owed.max(0.0) as usize) {
            let col = pick(&mut self.rng, self.cols.len());
            let slot = pick(&mut self.rng, RING);
            self.cols[col].cells[slot] = glyph_index(&mut self.rng);
        }
    }

    /// Paints the rain onto a `w` x `h` area.
    ///
    /// One column per terminal column, head first: the head is the brightest
    /// cell and the trail above it fades on [`TAIL_CURVE`]. Cells too dark to
    /// see are left out entirely — see [`MIN_VISIBLE`] for why that is a
    /// correctness point and not a saving.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )] // screen coords: small integers, bounds-checked below
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let set = self.charset.glyphs();
        let fh = f32::from(h);
        let rows = i32::from(h);
        let mut out = Vec::with_capacity(usize::from(w) * usize::from(h) / 3);
        for (x, col) in self.cols.iter().take(usize::from(w)).enumerate() {
            let head = col.head * fh;
            if head < 0.0 {
                // Still falling in from above: nothing of this column is on
                // screen, and its trail is above the head, not below it.
                continue;
            }
            let head_row = head as i32;
            // At least one cell, and never longer than the screen is tall.
            let steps = ((col.len * fh) as i32).clamp(1, rows);
            for k in 0..=steps {
                let row = head_row - k;
                if row < 0 {
                    // Walked off the top; every further step is further off.
                    break;
                }
                if row >= rows {
                    // The head is below the bottom edge — its trail is still
                    // coming, so keep walking up rather than giving up.
                    continue;
                }
                let along = k as f32 / steps as f32;
                let glow = (1.0 - along).powf(TAIL_CURVE);
                let lit = if k == 0 {
                    RAIN_HEAD
                } else {
                    lerp_rgb(RAIN_TAIL, RAIN_BODY, glow)
                };
                // The column's own brightness dims the whole of it, tail
                // included — that is what sets a column back rather than
                // merely capping its head. Applied to the color and not to
                // `glow`, or the darkest end of every column would come out
                // the same [`RAIN_TAIL`] whatever the column's gain, and the
                // depth would be gone exactly where it shows most.
                let color = lerp_rgb(BLACK, lit, col.gain);
                if color.1 < MIN_VISIBLE {
                    continue;
                }
                let slot = usize::try_from(row).unwrap_or(0) % RING;
                out.push(Glyph {
                    x: x as u16,
                    y: row as u16,
                    ch: set[usize::from(col.cells[slot]) % set.len()],
                    color,
                });
            }
        }
        out
    }
}
