#![allow(dead_code)]
// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! **Ported verbatim from plank's `src/arcade/minions.rs`**, with one change:
//! the imports. The poses, the timing and the palette are the original, and a
//! host-side test compares this face against the built-in one glyph for glyph.
//!
//! The sprite sheet travels with the plugin (`resources/minions.txt`) and is
//! packed by the plugin's own `build.rs`, because a plugin that read plank's
//! source tree at build time would not be a plugin — it would be a coupled
//! artifact that happens to compile to wasm.
//!
//! Two minions on the shore of a night lake, giggling at each other.
//!
//! Weather rather than a game, like the starfield: no score, no lives, nothing
//! to end. It keeps the arcade's contract — state advances only through an
//! injected `dt`, the only randomness comes from a seeded [`Rng`], and drawing
//! hands [`Glyph`]s to `tui.rs` — so a given seed always plays out the same
//! way and the whole thing is testable without a terminal.
//!
//! # Where the picture comes from
//!
//! Six poses (idle, blink, laugh, for each of the two) live in
//! `src/resources/minions.txt` as ASCII paint-by-numbers, are packed at build
//! time by [`codec`], and are decoded once here on first use. What the binary
//! carries is [`codec::BLOB`]-sized, not sheet-sized: see [`SHEET_BYTES`] and
//! [`BLOB_BYTES`], which `build.rs` writes out from the actual files so the
//! number cannot go stale.
//!
//! Everything else is generated rather than drawn: the walk, the bob, the
//! lean, the nudge, the laughter puffs, the stars and the reflection in the
//! water. Six poses is the entire art budget.
//!
//! # Transparency, and what a terminal will actually give you
//!
//! A cell holds one character and one foreground colour. There is no alpha to
//! composite with — but there are three things that read as one, and this
//! module uses all three:
//!
//! - **Not drawing.** The only true transparency a terminal has. Every ink
//!   that fades below [`FAINTEST`] is dropped rather than dimmed, so over live
//!   model output (`t`) the text underneath survives instead of being punched
//!   out by a cell too dark to see. It is also why the art is mostly `.`.
//! - **Coverage.** `█ ▓ ▒ ░` fill a known fraction of the cell, and the
//!   terminal composites them against whatever is behind for free. The ramp
//!   *is* the alpha channel: [`Ink::dens`] is where an ink starts on it, and
//!   fading walks it down. Rounded shoulders, the fade-in when the screen
//!   opens and the reflection are all the same mechanism.
//! - **Colour.** Glyphs that are shapes rather than fills — the goggle rims,
//!   the eyes, the mouth — cannot lose coverage without losing their meaning,
//!   so they fade toward the night instead.
//!
//! The reflection uses all three at once, which is why it looks like water:
//! fewer dots, bluer, and the faintest rows simply are not there.

use crate::support::{Glyph, MAX_STEP_MS, Rgb, Rng, lerp_rgb};

pub mod codec;

use codec::{POSE_CELLS, SHEET_H, SHEET_W};

// `SHEET_BYTES` and `BLOB_BYTES`, measured by `build.rs` on the files it
// actually packed.
include!(concat!(env!("OUT_DIR"), "/minions_stats.rs"));

/// The packed sprite sheet: the only art in the executable.
static BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/minions.bin"));

/// The packing has to keep earning its complexity. A ceiling rather than a
/// target — the art packs to about a seventh, and a fifth leaves room to draw
/// another minion — but a change to the art or the codec that quietly gives
/// most of it back fails the build rather than shipping.
const _: () = assert!(BLOB_BYTES * 5 <= SHEET_BYTES, "the sprite blob got fat");

/// The decoded sheet, unpacked on first use and kept for the process.
///
/// A screensaver that has come up once will come up again, and 1.4 KB is not
/// worth reclaiming; decoding on every open, however, would put a loop in the
/// path of the first frame for no reason at all.
fn sheet() -> &'static [u8] {
    static SHEET: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    SHEET.get_or_init(|| codec::decode(BLOB))
}

// ---------------------------------------------------------------------------
// Inks
// ---------------------------------------------------------------------------

/// The coverage ramp, from nothing to a full cell. Index is [`Ink::dens`], and
/// fading an ink is walking down this — index 0 means the cell is left alone.
const RAMP: [char; 5] = [' ', '░', '▒', '▓', '█'];

/// How a legend byte is painted.
///
/// `dens` is the ink's place on [`RAMP`] — 4 for a solid fill, 2 for the soft
/// edge that rounds a shoulder — or 0 for a *shape*: a glyph whose identity is
/// its outline (a goggle rim, an eye, a corner of a mouth) and which therefore
/// carries `ch` unchanged and fades by colour alone.
#[derive(Debug, Clone, Copy)]
struct Ink {
    ch: char,
    rgb: Rgb,
    dens: u8,
}

impl Ink {
    /// A shape: drawn as itself, faded by colour.
    const fn shape(ch: char, rgb: Rgb) -> Self {
        Self { ch, rgb, dens: 0 }
    }

    /// A fill at the given place on [`RAMP`].
    const fn fill(dens: u8, rgb: Rgb) -> Self {
        Self {
            ch: RAMP[dens as usize],
            rgb,
            dens,
        }
    }
}

/// Minion yellow, and the darker shade the overall straps sit on.
const SKIN: Rgb = (250, 205, 66);
const SKIN_DARK: Rgb = (206, 158, 40);
/// The goggle strap, and the brighter metal of the rim.
const STRAP: Rgb = (92, 98, 108);
const RIM: Rgb = (198, 204, 214);
/// Eye white, and the mouth's dark interior.
const EYE: Rgb = (246, 246, 242);
const MOUTH: Rgb = (84, 52, 36);
/// Denim, and the pale stitching of the pocket.
const DENIM: Rgb = (48, 92, 164);
const STITCH: Rgb = (150, 190, 238);
/// Gloves and shoes. Black in the film; a dark slate here, because true black
/// on a black sky is a minion with no hands.
const LEATHER: Rgb = (58, 58, 70);
/// The hair, and the night everything fades into.
const HAIR: Rgb = (78, 70, 60);
const NIGHT: Rgb = (0, 0, 0);
/// What the water pulls a reflection toward, and the lighter blue a ripple
/// catches on its way past.
const WATER: Rgb = (16, 30, 62);
const RIPPLE: Rgb = (54, 88, 150);

/// The ink table, in [`codec::LEGEND`] order — the byte at index `i` in the
/// sheet is painted by `INKS[i]`. A test holds the two in step.
///
/// Index 0 is the transparent cell and is never painted; it is here so the
/// table can be indexed by ink code without an offset.
const INKS: [Ink; 30] = [
    Ink::fill(0, NIGHT),     // .  nothing
    Ink::fill(4, SKIN),      // Y  body
    Ink::fill(4, SKIN_DARK), // y  body, in shadow
    Ink::fill(2, SKIN),      // s  body, soft edge
    Ink::fill(1, SKIN),      // t  body, faint edge
    Ink::fill(4, STRAP),     // S  goggle strap
    Ink::shape('╭', RIM),    // [
    Ink::shape('╮', RIM),    // ]
    Ink::shape('╰', RIM),    // {
    Ink::shape('╯', RIM),    // }
    Ink::shape('│', RIM),    // |
    Ink::shape('─', RIM),    // -
    Ink::fill(4, EYE),       // W  eye white
    Ink::shape('◉', EYE),    // e  open eye
    Ink::shape('─', MOUTH),  // c  closed eye
    Ink::shape('^', MOUTH),  // ^  eye squeezed shut
    Ink::shape('╰', MOUTH),  // <  smile, left corner
    Ink::shape('─', MOUTH),  // =  smile
    Ink::shape('╯', MOUTH),  // >  smile, right corner
    Ink::shape('▗', MOUTH),  // 1  open mouth, upper row
    Ink::shape('▄', MOUTH),  // 2
    Ink::shape('▖', MOUTH),  // 3
    Ink::shape('▝', MOUTH),  // 4  open mouth, lower row
    Ink::shape('▀', MOUTH),  // 5
    Ink::shape('▘', MOUTH),  // 6
    Ink::fill(4, DENIM),     // B  overalls
    Ink::fill(2, STITCH),    // p  pocket stitching
    Ink::fill(4, LEATHER),   // G  glove
    Ink::fill(4, LEATHER),   // F  shoe
    Ink::shape('╵', HAIR),   // h  hair
];

/// The two inks the code reaches for by hand rather than through the art: the
/// arm that comes out of a sprite during the nudge, and the glove on the end
/// of it. Named because an index into [`INKS`] is unreadable and a wrong one
/// paints a blue elbow. A test ties them back to their legend bytes.
const INK_BODY: usize = 1;
const INK_GLOVE: usize = 27;

/// Below this, an ink is dropped instead of painted.
///
/// Not a performance tweak: over live model output every glyph *replaces* the
/// character under it, so a cell too faint to see would silently punch a hole
/// in the text. Dropping it is both the honest thing and the transparent one.
const FAINTEST: f32 = 0.34;

/// Paints one ink at `alpha`, or nothing when there is not enough of it left.
///
/// Fills lose coverage *and* colour: the ramp is the alpha, and pulling the
/// colour along keeps a half-covered cell from reading as a hard dotted edge.
/// Shapes only lose colour — a goggle rim at quarter coverage is not a fainter
/// rim, it is a different glyph.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // 0..=4 by construction
fn paint(ink: Ink, alpha: f32, toward: Rgb) -> Option<(char, Rgb)> {
    let alpha = alpha.clamp(0.0, 1.0);
    if ink.dens == 0 {
        return (alpha >= FAINTEST).then(|| (ink.ch, lerp_rgb(toward, ink.rgb, alpha)));
    }
    let level = (f32::from(ink.dens) * alpha).round() as usize;
    (level > 0).then(|| (RAMP[level.min(4)], lerp_rgb(toward, ink.rgb, alpha)))
}

// ---------------------------------------------------------------------------
// The skit
// ---------------------------------------------------------------------------

/// Poses in the sheet, in the order `minions.txt` lists them.
const KEVIN_IDLE: usize = 0;
const KEVIN_BLINK: usize = 1;
const KEVIN_LAUGH: usize = 2;
const STUART_IDLE: usize = 3;
const STUART_BLINK: usize = 4;
const STUART_LAUGH: usize = 5;

/// The sheet's size again, as screen offsets rather than grid indices.
///
/// The two never meet — one indexes the art, the other places it — and naming
/// both beats converting at a dozen call sites.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // 15 and 16
const SHEET_WI: i32 = SHEET_W as i32;
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
const SHEET_HI: i32 = SHEET_H as i32;
/// The pair stands shoulder to shoulder, overlapping by a column.
const PAIR_W: i32 = SHEET_WI * 2 - 1;
/// The row the overalls start on. A lean pivots here: heads and shoulders
/// swing, legs stay planted, which is the difference between leaning and
/// sliding.
const HIP_ROW: usize = 10;
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // 10
const HIP_ROW_I: i32 = HIP_ROW as i32;

/// How long a blink lasts, and how far apart blinks are dealt.
const BLINK_MS: u64 = 150;
const BLINK_GAP_MIN: u64 = 2_400;
const BLINK_GAP_MAX: u64 = 7_000;

/// How long the pair walks before one of them starts something.
const WALK_MS_MIN: u64 = 4_000;
const WALK_MS_MAX: u64 = 9_000;
/// The nudge itself: elbow in, and the recoil.
const NUDGE_MS: u64 = 700;
/// How long they laugh about it.
const GIGGLE_MS_MIN: u64 = 2_200;
const GIGGLE_MS_MAX: u64 = 3_800;

/// Walking pace, as a fraction of the crossing per second, before the user's
/// multiplier. About a minute to cross an 80-column terminal: this is a
/// screensaver, and the point of the drift is that nothing sits burnt into one
/// place for an hour.
const WALK_PER_SEC: f32 = 0.017;
/// How far the user can push the pace either way, and where it starts.
const SPEED_MIN: f32 = 0.2;
const SPEED_MAX: f32 = 4.0;
const SPEED_DEFAULT: f32 = 1.0;

/// Bob period while walking and while laughing, in milliseconds. Laughing is
/// faster and that is most of what sells it.
const BOB_WALK_MS: f32 = 760.0;
const BOB_GIGGLE_MS: f32 = 260.0;

/// How long the whole scene takes to fade up when the screensaver opens.
const FADE_IN_MS: f32 = 1_400.0;

/// A puff of laughter: how long one lives and how far it rises.
const PUFF_MS: u64 = 1_500;
const PUFF_RISE: f32 = 3.6;
/// How often a puff is let go while they are laughing.
const PUFF_EVERY_MS: u64 = 340;
/// What they say. Minion-speak, and every one of them ASCII: a laughter puff
/// that draws as boxes is worse than no laughter puff.
const PUFFS: [&str; 6] = ["he he", "ha ha", "hi hi!", "bee do", "banana", "poopaye"];
/// The colour a puff is written in, before it starts fading.
const PUFF_INK: Rgb = (236, 228, 206);

/// Stars in the sky, and the two ends of their twinkle.
const STARS: usize = 70;
const STAR_DIM: Rgb = (44, 52, 86);
const STAR_BRIGHT: Rgb = (208, 218, 250);

/// Where the shore sits, as a fraction of the height, and the least height
/// that leaves room for a reflection underneath it.
const SHORE: f32 = 0.74;
const REFLECT_MIN_H: u16 = 21;
/// How strongly the reflection starts, how much of that is left at its deepest
/// row, and how far the ripples shift a row sideways.
const REFLECT_ALPHA: f32 = 0.62;
const REFLECT_DEPTH_FADE: f32 = 0.75;
const REFLECT_WOBBLE: f32 = 1.4;

/// What phase of the skit is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Beat {
    /// Walking along the shore, blinking now and then.
    Walk,
    /// One elbow going in, and the other one noticing.
    Nudge,
    /// Both of them, about it.
    Giggle,
}

/// One rising puff of laughter, in cells relative to the pair.
#[derive(Debug, Clone, Copy)]
struct Puff {
    dx: f32,
    dy: f32,
    born: u64,
    text: &'static str,
}

/// One pose, placed: which one, where, how far it leans and how strongly it is
/// drawn. Named fields rather than nine positional arguments — half of them
/// are bare numbers, and `(x, shore, lean, -bob, true)` says nothing.
#[derive(Debug, Clone, Copy)]
struct Stamp {
    pose: usize,
    x: i32,
    y: i32,
    /// Sideways shift of everything above [`HIP_ROW`]: heads and shoulders
    /// swing, legs stay planted.
    lean: i32,
    /// Mirrored about `y`, into the water.
    mirrored: bool,
    /// Vertical offset for the mirrored copy alone, so the reflection sinks as
    /// the minion rises.
    sink: i32,
    /// How strongly the whole stamp is painted, before depth takes its share.
    fade: f32,
}

/// One star, in fractions of the screen, with its own twinkle phase.
#[derive(Debug, Clone, Copy)]
struct Twinkle {
    x: f32,
    y: f32,
    phase: f32,
}

/// Two minions by the water, walking, nudging each other and laughing.
#[derive(Debug)]
pub struct Skit {
    rng: Rng,
    /// Milliseconds since the skit opened. Drives every phase, so a resize or
    /// a dropped frame changes nothing about where the animation is.
    now: u64,
    /// Where the pair is along its crossing, in `[0, 1)`. Normalized so the
    /// step function needs no terminal size (it is handed only a `dt`) and a
    /// resize re-places the pair instead of stranding it off-screen.
    at: f32,
    /// The user's pace multiplier.
    speed: f32,
    beat: Beat,
    /// When the current beat gives way to the next.
    until: u64,
    /// When each of the two blinks next, and when that blink ends.
    blink_at: [u64; 2],
    puffs: Vec<Puff>,
    /// When the last puff was let go.
    last_puff: u64,
    stars: Vec<Twinkle>,
    /// Night sky and water, or the two of them and nothing else (`r`).
    scenery: bool,
}

impl Skit {
    /// Opens on a pair already walking, a sky already dealt.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let stars = (0..STARS)
            .map(|_| Twinkle {
                x: rng.next_f32(),
                y: rng.next_f32(),
                phase: rng.range(0.0, std::f32::consts::TAU),
            })
            .collect();
        let mut skit = Self {
            now: 0,
            at: rng.next_f32(),
            speed: SPEED_DEFAULT,
            beat: Beat::Walk,
            until: rng.range_ms(WALK_MS_MIN, WALK_MS_MAX),
            blink_at: [0; 2],
            puffs: Vec::new(),
            last_puff: 0,
            stars,
            scenery: true,
            rng,
        };
        skit.blink_at = [skit.next_blink(), skit.next_blink()];
        skit
    }

    /// Current pace, as a multiple of the default walk.
    #[must_use]
    pub const fn speed(&self) -> f32 {
        self.speed
    }

    /// Scales the pace, clamped to what stays watchable.
    pub const fn scale_speed(&mut self, factor: f32) {
        self.speed = (self.speed * factor).clamp(SPEED_MIN, SPEED_MAX);
    }

    /// Whether the sky and the lake are drawn, or only the pair.
    #[must_use]
    pub const fn scenery(&self) -> bool {
        self.scenery
    }

    /// Turns the scenery off — the sensible thing over live model output,
    /// where every star painted costs a character of the text underneath.
    pub const fn toggle_scenery(&mut self) {
        self.scenery = !self.scenery;
    }

    /// When the next blink is due, from now.
    fn next_blink(&mut self) -> u64 {
        self.now + self.rng.range_ms(BLINK_GAP_MIN, BLINK_GAP_MAX)
    }

    /// Advances the skit by `dt_ms`.
    #[allow(clippy::cast_precision_loss)] // dt is bounded by MAX_STEP_MS
    pub fn step(&mut self, dt_ms: u64) {
        let dt = dt_ms.min(MAX_STEP_MS);
        self.now += dt;

        if self.now >= self.until {
            self.advance_beat();
        }
        // They stop to laugh. Walking through the punchline reads as two
        // people who are not actually talking to each other.
        if self.beat == Beat::Walk {
            let travel = WALK_PER_SEC * self.speed * (dt as f32 / 1000.0);
            self.at = (self.at + travel).fract();
            for i in 0..2 {
                if self.now >= self.blink_at[i] + BLINK_MS {
                    self.blink_at[i] = self.next_blink();
                }
            }
        }
        if self.beat == Beat::Giggle && self.now >= self.last_puff + PUFF_EVERY_MS {
            self.spawn_puff();
        }
        let now = self.now;
        self.puffs.retain(|p| now < p.born + PUFF_MS);
    }

    /// Moves to whatever comes after the beat that just ran out.
    fn advance_beat(&mut self) {
        let (next, len) = match self.beat {
            Beat::Walk => (Beat::Nudge, NUDGE_MS),
            Beat::Nudge => (
                Beat::Giggle,
                self.rng.range_ms(GIGGLE_MS_MIN, GIGGLE_MS_MAX),
            ),
            Beat::Giggle => (Beat::Walk, self.rng.range_ms(WALK_MS_MIN, WALK_MS_MAX)),
        };
        self.beat = next;
        self.until = self.now + len;
        if next == Beat::Walk {
            // Nobody blinks mid-laugh — the eyes are already shut — so the
            // timers are re-dealt on the way out rather than left in the past.
            self.blink_at = [self.next_blink(), self.next_blink()];
        }
    }

    /// Lets go of one puff of laughter, over one head or the other.
    fn spawn_puff(&mut self) {
        let over_kevin = self.rng.next_u64().is_multiple_of(2);
        let pick = self.rng.next_u64() % u64::try_from(PUFFS.len()).unwrap_or(1);
        let text = PUFFS[usize::try_from(pick).unwrap_or(0)];
        self.puffs.push(Puff {
            // Over one head or the other: the heads sit around columns 7 and
            // 21 of the pair, and a puff leans out from there.
            dx: if over_kevin {
                self.rng.range(1.0, 8.0)
            } else {
                self.rng.range(15.0, 22.0)
            },
            dy: self.rng.range(-1.5, 0.5),
            born: self.now,
            text,
        });
        self.last_puff = self.now;
    }

    /// Whether the given minion is mid-blink.
    fn blinking(&self, who: usize) -> bool {
        (self.blink_at[who]..self.blink_at[who] + BLINK_MS).contains(&self.now)
    }

    /// The pose each of the two is holding right now.
    fn poses(&self) -> (usize, usize) {
        match self.beat {
            Beat::Giggle => (KEVIN_LAUGH, STUART_LAUGH),
            // The nudge lands before the laugh: he is still deadpan while the
            // elbow goes in, which is the joke.
            Beat::Nudge => (KEVIN_IDLE, STUART_LAUGH),
            Beat::Walk => (
                if self.blinking(0) {
                    KEVIN_BLINK
                } else {
                    KEVIN_IDLE
                },
                if self.blinking(1) {
                    STUART_BLINK
                } else {
                    STUART_IDLE
                },
            ),
        }
    }

    /// How far into the current beat we are, in `[0, 1]`.
    #[allow(clippy::cast_precision_loss)] // beats are seconds, not eons
    fn through_beat(&self, len: u64) -> f32 {
        let started = self.until.saturating_sub(len);
        ((self.now.saturating_sub(started)) as f32 / len as f32).clamp(0.0, 1.0)
    }

    /// A phase in radians from a period in milliseconds.
    #[allow(clippy::cast_precision_loss)] // wall time in ms, at f32 precision
    fn phase(&self, period_ms: f32, offset: f32) -> f32 {
        (self.now as f32 / period_ms).mul_add(std::f32::consts::TAU, offset)
    }

    /// The vertical bob of one of the two, in cells.
    #[allow(clippy::cast_possible_truncation)] // a bob is one cell either way
    fn bob(&self, who: usize) -> i32 {
        let (period, reach) = if self.beat == Beat::Giggle {
            (BOB_GIGGLE_MS, 1.0)
        } else {
            (BOB_WALK_MS, 0.6)
        };
        // Half a period apart, so the two never bounce in lockstep.
        let offset = if who == 0 { 0.0 } else { std::f32::consts::PI };
        (self.phase(period, offset).sin() * reach).round() as i32
    }

    /// How far each of the two is leaning, in cells: negative is leftward.
    ///
    /// Stuart's elbow goes in first and Kevin gives way a beat later — the
    /// delay is the whole gag, and it is why this is two numbers and not one.
    fn leans(&self) -> (i32, i32) {
        match self.beat {
            Beat::Nudge => {
                let t = self.through_beat(NUDGE_MS);
                let stuart = if t > 0.15 { -1 } else { 0 };
                let kevin = if t > 0.35 { -1 } else { 0 };
                (kevin, stuart)
            }
            Beat::Giggle => {
                // Rocking, in opposite directions, at half the bob's rate.
                let swing = self.phase(BOB_GIGGLE_MS * 2.0, 0.0).sin();
                let lean = if swing > 0.4 {
                    1
                } else if swing < -0.4 {
                    -1
                } else {
                    0
                };
                (-lean, lean)
            }
            Beat::Walk => (0, 0),
        }
    }

    /// The whole scene, painted onto a `w` x `h` area.
    ///
    /// Nothing here fails on a small terminal: every cell is placed and then
    /// bounds-checked, so a screen too short simply shows the part of the
    /// scene that fits.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // screen coordinates: small integers, bounds-checked in `put`
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let fade = (self.now as f32 / FADE_IN_MS).min(1.0);
        let mut out = Vec::with_capacity(700);

        let shore = (f32::from(h) * SHORE) as i32;
        // Feet on the shore: the sheet's last row is the shoes.
        let top = shore - SHEET_HI;
        let x0 = (self.at * (f32::from(w) + PAIR_W as f32)) as i32 - PAIR_W;

        if self.scenery {
            self.sky(&mut out, w, h, shore, fade);
            self.water(&mut out, w, h, shore, fade);
        }

        let (kevin, stuart) = self.poses();
        let (kevin_lean, stuart_lean) = self.leans();
        let pair = [
            (kevin, x0, self.bob(0), kevin_lean),
            (stuart, x0 + SHEET_WI - 1, self.bob(1), stuart_lean),
        ];

        if self.scenery && h >= REFLECT_MIN_H {
            for (pose, x, bob, lean) in pair {
                // The reflection rises when the minion sinks: the water is a
                // mirror, and a reflection that bobbed along with it would
                // read as a second minion standing in the lake.
                let stamp = Stamp {
                    pose,
                    x,
                    y: shore,
                    lean,
                    mirrored: true,
                    sink: -bob,
                    fade,
                };
                self.stamp(&mut out, &stamp, w, h);
            }
        }
        for (pose, x, bob, lean) in pair {
            let stamp = Stamp {
                pose,
                x,
                y: top + bob,
                lean,
                mirrored: false,
                sink: 0,
                fade,
            };
            self.stamp(&mut out, &stamp, w, h);
        }
        self.laughter(&mut out, x0, top, w, h, fade);
        out
    }

    /// Stamps one pose.
    ///
    /// Mirrored, it becomes the reflection: fainter with depth, pulled toward
    /// the water, and shifted row by row by the ripples.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn stamp(&self, out: &mut Vec<Glyph>, s: &Stamp, w: u16, h: u16) {
        let cells = &sheet()[s.pose * POSE_CELLS..(s.pose + 1) * POSE_CELLS];
        let ripple = self.phase(2_600.0, 0.0);
        // Each row is wanted twice over: as an index into the art and as an
        // offset on the screen. `enumerate` hands out both.
        for (row, row_i) in (0..SHEET_HI).enumerate() {
            // Depth below the shore, for the reflection: 0 at the waterline.
            let depth = SHEET_HI - 1 - row_i;
            let (sy, alpha, toward) = if s.mirrored {
                let fall = depth as f32 / SHEET_HI as f32;
                (
                    s.y + depth + s.sink,
                    s.fade * REFLECT_ALPHA * fall.mul_add(-REFLECT_DEPTH_FADE, 1.0),
                    WATER,
                )
            } else {
                (s.y + row_i, s.fade, NIGHT)
            };
            let wobble = if s.mirrored {
                ((ripple + depth as f32 * 0.8).sin() * REFLECT_WOBBLE).round() as i32
            } else {
                0
            };
            let swing = if row < HIP_ROW { s.lean } else { 0 };
            for (col, col_i) in (0..SHEET_WI).enumerate() {
                let ink = cells[row * SHEET_W + col];
                if ink == 0 {
                    continue;
                }
                let Some((ch, rgb)) = paint(INKS[ink as usize], alpha, toward) else {
                    continue;
                };
                put(out, s.x + col_i + swing + wobble, sy, ch, rgb, w, h);
            }
        }
        if self.beat == Beat::Nudge && s.pose == STUART_LAUGH && !s.mirrored {
            self.elbow(out, s.x, s.y, w, h, s.fade);
        }
    }

    /// The elbow itself: two cells of arm reaching out of Stuart's left side
    /// into Kevin's ribs, drawn only for the fraction of a second it is in.
    ///
    /// Procedural rather than a seventh pose: a pose costs a grid, and this
    /// costs two cells and a line of arithmetic.
    #[allow(clippy::cast_possible_truncation)]
    fn elbow(&self, out: &mut Vec<Glyph>, x: i32, y: i32, w: u16, h: u16, fade: f32) {
        let through = self.through_beat(NUDGE_MS);
        if !(0.15..0.75).contains(&through) {
            return;
        }
        let row = y + HIP_ROW_I + 1;
        for (dx, ink) in [(-1, INKS[INK_BODY]), (-2, INKS[INK_GLOVE])] {
            if let Some((ch, rgb)) = paint(ink, fade, NIGHT) {
                put(out, x + dx, row, ch, rgb, w, h);
            }
        }
    }

    /// The stars, twinkling above the shore.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn sky(&self, out: &mut Vec<Glyph>, w: u16, h: u16, shore: i32, fade: f32) {
        if shore <= 0 {
            return;
        }
        let spin = self.phase(3_400.0, 0.0);
        for star in &self.stars {
            let bright = (spin + star.phase).sin().mul_add(0.5, 0.5);
            // A star is a shape: no coverage to lose, so it fades by colour and
            // then stops being drawn at all.
            let alpha = fade * bright.mul_add(0.7, 0.3);
            if alpha < FAINTEST {
                continue;
            }
            let ch = if bright > 0.82 { '+' } else { '·' };
            let rgb = lerp_rgb(STAR_DIM, STAR_BRIGHT, bright);
            let x = (star.x * f32::from(w)) as i32;
            let y = (star.y * shore as f32) as i32;
            put(out, x, y, ch, lerp_rgb(NIGHT, rgb, alpha), w, h);
        }
    }

    /// The lake: a waterline, and ripples drifting across it.
    ///
    /// The ripples are placed by a hash of the row and the drift rather than by
    /// the [`Rng`], because they have to be a pure function of the clock — the
    /// scene has no per-ripple state, and a screensaver may run for hours.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn water(&self, out: &mut Vec<Glyph>, w: u16, h: u16, shore: i32, fade: f32) {
        if shore >= i32::from(h) {
            return;
        }
        if let Some((ch, rgb)) = paint(Ink::fill(1, WATER), fade, NIGHT) {
            for x in 0..i32::from(w) {
                put(out, x, shore, ch, rgb, w, h);
            }
        }
        let drift = self.now / 220;
        for y in (shore + 1)..i32::from(h) {
            let depth = (y - shore) as f32 / f32::from(h).max(1.0);
            let alpha = fade * 0.3f32.mul_add(-depth, 0.6);
            let Some((ch, rgb)) = paint(Ink::shape('~', RIPPLE), alpha, NIGHT) else {
                continue;
            };
            for nth in 0..=(i32::from(w) / 14) {
                // Hashed rather than dealt: the lake has no per-ripple state to
                // keep, and a screensaver may be up for hours. It has to be a
                // *mixing* hash — the obvious `row * a + i * b` puts every
                // ripple of a row within a cell or two of the last one, which
                // reads as a dashed line rather than water.
                let key = u64::from(y.unsigned_abs()) << 8 | u64::from(nth.unsigned_abs());
                let x = i32::try_from(scatter(key).wrapping_add(drift) % u64::from(w)).unwrap_or(0);
                put(out, x, y, ch, rgb, w, h);
            }
        }
    }

    /// The laughter, rising and thinning out.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn laughter(&self, out: &mut Vec<Glyph>, x0: i32, top: i32, w: u16, h: u16, fade: f32) {
        for puff in &self.puffs {
            let age = (self.now.saturating_sub(puff.born)) as f32 / PUFF_MS as f32;
            // Ease out: a puff leaves quickly and lingers as it goes faint,
            // which is what smoke does and what a hard linear fade does not.
            let alpha = fade * (1.0 - age * age);
            if alpha < FAINTEST {
                continue;
            }
            let y = top + (puff.dy - age * PUFF_RISE).round() as i32;
            let x = x0 + puff.dx.round() as i32;
            for (i, ch) in puff.text.chars().enumerate() {
                if ch == ' ' {
                    continue;
                }
                put(
                    out,
                    x + i32::try_from(i).unwrap_or(0),
                    y,
                    ch,
                    lerp_rgb(NIGHT, PUFF_INK, alpha),
                    w,
                    h,
                );
            }
        }
    }
}

/// Spreads consecutive inputs across the whole range: the finalizer of
/// splitmix64, which is all that is needed to place a ripple.
const fn scatter(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Places one glyph, dropping it when it falls outside the area.
fn put(out: &mut Vec<Glyph>, x: i32, y: i32, ch: char, color: Rgb, w: u16, h: u16) {
    if x < 0 || y < 0 || x >= i32::from(w) || y >= i32::from(h) {
        return;
    }
    out.push(Glyph {
        x: u16::try_from(x).unwrap_or(0),
        y: u16::try_from(y).unwrap_or(0),
        ch,
        color,
    });
}

/// Milliseconds in a range, dealt from the field's own generator.
///
/// An extension trait rather than an inherent `impl`: `Rng` now lives in the
/// shared support crate, so it cannot be extended here directly — and it should
/// not be, since this is a skit's idea of a duration and not something every
/// guest needs.
trait RangeMs {
    fn range_ms(&mut self, lo: u64, hi: u64) -> u64;
}

impl RangeMs for Rng {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // ranges are ms, and small
    fn range_ms(&mut self, lo: u64, hi: u64) -> u64 {
        lo + (self.next_u64() % (hi - lo + 1))
    }
}
