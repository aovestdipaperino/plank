// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! What a face needs from the host that a guest has to bring itself.
//!
//! These are ports, not reimplementations: [`Rng`] is `arcade::Rng` and the
//! colour helpers are `anim`'s, copied because a guest cannot link plank. They
//! must stay byte-identical in behaviour — a face's whole testability rests on
//! "the same seed rains the same way", and a subtly different `next_f32` would
//! break that quietly, in a way no test on either side would catch.

/// An RGB triple, as the glyph wire format carries it.
pub type Rgb = (u8, u8, u8);

/// One glyph to paint.
#[derive(Debug, Clone, Copy)]
pub struct Glyph {
    pub x: u16,
    pub y: u16,
    pub ch: char,
    pub color: Rgb,
}

/// Longest `dt` a face integrates in one step.
///
/// The host clamps this too, and deliberately so: the host's clamp protects it
/// from a runaway guest, and this one keeps a face's own physics sane if it is
/// ever driven by something else. Same constant, two independent reasons.
pub const MAX_STEP_MS: u64 = 100;

/// Seeded xorshift — `arcade::Rng`, ported.
///
/// `Debug` because the faces derive it: plank lints for
/// `missing_debug_implementations`, and a face carried across should not have
/// to be edited to satisfy a lint the guest does not run.
///
/// A guest gets no ambient randomness, which is what makes a seeded frame
/// replayable. Every face draws from here and nowhere else.
#[derive(Debug)]
pub struct Rng(u64);

impl Rng {
    /// Seeds the generator. Zero is replaced (xorshift is stuck at zero).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// Next raw 64-bit value.
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Next value in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Take the top 24 bits: exactly the f32 mantissa width, so the divide
        // is lossless and the result never rounds up to 1.0.
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }

    /// Next value in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// Linear interpolation between two channel values.
#[must_use]
pub fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let t = t.clamp(0.0, 1.0);
    let a = f32::from(a);
    let b = f32::from(b);
    let v = (a + (b - a) * t).round().clamp(0.0, 255.0) as u8;
    v
}

/// Linear interpolation between two colours.
#[must_use]
pub fn lerp_rgb(a: Rgb, b: Rgb, t: f32) -> Rgb {
    (
        lerp_u8(a.0, b.0, t),
        lerp_u8(a.1, b.1, t),
        lerp_u8(a.2, b.2, t),
    )
}

/// Packs glyphs into the `PGLY` buffer `frame_step` returns.
///
/// Written here rather than pulled from the host: a guest is on the far side
/// of the ABI by definition, and the format has to be writable by someone who
/// has only the spec. If this and `wasmglyph::decode` ever disagree, the ABI
/// is what is wrong, not this file.
#[must_use]
pub fn encode(glyphs: &[Glyph], w: u16, h: u16) -> Vec<u8> {
    // The count field is a u16, so a face that somehow emitted more than that
    // is truncated here rather than writing a count that disagrees with the
    // body — which would fail to decode for reasons its author could not see.
    let count = u16::try_from(glyphs.len()).unwrap_or(u16::MAX);
    let mut out = Vec::with_capacity(12 + usize::from(count) * 12);
    out.extend_from_slice(b"PGLY");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    for g in glyphs.iter().take(usize::from(count)) {
        out.extend_from_slice(&g.x.to_le_bytes());
        out.extend_from_slice(&g.y.to_le_bytes());
        out.extend_from_slice(&(g.ch as u32).to_le_bytes());
        out.extend_from_slice(&[g.color.0, g.color.1, g.color.2, 0]);
    }
    out
}

/// Reads a numeric field out of a flat JSON payload.
///
/// The host's payloads are flat string maps by design, so a full parser would
/// be several kilobytes of guest for no benefit. Anything absent or malformed
/// reads as `0.0`, which every caller treats as "not given".
#[must_use]
pub fn num(input: &str, key: &str) -> f32 {
    input
        .split_once(&format!("\"{key}\":"))
        .and_then(|(_, rest)| {
            rest.trim_start()
                .split([',', '}'])
                .next()
                .and_then(|n| n.trim().trim_matches('"').parse::<f32>().ok())
        })
        .unwrap_or(0.0)
}

/// Reads an integer field out of a flat JSON payload.
///
/// Separate from [`num`] because a seed is a u64 and `f32` has 24 bits of
/// mantissa: routing a seed through `num` silently rounds it, and the face
/// then rains a *different* rain than the one the host asked for. Nothing
/// visible goes wrong — it is still rain — which is exactly what makes the
/// bug survive: only a glyph-for-glyph comparison against the built-in face
/// catches it.
#[must_use]
pub fn int(input: &str, key: &str) -> u64 {
    input
        .split_once(&format!("\"{key}\":"))
        .and_then(|(_, rest)| {
            rest.trim_start()
                .split([',', '}'])
                .next()
                .and_then(|n| n.trim().trim_matches('"').parse::<u64>().ok())
        })
        .unwrap_or(0)
}

/// Reads a string field out of a flat JSON payload.
#[must_use]
pub fn text(input: &str, key: &str) -> String {
    input
        .split_once(&format!("\"{key}\":"))
        .and_then(|(_, rest)| rest.trim_start().strip_prefix('"'))
        .and_then(|rest| rest.split('"').next())
        .unwrap_or("")
        .to_string()
}
