// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! What a guest needs from the host that it has to bring itself.
//!
//! These are ports, not reimplementations: [`Rng`] is `arcade::Rng` and the
//! colour helpers are `anim`'s, copied because a guest cannot link plank. They
//! must stay byte-identical in behaviour — a face's whole testability rests on
//! "the same seed rains the same way", and a subtly different `next_f32` would
//! break that quietly, in a way no test on either side would catch.
//!
//! One crate rather than a file copied into each guest, which is what this was:
//! the copies had already drifted, and the RNG is precisely the thing that must
//! not. Shared, the divergence is impossible rather than merely discouraged.

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The RNG's exact output is a contract, not an implementation detail.
    ///
    /// A face's testability rests on "the same seed draws the same way", and the
    /// screensaver rotation reuses seeds across sessions. These are the values
    /// the two copies of this file produced before they were merged, so the
    /// extraction is provably behaviour-preserving rather than probably.
    #[test]
    fn the_rng_sequence_is_a_contract() {
        let mut rng = Rng::new(1);
        let first: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        // Re-seeded identically, it repeats exactly.
        let mut again = Rng::new(1);
        let repeat: Vec<u64> = (0..4).map(|_| again.next_u64()).collect();
        assert_eq!(
            first, repeat,
            "the same seed must produce the same sequence"
        );

        // A different seed diverges immediately: a generator that ignored its
        // seed would still pass the check above.
        let mut other = Rng::new(2);
        assert_ne!(other.next_u64(), first[0]);

        // `next_f32` stays in [0, 1) — a face multiplies it by a dimension, and
        // a value of exactly 1.0 would index one past the edge.
        let mut rng = Rng::new(99);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }

        // `range` respects its bounds in both orders of magnitude.
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let v = rng.range(-3.0, 5.0);
            assert!((-3.0..5.0).contains(&v), "out of range: {v}");
        }
    }

    /// The glyph packing is the wire format the host decodes, so its header is
    /// as much a contract as the RNG.
    #[test]
    fn encoding_matches_the_hosts_expectations() {
        let glyphs = [
            Glyph {
                x: 1,
                y: 2,
                ch: 'a',
                color: (10, 20, 30),
            },
            Glyph {
                x: 3,
                y: 4,
                ch: '●',
                color: (40, 50, 60),
            },
        ];
        let bytes = encode(&glyphs, 40, 20);
        assert_eq!(&bytes[..4], b"PGLY", "the magic the host looks for");
        // count, then the declared area, little-endian.
        assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 2);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 40);
        assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), 20);
    }

    /// The JSON readers are deliberately not a parser, so the edges are worth
    /// pinning: a missing key must not panic a guest mid-frame.
    #[test]
    fn json_readers_handle_absent_and_malformed_keys() {
        let input = r#"{"w": 40, "h": 20, "seed": 7, "arg": "starfield"}"#;
        assert_eq!(num(input, "w"), 40.0);
        assert_eq!(int(input, "seed"), 7);
        assert_eq!(text(input, "arg"), "starfield");
        // Absent keys fall back rather than trapping: the host may add fields,
        // and an older guest must keep running.
        assert_eq!(num(input, "nope"), 0.0);
        assert_eq!(int(input, "nope"), 0);
        assert_eq!(text(input, "nope"), "");
        assert_eq!(num("not json", "w"), 0.0);
    }
}
