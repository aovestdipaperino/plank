//! The NVFP4 block format, decoded exactly as upstream ggml does it
//! (`dequantize_row_nvfp4` and `ggml_ue4m3_to_fp32`, llama.cpp `970b51c`).
//!
//! A block holds 64 weights in 36 bytes: four UE4M3 scales, one for each run
//! of 16 values, then 32 bytes of 4-bit E2M1 codes. Within run `s`, byte `j`
//! carries value `j` in its low nibble and value `j + 8` in its high nibble.
//! The code table is E2M1 doubled, and the scale decode halves to match.

/// Values per block.
pub const QK: usize = 64;
/// Values per scale.
pub const QK_SUB: usize = 16;
/// Scales per block.
pub const SUBS: usize = QK / QK_SUB;
/// Bytes per block: the scales, then the packed codes.
pub const BLOCK_BYTES: usize = SUBS + QK / 2;

/// E2M1 values, doubled (`kvalues_fp4` upstream).
pub const KVALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Decodes a UE4M3 scale byte, halved to pair with the doubled table.
///
/// 4 exponent bits (bias 7), 3 mantissa bits, no sign. `0x00` and the NaN
/// byte `0x7F` both give 0, and bit 7 is ignored, as upstream.
#[must_use]
pub fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = i32::from((x >> 3) & 0xF);
    let man = f32::from(x & 0x7);
    let raw = if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    raw * 0.5
}

/// Encodes a positive value as UE4M3 (`ggml_fp32_to_ue4m3`): round to
/// nearest on the mantissa, saturating to `0x7E` from exponent 15 up.
/// Non-positive and NaN give 0.
#[must_use]
// The casts mirror the C: exponent and mantissa fit their types by construction.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn f32_to_ue4m3(x: f32) -> u8 {
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    let x = x.min(448.0);
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_man = (bits >> 20) & 0x7;
    let mut exp = fp32_exp + 7;
    if exp <= 0 {
        // Subnormal: value = man * 2^-9.
        let man = ((x * 512.0 + 0.5) as i32).min(7);
        return if man < 1 { 0 } else { man as u8 };
    }
    if exp >= 15 {
        return 0x7E;
    }
    let mut man = fp32_man + ((bits >> 19) & 1);
    if man > 7 {
        man = 0;
        exp += 1;
        if exp >= 15 {
            return 0x7E;
        }
    }
    ((exp as u8) << 3) | man as u8
}

/// Decodes whole blocks into `out`, one value per weight, unscaled by the
/// tensor scale. `blocks` must be whole 36-byte blocks and `out` must hold
/// exactly `blocks.len() / 36 * 64` values; debug builds assert both, and a
/// release build decodes only as many blocks as both sides cover.
pub fn dequant_row(blocks: &[u8], out: &mut [f32]) {
    debug_assert_eq!(blocks.len() % BLOCK_BYTES, 0);
    debug_assert_eq!(out.len(), blocks.len() / BLOCK_BYTES * QK);
    for (blk, ob) in blocks
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK>().0.iter_mut())
    {
        for s in 0..SUBS {
            let d = ue4m3_to_f32(blk[s]);
            let qs = &blk[SUBS + s * QK_SUB / 2..SUBS + (s + 1) * QK_SUB / 2];
            let yb = &mut ob[s * QK_SUB..(s + 1) * QK_SUB];
            for (j, &q) in qs.iter().enumerate() {
                yb[j] = f32::from(KVALUES[usize::from(q & 0x0F)]) * d;
                yb[j + QK_SUB / 2] = f32::from(KVALUES[usize::from(q >> 4)]) * d;
            }
        }
    }
}

/// The dot product of one row of blocks with `x`, unscaled by the tensor
/// scale. Each run of 16 is summed, then multiplied by its scale.
#[must_use]
pub fn dot_row(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (blk, xb) in row
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(x.as_chunks::<QK>().0.iter())
    {
        for s in 0..SUBS {
            let qs = &blk[SUBS + s * QK_SUB / 2..SUBS + (s + 1) * QK_SUB / 2];
            let xs = &xb[s * QK_SUB..(s + 1) * QK_SUB];
            let mut sum = 0.0f32;
            for (j, &q) in qs.iter().enumerate() {
                sum += f32::from(KVALUES[usize::from(q & 0x0F)]) * xs[j]
                    + f32::from(KVALUES[usize::from(q >> 4)]) * xs[j + QK_SUB / 2];
            }
            acc += sum * ue4m3_to_f32(blk[s]);
        }
    }
    acc
}

/// Quantizes whole blocks of `x` (`quantize_row_nvfp4_ref`): per run of 16,
/// the scale maps the largest magnitude onto E2M1's 6, and each value takes
/// the nearest code. Used to build realistic test weights.
#[must_use]
pub fn quantize_row(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / QK * BLOCK_BYTES);
    for xb in x.as_chunks::<QK>().0 {
        let mut blk = [0u8; BLOCK_BYTES];
        for s in 0..SUBS {
            let xs = &xb[s * QK_SUB..(s + 1) * QK_SUB];
            let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let ue = f32_to_ue4m3(amax / 6.0);
            blk[s] = ue;
            let d = ue4m3_to_f32(ue);
            for j in 0..QK_SUB / 2 {
                let lo = best_index(xs[j], d);
                let hi = best_index(xs[QK_SUB / 2 + j], d);
                blk[SUBS + s * (QK_SUB / 2) + j] = lo | (hi << 4);
            }
        }
        out.extend_from_slice(&blk);
    }
    out
}

/// The code whose value is nearest `x` at scale `d`, first one on ties
/// (`best_index_mxfp4`).
fn best_index(x: f32, d: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = (f32::from(KVALUES[0]) * d - x).abs();
    for i in 1..16u8 {
        let e = (f32::from(KVALUES[usize::from(i)]) * d - x).abs();
        if e < best_err {
            best = i;
            best_err = e;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue4m3_decodes_the_upstream_corner_values() {
        assert_eq!(ue4m3_to_f32(0x00), 0.0);
        assert_eq!(
            ue4m3_to_f32(0x7F),
            0.0,
            "the E4M3 NaN byte decodes to 0 upstream"
        );
        assert_eq!(ue4m3_to_f32(0x38), 0.5, "exp 7, man 0 is 1.0, halved");
        assert_eq!(ue4m3_to_f32(0x3C), 0.75);
        assert_eq!(
            ue4m3_to_f32(0x01),
            1.0 / 1024.0,
            "smallest subnormal, halved"
        );
        assert_eq!(ue4m3_to_f32(0x07), 7.0 / 1024.0);
        assert_eq!(ue4m3_to_f32(0x08), 1.0 / 128.0, "smallest normal, halved");
        assert_eq!(ue4m3_to_f32(0x7E), 224.0, "largest finite, halved");
        assert_eq!(ue4m3_to_f32(0xB8), 0.5, "bit 7 is ignored");
    }

    #[test]
    fn ue4m3_round_trips_below_the_saturating_exponent() {
        // Upstream's encoder saturates exponent 15 to 0x7E, so only 0x01..0x77
        // round-trip. The decoder halves, so the encoder gets twice its output.
        for x in 1u8..0x78 {
            assert_eq!(f32_to_ue4m3(2.0 * ue4m3_to_f32(x)), x, "byte {x:#04x}");
        }
        assert_eq!(f32_to_ue4m3(0.0), 0);
        assert_eq!(f32_to_ue4m3(-1.0), 0);
        assert_eq!(f32_to_ue4m3(f32::NAN), 0);
        assert_eq!(f32_to_ue4m3(1000.0), 0x7E);
    }

    /// One block whose scales are all 0x38 (0.5) and whose sub-block `s`
    /// byte `j` is `j | (j + 8) << 4`.
    fn known_block() -> [u8; BLOCK_BYTES] {
        let mut b = [0u8; BLOCK_BYTES];
        b[..SUBS].fill(0x38);
        for s in 0..SUBS {
            for j in 0..8u8 {
                b[SUBS + s * 8 + usize::from(j)] = j | ((j + 8) << 4);
            }
        }
        b
    }

    #[test]
    fn dequant_places_low_nibbles_first_and_high_nibbles_eight_later() {
        let mut out = [0f32; QK];
        dequant_row(&known_block(), &mut out);
        for s in 0..SUBS {
            for j in 0..8 {
                assert_eq!(out[s * 16 + j], f32::from(KVALUES[j]) * 0.5);
                assert_eq!(out[s * 16 + j + 8], f32::from(KVALUES[j + 8]) * 0.5);
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // i is below 128, exact in f32
    fn dot_row_equals_dequant_then_dot() {
        let mut row = known_block().to_vec();
        row.extend_from_slice(&known_block());
        row[1] = 0x45; // vary one scale
        let x: Vec<f32> = (0..2 * QK).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut w = vec![0f32; 2 * QK];
        dequant_row(&row, &mut w);
        let want: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((dot_row(&row, &x) - want).abs() <= 1e-5 * want.abs().max(1.0));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // values are below 211, exact in f32
    fn quantize_then_dequantize_stays_within_one_grid_step() {
        let x: Vec<f32> = (0..4 * QK)
            .map(|i| ((i * 7919) % 211) as f32 / 105.0 - 1.0)
            .collect();
        let q = quantize_row(&x);
        assert_eq!(q.len(), 4 * BLOCK_BYTES);
        let mut back = vec![0f32; x.len()];
        dequant_row(&q, &mut back);
        for (sub, (xs, bs)) in x.chunks(QK_SUB).zip(back.chunks(QK_SUB)).enumerate() {
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            // The widest E2M1 gap (4..6) is 2 * amax/6, so the nearest grid
            // point is within amax/6; UE4M3 rounding of the scale adds < 7%.
            let bound = amax / 6.0 * 1.07 + 1e-6;
            for (a, b) in xs.iter().zip(bs) {
                assert!(
                    (a - b).abs() <= bound,
                    "sub-block {sub}: {a} came back as {b}"
                );
            }
        }
    }
}
