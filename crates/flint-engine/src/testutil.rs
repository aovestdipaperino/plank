//! Shared test helpers: a seeded RNG, random weights, and the naive oracles
//! every kernel is checked against.

use std::sync::Arc;

use crate::nvfp4::block::quantize_row;
use crate::nvfp4::cpu::silu;
use crate::nvfp4::tensor::{MoeExperts, NvWeight};

/// `SplitMix64`, as in gemma-engine's sampler: reproducible, no dependency.
pub(crate) struct Rng(u64);

// Converting 24 random bits to f32 is exact.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    pub(crate) fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.uniform()).collect()
    }

    pub(crate) fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

/// A realistic weight: random values quantized row by row, a random scale
/// in `[0.5, 1.5)` per expert.
pub(crate) fn weight(rng: &mut Rng, cols: usize, rows: usize, experts: usize) -> NvWeight {
    let data: Vec<u8> = (0..rows * experts)
        .flat_map(|_| quantize_row(&rng.vec(cols)))
        .collect();
    let scales = (0..experts).map(|_| 1.0 + 0.5 * rng.uniform()).collect();
    NvWeight::owned(cols, rows, experts, data, scales).unwrap()
}

/// Every `got[i]` within `rel` of `want[i]`, relative to the largest `|want|`.
pub(crate) fn assert_close(got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= rel * scale,
            "index {i}: got {g}, want {w} (scale {scale})"
        );
    }
}

/// `x [tokens, cols]` times the dequantized expert, accumulated in f64.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn naive_matmul(w: &NvWeight, expert: usize, x: &[f32], tokens: usize) -> Vec<f32> {
    let m = w.dequant(expert);
    let mut y = vec![0f32; tokens * w.rows()];
    for t in 0..tokens {
        let xt = &x[t * w.cols()..(t + 1) * w.cols()];
        for r in 0..w.rows() {
            let row = &m[r * w.cols()..(r + 1) * w.cols()];
            y[t * w.rows() + r] = row
                .iter()
                .zip(xt)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>() as f32;
        }
    }
    y
}

/// The `MoE` layer, written out with `naive_matmul`.
// Same conventional single-letter names as the kernel it checks.
#[allow(clippy::many_single_char_names)]
pub(crate) fn naive_moe(
    m: &MoeExperts,
    x: &[f32],
    tokens: usize,
    ids: &[u32],
    route: &[f32],
    k_used: usize,
) -> Vec<f32> {
    let d = m.hidden();
    let mut y = vec![0f32; tokens * d];
    for t in 0..tokens {
        let xt = &x[t * d..(t + 1) * d];
        for j in 0..k_used {
            let e = ids[t * k_used + j] as usize;
            let g = naive_matmul(m.gate(), e, xt, 1);
            let u = naive_matmul(m.up(), e, xt, 1);
            let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            let o = naive_matmul(m.down(), e, &h, 1);
            for (acc, v) in y[t * d..(t + 1) * d].iter_mut().zip(&o) {
                *acc += route[t * k_used + j] * v;
            }
        }
    }
    y
}

/// A random `MoE` layer: `count` experts of hidden size `d` and ffn size `f`.
pub(crate) fn experts(rng: &mut Rng, d: usize, f: usize, count: usize) -> MoeExperts {
    MoeExperts::new(
        Arc::new(weight(rng, d, f, count)),
        Arc::new(weight(rng, d, f, count)),
        Arc::new(weight(rng, f, d, count)),
    )
    .unwrap()
}
