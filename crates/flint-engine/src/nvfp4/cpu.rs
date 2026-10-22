//! The CPU kernels: decode one block at a time in registers, f32 throughout,
//! rows in parallel with rayon. They are also the reference the Metal
//! kernels are tested against.

use rayon::prelude::*;

use crate::nvfp4::block::dot_row;
use crate::nvfp4::tensor::{MoeExperts, NvWeight};
use crate::{Result, err};

/// `x / (1 + e^-x)`.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `y[t][r] = scale(expert) * sum_k W[expert][r][k] * x[t][k]`.
///
/// # Errors
/// `expert` out of range, or `x`/`y` not `[tokens, cols]`/`[tokens, rows]`.
pub fn matmul(w: &NvWeight, expert: usize, x: &[f32], tokens: usize, y: &mut [f32]) -> Result<()> {
    if expert >= w.experts() {
        return Err(err(format!("expert {expert} of {}", w.experts())));
    }
    if x.len() != tokens * w.cols() || y.len() != tokens * w.rows() {
        return Err(err(format!(
            "matmul wants x [{tokens}, {}] and y [{tokens}, {}], got {} and {} values",
            w.cols(),
            w.rows(),
            x.len(),
            y.len()
        )));
    }
    let s = w.scale(expert);
    for (xt, yt) in x.chunks_exact(w.cols()).zip(y.chunks_exact_mut(w.rows())) {
        yt.par_iter_mut()
            .enumerate()
            .for_each(|(r, out)| *out = s * dot_row(w.row(expert, r), xt));
    }
    Ok(())
}

/// The gathered `MoE`: for each token, each of its `k_used` slots runs expert
/// `ids[slot]` as `down(silu(gate x) * up x)` (each output scaled by that
/// expert's scale), multiplies it by `route[slot]`, and the slots are summed
/// in order. `y` is overwritten.
///
/// # Errors
/// Lengths that do not match `tokens`, `k_used` and the expert shapes, or an
/// id past the last expert.
// x, y, h, d, f, t, j, e, r are the conventional names for activations and dims.
#[allow(clippy::many_single_char_names)]
pub fn moe(
    m: &MoeExperts,
    x: &[f32],
    tokens: usize,
    ids: &[u32],
    route: &[f32],
    k_used: usize,
    y: &mut [f32],
) -> Result<()> {
    let (d, f) = (m.hidden(), m.ffn());
    if x.len() != tokens * d
        || y.len() != tokens * d
        || ids.len() != tokens * k_used
        || route.len() != ids.len()
    {
        return Err(err(format!(
            "moe wants x and y [{tokens}, {d}], ids and route [{tokens}, {k_used}]"
        )));
    }
    if let Some(bad) = ids.iter().find(|&&e| e as usize >= m.count()) {
        return Err(err(format!("expert {bad} of {}", m.count())));
    }
    y.fill(0.0);
    let mut h = vec![0f32; f];
    let mut out = vec![0f32; d];
    for t in 0..tokens {
        let xt = &x[t * d..(t + 1) * d];
        for j in 0..k_used {
            let e = ids[t * k_used + j] as usize;
            let (gs, us, ds) = (m.gate().scale(e), m.up().scale(e), m.down().scale(e));
            h.par_iter_mut().enumerate().for_each(|(r, hv)| {
                let g = gs * dot_row(m.gate().row(e, r), xt);
                let u = us * dot_row(m.up().row(e, r), xt);
                *hv = silu(g) * u;
            });
            out.par_iter_mut()
                .enumerate()
                .for_each(|(r, o)| *o = ds * dot_row(m.down().row(e, r), &h));
            let rw = route[t * k_used + j];
            for (acc, v) in y[t * d..(t + 1) * d].iter_mut().zip(&out) {
                *acc += rw * v;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Rng, assert_close, experts, naive_matmul, naive_moe, weight};

    #[test]
    fn matmul_matches_the_dequantized_oracle() {
        let mut rng = Rng::new(1);
        for (cols, rows, tokens) in [(64, 1, 1), (192, 33, 3), (2048, 100, 2)] {
            let w = weight(&mut rng, cols, rows, 1);
            let x = rng.vec(cols * tokens);
            let mut y = vec![0f32; rows * tokens];
            matmul(&w, 0, &x, tokens, &mut y).unwrap();
            assert_close(&y, &naive_matmul(&w, 0, &x, tokens), 1e-5);
        }
    }

    #[test]
    fn matmul_reads_the_right_expert_and_its_scale() {
        let mut rng = Rng::new(2);
        let w = weight(&mut rng, 128, 7, 3);
        let x = rng.vec(128);
        let mut y = vec![0f32; 7];
        matmul(&w, 2, &x, 1, &mut y).unwrap();
        assert_close(&y, &naive_matmul(&w, 2, &x, 1), 1e-5);
    }

    #[test]
    fn moe_matches_the_written_out_layer() {
        let mut rng = Rng::new(3);
        let m = experts(&mut rng, 128, 64, 8);
        let (tokens, k_used) = (3, 3);
        let x = rng.vec(tokens * 128);
        let ids: Vec<u32> = (0..tokens * k_used).map(|_| rng.below(8)).collect();
        let route: Vec<f32> = (0..tokens * k_used).map(|_| rng.uniform().abs()).collect();
        let mut y = vec![0f32; tokens * 128];
        moe(&m, &x, tokens, &ids, &route, k_used, &mut y).unwrap();
        assert_close(&y, &naive_moe(&m, &x, tokens, &ids, &route, k_used), 1e-4);
    }

    #[test]
    fn bad_shapes_and_ids_are_errors() {
        let mut rng = Rng::new(4);
        let w = weight(&mut rng, 64, 2, 1);
        assert!(
            matmul(&w, 1, &[0.0; 64], 1, &mut [0.0; 2]).is_err(),
            "expert out of range"
        );
        assert!(
            matmul(&w, 0, &[0.0; 63], 1, &mut [0.0; 2]).is_err(),
            "short input"
        );
        assert!(
            matmul(&w, 0, &[0.0; 64], 1, &mut [0.0; 3]).is_err(),
            "wrong output"
        );
        let m = experts(&mut rng, 64, 64, 2);
        let mut y = vec![0f32; 64];
        assert!(
            moe(&m, &[0.0; 64], 1, &[2], &[1.0], 1, &mut y)
                .unwrap_err()
                .0
                .contains("expert 2")
        );
        assert!(moe(&m, &[0.0; 64], 1, &[0, 1], &[1.0], 1, &mut y).is_err());
    }
}
