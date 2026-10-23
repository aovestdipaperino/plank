//! Flint's feed-forward blocks: the dense `SwiGLU` of the leading layers, and
//! the routed `MoE` block (`build_moe_ffn` with sigmoid gating) plus the
//! shared expert that every later layer adds.

use std::sync::Arc;

use candle_core::{Result, Tensor};

use crate::nvfp4::ops;
use crate::nvfp4::tensor::{MoeExperts, NvWeight};

/// `build_moe_ffn`'s floor on the routing-weight sum (the smallest f16
/// normal), so a row of vanishing probabilities never divides by zero.
const MIN_WEIGHT_SUM: f64 = 6.103_515_625e-5;
/// [`MIN_WEIGHT_SUM`] for the Metal kernels, exact in f32.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const MIN_WEIGHT_SUM_F32: f32 = MIN_WEIGHT_SUM as f32;

/// `down(silu(gate x) * up x)`; each projection's tensor scale is applied
/// by the op, so the gate is scaled before the `SiLU`.
#[derive(Debug)]
pub(crate) struct Dense {
    pub gate: Arc<NvWeight>,
    pub up: Arc<NvWeight>,
    pub down: Arc<NvWeight>,
}

impl Dense {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = ops::matmul(&self.gate, x)?.silu()?;
        let u = ops::matmul(&self.up, x)?;
        ops::matmul(&self.down, &(g * u)?.contiguous()?)
    }
}

/// The routed experts of one layer plus its shared expert.
#[derive(Debug)]
pub(crate) struct Moe {
    /// `[experts, hidden]`, widened from BF16.
    pub router: Tensor,
    pub experts: MoeExperts,
    pub shared: Dense,
    pub k: usize,
    pub norm: bool,
    pub scale: f32,
}

impl Moe {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = x.matmul(&self.router.t()?)?;
        let (ids, weights) = route(&logits, self.k, self.norm, self.scale)?;
        let routed = ops::moe(&self.experts, x, &ids, &weights)?;
        routed + self.shared.forward(x)?
    }
}

/// Sigmoid gating: probabilities `sigmoid(logits)`, the `k` largest per
/// row, their probabilities as weights, divided by the (floored) row sum
/// when `norm`, then times `scale` unless it is 0 or 1. Returns the expert
/// ids (`u32`) and weights (`f32`), both `[t, k]`.
pub(crate) fn route(logits: &Tensor, k: usize, norm: bool, scale: f32) -> Result<(Tensor, Tensor)> {
    if let Some(fused) = route_fused(logits, k, norm, scale)? {
        return Ok(fused);
    }
    route_generic(logits, k, norm, scale)
}

/// [`route`] as one Metal kernel (`moe_route`) instead of nine ops, or
/// `None` off Metal or for a `k` the kernel does not take. Ties between
/// equal probabilities go to the lower expert id.
#[cfg(target_os = "macos")]
fn route_fused(
    logits: &Tensor,
    k: usize,
    norm: bool,
    scale: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle_core::Storage;
    use candle_core::op::BackpropOp;

    use crate::nvfp4::metal::{ROUTE_MAX_EXPERTS, ROUTE_MAX_K, moe_route};

    let (tokens, experts) = logits.dims2()?;
    if !logits.device().is_metal()
        || k == 0
        || k > ROUTE_MAX_K
        || k > experts
        || experts > ROUTE_MAX_EXPERTS
        || tokens == 0
    {
        return Ok(None);
    }
    let logits = logits.contiguous()?;
    let (store, layout) = logits.storage_and_layout();
    let Storage::Metal(m) = &*store else {
        return Ok(None);
    };
    let applied = if scale.abs() > f32::EPSILON && (scale - 1.0).abs() > f32::EPSILON {
        scale
    } else {
        1.0
    };
    // The floor is the smallest f16 normal, exact in f32.
    #[allow(clippy::cast_possible_truncation)]
    let (ids, weights) = moe_route(
        m,
        layout.start_offset() * 4,
        (tokens, experts),
        k,
        norm,
        (applied, MIN_WEIGHT_SUM as f32),
    )?;
    let wrap = |s| Tensor::from_storage(Storage::Metal(s), (tokens, k), BackpropOp::none(), false);
    Ok(Some((wrap(ids), wrap(weights))))
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_wraps)]
fn route_fused(_: &Tensor, _: usize, _: bool, _: f32) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}

/// [`route`] from candle ops, on any device.
pub(crate) fn route_generic(
    logits: &Tensor,
    k: usize,
    norm: bool,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let probs = candle_nn::ops::sigmoid(logits)?;
    let ids = probs
        .arg_sort_last_dim(false)?
        .narrow(1, 0, k)?
        .contiguous()?;
    let mut weights = probs.gather(&ids, 1)?;
    if norm {
        let sum = weights.sum_keepdim(1)?.maximum(MIN_WEIGHT_SUM)?;
        weights = weights.broadcast_div(&sum)?;
    }
    if scale.abs() > f32::EPSILON && (scale - 1.0).abs() > f32::EPSILON {
        weights = (weights * f64::from(scale))?;
    }
    Ok((ids, weights.contiguous()?))
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn route_picks_the_top_sigmoids_and_normalises() {
        let dev = Device::Cpu;
        let logits = Tensor::new(
            &[[0.1f32, 2.0, -1.0, 3.0, 0.5], [-2.0, -3.0, 1.0, 0.0, -0.5]],
            &dev,
        )
        .unwrap();
        let (ids, w) = route(&logits, 2, true, 1.0).unwrap();
        assert_eq!(ids.to_vec2::<u32>().unwrap(), vec![vec![3, 1], vec![2, 3]]);
        let w = w.to_vec2::<f32>().unwrap();
        let (a, b) = (sigmoid(3.0), sigmoid(2.0));
        assert!((w[0][0] - a / (a + b)).abs() < 1e-6 && (w[0][1] - b / (a + b)).abs() < 1e-6);
        assert!((w[1][0] + w[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn route_without_norm_keeps_raw_probabilities_and_applies_the_scale() {
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[0.0f32, 1.0]], &dev).unwrap();
        let (_, w) = route(&logits, 1, false, 2.5).unwrap();
        assert!((w.to_vec2::<f32>().unwrap()[0][0] - 2.5 * sigmoid(1.0)).abs() < 1e-6);
    }

    #[test]
    fn route_clamps_a_vanishing_sum() {
        let dev = Device::Cpu;
        // sigmoid(-20) is about 2.1e-9, so the row's sum sits far below the
        // floor: divided by the floor the weights stay tiny, where dividing
        // by their own sum would make them 0.5 each. sigmoid(-100) is 0 in
        // f32, and the floor keeps 0 / 0 from becoming NaN.
        let logits =
            Tensor::new(&[[-20f32, -20.0, -20.0], [-100.0, -100.0, -100.0]], &dev).unwrap();
        let (_, w) = route(&logits, 2, true, 1.0).unwrap();
        let w = w.to_vec2::<f32>().unwrap();
        #[allow(clippy::cast_possible_truncation)] // the floor is an exact f32
        let want = sigmoid(-20.0) / MIN_WEIGHT_SUM as f32;
        assert!(
            w[0].iter().all(|x| (x - want).abs() <= 1e-3 * want),
            "{:?} against {want}",
            w[0]
        );
        assert!(w[0].iter().sum::<f32>() < 1e-3, "{:?}", w[0]);
        assert!(
            w[1].iter().all(|x| x.is_finite() && *x < 1e-30),
            "{:?}",
            w[1]
        );
    }

    /// The fused `moe_route` kernel against the candle op chain on Metal,
    /// with and without normalisation and scale, at decode and prefill
    /// widths.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_fused_route_matches_the_op_chain_on_metal() {
        let Ok(dev) = candle_core::Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let mut seed = 7u64;
        for (tokens, norm, scale) in [(1usize, true, 1.0f32), (37, true, 2.5), (5, false, 0.0)] {
            let v: Vec<f32> = (0..tokens * 256)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    #[allow(clippy::cast_precision_loss)]
                    let x = ((seed >> 40) as f32) / (1u64 << 24) as f32;
                    x * 8.0 - 4.0
                })
                .collect();
            let logits = Tensor::from_vec(v, (tokens, 256), &dev).unwrap();
            let (want_ids, want_w) = route_generic(&logits, 6, norm, scale).unwrap();
            let (got_ids, got_w) = route(&logits, 6, norm, scale).unwrap();
            assert_eq!(
                got_ids.to_vec2::<u32>().unwrap(),
                want_ids.to_vec2::<u32>().unwrap(),
                "{tokens} tokens"
            );
            let (w, g) = (
                want_w.to_vec2::<f32>().unwrap(),
                got_w.to_vec2::<f32>().unwrap(),
            );
            for (a, b) in w.iter().flatten().zip(g.iter().flatten()) {
                assert!((a - b).abs() < 1e-6, "{a} vs {b}");
            }
        }
    }
}
