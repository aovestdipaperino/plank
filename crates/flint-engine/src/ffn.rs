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
        let logits = Tensor::new(&[[-100f32, -100.0, -100.0]], &dev).unwrap();
        let (_, w) = route(&logits, 2, true, 1.0).unwrap();
        assert!(w.to_vec2::<f32>().unwrap()[0].iter().all(|x| x.is_finite()));
    }
}
