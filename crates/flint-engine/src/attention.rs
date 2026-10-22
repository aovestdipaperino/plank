//! Flint self-attention (`llama_model_flint::graph`): grouped-query
//! attention whose q and k get a per-head `RMSNorm`, `NeoX` `RoPE` on the layers
//! that have it, a sliding window on the layers that have one, and one
//! learned sink logit per head that takes softmax mass but adds no value.

use std::sync::Arc;

use candle_core::{Device, Result, Tensor, bail};
use gemma_engine::kv::LayerKv;

use crate::nvfp4::ops;
use crate::nvfp4::tensor::NvWeight;
use crate::rope::Rope;

/// One layer's attention weights and shape.
#[derive(Debug)]
pub(crate) struct Attention {
    pub q: Arc<NvWeight>,
    pub k: Arc<NvWeight>,
    pub v: Arc<NvWeight>,
    pub o: Arc<NvWeight>,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    /// `[heads]`.
    pub sinks: Tensor,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub window: Option<usize>,
    pub rope: bool,
    pub eps: f32,
}

fn kv_err(e: gemma_engine::Error) -> candle_core::Error {
    candle_core::Error::msg(e.0)
}

impl Attention {
    /// `x` is `[t, hidden]` at positions `offset..`; returns `[t, hidden]`
    /// and extends `cache`. `mask` is this layer type's mask for the forward
    /// (`None` for a single token, which sees every key left after the
    /// window narrowing).
    // q, k, v, x and t are the attention formula's own names.
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        offset: usize,
        cache: &mut LayerKv,
        rope: &Rope,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        let heads = |w: &Arc<NvWeight>, n: usize| -> Result<Tensor> {
            ops::matmul(w, x)?.reshape((t, n, self.head_dim))
        };
        let q = candle_nn::ops::rms_norm(
            &heads(&self.q, self.heads)?.contiguous()?,
            &self.q_norm,
            self.eps,
        )?;
        let k = candle_nn::ops::rms_norm(
            &heads(&self.k, self.kv_heads)?.contiguous()?,
            &self.k_norm,
            self.eps,
        )?;
        let v = heads(&self.v, self.kv_heads)?;
        // [1, heads, t, head_dim]
        let shape = |x: Tensor| -> Result<Tensor> { x.transpose(0, 1)?.unsqueeze(0)?.contiguous() };
        let (q, k, v) = (shape(q)?, shape(k)?, shape(v)?);
        let (q, k) = if self.rope {
            (rope.apply(&q, offset)?, rope.apply(&k, offset)?)
        } else {
            (q, k)
        };
        cache.append(&k, &v).map_err(kv_err)?;
        let Some((keys, values)) = cache.view().map_err(kv_err)? else {
            bail!("Flint KV cache is empty after an append")
        };
        let (base, stored) = (cache.base(), cache.len());
        if stored != offset + t {
            bail!(
                "Flint KV cache holds {base}..{stored}, expected ..{}",
                offset + t
            )
        }
        // A sliding layer's earliest query sees no key before
        // `offset + 1 - window`; drop those before scoring.
        let start = self.window.map_or(0, |w| (offset + 1).saturating_sub(w));
        if start < base {
            bail!("Flint KV cache starts at {base}, attention needs keys from {start}")
        }
        let (keys, values) = if start > base {
            (
                keys.narrow(2, start - base, stored - start)?,
                values.narrow(2, start - base, stored - start)?,
            )
        } else {
            (keys, values)
        };
        // Head dims are small integers, exact in f64.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let ctx = attend(&q, &keys, &values, &self.sinks, mask, scale)?;
        let ctx = ctx
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((t, self.heads * self.head_dim))?;
        ops::matmul(&self.o, &ctx)
    }
}

/// `softmax([q·kᵀ·scale + mask, sink]) · v` without the sink's column.
///
/// `query` is `[1, heads, t, d]`, `key`/`value` `[1, kv_heads, keys, d]`,
/// `sinks` `[heads]`, `mask` (when given) `[1, 1, t, keys]` of 0 and -inf.
/// Query head `h` reads KV head `h / (heads / kv_heads)`.
pub(crate) fn attend(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let (_, heads, t, d) = query.dims4()?;
    let (_, kv_heads, keys, _) = key.dims4()?;
    let rep = heads / kv_heads;
    let mut scores = query
        .reshape((1, kv_heads, rep * t, d))?
        .matmul(&key.transpose(2, 3)?.contiguous()?)?
        .reshape((1, heads, t, keys))?;
    scores = (scores * scale)?;
    if let Some(mask) = mask {
        if mask.dims() != [1, 1, t, keys] {
            bail!(
                "attention mask {:?} does not fit [1, 1, {t}, {keys}]",
                mask.dims()
            )
        }
        scores = scores.broadcast_add(mask)?;
    }
    let sink = sinks
        .reshape((1, heads, 1, 1))?
        .broadcast_as((1, heads, t, 1))?
        .contiguous()?;
    let probs = candle_nn::ops::softmax_last_dim(&Tensor::cat(&[&scores, &sink], 3)?)?
        .narrow(3, 0, keys)?
        .contiguous()?;
    probs
        .reshape((1, kv_heads, rep * t, keys))?
        .matmul(value)?
        .reshape((1, heads, t, d))
}

/// The additive mask for `t` queries at `offset..` over keys at
/// `key_start..key_start + keys`: `-inf` where the key is in the future,
/// or (with a window) `window` or more positions behind the query.
pub(crate) fn causal_mask(
    t: usize,
    offset: usize,
    key_start: usize,
    keys: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let mut m = Vec::with_capacity(t * keys);
    for i in 0..t {
        let qp = offset + i;
        for j in 0..keys {
            let kp = key_start + j;
            let masked = kp > qp || window.is_some_and(|w| qp - kp >= w);
            m.push(if masked { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    Tensor::from_vec(m, (1, 1, t, keys), device)
}
#[cfg(test)]
// q, k, v, d and t are the attention formula's own names.
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // 24 bits to f32 is exact.
                #[allow(clippy::cast_precision_loss)]
                let v = ((s >> 40) as f32) / (1u64 << 24) as f32;
                v * 2.0 - 1.0
            })
            .collect()
    }

    /// softmax with a sink, written out in f64 loops.
    #[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
    fn naive(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        (heads, kv_heads, t, keys, d): (usize, usize, usize, usize, usize),
        key_start: usize,
        offset: usize,
        window: Option<usize>,
        scale: f64,
    ) -> Vec<f32> {
        let rep = heads / kv_heads;
        let mut out = vec![0f32; heads * t * d];
        for h in 0..heads {
            let g = h / rep;
            for i in 0..t {
                let qp = offset + i;
                let mut logits = Vec::new();
                for j in 0..keys {
                    let kp = key_start + j;
                    let masked = kp > qp || window.is_some_and(|w| qp - kp >= w);
                    let dot: f64 = (0..d)
                        .map(|x| {
                            f64::from(q[(h * t + i) * d + x]) * f64::from(k[(g * keys + j) * d + x])
                        })
                        .sum();
                    logits.push(if masked {
                        f64::NEG_INFINITY
                    } else {
                        dot * scale
                    });
                }
                let m = logits.iter().copied().fold(f64::from(sinks[h]), f64::max);
                let denom: f64 = logits.iter().map(|l| (l - m).exp()).sum::<f64>()
                    + (f64::from(sinks[h]) - m).exp();
                for x in 0..d {
                    let acc: f64 = (0..keys)
                        .map(|j| {
                            (logits[j] - m).exp() / denom * f64::from(v[(g * keys + j) * d + x])
                        })
                        .sum();
                    out[(h * t + i) * d + x] = acc as f32;
                }
            }
        }
        out
    }

    fn check(t: usize, keys: usize, key_start: usize, offset: usize, window: Option<usize>) {
        let (heads, kv_heads, d) = (4, 2, 8);
        let dev = Device::Cpu;
        let (qv, kv, vv) = (
            rand(heads * t * d, 1),
            rand(kv_heads * keys * d, 2),
            rand(kv_heads * keys * d, 3),
        );
        let sv = vec![0.5, -1.0, 2.0, 0.0];
        let q = Tensor::from_vec(qv.clone(), (1, heads, t, d), &dev).unwrap();
        let k = Tensor::from_vec(kv.clone(), (1, kv_heads, keys, d), &dev).unwrap();
        let v = Tensor::from_vec(vv.clone(), (1, kv_heads, keys, d), &dev).unwrap();
        let s = Tensor::from_vec(sv.clone(), heads, &dev).unwrap();
        let mask = (t > 1).then(|| causal_mask(t, offset, key_start, keys, window, &dev).unwrap());
        let scale = 1.0 / f64::from(u8::try_from(d).unwrap()).sqrt();
        let got: Vec<f32> = attend(&q, &k, &v, &s, mask.as_ref(), scale)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let want = naive(
            &qv,
            &kv,
            &vv,
            &sv,
            (heads, kv_heads, t, keys, d),
            key_start,
            offset,
            window,
            scale,
        );
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "t={t} keys={keys} start={key_start} window={window:?} [{i}] {a} vs {b}"
            );
        }
    }

    #[test]
    fn attend_matches_the_written_out_softmax_with_sinks() {
        check(1, 6, 0, 5, None); // decode, global
        check(5, 5, 0, 0, None); // prefill from zero, causal
        check(4, 7, 3, 6, Some(4)); // sliding prefill past the window
        check(1, 4, 6, 9, Some(4)); // sliding decode, keys already narrowed
    }

    #[test]
    fn a_sink_takes_probability_mass() {
        let dev = Device::Cpu;
        let q = Tensor::ones((1, 1, 1, 4), candle_core::DType::F32, &dev).unwrap();
        let k = Tensor::ones((1, 1, 2, 4), candle_core::DType::F32, &dev).unwrap();
        let v = Tensor::ones((1, 1, 2, 4), candle_core::DType::F32, &dev).unwrap();
        let none = Tensor::new(&[f32::NEG_INFINITY], &dev).unwrap();
        let big = Tensor::new(&[100f32], &dev).unwrap();
        let a: Vec<f32> = attend(&q, &k, &v, &none, None, 1.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b: Vec<f32> = attend(&q, &k, &v, &big, None, 1.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!((a[0] - 1.0).abs() < 1e-6, "no sink: values average to 1");
        assert!(b[0] < 1e-6, "a dominant sink leaves almost nothing");
    }
}
