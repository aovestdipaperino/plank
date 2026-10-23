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
    /// window narrowing), built only if the op-chain attention runs.
    // q, k, v, x and t are the attention formula's own names.
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        offset: usize,
        cache: &mut LayerKv,
        rope: &Rope,
        mask: Option<&LazyMask<'_>>,
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
        // Checked before the append, so a refused call leaves the cache as
        // it was.
        if cache.len() != offset {
            bail!(
                "Flint KV cache holds {}..{}, expected ..{offset}",
                cache.base(),
                cache.len()
            )
        }
        cache.append(&k, &v).map_err(kv_err)?;
        let Some((keys, values)) = cache.view().map_err(kv_err)? else {
            bail!("Flint KV cache is empty after an append")
        };
        let (base, stored) = (cache.base(), cache.len());
        // Head dims are small integers, exact in f64.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        if mask.is_some()
            && let Some(ctx) = prefill_attend(
                &q,
                (&keys, &values),
                &self.sinks,
                (offset, base),
                self.window.unwrap_or(0),
                scale,
            )?
        {
            return self.project(&ctx, t);
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
        let fused = if mask.is_none() {
            decode_attend(&q, &keys, &values, &self.sinks, scale)?
        } else {
            None
        };
        let ctx = match fused {
            Some(ctx) => ctx,
            // The op chain multiplies in f32, whatever the cache holds.
            None => attend(
                &q,
                &keys.to_dtype(candle_core::DType::F32)?,
                &values.to_dtype(candle_core::DType::F32)?,
                &self.sinks,
                mask.map(LazyMask::get).transpose()?,
                scale,
            )?,
        };
        self.project(&ctx, t)
    }

    /// The output projection of `ctx` `[1, heads, t, head_dim]`.
    fn project(&self, ctx: &Tensor, t: usize) -> Result<Tensor> {
        let ctx = ctx
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((t, self.heads * self.head_dim))?;
        ops::matmul(&self.o, &ctx)
    }
}

/// A causal mask built on first use: the fused prefill attention needs
/// none, so on Metal it is never built.
pub(crate) struct LazyMask<'a> {
    cell: std::cell::OnceCell<Tensor>,
    build: Box<dyn Fn() -> Result<Tensor> + 'a>,
}

impl<'a> LazyMask<'a> {
    pub(crate) fn new(build: impl Fn() -> Result<Tensor> + 'a) -> Self {
        Self {
            cell: std::cell::OnceCell::new(),
            build: Box::new(build),
        }
    }

    /// The mask, built now if it was not yet.
    pub(crate) fn get(&self) -> Result<&Tensor> {
        if let Some(m) = self.cell.get() {
            return Ok(m);
        }
        let m = (self.build)()?;
        Ok(self.cell.get_or_init(|| m))
    }
}

/// `softmax([q·kᵀ·scale + mask, sink]) · v` without the sink's column.
///
/// `query` is `[1, heads, t, d]`, `key`/`value` `[1, kv_heads, keys, d]`,
/// `sinks` `[heads]`, `mask` `[1, 1, t, keys]` of 0 and -inf, required when
/// `t > 1` (a single query attends to every key it is given).
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
    if mask.is_none() && t > 1 {
        bail!("attention over {t} queries needs a causal mask")
    }
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

/// [`attend`] for a single query (over an f32 or f16 cache) through the
/// fused Metal kernels
/// (`fused.metal`), which read the cache in place instead of copying
/// it into a transposed key matrix and a score tensor per head. `None` when
/// they do not apply: not on Metal, more than one query, or a shape or
/// layout they do not take; the caller then runs [`attend`].
#[cfg(target_os = "macos")]
pub(crate) fn decode_attend(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    sinks: &Tensor,
    scale: f64,
) -> Result<Option<Tensor>> {
    use candle_core::Storage;
    use candle_core::op::BackpropOp;

    use crate::nvfp4::metal::{ATTN_HEAD_DIM, ATTN_SG, decode_attention};

    let (_, heads, t, d) = query.dims4()?;
    let (_, kv_heads, keys, key_d) = key.dims4()?;
    let fits = query.device().is_metal()
        && t == 1
        && d == ATTN_HEAD_DIM
        && key_d == d
        && value.dims() == key.dims()
        && value.dtype() == key.dtype()
        && matches!(
            key.dtype(),
            candle_core::DType::F32 | candle_core::DType::F16
        )
        && keys > 0
        && kv_heads > 0
        && heads.is_multiple_of(kv_heads)
        && heads / kv_heads <= ATTN_SG;
    if !fits {
        return Ok(None);
    }
    let query = query.contiguous()?;
    let sinks = sinks.contiguous()?;
    let (q_store, q_layout) = query.storage_and_layout();
    let (k_store, k_layout) = key.storage_and_layout();
    let (v_store, v_layout) = value.storage_and_layout();
    let (s_store, s_layout) = sinks.storage_and_layout();
    let (
        Storage::Metal(q_metal),
        Storage::Metal(k_metal),
        Storage::Metal(v_metal),
        Storage::Metal(s_metal),
    ) = (&*q_store, &*k_store, &*v_store, &*s_store)
    else {
        return Ok(None);
    };
    let (Some(k_op), Some(v_op)) = (
        kv_operand(k_metal, k_layout, d),
        kv_operand(v_metal, v_layout, d),
    ) else {
        return Ok(None);
    };
    if !q_layout.start_offset().is_multiple_of(4) {
        return Ok(None);
    }
    // The kernel takes the scale as f32, as every score is.
    #[allow(clippy::cast_possible_truncation)]
    let out = decode_attention(
        q_metal,
        q_layout.start_offset() * 4,
        &k_op,
        &v_op,
        (s_metal, s_layout.start_offset() * 4),
        heads,
        kv_heads,
        keys,
        scale as f32,
    )?;
    Ok(Some(Tensor::from_storage(
        Storage::Metal(out),
        (1, heads, 1, d),
        BackpropOp::none(),
        false,
    )))
}

/// Where [`decode_attend`] reads a KV operand: `[1, kv_heads, keys, d]` with
/// each key's values adjacent and every head's first key on a 16-byte
/// boundary, as the kernel's float4 loads need. `None` for any other layout.
#[cfg(target_os = "macos")]
fn kv_operand<'a>(
    storage: &'a candle_core::MetalStorage,
    layout: &candle_core::Layout,
    d: usize,
) -> Option<crate::nvfp4::metal::KvOperand<'a>> {
    let stride = layout.stride();
    let size = candle_core::backend::BackendStorage::dtype(storage).size_in_bytes();
    (stride[3] == 1
        && stride[2] == d
        && stride[1].is_multiple_of(4)
        && layout.start_offset().is_multiple_of(4))
    .then(|| crate::nvfp4::metal::KvOperand {
        storage,
        offset: layout.start_offset() * size,
        head_stride: stride[1],
    })
}

/// Attention for many queries through the fused prefill kernel
/// (`attn_prefill` in `fused.metal`), which reads the cache in place and
/// masks from positions instead of a mask tensor. `query` is
/// `[1, heads, t, d]` at absolute positions `q_pos0..`, `key`/`value` the
/// whole live cache view from absolute position `key_base`, `window` 0 for
/// none. Returns `[1, heads, t, d]`, or `None` when the kernel does not
/// apply (not on Metal, or a shape or layout it does not take).
#[cfg(target_os = "macos")]
pub(crate) fn prefill_attend(
    query: &Tensor,
    (key, value): (&Tensor, &Tensor),
    sinks: &Tensor,
    (q_pos0, key_base): (usize, usize),
    window: usize,
    scale: f64,
) -> Result<Option<Tensor>> {
    use candle_core::Storage;
    use candle_core::op::BackpropOp;

    use crate::nvfp4::metal::{ATTN_HEAD_DIM, PF_Q, PrefillQueries, prefill_attention};

    let (_, heads, t, d) = query.dims4()?;
    let (_, kv_heads, _, key_d) = key.dims4()?;
    let fits = query.device().is_metal()
        && t > 1
        && d == ATTN_HEAD_DIM
        && key_d == d
        && value.dims() == key.dims()
        && value.dtype() == key.dtype()
        && matches!(
            key.dtype(),
            candle_core::DType::F32 | candle_core::DType::F16
        )
        && kv_heads > 0
        && heads.is_multiple_of(kv_heads);
    if !fits {
        return Ok(None);
    }
    // Whole 32-row blocks of queries per head; the padding rows come out
    // zero and are cut off below.
    let t_pad = t.div_ceil(PF_Q) * PF_Q;
    // With an f16 cache the kernel multiplies in f16 (accumulating in f32),
    // so the queries go to f16 too.
    let query = query.to_dtype(key.dtype())?;
    let query = if t_pad > t {
        let pad = Tensor::zeros((1, heads, t_pad - t, d), query.dtype(), query.device())?;
        Tensor::cat(&[&query, &pad], 2)?
    } else {
        query.contiguous()?
    };
    let sinks = sinks.contiguous()?;
    let (q_store, q_layout) = query.storage_and_layout();
    let (k_store, k_layout) = key.storage_and_layout();
    let (v_store, v_layout) = value.storage_and_layout();
    let (s_store, s_layout) = sinks.storage_and_layout();
    let (
        Storage::Metal(q_metal),
        Storage::Metal(k_metal),
        Storage::Metal(v_metal),
        Storage::Metal(s_metal),
    ) = (&*q_store, &*k_store, &*v_store, &*s_store)
    else {
        return Ok(None);
    };
    let (Some(k_op), Some(v_op)) = (
        kv_operand(k_metal, k_layout, d),
        kv_operand(v_metal, v_layout, d),
    ) else {
        return Ok(None);
    };
    let queries = PrefillQueries {
        storage: q_metal,
        offset: q_layout.start_offset() * query.dtype().size_in_bytes(),
        t,
        t_pad,
        pos0: q_pos0,
    };
    // The kernel takes the scale as f32, as every score is.
    #[allow(clippy::cast_possible_truncation)]
    let out = prefill_attention(
        &queries,
        &k_op,
        &v_op,
        (s_metal, s_layout.start_offset() * 4),
        (heads, kv_heads),
        key_base,
        window,
        scale as f32,
    )?;
    let out = Tensor::from_storage(
        Storage::Metal(out),
        (1, heads, t_pad, d),
        BackpropOp::none(),
        false,
    );
    Ok(Some(out.narrow(2, 0, t)?))
}

/// Off macOS there is no fused prefill: always [`attend`].
#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn prefill_attend(
    _query: &Tensor,
    _kv: (&Tensor, &Tensor),
    _sinks: &Tensor,
    _positions: (usize, usize),
    _window: usize,
    _scale: f64,
) -> Result<Option<Tensor>> {
    Ok(None)
}

/// Off macOS there are no Metal kernels: always [`attend`].
#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn decode_attend(
    _query: &Tensor,
    _key: &Tensor,
    _value: &Tensor,
    _sinks: &Tensor,
    _scale: f64,
) -> Result<Option<Tensor>> {
    Ok(None)
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

    fn tiny_attention() -> Attention {
        use crate::testutil::{Rng, weight};
        let (hidden, heads, kv_heads, head_dim) = (64, 2, 1, 64);
        let mut rng = Rng::new(11);
        let dev = Device::Cpu;
        let ones = |n: usize| Tensor::ones(n, candle_core::DType::F32, &dev).unwrap();
        Attention {
            q: Arc::new(weight(&mut rng, hidden, heads * head_dim, 1)),
            k: Arc::new(weight(&mut rng, hidden, kv_heads * head_dim, 1)),
            v: Arc::new(weight(&mut rng, hidden, kv_heads * head_dim, 1)),
            o: Arc::new(weight(&mut rng, heads * head_dim, hidden, 1)),
            q_norm: ones(head_dim),
            k_norm: ones(head_dim),
            sinks: ones(heads),
            heads,
            kv_heads,
            head_dim,
            window: None,
            rope: true,
            eps: 1e-6,
        }
    }

    #[test]
    fn a_cache_that_does_not_end_at_the_offset_is_refused_and_left_unchanged() {
        let dev = Device::Cpu;
        let attn = tiny_attention();
        let rope = Rope::new(64, 10_000.0, 16, &dev).unwrap();
        let mut cache = LayerKv::default();
        let x = Tensor::from_vec(rand(2 * 64, 4), (2, 64), &dev).unwrap();
        let mask = LazyMask::new(|| causal_mask(2, 0, 0, 2, None, &dev));
        attn.forward(&x, 0, &mut cache, &rope, Some(&mask)).unwrap();
        assert_eq!(cache.len(), 2);
        let one = x.narrow(0, 0, 1).unwrap();
        assert!(attn.forward(&one, 5, &mut cache, &rope, None).is_err());
        assert_eq!(cache.len(), 2, "a refused forward appends nothing");
    }

    #[test]
    fn several_queries_without_a_mask_are_refused() {
        let dev = Device::Cpu;
        let q = Tensor::ones((1, 1, 2, 4), candle_core::DType::F32, &dev).unwrap();
        let k = Tensor::ones((1, 1, 2, 4), candle_core::DType::F32, &dev).unwrap();
        let s = Tensor::new(&[0f32], &dev).unwrap();
        assert!(attend(&q, &k, &k, &s, None, 1.0).is_err());
    }

    /// The fused single-query kernels against [`attend`] on Metal: several
    /// grouping ratios, key counts on both sides of a chunk boundary, and
    /// keys narrowed out of a larger cache buffer at an offset, as the
    /// sliding layers read them. Large score magnitudes check the max
    /// subtraction across chunks.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_fused_decode_kernels_match_attend_on_metal() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let d = 128;
        for (case, &(heads, kv_heads, keys, cap, start, amp)) in [
            (32, 4, 1, 256, 0, 1.0f32),
            (32, 4, 7, 256, 0, 1.0),
            (32, 4, 128, 256, 0, 1.0),
            (32, 4, 129, 512, 0, 4.0),
            (32, 4, 1000, 1024, 5, 1.0),
            (8, 8, 300, 512, 3, 1.0),
            (16, 2, 2500, 4096, 100, 8.0),
        ]
        .iter()
        .enumerate()
        {
            let seed = case as u64 * 10;
            let buf = |seed: u64| {
                let v: Vec<f32> = rand(kv_heads * cap * d, seed)
                    .iter()
                    .map(|x| x * amp)
                    .collect();
                Tensor::from_vec(v, (1, kv_heads, cap, d), &dev).unwrap()
            };
            let k = buf(seed + 1).narrow(2, start, keys).unwrap();
            let v = buf(seed + 2).narrow(2, start, keys).unwrap();
            let q = Tensor::from_vec(rand(heads * d, seed + 3), (1, heads, 1, d), &dev).unwrap();
            let sinks = Tensor::from_vec(rand(heads, seed + 4), heads, &dev).unwrap();
            let scale = 1.0 / 128f64.sqrt();
            let want: Vec<f32> = attend(&q, &k, &v, &sinks, None, scale)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let got: Vec<f32> = decode_attend(&q, &k, &v, &sinks, scale)
                .unwrap()
                .expect("the fused kernels take this shape")
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-4 * amp, "case {case}: max diff {worst}");
        }
    }

    /// The fused prefill kernel against [`attend`] with an explicit causal
    /// mask on Metal: a fresh cache, queries after earlier keys, a sliding
    /// window over a cache whose base is past 0, query counts on both sides
    /// of the 32-row block, and f32 and f16 caches.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_fused_prefill_matches_attend_on_metal() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let d = 128;
        let scale = 1.0 / 128f64.sqrt();
        for (case, &(heads, kv_heads, t, offset, base, window, half)) in [
            (32, 4, 37, 0, 0, 0, false),
            (8, 2, 64, 100, 0, 0, true),
            (16, 4, 33, 300, 40, 50, true),
            (8, 8, 2, 7, 0, 3, false),
        ]
        .iter()
        .enumerate()
        {
            let seed = 100 + case as u64 * 10;
            let stored = offset + t - base;
            let buf = |seed: u64| {
                let v = Tensor::from_vec(
                    rand(kv_heads * stored * d, seed),
                    (1, kv_heads, stored, d),
                    &dev,
                )
                .unwrap();
                if half {
                    v.to_dtype(candle_core::DType::F16).unwrap()
                } else {
                    v
                }
            };
            let (k, v) = (buf(seed + 1), buf(seed + 2));
            let q =
                Tensor::from_vec(rand(heads * t * d, seed + 3), (1, heads, t, d), &dev).unwrap();
            let sinks = Tensor::from_vec(rand(heads, seed + 4), heads, &dev).unwrap();
            let got: Vec<f32> = prefill_attend(&q, (&k, &v), &sinks, (offset, base), window, scale)
                .unwrap()
                .expect("the prefill kernel takes this shape")
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let start = if window == 0 {
                base
            } else {
                base.max((offset + 1).saturating_sub(window))
            };
            let wide = |x: &Tensor| {
                x.narrow(2, start - base, stored - (start - base))
                    .unwrap()
                    .to_dtype(candle_core::DType::F32)
                    .unwrap()
            };
            let mask = causal_mask(
                t,
                offset,
                start,
                offset + t - start,
                (window > 0).then_some(window),
                &dev,
            )
            .unwrap();
            let want: Vec<f32> = attend(&q, &wide(&k), &wide(&v), &sinks, Some(&mask), scale)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // An f16 cache multiplies in f16.
            let tol = if half { 3e-3 } else { 1e-4 };
            assert!(worst < tol, "case {case}: max diff {worst}");
        }
    }

    /// More than one query, or a head size the kernels do not take, falls
    /// back to [`attend`].
    #[test]
    fn the_fused_decode_kernels_decline_what_they_do_not_take() {
        let dev = Device::new_metal(0).unwrap_or(Device::Cpu);
        let z = |s: (usize, usize, usize, usize)| {
            Tensor::zeros(s, candle_core::DType::F32, &dev).unwrap()
        };
        let sinks = Tensor::zeros(32, candle_core::DType::F32, &dev).unwrap();
        let k = z((1, 4, 10, 128));
        assert!(
            decode_attend(&z((1, 32, 2, 128)), &k, &k, &sinks, 1.0)
                .unwrap()
                .is_none()
        );
        let k64 = z((1, 4, 10, 64));
        assert!(
            decode_attend(&z((1, 32, 1, 64)), &k64, &k64, &sinks, 1.0)
                .unwrap()
                .is_none()
        );
        let cpu = |s| Tensor::zeros(s, candle_core::DType::F32, &Device::Cpu).unwrap();
        let kc = cpu((1, 4, 10, 128));
        let sc = Tensor::zeros(32, candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(
            decode_attend(&cpu((1, 32, 1, 128)), &kc, &kc, &sc, 1.0)
                .unwrap()
                .is_none()
        );
    }
}
