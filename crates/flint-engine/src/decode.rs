//! The single-token decode graph on Metal.
//!
//! A decode step through candle ops is about 2,000 kernels, each in its own
//! encoder, with a fresh allocation per intermediate; at about 1.3 us a
//! kernel of overhead that is as much time as the weights take to stream.
//! The graph encodes the whole forward of one token into one encoder
//! instead, about 14 dispatches a layer, over scratch buffers allocated
//! once, and writes the new K/V straight into the cache:
//!
//! - `dec_rmsnorm` normalises the residual stream (adding the attention
//!   output first from the second norm of a layer on);
//! - `dec_mv3_nvfp4` runs q, k and v as one matvec;
//! - `dec_qk_norm_rope` normalises and rotates q and k and stores k and v in
//!   the cache;
//! - `attn_dec_partial`/`attn_dec_combine` attend over the cache in place;
//! - `mv_nvfp4` projects the attention output, and later the shared
//!   expert's and the dense layers' down projections and the output head;
//! - `dec_mv_f32` and `moe_route` route, `moe_gate_up`/`moe_down` run the
//!   experts (and, with one fixed slot, the dense and shared FFNs' gate and
//!   up), and `dec_combine` adds the FFN back into the residual stream.
//!
//! A serial compute encoder runs its dispatches in order, each seeing the
//! last one's writes, which is the only ordering the graph needs.
//! [`Model::forward`] uses it for every single-token step on Metal whose
//! shapes it takes ([`DecodeGraph::new`] says which) and the candle path
//! for everything else, so prefill and the CPU are unchanged.

use candle_core::{DType, Device, MetalDevice, Storage, Tensor};
use std::ptr::NonNull;

use candle_metal_kernels::metal::{Buffer, ComputePipeline, MTLResourceOptions};
use gemma_engine::kv::KvCache;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLDispatchType, MTLSize,
};

use crate::model::{Ffn, Layer, Model};
use crate::nvfp4::metal::{
    ATTN_HEAD_DIM, ATTN_MAX_CHUNKS, ATTN_SG, AttnParams, COMBINE_GROUPS, COMBINE_PER_THREAD,
    MV_ROWS_PER_SG, MV_ROWS_PER_TG, MV_SG_PER_TG, MV_THREADS, MatParams, MoeParams, NORM_THREADS,
    Pipelines, ROUTE_MAX_EXPERTS, ROUTE_MAX_K, ROUTE_THREADS, RouteParams, SIMD_WIDTH, attn_split,
    binding, grid, pipelines, u32_of,
};
use crate::nvfp4::tensor::NvWeight;
use crate::{Result, err};

/// A concurrent compute encoder: dispatches may overlap until a barrier, so
/// [`Enc::dispatch`] ends each with one and [`Enc::dispatch_overlap`] lets
/// the next dispatch run beside it.
struct Enc(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>);

impl Enc {
    fn pipeline(&self, p: &ComputePipeline) {
        self.0.setComputePipelineState(p.as_ref());
    }

    fn buffer(&self, index: usize, b: &Buffer, offset: usize) {
        // SAFETY: the buffer is retained by the graph or the cache for the
        // whole step, and every offset is within it.
        unsafe {
            self.0
                .setBuffer_offset_atIndex(Some(b.as_ref()), offset, index);
        };
    }

    fn bytes<T>(&self, index: usize, v: &T) {
        // SAFETY: Metal copies `size_of::<T>()` bytes from `v` now.
        unsafe {
            self.0
                .setBytes_length_atIndex(NonNull::from(v).cast(), size_of::<T>(), index);
        };
    }

    /// Dispatches, then a barrier: nothing later starts before this has
    /// finished writing.
    fn dispatch(&self, groups: MTLSize, threads: MTLSize) {
        self.dispatch_overlap(groups, threads);
        self.0.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    /// Dispatches without a barrier: the next dispatch may run beside this
    /// one, so it must not read what this one writes.
    fn dispatch_overlap(&self, groups: MTLSize, threads: MTLSize) {
        self.0
            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    }
}

/// A buffer and the byte offset its data starts at.
#[derive(Debug, Clone)]
struct Slice {
    buf: Buffer,
    off: usize,
}

/// A contiguous f32 Metal tensor as a [`Slice`]. The graph holds a retained
/// reference to the buffer, so it outlives nothing it needs; the model's
/// tensors are never written after loading.
fn slice_of(t: &Tensor) -> Result<Slice> {
    if t.dtype() != DType::F32 {
        return Err(err("decode graph: a model tensor is not f32"));
    }
    raw_slice(t)
}

/// [`slice_of`] for a tensor of any element type (a KV buffer).
fn raw_slice(t: &Tensor) -> Result<Slice> {
    if !t.is_contiguous() {
        return Err(err("decode graph: a tensor is not contiguous"));
    }
    let (store, layout) = t.storage_and_layout();
    let Storage::Metal(m) = &*store else {
        return Err(err("decode graph: a model tensor is not on Metal"));
    };
    Ok(Slice {
        buf: m.buffer().clone(),
        off: layout.start_offset() * t.dtype().size_in_bytes(),
    })
}

/// An NVFP4 weight bound to the device: its codes, its per-expert scales,
/// and the shape the kernels' parameters need.
#[derive(Debug)]
struct Nv {
    codes: Slice,
    scales: Buffer,
    cols: usize,
    rows: usize,
    row_bytes: usize,
    expert_bytes: usize,
    /// Experts the weight holds (1 for a dense one).
    experts: usize,
    scale: f32,
}

fn nv(dev: &MetalDevice, w: &NvWeight) -> Result<Nv> {
    let (buf, off, scales) = binding(dev, w)?;
    Ok(Nv {
        codes: Slice {
            buf: (*buf).clone(),
            off,
        },
        scales: (*scales).clone(),
        cols: w.cols(),
        rows: w.rows(),
        row_bytes: w.row_bytes(),
        expert_bytes: w.expert_bytes(),
        experts: w.experts(),
        scale: w.scale(0),
    })
}

/// The gate, up and down projections of one FFN, routed or not.
#[derive(Debug)]
struct Swiglu {
    gate: Nv,
    up: Nv,
    down: Nv,
}

/// A routed FFN: the router, its experts and the shared expert.
#[derive(Debug)]
struct MoeRefs {
    /// `[experts, hidden]` f32.
    router: Slice,
    experts: Swiglu,
    shared: Swiglu,
    k: usize,
    norm: bool,
    /// Applied to every routing weight (1 when the file sets none).
    scale: f32,
}

#[derive(Debug)]
enum LayerFfn {
    Dense(Box<Swiglu>),
    Moe(Box<MoeRefs>),
}

#[derive(Debug)]
struct LayerRefs {
    attn_norm: Slice,
    ffn_norm: Slice,
    q: Nv,
    k: Nv,
    v: Nv,
    o: Nv,
    q_norm: Slice,
    k_norm: Slice,
    sinks: Slice,
    window: Option<usize>,
    rope: bool,
    ffn: LayerFfn,
}

/// One layer's cache as a step writes and reads it.
#[derive(Clone, Copy)]
struct CacheRefs<'a> {
    /// The absolute position of buffer index 0.
    base: usize,
    /// Positions the buffers hold.
    cap: usize,
    /// f16 buffers (else f32).
    half: bool,
    k: &'a Slice,
    v: &'a Slice,
}

/// The scratch buffers one step runs through, sized for the widest layer.
#[derive(Debug)]
struct Scratch {
    /// The residual stream.
    x: Buffer,
    /// The normalised stream each projection reads.
    xn: Buffer,
    /// The stream after attention, before the FFN.
    h1: Buffer,
    qkv: Buffer,
    q: Buffer,
    /// The attention output, `heads × head_dim`.
    ctx: Buffer,
    attn_out: Buffer,
    router_logits: Buffer,
    ids: Buffer,
    route: Buffer,
    /// Gate·up activations of the routed experts or a dense FFN.
    ffn_h: Buffer,
    /// The routed experts' weighted outputs, one row per slot.
    ys: Buffer,
    /// Gate·up activations of the shared expert.
    shared_h: Buffer,
    /// The shared expert's or a dense FFN's output.
    ffn_out: Buffer,
    logits: Buffer,
    part_o: Buffer,
    part_ml: Buffer,
    /// One slot naming expert 0, for running a single FFN through the
    /// `moe_gate_up` kernel.
    zero_id: Buffer,
}

/// The decode graph of one model on one Metal device.
#[derive(Debug)]
pub(crate) struct DecodeGraph {
    /// The graph's own queue: its steps run in one concurrent encoder each,
    /// which candle's pool does not offer.
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: std::sync::Arc<Pipelines>,
    layers: Vec<LayerRefs>,
    output_norm: Slice,
    output: Nv,
    cos: Slice,
    sin: Slice,
    scratch: Scratch,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    eps: f32,
}

fn swiglu(dev: &MetalDevice, gate: &NvWeight, up: &NvWeight, down: &NvWeight) -> Result<Swiglu> {
    Ok(Swiglu {
        gate: nv(dev, gate)?,
        up: nv(dev, up)?,
        down: nv(dev, down)?,
    })
}

/// One layer's weights and tensors, bound to `dev`.
fn layer_refs(dev: &MetalDevice, layer: &Layer) -> Result<LayerRefs> {
    let a = &layer.attn;
    let ffn = match &layer.ffn {
        Ffn::Dense(d) => LayerFfn::Dense(Box::new(swiglu(dev, &d.gate, &d.up, &d.down)?)),
        Ffn::Moe(m) => LayerFfn::Moe(Box::new(MoeRefs {
            router: slice_of(&m.router)?,
            experts: swiglu(dev, m.experts.gate(), m.experts.up(), m.experts.down())?,
            shared: swiglu(dev, &m.shared.gate, &m.shared.up, &m.shared.down)?,
            k: m.k,
            norm: m.norm,
            scale: if m.scale.abs() > f32::EPSILON && (m.scale - 1.0).abs() > f32::EPSILON {
                m.scale
            } else {
                1.0
            },
        })),
    };
    Ok(LayerRefs {
        attn_norm: slice_of(&layer.attn_norm)?,
        ffn_norm: slice_of(&layer.ffn_norm)?,
        q: nv(dev, &a.q)?,
        k: nv(dev, &a.k)?,
        v: nv(dev, &a.v)?,
        o: nv(dev, &a.o)?,
        q_norm: slice_of(&a.q_norm)?,
        k_norm: slice_of(&a.k_norm)?,
        sinks: slice_of(&a.sinks)?,
        window: a.window,
        rope: a.rope,
        ffn,
    })
}

impl Scratch {
    /// Buffers for `model`'s widest layer and its whole context. Shared
    /// storage: the CPU writes the embedding and reads the logits in place,
    /// and candle's `new_private_buffer` is GPU-only on macOS.
    fn new(dev: &MetalDevice, model: &Model) -> Result<Self> {
        let c = model.config();
        let (qd, kd) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let words = |n: usize| -> Result<Buffer> {
            dev.device()
                .new_buffer(n.max(1) * 4, MTLResourceOptions::StorageModeShared)
                .map_err(|e| err(format!("decode graph: {e}")))
        };
        let chunks = ATTN_MAX_CHUNKS;
        let rep = c.heads / c.kv_heads;
        let zero_id = words(1)?;
        // SAFETY: a fresh shared-storage buffer of one u32, not yet seen by
        // the GPU. Metal buffers are page-aligned, so the cast is aligned.
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            zero_id.contents().cast::<u32>().write(0);
        };
        Ok(Self {
            x: words(c.hidden)?,
            xn: words(c.hidden)?,
            h1: words(c.hidden)?,
            qkv: words(qd + 2 * kd)?,
            q: words(qd)?,
            ctx: words(qd)?,
            attn_out: words(c.hidden)?,
            router_logits: words(c.experts)?,
            ids: words(c.experts_used)?,
            route: words(c.experts_used)?,
            ffn_h: words(c.ffn.max(c.experts_used * c.expert_ffn))?,
            ys: words(c.experts_used * c.hidden)?,
            shared_h: words(c.shared_ffn.max(c.ffn))?,
            ffn_out: words(c.hidden)?,
            logits: words(model.vocab())?,
            part_o: words(c.kv_heads * chunks * rep * ATTN_HEAD_DIM)?,
            part_ml: words(c.kv_heads * chunks * rep * 2)?,
            zero_id,
        })
    }
}

impl DecodeGraph {
    /// The graph for `model`, or `None` when it cannot run it: not on Metal,
    /// a head size other than the attention kernels' `ATTN_HEAD_DIM`, more
    /// query heads per KV head than they take, more experts or picks than
    /// the router kernel takes, or projection widths a fused matvec cannot
    /// split.
    ///
    /// # Errors
    /// A Metal allocation or binding failure.
    pub(crate) fn new(model: &Model) -> Result<Option<Self>> {
        let Device::Metal(dev) = model.device() else {
            return Ok(None);
        };
        let c = model.config();
        let (qd, kd) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let fits = c.head_dim == ATTN_HEAD_DIM
            && c.kv_heads > 0
            && c.heads.is_multiple_of(c.kv_heads)
            && c.heads / c.kv_heads <= ATTN_SG
            && c.experts_used <= ROUTE_MAX_K
            && c.experts <= ROUTE_MAX_EXPERTS
            && c.hidden.is_multiple_of(4)
            && c.hidden <= NORM_THREADS * COMBINE_PER_THREAD
            && qd.is_multiple_of(MV_ROWS_PER_SG)
            && kd.is_multiple_of(MV_ROWS_PER_SG);
        if !fits {
            return Ok(None);
        }
        let layers = model
            .layers()
            .iter()
            .map(|layer| layer_refs(dev, layer))
            .collect::<Result<Vec<_>>>()?;
        let (cos, sin) = model.rope().tables();
        let scratch = Scratch::new(dev, model)?;
        let raw: &ProtocolObject<dyn MTLDevice> = dev.device().as_ref();
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| err("decode graph: no command queue"))?;
        Ok(Some(Self {
            queue,
            pipes: pipelines(dev)?,
            layers,
            output_norm: slice_of(model.output_norm())?,
            output: nv(dev, model.output())?,
            cos: slice_of(cos)?,
            sin: slice_of(sin)?,
            scratch,
            hidden: c.hidden,
            heads: c.heads,
            kv_heads: c.kv_heads,
            vocab: model.vocab(),
            eps: c.rms_eps,
        }))
    }

    /// One token at position `offset`: extends every layer of `cache` by one
    /// position and returns the logits. The caller holds the model's GPU
    /// lock, so nothing else encodes on the device meanwhile.
    ///
    /// # Errors
    /// A cache that does not end at `offset`, a sliding window reaching
    /// below a restored cache's base, or a device failure.
    pub(crate) fn step(
        &self,
        model: &Model,
        token: u32,
        offset: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let device = model.device();
        // Earlier candle work on the cache (a prefill, a restore) may sit in
        // another pool command buffer; it must land before this one reads it.
        device.synchronize()?;
        let s = &self.scratch;
        // SAFETY: `x` holds `hidden` f32 in shared storage, page-aligned, and
        // no GPU work touches it: the synchronize above waited for
        // everything encoded.
        #[allow(clippy::cast_ptr_alignment)]
        let x =
            unsafe { std::slice::from_raw_parts_mut(s.x.contents().cast::<f32>(), self.hidden) };
        model.embed_row(token, x)?;
        let mut grew = false;
        for kv in &mut cache.layers {
            if kv.len() != offset {
                return Err(err(format!(
                    "Flint KV cache holds {}..{}, expected ..{offset}",
                    kv.base(),
                    kv.len()
                )));
            }
            grew |= kv.reserve(1, self.kv_heads, ATTN_HEAD_DIM, device)?;
        }
        if grew {
            // The growth copies are candle kernels in the pool; finish them
            // before the graph writes into the new buffers.
            device.synchronize()?;
        }
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| err("decode graph: no command buffer"))?;
        let raw = cb
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or_else(|| err("decode graph: no compute encoder"))?;
        let enc = Enc(raw);
        // Each layer's FFN ends by normalising for the next layer (the last,
        // for the output head), so only the first layer's norm stands alone.
        if let Some(first) = self.layers.first() {
            self.norm(&enc, &s.x, None, &first.attn_norm, &s.xn);
        }
        for (i, (layer, kv)) in self.layers.iter().zip(&cache.layers).enumerate() {
            let next_norm = self
                .layers
                .get(i + 1)
                .map_or(&self.output_norm, |l| &l.attn_norm);
            let Some((kb, vb)) = kv.buffers() else {
                return Err(err("Flint KV cache has no buffers after a reserve"));
            };
            let (k_slice, v_slice) = (raw_slice(kb)?, raw_slice(vb)?);
            let cap = kb.dims4()?.2;
            let half = match kv.dtype() {
                DType::F16 => true,
                DType::F32 => false,
                other => return Err(err(format!("decode graph: a {other:?} KV cache"))),
            };
            let cache = CacheRefs {
                base: kv.base(),
                cap,
                half,
                k: &k_slice,
                v: &v_slice,
            };
            self.encode_layer(&enc, layer, offset, &cache, next_norm)?;
        }
        self.matvec(&enc, &self.output, &s.xn, &s.logits)?;
        enc.0.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(err(format!("decode graph: {e}")));
        }
        for kv in &mut cache.layers {
            kv.advance(1)?;
        }
        let mut logits = vec![0f32; self.vocab];
        // SAFETY: the logits buffer holds `vocab` f32 in shared storage,
        // page-aligned, and the synchronize above waited for the step that
        // wrote them.
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            std::ptr::copy_nonoverlapping(
                s.logits.contents().cast::<f32>(),
                logits.as_mut_ptr(),
                self.vocab,
            );
        }
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer(
        &self,
        enc: &Enc,
        l: &LayerRefs,
        pos: usize,
        cache: &CacheRefs<'_>,
        next_norm: &Slice,
    ) -> Result<()> {
        let s = &self.scratch;
        let d = ATTN_HEAD_DIM;
        let CacheRefs {
            base,
            cap,
            half,
            k: kc,
            v: vc,
        } = *cache;
        self.qkv(enc, l, &s.xn, &s.qkv)?;
        let slot = pos - base;
        let qk = QkParams {
            heads: u32_of(self.heads)?,
            kv_heads: u32_of(self.kv_heads)?,
            rope: u32::from(l.rope),
            eps: self.eps,
            pos: u32_of(pos)?,
            slot: u32_of(slot)?,
            k_head_stride: u32_of(cap * d)?,
            v_head_stride: u32_of(cap * d)?,
        };
        enc.pipeline(if half {
            &self.pipes.dec_qk_norm_rope_f16
        } else {
            &self.pipes.dec_qk_norm_rope
        });
        enc.buffer(0, &s.qkv, 0);
        enc.buffer(1, &l.q_norm.buf, l.q_norm.off);
        enc.buffer(2, &l.k_norm.buf, l.k_norm.off);
        enc.buffer(3, &self.cos.buf, self.cos.off);
        enc.buffer(4, &self.sin.buf, self.sin.off);
        enc.buffer(5, &s.q, 0);
        enc.buffer(6, &kc.buf, kc.off);
        enc.buffer(7, &vc.buf, vc.off);
        enc.bytes(8, &qk);
        enc.dispatch(grid(self.heads + 2 * self.kv_heads, 1), grid(d, 1));

        // A sliding layer's query sees no key `window` or more behind it.
        let start = l.window.map_or(0, |w| (pos + 1).saturating_sub(w));
        if start < base {
            return Err(err(format!(
                "Flint KV cache starts at {base}, attention needs keys from {start}"
            )));
        }
        let keys = pos + 1 - start;
        let rep = self.heads / self.kv_heads;
        let (chunks, per_tg) = attn_split(keys);
        // The kernel takes the scale as f32, as every score is.
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let attn = AttnParams {
            keys: u32_of(keys)?,
            chunks: u32_of(chunks)?,
            rep: u32_of(rep)?,
            k_head_stride: u32_of(cap * d)?,
            v_head_stride: u32_of(cap * d)?,
            scale: (1.0 / (d as f64).sqrt()) as f32,
            per_tg: u32_of(per_tg)?,
        };
        let first_key = (start - base) * d * if half { 2 } else { 4 };
        enc.pipeline(if half {
            &self.pipes.attn_partial_f16
        } else {
            &self.pipes.attn_partial
        });
        enc.buffer(0, &s.q, 0);
        enc.buffer(1, &kc.buf, kc.off + first_key);
        enc.buffer(2, &vc.buf, vc.off + first_key);
        enc.buffer(3, &s.part_o, 0);
        enc.buffer(4, &s.part_ml, 0);
        enc.bytes(5, &attn);
        enc.dispatch(grid(chunks, self.kv_heads), grid(ATTN_SG * SIMD_WIDTH, 1));
        enc.pipeline(&self.pipes.attn_combine);
        enc.buffer(0, &s.part_o, 0);
        enc.buffer(1, &s.part_ml, 0);
        enc.buffer(2, &l.sinks.buf, l.sinks.off);
        enc.buffer(3, &s.ctx, 0);
        enc.bytes(4, &attn);
        enc.dispatch(grid(self.heads, 1), grid(COMBINE_GROUPS * d, 1));

        self.matvec(enc, &l.o, &s.ctx, &s.attn_out)?;
        self.norm(enc, &s.x, Some((&s.attn_out, &s.h1)), &l.ffn_norm, &s.xn);

        self.encode_ffn(enc, &l.ffn, next_norm)
    }

    /// The FFN half of a layer: from the normalised stream back into the
    /// residual stream.
    fn encode_ffn(&self, enc: &Enc, ffn: &LayerFfn, next_norm: &Slice) -> Result<()> {
        let s = &self.scratch;
        match ffn {
            LayerFfn::Dense(f) => {
                self.gate_up(enc, f, &s.zero_id, 1, &s.ffn_h)?;
                self.matvec(enc, &f.down, &s.ffn_h, &s.ffn_out)?;
                self.combine(enc, 0, Some(&s.ffn_out), next_norm)?;
            }
            LayerFfn::Moe(m) => {
                let MoeRefs {
                    router,
                    experts,
                    shared,
                    k,
                    norm,
                    scale,
                } = &**m;
                let kn = [u32_of(self.hidden)?, u32_of(experts.gate.experts)?];
                enc.pipeline(&self.pipes.dec_mv_f32);
                enc.buffer(0, &router.buf, router.off);
                enc.buffer(1, &s.xn, 0);
                enc.buffer(2, &s.router_logits, 0);
                enc.bytes(3, &kn);
                // The router runs beside the shared expert's gate/up, and the
                // top-k beside its down projection: the two chains share
                // only the normalised stream they both read.
                enc.dispatch_overlap(
                    grid(experts.gate.experts.div_ceil(MV_SG_PER_TG), 1),
                    grid(MV_THREADS, 1),
                );
                self.gate_up(enc, shared, &s.zero_id, 1, &s.shared_h)?;
                // The floor is the smallest f16 normal, exact in f32.
                let rp = RouteParams {
                    experts: u32_of(experts.gate.experts)?,
                    k: u32_of(*k)?,
                    tokens: 1,
                    norm: u32::from(*norm),
                    scale: *scale,
                    min_sum: crate::ffn::MIN_WEIGHT_SUM_F32,
                };
                enc.pipeline(&self.pipes.moe_route);
                enc.buffer(0, &s.router_logits, 0);
                enc.buffer(1, &s.ids, 0);
                enc.buffer(2, &s.route, 0);
                enc.bytes(3, &rp);
                enc.dispatch_overlap(grid(1, 1), grid(ROUTE_THREADS, 1));
                self.matvec(enc, &shared.down, &s.shared_h, &s.ffn_out)?;
                self.gate_up(enc, experts, &s.ids, *k, &s.ffn_h)?;
                self.down(enc, experts, *k)?;
                self.combine(enc, *k, Some(&s.ffn_out), next_norm)?;
            }
        }
        Ok(())
    }

    /// `y = rmsnorm(a (+ b)) * w`; with `add = Some((b, sum))` also stores
    /// `a + b` in `sum`.
    fn norm(&self, enc: &Enc, a: &Buffer, add: Option<(&Buffer, &Buffer)>, w: &Slice, y: &Buffer) {
        let p = NormParams {
            n: u32::try_from(self.hidden).unwrap_or(u32::MAX),
            eps: self.eps,
            add: u32::from(add.is_some()),
        };
        enc.pipeline(&self.pipes.dec_rmsnorm);
        enc.buffer(0, a, 0);
        enc.buffer(1, add.map_or(a, |(b, _)| b), 0);
        enc.buffer(2, &w.buf, w.off);
        enc.buffer(3, add.map_or(y, |(_, sum)| sum), 0);
        enc.buffer(4, y, 0);
        enc.bytes(5, &p);
        enc.dispatch(grid(1, 1), grid(NORM_THREADS, 1));
    }

    fn qkv(&self, enc: &Enc, l: &LayerRefs, x: &Buffer, y: &Buffer) -> Result<()> {
        let p = Mv3Params {
            k: u32_of(l.q.cols)?,
            row_bytes: u32_of(l.q.row_bytes)?,
            n0: u32_of(l.q.rows)?,
            n1: u32_of(l.k.rows)?,
            n2: u32_of(l.v.rows)?,
            s0: l.q.scale,
            s1: l.k.scale,
            s2: l.v.scale,
        };
        let rows = l.q.rows + l.k.rows + l.v.rows;
        enc.pipeline(&self.pipes.dec_mv3);
        enc.buffer(0, &l.q.codes.buf, l.q.codes.off);
        enc.buffer(1, &l.k.codes.buf, l.k.codes.off);
        enc.buffer(2, &l.v.codes.buf, l.v.codes.off);
        enc.buffer(3, x, 0);
        enc.buffer(4, y, 0);
        enc.bytes(5, &p);
        enc.dispatch(grid(rows.div_ceil(MV_ROWS_PER_TG), 1), grid(MV_THREADS, 1));
        Ok(())
    }

    fn matvec(&self, enc: &Enc, w: &Nv, x: &Buffer, y: &Buffer) -> Result<()> {
        let p = MatParams {
            k: u32_of(w.cols)?,
            n: u32_of(w.rows)?,
            tokens: 1,
            row_bytes: u32_of(w.row_bytes)?,
            scale: w.scale,
        };
        enc.pipeline(&self.pipes.mv);
        enc.buffer(0, &w.codes.buf, w.codes.off);
        enc.buffer(1, x, 0);
        enc.buffer(2, y, 0);
        enc.bytes(3, &p);
        enc.dispatch(
            grid(w.rows.div_ceil(MV_ROWS_PER_TG), 1),
            grid(MV_THREADS, 1),
        );
        Ok(())
    }

    fn moe_params(f: &Swiglu, slots: usize) -> Result<MoeParams> {
        Ok(MoeParams {
            d: u32_of(f.gate.cols)?,
            f: u32_of(f.gate.rows)?,
            slots: u32_of(slots)?,
            k_used: u32_of(slots)?,
            experts: u32_of(f.gate.experts)?,
            gu_row_bytes: u32_of(f.gate.row_bytes)?,
            gu_expert_bytes: u32_of(f.gate.expert_bytes)?,
            down_row_bytes: u32_of(f.down.row_bytes)?,
            down_expert_bytes: u32_of(f.down.expert_bytes)?,
        })
    }

    /// `silu(gate x) * up x` for each of `slots` experts named in `ids`,
    /// reading the normalised stream.
    fn gate_up(&self, enc: &Enc, f: &Swiglu, ids: &Buffer, slots: usize, h: &Buffer) -> Result<()> {
        let p = Self::moe_params(f, slots)?;
        enc.pipeline(&self.pipes.moe_gate_up);
        enc.buffer(0, &f.gate.codes.buf, f.gate.codes.off);
        enc.buffer(1, &f.up.codes.buf, f.up.codes.off);
        enc.buffer(2, &self.scratch.xn, 0);
        enc.buffer(3, ids, 0);
        enc.buffer(4, &f.gate.scales, 0);
        enc.buffer(5, &f.up.scales, 0);
        enc.buffer(6, h, 0);
        enc.bytes(7, &p);
        enc.dispatch(
            grid(f.gate.rows.div_ceil(MV_ROWS_PER_TG), slots),
            grid(MV_THREADS, 1),
        );
        Ok(())
    }

    /// The routed experts' down projections, each weighted by its route.
    fn down(&self, enc: &Enc, f: &Swiglu, slots: usize) -> Result<()> {
        let s = &self.scratch;
        let p = Self::moe_params(f, slots)?;
        enc.pipeline(&self.pipes.moe_down);
        enc.buffer(0, &f.down.codes.buf, f.down.codes.off);
        enc.buffer(1, &s.ffn_h, 0);
        enc.buffer(2, &s.ids, 0);
        enc.buffer(3, &f.down.scales, 0);
        enc.buffer(4, &s.route, 0);
        enc.buffer(5, &s.ys, 0);
        enc.bytes(6, &p);
        enc.dispatch(
            grid(f.down.rows.div_ceil(MV_ROWS_PER_TG), slots),
            grid(MV_THREADS, 1),
        );
        Ok(())
    }

    /// `x = h1 + (the slots of ys + extra)`.
    /// `x = h1 + (the slots of ys + extra)`, then `xn = rmsnorm(x) * w`.
    fn combine(&self, enc: &Enc, slots: usize, extra: Option<&Buffer>, w: &Slice) -> Result<()> {
        let s = &self.scratch;
        let p = CombineParams {
            d: u32_of(self.hidden)?,
            slots: u32_of(slots)?,
            extra: u32::from(extra.is_some()),
            eps: self.eps,
        };
        enc.pipeline(&self.pipes.dec_combine_norm);
        enc.buffer(0, &s.h1, 0);
        enc.buffer(1, &s.ys, 0);
        enc.buffer(2, extra.unwrap_or(&s.ys), 0);
        enc.buffer(3, &s.x, 0);
        enc.buffer(4, &w.buf, w.off);
        enc.buffer(5, &s.xn, 0);
        enc.bytes(6, &p);
        enc.dispatch(grid(1, 1), grid(NORM_THREADS, 1));
        Ok(())
    }
}

/// Mirrors `NormParams` in `decode.metal`.
#[repr(C)]
struct NormParams {
    n: u32,
    eps: f32,
    add: u32,
}

/// Mirrors `Mv3Params` in `decode.metal`.
#[repr(C)]
struct Mv3Params {
    k: u32,
    row_bytes: u32,
    n0: u32,
    n1: u32,
    n2: u32,
    s0: f32,
    s1: f32,
    s2: f32,
}

/// Mirrors `QkParams` in `decode.metal`.
#[repr(C)]
struct QkParams {
    heads: u32,
    kv_heads: u32,
    rope: u32,
    eps: f32,
    pos: u32,
    slot: u32,
    k_head_stride: u32,
    v_head_stride: u32,
}

/// Mirrors `CombineParams` in `decode.metal`.
#[repr(C)]
struct CombineParams {
    d: u32,
    slots: u32,
    extra: u32,
    eps: f32,
}
