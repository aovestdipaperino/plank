//! The Metal side of the NVFP4 kernels: one library compiled from
//! `nvfp4.metal` per device, weights bound to buffers, and the dispatches.
//!
//! A file-backed weight binds to a single no-copy buffer over the whole
//! mapping when Metal accepts one that large, so the weights are never
//! copied; otherwise, and for owned weights, each weight is copied once into
//! its own buffer.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use block2::RcBlock;
use candle_core::backend::BackendStorage;
use candle_core::{DType, MetalDevice, MetalStorage, Shape};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Library};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize};

use crate::file::MappedFile;
use crate::nvfp4::ops::{MatDims, MoeDims};
use crate::nvfp4::tensor::{MoeExperts, NvWeight};

/// The kernels, without their launch constants: [`source`] prepends those.
const SOURCE: &str = include_str!("nvfp4.metal");
/// The single-token attention and routing kernels, compiled into the same
/// library.
const FUSED_SOURCE: &str = include_str!("fused.metal");
/// The decode graph's own kernels, compiled after the two above.
const DECODE_SOURCE: &str = include_str!("decode.metal");
/// The `MoE` grouped by expert, for many tokens.
const GROUPED_SOURCE: &str = include_str!("moe_grouped.metal");

/// From this many tokens up, a matmul runs the tiled `mm_nvfp4` kernel.
pub(crate) const MM_MIN_TOKENS: usize = 8;
/// The simdgroup width every kernel assumes.
pub(crate) const SIMD_WIDTH: usize = 32;
/// Weight rows each simdgroup of the matvec kernels computes.
pub(crate) const MV_ROWS_PER_SG: usize = 2;
/// Simdgroups per matvec threadgroup.
pub(crate) const MV_SG_PER_TG: usize = 2;
/// Weight rows per matvec threadgroup.
pub(crate) const MV_ROWS_PER_TG: usize = MV_ROWS_PER_SG * MV_SG_PER_TG;
/// Threads per matvec threadgroup.
pub(crate) const MV_THREADS: usize = SIMD_WIDTH * MV_SG_PER_TG;
/// Weight rows and tokens per `mm_nvfp4` tile (`MM_TN`, `MM_TT`).
const MM_TILE: usize = 32;
/// Four simdgroups, in a 2x2 grid of 16x16 output patches.
const MM_THREADS: usize = 4 * SIMD_WIDTH;
/// Threads per `moe_sum` threadgroup, one output column each.
const SUM_THREADS: usize = 256;
/// The head dimension the attention kernels take: four values per lane.
pub(crate) const ATTN_HEAD_DIM: usize = 4 * SIMD_WIDTH;
/// Keys per block of `attn_dec_partial`: one per lane.
pub(crate) const ATTN_BLOCK: usize = SIMD_WIDTH;
/// The most `attn_dec_partial` threadgroups one KV head's keys split into:
/// enough to fill the GPU, few enough that `attn_dec_combine` merges them
/// quickly at any depth.
pub(crate) const ATTN_MAX_CHUNKS: usize = 64;
/// Queries per `attn_prefill` threadgroup: four simdgroups of 8 rows.
pub(crate) const PF_Q: usize = 32;
/// Keys per `attn_prefill` block.
const PF_K: usize = 32;
/// Threads per `attn_prefill` threadgroup.
const PF_THREADS: usize = 4 * SIMD_WIDTH;
/// Groups of `ATTN_HEAD_DIM` threads in an `attn_dec_combine` threadgroup,
/// each summing every `COMBINE_GROUPS`-th run.
pub(crate) const COMBINE_GROUPS: usize = 8;

/// How `keys` keys split over `attn_dec_partial` threadgroups:
/// `(chunks, keys per threadgroup)`, as many single-block runs as fit in
/// [`ATTN_MAX_CHUNKS`], longer runs beyond.
pub(crate) fn attn_split(keys: usize) -> (usize, usize) {
    let blocks = keys.div_ceil(ATTN_BLOCK).max(1);
    let per_tg = blocks.div_ceil(blocks.min(ATTN_MAX_CHUNKS)) * ATTN_BLOCK;
    (keys.div_ceil(per_tg), per_tg)
}
/// Simdgroups per `attn_dec_partial` threadgroup, which is also the most
/// query heads one KV head may serve.
pub(crate) const ATTN_SG: usize = 8;
/// The most experts `moe_route` picks per token.
pub(crate) const ROUTE_MAX_K: usize = 8;
/// Threads per `moe_route` threadgroup, which routes one token.
pub(crate) const ROUTE_THREADS: usize = 256;
/// Experts each `moe_route` thread holds, so a token may have up to
/// `ROUTE_THREADS * ROUTE_PER_THREAD` experts.
const ROUTE_PER_THREAD: usize = 4;
/// The most experts `moe_route` takes.
pub(crate) const ROUTE_MAX_EXPERTS: usize = ROUTE_THREADS * ROUTE_PER_THREAD;
/// Values of the residual stream each `dec_combine_norm` thread holds, so
/// the graph takes a model width up to `NORM_THREADS * COMBINE_PER_THREAD`.
pub(crate) const COMBINE_PER_THREAD: usize = 4;
/// Slots per tile of the grouped `MoE` kernels.
const MOE_TT: usize = 16;
/// Weight rows per `moe_mm_gate_up` threadgroup: an 8-row strip per
/// simdgroup.
const MOE_GU_TN: usize = 32;
/// Weight rows per `moe_mm_down` threadgroup: a 16-row strip per simdgroup.
const MOE_DN_TN: usize = 64;
/// Threads in `moe_group`'s one threadgroup.
const GROUP_THREADS: usize = 1024;
/// The most experts `moe_group` sorts slots over.
const GROUP_MAX_EXPERTS: usize = 1024;
/// Threads in the decode graph's one `dec_rmsnorm` threadgroup.
pub(crate) const NORM_THREADS: usize = 1024;

// The 2x2 simdgroups of 16x16 patches cover exactly one tile.
const _: () = assert!(MM_TILE == 2 * 16);
// Each `mm_nvfp4` thread loads one 16-value run (four per 64-value block) of
// one weight row and one token, so the threads must cover the tile's rows.
const _: () = assert!(MM_THREADS == 4 * MM_TILE);

/// The first line of the generated prelude.
const PRELUDE_MARK: &str = "// Launch constants, generated by metal.rs.\n";

/// The shader source: a `#define` prelude made from the constants above, so
/// the kernels and their dispatches cannot disagree, then `nvfp4.metal`.
fn source() -> String {
    format!(
        "{PRELUDE_MARK}\
         #define SIMD_WIDTH {SIMD_WIDTH}\n\
         #define MV_ROWS_PER_SG {MV_ROWS_PER_SG}\n\
         #define MV_SG_PER_TG {MV_SG_PER_TG}\n\
         #define MM_TN {MM_TILE}\n\
         #define MM_TT {MM_TILE}\n\
         #define MM_THREADS {MM_THREADS}\n\
         #define SUM_THREADS {SUM_THREADS}\n\
         #define ATTN_HEAD_DIM {ATTN_HEAD_DIM}\n\
         #define ATTN_BLOCK {ATTN_BLOCK}\n\
         #define COMBINE_GROUPS {COMBINE_GROUPS}\n\
         #define PF_Q {PF_Q}\n\
         #define PF_K {PF_K}\n\
         #define PF_THREADS {PF_THREADS}\n\
         #define ATTN_SG {ATTN_SG}\n\
         #define ROUTE_MAX_K {ROUTE_MAX_K}\n\
         #define ROUTE_THREADS {ROUTE_THREADS}\n\
         #define ROUTE_PER_THREAD {ROUTE_PER_THREAD}\n\
         #define NORM_THREADS {NORM_THREADS}\n\
         #define COMBINE_PER_THREAD {COMBINE_PER_THREAD}\n\
         #define GROUP_THREADS {GROUP_THREADS}\n\
         #define MOE_TT {MOE_TT}\n\
         #define MOE_GU_TN {MOE_GU_TN}\n\
         #define MOE_DN_TN {MOE_DN_TN}\n\
         #define GROUP_MAX_EXPERTS {GROUP_MAX_EXPERTS}\n\
         #line 1 \"nvfp4.metal\"\n\
         {SOURCE}\n\
         #line 1 \"fused.metal\"\n\
         {FUSED_SOURCE}\n\
         #line 1 \"decode.metal\"\n\
         {DECODE_SOURCE}\n\
         #line 1 \"moe_grouped.metal\"\n\
         {GROUPED_SOURCE}"
    )
}

/// The compiled kernels of one device.
#[derive(Debug)]
pub(crate) struct Pipelines {
    pub mv: ComputePipeline,
    pub mm: ComputePipeline,
    pub moe_gate_up: ComputePipeline,
    pub moe_down: ComputePipeline,
    pub moe_sum: ComputePipeline,
    pub attn_partial: ComputePipeline,
    pub attn_partial_f16: ComputePipeline,
    pub attn_combine: ComputePipeline,
    pub moe_route: ComputePipeline,
    pub dec_rmsnorm: ComputePipeline,
    pub dec_mv3: ComputePipeline,
    pub dec_qk_norm_rope: ComputePipeline,
    pub dec_qk_norm_rope_f16: ComputePipeline,
    pub dec_mv_f32: ComputePipeline,
    pub dec_combine_norm: ComputePipeline,
    pub moe_group: ComputePipeline,
    pub attn_prefill: ComputePipeline,
    pub attn_prefill_f16: ComputePipeline,
    pub moe_mm_gate_up: ComputePipeline,
    pub moe_mm_down: ComputePipeline,
}

static PIPELINES: Mutex<Vec<(u64, Arc<Pipelines>)>> = Mutex::new(Vec::new());

fn metal_err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::msg(format!("nvfp4 metal: {e}"))
}

/// The kernels for `dev`, compiled on first use. Compiler and pipeline
/// failures come back as errors (never a panic, which would also poison the
/// cache lock).
pub(crate) fn pipelines(dev: &MetalDevice) -> candle_core::Result<Arc<Pipelines>> {
    let id = dev.device().registry_id();
    let mut cache = PIPELINES.lock().map_err(metal_err)?;
    if let Some((_, p)) = cache.iter().find(|(i, _)| *i == id) {
        return Ok(Arc::clone(p));
    }
    let raw: &ProtocolObject<dyn MTLDevice> = dev.device().as_ref();
    let lib = raw
        .newLibraryWithSource_options_error(&NSString::from_str(&source()), None)
        .map_err(|e| metal_err(format!("nvfp4.metal does not compile: {e}")))?;
    let lib = Library::new(lib);
    let get = |name: &str, threads: usize| -> candle_core::Result<ComputePipeline> {
        let f = lib.get_function(name, None).map_err(metal_err)?;
        let pipe = raw
            .newComputePipelineStateWithFunction_error(f.as_ref())
            .map_err(|e| metal_err(format!("{name}: {e}")))?;
        // The kernels reduce with `simd_sum` over 32 lanes and size their
        // threadgroups in multiples of 32.
        let width = pipe.threadExecutionWidth();
        if width != SIMD_WIDTH {
            return Err(metal_err(format!(
                "{name}: simdgroups are {width} wide, the kernels need {SIMD_WIDTH}"
            )));
        }
        // Each kernel is dispatched with a fixed threadgroup size.
        let most = pipe.maxTotalThreadsPerThreadgroup();
        if most < threads {
            return Err(metal_err(format!(
                "{name}: threadgroups hold at most {most} threads, the kernel needs {threads}"
            )));
        }
        Ok(ComputePipeline::new(pipe))
    };
    let p = Arc::new(Pipelines {
        mv: get("mv_nvfp4", MV_THREADS)?,
        mm: get("mm_nvfp4", MM_THREADS)?,
        moe_gate_up: get("moe_gate_up", MV_THREADS)?,
        moe_down: get("moe_down", MV_THREADS)?,
        moe_sum: get("moe_sum", SUM_THREADS)?,
        attn_partial: get("attn_dec_partial_f32", ATTN_SG * SIMD_WIDTH)?,
        attn_partial_f16: get("attn_dec_partial_f16", ATTN_SG * SIMD_WIDTH)?,
        attn_combine: get("attn_dec_combine", COMBINE_GROUPS * ATTN_HEAD_DIM)?,
        moe_route: get("moe_route", ROUTE_THREADS)?,
        dec_rmsnorm: get("dec_rmsnorm", NORM_THREADS)?,
        dec_mv3: get("dec_mv3_nvfp4", MV_THREADS)?,
        dec_qk_norm_rope: get("dec_qk_norm_rope_f32", ATTN_HEAD_DIM)?,
        dec_qk_norm_rope_f16: get("dec_qk_norm_rope_f16", ATTN_HEAD_DIM)?,
        dec_mv_f32: get("dec_mv_f32", MV_THREADS)?,
        dec_combine_norm: get("dec_combine_norm", NORM_THREADS)?,
        moe_group: get("moe_group", GROUP_THREADS)?,
        attn_prefill: get("attn_prefill_f32", PF_THREADS)?,
        attn_prefill_f16: get("attn_prefill_f16", PF_THREADS)?,
        moe_mm_gate_up: get("moe_mm_gate_up", MM_THREADS)?,
        moe_mm_down: get("moe_mm_down", MM_THREADS)?,
    });
    cache.push((id, Arc::clone(&p)));
    Ok(p)
}

/// Where a weight lives on the GPU.
#[derive(Debug)]
pub(crate) struct Binding {
    /// The `registry_id` of the device the buffers belong to.
    pub device: u64,
    pub buf: Arc<Buffer>,
    pub offset: usize,
    /// The weight's scales as f32, one per expert.
    pub scales: Arc<Buffer>,
}

fn copied(dev: &MetalDevice, bytes: &[u8]) -> candle_core::Result<Arc<Buffer>> {
    let b = dev
        .device()
        .new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
        .map_err(metal_err)?;
    Ok(Arc::new(b))
}

/// `maxBufferLength` of `dev`.
pub(crate) fn max_buffer_len_of(dev: &MetalDevice) -> Option<u64> {
    let raw: &ProtocolObject<dyn MTLDevice> = dev.device().as_ref();
    u64::try_from(raw.maxBufferLength()).ok()
}

/// `maxBufferLength` of Metal device 0, or `None` when none opens.
pub(crate) fn max_buffer_len() -> Option<u64> {
    let candle_core::Device::Metal(dev) = candle_core::Device::new_metal(0).ok()? else {
        return None;
    };
    max_buffer_len_of(&dev)
}

/// One buffer over the whole mapping, without a copy, or `None` when Metal
/// will not make one (too large for `maxBufferLength`, or refused).
fn no_copy(dev: &MetalDevice, file: &MappedFile) -> Option<Arc<Buffer>> {
    let bytes = file.bytes();
    // SAFETY: sysconf has no preconditions.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    // The mapping is page-aligned and extends to the end of its last page.
    let len = bytes.len().div_ceil(page) * page;
    let raw: &ProtocolObject<dyn MTLDevice> = dev.device().as_ref();
    if len > raw.maxBufferLength() {
        return None;
    }
    let ptr = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())?;
    // The deallocator owns a clone of the mapping. Metal keeps the block
    // until it frees the buffer, which is after every command buffer that
    // retained the buffer has completed, so the pages stay mapped for as
    // long as the GPU can read them, however early the file is dropped.
    let keep = Arc::clone(file.map());
    let release = RcBlock::new(move |_: NonNull<c_void>, _: usize| {
        let _ = &keep;
    });
    // SAFETY: `ptr..ptr+len` is the page-aligned mapping rounded up to its
    // last page, all of it mapped and read-only for the mapping's lifetime.
    // That lifetime covers the buffer's: the deallocator block above holds an
    // `Arc` of the `Mmap`, released only when Metal frees the buffer. Metal
    // never writes the memory (no kernel binds a weight for writing) and the
    // deallocator does not free it, the `Mmap` drop does.
    // Metal may run the deallocator on any thread. That is sound only because
    // the block captures nothing but an `Arc<Mmap>`, which is `Send + Sync`:
    // do not add captures to it.
    let raw_buf = unsafe {
        raw.newBufferWithBytesNoCopy_length_options_deallocator(
            ptr,
            len,
            MTLResourceOptions::StorageModeShared,
            Some(&release),
        )
    }?;
    Some(Arc::new(Buffer::new(raw_buf)))
}

fn other_device(bound: u64, asked: u64) -> candle_core::Error {
    metal_err(format!(
        "weight is bound to another Metal device ({bound:#x}, not {asked:#x}); \
         a weight serves one device"
    ))
}

/// The weight's buffer, byte offset and scale buffer, bound on first use.
/// A weight, and the mapping it borrows, bind to the first device that uses
/// them; asking for another device is an error.
pub(crate) fn binding(
    dev: &MetalDevice,
    w: &NvWeight,
) -> candle_core::Result<(Arc<Buffer>, usize, Arc<Buffer>)> {
    let id = dev.device().registry_id();
    let mut slot = w.metal.lock().map_err(metal_err)?;
    if let Some(b) = slot.as_ref() {
        if b.device != id {
            return Err(other_device(b.device, id));
        }
        return Ok((Arc::clone(&b.buf), b.offset, Arc::clone(&b.scales)));
    }
    let shared = match w.mapping() {
        Some((file, offset)) => {
            let mut whole = file.metal.lock().map_err(metal_err)?;
            let (bound, buf) = whole.get_or_insert_with(|| (id, no_copy(dev, file)));
            if *bound != id {
                return Err(other_device(*bound, id));
            }
            buf.clone().map(|b| (b, offset))
        }
        None => None,
    };
    let (buf, offset) = match shared {
        Some(b) => b,
        None => (copied(dev, w.bytes())?, 0),
    };
    let scale_bytes: Vec<u8> = w.scales().iter().flat_map(|s| s.to_le_bytes()).collect();
    let scales = copied(dev, &scale_bytes)?;
    *slot = Some(Binding {
        device: id,
        buf: Arc::clone(&buf),
        offset,
        scales: Arc::clone(&scales),
    });
    Ok((buf, offset, scales))
}

/// `n` as a u32 kernel parameter, or an error when it does not fit.
pub(crate) fn u32_of(n: usize) -> candle_core::Result<u32> {
    u32::try_from(n).map_err(|_| metal_err(format!("{n} does not fit a u32 kernel parameter")))
}

/// A 2-D Metal size (depth 1).
pub(crate) fn grid(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

/// Mirrors `MatParams` in the shader.
#[repr(C)]
pub(crate) struct MatParams {
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) tokens: u32,
    pub(crate) row_bytes: u32,
    pub(crate) scale: f32,
}

/// `x [tokens, cols]` times a dense weight on the GPU.
pub(crate) fn matmul(
    w: &NvWeight,
    x: &MetalStorage,
    dims: MatDims,
) -> candle_core::Result<(MetalStorage, Shape)> {
    if x.dtype() != DType::F32 {
        candle_core::bail!("nvfp4 matmul wants f32 input");
    }
    let dev = x.device();
    let pipes = pipelines(dev)?;
    let (wbuf, woff, _) = binding(dev, w)?;
    let (tokens, n) = (dims.tokens, w.rows());
    let out = dev.new_buffer(tokens * n, DType::F32, "nvfp4_matmul")?;
    let params = MatParams {
        k: u32_of(w.cols())?,
        n: u32_of(n)?,
        tokens: u32_of(tokens)?,
        row_bytes: u32_of(w.row_bytes())?,
        scale: w.scale(0),
    };
    let enc = dev.command_encoder()?;
    enc.set_label("nvfp4_matmul");
    enc.set_buffer(0, Some(&wbuf), woff);
    enc.set_buffer(1, Some(x.buffer()), dims.start * 4);
    enc.set_buffer(2, Some(&out), 0);
    enc.set_bytes(3, &params);
    if tokens < MM_MIN_TOKENS {
        enc.set_compute_pipeline_state(&pipes.mv);
        enc.dispatch_thread_groups(
            grid(n.div_ceil(MV_ROWS_PER_TG), tokens),
            grid(MV_THREADS, 1),
        );
    } else {
        enc.set_compute_pipeline_state(&pipes.mm);
        enc.dispatch_thread_groups(
            grid(n.div_ceil(MM_TILE), tokens.div_ceil(MM_TILE)),
            grid(MM_THREADS, 1),
        );
    }
    // Each encoder must end before the next `command_encoder()`, or candle
    // hands out a different pool command buffer and the passes lose order.
    drop(enc);
    Ok((
        MetalStorage::new(out, dev.clone(), tokens * n, DType::F32),
        Shape::from((tokens, n)),
    ))
}

/// Mirrors `MoeParams` in the shader.
#[repr(C)]
pub(crate) struct MoeParams {
    pub(crate) d: u32,
    pub(crate) f: u32,
    pub(crate) slots: u32,
    pub(crate) k_used: u32,
    pub(crate) experts: u32,
    pub(crate) gu_row_bytes: u32,
    pub(crate) gu_expert_bytes: u32,
    pub(crate) down_row_bytes: u32,
    pub(crate) down_expert_bytes: u32,
}

/// The gathered `MoE` on the GPU: gate/up per slot, down per slot weighted
/// by the route, then the slots summed in order.
pub(crate) fn moe(
    m: &MoeExperts,
    x: &MetalStorage,
    ids: &MetalStorage,
    route: &MetalStorage,
    dims: MoeDims,
) -> candle_core::Result<(MetalStorage, Shape)> {
    if x.dtype() != DType::F32 || ids.dtype() != DType::U32 || route.dtype() != DType::F32 {
        candle_core::bail!("moe wants f32 x, u32 ids and f32 route");
    }
    let dev = x.device();
    let pipes = pipelines(dev)?;
    let (gate, gate_off, gate_scales) = binding(dev, m.gate())?;
    let (up, up_off, up_scales) = binding(dev, m.up())?;
    let (down, down_off, down_scales) = binding(dev, m.down())?;
    let (tokens, hidden, ffn) = (dims.tokens, m.hidden(), m.ffn());
    let slots = tokens * dims.k_used;
    let params = MoeParams {
        d: u32_of(hidden)?,
        f: u32_of(ffn)?,
        slots: u32_of(slots)?,
        k_used: u32_of(dims.k_used)?,
        experts: u32_of(m.count())?,
        gu_row_bytes: u32_of(m.gate().row_bytes())?,
        gu_expert_bytes: u32_of(m.gate().expert_bytes())?,
        down_row_bytes: u32_of(m.down().row_bytes())?,
        down_expert_bytes: u32_of(m.down().expert_bytes())?,
    };
    let h = dev.new_buffer(slots * ffn, DType::F32, "nvfp4_moe_h")?;
    let ys = dev.new_buffer(slots * hidden, DType::F32, "nvfp4_moe_ys")?;
    let out = dev.new_buffer(tokens * hidden, DType::F32, "nvfp4_moe_out")?;
    if tokens >= MM_MIN_TOKENS && m.count() <= GROUP_MAX_EXPERTS {
        grouped(
            &pipes,
            dev,
            [
                (&*gate, gate_off, &*gate_scales),
                (&*up, up_off, &*up_scales),
                (&*down, down_off, &*down_scales),
            ],
            (x, ids, route),
            dims,
            &params,
            (&h, &ys),
        )?;
    } else {
        // Each scoped encoder below ends before the next `command_encoder()`;
        // otherwise candle hands out a different pool command buffer and the
        // three passes lose their order.
        {
            let enc = dev.command_encoder()?;
            enc.set_label("nvfp4_moe_gate_up");
            enc.set_compute_pipeline_state(&pipes.moe_gate_up);
            enc.set_buffer(0, Some(&gate), gate_off);
            enc.set_buffer(1, Some(&up), up_off);
            enc.set_buffer(2, Some(x.buffer()), dims.x_start * 4);
            enc.set_buffer(3, Some(ids.buffer()), dims.ids_start * 4);
            enc.set_buffer(4, Some(&gate_scales), 0);
            enc.set_buffer(5, Some(&up_scales), 0);
            enc.set_buffer(6, Some(&h), 0);
            enc.set_bytes(7, &params);
            enc.dispatch_thread_groups(
                grid(ffn.div_ceil(MV_ROWS_PER_TG), slots),
                grid(MV_THREADS, 1),
            );
        }
        // gate/up's encoder has ended (see above).
        {
            let enc = dev.command_encoder()?;
            enc.set_label("nvfp4_moe_down");
            enc.set_compute_pipeline_state(&pipes.moe_down);
            enc.set_buffer(0, Some(&down), down_off);
            enc.set_buffer(1, Some(&h), 0);
            enc.set_buffer(2, Some(ids.buffer()), dims.ids_start * 4);
            enc.set_buffer(3, Some(&down_scales), 0);
            enc.set_buffer(4, Some(route.buffer()), dims.route_start * 4);
            enc.set_buffer(5, Some(&ys), 0);
            enc.set_bytes(6, &params);
            enc.dispatch_thread_groups(
                grid(hidden.div_ceil(MV_ROWS_PER_TG), slots),
                grid(MV_THREADS, 1),
            );
        }
    }
    // down's encoder has ended (see above).
    {
        let enc = dev.command_encoder()?;
        enc.set_label("nvfp4_moe_sum");
        enc.set_compute_pipeline_state(&pipes.moe_sum);
        enc.set_buffer(0, Some(&ys), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.set_bytes(2, &params);
        enc.dispatch_thread_groups(
            grid(hidden.div_ceil(SUM_THREADS), tokens),
            grid(SUM_THREADS, 1),
        );
    }
    Ok((
        MetalStorage::new(out, dev.clone(), tokens * hidden, DType::F32),
        Shape::from((tokens, hidden)),
    ))
}

/// Mirrors `PrefillParams` in the shader.
#[repr(C)]
struct PrefillParams {
    t: u32,
    t_pad: u32,
    q_pos0: u32,
    key_base: u32,
    window: u32,
    rep: u32,
    kv_head_stride: u32,
    scale: f32,
}

/// Where prefill attention's queries sit: `[heads][t_pad][ATTN_HEAD_DIM]`
/// f32 from `offset` bytes, `t` of each head's `t_pad` rows live, the first
/// at absolute position `pos0`.
pub(crate) struct PrefillQueries<'a> {
    pub storage: &'a MetalStorage,
    pub offset: usize,
    pub t: usize,
    pub t_pad: usize,
    pub pos0: usize,
}

/// Attention for many queries over a cache (`attn_prefill`): `k`/`v` hold
/// absolute positions from `key_base` up to the last query's, `window` is
/// 0 for none. Returns `[heads][t_pad][ATTN_HEAD_DIM]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attention(
    q: &PrefillQueries<'_>,
    k: &KvOperand<'_>,
    v: &KvOperand<'_>,
    sinks: (&MetalStorage, usize),
    (heads, kv_heads): (usize, usize),
    key_base: usize,
    window: usize,
    scale: f32,
) -> candle_core::Result<MetalStorage> {
    let kv_dtype = k.storage.dtype();
    if q.storage.dtype() != DType::F32
        || v.storage.dtype() != kv_dtype
        || !matches!(kv_dtype, DType::F32 | DType::F16)
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || k.head_stride != v.head_stride
        || !q.t_pad.is_multiple_of(PF_Q)
        || q.t > q.t_pad
    {
        candle_core::bail!("prefill attention cannot take these operands");
    }
    let dev = q.storage.device();
    let pipes = pipelines(dev)?;
    let params = PrefillParams {
        t: u32_of(q.t)?,
        t_pad: u32_of(q.t_pad)?,
        q_pos0: u32_of(q.pos0)?,
        key_base: u32_of(key_base)?,
        window: u32_of(window)?,
        rep: u32_of(heads / kv_heads)?,
        kv_head_stride: u32_of(k.head_stride)?,
        scale,
    };
    let n = heads * q.t_pad * ATTN_HEAD_DIM;
    let out = dev.new_buffer(n, DType::F32, "attn_prefill_out")?;
    {
        let enc = dev.command_encoder()?;
        enc.set_label("attn_prefill");
        enc.set_compute_pipeline_state(if kv_dtype == DType::F16 {
            &pipes.attn_prefill_f16
        } else {
            &pipes.attn_prefill
        });
        enc.set_buffer(0, Some(q.storage.buffer()), q.offset);
        enc.set_buffer(1, Some(k.storage.buffer()), k.offset);
        enc.set_buffer(2, Some(v.storage.buffer()), v.offset);
        enc.set_buffer(3, Some(sinks.0.buffer()), sinks.1);
        enc.set_buffer(4, Some(&out), 0);
        enc.set_bytes(5, &params);
        enc.dispatch_thread_groups(grid(q.t_pad / PF_Q, heads), grid(PF_THREADS, 1));
    }
    Ok(MetalStorage::new(out, dev.clone(), n, DType::F32))
}

/// A weight bound for a kernel: its buffer, byte offset and scale buffer.
type Bound<'a> = (&'a Buffer, usize, &'a Buffer);

/// The `MoE`'s gate/up and down for many tokens, grouped by expert
/// (`moe_grouped.metal`): `moe_group` sorts the slots by expert and lists
/// tiles of one expert's slots, then one tiled matmul per (row tile, tile)
/// reads each expert's weights once per tile instead of once per slot.
/// Writes `h` and `ys` per slot, as the per-slot kernels do, for `moe_sum`.
fn grouped(
    pipes: &Pipelines,
    dev: &MetalDevice,
    [gate, up, down]: [Bound<'_>; 3],
    (x, ids, route): (&MetalStorage, &MetalStorage, &MetalStorage),
    dims: MoeDims,
    params: &MoeParams,
    (h, ys): (&Buffer, &Buffer),
) -> candle_core::Result<()> {
    let slots = dims.tokens * dims.k_used;
    let (hidden, ffn) = (params.d as usize, params.f as usize);
    // At most one partial tile per expert beyond the full ones.
    let max_tiles = slots.div_ceil(MOE_TT) + slots.min(params.experts as usize);
    let perm = dev.new_buffer(slots, DType::U32, "nvfp4_moe_perm")?;
    // A `uint3` takes 16 bytes in a Metal buffer.
    let tiles = dev.new_buffer(max_tiles * 4, DType::U32, "nvfp4_moe_tiles")?;
    let n_tiles = dev.new_buffer(1, DType::U32, "nvfp4_moe_n_tiles")?;
    // Each scoped encoder ends before the next `command_encoder()`, so the
    // passes keep their order (see `moe`).
    {
        let enc = dev.command_encoder()?;
        enc.set_label("nvfp4_moe_group");
        enc.set_compute_pipeline_state(&pipes.moe_group);
        enc.set_buffer(0, Some(ids.buffer()), dims.ids_start * 4);
        enc.set_buffer(1, Some(&perm), 0);
        enc.set_buffer(2, Some(&tiles), 0);
        enc.set_buffer(3, Some(&n_tiles), 0);
        enc.set_buffer(4, Some(ys), 0);
        enc.set_bytes(5, params);
        enc.dispatch_thread_groups(grid(1, 1), grid(GROUP_THREADS, 1));
    }
    {
        let enc = dev.command_encoder()?;
        enc.set_label("nvfp4_moe_mm_gate_up");
        enc.set_compute_pipeline_state(&pipes.moe_mm_gate_up);
        enc.set_buffer(0, Some(gate.0), gate.1);
        enc.set_buffer(1, Some(up.0), up.1);
        enc.set_buffer(2, Some(x.buffer()), dims.x_start * 4);
        enc.set_buffer(3, Some(&perm), 0);
        enc.set_buffer(4, Some(&tiles), 0);
        enc.set_buffer(5, Some(&n_tiles), 0);
        enc.set_buffer(6, Some(gate.2), 0);
        enc.set_buffer(7, Some(up.2), 0);
        enc.set_buffer(8, Some(h), 0);
        enc.set_bytes(9, params);
        enc.dispatch_thread_groups(
            grid(ffn.div_ceil(MOE_GU_TN), max_tiles),
            grid(MM_THREADS, 1),
        );
    }
    {
        let enc = dev.command_encoder()?;
        enc.set_label("nvfp4_moe_mm_down");
        enc.set_compute_pipeline_state(&pipes.moe_mm_down);
        enc.set_buffer(0, Some(down.0), down.1);
        enc.set_buffer(1, Some(h), 0);
        enc.set_buffer(2, Some(&perm), 0);
        enc.set_buffer(3, Some(&tiles), 0);
        enc.set_buffer(4, Some(&n_tiles), 0);
        enc.set_buffer(5, Some(down.2), 0);
        enc.set_buffer(6, Some(route.buffer()), dims.route_start * 4);
        enc.set_buffer(7, Some(ys), 0);
        enc.set_bytes(8, params);
        enc.dispatch_thread_groups(
            grid(hidden.div_ceil(MOE_DN_TN), max_tiles),
            grid(MM_THREADS, 1),
        );
    }
    Ok(())
}

/// Mirrors `AttnParams` in the shader.
#[repr(C)]
pub(crate) struct AttnParams {
    pub(crate) keys: u32,
    pub(crate) chunks: u32,
    pub(crate) rep: u32,
    pub(crate) k_head_stride: u32,
    pub(crate) v_head_stride: u32,
    pub(crate) scale: f32,
    pub(crate) per_tg: u32,
}

/// Where single-token attention reads one KV operand: the buffer (f32 or
/// f16), the byte offset of head 0's first key, and the values between
/// heads.
pub(crate) struct KvOperand<'a> {
    pub storage: &'a MetalStorage,
    pub offset: usize,
    pub head_stride: usize,
}

/// Single-token attention on the GPU: `q` is `heads × ATTN_HEAD_DIM`
/// (contiguous from `q_offset` bytes), `k`/`v` hold `keys` positions per KV
/// head, `sinks` one logit per head. Returns `[heads × ATTN_HEAD_DIM]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_attention(
    q: &MetalStorage,
    q_offset: usize,
    k: &KvOperand<'_>,
    v: &KvOperand<'_>,
    sinks: (&MetalStorage, usize),
    heads: usize,
    kv_heads: usize,
    keys: usize,
    scale: f32,
) -> candle_core::Result<MetalStorage> {
    let kv_dtype = k.storage.dtype();
    if q.dtype() != DType::F32
        || sinks.0.dtype() != DType::F32
        || v.storage.dtype() != kv_dtype
        || !matches!(kv_dtype, DType::F32 | DType::F16)
    {
        candle_core::bail!("decode attention wants an f32 q and sinks and an f32 or f16 cache");
    }
    if keys == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || heads / kv_heads > ATTN_SG {
        candle_core::bail!(
            "decode attention cannot take {heads} heads over {kv_heads} KV heads and {keys} keys"
        );
    }
    let dev = q.device();
    let pipes = pipelines(dev)?;
    let rep = heads / kv_heads;
    let (chunks, per_tg) = attn_split(keys);
    let params = AttnParams {
        keys: u32_of(keys)?,
        chunks: u32_of(chunks)?,
        rep: u32_of(rep)?,
        k_head_stride: u32_of(k.head_stride)?,
        v_head_stride: u32_of(v.head_stride)?,
        scale,
        per_tg: u32_of(per_tg)?,
    };
    let slots = kv_heads * chunks * rep;
    let part_o = dev.new_buffer(slots * ATTN_HEAD_DIM, DType::F32, "attn_dec_part_o")?;
    let part_ml = dev.new_buffer(slots * 2, DType::F32, "attn_dec_part_ml")?;
    let out = dev.new_buffer(heads * ATTN_HEAD_DIM, DType::F32, "attn_dec_out")?;
    // Each scoped encoder ends before the next `command_encoder()`, so the
    // two passes keep their order (see `moe`).
    {
        let enc = dev.command_encoder()?;
        enc.set_label("attn_dec_partial");
        enc.set_compute_pipeline_state(if kv_dtype == DType::F16 {
            &pipes.attn_partial_f16
        } else {
            &pipes.attn_partial
        });
        enc.set_buffer(0, Some(q.buffer()), q_offset);
        enc.set_buffer(1, Some(k.storage.buffer()), k.offset);
        enc.set_buffer(2, Some(v.storage.buffer()), v.offset);
        enc.set_buffer(3, Some(&part_o), 0);
        enc.set_buffer(4, Some(&part_ml), 0);
        enc.set_bytes(5, &params);
        enc.dispatch_thread_groups(grid(chunks, kv_heads), grid(ATTN_SG * SIMD_WIDTH, 1));
    }
    {
        let enc = dev.command_encoder()?;
        enc.set_label("attn_dec_combine");
        enc.set_compute_pipeline_state(&pipes.attn_combine);
        enc.set_buffer(0, Some(&part_o), 0);
        enc.set_buffer(1, Some(&part_ml), 0);
        enc.set_buffer(2, Some(sinks.0.buffer()), sinks.1);
        enc.set_buffer(3, Some(&out), 0);
        enc.set_bytes(4, &params);
        enc.dispatch_thread_groups(grid(heads, 1), grid(COMBINE_GROUPS * ATTN_HEAD_DIM, 1));
    }
    Ok(MetalStorage::new(
        out,
        dev.clone(),
        heads * ATTN_HEAD_DIM,
        DType::F32,
    ))
}

/// Mirrors `RouteParams` in the shader.
#[repr(C)]
pub(crate) struct RouteParams {
    pub(crate) experts: u32,
    pub(crate) k: u32,
    pub(crate) tokens: u32,
    pub(crate) norm: u32,
    pub(crate) scale: f32,
    pub(crate) min_sum: f32,
}

/// Sigmoid top-`k` routing of `logits [tokens, experts]` (contiguous from
/// `offset` bytes) on the GPU. Returns the ids (`u32`) and weights (`f32`),
/// both `tokens × k`; `scale` multiplies every weight.
pub(crate) fn moe_route(
    logits: &MetalStorage,
    offset: usize,
    (tokens, experts): (usize, usize),
    k: usize,
    norm: bool,
    (scale, min_sum): (f32, f32),
) -> candle_core::Result<(MetalStorage, MetalStorage)> {
    if logits.dtype() != DType::F32 {
        candle_core::bail!("moe_route wants f32 logits");
    }
    if k == 0 || k > ROUTE_MAX_K || k > experts || experts > ROUTE_MAX_EXPERTS {
        candle_core::bail!("moe_route cannot pick {k} of {experts} experts");
    }
    let dev = logits.device();
    let pipes = pipelines(dev)?;
    let params = RouteParams {
        experts: u32_of(experts)?,
        k: u32_of(k)?,
        tokens: u32_of(tokens)?,
        norm: u32::from(norm),
        scale,
        min_sum,
    };
    let ids = dev.new_buffer(tokens * k, DType::U32, "moe_route_ids")?;
    let weights = dev.new_buffer(tokens * k, DType::F32, "moe_route_weights")?;
    {
        let enc = dev.command_encoder()?;
        enc.set_label("moe_route");
        enc.set_compute_pipeline_state(&pipes.moe_route);
        enc.set_buffer(0, Some(logits.buffer()), offset);
        enc.set_buffer(1, Some(&ids), 0);
        enc.set_buffer(2, Some(&weights), 0);
        enc.set_bytes(3, &params);
        enc.dispatch_thread_groups(grid(tokens, 1), grid(ROUTE_THREADS, 1));
    }
    Ok((
        MetalStorage::new(ids, dev.clone(), tokens * k, DType::U32),
        MetalStorage::new(weights, dev.clone(), tokens * k, DType::F32),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key lands in exactly one run, runs are whole blocks, and no KV
    /// head splits into more than `ATTN_MAX_CHUNKS` runs.
    #[test]
    fn attention_splits_into_bounded_whole_block_runs() {
        assert_eq!(attn_split(1), (1, ATTN_BLOCK));
        assert_eq!(attn_split(64), (2, ATTN_BLOCK));
        for keys in [1, 31, 32, 33, 2048, 2049, 20_000, 262_144] {
            let (chunks, per_tg) = attn_split(keys);
            assert!(chunks <= ATTN_MAX_CHUNKS, "{keys}");
            assert_eq!(per_tg % ATTN_BLOCK, 0, "{keys}");
            assert!(
                chunks * per_tg >= keys && (chunks - 1) * per_tg < keys,
                "{keys}"
            );
        }
    }

    #[test]
    fn the_shader_takes_its_launch_constants_from_the_host() {
        let src = source();
        for (name, value) in [
            ("SIMD_WIDTH", SIMD_WIDTH),
            ("MV_ROWS_PER_SG", MV_ROWS_PER_SG),
            ("MV_SG_PER_TG", MV_SG_PER_TG),
            ("MM_TN", MM_TILE),
            ("MM_TT", MM_TILE),
            ("MM_THREADS", MM_THREADS),
            ("SUM_THREADS", SUM_THREADS),
            ("ATTN_HEAD_DIM", ATTN_HEAD_DIM),
            ("ATTN_BLOCK", ATTN_BLOCK),
            ("COMBINE_GROUPS", COMBINE_GROUPS),
            ("PF_Q", PF_Q),
            ("PF_K", PF_K),
            ("PF_THREADS", PF_THREADS),
            ("ATTN_SG", ATTN_SG),
            ("ROUTE_MAX_K", ROUTE_MAX_K),
            ("ROUTE_THREADS", ROUTE_THREADS),
            ("ROUTE_PER_THREAD", ROUTE_PER_THREAD),
            ("NORM_THREADS", NORM_THREADS),
            ("COMBINE_PER_THREAD", COMBINE_PER_THREAD),
            ("GROUP_THREADS", GROUP_THREADS),
            ("MOE_TT", MOE_TT),
            ("MOE_GU_TN", MOE_GU_TN),
            ("MOE_DN_TN", MOE_DN_TN),
            ("GROUP_MAX_EXPERTS", GROUP_MAX_EXPERTS),
        ] {
            let line = format!("#define {name} {value}\n");
            assert!(src.starts_with(PRELUDE_MARK), "the prelude comes first");
            assert!(src.contains(&line), "missing {line:?}");
            assert!(
                !SOURCE.contains(&format!("#define {name} "))
                    && !FUSED_SOURCE.contains(&format!("#define {name} "))
                    && !DECODE_SOURCE.contains(&format!("#define {name} "))
                    && !GROUPED_SOURCE.contains(&format!("#define {name} ")),
                "a kernel file still defines {name} itself"
            );
        }
        for literal in ["e += 128", "* 256 + tid", "sb += 32", "tid >> 2"] {
            assert!(!SOURCE.contains(literal), "{literal:?} is hard-coded");
        }
    }

    /// Prints the device's `maxBufferLength`: a file larger than this cannot
    /// be one no-copy buffer and falls back to copying every weight.
    #[test]
    #[ignore = "prints a device limit for FINDINGS.md"]
    // A display figure; precision loss in the GiB conversion is irrelevant.
    #[allow(clippy::cast_precision_loss)]
    fn print_max_buffer_length() {
        let Ok(dev) = candle_core::Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let candle_core::Device::Metal(dev) = dev else {
            return;
        };
        let raw: &ProtocolObject<dyn MTLDevice> = dev.device().as_ref();
        let n = raw.maxBufferLength();
        println!(
            "maxBufferLength = {n} bytes ({:.2} GiB)",
            n as f64 / f64::from(1u32 << 30)
        );
    }
}
