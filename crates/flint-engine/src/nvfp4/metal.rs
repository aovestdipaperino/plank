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
use crate::nvfp4::ops::MatDims;
use crate::nvfp4::tensor::NvWeight;

const SOURCE: &str = include_str!("nvfp4.metal");

/// From this many tokens up, a matmul runs the tiled `mm_nvfp4` kernel.
pub(crate) const MM_MIN_TOKENS: usize = 8;
/// `MV_ROWS_PER_SG * MV_SG_PER_TG` in the shader.
const MV_ROWS_PER_TG: usize = 8;
/// `32 * MV_SG_PER_TG`.
const MV_THREADS: usize = 64;
/// Weight rows and tokens per `mm_nvfp4` tile (`MM_TN`, `MM_TT`).
const MM_TILE: usize = 32;
/// Four simdgroups.
const MM_THREADS: usize = 128;
/// The simdgroup width every kernel assumes.
const SIMD_WIDTH: usize = 32;

/// The compiled kernels of one device.
#[derive(Debug)]
pub(crate) struct Pipelines {
    pub mv: ComputePipeline,
    pub mm: ComputePipeline,
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
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
        .map_err(|e| metal_err(format!("nvfp4.metal does not compile: {e}")))?;
    let lib = Library::new(lib);
    let get = |name: &str| -> candle_core::Result<ComputePipeline> {
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
        Ok(ComputePipeline::new(pipe))
    };
    let p = Arc::new(Pipelines {
        mv: get("mv_nvfp4")?,
        mm: get("mm_nvfp4")?,
    });
    cache.push((id, Arc::clone(&p)));
    Ok(p)
}

/// Where a weight lives on the GPU.
#[derive(Debug)]
pub(crate) struct Binding {
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

/// The weight's buffer, byte offset and scale buffer, bound on first use.
pub(crate) fn binding(
    dev: &MetalDevice,
    w: &NvWeight,
) -> candle_core::Result<(Arc<Buffer>, usize, Arc<Buffer>)> {
    let mut slot = w.metal.lock().map_err(metal_err)?;
    if let Some(b) = slot.as_ref() {
        return Ok((Arc::clone(&b.buf), b.offset, Arc::clone(&b.scales)));
    }
    let shared = match w.mapping() {
        Some((file, offset)) => {
            let mut whole = file.metal.lock().map_err(metal_err)?;
            let buf = whole.get_or_insert_with(|| no_copy(dev, file)).clone();
            buf.map(|b| (b, offset))
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
struct MatParams {
    k: u32,
    n: u32,
    tokens: u32,
    row_bytes: u32,
    scale: f32,
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
    drop(enc);
    Ok((
        MetalStorage::new(out, dev.clone(), tokens * n, DType::F32),
        Shape::from((tokens, n)),
    ))
}
