//! NVFP4 kernel throughput on the real Flint GGUF, on Metal.
//!
//! ```sh
//! PLANK_FLINT_GGUF=~/.plank/models/flint-nvfp4.gguf \
//!   cargo run -p flint-engine --features candle --release --example bench_kernels
//! ```
//!
//! Decode reads every weight byte once per token, so GB/s against the
//! machine's memory bandwidth is the figure of merit.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use flint_engine::file::GgufFile;
use flint_engine::nvfp4::ops;
use flint_engine::nvfp4::tensor::MoeExperts;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::var_os("PLANK_FLINT_GGUF").ok_or("set PLANK_FLINT_GGUF to the Flint GGUF")?,
    );
    let file = GgufFile::open(&path)?;
    let dev = Device::new_metal(0)?;

    let q = Arc::new(file.nv_weight("blk.2.attn_q")?);
    let x1 = Tensor::ones((1, q.cols()), DType::F32, &dev)?;
    bench("attn_q decode", q.bytes().len(), 200, &dev, || {
        ops::matmul(&q, &x1).map(drop)
    })?;
    let x512 = Tensor::ones((512, q.cols()), DType::F32, &dev)?;
    bench("attn_q prefill x512", q.bytes().len(), 20, &dev, || {
        ops::matmul(&q, &x512).map(drop)
    })?;

    let head = Arc::new(file.nv_weight("output")?);
    let xh = Tensor::ones((1, head.cols()), DType::F32, &dev)?;
    bench("lm_head decode", head.bytes().len(), 50, &dev, || {
        ops::matmul(&head, &xh).map(drop)
    })?;

    let m = MoeExperts::new(
        Arc::new(file.nv_weight("blk.2.ffn_gate_exps")?),
        Arc::new(file.nv_weight("blk.2.ffn_up_exps")?),
        Arc::new(file.nv_weight("blk.2.ffn_down_exps")?),
    )?;
    let xm = Tensor::ones((1, m.hidden()), DType::F32, &dev)?;
    let ids = Tensor::new(&[[3u32, 17, 42, 99, 128, 250]], &dev)?;
    let route = Tensor::new(&[[0.3f32, 0.2, 0.2, 0.1, 0.1, 0.1]], &dev)?;
    let moe_bytes = 6 * (m.gate().expert_bytes() + m.up().expert_bytes() + m.down().expert_bytes());
    bench("moe decode (6 experts)", moe_bytes, 200, &dev, || {
        ops::moe(&m, &xm, &ids, &route).map(drop)
    })?;
    Ok(())
}

// Throughput figures; precision loss in the f64 conversion is irrelevant.
#[allow(clippy::cast_precision_loss)]
fn bench(
    name: &str,
    bytes: usize,
    iters: u32,
    dev: &Device,
    mut f: impl FnMut() -> candle_core::Result<()>,
) -> candle_core::Result<()> {
    f()?;
    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    dev.synchronize()?;
    let secs = start.elapsed().as_secs_f64() / f64::from(iters);
    println!(
        "{name:28} {:9.1} us {:8.1} GB/s",
        secs * 1e6,
        bytes as f64 / secs / 1e9
    );
    Ok(())
}
