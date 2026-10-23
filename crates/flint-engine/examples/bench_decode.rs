//! Prefill and decode speed of the Flint model on Metal.
//!
//! ```sh
//! PLANK_FLINT_GGUF=~/.plank/models/flint-nvfp4.gguf \
//!   cargo run -p flint-engine --features candle --release --example bench_decode
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use flint_engine::Session;
use flint_engine::model::Model;

// Throughput figures; precision loss in the conversions is irrelevant.
#[allow(clippy::cast_precision_loss)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(std::env::var_os("PLANK_FLINT_GGUF").ok_or("set PLANK_FLINT_GGUF")?);
    let dev = Device::new_metal(0)?;
    let model = Model::open_with_ctx(&path, &dev, 8192)?;
    let prompt: Vec<u32> = (0..512u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    // One discarded iteration first, so pipeline compilation and the first
    // touch of the mapped weights do not land in the measured runs.
    let mut warm = Session::new(model.clone(), 8192);
    warm.prefill(&prompt[..32], &|| false)?;
    for &t in &prompt[32..40] {
        warm.step(t)?;
    }
    dev.synchronize()?;
    drop(warm);
    for (label, depth) in [("short", 32usize), ("512", 512)] {
        let mut s = Session::new(model.clone(), 8192);
        let start = Instant::now();
        let mut logits = s
            .prefill(&prompt[..depth], &|| false)?
            .ok_or("interrupted")?;
        dev.synchronize()?;
        let prefill = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for _ in 0..64 {
            let next = logits
                .iter()
                .enumerate()
                .fold((0u32, f32::NEG_INFINITY), |b, (i, &x)| {
                    if x > b.1 {
                        (u32::try_from(i).unwrap_or(0), x)
                    } else {
                        b
                    }
                });
            logits = s.step(next.0)?;
        }
        let decode = start.elapsed().as_secs_f64();
        println!(
            "context {label:>5}: prefill {depth} tokens in {prefill:.2} s ({:.0} tok/s), decode 64 tokens {:.1} tok/s",
            depth as f64 / prefill,
            64.0 / decode
        );
    }
    Ok(())
}
