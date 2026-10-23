//! Decode speed of the Flint model on Metal at increasing context depths,
//! where the global layers' attention reads ever more keys: greedy
//! generation of [`STEPS`] tokens from each depth, best of [`ROUNDS`].
//!
//! ```sh
//! PLANK_FLINT_GGUF=~/.plank/models/flint-nvfp4.gguf \
//!   cargo run -p flint-engine --features candle --release --example bench_depth
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use flint_engine::Session;
use flint_engine::model::Model;

/// Decode steps per measurement; the best of [`ROUNDS`] is reported.
const STEPS: u32 = 64;
const ROUNDS: usize = 5;

// Throughput figures; precision loss in the conversions is irrelevant.
#[allow(clippy::cast_precision_loss)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(std::env::var_os("PLANK_FLINT_GGUF").ok_or("set PLANK_FLINT_GGUF")?);
    let dev = Device::new_metal(0)?;
    let model = Model::open_with_ctx(&path, &dev, 32_768)?;
    let prompt: Vec<u32> = (0..20_000u32).map(|i| 1000 + (i * 7919) % 50_000).collect();
    let mut s = Session::new(model.clone(), 32_768);
    let mut done = 0;
    for depth in [64usize, 2048, 10_240, 20_000] {
        s.prefill(&prompt[done..depth], &|| false)?
            .ok_or("interrupted")?;
        done = depth;
        let mut best = 0f64;
        for round in 0..ROUNDS {
            // Greedy generation from `depth`: each step feeds back the last
            // argmax, so the experts vary as in a real reply. The cache is
            // cut back to `depth` after each round.
            let mut tok = 1000 + u32::try_from(round).unwrap_or(0);
            dev.synchronize()?;
            let start = Instant::now();
            for _ in 0..STEPS {
                let logits = s.step(tok)?;
                tok = argmax(&logits);
            }
            best = best.max(f64::from(STEPS) / start.elapsed().as_secs_f64());
            s.truncate(depth);
        }
        println!("depth {depth:>6}: decode {best:.1} tok/s");
    }
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .and_then(|(i, _)| u32::try_from(i).ok())
        .unwrap_or(0)
}
