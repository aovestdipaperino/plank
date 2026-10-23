//! One measurement for a head-to-head: prefill `depth` tokens, then decode
//! 128 greedily and print the decode rate (as Ollama's eval rate is).
use candle_core::Device;
use flint_engine::Session;
use flint_engine::model::Model;
use std::path::PathBuf;
use std::time::Instant;
// Throughput figures and small token ids; the conversions cannot lose
// anything that matters.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let depth: usize = std::env::args().nth(1).ok_or("depth")?.parse()?;
    let path = PathBuf::from(std::env::var_os("PLANK_FLINT_GGUF").ok_or("set PLANK_FLINT_GGUF")?);
    let dev = Device::new_metal(0)?;
    let model = Model::open_with_ctx(&path, &dev, 32_768)?;
    let prompt: Vec<u32> = (0..depth as u32)
        .map(|i| 1000 + (i * 7919) % 50_000)
        .collect();
    let mut s = Session::new(model.clone(), 32_768);
    let t = Instant::now();
    let mut logits = s.prefill(&prompt, &|| false)?.ok_or("interrupted")?;
    let prefill = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for _ in 0..128 {
        let tok = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i as u32);
        logits = s.step(tok)?;
    }
    println!(
        "prefill {:.1} tok/s, decode {:.1} tok/s",
        depth as f64 / prefill,
        128.0 / t.elapsed().as_secs_f64()
    );
    Ok(())
}
