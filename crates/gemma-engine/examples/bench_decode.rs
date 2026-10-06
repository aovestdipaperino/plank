//! Decode throughput against context length on the real model:
//!
//! ```sh
//! PLANK_GEMMA_GGUF=~/.plank/models/gemma-4-E4B-it-Q4_K_M.gguf PLANK_NO_DS4=1 \
//!   cargo run -p gemma-engine --features candle --release --example bench_decode
//! ```
//!
//! For each context length N it prefills N filler tokens, then times 32
//! `step` calls. `PLANK_GEMMA_PROFILE=1` adds a per-region breakdown of
//! decode steps at N=2048 (with the device synchronized around every region,
//! so that run's own rate is not a throughput). `PLANK_GEMMA_BENCH_N` takes a
//! comma-separated list of context lengths instead of the default set.
//! Metal when available, unless `PLANK_GEMMA_DEVICE=cpu`.

use std::time::Instant;

use gemma_engine::model::Model;
use gemma_engine::profile;
use gemma_engine::session::Session;

const STEPS: usize = 32;
const PROFILE_N: usize = 2048;

fn device() -> candle_core::Device {
    if std::env::var("PLANK_GEMMA_DEVICE").as_deref() == Ok("cpu") {
        return candle_core::Device::Cpu;
    }
    candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
}

/// Ordinary text tokens, varied so attention is not degenerate.
fn filler(n: usize, bos: Option<u32>) -> Vec<u32> {
    let mut out: Vec<u32> = bos.into_iter().collect();
    let mut x: u32 = 12345;
    while out.len() < n {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        out.push(1000 + (x >> 8) % 60_000);
    }
    out
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let path = std::env::var("PLANK_GEMMA_GGUF").expect("set PLANK_GEMMA_GGUF");
    let ns: Vec<usize> = std::env::var("PLANK_GEMMA_BENCH_N").map_or_else(
        |_| vec![32, 512, 1024, 2048, 4096],
        |s| s.split(',').map(|n| n.trim().parse().expect("N")).collect(),
    );
    let max_n = ns.iter().copied().max().unwrap_or(0);
    let device = device();
    eprintln!("device: {device:?}");
    let model = Model::open_with_ctx(path.as_ref(), &device, max_n + 2 * STEPS + 16).expect("open");
    let profiling = profile::enabled();
    profile::set_enabled(false);

    // Warm-up: compile the kernels before anything is timed.
    {
        let mut s = Session::new(model.clone(), max_n + 2 * STEPS + 16);
        s.prefill(&filler(16, model.tokenizer.bos()), &|| false)
            .expect("warm-up prefill");
        for _ in 0..4 {
            s.step(1234).expect("warm-up step");
        }
    }

    println!(
        "{:>6} {:>12} {:>12} {:>10}",
        "N", "prefill t/s", "decode t/s", "ms/step"
    );
    for &n in &ns {
        let mut s = Session::new(model.clone(), max_n + 2 * STEPS + 16);
        let toks = filler(n, model.tokenizer.bos());
        let started = Instant::now();
        s.prefill(&toks, &|| false).expect("prefill");
        let prefill = started.elapsed().as_secs_f64();
        let started = Instant::now();
        for i in 0..STEPS {
            s.step(filler(n + i + 1, None)[n + i]).expect("step");
        }
        let decode = started.elapsed().as_secs_f64();
        println!(
            "{n:>6} {:>12.1} {:>12.2} {:>10.1}",
            n as f64 / prefill,
            STEPS as f64 / decode,
            decode * 1000.0 / STEPS as f64
        );
        if profiling && n == PROFILE_N {
            let _ = profile::take();
            profile::set_enabled(true);
            let started = Instant::now();
            for _ in 0..STEPS {
                s.step(1234).expect("profiled step");
            }
            let total = started.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;
            profile::set_enabled(false);
            println!("\nper-step breakdown at N={n} (synchronized, {total:.1} ms/step):");
            let mut sum = 0.0;
            for (name, d) in profile::take() {
                let ms = d.as_secs_f64() * 1000.0 / STEPS as f64;
                sum += ms;
                println!("  {name:<18} {ms:>8.2} ms");
            }
            println!("  {:<18} {:>8.2} ms\n", "(sum)", sum);

            // One 512-token prefill chunk at the end of the context.
            let keep = s.tokens().len() - 512;
            let chunk = s.tokens()[keep..].to_vec();
            s.truncate(keep);
            profile::set_enabled(true);
            let started = Instant::now();
            s.prefill(&chunk, &|| false).expect("profiled prefill");
            let total = started.elapsed().as_secs_f64() * 1000.0;
            profile::set_enabled(false);
            println!(
                "prefill breakdown, 512 tokens at offset {keep} (synchronized, {total:.1} ms):"
            );
            let mut sum = 0.0;
            for (name, d) in profile::take() {
                let ms = d.as_secs_f64() * 1000.0;
                sum += ms;
                println!("  {name:<18} {ms:>8.2} ms");
            }
            println!("  {:<18} {:>8.2} ms\n", "(sum)", sum);
        }
    }
}
