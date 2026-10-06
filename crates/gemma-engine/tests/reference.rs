//! Opt-in check of the forward pass against llama.cpp on the real model:
//!
//! ```sh
//! PLANK_GEMMA_GGUF=~/.plank/models/gemma-4-E4B-it-Q4_K_M.gguf PLANK_NO_DS4=1 \
//!   cargo test -p gemma-engine --features candle --release --test reference -- --ignored --nocapture
//! ```
//!
//! `fixtures/e4b_reference.json` holds, per prompt, llama.cpp's tokens (BOS
//! first), its top-10 next-token probabilities and its 16-token greedy
//! continuation, recorded on llama.cpp's Metal backend; its `source` field
//! has the exact commands.
//!
//! The model runs on Metal when there is one, like the reference.
//! `PLANK_GEMMA_DEVICE=cpu` forces the CPU, where the first case misses the
//! probability tolerance: candle's CPU quantized matmul rounds every
//! activation to `Q8_K`, as llama.cpp's own CPU backend does, and on that
//! nearly tied prompt the rounding alone moves the top probabilities by up to
//! 0.065 (llama.cpp's CPU backend is itself 0.047 from its Metal one there).
#![cfg(feature = "candle")]

use std::sync::{Arc, OnceLock};

use gemma_engine::model::Model;
use gemma_engine::sample::Sampler;
use gemma_engine::session::Session;
use serde_json::Value;

/// Probability tolerance on each of llama.cpp's top-5 tokens.
const PROB_TOLERANCE: f64 = 0.03;
/// Greedy tokens that must agree before quantization drift may diverge.
const GREEDY_PREFIX: usize = 8;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/e4b_reference.json")).unwrap()
}

/// One model for every test in this file: two would not fit in memory.
fn model() -> Option<Arc<Model>> {
    static MODEL: OnceLock<Option<Arc<Model>>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let path = std::env::var("PLANK_GEMMA_GGUF").ok()?;
            Some(Model::open(path.as_ref(), &device()).unwrap())
        })
        .clone()
}

/// Metal unless `PLANK_GEMMA_DEVICE=cpu` or there is none.
fn device() -> candle_core::Device {
    if std::env::var("PLANK_GEMMA_DEVICE").as_deref() == Ok("cpu") {
        return candle_core::Device::Cpu;
    }
    candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|t| u32::try_from(t.as_u64().unwrap()).unwrap())
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&l| (f64::from(l) - f64::from(max)).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(logits: &[f32]) -> u32 {
    let best = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    u32::try_from(best).unwrap()
}

#[test]
#[ignore = "needs PLANK_GEMMA_GGUF"]
fn tokens_match_llama_cpp() {
    let Some(model) = model() else { return };
    for case in fixture()["cases"].as_array().unwrap() {
        let mut got: Vec<u32> = model.tokenizer.bos().into_iter().collect();
        got.extend(
            model
                .tokenizer
                .encode_trusted(case["prompt"].as_str().unwrap()),
        );
        assert_eq!(
            got,
            ids(&case["tokens"]),
            "tokens differ for {:?}",
            case["prompt"]
        );
    }
}

#[test]
#[ignore = "needs PLANK_GEMMA_GGUF"]
fn next_token_and_greedy_continuation_match_llama_cpp() {
    let Some(model) = model() else { return };
    let close = model.tokenizer.control_id("<turn|>").unwrap();
    let eos = model.tokenizer.eos();
    let mut failures = Vec::new();
    for (index, case) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        let tokens = ids(&case["tokens"]);
        let mut session = Session::new(model.clone(), 4096);
        let started = std::time::Instant::now();
        let mut logits = session.prefill(&tokens, &|| false).unwrap().unwrap();
        let prefill = started.elapsed();

        let top = case["top"].as_array().unwrap();
        let probs = softmax(&logits);
        eprintln!("case {index}: {} tokens", tokens.len());
        eprintln!(
            "  {:>8} {:>10} {:>10} {:>8}",
            "id", "llama.cpp", "ours", "delta"
        );
        let mut worst = 0f64;
        for pair in top.iter().take(5) {
            let id = usize::try_from(pair[0].as_u64().unwrap()).unwrap();
            let want = pair[1].as_f64().unwrap();
            let delta = (probs[id] - want).abs();
            worst = worst.max(delta);
            eprintln!("  {id:>8} {want:>10.5} {:>10.5} {delta:>8.5}", probs[id]);
        }
        let want_best = u32::try_from(top[0][0].as_u64().unwrap()).unwrap();
        if argmax(&logits) != want_best {
            failures.push(format!(
                "case {index}: argmax {} vs llama.cpp {want_best}",
                argmax(&logits)
            ));
        }

        let want_greedy = ids(&case["greedy"]);
        let mut sampler = Sampler::new(0);
        let mut got_greedy = Vec::new();
        let started = std::time::Instant::now();
        for _ in 0..want_greedy.len() {
            let t = sampler.sample(&logits, 0.0, 1.0, 0.0, true);
            got_greedy.push(t);
            if t == close || t == eos {
                break;
            }
            logits = session.step(t).unwrap();
        }
        let decode = started.elapsed();
        let matching = got_greedy
            .iter()
            .zip(&want_greedy)
            .take_while(|(a, b)| a == b)
            .count();
        #[allow(clippy::cast_precision_loss)]
        let (prefill_rate, decode_rate) = (
            tokens.len() as f64 / prefill.as_secs_f64(),
            got_greedy.len() as f64 / decode.as_secs_f64(),
        );
        eprintln!(
            "  top-5 max delta {worst:.5}; greedy {matching}/{} match; prefill {prefill_rate:.1} tok/s, decode {decode_rate:.1} tok/s",
            want_greedy.len()
        );
        eprintln!("  greedy ours  {got_greedy:?}\n  greedy llama {want_greedy:?}");

        for pair in top.iter().take(5) {
            let id = usize::try_from(pair[0].as_u64().unwrap()).unwrap();
            let want = pair[1].as_f64().unwrap();
            if (probs[id] - want).abs() >= PROB_TOLERANCE {
                failures.push(format!(
                    "case {index} token {id}: probability {} vs llama.cpp {want}",
                    probs[id]
                ));
            }
        }
        let need = GREEDY_PREFIX.min(want_greedy.len());
        if matching < need {
            failures.push(format!(
                "case {index}: greedy agrees on {matching} tokens, need {need}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
