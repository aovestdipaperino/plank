//! Opt-in checks of the tokenizer and the forward pass against llama.cpp
//! 970b51c on the real model (fixtures from `tools/flint_fixtures.py`):
//!
//! ```sh
//! PLANK_FLINT_GGUF=~/.plank/models/flint-nvfp4.gguf \
//!   cargo test -p flint-engine --features candle --release --test reference_model -- --ignored --nocapture
//! ```
#![cfg(feature = "candle")]

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use candle_core::Device;
use flint_engine::Session;
use flint_engine::model::Model;
use flint_engine::tokenizer::FlintTokenizer;
use serde_json::Value;

/// Probability tolerance on each of llama.cpp's top-5 tokens (spec).
const PROB_TOLERANCE: f64 = 0.005;
/// Greedy tokens that must agree on the short cases (spec), or up to EOG.
const GREEDY_PREFIX: usize = 64;

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn path() -> PathBuf {
    PathBuf::from(std::env::var_os("PLANK_FLINT_GGUF").expect("set PLANK_FLINT_GGUF"))
}

fn model() -> Arc<Model> {
    static M: OnceLock<Arc<Model>> = OnceLock::new();
    M.get_or_init(|| {
        let dev = Device::new_metal(0).unwrap_or(Device::Cpu);
        eprintln!("reference device: {dev:?}");
        Model::open_with_ctx(&path(), &dev, 32768).unwrap()
    })
    .clone()
}

fn tokenizer() -> FlintTokenizer {
    FlintTokenizer::from_header(model().header()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn plain_text_tokenizes_like_llama_cpp() {
    let t = tokenizer();
    let bytes = |v: &[u32]| {
        v.iter()
            .flat_map(|&i| t.piece_bytes(i))
            .collect::<Vec<u8>>()
    };
    for case in fixture("flint_tokenizer.json")["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let (got, want) = (t.encode_plain(text), ids(&case["ids"]));
        if want.iter().any(|&i| t.is_control(i)) {
            // llama.cpp matches user-defined tokens (`<think>`, `<tool_call>`,
            // ...) even in plain text; plank keeps plain text inert by
            // design (FINDINGS), so ours spells them out: same bytes, no
            // control ids.
            assert!(got.iter().all(|&i| !t.is_control(i)), "{text:?}: {got:?}");
            assert_eq!(bytes(&got), bytes(&want), "{text:?}");
        } else {
            assert_eq!(got, want, "{text:?}");
        }
    }
}

#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn rendered_prompts_tokenize_like_llama_cpp() {
    let t = tokenizer();
    for case in fixture("flint_reference.json")["cases"].as_array().unwrap() {
        let Some(prompt) = case["prompt"].as_str() else {
            continue;
        };
        assert_eq!(
            t.encode_trusted(prompt),
            ids(&case["tokens"]),
            "case {}",
            case["name"]
        );
    }
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = logits.iter().map(|l| f64::from(l - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn argmax(v: &[f32]) -> u32 {
    let (i, _) =
        v.iter().enumerate().fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &x)| if x > b.1 { (i, x) } else { b },
        );
    u32::try_from(i).unwrap()
}

#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn next_token_probabilities_and_greedy_runs_match_llama_cpp() {
    let m = model();
    let tok = tokenizer();
    let f = fixture("flint_reference.json");
    let mut failures = Vec::new();
    for case in f["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let tokens = ids(&case["tokens"]);
        let mut s = Session::new(m.clone(), m.context_length());
        let logits = s.prefill(&tokens, &|| false).unwrap().unwrap();
        let p = softmax(&logits);
        let top = case["top"].as_array().unwrap();
        let want_top = u32::try_from(top[0][0].as_u64().unwrap()).unwrap();
        if argmax(&logits) != want_top {
            failures.push(format!(
                "{name}: argmax {} != llama.cpp {want_top}",
                argmax(&logits)
            ));
        }
        let mut max_diff = 0f64;
        for pair in top.iter().take(5) {
            let id = usize::try_from(pair[0].as_u64().unwrap()).unwrap();
            let want = pair[1].as_f64().unwrap();
            max_diff = max_diff.max((p[id] - want).abs());
            if (p[id] - want).abs() > PROB_TOLERANCE {
                failures.push(format!("{name}: p[{id}] {:.5} vs {want:.5}", p[id]));
            }
        }
        let want = ids(&case["greedy"]);
        let limit = if name == "long" {
            want.len()
        } else {
            want.len().min(GREEDY_PREFIX)
        };
        let mut got = vec![argmax(&logits)];
        while got.len() < limit && !tok.is_eog(*got.last().unwrap()) {
            got.push(argmax(&s.step(*got.last().unwrap()).unwrap()));
        }
        let agree = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
            failures.push(format!(
                "{name}: greedy diverges at {i}: {:?} vs {:?}",
                &got[i..],
                &want[i..]
            ));
        }
        eprintln!(
            "{name}: {} prompt tokens, argmax {} (llama.cpp {want_top}) p={:.4}, \
             max top-5 |dp| {max_diff:.6}, greedy {agree}/{} agree ({} generated)",
            tokens.len(),
            argmax(&logits),
            p[want_top as usize],
            want.len(),
            got.len()
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
