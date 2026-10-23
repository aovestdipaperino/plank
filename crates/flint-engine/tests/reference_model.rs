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

/// Probability tolerance on each of llama.cpp's top-5 first tokens (spec).
const PROB_TOLERANCE: f64 = 0.005;
/// Each of the f64 `NumPy` oracle's top-20 probabilities must be within this
/// of ours, at every position. The user's decision: llama.cpp rounds
/// activations to `Q8_0` (its NVFP4 matmuls run on the CPU) and keeps an f16
/// KV, so it moves single probabilities by tenths and cannot be a precise
/// per-position oracle (FINDINGS, "Flint against llama.cpp on the real
/// file"); `tools/flint_numpy_ref.py` can.
const NUMPY_TOLERANCE: f64 = 1e-3;

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

fn pair(v: &Value) -> (u32, f64) {
    (
        u32::try_from(v[0].as_u64().unwrap()).unwrap(),
        v[1].as_f64().unwrap(),
    )
}

/// llama.cpp's first token on all six cases, and its free greedy run on
/// `long`. The per-position `steps` in the fixture are recorded evidence of
/// its noise; no test asserts on them.
#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn first_tokens_and_the_long_greedy_run_match_llama_cpp() {
    let m = model();
    let tok = tokenizer();
    let fx = fixture("flint_reference.json");
    let mut failures = Vec::new();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let tokens = ids(&case["tokens"]);
        let mut session = Session::new(m.clone(), m.context_length());
        let logits = session.prefill(&tokens, &|| false).unwrap().unwrap();
        let probs = softmax(&logits);
        let top = case["top"].as_array().unwrap();
        let (want_top, _) = pair(&top[0]);
        if argmax(&logits) != want_top {
            failures.push(format!(
                "{name}: argmax {} != llama.cpp {want_top}",
                argmax(&logits)
            ));
        }
        let mut max_diff = 0f64;
        for w in top.iter().take(5) {
            let (id, want) = pair(w);
            let got = probs[id as usize];
            max_diff = max_diff.max((got - want).abs());
            if (got - want).abs() > PROB_TOLERANCE {
                failures.push(format!("{name}: p[{id}] {got:.5} vs {want:.5}"));
            }
        }
        eprintln!(
            "{name}: {} prompt tokens, argmax {} (llama.cpp {want_top}), max top-5 |dp| {max_diff:.6}",
            tokens.len(),
            argmax(&logits)
        );
        if name == "long" {
            let greedy = ids(&case["greedy"]);
            let mut got = vec![argmax(&logits)];
            while got.len() < greedy.len() && !tok.is_eog(*got.last().unwrap()) {
                got.push(argmax(&session.step(*got.last().unwrap()).unwrap()));
            }
            let agree = got.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
            if agree < greedy.len() {
                failures.push(format!(
                    "{name}: greedy diverges at {agree}: {:?} vs {:?}",
                    &got[agree.min(got.len())..],
                    &greedy[agree..]
                ));
            }
            eprintln!("{name}: free greedy {agree}/{} agree", greedy.len());
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Prefill, then step through the oracle's forced tokens, checking the
/// argmax and every top-20 probability at each of the positions.
#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn probabilities_match_the_f64_numpy_oracle() {
    let m = model();
    let fx = fixture("flint_numpy_reference.json");
    let mut failures = Vec::new();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let tokens = ids(&case["tokens"]);
        let forced = ids(&case["forced"]);
        let tops = case["top"].as_array().unwrap();
        assert_eq!(tops.len(), forced.len() + 1, "{name}");
        let mut session = Session::new(m.clone(), m.context_length());
        let mut logits = session.prefill(&tokens, &|| false).unwrap().unwrap();
        for (i, want) in tops.iter().enumerate() {
            let want = want.as_array().unwrap();
            let probs = softmax(&logits);
            let (want_top, want_p) = pair(&want[0]);
            let ok_argmax = argmax(&logits) == want_top;
            if !ok_argmax {
                failures.push(format!(
                    "{name} position {i}: argmax {} != oracle {want_top}",
                    argmax(&logits)
                ));
            }
            let mut max_diff = 0f64;
            for w in want {
                let (id, wp) = pair(w);
                let got = probs[id as usize];
                max_diff = max_diff.max((got - wp).abs());
                if (got - wp).abs() > NUMPY_TOLERANCE {
                    failures.push(format!("{name} position {i}: p[{id}] {got:.6} vs {wp:.6}"));
                }
            }
            eprintln!(
                "{name} position {i}: argmax {} (oracle {want_top}, p {want_p:.4}) {}, max top-20 |dp| {max_diff:.2e}",
                argmax(&logits),
                if ok_argmax { "agrees" } else { "DIFFERS" }
            );
            if let Some(&t) = forced.get(i) {
                logits = session.step(t).unwrap();
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
