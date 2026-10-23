//! The unified API over the Flint backend, on a tiny generated Flint GGUF
//! run on the CPU, so it needs no model download and no GPU.

#![cfg(feature = "flint")]

use std::path::PathBuf;
use std::sync::Arc;

use flint_engine::config::{FlintConfig, Sampling};
use flint_engine::testgguf::{FLINT_MARKERS, tiny_flint_with_vocab};
use local_inference_engine::{Family, Model, Options, Session, Think};

/// `flint-engine`'s test config with a 256-token window.
fn config() -> FlintConfig {
    FlintConfig {
        layers: 4,
        hidden: 128,
        ffn: 128,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        rope_base: 10000.0,
        rms_eps: 1e-6,
        context_length: 256,
        sliding_window: 4,
        swa: vec![false, true, true, false],
        rope: vec![false, true, true, false],
        dense_layers: 1,
        experts: 8,
        experts_used: 2,
        expert_ffn: 64,
        shared_ffn: 64,
        weights_norm: true,
        weights_scale: 1.0,
        sampling: Sampling {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
        },
    }
}

/// A tiny Flint GGUF in a directory of its own.
fn tiny(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lie-flint-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    std::fs::write(
        &path,
        tiny_flint_with_vocab(&config(), &FLINT_MARKERS, 0.5, 54),
    )
    .unwrap();
    path
}

fn open(name: &str) -> Arc<Model> {
    let path = tiny(name);
    Arc::new(Model::open(&Options::new(&path).ctx_size(256).cpu(true)).unwrap())
}

fn text(model: &Model, ids: &[i32]) -> String {
    String::from_utf8(ids.iter().flat_map(|&t| model.token_bytes(t)).collect()).unwrap()
}

#[test]
fn a_flint_gguf_is_routed_to_the_flint_backend() {
    let path = tiny("route");
    assert_eq!(Family::of(&path), Family::Flint);
    let model = Model::open(&Options::new(&path).ctx_size(256).cpu(true)).unwrap();
    assert_eq!(model.family(), Family::Flint);
    assert!(model.as_ds4().is_none());
    assert!(model.as_flint().is_some());
    assert_eq!(model.name(), "Flint");
    assert_eq!(model.ctx_size(), 256);
}

#[test]
fn a_non_gguf_file_falls_through_to_ds4() {
    let dir = std::env::temp_dir().join(format!("lie-flint-not-gguf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("not.gguf");
    std::fs::write(&path, b"not a gguf").unwrap();
    assert_eq!(Family::of(&path), Family::Ds4);
}

#[test]
fn the_chat_is_system_user_and_the_assistant_opener_with_no_bos() {
    let model = open("chat");
    let with = model.encode_chat("be brief", "hello", Think::Off).unwrap();
    assert_eq!(
        text(&model, &with),
        "<|system|>be brief<|user|>hello<|assistant|>"
    );
    let without = model.encode_chat("", "hello", Think::Off).unwrap();
    assert_eq!(text(&model, &without), "<|user|>hello<|assistant|>");
    // The template has no thinking switch.
    assert_eq!(
        model.encode_chat("be brief", "hello", Think::Max).unwrap(),
        with
    );
    assert_eq!(model.tokenize_rendered("<|user|>").unwrap().len(), 1);
}

#[test]
fn a_callers_text_never_turns_into_control_tokens() {
    let model = open("plain");
    let ids = model
        .encode_chat("sys <|user|>", "hi <|assistant|>", Think::Off)
        .unwrap();
    let controls = model.tokenize_rendered("<|user|>").unwrap()[0];
    let assistant = model.tokenize_rendered("<|assistant|>").unwrap()[0];
    assert_eq!(ids.iter().filter(|&&t| t == controls).count(), 1);
    assert_eq!(ids.iter().filter(|&&t| t == assistant).count(), 1);
}

#[test]
fn sync_eval_and_sample_run_a_greedy_generation() {
    let model = open("generate");
    let mut session = Session::new(&model, 128).unwrap();
    let prompt = model.encode_chat("be brief", "hi", Think::Off).unwrap();
    let mut events = 0;
    session
        .sync_with_progress(&prompt, &mut |_, _, _| events += 1)
        .unwrap();
    assert!(events > 0);
    assert_eq!(session.pos(), i32::try_from(prompt.len()).unwrap());

    let mut rng = 1;
    let first = session.sample(0.0, 0, 1.0, 0.0, &mut rng);
    assert!(first >= 0);
    let top = session.top_logprobs(3);
    assert_eq!(top.len(), 3);
    assert_eq!(top[0].id, first, "greedy takes the likeliest token");
    assert!(top.windows(2).all(|w| w[0].logprob >= w[1].logprob));

    session.eval(first).unwrap();
    assert_eq!(session.pos(), i32::try_from(prompt.len() + 1).unwrap());
    assert!(session.sample(0.0, 0, 1.0, 0.0, &mut rng) >= 0);
    assert!(session.as_flint().is_some());
}

#[test]
fn a_resync_reuses_the_prefix_and_reproduces_the_logits() {
    let model = open("resync");
    let mut session = Session::new(&model, 128).unwrap();
    let a = model
        .encode_chat("sys", "first question", Think::Off)
        .unwrap();
    session.sync(&a).unwrap();
    let before = session.top_logprobs(5);

    let b = model.encode_chat("sys", "another one", Think::Off).unwrap();
    session.sync(&b).unwrap();
    assert_eq!(session.pos(), i32::try_from(b.len()).unwrap());

    session.sync(&a).unwrap();
    let after = session.top_logprobs(5);
    assert_eq!(
        before.iter().map(|s| s.id).collect::<Vec<_>>(),
        after.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    for (x, y) in before.iter().zip(&after) {
        assert!((x.logprob - y.logprob).abs() < 1e-4, "{x:?} vs {y:?}");
    }

    session.sync(&a).unwrap();
    assert_eq!(session.top_logprobs(1)[0].id, before[0].id);

    session.invalidate();
    assert_eq!(session.pos(), 0);
    assert_eq!(session.sample(0.0, 0, 1.0, 0.0, &mut 0), -1);
}

#[test]
fn top_k_restricts_sampling_to_the_likeliest_tokens() {
    let model = open("topk");
    let mut session = Session::new(&model, 128).unwrap();
    session
        .sync(&model.encode_chat("", "x", Think::Off).unwrap())
        .unwrap();
    let best = session.top_logprobs(1)[0].id;
    let mut rng = 99;
    for _ in 0..20 {
        assert_eq!(session.sample(5.0, 1, 1.0, 0.0, &mut rng), best);
    }
}

#[test]
fn a_prompt_longer_than_the_context_is_refused() {
    let model = open("full");
    let mut session = Session::new(&model, 8).unwrap();
    let prompt = model
        .encode_chat("", "a long enough prompt", Think::Off)
        .unwrap();
    assert!(prompt.len() > 8);
    let err = session.sync(&prompt).unwrap_err();
    assert!(err.to_string().contains("context full"), "{err}");
}
