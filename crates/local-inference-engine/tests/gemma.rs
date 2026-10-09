//! The unified API over the Gemma backend, on a tiny generated Gemma 4 GGUF
//! run on the CPU, so it needs no model download and no GPU.

#![cfg(feature = "gemma")]

use std::path::PathBuf;
use std::sync::Arc;

use gemma_engine::testgguf::{TinyConfig, write_tiny};
use local_inference_engine::{Family, Model, Options, Session, Think};

/// A tiny Gemma 4 GGUF in a directory of its own.
fn tiny(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lie-gemma-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    write_tiny(&path, &TinyConfig::default()).unwrap();
    path
}

fn open(name: &str) -> Arc<Model> {
    let path = tiny(name);
    Arc::new(Model::open(&Options::new(&path).ctx_size(256).cpu(true)).unwrap())
}

#[test]
fn a_gemma_gguf_is_routed_to_the_gemma_backend() {
    let path = tiny("route");
    assert_eq!(Family::of(&path), Family::Gemma);
    let model = Model::open(&Options::new(&path).ctx_size(256).cpu(true)).unwrap();
    assert_eq!(model.family(), Family::Gemma);
    assert!(model.as_ds4().is_none());
    assert!(model.as_gemma().is_some());
    assert!(model.name().starts_with("Gemma 4"), "{}", model.name());
    assert_eq!(model.ctx_size(), 256);
}

#[test]
fn a_non_gguf_file_falls_through_to_ds4() {
    let dir = std::env::temp_dir().join(format!("lie-not-gguf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("not.gguf");
    std::fs::write(&path, b"not a gguf").unwrap();
    assert_eq!(Family::of(&path), Family::Ds4);
}

#[test]
fn the_chat_ends_on_the_model_turn_and_thinking_adds_the_think_token() {
    let model = open("chat");
    let off = model.encode_chat("", "hello", Think::Off).unwrap();
    let on = model.encode_chat("", "hello", Think::High).unwrap();
    let text = |ids: &[i32]| {
        String::from_utf8(ids.iter().flat_map(|&t| model.token_bytes(t)).collect()).unwrap()
    };
    assert!(text(&off).ends_with("<|turn>model\n"), "{}", text(&off));
    assert!(!text(&off).contains("<|think|>"), "{}", text(&off));
    assert!(text(&on).contains("<|think|>"), "{}", text(&on));
    assert_eq!(model.tokenize_rendered("<|turn>").unwrap().len(), 1);
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

    // Back to `a`, through a shared prefix: the same distribution as a cold
    // prefill of `a`.
    session.sync(&a).unwrap();
    let after = session.top_logprobs(5);
    assert_eq!(
        before.iter().map(|s| s.id).collect::<Vec<_>>(),
        after.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    for (x, y) in before.iter().zip(&after) {
        assert!((x.logprob - y.logprob).abs() < 1e-4, "{x:?} vs {y:?}");
    }

    // Syncing the same prompt again keeps everything and still has logits.
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
