//! Regression test for the vision-embedding ownership transfer.
//!
//! `ds4_prompt_append_vision` moves an embedding's `data` pointer into the
//! output span instead of copying the floats, and the span is later released
//! with `ds4_vision_embedding_free` (a plain `free`). plank used to hand it a
//! pointer into a Rust `Vec<f32>`, so the buffer was freed twice — once by C,
//! once by the `Vec` — which aborts the process with
//! `malloc: pointer being freed was not allocated`. The `rebuild_vision_spans`
//! path was worse: its `Vec` dropped at the end of the block, leaving the span
//! dangling for the rest of the turn.
//!
//! The crash only shows up once a *second* turn runs over a transcript whose
//! user message carries an image, because that is what drives the kept-span
//! rebuild. So this test takes two turns and then drops the session; a
//! regression aborts the whole test binary rather than failing an assertion,
//! which is exactly the symptom being guarded.
//!
//! Needs the real engine and the model files, so it self-skips everywhere else
//! (CI included).

#![cfg(ds4_engine)]

use plank::engine::{Engine, GenerationOptions, Prompt, VisionImage};

/// A 64x64 PNG gradient, written fresh so the test owns its input.
fn write_test_png(path: &std::path::Path) {
    let mut buf = image::RgbImage::new(64, 64);
    for (x, y, px) in buf.enumerate_pixels_mut() {
        // 0..64 scaled by 4 stays inside a byte, so the conversion is exact.
        let component = |v: u32| u8::try_from(v * 4).unwrap_or(u8::MAX);
        *px = image::Rgb([component(x), component(y), 128]);
    }
    buf.save(path).expect("write test png");
}

#[test]
fn two_turns_over_an_image_message_do_not_double_free_the_embedding() {
    let home = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default());
    let model = home.join(".plank/ds4flash.gguf");
    let vision = home.join(".plank/ds4flash.vision.gguf");
    if !model.exists() || !vision.exists() {
        eprintln!("skipping: no model at {} (+ vision)", model.display());
        return;
    }

    let dir = std::env::temp_dir().join(format!("plank-vision-own-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let png = dir.join("gradient.png");
    write_test_png(&png);

    // DSpark is on by default and needs its support GGUF wired up; this test
    // is about ownership, not decode speed, so keep the plain target path.
    let tuning = plank::config::EngineTuning {
        dspark: false,
        ..plank::config::EngineTuning::default()
    };
    let mut session = plank::ds4engine::Ds4Session::open(
        &model,
        plank::ffi::Ds4Backend::Metal,
        8192,
        0,
        100,
        &tuning,
    )
    .expect("open engine");
    assert!(session.has_vision(), "vision encoder should be loaded");

    let embedding = session
        .vision_encode_file(&png.to_string_lossy())
        .expect("encode the test image");
    assert!(
        embedding.token_count > 0 && !embedding.data.is_empty(),
        "encoder produced an empty embedding",
    );
    // Metadata has to survive the encoder's own `ds4_vision_embedding_free`,
    // which memsets the struct it is handed: reading these fields after that
    // free zeroed `layout` and the token grid, and the C then rejected every
    // append with "invalid DeepSeek vision embedding layout" — silently, since
    // a failed append just yields an empty token delta.
    assert!(
        embedding.layout != 0,
        "layout was lost (read after the C free zeroed the struct)",
    );
    assert!(
        embedding.grid_width > 0 && embedding.grid_height > 0,
        "token grid was lost: {}x{}",
        embedding.grid_width,
        embedding.grid_height,
    );
    assert!(
        embedding.width > 0 && embedding.height > 0,
        "image dimensions were lost",
    );
    assert!(
        embedding.fingerprint != [0_u8; 32],
        "fingerprint was lost, so image identity checks would collapse",
    );

    // The transcript shape `sync_engine_images` produces: the image hangs off a
    // user message, keyed by the exact section text the engine tokenizes.
    let text = "Summarize this image.";
    let transcript = format!("[user]\n{text}\n");
    let image = VisionImage {
        path: png.to_string_lossy().into_owned(),
        embedding,
    };

    let opts = GenerationOptions {
        n_predict: 1,
        ctx_size: 8192,
        ..GenerationOptions::default()
    };
    let no = || false;

    // Turn one appends the image as a multimodal user message (ownership moves
    // into a fresh span). Turn two matches the same message as a *kept* span and
    // rebuilds its span from a re-cloned pending image — the path that used to
    // dangle. `set_pending_images` re-sends clones before every turn, as
    // `Agent::sync_engine_images` does.
    for turn in 0..2 {
        session.set_pending_images(vec![(transcript.clone(), image.clone())]);
        session
            .generate(Prompt::Flat(&transcript), &opts, &no, &no, &mut |_event| {})
            .unwrap_or_else(|e| panic!("turn {turn} failed: {e}"));
    }

    // Dropping the session frees every live span. Under the bug this is the
    // second free of a buffer Rust already released.
    drop(session);
    let _ = std::fs::remove_dir_all(&dir);
}
