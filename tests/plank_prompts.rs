//! Byte-lock fixtures for model-facing text plank deliberately does **not**
//! share with the `refs/ds4` C reference.
//!
//! This is the mirror of `c_parity.rs`. That suite locks the trained wire
//! format *to* the C, and a divergence there is a bug. The text here has no C
//! counterpart to check against — it is plank's own — so nothing would notice
//! an accidental edit. That matters most where the prompt is one half of a
//! contract whose other half is a parser: the compaction prompt names the
//! `<summary>`/`<analysis>` tags that `compact::extract_summary` looks for, and
//! a typo in either would fail silently at runtime (an empty or
//! reasoning-polluted summary) rather than in CI.
//!
//! Regenerate with `PLANK_REGEN_FIXTURES=1 cargo test`, the same command the
//! parity fixtures use, then review the diff before committing.

use std::path::{Path, PathBuf};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Compares `actual` to the fixture, or rewrites it under `PLANK_REGEN_FIXTURES`.
fn assert_fixture_eq(name: &str, actual: &str) {
    let path = fixture_path(name);
    if std::env::var_os("PLANK_REGEN_FIXTURES").is_some() {
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("missing fixture {name} ({e}); regenerate with PLANK_REGEN_FIXTURES=1")
    });
    if expected == actual {
        return;
    }
    let at = expected
        .bytes()
        .zip(actual.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    let from = at.saturating_sub(60);
    panic!(
        "{name} changed at byte {at}\n--- expected ---\n{:?}\n--- actual ---\n{:?}\n\
         If the change is intentional, regenerate with PLANK_REGEN_FIXTURES=1.",
        &expected[from..(at + 60).min(expected.len())],
        &actual[from..(at + 60).min(actual.len())],
    );
}

/// The compaction prompt is plank's own text, not the C's: the C asks for a
/// free-form bulleted summary, plank asks for eight numbered sections wrapped
/// in `<summary>` tags because that is what `extract_summary` parses. Locked
/// with a reason so the trailing `Compaction reason:` line is covered too.
/// Locked *without* custom instructions: that is the automatic-compaction form,
/// the one every turn can hit, and pinning it keeps `/compact <instructions>`
/// from silently changing the default prompt.
#[test]
fn compaction_prompt_matches_fixture() {
    assert_fixture_eq(
        "compaction_prompt.txt",
        &plank::compact::make_prompt("low context", ""),
    );
}

/// `/compact <instructions>` adds to the pinned prompt rather than reshaping it:
/// the fixture text must still be present in full, with the instructions spliced
/// in ahead of the closing no-tools trailer.
#[test]
fn compaction_prompt_with_instructions_extends_the_fixture() {
    let base = plank::compact::make_prompt("low context", "");
    let with = plank::compact::make_prompt("low context", "focus on the parser bug");
    let (body, tail) = base
        .split_once("After the summary, stop.")
        .expect("the pinned prompt ends with the no-tools trailer");
    assert!(with.starts_with(body), "the pinned body must be unchanged");
    assert!(
        with.ends_with(&format!("After the summary, stop.{tail}")),
        "the pinned trailer must stay last"
    );
    assert!(with.contains("focus on the parser bug"));
}
