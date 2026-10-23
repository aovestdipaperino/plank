//! Pins plank's Flint tools prompt byte for byte.
//!
//! There is no C reference for this dialect, so these fixtures are the
//! reference: any drift is a change to what the model reads, and must be
//! reviewed. They pin plank's text and the fenced `tojson` tool lines; the
//! template scaffold the lines are lifted into is pinned by flint-engine's
//! `template_parity`. Regenerate with
//! `PLANK_REGEN_FIXTURES=1 cargo test --test flint_parity`.

use std::path::Path;

fn check(name: &str, got: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/flint")
        .join(name);
    if std::env::var_os("PLANK_REGEN_FIXTURES").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("missing fixture {name}; regenerate with PLANK_REGEN_FIXTURES=1")
    });
    assert!(
        got == want,
        "{name} drifted; review and regenerate if intended"
    );
}

#[test]
fn flint_tools_prompt_fixtures() {
    check(
        "tools_on.txt",
        &plank::sysprompt::flint_tools_prompt_for_tests(true),
    );
    check(
        "tools_off.txt",
        &plank::sysprompt::flint_tools_prompt_for_tests(false),
    );
}
