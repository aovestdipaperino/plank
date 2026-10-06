//! Pins the Gemma 4 tools prompt byte for byte.
//!
//! There is no C reference for this dialect, so these fixtures are the
//! reference: any drift is a change to what the model reads, and must be
//! reviewed. The declaration syntax itself follows the model's own chat
//! template (`format_function_declaration`), pinned by unit tests in
//! `src/sysprompt.rs`. Regenerate with
//! `PLANK_REGEN_FIXTURES=1 cargo test --test gemma_parity`.

use std::path::Path;

fn check(name: &str, got: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gemma")
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
fn gemma_tools_prompt_fixtures() {
    check(
        "tools_on.txt",
        &plank::sysprompt::gemma_tools_prompt_for_tests(true),
    );
    check(
        "tools_off.txt",
        &plank::sysprompt::gemma_tools_prompt_for_tests(false),
    );
}

#[test]
fn gemma_profile_prompt_fixture() {
    check(
        "profile.txt",
        &plank::sysprompt::gemma_profile_prompt_for_tests(
            "You are HAL.\n\n{{plank:tool-protocol}}\n\nBe brief.\n",
            &["read", "edit", "bash", "glob"],
        ),
    );
}
