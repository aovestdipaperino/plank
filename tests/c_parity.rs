//! Byte-diff parity tests against the `refs/ds4` C reference.
//!
//! The model was trained on the C agent's exact bytes, so the wire-facing
//! text (tools prompt, DSML syntax, tool-result framing) must stay
//! byte-for-byte identical to `refs/ds4/ds4_agent.c`. Two layers enforce it:
//!
//! 1. **Fixtures** (`tests/fixtures/`): committed snapshots of the reference
//!    bytes, compared on every `cargo test` — including CI checkouts without
//!    the submodule. Regenerate with `PLANK_REGEN_FIXTURES=1 cargo test`,
//!    then review the diff before committing.
//! 2. **The C source itself**: when the `refs/ds4` submodule is present, the
//!    named C string constants are decoded straight out of `ds4_agent.c` and
//!    compared, so the fixtures cannot silently drift from the reference.
//!
//! Nondeterministic spans (timestamps) are masked in fixtures with `«MASK»`;
//! [`assert_masked_eq`] compares everything around the masks byte-exactly.

use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Compares `actual` to the fixture, or rewrites the fixture when
/// `PLANK_REGEN_FIXTURES` is set. Byte-exact: any drift is a wire change.
fn assert_fixture_eq(name: &str, actual: &str) {
    let path = fixture_path(name);
    if std::env::var_os("PLANK_REGEN_FIXTURES").is_some() {
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("missing fixture {name} ({e}); regenerate with PLANK_REGEN_FIXTURES=1")
    });
    assert_identical(&expected, actual, name);
}

/// Like [`assert_fixture_eq`] but the fixture may contain `«MASK»` markers
/// that match any (possibly empty) span in `actual`.
fn assert_masked_fixture_eq(name: &str, actual: &str, regen_value: &str) {
    let path = fixture_path(name);
    if std::env::var_os("PLANK_REGEN_FIXTURES").is_some() {
        std::fs::write(&path, regen_value).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("missing fixture {name} ({e}); regenerate with PLANK_REGEN_FIXTURES=1")
    });
    assert_masked_eq(&expected, actual, name);
}

/// Asserts byte equality, reporting the first differing byte in context.
fn assert_identical(expected: &str, actual: &str, what: &str) {
    if expected == actual {
        return;
    }
    let pos = expected
        .bytes()
        .zip(actual.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(actual.len()));
    let ctx = |s: &str| {
        let lo = pos.saturating_sub(40);
        let hi = (pos + 40).min(s.len());
        s.get(lo..hi).map_or_else(
            || format!("<non-utf8 boundary near byte {pos}>"),
            str::to_string,
        )
    };
    panic!(
        "{what}: first differing byte at offset {pos}\n  expected …{:?}…\n  actual   …{:?}…",
        ctx(expected),
        ctx(actual)
    );
}

/// Compares `actual` against `pattern`, where `«MASK»` in the pattern matches
/// any span of `actual`. Segments between masks must match byte-exactly and
/// in order; the pattern must consume `actual` completely.
fn assert_masked_eq(pattern: &str, actual: &str, what: &str) {
    const MASK: &str = "«MASK»";
    let segments: Vec<&str> = pattern.split(MASK).collect();
    let mut rest = actual;
    let last = segments.len() - 1;
    for (i, seg) in segments.iter().enumerate() {
        if i == 0 {
            let Some(r) = rest.strip_prefix(seg) else {
                panic!("{what}: output does not start with expected prefix {seg:?}");
            };
            rest = r;
        } else if i == last {
            assert!(
                rest.ends_with(seg) || (seg.is_empty()),
                "{what}: output does not end with expected suffix {seg:?} (tail was {rest:?})"
            );
            rest = "";
        } else {
            let Some(at) = rest.find(seg) else {
                panic!("{what}: expected segment {seg:?} not found after masks");
            };
            rest = &rest[at + seg.len()..];
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture layer: always runs, submodule or not.
// ---------------------------------------------------------------------------

#[test]
fn tools_prompt_matches_fixture() {
    assert_fixture_eq(
        "tools_prompt.txt",
        &plank::sysprompt::build_tools_prompt(&[]),
    );
}

#[test]
fn dsml_syntax_reminder_matches_fixture() {
    assert_fixture_eq(
        "dsml_reminder.txt",
        plank::sysprompt::dsml_syntax_reminder(),
    );
}

#[test]
fn system_prompt_reminder_matches_fixture() {
    assert_fixture_eq(
        "system_prompt_reminder.txt",
        &plank::sysprompt::build_system_prompt_reminder(&[]),
    );
}

#[test]
fn datetime_context_matches_fixture_modulo_timestamp() {
    // The timestamp is local-timezone dependent — that span is masked.
    let line =
        plank::sysprompt::datetime_context_line(UNIX_EPOCH + Duration::from_secs(1_700_000_000));
    let regen = {
        // Rebuild the masked form from the live line: mask the span between
        // the fixed prefix and the fixed suffix.
        let prefix = "Current local date and time at session start: ";
        let suffix = ". Use this only when date or time matters.";
        assert!(
            line.starts_with(prefix) && line.ends_with(suffix),
            "unexpected shape: {line:?}"
        );
        format!("{prefix}«MASK»{suffix}")
    };
    assert_masked_fixture_eq("datetime_context.txt", &line, &regen);
}

#[test]
fn tool_result_framing_matches_reference() {
    use plank::dsml::ToolCall;
    use plank::tools::{ToolContext, dispatch_all};

    let mut ctx = ToolContext::new(std::env::temp_dir());
    // Unknown tool: exercises both the per-call header and the error text.
    let call = ToolCall {
        name: "nope".to_string(),
        args: Vec::new(),
    };
    let out = dispatch_all(&[call], &mut ctx);
    assert_fixture_eq("tool_result_unknown.txt", &out);

    // Empty block: the C emits a fixed error line.
    let out = dispatch_all(&[], &mut ctx);
    assert_identical(
        "Tool error: empty tool call block\n",
        &out,
        "empty tool call block",
    );
}

// ---------------------------------------------------------------------------
// Source layer: decode the constants straight out of ds4_agent.c when the
// submodule is checked out, so fixtures cannot drift from the reference.
// ---------------------------------------------------------------------------

fn c_source() -> Option<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("refs/ds4")
        .join("ds4_agent.c");
    std::fs::read_to_string(path).ok()
}

/// Decodes the concatenated C string literals initializing
/// `static const char <name>[] = ...;` in `src`.
fn extract_c_string_constant(src: &str, name: &str) -> String {
    let decl = format!("static const char {name}[] =");
    let start = src
        .find(&decl)
        .unwrap_or_else(|| panic!("constant {name} not found in ds4_agent.c"));
    let mut out = String::new();
    let bytes = &src.as_bytes()[start + decl.len()..];
    let mut i = 0;
    loop {
        // Skip whitespace/newlines between literals.
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        match bytes.get(i) {
            Some(b'"') => i += 1,
            Some(b';') => break,
            other => panic!("unexpected token {other:?} while reading {name}"),
        }
        // Decode one literal.
        while i < bytes.len() {
            match bytes[i] {
                b'"' => {
                    i += 1;
                    break;
                }
                b'\\' => {
                    i += 1;
                    let (ch, used) = match bytes[i] {
                        b'n' => ('\n', 1),
                        b't' => ('\t', 1),
                        b'r' => ('\r', 1),
                        b'0' => ('\0', 1),
                        b'\\' => ('\\', 1),
                        b'"' => ('"', 1),
                        b'\'' => ('\'', 1),
                        b'x' => {
                            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap();
                            (u8::from_str_radix(hex, 16).unwrap() as char, 3)
                        }
                        other => panic!("unhandled escape \\{} in {name}", other as char),
                    };
                    out.push(ch);
                    i += used;
                }
                _ => {
                    // Copy the full UTF-8 character (DSML uses U+FF5C).
                    let s = &src[start + decl.len()..];
                    let ch = s[i..].chars().next().unwrap();
                    out.push(ch);
                    i += ch.len_utf8();
                }
            }
        }
    }
    out
}

#[test]
fn tools_prompt_matches_c_source() {
    let Some(src) = c_source() else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };
    let mut expected = extract_c_string_constant(&src, "agent_tools_prompt_intro");
    expected.push_str(&extract_c_string_constant(
        &src,
        "agent_tools_prompt_edit_line",
    ));
    expected.push_str(&extract_c_string_constant(
        &src,
        "agent_tools_prompt_after_edit",
    ));
    // The base is what must match C byte-for-byte. Native plank tools (glob)
    // and MCP tools are layered on top by `build_tools_prompt`, outside the
    // trained table — see `append_native_extra_schemas`.
    assert_identical(
        &expected,
        &plank::sysprompt::build_tools_prompt_base(),
        "tools prompt base vs C",
    );
}

#[test]
fn dsml_reminder_matches_c_source() {
    let Some(src) = c_source() else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };
    let expected = extract_c_string_constant(&src, "agent_dsml_syntax_reminder");
    assert_identical(
        &expected,
        plank::sysprompt::dsml_syntax_reminder(),
        "DSML reminder vs C",
    );
}

#[test]
fn tool_result_header_format_matches_c_source() {
    let Some(src) = c_source() else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };
    // The C frames each result with snprintf("Tool result %d (%s):\n", …) and
    // emits a fixed line for an empty block; both literals must be present.
    assert!(
        src.contains(r#""Tool result %d (%s):\n""#),
        "C header format changed"
    );
    assert!(
        src.contains(r#""Tool error: empty tool call block\n""#),
        "C empty-block text changed"
    );
}
