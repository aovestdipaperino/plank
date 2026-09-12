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
        &plank::sysprompt::build_tools_prompt(&[], true),
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
fn dsml41_tools_prompt_matches_fixture() {
    assert_fixture_eq(
        "tools_prompt_dsml41.txt",
        &plank::sysprompt::dsml41_tools_prompt(&plank::sysprompt::build_tools_prompt(&[], true)),
    );
}

#[test]
fn dsml41_syntax_reminder_matches_fixture() {
    assert_fixture_eq(
        "dsml41_reminder.txt",
        plank::sysprompt::dsml41_syntax_reminder(),
    );
}

#[test]
fn system_prompt_reminder_matches_fixture() {
    assert_fixture_eq(
        "system_prompt_reminder.txt",
        // Empty roster: an agent roster must never perturb the parity bytes.
        &plank::sysprompt::build_system_prompt_reminder(&[], true),
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
    c_file("ds4_agent.c")
}

/// The engine core, which owns the chat encoding and the reasoning-effort
/// preamble (the agent lives one layer above it).
fn c_core_source() -> Option<String> {
    c_file("ds4.c")
}

fn c_file(name: &str) -> Option<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("refs/ds4")
        .join(name);
    std::fs::read_to_string(path).ok()
}

/// Inlines object-like `#define NAME "..."` macros into the rest of `src`.
///
/// The C spells shared prompt sentences as string macros so two variants of a
/// section can share them. [`extract_c_string_constant`] only understands
/// literals, so the macro name has to become its literal before it runs.
/// Continuation backslashes let a macro body sit on the following line.
fn expand_string_macros(src: &str) -> String {
    let mut macros: Vec<(String, String)> = Vec::new();
    // Join continuation lines so a macro body always follows its name.
    let joined = src.replace("\\\n", " ");
    for line in joined.lines() {
        let Some(rest) = line.trim_start().strip_prefix("#define ") else {
            continue;
        };
        let Some((name, body)) = rest.split_once(char::is_whitespace) else {
            continue;
        };
        let body = body.trim();
        // Object-like macros only, and only those whose body is a literal.
        if name.contains('(') || !body.starts_with('"') || !body.ends_with('"') {
            continue;
        }
        macros.push((name.to_string(), body.to_string()));
    }
    // Longest name first, so one macro's name cannot be a prefix of another's.
    macros.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));
    let mut out = joined;
    for (name, body) in macros {
        // Skip the defining line itself: it is not inside any constant we read.
        out = out
            .lines()
            .map(|line| {
                if line.trim_start().starts_with("#define ") {
                    line.to_string()
                } else {
                    line.replace(&name, &body)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    out
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
    // The edit section is spelled with a `#define`d string macro, which the
    // literal decoder below cannot see through; expand it first.
    let src = expand_string_macros(&src);
    let mut expected = extract_c_string_constant(&src, "agent_tools_prompt_intro");
    // The intro ends with AGENT_TOOL_CONTRACTS, which plank does not adopt:
    // two of its sentences are false of plank — a 128 KiB read cap it does not
    // have (that is the C's `AGENT_TOOL_MAX_BYTES` buffer limit) and hard-link
    // rejection it never checks. Here the block is the intro's tail, so the
    // subtraction is a truncation. Everything before it still has to match
    // byte for byte, so upstream rewording of the rest still surfaces.
    let contracts_at = expected
        .find(CONTRACTS_HEAD)
        .expect("intro still ends with the contracts block plank omits");
    expected.truncate(contracts_at);
    // plank ships the `[upto]` variant: its edit tool implements the anchor,
    // so it takes the prompt that teaches it. The C's `_edit_exact` sibling
    // (its default since `--edit-upto` became opt-in) is deliberately not the
    // one plank mirrors — see `sysprompt::TOOLS_PROMPT_EDIT_LINE`.
    expected.push_str(&extract_c_string_constant(
        &src,
        "agent_tools_prompt_edit_upto",
    ));
    // `agent_build_dsml_tools_prompt` splices the vision schema in just before
    // `\n# Rules\n` rather than carrying it in the after-edit block, so the
    // assembly here has to do the same. plank's base prompt is the vision=true
    // variant: the encoder is always offered on the DeepSeek path.
    let after_edit = extract_c_string_constant(&src, "agent_tools_prompt_after_edit");
    let rules_at = after_edit
        .find("\n# Rules\n")
        .expect("after-edit block still has a Rules section to splice before");
    expected.push_str(&after_edit[..rules_at]);
    expected.push_str("\n{\"type\":\"function\",\"function\":");
    expected.push_str(&extract_c_string_constant(&src, "agent_vision_tool_schema"));
    expected.push_str("}\n");
    expected.push_str(&after_edit[rules_at..]);
    // The base is what must match C byte-for-byte. Native plank tools (glob)
    // and MCP tools are layered on top by `build_tools_prompt`, outside the
    // trained table — see `append_native_extra_schemas`.
    assert_identical(
        &expected,
        &plank::sysprompt::build_tools_prompt_base(true),
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

/// The V4.1 reminder is its own C constant (`agent_dsml41_syntax_reminder`),
/// not a derivation, so it is pinned straight against the C source.
#[test]
fn dsml41_reminder_matches_c_source() {
    let Some(src) = c_source() else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };
    let expected = extract_c_string_constant(&src, "agent_dsml41_syntax_reminder");
    assert_identical(
        &expected,
        plank::sysprompt::dsml41_syntax_reminder(),
        "DSML 4.1 reminder vs C",
    );
}

/// `agent_dsml41_tools_prompt` is a rewrite rather than a constant, so parity
/// is checked on its output: the C's own reminder constant is the shortest
/// text carrying all three tags, and running plank's port over the V4 constant
/// must produce the C's V4.1 constant byte-for-byte.
#[test]
fn dsml41_rewrite_matches_c_source() {
    let Some(src) = c_source() else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };
    let v4 = extract_c_string_constant(&src, "agent_dsml_syntax_reminder");
    let v41 = extract_c_string_constant(&src, "agent_dsml41_syntax_reminder");
    assert_identical(
        &v41,
        &plank::sysprompt::dsml41_tools_prompt(&v4),
        "DSML 4.1 rewrite vs C",
    );
    // And the predicate the C applies: the name must be followed by `>` or a
    // space, so `p[n] == '>' || p[n] == ' '` is still what the C tests.
    assert!(
        src.contains("p[n] == '>' || p[n] == ' '"),
        "C DSML 4.1 rewrite predicate changed"
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

/// The `/think max` preamble is model-facing text prepended ahead of the system
/// prompt, so it is under the same byte-for-byte rule as the system prompt
/// itself: the model was trained against exactly these words.
#[test]
fn think_max_prefix_matches_c_source() {
    let Some(src) = c_core_source() else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };
    let expected = extract_c_string_constant(&src, "DS4_REASONING_EFFORT_MAX_PREFIX");
    assert_identical(
        &expected,
        plank::engine::THINK_MAX_PREFIX,
        "think-max preamble vs C",
    );
}

/// The context floor `/think max` enforces is the C's, so the two cannot drift
/// into disagreeing about when the level is usable.
#[test]
fn think_max_min_context_matches_c_source() {
    let Some(src) = c_core_source() else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };
    let decl = "#define DS4_THINK_MAX_MIN_CONTEXT ";
    let start = src
        .find(decl)
        .expect("DS4_THINK_MAX_MIN_CONTEXT not found in ds4.c")
        + decl.len();
    let digits: String = src[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    let expected: i32 = digits.parse().expect("min-context define is not a number");
    assert_eq!(
        expected,
        plank::engine::THINK_MAX_MIN_CONTEXT,
        "think-max minimum context vs C"
    );
}

/// `chat_push_think_prefix`'s fall-through arm (non-GLM, non-DeepSeek41 model
/// families — this covers V4) only appends `DS4_REASONING_EFFORT_MAX_PREFIX`
/// when the mode is exactly `DS4_THINK_MAX`. plank relies on that: it always
/// passes a numeric effort level (e.g. `Level(25)` for `/think low`) down to
/// this same C function on every family, and it is a documented byte-for-byte
/// no-op on V4 *only* because this arm ignores everything but `DS4_THINK_MAX`.
/// If a submodule bump changed this arm to react to numeric levels — or to
/// any non-MAX mode — V4 users at `low` would silently start getting a
/// different system prompt, and nothing else would catch it (a full
/// end-to-end check needs an 81 GB engine open).
#[test]
fn think_prefix_fallthrough_arm_matches_c_source() {
    let Some(src) = c_core_source() else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };

    let fn_start = src
        .find("static void chat_push_think_prefix(")
        .expect("chat_push_think_prefix not found in ds4.c");
    let brace_start = src[fn_start..]
        .find('{')
        .map(|i| fn_start + i)
        .expect("chat_push_think_prefix has no body");
    // Balance braces to find the function's closing brace.
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut i = brace_start;
    let body_end = loop {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    break i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    };
    let body = &src[brace_start..body_end];

    assert!(
        body.contains("DS4_MODEL_FAMILY_GLM_DSA") && body.contains("DS4_MODEL_FAMILY_DEEPSEEK41"),
        "chat_push_think_prefix no longer branches on GLM_DSA/DeepSeek41 \
         first; the fall-through arm this test pins may no longer be the \
         non-GLM, non-DeepSeek41 branch — re-derive the test against the new \
         shape"
    );

    // The final `else if` in the chain (with no preceding family check) is
    // the fall-through arm for every other family, V4 included. Take the
    // text after the LAST "else if" so branch reordering upstream of it
    // (e.g. swapping the GLM_DSA and DeepSeek41 blocks) cannot affect which
    // clause this test reads.
    let last_else_if = body
        .rsplit("else if")
        .next()
        .expect("chat_push_think_prefix has no else-if fall-through arm");
    let cond_end = last_else_if
        .find('{')
        .expect("fall-through arm's condition has no body");
    // Collapse whitespace so reformatting (line breaks, extra spaces) can't
    // trip the check.
    let condition: String = last_else_if[..cond_end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(
        condition, "(think_mode == DS4_THINK_MAX)",
        "chat_push_think_prefix's fall-through arm no longer guards on \
         `think_mode == DS4_THINK_MAX` alone (now: {condition:?}). On V4 and \
         any other non-GLM, non-DeepSeek41 family, plank passes a numeric \
         effort level to this function on every ThinkMode, trusting this arm \
         to ignore it outside DS4_THINK_MAX; if upstream widened this \
         condition, V4 users at low/medium would silently start getting a \
         different system prompt than before, with no test catching it \
         short of opening the 81 GB engine."
    );
}

/// plank's Metal kernel table must list exactly what the C engine requires.
///
/// `ds4_gpu_full_source` treats every entry in its `required_sources` array as
/// mandatory and aborts the whole startup ("metal backend unavailable") when
/// one cannot be found. plank has to name each file explicitly, because the
/// C's fallback search paths only resolve relative to the submodule root. So a
/// submodule bump that ships a new kernel silently produces a build that
/// cannot open any model — which is exactly what the Qwen3.8 bump did, adding
/// `qwen4.metal` and `qwen4_vision.metal`.
///
/// Order matters as documentation, not to the engine, so this compares the
/// pairs as sets and reports each side's surplus.
#[test]
fn metal_kernels_match_the_c_reference() {
    let Some(src) = c_file("ds4_metal.m") else {
        eprintln!("refs/ds4 submodule absent; skipping source-layer parity check");
        return;
    };
    let table = src
        .split_once("required_sources = @[")
        .expect("required_sources table")
        .1
        .split_once("];")
        .expect("end of required_sources table")
        .0;

    // Each row is `@[@"VAR", @"metal/file.metal"]`; take the quoted pairs.
    let mut from_c: Vec<(String, String)> = Vec::new();
    for row in table.split("@[").skip(1) {
        let mut quoted = row.split('"').skip(1).step_by(2);
        let (Some(var), Some(path)) = (quoted.next(), quoted.next()) else {
            continue;
        };
        let file = path.rsplit('/').next().unwrap_or(path);
        from_c.push((var.to_owned(), file.to_owned()));
    }
    assert!(
        from_c.len() > 20,
        "parsed only {} rows out of the C table; the parser drifted from the \
         source layout rather than the table shrinking",
        from_c.len()
    );

    let ours: Vec<(String, String)> = plank::ds4engine::METAL_KERNEL_SOURCES
        .iter()
        .map(|(v, f)| ((*v).to_owned(), (*f).to_owned()))
        .collect();

    let missing: Vec<_> = from_c.iter().filter(|e| !ours.contains(e)).collect();
    let extra: Vec<_> = ours.iter().filter(|e| !from_c.contains(e)).collect();
    assert!(
        missing.is_empty(),
        "the C requires kernels plank never points at, so startup aborts with \
         \"metal backend unavailable\": {missing:?}"
    );
    assert!(
        extra.is_empty(),
        "plank points at kernels the C no longer requires: {extra:?}"
    );
}

/// The C sentence that opens `AGENT_TOOL_CONTRACTS`, which plank omits.
const CONTRACTS_HEAD: &str = "Read output is limited to 128 KiB.";
/// Every committed manifest must parse with plank's own parser.
///
/// They are data files, so nothing else compiles them: a typo in a URL, a
/// truncated hash, or a kind this build cannot install would otherwise only
/// surface as a failed download on a user's machine.
#[test]
fn the_committed_manifests_parse_and_name_installable_kinds() {
    for (set, name) in [(plank::manifest::ModelSet::Ds4, "ds4.manifest")] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(name);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let m = plank::manifest::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(set.manifest_name(), name, "set names its own file");

        // Every kind this build installs for the set must be present, or a
        // swap would never find the set complete and would silently install
        // nothing at all.
        for kind in set.kinds() {
            let entry = m
                .files
                .get(*kind)
                .unwrap_or_else(|| panic!("{name} omits the {kind} artifact"));
            assert!(entry.bytes > 0, "{name}: {kind} has no size");
            assert!(
                entry.url.starts_with("https://"),
                "{name}: {kind} url is not https"
            );
            assert!(
                plank::manifest::local_path_for(set, kind).is_some(),
                "{name}: {kind} has nowhere to install"
            );
        }
    }
}
