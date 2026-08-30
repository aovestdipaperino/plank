// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! System prompt rendering: tool prompt text, reminders, datetime context.
//!
//! Port of the prompt-construction half of the "System Prompt Rendering And
//! Worker Output Queues" section of `ds4_agent.c` (roughly lines 703-1065).
//! The long tool-protocol strings are model-facing and replicated verbatim
//! from the C reference.

use std::time::{SystemTime, UNIX_EPOCH};

/// The tools-prompt line forbidding tool calls inside thinking, verbatim from
/// the C (`ds4_agent.c:718`) and a substring of [`TOOLS_PROMPT_INTRO`].
///
/// Removed from the built prompt when `engine.thinkingToolCalls` is on, so the
/// prompt matches the behavior. The C constants themselves are never edited —
/// `tests/c_parity.rs` locks them against `refs/ds4`.
pub use trace_stream::viz::IN_THINK_PROHIBITION;

/// Introductory section of the tools prompt (verbatim from C).
const TOOLS_PROMPT_INTRO: &str = "You are a coding agent running in a local workspace. Use tools for local file and system work. \
Avoid printing large file contents or large code blocks as answers; create or edit files with tools, \
then summarize results briefly.\n\n\
## Tools\n\n\
You have access to native DSML tools. Invoke tools by writing exactly this shape:\n\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\n\
Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.\n\n\
String parameters use raw text and string=\"true\". Numbers and booleans use JSON text and string=\"false\".\n\n\
Read defaults to a context-sized bounded chunk, not the whole file. \
For first looks at large files, prefer read with explicit max_lines around 80-160; \
if read says more lines are available, call more with count=<lines> to read the next chunk. \
The read result also reports continue_offset=N, which is the next start_line if you need to jump manually. \
If the user explicitly asks you to read a complete file into context, call read with whole=true. \
A whole-file read may fail if the result would not fit the current context; then explain that and use chunks.\n\n";

/// Editing-instructions section of the tools prompt (verbatim from C).
///
/// This is the C's `agent_tools_prompt_edit_upto` variant: plank's edit tool
/// implements `[upto]` anchoring, so it takes the prompt that teaches it. The
/// C also carries an `agent_tools_prompt_edit_exact` variant it now selects by
/// default (`--edit-upto` opts back in); plank has not adopted that split.
const TOOLS_PROMPT_EDIT_LINE: &str = "## Editing files\n\n\
When editing files, state the target filename before the edit; for the edit tool, put path first.\n\
Use write for new files or deliberate whole-file replacement. Use edit with path, old, and new for changes. \
The old text must match exactly once in the current file; otherwise edit fails for safety.\n\
For large replacements, prefer anchored old text: write the first lines, then [upto], then the final lines. \
The tool replaces everything from the head through the tail. If the head or tail is ambiguous, the edit fails.\n\
After [upto], always write unique final lines before closing old; never close old immediately after [upto].\n\
Do not use a generic tail anchor like:\n\
- BigNum bignum_add(BigNum *a, BigNum *b) {\n\
- [upto]\n\
- }\n\
because the closing brace may match many functions. Instead include final lines that are unique near that function, \
for example its last calculation and return line before the brace.\n\
Example anchored edit:\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"edit\">\n\
<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/example.c</｜DSML｜parameter>\n\
<｜DSML｜parameter name=\"old\" string=\"true\">static int parse(void) {\n    int ok = 0;\n\
[upto]\n    return ok;\n\
}</｜DSML｜parameter>\n\
<｜DSML｜parameter name=\"new\" string=\"true\">static int parse(void) {\n    return parse_impl();\n\
}</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\
To insert text, use edit with old set to an exact unique anchor and new set to that anchor plus the added text.\n\
Use read raw=true only when you need plain file text without line numbers or read annotations.\n\n";

/// Trailing section of the tools prompt: web tools, schemas, rules.
///
/// Byte-identical to the C `agent_tools_prompt_after_edit`. Kept as a
/// resource file because a `\`-continued Rust string literal strips the
/// next line's leading whitespace, silently deleting the indentation the
/// JSON schemas carry (see FINDINGS.md); `tests/c_parity.rs` enforces the
/// byte identity.
const TOOLS_PROMPT_AFTER_EDIT: &str = include_str!("resources/tools_prompt_after_edit.txt");

/// Token-estimate distance after which the system prompt reminder is re-injected.
pub const SYSTEM_PROMPT_REMINDER_TOKENS: i32 = 50_000;

/// Selects which system prompt a backend receives (design §4.4).
///
/// The `Ds4` prompt is the byte-parity DS4 prompt (DSML-in-prose tool
/// instructions the local model was trained on); it must never be sent to a
/// third-party provider. The `Provider` prompt is plank's own text — the same
/// behavioral guidance minus the DSML syntax instructions, since native tool
/// definitions replace them. The `Provider` variant is deliberately *not* under
/// `tests/c_parity.rs`: it is free to evolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemPrompt {
    /// The byte-parity DS4 prompt (local / remote-ds4 engines).
    Ds4,
    /// The provider-facing prompt (OpenAI-compatible / Anthropic engines).
    Provider,
}

/// The provider-facing system prompt (design §4.4).
///
/// Same behavioral guidance as the DS4 prompt's prose — role, editing norms,
/// web-tool norms, the workspace rules — but **without** the DSML tool-call
/// syntax section (native provider tool definitions replace it) and without the
/// verbatim DSML JSON-schema dump. Non-empty `-sys` user text is appended.
#[must_use]
pub fn provider_system_prompt(user_system: &str) -> String {
    let mut out = String::from(
        "You are Plank, a coding agent running in a local workspace. Use the provided tools for local \
file and system work. Avoid printing large file contents or large code blocks as answers; create \
or edit files with tools, then summarize results briefly.\n\n\
## Reading files\n\n\
read defaults to a bounded chunk: a path alone returns the first 500 lines, not the whole file. \
If read reports more lines are available, call more with count=<lines> for the next chunk. Pass \
whole=true only when explicitly asked to read a complete file into context.\n\n\
## Editing files\n\n\
Use write for new files or deliberate whole-file replacement. Use edit with path, old and new for \
changes; old must match exactly once. For large replacements prefer anchored old text: the first \
lines, then [upto], then unique final lines — never close old immediately after [upto].\n\n\
## Web\n\n\
Use google_search to find web pages and visit_page to read a known URL. The first web call may \
ask permission to start a browser.\n\n\
## Rules\n\n\
- Prefer read/search to get anchors, then anchored edit to avoid retyping large text.\n\
- Write code that is reliable; keep a clear mental model of complex parts.\n\
- Preserve the current system configuration integrity unless explicitly asked otherwise.\n",
    );
    if crate::settings::active().git.sign_commits {
        out.push('\n');
        out.push_str(COMMIT_SIGNATURE_INSTRUCTION);
    }
    if !user_system.is_empty() {
        out.push('\n');
        out.push_str(user_system);
    }
    out
}

/// Names of every tool advertised this session, native and MCP.
///
/// Feeds the renderer's pseudo-tool detector, which needs to know which bare
/// `<name>` tags are the model reaching for a real tool.
#[must_use]
pub fn tool_names(mcp: &[crate::tools::mcp::McpServer]) -> Vec<String> {
    provider_tool_registry(mcp)
        .into_iter()
        .map(|spec| spec.name)
        .collect()
}

/// Builds the machine-readable tool registry for a provider engine (§4.3).
///
/// The static tool schemas already live as JSON in the DS4 tools prompt
/// resource (the `### Available Tool Schemas` section, `OpenAI` function shape);
/// this parses them into structured [`crate::engine::ToolSpec`]s — single
/// source of truth — and appends any loaded MCP tools.
#[must_use]
pub fn provider_tool_registry(
    mcp_servers: &[crate::tools::mcp::McpServer],
) -> Vec<crate::engine::ToolSpec> {
    let mut specs = parse_builtin_tool_schemas();
    // Native plank tools beyond the C table are appended to the text prompt by
    // `append_native_extra_schemas`; mirror them here so provider engines see
    // the same table.
    specs.push(crate::engine::ToolSpec {
        name: "glob".to_string(),
        description: "Find files by name pattern across a directory tree. Use this instead of shelling out to find or ls. '**' crosses directory boundaries, '*' matches within one path component. Results are paths relative to the search root, sorted, capped at 100.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "glob pattern, e.g. '*.rs', '**/*test*', 'src/**/mod.rs'"},
                "path": {"type": "string", "description": "directory to search from; defaults to the working directory"}
            },
            "required": ["pattern"]
        }),
    });
    specs.push(crate::engine::ToolSpec {
        name: "skill".to_string(),
        description: "Invoke an installed skill (a packaged procedure) by name; its instructions are returned for you to follow. Call with no name to list the installed skills first.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "skill name; omit to enumerate installed skills"},
                "args": {"type": "string", "description": "arguments passed to the skill"}
            }
        }),
    });
    specs.push(crate::engine::ToolSpec {
        name: "task".to_string(),
        description: "Track a plan that survives context compaction. op='add' appends a pending task (needs 'subject') and returns its id; op='update' changes a task's status/subject (needs 'id'); op='list' returns every task. Statuses: pending, in_progress, completed. The current list is shown to you each turn, so use 'list' only to recover it.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "op": {"type": "string", "description": "add, update, or list"},
                "id": {"type": "string", "description": "task id (for update)"},
                "subject": {"type": "string", "description": "task description (for add; optional rename on update)"},
                "status": {"type": "string", "description": "pending, in_progress, or completed (for update)"},
                "active_form": {"type": "string", "description": "present-tense form shown while the task is in progress, e.g. 'Refactoring the parser'"}
            },
            "required": ["op"]
        }),
    });
    specs.push(crate::engine::ToolSpec {
        name: "ask".to_string(),
        description: "Ask the user a multiple-choice question and block until they answer. Use this instead of guessing when a turn is genuinely ambiguous. 'question' is the full question, 'header' a short (~12 char) label, 'options' a JSON array of 2 to 7 {\"label\",\"description\"} choices. Set 'multi' to true to allow several selections. Returns the selected label(s). In non-interactive mode it returns immediately telling you no user is available.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "the full question, phrased as a question"},
                "header": {"type": "string", "description": "short UI label, ~12 characters"},
                "options": {"type": "string", "description": "JSON array of 2 to 7 {\"label\",\"description\"} objects"},
                "multi": {"type": "string", "description": "true to allow selecting more than one option (default false)"}
            },
            "required": ["question", "header", "options"]
        }),
    });
    push_agent_and_plan_specs(&mut specs);
    for server in mcp_servers {
        // No `alive` filter, deliberately: a non-alive server is either an
        // offline shadow — whose tools must stay advertised so this registry
        // and the text prompt agree, letting dispatch answer with the "not
        // running" message instead of the provider rejecting an unknown tool —
        // or a mid-session death, where dispatch reports the failure.
        for tool in &server.tools {
            // Only primary tools get a full schema, exactly as on the text path
            // (`append_tool_schemas`). A server like tokensave advertises 82
            // tools whose schemas are 63 KB of JSON — resent on every request,
            // where they inflate prefill and slow every subsequent decode by
            // enlarging the KV context. The rest stay reachable through the
            // directory below.
            if !tool.primary {
                continue;
            }
            let parameters = serde_json::from_str::<serde_json::Value>(&tool.schema_json)
                .unwrap_or_else(|_| serde_json::json!({ "type": "object", "properties": {} }));
            specs.push(crate::engine::ToolSpec {
                // Namespaced, exactly as the text path spells it
                // (`append_one_schema`). Dispatch routes MCP calls on the
                // `mcp__` prefix, so a bare name reaches the unknown-tool
                // fallthrough and the model is told the tool does not exist —
                // while the directory advertises the qualified spelling, which
                // is a contradiction it cannot act on.
                name: format!("mcp__{}__{}", server.name, tool.name),
                description: tool.description.clone(),
                parameters,
            });
        }
    }
    push_mcp_directory_specs(&mut specs, mcp_servers);
    // Resource tools, advertised only when a server actually publishes
    // resources — mirroring `append_resource_tool_schemas` for the text path.
    if mcp_servers.iter().any(|s| !s.resources().is_empty()) {
        specs.push(crate::engine::ToolSpec {
            name: "mcp_list_resources".to_string(),
            description: "List resources published by connected MCP servers, as {server}:{uri}. Optional 'server' filters to one server.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"server": {"type": "string"}}
            }),
        });
        specs.push(crate::engine::ToolSpec {
            name: "mcp_read_resource".to_string(),
            description: "Read one MCP resource's contents. Both 'server' and 'uri' are required (as listed by mcp_list_resources). Text inlines; binary reports type and size.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"server": {"type": "string"}, "uri": {"type": "string"}},
                "required": ["server", "uri"]
            }),
        });
    }
    specs
}

/// The `agent` tool's `name` description, used verbatim by both schema paths.
///
/// No roster is interpolated into it: definitions are advertised in the session
/// context instead (`context::agent_roster_context`), so that editing one
/// rebuilds the small project-tier cache rather than invalidating the
/// fingerprinted system prompt. Keeping this text fixed is also what keeps the
/// C-parity fixtures valid.
const AGENT_NAME_DESC_EMPTY: &str =
    "optional configured agent name to act as; omit for a general-purpose sub-agent";

/// Pushes the provider-path [`ToolSpec`](crate::engine::ToolSpec)s for the
/// `agent` and plan-mode tools (issue #50). Mirrors the text-path schemas in
/// [`append_agent_and_plan_schemas`]; split out to keep
/// [`provider_tool_registry`] under the function-length lint.
/// Pushes the two specs that keep *non-primary* MCP tools usable on the
/// provider path: `mcp_describe` for their schemas and `mcp_call` to invoke
/// them.
///
/// The text path can leave directory tools undeclared because the model writes
/// DSML there — free text, able to name any tool. A provider request has no
/// such freedom: an OpenAI-compatible gateway rejects a function name that was
/// never declared, and llama.cpp constrains tool-call names with a grammar
/// built from the `tools` array, so an undeclared name is literally
/// ungeneratable. `mcp_call` is therefore the one declared door to all of them,
/// and its description carries the directory so the model knows what exists.
fn push_mcp_directory_specs(
    specs: &mut Vec<crate::engine::ToolSpec>,
    mcp_servers: &[crate::tools::mcp::McpServer],
) {
    let directory = crate::tools::mcp::directory_listing(mcp_servers);
    if directory.is_empty() {
        return;
    }
    specs.push(crate::engine::ToolSpec {
        name: "mcp_describe".to_string(),
        description: "Return the full parameter schema of directory MCP tools (those not listed as function specs). Accepts one or more space-separated tool names. Call this before the first use of a directory tool.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "tools": {"type": "string", "description": "space-separated full tool names, e.g. 'mcp__srv__alpha mcp__srv__beta'"}
            },
            "required": ["tools"]
        }),
    });
    specs.push(crate::engine::ToolSpec {
        name: "mcp_call".to_string(),
        description: format!(
            "Invoke one of the MCP directory tools listed below. These tools have no function spec of their own, so call them through this one: pass the full tool name and its arguments as a JSON object. Use mcp_describe first to get a tool's parameter schema.\n\nDirectory:\n{directory}"
        ),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "full tool name, e.g. 'mcp__srv__alpha'"},
                "arguments": {"type": "string", "description": "the tool's arguments as a JSON object, e.g. '{\"path\": \"src\"}'; omit for a tool that takes none"}
            },
            "required": ["name"]
        }),
    });
}

fn push_agent_and_plan_specs(specs: &mut Vec<crate::engine::ToolSpec>) {
    // No enum of definition names: the roster lives in the session context
    // (`context::agent_roster_context`) so that editing a definition rebuilds
    // the small project-tier cache instead of invalidating the fingerprinted
    // system prompt. An unmatched name still falls back with a note.
    let name_prop = serde_json::json!({
        "type": "string", "description": AGENT_NAME_DESC_EMPTY
    });
    specs.push(crate::engine::ToolSpec {
        name: "agent".to_string(),
        description: "Delegate a self-contained sub-task to a fresh sub-agent that works in its own scoped context and returns only a final report. Use this to keep your own context small: hand off open-ended research or a bounded multi-step chore, then continue from its report. 'task' is a complete, standalone instruction; 'name' optionally selects a configured agent persona. The sub-agent cannot ask you questions, so make 'task' fully specified.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "the complete, standalone task to delegate; include all needed context"},
                "name": name_prop
            },
            "required": ["task"]
        }),
    });
    specs.push(crate::engine::ToolSpec {
        name: "EnterPlanMode".to_string(),
        description: "Enter read-only plan mode: research and design without changing anything. While it is active, write/edit/bash are refused; only read-only tools work. Use it when a task is risky or ambiguous and the user should approve an approach before you edit. Exit with ExitPlanMode.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    });
    specs.push(crate::engine::ToolSpec {
        name: "ExitPlanMode".to_string(),
        description: "Leave plan mode by presenting your proposed plan for the user's approval. On approval the read-only gate lifts and you may edit; otherwise plan mode stays on and you should refine the plan. 'plan' is the full proposed plan.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "the full plan to carry out, for the user to approve"}
            },
            "required": ["plan"]
        }),
    });
    push_worktree_specs(specs);
}

/// Pushes the provider-path specs for the worktree tools. Mirrors the text-path
/// schemas in [`append_worktree_schemas`].
fn push_worktree_specs(specs: &mut Vec<crate::engine::ToolSpec>) {
    specs.push(crate::engine::ToolSpec {
        name: "EnterWorktree".to_string(),
        description: "Move into an isolated git worktree: a separate checkout of this repository, on its own branch, where your edits cannot affect the main working copy. Every tool's working directory switches into it until you call ExitWorktree. Use this ONLY when the user explicitly asks for a worktree or for isolation from their current checkout. Do NOT use it for ordinary branch or feature work — run git through bash for that. 'name' is the worktree name; letters, digits, '.', '_', '-' and '/' only.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "name for the worktree, e.g. 'refactor-parser'"}
            },
            "required": ["name"]
        }),
    });
    specs.push(crate::engine::ToolSpec {
        name: "ExitWorktree".to_string(),
        description: "Leave the worktree entered with EnterWorktree and return to the original working directory. action 'keep' leaves the worktree and its branch on disk so the user can review, merge, or resume the work; action 'remove' deletes both. A remove is refused when the worktree holds uncommitted files or commits that are not on the base branch, unless you also pass discard_changes true — so prefer 'keep' whenever there is work worth saving.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "'keep' to leave the worktree in place, 'remove' to delete it"},
                "discard_changes": {"type": "string", "description": "true to remove the worktree even though it holds work that would be lost"}
            },
            "required": ["action"]
        }),
    });
}

/// Parses the built-in OpenAI-shaped tool schemas out of the DS4 tools prompt
/// resource into [`crate::engine::ToolSpec`]s.
fn parse_builtin_tool_schemas() -> Vec<crate::engine::ToolSpec> {
    let text = TOOLS_PROMPT_AFTER_EDIT;
    let Some(start) = text.find("### Available Tool Schemas") else {
        return Vec::new();
    };
    let rest = &text[start..];
    // The schema blocks end at the trailing "# Rules" section.
    let region = rest.split("# Rules").next().unwrap_or(rest);
    // Skip the header line itself.
    let region = region.split_once('\n').map_or(region, |(_, body)| body);
    let mut specs = Vec::new();
    // Consecutive JSON objects, blank-line separated; a streaming deserializer
    // tolerates the interspersed whitespace and stops cleanly at the tail.
    let stream = serde_json::Deserializer::from_str(region).into_iter::<serde_json::Value>();
    for value in stream.flatten() {
        let Some(func) = value.get("function") else {
            continue;
        };
        let Some(name) = func.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let description = func
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let parameters = func
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} }));
        specs.push(crate::engine::ToolSpec {
            name: name.to_string(),
            description,
            parameters,
        });
    }
    specs
}

/// Builds the full tools prompt (intro, editing, schemas, rules, MCP tools).
///
/// Mirrors `agent_build_tools_prompt`: the three verbatim C string constants
/// followed by the schemas of any MCP tools loaded at startup.
#[must_use]
pub fn build_tools_prompt(mcp_servers: &[crate::tools::mcp::McpServer], parity: bool) -> String {
    build_tools_prompt_parts(mcp_servers, parity).0
}

/// [`build_tools_prompt`] plus the byte length of its leading **trusted** span.
///
/// Everything up to that offset is plank's own control text — the C-derived
/// base, the two inserted notes, and the native tool schemas. Everything after
/// it comes from MCP servers, which are third-party processes whose tool names,
/// descriptions, and JSON schemas are arbitrary text.
///
/// The distinction exists because the trusted span is tokenized differently:
/// see [`SplitSystemPrompt::trusted_len`].
fn build_tools_prompt_parts(
    mcp_servers: &[crate::tools::mcp::McpServer],
    parity: bool,
) -> (String, usize) {
    build_tools_prompt_parts_with_wasm(mcp_servers, &[], parity)
}

/// [`build_tools_prompt_parts`] with WASM component tools folded in.
///
/// Their schemas land **after** `trusted_len`, beside MCP's and for the same
/// reason: a component's tool names, descriptions and schemas are arbitrary
/// third-party text, and the trusted span is exactly the part plank wrote.
fn build_tools_prompt_parts_with_wasm(
    mcp_servers: &[crate::tools::mcp::McpServer],
    wasm_tools: &[&crate::wasmreg::WasmTool],
    parity: bool,
) -> (String, usize) {
    let mut out = build_tools_prompt_base(parity);
    insert_marker_spelling_note(&mut out);
    insert_document_read_note(&mut out);
    append_native_extra_schemas(&mut out);
    let trusted_len = out.len();
    crate::tools::mcp::append_tool_schemas(&mut out, mcp_servers);
    crate::tools::mcp::append_resource_tool_schemas(&mut out, mcp_servers);
    crate::tools::mcp::append_server_instructions(&mut out, mcp_servers);
    append_wasm_tool_schemas(&mut out, wasm_tools);
    (out, trusted_len)
}

/// Appends one function schema per WASM component tool, in the same shape the
/// MCP block uses so the model sees one convention rather than two.
fn append_wasm_tool_schemas(out: &mut String, tools: &[&crate::wasmreg::WasmTool]) {
    use std::fmt::Write as _;

    if tools.is_empty() {
        return;
    }
    for t in tools {
        let _ = write!(
            out,
            "\n{{\n  \"type\": \"function\",\n  \"function\": {{\n    \"name\": \"{}\",\n    \"description\": ",
            t.exposed
        );
        crate::tools::mcp::json_escape(out, &t.description);
        let _ = write!(out, ",\n    \"parameters\": {}\n  }}\n}}\n", t.schema);
    }
}

/// The C-derived tools prompt with nothing appended.
///
/// This is what the parity suite locks byte-for-byte against `refs/ds4`: it is
/// exactly the three C string constants. Native plank tools (see
/// [`append_native_extra_schemas`]) and MCP tools are layered on top by
/// [`build_tools_prompt`], the same way MCP has always extended it, so the
/// trained-table parity guarantee stays intact for the base.
///
/// `parity` selects strict C output. With `parity` false the single
/// [`IN_THINK_PROHIBITION`] line is stripped, because plank then dispatches
/// those calls (see `engine.thinkingToolCalls`); everything else is identical.
#[must_use]
pub fn build_tools_prompt_base(parity: bool) -> String {
    let mut out = String::with_capacity(
        TOOLS_PROMPT_INTRO.len() + TOOLS_PROMPT_EDIT_LINE.len() + TOOLS_PROMPT_AFTER_EDIT.len(),
    );
    out.push_str(TOOLS_PROMPT_INTRO);
    out.push_str(TOOLS_PROMPT_EDIT_LINE);
    out.push_str(TOOLS_PROMPT_AFTER_EDIT);
    if !parity {
        // The line and the blank line separating it from the next paragraph.
        out = out.replace(&format!("{IN_THINK_PROHIBITION}\n\n"), "");
    }
    out
}

/// The line warning the model off the `SSML` misspelling of the marker.
///
/// Not in the C constants, so it lives outside [`build_tools_prompt_base`] and
/// the parity suite keeps locking the base byte-for-byte. See
/// [`crate::dsml::MARKER_NAMES`]: `<｜SSML｜…>` is accepted as a recovery
/// alias, and this line exists so the model does not learn it as a second
/// legal syntax.
const MARKER_SPELLING_NOTE: &str =
    "The marker is spelled DSML. SSML is not supported; do not write <｜SSML｜…>.\n\n";

/// Inserts [`MARKER_SPELLING_NOTE`] directly after the tool-call shape block.
///
/// Anchored on the shape's closing tag rather than appended at the end so the
/// warning sits next to the syntax it is about, instead of behind the tool
/// schemas. A missing anchor leaves the prompt untouched — the alias in
/// `dsml.rs` still recovers the call, so this is advisory, not load-bearing.
fn insert_marker_spelling_note(out: &mut String) {
    const ANCHOR: &str = "</｜DSML｜tool_calls>\n\n";
    if let Some(at) = out.find(ANCHOR) {
        out.insert_str(at + ANCHOR.len(), MARKER_SPELLING_NOTE);
    }
}

/// The line telling the model that `read` already handles PDFs.
///
/// `docs/LITEPARSE.md` argued the prompt cost of document ingestion was zero
/// because "the model does not learn anything new — documents simply stop being
/// unreadable". That turned out to be false in the one way that matters: a
/// model that believes a `.pdf` is unreadable never calls `read` on one. It
/// shells out to `pdftotext` instead, which is slower, unpaged, and absent on
/// most machines. One sentence next to the reading rules is what makes the
/// feature reachable.
#[cfg(feature = "docparse")]
const DOCUMENT_READ_NOTE: &str = "PDFs are readable: read on a .pdf serves the document as Markdown and pages through it \
     exactly like a text file. Never shell out to pdftotext or a PDF library.\n\n";

/// Inserts [`DOCUMENT_READ_NOTE`] after the paragraph on bounded reads.
///
/// Anchored on the last sentence of that paragraph so the note sits with the
/// other `read` guidance rather than behind the tool schemas. A missing anchor
/// leaves the prompt untouched — routing still works, the model just may not
/// think to use it. Gated on `docparse`: in a build without the parser, `read`
/// on a PDF fails, and promising otherwise would be worse than saying nothing.
#[cfg(feature = "docparse")]
fn insert_document_read_note(out: &mut String) {
    const ANCHOR: &str = "then explain that and use chunks.\n\n";
    if let Some(at) = out.find(ANCHOR) {
        out.insert_str(at + ANCHOR.len(), DOCUMENT_READ_NOTE);
    }
}

/// Without a document parser linked in, the prompt says nothing about PDFs.
#[cfg(not(feature = "docparse"))]
fn insert_document_read_note(_out: &mut String) {}

/// Appends the schemas of native tools plank adds beyond the C-trained table.
///
/// These tools are **not** in the model's training-time tool table, which is
/// why issue #32 requires measuring that the model actually calls them. They
/// are appended here rather than baked into the C constants so the parity
/// suite keeps verifying the base against the reference.
/// The `task` tool schema (text path). Ends on the
/// object close so it slots between the skill and ask blocks.
const TASK_SCHEMA: &str = "{\n\
     \x20 \"type\": \"function\",\n\
     \x20 \"function\": {\n\
     \x20   \"name\": \"task\",\n\
     \x20   \"description\": \"Track a plan that survives context compaction. op='add' appends a pending task (needs 'subject') and returns its id; op='update' changes a task's status/subject (needs 'id'); op='list' returns every task. Statuses: pending, in_progress, completed. The current list is shown to you each turn, so use 'list' only to recover it.\",\n\
     \x20   \"parameters\": {\n\
     \x20     \"type\": \"object\",\n\
     \x20     \"properties\": {\n\
     \x20       \"op\": {\"type\": \"string\", \"description\": \"add, update, or list\"},\n\
     \x20       \"id\": {\"type\": \"string\", \"description\": \"task id (for update)\"},\n\
     \x20       \"subject\": {\"type\": \"string\", \"description\": \"task description (for add; optional rename on update)\"},\n\
     \x20       \"status\": {\"type\": \"string\", \"description\": \"pending, in_progress, or completed (for update)\"},\n\
     \x20       \"active_form\": {\"type\": \"string\", \"description\": \"present-tense form shown while the task is in progress, e.g. 'Refactoring the parser'\"}\n\
     \x20     },\n\
     \x20     \"required\": [\"op\"]\n\
     \x20   }\n\
     \x20 }\n\
     }\n";

fn append_native_extra_schemas(out: &mut String) {
    out.push_str(
        "\n{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"glob\",\n\
         \x20   \"description\": \"Find files by name pattern across a directory tree. Use this instead of shelling out to find or ls. '**' crosses directory boundaries, '*' matches within one path component. Results are paths relative to the search root, sorted, capped at 100.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"pattern\": {\"type\": \"string\", \"description\": \"glob pattern, e.g. '*.rs', '**/*test*', 'src/**/mod.rs'\"},\n\
         \x20       \"path\": {\"type\": \"string\", \"description\": \"directory to search from; defaults to the working directory\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"pattern\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n\
         {\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"skill\",\n\
         \x20   \"description\": \"Invoke an installed skill (a packaged procedure) by name; its instructions are returned for you to follow. Call with no name to list the installed skills first.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"name\": {\"type\": \"string\", \"description\": \"skill name; omit to enumerate installed skills\"},\n\
         \x20       \"args\": {\"type\": \"string\", \"description\": \"arguments passed to the skill\"}\n\
         \x20     }\n\
         \x20   }\n\
         \x20 }\n\
         }\n",
    );
    out.push_str(TASK_SCHEMA);
    out.push_str(
        "{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"ask\",\n\
         \x20   \"description\": \"Ask the user a multiple-choice question and block until they answer. Use this instead of guessing when a turn is genuinely ambiguous. 'question' is the full question, 'header' a short (~12 char) label, 'options' a JSON array of 2 to 7 {\\\"label\\\",\\\"description\\\"} choices. Set 'multi' to true to allow several selections. Returns the selected label(s). In non-interactive mode it returns immediately telling you no user is available.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"question\": {\"type\": \"string\", \"description\": \"the full question, phrased as a question\"},\n\
         \x20       \"header\": {\"type\": \"string\", \"description\": \"short UI label, ~12 characters\"},\n\
         \x20       \"options\": {\"type\": \"string\", \"description\": \"JSON array of 2 to 7 {\\\"label\\\",\\\"description\\\"} objects\"},\n\
         \x20       \"multi\": {\"type\": \"string\", \"description\": \"true to allow selecting more than one option (default false)\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"question\", \"header\", \"options\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n",
    );
    append_agent_and_plan_schemas(out);
    // The `recall` (M8), `fanout` (M9) and `run_code` (M10) tools are
    // deliberate deviations from the C reference: the C agent has none of them.
    // They are advertised by default and can be switched off individually
    // (`tools.recall` / `tools.fanout` / `tools.runCode`). Because they are in
    // the prompt, `fp1` differs from the C agent's fingerprint — the versioned
    // deviation documented in docs/SYSTEM-PROMPT-OVERRIDES.md. What parity
    // still holds byte-for-byte is the C-*derived* text, which
    // `tools_prompt_matches_c_source` checks independently of this list.
    if crate::settings::active().tools.recall {
        append_recall_schema(out);
    }
    if crate::settings::active().tools.fanout {
        append_fanout_schema(out);
    }
    if crate::settings::active().tools.run_code {
        append_run_code_schema(out);
    }
}

/// Appends the `recall` tool schema (M8): search prior sessions and the
/// current one's pre-compaction portion, scoped to the current project.
fn append_recall_schema(out: &mut String) {
    out.push_str(
        "{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"recall\",\n\
         \x20   \"description\": \"Search your prior sessions and the current one's pre-compaction portion for a query, scoped to the current project. Returns matching session titles, ages and snippets.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"query\": {\"type\": \"string\", \"description\": \"the text to search for\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"query\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n",
    );
}

/// Appends the `fanout` tool schema (M9): run independent subtasks and join
/// their reports deterministically. The description deliberately promises a
/// deterministic join, not speed — on the `ds4_engine` path subtasks are
/// interleaved on one Metal queue, not parallel.
fn append_fanout_schema(out: &mut String) {
    out.push_str(
        "{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"fanout\",\n\
         \x20   \"description\": \"Run a list of independent subtasks, each delegated to a named sub-agent, and join their reports into one deterministic result. Subtasks run serially on the shared engine — this buys structure, not speed.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"subtasks\": {\"type\": \"array\", \"description\": \"JSON array of {\\\"name\\\": <agent name>, \\\"task\\\": <task text>} objects\", \"items\": {\"type\": \"object\"}}\n\
         \x20     },\n\
         \x20     \"required\": [\"subtasks\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n",
    );
}

/// Appends the `run_code` tool schema (M10): run a small script of named
/// operations (read/glob/edit/bash) through the existing tool dispatch path,
/// so the consent and sandbox checks apply. The minimal viable guest language
/// is a sequence of operations; a full interpreted guest compiled to the WASM
/// host is a documented follow-up.
fn append_run_code_schema(out: &mut String) {
    out.push_str(
        "{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"run_code\",\n\
         \x20   \"description\": \"Run a small script of named operations (read <path>, glob <pattern>, edit <path> <old> <new>, bash <command>) through the same checks as the tools, collecting outputs into one result. One line per operation, one operation per line.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"script\": {\"type\": \"string\", \"description\": \"the script, one operation per line\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"script\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n",
    );
}

/// Appends the `agent` (sub-agent delegation) and plan-mode tool schemas
/// (issue #50). Split from [`append_native_extra_schemas`] to keep each under
/// the function-length lint; both are native tools outside the C-trained table.
fn append_agent_and_plan_schemas(out: &mut String) {
    // Built as an owned line rather than inlined into the continued literal: a
    // `\`-continued literal strips the next line's leading whitespace, which
    // would corrupt the schema indentation (see CLAUDE.md). No roster is listed
    // here — see `push_agent_and_plan_specs` — so this is byte-identical to the
    // pre-roster build, which is what keeps the parity fixtures valid.
    let name_line = format!(
        "\x20       \"name\": {{\"type\": \"string\", \"description\": \"{AGENT_NAME_DESC_EMPTY}\"}}\n"
    );
    out.push_str(
        "{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"agent\",\n\
         \x20   \"description\": \"Delegate a self-contained sub-task to a fresh sub-agent that works in its own scoped context and returns only a final report. Use this to keep your own context small: hand off open-ended research (locate where X is handled, summarize how Y works) or a bounded multi-step chore, then continue from its report. 'task' is a complete, standalone instruction; 'name' optionally selects a configured agent persona. The sub-agent cannot ask you questions, so make 'task' fully specified.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"task\": {\"type\": \"string\", \"description\": \"the complete, standalone task to delegate; include all needed context\"},\n",
    );
    out.push_str(&name_line);
    out.push_str(
        "\x20     },\n\
         \x20     \"required\": [\"task\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n",
    );
    out.push_str(
        "{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"EnterPlanMode\",\n\
         \x20   \"description\": \"Enter read-only plan mode: research and design without changing anything. While it is active, write/edit/bash are refused; only read-only tools work. Use it when a task is risky or ambiguous and the user should approve an approach before you edit. Exit with ExitPlanMode.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {}\n\
         \x20   }\n\
         \x20 }\n\
         }\n\
         {\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"ExitPlanMode\",\n\
         \x20   \"description\": \"Leave plan mode by presenting your proposed plan for the user's approval. On approval the read-only gate lifts and you may edit; otherwise plan mode stays on and you should refine the plan. 'plan' is the full proposed plan.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"plan\": {\"type\": \"string\", \"description\": \"the full plan to carry out, for the user to approve\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"plan\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n",
    );
    append_worktree_schemas(out);
}

/// Appends the `EnterWorktree` / `ExitWorktree` schemas.
///
/// The descriptions are deliberately restrictive. A worktree is a whole second
/// checkout, and a model that reaches for one whenever it sees the word
/// "branch" would strand the user's work in a directory they never asked for —
/// so the tool is scoped to an explicit request, and ordinary branch work is
/// pointed back at `bash` and git.
fn append_worktree_schemas(out: &mut String) {
    out.push_str(
        "{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"EnterWorktree\",\n\
         \x20   \"description\": \"Move into an isolated git worktree: a separate checkout of this repository, on its own branch, where your edits cannot affect the main working copy. Every tool's working directory switches into it until you call ExitWorktree. Use this ONLY when the user explicitly asks for a worktree or for isolation from their current checkout. Do NOT use it for ordinary branch or feature work — run git through bash for that. 'name' is the worktree name; letters, digits, '.', '_', '-' and '/' only.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"name\": {\"type\": \"string\", \"description\": \"name for the worktree, e.g. 'refactor-parser'\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"name\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n\
         {\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"ExitWorktree\",\n\
         \x20   \"description\": \"Leave the worktree entered with EnterWorktree and return to the original working directory. action 'keep' leaves the worktree and its branch on disk so the user can review, merge, or resume the work; action 'remove' deletes both. A remove is refused when the worktree holds uncommitted files or commits that are not on the base branch, unless you also pass discard_changes true — so prefer 'keep' whenever there is work worth saving.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"action\": {\"type\": \"string\", \"description\": \"'keep' to leave the worktree in place, 'remove' to delete it\"},\n\
         \x20       \"discard_changes\": {\"type\": \"string\", \"description\": \"true to remove the worktree even though it holds work that would be lost\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"action\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n",
    );
}

/// Returns the short DSML syntax reminder (verbatim from C).
#[must_use]
pub fn dsml_syntax_reminder() -> &'static str {
    "DSML syntax reminder:\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n"
}

/// Builds the full system prompt reminder block, framed like the C version.
///
/// Mirrors `agent_build_system_prompt_reminder`: the tools prompt wrapped in
/// start/end reminder markers.
#[must_use]
pub fn build_system_prompt_reminder(
    mcp_servers: &[crate::tools::mcp::McpServer],
    parity: bool,
) -> String {
    let mut out = String::from("\n\n[System prompt reminder follows.]\n");
    out.push_str(&build_tools_prompt_parts(mcp_servers, parity).0);
    out.push_str("[End system prompt reminder.]\n\n");
    out
}

/// The attribution trailer plank asks the model to end its commit messages
/// with, disabled by `git.signCommits` in `settings.json`.
pub const COMMIT_SIGNATURE_TRAILER: &str = "--Co-Authored by Plank (https://plank-agent.dev)";

/// Instruction that puts [`COMMIT_SIGNATURE_TRAILER`] on the commits the model
/// makes.
///
/// It is appended *after* [`SplitSystemPrompt::trusted_len`] on purpose: it is
/// plank's own text, but nothing in it teaches DSML, so there is no reason to
/// widen the span that gets the control-text tokenizer. It is settings-derived
/// yet still cache-stable — `settings.json` does not change within a session,
/// and editing it correctly re-fingerprints `sysprompt.kv`.
pub const COMMIT_SIGNATURE_INSTRUCTION: &str = concat!(
    "## Commits\n\n",
    "When you create a git commit, end its message with a blank line followed by the",
    " single line `--Co-Authored by Plank (https://plank-agent.dev)`. Leave the trailer out only if the user",
    " asks you to, or if the message already ends with it.\n"
);

/// Composes the initial system prompt: tools prompt plus optional user text.
///
/// Mirrors `agent_append_system_prompt`: the built-in tools prompt comes
/// first, and non-empty user `-sys` text is appended after a blank line. The
/// two halves are tokenized differently, as in the C — see
/// [`SplitSystemPrompt::trusted_len`], which [`build_system_prompt_parts`] reports.
/// This function returns only the composed text, for callers that render or
/// fingerprint it rather than tokenize it.
#[must_use]
/// **Cache-boundary rule** (docs/SYSTEM-PROMPT.md): everything composed here
/// enters the fingerprinted `sysprompt.kv` KV prefix, so only inputs that are
/// stable across sessions are allowed — the verbatim tools prompt, MCP
/// schemas/instructions, and `-sys` text. Per-session data (date, git state,
/// AGENTS.md) belongs in [`crate::context::ContextContent`] instead; the
/// `fingerprinted_prompt_contains_no_volatile_bytes` test guards this.
pub fn build_system_prompt(
    user_system: &str,
    mcp_servers: &[crate::tools::mcp::McpServer],
    parity: bool,
) -> String {
    build_system_prompt_parts(user_system, mcp_servers, parity).text
}

/// A composed system prompt together with the boundary that decides how each
/// half is tokenized.
#[derive(Debug, Clone)]
pub struct SplitSystemPrompt {
    /// The composed prompt, identical to [`build_system_prompt`]'s output.
    pub text: String,
    /// Byte length of the leading span that is **trusted control text**.
    ///
    /// This span is tokenized as *rendered chat*, so the literal `｜DSML｜` in
    /// its examples becomes the model's dedicated DSML vocabulary token rather
    /// than a spelled-out BPE sequence — the form the model was trained on.
    /// The C does the same in `agent_append_system_prompt`, with the comment:
    ///
    /// > The built-in tool prompt is trusted DS4 control text. Tokenize it like
    /// > a rendered chat prompt so the literal ｜DSML｜ markers in the examples
    /// > become the model's dedicated DSML token. Do not apply that tokenizer
    /// > to user supplied -sys text: arbitrary user text containing
    /// > `<｜User｜>`, `<think>`, or `｜DSML｜` must remain plain content, not
    /// > control tokens.
    ///
    /// That prohibition is a prompt-injection boundary, and plank's is **wider
    /// than the C's**: the C's tools prompt is entirely built in, while plank
    /// appends MCP tool schemas and server instructions to it. Those come from
    /// third-party processes, so they sit *outside* the trusted span alongside
    /// `-sys` text. Every `｜DSML｜` the prompt teaches is in the built-in part,
    /// so nothing is lost by drawing the line there.
    ///
    /// Widening this span to cover MCP or user text would let either forge a
    /// turn boundary. Do not.
    pub trusted_len: usize,
}

/// [`build_system_prompt`] with the trusted/untrusted boundary reported.
#[must_use]
pub fn build_system_prompt_parts(
    user_system: &str,
    mcp_servers: &[crate::tools::mcp::McpServer],
    parity: bool,
) -> SplitSystemPrompt {
    build_system_prompt_parts_with_wasm(user_system, mcp_servers, &[], parity)
}

/// [`build_system_prompt_parts`] with WASM component tools folded in.
///
/// A separate entry point rather than a fourth parameter on the existing one:
/// every caller that has no components — every test, every sub-agent path —
/// keeps working unchanged, and the one caller that does have them says so
/// explicitly.
#[must_use]
pub fn build_system_prompt_parts_with_wasm(
    user_system: &str,
    mcp_servers: &[crate::tools::mcp::McpServer],
    wasm_tools: &[&crate::wasmreg::WasmTool],
    parity: bool,
) -> SplitSystemPrompt {
    let (mut text, trusted_len) =
        build_tools_prompt_parts_with_wasm(mcp_servers, wasm_tools, parity);
    if crate::settings::active().git.sign_commits {
        text.push_str("\n\n");
        text.push_str(COMMIT_SIGNATURE_INSTRUCTION);
    }
    if !user_system.is_empty() {
        text.push_str("\n\n");
        text.push_str(user_system);
    }
    SplitSystemPrompt { text, trusted_len }
}

/// Formats the session-start datetime context line for the given instant.
///
/// Mirrors `agent_worker_maybe_append_datetime_context`: the timestamp is the
/// local time formatted as `%Y-%m-%d %H:%M:%S %Z`, falling back to the raw
/// Unix seconds if formatting fails.
#[must_use]
pub fn datetime_context_line(now: SystemTime) -> String {
    let secs = match now.duration_since(UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        Err(e) => -i64::try_from(e.duration().as_secs()).unwrap_or(i64::MAX),
    };
    let when = format_local(secs).unwrap_or_else(|| secs.to_string());
    format!(
        "Current local date and time at session start: {when}. \
         Use this only when date or time matters."
    )
}

/// Formats Unix seconds as local time `%Y-%m-%d %H:%M:%S %Z`, or `None` on failure.
fn format_local(secs: i64) -> Option<String> {
    let t: libc::time_t = secs;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: `t` and `tm` are valid for reads/writes; localtime_r fills `tm`
    // or returns NULL on failure.
    if unsafe { libc::localtime_r(&raw const t, &raw mut tm) }.is_null() {
        return None;
    }
    let mut buf = [0u8; 128];
    let fmt = c"%Y-%m-%d %H:%M:%S %Z";
    // SAFETY: `buf` is a writable buffer of the given length, `fmt` and `tm`
    // are valid; strftime NUL-terminates on success and returns 0 on failure.
    let n = unsafe {
        libc::strftime(
            buf.as_mut_ptr().cast::<libc::c_char>(),
            buf.len(),
            fmt.as_ptr(),
            &raw const tm,
        )
    };
    if n == 0 {
        return None;
    }
    Some(String::from_utf8_lossy(&buf[..n]).into_owned())
}

/// Pressure-controlled policy for re-injecting the system prompt reminder.
///
/// Mirrors `agent_worker_maybe_append_system_prompt_reminder` together with
/// `agent_worker_note_system_prompt_seen`. Positions are token-estimate
/// offsets into the transcript (the C code uses `transcript.len`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SystemPromptReminder {
    last_reminder_at: i32,
}

impl SystemPromptReminder {
    /// Creates a policy that has not yet seen a system prompt.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that the system prompt was (re)seen at `current_pos`.
    pub fn note_seen(&mut self, current_pos: i32) {
        self.last_reminder_at = current_pos;
    }

    /// Decides whether to re-inject the reminder at `current_pos`.
    ///
    /// Returns `true` when at least [`SYSTEM_PROMPT_REMINDER_TOKENS`] have
    /// accumulated since the prompt was last seen; the caller must then
    /// inject [`build_system_prompt_reminder`]. As in the C code, a
    /// non-positive last-seen position only records the current position.
    pub fn should_remind(&mut self, current_pos: i32) -> bool {
        if self.last_reminder_at <= 0 {
            self.note_seen(current_pos);
            return false;
        }
        if current_pos - self.last_reminder_at < SYSTEM_PROMPT_REMINDER_TOKENS {
            return false;
        }
        self.note_seen(current_pos);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn non_trained_tool_schemas_are_always_present() {
        // The `task`, `agent`, and plan-mode tools used to be gated behind
        // `settings.tools`; they are unconditional now, so a default-settings
        // build must still advertise all four.
        let specs = provider_tool_registry(&[]);
        for name in ["task", "agent", "EnterPlanMode", "ExitPlanMode"] {
            assert!(
                specs.iter().any(|s| s.name == name),
                "provider registry is missing {name}"
            );
        }
        let mut text = String::new();
        append_native_extra_schemas(&mut text);
        for name in [
            "\"task\"",
            "\"agent\"",
            "\"EnterPlanMode\"",
            "\"ExitPlanMode\"",
        ] {
            assert!(text.contains(name), "text schemas are missing {name}");
        }
    }

    /// The three plank-native tools (M8/M9/M10) are advertised by default and
    /// can be switched off. They are still deviations from the C reference —
    /// the C agent has none of them — so `fp1` differs from the C agent's
    /// fingerprint for every session. What parity still guarantees is that the
    /// C-*derived* text is byte-identical, which `tools_prompt_matches_c_source`
    /// checks independently of this schema list.
    #[test]
    fn recall_schema_is_advertised_by_default_and_can_be_disabled() {
        let mut text = String::new();
        append_native_extra_schemas(&mut text);
        assert!(
            text.contains("\"recall\""),
            "recall is advertised by default"
        );
        let mut s = crate::settings::Settings::default();
        s.tools.recall = false;
        crate::settings::install_for_test(s);
        let mut text = String::new();
        append_native_extra_schemas(&mut text);
        assert!(
            !text.contains("\"recall\""),
            "tools.recall = false removes the schema"
        );
    }

    #[test]
    fn fanout_schema_is_advertised_by_default_and_can_be_disabled() {
        let mut text = String::new();
        append_native_extra_schemas(&mut text);
        assert!(
            text.contains("\"fanout\""),
            "fanout is advertised by default"
        );
        // The description must promise a deterministic join, not speed: on the
        // ds4_engine path subtasks are interleaved on one Metal queue.
        assert!(text.contains("deterministic"), "{text}");
        let mut s = crate::settings::Settings::default();
        s.tools.fanout = false;
        crate::settings::install_for_test(s);
        let mut text = String::new();
        append_native_extra_schemas(&mut text);
        assert!(
            !text.contains("\"fanout\""),
            "tools.fanout = false removes the schema"
        );
    }

    #[test]
    fn run_code_schema_is_advertised_by_default_and_can_be_disabled() {
        let mut text = String::new();
        append_native_extra_schemas(&mut text);
        assert!(
            text.contains("\"run_code\""),
            "run_code is advertised by default"
        );
        let mut s = crate::settings::Settings::default();
        s.tools.run_code = false;
        crate::settings::install_for_test(s);
        let mut text = String::new();
        append_native_extra_schemas(&mut text);
        assert!(
            !text.contains("\"run_code\""),
            "tools.runCode = false removes the schema"
        );
    }

    #[test]
    fn tools_prompt_contains_verbatim_phrases() {
        let p = build_tools_prompt(&[], true);
        assert!(p.starts_with("You are a coding agent running in a local workspace."));
        assert!(p.contains("<｜DSML｜tool_calls>"));
        assert!(p.contains("Tool calls are not allowed inside <think></think>"));
        assert!(p.contains("## Editing files"));
        assert!(p.contains("never close old immediately after [upto]"));
        assert!(p.contains("Use google_search to find web pages."));
        assert!(p.contains("### Available Tool Schemas"));
        assert!(p.contains("\"name\": \"bash_stop\""));
        assert!(p.contains("- Always use strict syntax for DSML tool stanzas.\n"));
        // The C-derived base ends with the rules text; native tools (glob) are
        // appended on top of it by `build_tools_prompt`.
        assert!(
            build_tools_prompt_base(true)
                .ends_with("unless explicitly asked otherwise by the user.\n")
        );
        assert!(
            p.contains("\"name\": \"glob\""),
            "native glob tool is appended"
        );
    }

    /// Guards the static/volatile boundary (docs/SYSTEM-PROMPT.md): the
    /// composed system prompt is what `sysprompt.kv` fingerprints, so any
    /// per-session bytes (date, git state, AGENTS.md) sneaking in would make
    /// the disk snapshot rebuild on every launch. Volatile context belongs in
    /// the first user turn (`context::ContextContent`), never here.
    #[test]
    fn fingerprinted_prompt_contains_no_volatile_bytes() {
        let a = build_system_prompt("user -sys text", &[], true);
        let b = build_system_prompt("user -sys text", &[], true);
        assert_eq!(a, b, "system prompt must be deterministic");
        let today = crate::context::current_local_iso_date();
        assert!(
            !a.contains(&today),
            "today's date leaked into the cached prefix"
        );
        for marker in [
            "Today's date",
            "This is the git status",
            "Current branch:",
            "Main branch",
            "Git user:",
            "Agent instructions:",
        ] {
            assert!(
                !a.contains(marker),
                "volatile marker {marker:?} leaked into the cached prefix"
            );
        }
    }

    #[test]
    fn commit_signature_instruction_is_in_the_default_prompt() {
        // Default settings sign commits, and the trailer must land outside the
        // trusted control-text span.
        let parts = build_system_prompt_parts("", &[], true);
        assert!(
            parts
                .text
                .contains("--Co-Authored by Plank (https://plank-agent.dev)")
        );
        assert!(
            !parts.text[..parts.trusted_len]
                .contains("--Co-Authored by Plank (https://plank-agent.dev)")
        );
        assert!(
            provider_system_prompt("").contains("--Co-Authored by Plank (https://plank-agent.dev)")
        );
    }

    #[test]
    fn provider_system_prompt_omits_dsml() {
        let p = provider_system_prompt("Be terse.");
        // The provider prompt must not teach DSML syntax (native tools replace
        // it) and must not carry DS4-only framing (design §4.4 / constraint 3).
        assert!(!p.contains("DSML"));
        assert!(!p.contains("<｜DSML｜"));
        assert!(!p.contains("### Available Tool Schemas"));
        // But it keeps the behavioral guidance and appends user -sys text.
        assert!(p.contains("Editing files"));
        assert!(p.ends_with("Be terse."));
    }

    /// The C-parity fixtures depend on this: an empty roster must not perturb a
    /// single byte of either schema path. If this fails, the fix is in the code,
    /// never `PLANK_REGEN_FIXTURES=1`.
    #[test]
    fn empty_roster_emits_todays_bytes_exactly() {
        let text = build_tools_prompt(&[], true);
        assert!(text.contains("\"name\": \"agent\""), "agent schema present");
        assert!(
            !text.contains("\"enum\""),
            "no enum key with an empty roster: {text}"
        );
        let specs = provider_tool_registry(&[]);
        let agent = specs
            .iter()
            .find(|s| s.name == "agent")
            .expect("agent spec");
        assert!(
            agent.parameters["properties"]["name"].get("enum").is_none(),
            "no enum with an empty roster"
        );
    }

    /// The roster deliberately does *not* reach either prompt shape any more: it
    /// lives in the session context (`context::agent_roster_context`) so editing
    /// a definition rebuilds the small project-tier cache instead of
    /// invalidating the fingerprinted system prompt. Both shapes must therefore
    /// be independent of what is on disk, which is also what keeps the parity
    /// fixtures valid.
    #[test]
    fn no_roster_reaches_either_prompt_shape() {
        let (text, trusted) = build_tools_prompt_parts(&[], true);
        assert!(!text.contains("\"enum\""), "no enum key at all: {text}");
        assert!(
            text.contains("\"name\": {\"type\": \"string\", \"description\":"),
            "`name` is a plain string: {text}"
        );

        let specs = provider_tool_registry(&[]);
        let agent = specs
            .iter()
            .find(|s| s.name == "agent")
            .expect("agent spec");
        assert!(
            agent.parameters["properties"]["name"].get("enum").is_none(),
            "no enum in the structured spec either"
        );

        // Definitions on disk cannot move a byte of it. Same inputs, same
        // output, regardless of what a roster would have said.
        let (again, trusted_again) = build_tools_prompt_parts(&[], true);
        assert_eq!(text, again);
        assert_eq!(trusted, trusted_again);
    }

    #[test]
    fn provider_tool_registry_advertises_only_primary_mcp_tools() {
        // Regression (perf): every MCP tool used to get a full function spec on
        // the provider path, ignoring `primary`. With tokensave connected that
        // was 82 schemas / 63 KB of JSON resent on every request, inflating
        // prefill and slowing every later decode by enlarging the KV context.
        // The text path had always filtered; this path now matches it.
        use crate::tools::mcp::{McpServer, McpTool};
        let tool = |name: &str, primary: bool| McpTool {
            name: name.to_string(),
            description: format!("does {name}"),
            schema_json: "{\"type\":\"object\",\"properties\":{}}".to_string(),
            primary,
        };
        let rec = crate::tools::mcp_advert::AdvertRecord {
            server: "srv".to_string(),
            instructions: String::new(),
            tools: vec![tool("kept", true), tool("hidden", false)],
            resources: Vec::new(),
        };
        let servers = vec![McpServer::offline(&rec)];
        let specs = provider_tool_registry(&servers);
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        // Namespaced: dispatch routes on the `mcp__` prefix, so a bare name
        // would be advertised and then rejected as an unknown tool.
        assert!(
            names.contains(&"mcp__srv__kept"),
            "primary tool must keep its schema, qualified: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("hidden")),
            "non-primary tool must not carry a schema: {names:?}"
        );
        assert!(
            !names.contains(&"kept"),
            "an unqualified name reaches the unknown-tool fallthrough: {names:?}"
        );
        // ...but it must stay *reachable*: a provider rejects an undeclared
        // function name, so dropping the schema without the escape hatch would
        // lose the tool entirely rather than just its parameters.
        assert!(names.contains(&"mcp_call"), "{names:?}");
        assert!(names.contains(&"mcp_describe"), "{names:?}");
        let mcp_call = specs.iter().find(|s| s.name == "mcp_call").unwrap();
        assert!(
            mcp_call
                .description
                .contains("mcp__srv__hidden: does hidden"),
            "the directory must name the hidden tool: {}",
            mcp_call.description
        );
        assert!(
            !mcp_call.description.contains("mcp__srv__kept"),
            "a primary tool has its own spec and must not be listed twice"
        );
    }

    #[test]
    fn every_provider_spec_name_is_routable_by_dispatch() {
        // The bug this pins: MCP specs were advertised under their bare name
        // while dispatch routes MCP calls on the `mcp__` prefix, so the model
        // was offered `tokensave_status`, called it, and was told no such tool
        // exists — with the mcp_call directory simultaneously advertising the
        // qualified spelling. Asserting the two paths agree is not enough; the
        // name has to be one dispatch will actually accept.
        use crate::tools::mcp::{McpServer, McpTool};
        let rec = crate::tools::mcp_advert::AdvertRecord {
            server: "srv".to_string(),
            instructions: String::new(),
            tools: vec![
                McpTool {
                    name: "alpha".to_string(),
                    description: "a".to_string(),
                    schema_json: "{}".to_string(),
                    primary: true,
                },
                McpTool {
                    name: "beta".to_string(),
                    description: "b".to_string(),
                    schema_json: "{}".to_string(),
                    primary: false,
                },
            ],
            resources: Vec::new(),
        };
        let specs = provider_tool_registry(&[McpServer::offline(&rec)]);
        // No hand-copied list of native tools here (it would rot against the
        // dispatch match): assert the precise invariant instead — a spec named
        // after a *server-advertised* tool must carry the `mcp__` prefix, since
        // that prefix is the only thing routing it to the MCP client.
        for advertised in ["alpha", "beta"] {
            assert!(
                !specs.iter().any(|s| s.name == advertised),
                "{advertised:?} is advertised bare; dispatch would reject it as unknown"
            );
        }
        let mcp_named: Vec<&str> = specs
            .iter()
            .filter(|s| s.name.contains("alpha") || s.name.contains("beta"))
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(
            mcp_named,
            vec!["mcp__srv__alpha"],
            "only the primary tool gets a spec, and it must be qualified"
        );
        // The directory must use the same spelling, so mcp_call receives a name
        // `invoke_mcp_tool` can split.
        let mcp_call = specs.iter().find(|s| s.name == "mcp_call").unwrap();
        assert!(mcp_call.description.contains("mcp__srv__beta"));
    }

    #[test]
    fn provider_tool_registry_omits_the_escape_hatch_when_every_tool_is_primary() {
        // The common case (a server with no `primaryTools` filter) must not pay
        // for two extra specs and an empty directory.
        use crate::tools::mcp::{McpServer, McpTool};
        let rec = crate::tools::mcp_advert::AdvertRecord {
            server: "srv".to_string(),
            instructions: String::new(),
            tools: vec![McpTool {
                name: "only".to_string(),
                description: "d".to_string(),
                schema_json: "{}".to_string(),
                primary: true,
            }],
            resources: Vec::new(),
        };
        let specs = provider_tool_registry(&[McpServer::offline(&rec)]);
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"mcp__srv__only"), "{names:?}");
        assert!(!names.contains(&"mcp_call"), "{names:?}");
        assert!(!names.contains(&"mcp_describe"), "{names:?}");
    }

    #[test]
    fn provider_tool_registry_parses_builtin_schemas() {
        let specs = provider_tool_registry(&[]);
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        for want in ["read", "write", "edit", "bash", "search", "google_search"] {
            assert!(names.contains(&want), "missing {want} in {names:?}");
        }
        let read = specs.iter().find(|s| s.name == "read").unwrap();
        assert_eq!(read.parameters["type"], "object");
        assert!(read.parameters["properties"].get("path").is_some());
        assert!(!read.description.is_empty());
    }

    #[test]
    fn dsml_reminder_shape() {
        let r = dsml_syntax_reminder();
        assert!(r.starts_with("DSML syntax reminder:\n"));
        assert!(r.contains("<｜DSML｜invoke name=\"$TOOL_NAME\">"));
        assert!(r.ends_with("</｜DSML｜tool_calls>\n"));
    }

    #[test]
    fn system_prompt_reminder_framing() {
        let r = build_system_prompt_reminder(&[], true);
        assert!(r.starts_with("\n\n[System prompt reminder follows.]\n"));
        assert!(r.ends_with("[End system prompt reminder.]\n\n"));
        assert!(r.contains("## Tools"));
    }

    /// The marker-spelling warning must reach every prompt surface the model
    /// sees, sit next to the shape it is about, and stay out of the C-locked
    /// base so the parity suite keeps checking that base byte-for-byte.
    /// The trusted span must cover every `｜DSML｜` the prompt teaches — that is
    /// the whole point of tokenizing it as rendered chat — and must stop before
    /// anything a third party controls.
    #[test]
    fn trusted_span_covers_the_dsml_examples_and_nothing_untrusted() {
        for parity in [true, false] {
            let p = build_system_prompt_parts("USER SYS TEXT", &[], parity);
            let trusted = &p.text[..p.trusted_len];
            let rest = &p.text[p.trusted_len..];

            // Every marker the prompt teaches is inside the trusted span.
            assert!(trusted.contains("<｜DSML｜tool_calls>"), "parity={parity}");
            assert!(trusted.contains("<｜DSML｜invoke"), "parity={parity}");
            assert!(trusted.contains("<｜DSML｜parameter"), "parity={parity}");
            assert!(
                !rest.contains("｜DSML｜"),
                "a DSML marker outside the trusted span would be spelled out: {rest:?}"
            );

            // User `-sys` text never is: as control tokens it could forge a turn.
            assert!(!trusted.contains("USER SYS TEXT"), "parity={parity}");
            assert!(rest.contains("USER SYS TEXT"), "parity={parity}");
        }
    }

    /// The boundary is a property of the built-in prompt alone, so everything
    /// added after it — MCP schemas in a live session, `-sys` text here — lands
    /// outside, and the trusted prefix itself never moves.
    #[test]
    fn trusted_span_ends_before_anything_appended() {
        let bare = build_system_prompt_parts("", &[], true);
        // Only the commit-signature instruction sits past the boundary here.
        assert_eq!(
            &bare.text[bare.trusted_len..],
            format!("\n\n{COMMIT_SIGNATURE_INSTRUCTION}"),
            "nothing untrusted but the signature note"
        );

        let with_user = build_system_prompt_parts("Be terse.", &[], true);
        assert_eq!(with_user.trusted_len, bare.trusted_len);
        assert!(with_user.text.len() > with_user.trusted_len);
        assert!(with_user.text.starts_with(&bare.text[..bare.trusted_len]));
    }

    /// The composed text is unchanged by the split: the KV fingerprint and the
    /// rendered transcript must see exactly what they always did.
    #[test]
    fn parts_text_matches_the_plain_composition() {
        for parity in [true, false] {
            for user in ["", "Be terse."] {
                assert_eq!(
                    build_system_prompt_parts(user, &[], parity).text,
                    build_system_prompt(user, &[], parity),
                );
            }
        }
    }

    #[test]
    fn marker_spelling_note_lands_after_the_shape_but_not_in_the_base() {
        for parity in [true, false] {
            let base = build_tools_prompt_base(parity);
            assert!(!base.contains("SSML"), "base must stay verbatim C");

            let prompt = build_tools_prompt(&[], parity);
            assert_eq!(prompt.matches(MARKER_SPELLING_NOTE).count(), 1);
            let shape_end = prompt.find("</｜DSML｜tool_calls>\n\n").unwrap()
                + "</｜DSML｜tool_calls>\n\n".len();
            assert!(prompt[shape_end..].starts_with(MARKER_SPELLING_NOTE));
            // The reminder re-sends the tools prompt, so it inherits the note.
            assert!(build_system_prompt_reminder(&[], parity).contains(MARKER_SPELLING_NOTE));
            assert!(build_system_prompt("", &[], parity).contains(MARKER_SPELLING_NOTE));
        }
    }

    /// The PDF note sits with the other `read` guidance, reaches the reminder,
    /// and stays out of the C-locked base.
    #[cfg(feature = "docparse")]
    #[test]
    fn document_read_note_lands_after_the_reading_rules_but_not_in_the_base() {
        for parity in [true, false] {
            assert!(
                !build_tools_prompt_base(parity).contains("PDFs are readable"),
                "base must stay verbatim C"
            );
            let prompt = build_tools_prompt(&[], parity);
            assert_eq!(prompt.matches(DOCUMENT_READ_NOTE).count(), 1);
            let anchor = "then explain that and use chunks.\n\n";
            let at = prompt.find(anchor).unwrap() + anchor.len();
            assert!(prompt[at..].starts_with(DOCUMENT_READ_NOTE));
            assert!(build_system_prompt_reminder(&[], parity).contains(DOCUMENT_READ_NOTE));
        }
    }

    #[test]
    fn system_prompt_composition() {
        // Default settings append the commit-signature instruction between the
        // tools prompt and the user's `-sys` text.
        let signature = format!("\n\n{COMMIT_SIGNATURE_INSTRUCTION}");
        assert_eq!(
            build_system_prompt("", &[], true),
            format!("{}{signature}", build_tools_prompt(&[], true))
        );
        let with_extra = build_system_prompt("Be terse.", &[], true);
        assert!(with_extra.starts_with(&build_tools_prompt(&[], true)));
        assert!(with_extra.ends_with("\n\nBe terse."));
    }

    #[test]
    fn non_parity_prompt_drops_only_the_in_think_prohibition() {
        let parity = build_tools_prompt_base(true);
        let permissive = build_tools_prompt_base(false);
        assert!(parity.contains(IN_THINK_PROHIBITION));
        assert!(
            !permissive.contains("Tool calls are not allowed inside"),
            "the prohibition must be gone in permissive mode"
        );
        // The prohibition appears exactly once in the parity prompt, so the
        // `.replace` in `build_tools_prompt_base` is anchored to a single,
        // unambiguous occurrence rather than risking a multi-match strip.
        assert_eq!(parity.matches(IN_THINK_PROHIBITION).count(), 1);
        // Independent of the strip logic itself: permissive is shorter by
        // exactly the prohibition line plus the two newlines that separated
        // it from the next paragraph, and nothing else.
        assert_eq!(
            parity.len() - permissive.len(),
            IN_THINK_PROHIBITION.len() + 2
        );
        // The layered builders inherit the behavior.
        assert!(!build_tools_prompt(&[], false).contains("Tool calls are not allowed inside"));
        assert!(build_tools_prompt(&[], true).contains(IN_THINK_PROHIBITION));
        assert!(
            !build_system_prompt_reminder(&[], false).contains("Tool calls are not allowed inside")
        );
        assert!(!build_system_prompt("", &[], false).contains("Tool calls are not allowed inside"));
    }

    #[test]
    fn reminder_policy_thresholds() {
        let mut r = SystemPromptReminder::new();
        // First call only records the position.
        assert!(!r.should_remind(1000));
        // Below threshold: no reminder.
        assert!(!r.should_remind(1000 + SYSTEM_PROMPT_REMINDER_TOKENS - 1));
        // At threshold: reminder fires and position resets.
        assert!(r.should_remind(1000 + SYSTEM_PROMPT_REMINDER_TOKENS));
        // Immediately after, no reminder again.
        assert!(!r.should_remind(1000 + SYSTEM_PROMPT_REMINDER_TOKENS + 10));
    }

    #[test]
    fn datetime_line_shape() {
        let now = UNIX_EPOCH + Duration::from_secs(1_752_800_000);
        let line = datetime_context_line(now);
        assert!(line.starts_with("Current local date and time at session start: "));
        assert!(line.ends_with("Use this only when date or time matters."));
        // Local date portion: YYYY-MM-DD HH:MM:SS.
        let ts = line
            .strip_prefix("Current local date and time at session start: ")
            .unwrap();
        let bytes = ts.as_bytes();
        assert_eq!(&bytes[4..5], b"-");
        assert_eq!(&bytes[7..8], b"-");
        assert_eq!(&bytes[10..11], b" ");
        assert_eq!(&bytes[13..14], b":");
        assert_eq!(&bytes[16..17], b":");
    }
}
