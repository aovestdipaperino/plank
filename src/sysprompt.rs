//! System prompt rendering: tool prompt text, reminders, datetime context.
//!
//! Port of the prompt-construction half of the "System Prompt Rendering And
//! Worker Output Queues" section of `ds4_agent.c` (roughly lines 703-1065).
//! The long tool-protocol strings are model-facing and replicated verbatim
//! from the C reference.

use std::time::{SystemTime, UNIX_EPOCH};

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
Read defaults to a bounded chunk: path alone returns the first 500 lines, not the whole file. \
If read says more lines are available, call more with count=<lines> to read the next chunk; \
more defaults to the next 500 lines. \
The read result also reports continue_offset=N, which is the next start_line if you need to jump manually. \
If the user explicitly asks you to read a complete file into context, call read with whole=true. \
A whole-file read may fail if the result would not fit the current context; then explain that and use chunks.\n\n";

/// Editing-instructions section of the tools prompt (verbatim from C).
const TOOLS_PROMPT_EDIT_LINE: &str = "## Editing files\n\n\
Use write for new files or deliberate whole-file replacement. Use edit with path, old, and new for changes. \
For edit, always put the edited file path as the first parameter. \
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
        "You are a coding agent running in a local workspace. Use the provided tools for local \
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
    if !user_system.is_empty() {
        out.push('\n');
        out.push_str(user_system);
    }
    out
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
        if !server.alive() {
            continue;
        }
        for tool in &server.tools {
            let parameters = serde_json::from_str::<serde_json::Value>(&tool.schema_json)
                .unwrap_or_else(|_| serde_json::json!({ "type": "object", "properties": {} }));
            specs.push(crate::engine::ToolSpec {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters,
            });
        }
    }
    // Resource tools, advertised only when a server actually publishes
    // resources — mirroring `append_resource_tool_schemas` for the text path.
    if mcp_servers
        .iter()
        .any(|s| s.alive() && !s.resources().is_empty())
    {
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

/// Pushes the provider-path [`ToolSpec`](crate::engine::ToolSpec)s for the
/// `agent` and plan-mode tools (issue #50). Mirrors the text-path schemas in
/// [`append_agent_and_plan_schemas`]; split out to keep
/// [`provider_tool_registry`] under the function-length lint.
fn push_agent_and_plan_specs(specs: &mut Vec<crate::engine::ToolSpec>) {
    specs.push(crate::engine::ToolSpec {
        name: "agent".to_string(),
        description: "Delegate a self-contained sub-task to a fresh sub-agent that works in its own scoped context and returns only a final report. Use this to keep your own context small: hand off open-ended research or a bounded multi-step chore, then continue from its report. 'task' is a complete, standalone instruction; 'name' optionally selects a configured agent persona. The sub-agent cannot ask you questions, so make 'task' fully specified.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "the complete, standalone task to delegate; include all needed context"},
                "name": {"type": "string", "description": "optional configured agent name to act as; omit for a general-purpose sub-agent"}
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
pub fn build_tools_prompt(mcp_servers: &[crate::tools::mcp::McpServer]) -> String {
    let mut out = build_tools_prompt_base();
    append_native_extra_schemas(&mut out);
    crate::tools::mcp::append_tool_schemas(&mut out, mcp_servers);
    crate::tools::mcp::append_resource_tool_schemas(&mut out, mcp_servers);
    crate::tools::mcp::append_server_instructions(&mut out, mcp_servers);
    out
}

/// The C-derived tools prompt with nothing appended.
///
/// This is what the parity suite locks byte-for-byte against `refs/ds4`: it is
/// exactly the three C string constants. Native plank tools (see
/// [`append_native_extra_schemas`]) and MCP tools are layered on top by
/// [`build_tools_prompt`], the same way MCP has always extended it, so the
/// trained-table parity guarantee stays intact for the base.
#[must_use]
pub fn build_tools_prompt_base() -> String {
    let mut out = String::with_capacity(
        TOOLS_PROMPT_INTRO.len() + TOOLS_PROMPT_EDIT_LINE.len() + TOOLS_PROMPT_AFTER_EDIT.len(),
    );
    out.push_str(TOOLS_PROMPT_INTRO);
    out.push_str(TOOLS_PROMPT_EDIT_LINE);
    out.push_str(TOOLS_PROMPT_AFTER_EDIT);
    out
}

/// Appends the schemas of native tools plank adds beyond the C-trained table.
///
/// These tools are **not** in the model's training-time tool table, which is
/// why issue #32 requires measuring that the model actually calls them. They
/// are appended here rather than baked into the C constants so the parity
/// suite keeps verifying the base against the reference.
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
         }\n\
         {\n\
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
         }\n\
         {\n\
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
}

/// Appends the `agent` (sub-agent delegation) and plan-mode tool schemas
/// (issue #50). Split from [`append_native_extra_schemas`] to keep each under
/// the function-length lint; both are native tools outside the C-trained table.
fn append_agent_and_plan_schemas(out: &mut String) {
    out.push_str(
        "{\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"agent\",\n\
         \x20   \"description\": \"Delegate a self-contained sub-task to a fresh sub-agent that works in its own scoped context and returns only a final report. Use this to keep your own context small: hand off open-ended research (locate where X is handled, summarize how Y works) or a bounded multi-step chore, then continue from its report. 'task' is a complete, standalone instruction; 'name' optionally selects a configured agent persona. The sub-agent cannot ask you questions, so make 'task' fully specified.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"task\": {\"type\": \"string\", \"description\": \"the complete, standalone task to delegate; include all needed context\"},\n\
         \x20       \"name\": {\"type\": \"string\", \"description\": \"optional configured agent name to act as; omit for a general-purpose sub-agent\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"task\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n\
         {\n\
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
pub fn build_system_prompt_reminder(mcp_servers: &[crate::tools::mcp::McpServer]) -> String {
    let mut out = String::from("\n\n[System prompt reminder follows.]\n");
    out.push_str(&build_tools_prompt(mcp_servers));
    out.push_str("[End system prompt reminder.]\n\n");
    out
}

/// Composes the initial system prompt: tools prompt plus optional user text.
///
/// Mirrors `agent_append_system_prompt`: the built-in tools prompt comes
/// first, and non-empty user `-sys` text is appended after a blank line. In
/// the C agent the two parts are tokenized differently (the built-in prompt
/// as rendered chat so DSML markers become control tokens, user text as plain
/// content); here both are returned as one composed string.
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
) -> String {
    let mut out = build_tools_prompt(mcp_servers);
    if !user_system.is_empty() {
        out.push_str("\n\n");
        out.push_str(user_system);
    }
    out
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
    fn tools_prompt_contains_verbatim_phrases() {
        let p = build_tools_prompt(&[]);
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
            build_tools_prompt_base().ends_with("unless explicitly asked otherwise by the user.\n")
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
        let a = build_system_prompt("user -sys text", &[]);
        let b = build_system_prompt("user -sys text", &[]);
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
        let r = build_system_prompt_reminder(&[]);
        assert!(r.starts_with("\n\n[System prompt reminder follows.]\n"));
        assert!(r.ends_with("[End system prompt reminder.]\n\n"));
        assert!(r.contains("## Tools"));
    }

    #[test]
    fn system_prompt_composition() {
        assert_eq!(build_system_prompt("", &[]), build_tools_prompt(&[]));
        let with_extra = build_system_prompt("Be terse.", &[]);
        assert!(with_extra.starts_with(&build_tools_prompt(&[])));
        assert!(with_extra.ends_with("\n\nBe terse."));
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
