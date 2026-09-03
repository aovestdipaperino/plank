// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Agent tool execution: argument parsing, shared context, and dispatch.
//!
//! Port of the "Tool Argument Parsing And File Tool Helpers" and "Tool
//! Dispatch" sections of `ds4_agent.c`. Tool calls arrive as parsed
//! [`crate::dsml::ToolCall`] values; each tool returns the exact text the C
//! agent would feed back to the model as the tool-role result, including the
//! `Tool error: ...` convention for failures. The browser web tools
//! (`google_search`, `visit_page`) live in [`web`].

pub mod ask;
pub mod bash;
pub mod diff;
pub mod edit;
pub mod files;
pub mod mcp;
pub mod mcp_advert;
pub mod web;
pub mod worktree;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::dsml::ToolCall;

/// Default timeout for bash commands, in seconds.
const BASH_DEFAULT_TIMEOUT_SEC: u64 = 3600;

/// Result of executing one tool call.
///
/// `output` is the model-visible observation text. `is_error` mirrors the C
/// convention: failures are plain text starting with `Tool error:`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    /// Model-visible observation text.
    pub output: String,
    /// True when the observation reports a tool failure.
    pub is_error: bool,
}

impl ToolResult {
    /// Wraps raw observation text, deriving `is_error` from the C convention.
    #[must_use]
    pub fn from_output(output: String) -> Self {
        let is_error = output.starts_with("Tool error:");
        Self { output, is_error }
    }
}

/// State for the `more` continuation tool: where the next read resumes.
#[derive(Debug, Clone)]
pub struct MoreState {
    /// Path of the file the previous truncated read came from, as the model
    /// spelled it. Shown in the continuation's output.
    pub path: String,
    /// Path the bytes are actually read from. Equal to `path` except for
    /// converted documents, where it points into `~/.plank/doc-cache`.
    pub read_path: String,
    /// 1-based line the next chunk starts at.
    pub next_line: usize,
    /// True when the previous read was in raw (bare) mode.
    pub bare: bool,
}

/// Approval hook for the web tools, mirroring `agent_web_confirm`.
///
/// Receives the approval prompt and returns true to allow web access.
pub type WebConfirmFn = Box<dyn FnMut(&str) -> bool + Send>;

/// Sink for agent system status lines published *while* a tool runs, mirroring
/// `agent_publish_system_status`.
///
/// Distinct from the drained-after-dispatch vectors ([`ToolContext::hook_warnings`],
/// [`ToolContext::task_completions`]): a "Searching Google for ..." notice is
/// only useful before the search returns, so it goes straight to the front end.
/// Receives the bare message; the front end styles it.
pub type StatusSinkFn = Box<dyn Fn(&str) + Send>;

/// Mutable state shared by all tools of one agent worker.
pub struct ToolContext {
    /// Working directory relative paths are resolved against.
    pub cwd: PathBuf,
    /// Continuation state for the `more` tool, if a read was truncated.
    pub more: Option<MoreState>,
    /// The last spilled tool result, for the `more` tool to continue reading
    /// it by spill id. Set by the dispatch-level spill policy.
    pub spill: Option<crate::spill::Spilled>,
    /// Session id, used to scope spill storage under `~/.plank/spill/<id>/`.
    pub session_id: String,
    /// The live transcript, for the `recall` tool to search the current
    /// session's pre-compaction portion alongside the saved-session index.
    /// Populated by the turn loop before dispatch; empty in tests.
    pub current_transcript: Vec<crate::session::Message>,
    /// Table of live and finished asynchronous bash jobs.
    pub bash: bash::BashJobs,
    /// Per-session web tool state (sticky approval flag).
    pub web: web::WebState,
    /// Web access approval hook; `None` auto-denies like non-interactive C.
    pub web_confirm: Option<WebConfirmFn>,
    /// Front end for system status lines emitted during a dispatch; `None`
    /// swallows them (non-interactive runs, tests).
    pub status_sink: Option<StatusSinkFn>,
    /// Live MCP servers started from the `.mcp.json` config, if any.
    pub mcp: Vec<mcp::McpServer>,
    /// Resolved paths of recent successful `read` calls, oldest first, for
    /// post-compaction re-injection (`compact::build_reinjection`).
    pub recent_reads: Vec<PathBuf>,
    /// Command hooks (PreToolUse/PostToolUse/Stop) from hooks.json configs.
    pub hooks: crate::hooks::Hooks,
    /// Plugins activated for this session, contributing skills, agents and
    /// templates alongside the local ones.
    pub plugins: crate::plugins::PluginSet,
    /// WASM components admitted this session, and the runtime they run in.
    /// Lives beside `plugins` because a component *is* a plugin contribution.
    pub wasm: crate::wasmreg::Session,
    /// Seatbelt sandbox policy for model-initiated bash commands.
    pub sandbox: crate::sandbox::Sandbox,
    /// User-only warnings from non-blocking hook failures, drained by the UI
    /// after each dispatch.
    pub hook_warnings: Vec<String>,
    /// Skills the model may invoke via the `skill` tool (issue #36).
    pub skills: Vec<crate::skills::Skill>,
    /// Skill invocations so far this turn; the turn driver resets it to 0 at
    /// the start of each turn. Bounds runaway skill-invokes-skill recursion.
    pub skill_invocations: usize,
    /// Current sub-agent nesting depth (issue #50). The turn driver increments
    /// this around a delegated `agent` tool run and the `agent` tool refuses
    /// once it reaches [`SUBAGENT_DEPTH_CAP`], bounding agent-invokes-agent
    /// recursion the same way [`SKILL_DEPTH_CAP`] bounds skills.
    pub subagent_depth: usize,
    /// The worktree this session has moved into, if any. In-memory only: a
    /// resumed session always starts where it was launched, never inside a
    /// worktree a previous run happened to enter.
    pub worktree: Option<crate::worktree::WorktreeSession>,
    /// True while a read-only plan-mode gate is active (issue #50). Mutating
    /// tools refuse until `ExitPlanMode` clears it.
    pub plan_mode: bool,
    /// Set by a tool hook's `{"continue": false}` response envelope; the turn
    /// driver halts the turn after the dispatch that produced it.
    pub hook_stop: Option<String>,
    /// Live model-visible task list (issue #35). The authoritative working copy
    /// during a turn; the driver mirrors it onto the session (which serializes
    /// it) so it survives compaction, `/resume`, and checkpoint rollback.
    pub tasks: crate::tasks::TaskList,
    /// Subjects of tasks the `task` tool just marked completed, drained by the
    /// UI after each dispatch to write the single dim completion log line.
    pub task_completions: Vec<String>,
    /// Diff previews from `edit`/`write` calls this dispatch, drained by the UI
    /// to render a git-style change card. Empty when nothing changed a file.
    pub edit_previews: Vec<diff::EditPreview>,
    /// Absolute path of the file a tool most recently wrote, drained by the UI
    /// to aim a bare `/open`. Distinct from [`edit_previews`](Self::edit_previews):
    /// creating a new file deliberately produces no diff card (the streaming
    /// preview already showed it) but is still the file the user wants to open.
    pub last_written: Option<PathBuf>,
    /// Front end that presents `ask` questions (issue #34); `None` in
    /// non-interactive mode, where `ask` fast-fails instead of blocking.
    pub asker: Option<Box<dyn ask::Asker>>,
    /// UI-thread handle to the `ask` rendezvous, set only under the TUI (the
    /// worker's [`asker`](Self::asker) parks requests here for the event loop to
    /// render). `None` for the plain REPL (stdin asker) and non-interactive mode.
    pub ask_bridge: Option<ask::AskBridge>,
    /// Live browser session for the web tools, created lazily on first web use
    /// and reused across turns (like the C agent keeping Chrome alive). Only on
    /// `ds4_engine` builds; the curl path needs no handle.
    #[cfg(ds4_engine)]
    pub web_browser: Option<crate::ds4web::WebBrowser>,
}

/// Most `skill` invocations allowed within one turn before the tool refuses,
/// bounding a skill whose text tells the model to invoke another skill.
pub const SKILL_DEPTH_CAP: usize = 8;

/// Maximum sub-agent nesting depth (issue #50). A depth of 1 means a top-level
/// turn may delegate to a sub-agent, but that sub-agent may not delegate again;
/// this bounds runaway agent-invokes-agent recursion.
pub const SUBAGENT_DEPTH_CAP: usize = 1;

/// Tools that mutate the workspace and are therefore refused while plan mode is
/// active (issue #50). Read-only tools stay available so the model can research
/// before proposing a plan. `bash` is included because it can run arbitrary
/// side-effecting commands.
const PLAN_MODE_BLOCKED_TOOLS: &[&str] = &["write", "edit", "bash", "EnterWorktree"];

/// True when `name` is a workspace-mutating tool blocked under plan mode.
#[must_use]
fn is_plan_mode_blocked(name: &str) -> bool {
    PLAN_MODE_BLOCKED_TOOLS.contains(&name)
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("cwd", &self.cwd)
            .field("more", &self.more)
            .field("bash", &self.bash)
            .field("web", &self.web)
            .field("web_confirm", &self.web_confirm.as_ref().map(|_| "<fn>"))
            .field("mcp", &self.mcp)
            .field("recent_reads", &self.recent_reads)
            .field("hooks", &self.hooks)
            .field("sandbox", &self.sandbox)
            .finish_non_exhaustive()
    }
}

impl ToolContext {
    /// Creates a context rooted at the given working directory.
    #[must_use]
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            more: None,
            spill: None,
            session_id: String::new(),
            current_transcript: Vec::new(),
            bash: bash::BashJobs::default(),
            web: web::WebState::default(),
            web_confirm: None,
            status_sink: None,
            mcp: Vec::new(),
            recent_reads: Vec::new(),
            hooks: crate::hooks::Hooks::default(),
            plugins: crate::plugins::PluginSet::default(),
            wasm: crate::wasmreg::Session::default(),
            sandbox: crate::sandbox::Sandbox::default(),
            hook_warnings: Vec::new(),
            skills: Vec::new(),
            skill_invocations: 0,
            subagent_depth: 0,
            worktree: None,
            plan_mode: false,
            hook_stop: None,
            tasks: crate::tasks::TaskList::new(),
            task_completions: Vec::new(),
            edit_previews: Vec::new(),
            last_written: None,
            asker: None,
            ask_bridge: None,
            #[cfg(ds4_engine)]
            web_browser: None,
        }
    }

    /// Publishes a system status line to the front end, if one is listening.
    /// Mirrors `agent_publishf_system_status`.
    pub fn publish_status(&self, msg: &str) {
        if let Some(sink) = self.status_sink.as_ref() {
            sink(msg);
        }
    }

    /// Records a successful file read for post-compaction re-injection:
    /// moves `path` to the newest slot and bounds the list.
    pub fn note_read(&mut self, path: PathBuf) {
        const RECENT_READS_CAP: usize = 16;
        self.recent_reads.retain(|p| *p != path);
        self.recent_reads.push(path);
        if self.recent_reads.len() > RECENT_READS_CAP {
            self.recent_reads
                .drain(..self.recent_reads.len() - RECENT_READS_CAP);
        }
    }

    /// Resolves a tool-provided path against the context working directory.
    #[must_use]
    pub fn resolve(&self, path: impl AsRef<Path>) -> PathBuf {
        let path = path.as_ref();
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }
}

/// Executes one parsed tool call and returns the model-visible result.
///
/// Mirrors `agent_execute_tool_call`: the same tool names the C agent
/// registers, minus the browser web tools.
#[allow(clippy::too_many_lines)]
pub fn dispatch(call: &ToolCall, ctx: &mut ToolContext) -> ToolResult {
    if call.name.is_empty() {
        return ToolResult::from_output("Tool error: missing tool name\n".to_string());
    }
    // Argument values feed argument matchers like `bash(git *)`.
    let arg_values: Vec<&str> = call.args.iter().map(|a| a.value.as_str()).collect();
    // PreToolUse hooks: exit 2 blocks the tool, its stderr becomes the
    // model-visible tool error.
    if !ctx.hooks.pre_tool_use.is_empty() {
        let input = crate::hooks::tool_event_input(
            "PreToolUse",
            &call.name,
            &mcp::args_to_json(call),
            None,
            &ctx.cwd,
        );
        let pre = crate::hooks::run_event_args(
            &ctx.hooks.pre_tool_use,
            &call.name,
            &arg_values,
            &input,
            &ctx.cwd,
        );
        ctx.hook_warnings.extend(pre.warnings);
        ctx.hook_warnings.extend(pre.system_messages);
        if ctx.hook_stop.is_none() {
            ctx.hook_stop = pre.stop_reason;
        }
        if let Some(msg) = pre.block {
            return ToolResult::from_output(format!(
                "Tool error: blocked by PreToolUse hook: {msg}\n"
            ));
        }
    }
    // pre_tool_use for WASM subscribers, after the shell hooks and before the
    // plan-mode gate: a component sees the same call a hook would have seen,
    // and a block reaches the model in the same shape.
    {
        let event = crate::wasmevents::Event::new(
            crate::wasmevents::EventKind::PreToolUse,
            vec![
                ("name", call.name.clone()),
                ("args", mcp::args_to_json(call)),
            ],
        );
        let wasm = &mut ctx.wasm;
        let out = wasm.registry.dispatch(&mut *wasm.host, &event);
        ctx.hook_warnings.extend(out.printed);
        ctx.hook_warnings.extend(out.warnings);
        if let Some((id, reason)) = out.blocked {
            return ToolResult::from_output(format!(
                "Tool error: blocked by wasm component {id}: {reason}\n"
            ));
        }
    }
    // Plan mode (issue #50): while the read-only gate is active, refuse any
    // workspace-mutating tool so the model researches and proposes before it
    // edits. The gate itself is entered/exited by dedicated tools below.
    if ctx.plan_mode && is_plan_mode_blocked(&call.name) {
        return ToolResult::from_output(format!(
            "Tool error: plan mode is active — {} is read-only until you call ExitPlanMode with your proposed plan and it is approved\n",
            call.name
        ));
    }
    // Dispatch-level wall-clock deadline (`tools.callTimeoutSec`): the outer
    // bound on a single tool call, off by default. Bash keeps its own
    // model-supplied timeout; this is the outer bound of the two. Measured
    // around the tool body only, so hooks still see the full output.
    let deadline = crate::settings::active().tools.call_timeout_sec;
    let start = std::time::Instant::now();
    let output = match call.name.as_str() {
        "EnterWorktree" => worktree::tool_enter_worktree(ctx, call),
        "ExitWorktree" => worktree::tool_exit_worktree(ctx, call),
        "EnterPlanMode" => tool_enter_plan_mode(ctx),
        "ExitPlanMode" => tool_exit_plan_mode(ctx, call),
        "read" => files::tool_read(ctx, call),
        "view_image" => tool_view_image(call),
        "more" => files::tool_more(ctx, call),
        "write" => files::tool_write(ctx, call),
        "list" => files::tool_list(ctx, call),
        "glob" => files::tool_glob(ctx, call),
        "edit" => edit::tool_edit(ctx, call),
        "search" => edit::tool_search(ctx, call),
        "bash" => bash::tool_bash(ctx, call),
        "bash_status" => bash::tool_bash_status_or_stop(ctx, call, false),
        "bash_stop" => bash::tool_bash_status_or_stop(ctx, call, true),
        "google_search" => web::tool_google_search(ctx, call),
        "visit_page" => web::tool_visit_page(ctx, call),
        "mcp_describe" => mcp::tool_mcp_describe(&ctx.mcp, call),
        "mcp_call" => mcp::tool_mcp_invoke(&mut ctx.mcp, call),
        "mcp_list_resources" => mcp::tool_mcp_list_resources(&ctx.mcp, call),
        "mcp_read_resource" => mcp::tool_mcp_read_resource(&mut ctx.mcp, call),
        "skill" => crate::skills::tool_skill(
            &ctx.skills,
            &mut ctx.skill_invocations,
            SKILL_DEPTH_CAP,
            call,
        ),
        "task" => crate::tasks::tool_task(&mut ctx.tasks, &mut ctx.task_completions, call),
        "ask" => ask::tool_ask(ctx.asker.as_mut(), call),
        "recall" => tool_recall(ctx, call),
        "run_code" => tool_run_code(ctx, call),
        name if name.starts_with("mcp__") => mcp::tool_mcp_call(&mut ctx.mcp, call),
        // A WASM component's tool. Checked before the unknown-tool fallthrough
        // and after every built-in, so a component can extend the table and
        // never shadow it.
        name if ctx.wasm.registry.tools().iter().any(|t| t.exposed == name) => {
            let args = mcp::args_to_json(call);
            let wasm = &mut ctx.wasm;
            match wasm.registry.run_tool(&mut *wasm.host, name, &args) {
                Ok(output) => output,
                Err(e) => format!("Tool error: {e}\n"),
            }
        }
        other => format!("Tool error: unknown tool: {other}\n"),
    };
    // PostToolUse hooks: exit 2 appends stderr to the model's observation.
    let mut output = output;
    if !ctx.hooks.post_tool_use.is_empty() {
        let input = crate::hooks::tool_event_input(
            "PostToolUse",
            &call.name,
            &mcp::args_to_json(call),
            Some(&output),
            &ctx.cwd,
        );
        let post = crate::hooks::run_event_args(
            &ctx.hooks.post_tool_use,
            &call.name,
            &arg_values,
            &input,
            &ctx.cwd,
        );
        ctx.hook_warnings.extend(post.warnings);
        ctx.hook_warnings.extend(post.system_messages);
        if ctx.hook_stop.is_none() {
            ctx.hook_stop = post.stop_reason;
        }
        if let Some(msg) = post.block {
            if !output.ends_with('\n') {
                output.push('\n');
            }
            let _ = writeln!(output, "[PostToolUse hook] {msg}");
        }
    }
    // post_tool_use for WASM subscribers. Transform: a replacement becomes the
    // observation the model sees, which is the whole point of the event — a
    // redactor or a summarizer has nowhere else to stand.
    {
        let event = crate::wasmevents::Event::new(
            crate::wasmevents::EventKind::PostToolUse,
            vec![
                ("name", call.name.clone()),
                ("args", mcp::args_to_json(call)),
                ("output", output.clone()),
            ],
        );
        let wasm = &mut ctx.wasm;
        let out = wasm.registry.dispatch(&mut *wasm.host, &event);
        ctx.hook_warnings.extend(out.printed);
        ctx.hook_warnings.extend(out.warnings);
        if let Some(replaced) = out.replaced {
            output = replaced;
        }
        if let Some((id, reason)) = out.blocked {
            if !output.ends_with('\n') {
                output.push('\n');
            }
            let _ = writeln!(output, "[wasm {id}] {reason}");
        }
    }
    // PostToolUseFailure hooks: fire only when the tool failed (the C
    // `Tool error:` convention); success never reaches here.
    if output.starts_with("Tool error:") && !ctx.hooks.post_tool_use_failure.is_empty() {
        fire_post_tool_failure(ctx, call, &arg_values, &mut output);
    }
    // Deadline notice: the tool exceeded `callTimeoutSec`. Post-hoc (the tool
    // already ran) but actionable — the model sees it exceeded the budget.
    if deadline > 0 && start.elapsed().as_secs() >= deadline {
        if !output.ends_with('\n') {
            output.push('\n');
        }
        let _ = writeln!(
            output,
            "[deadline] {call} exceeded tools.callTimeoutSec={deadline}s",
            call = call.name
        );
    }
    // Output spill (M4): applied AFTER the PostToolUse hooks, so a hook sees
    // the full output and only the model sees the preview. The full payload is
    // written to `~/.plank/spill/<session-id>/` and the inline result becomes a
    // bounded preview plus a locator the `more` tool can continue.
    let s = crate::settings::active().tools.clone();
    let policy = crate::spill::SpillPolicy {
        max_bytes: s.spill_max_bytes,
        preview_bytes: s.spill_preview_bytes,
    };
    let (preview, spilled) = crate::spill::apply(&policy, &ctx.session_id, &call.name, output);
    ctx.spill = spilled;
    ToolResult::from_output(preview)
}

/// Fires the `PostToolUseFailure` hooks and appends any exit-2 block message to
/// `output`, mirroring the `PostToolUse` block framing. Split out of `dispatch`
/// to keep it under the function-length lint.
fn fire_post_tool_failure(
    ctx: &mut ToolContext,
    call: &ToolCall,
    arg_values: &[&str],
    output: &mut String,
) {
    // plank has no per-tool interrupt tracking in the dispatch path, so the
    // `is_interrupt` flag the reference carries is always false here; it is
    // still emitted so hooks can rely on the field being present.
    let base = crate::hooks::tool_event_input(
        "PostToolUseFailure",
        &call.name,
        &mcp::args_to_json(call),
        Some(output),
        &ctx.cwd,
    );
    let input = format!("{},\"is_interrupt\":false}}", &base[..base.len() - 1]);
    let fail = crate::hooks::run_event_args(
        &ctx.hooks.post_tool_use_failure,
        &call.name,
        arg_values,
        &input,
        &ctx.cwd,
    );
    ctx.hook_warnings.extend(fail.warnings);
    ctx.hook_warnings.extend(fail.system_messages);
    if ctx.hook_stop.is_none() {
        ctx.hook_stop = fail.stop_reason;
    }
    if let Some(msg) = fail.block {
        if !output.ends_with('\n') {
            output.push('\n');
        }
        let _ = writeln!(output, "[PostToolUseFailure hook] {msg}");
    }
}

/// Handles `EnterPlanMode`: turns on the read-only plan gate (issue #50).
///
/// Idempotent — entering plan mode when already in it just reaffirms the gate.
fn tool_enter_plan_mode(ctx: &mut ToolContext) -> String {
    ctx.plan_mode = true;
    "Plan mode is on. You are now read-only: research with read/list/glob/search \
     and the web/MCP tools, but do not modify the workspace. When you have a \
     concrete plan, call ExitPlanMode with the plan in its 'plan' argument to \
     ask the user for approval before making changes.\n"
        .to_string()
}

/// Handles `ExitPlanMode`: presents the proposed plan for approval and, when
/// approved, lifts the read-only gate (issue #50).
///
/// With an interactive [`ask::Asker`] the user approves or rejects; a rejection
/// keeps the gate on. Without one (non-interactive / headless) the plan is
/// auto-approved so scripted runs are not wedged, mirroring the `ask` tool's
/// non-interactive fast-path.
fn tool_exit_plan_mode(ctx: &mut ToolContext, call: &ToolCall) -> String {
    if !ctx.plan_mode {
        return "Tool error: ExitPlanMode called but plan mode is not active\n".to_string();
    }
    let plan = call.arg_value("plan").unwrap_or("").trim();
    if plan.is_empty() {
        return "Tool error: ExitPlanMode requires a non-empty 'plan' describing what you intend to do\n"
            .to_string();
    }
    let Some(asker) = ctx.asker.as_mut() else {
        // No interactive user to approve; lift the gate and proceed.
        ctx.plan_mode = false;
        return "No interactive user is available to approve the plan \
                (non-interactive mode); plan mode lifted, proceed.\n"
            .to_string();
    };
    let req = ask::AskRequest {
        question: format!("Approve this plan?\n\n{plan}"),
        header: "Plan".to_string(),
        options: vec![
            ask::AskOption {
                label: "Approve".to_string(),
                description: "Proceed with the plan and allow edits".to_string(),
            },
            ask::AskOption {
                label: "Keep planning".to_string(),
                description: "Stay read-only and refine the plan".to_string(),
            },
        ],
        multi: false,
    };
    match asker.ask(req) {
        ask::AskOutcome::Answered(labels) if labels.iter().any(|l| l == "Approve") => {
            ctx.plan_mode = false;
            "Plan approved. Plan mode is off; you may now modify the workspace to carry it out.\n"
                .to_string()
        }
        _ => {
            "Plan not approved; plan mode stays on. Refine the plan and call ExitPlanMode again.\n"
                .to_string()
        }
    }
}

/// Executes all calls of one DSML block, framing each result with its label.
///
/// Mirrors `agent_execute_tool_calls`, so the model can associate
/// observations with calls.
pub fn dispatch_all(calls: &[ToolCall], ctx: &mut ToolContext) -> String {
    if calls.is_empty() {
        return "Tool error: empty tool call block\n".to_string();
    }
    // Diff previews accumulate per dispatch; clear any a prior caller left
    // undrained so cards never leak between turns.
    ctx.edit_previews.clear();
    let mut all = String::new();
    for (i, call) in calls.iter().enumerate() {
        let res = dispatch(call, ctx);
        let name = if call.name.is_empty() {
            "unknown"
        } else {
            call.name.as_str()
        };
        let _ = writeln!(all, "Tool result {} ({}):", i + 1, name);
        all.push_str(&res.output);
        if !res.output.is_empty() && !res.output.ends_with('\n') {
            all.push('\n');
        }
    }
    all
}

/// Parses a bash timeout in seconds, clamped to `1..=86400`.
///
/// Mirrors `agent_parse_timeout`: missing or malformed values yield 3600.
#[must_use]
pub fn parse_timeout(s: Option<&str>) -> u64 {
    let Some(s) = s else {
        return BASH_DEFAULT_TIMEOUT_SEC;
    };
    let s = s.trim();
    // strtod stops at the first non-numeric byte; approximate by trying
    // progressively shorter prefixes of the leading float-looking run.
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
        .unwrap_or(s.len());
    let Ok(v) = s[..end].parse::<f64>() else {
        return BASH_DEFAULT_TIMEOUT_SEC;
    };
    if v <= 0.0 || !v.is_finite() {
        return BASH_DEFAULT_TIMEOUT_SEC;
    }
    let v = v.clamp(1.0, 24.0 * 3600.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        v as u64
    }
}

/// Parses an integer argument with a default and clamping range.
///
/// Mirrors `agent_parse_int_default`: trailing whitespace is tolerated, any
/// other trailing text falls back to the default.
#[must_use]
pub fn parse_int_default(s: Option<&str>, def: i64, min: i64, max: i64) -> i64 {
    let Some(s) = s else { return def };
    let t = s.trim();
    if t.is_empty() {
        return def;
    }
    match t.parse::<i64>() {
        Ok(v) => v.clamp(min, max),
        Err(_) => def,
    }
}

/// Parses a boolean argument, accepting true/yes/1 and false/no/0.
#[must_use]
pub fn parse_bool_default(s: Option<&str>, def: bool) -> bool {
    let Some(s) = s else { return def };
    if s.is_empty() {
        return def;
    }
    if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes") || s == "1" {
        return true;
    }
    if s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("no") || s == "0" {
        return false;
    }
    def
}

/// Implements the `recall` tool (M8): searches prior sessions and the current
/// one's pre-compaction portion, scoped to the current project. Off by default
/// (`tools.recall`); when off, the tool is not advertised and dispatches as
/// unknown. Results are bounded through the M4 spill policy like any other
/// tool result.
/// `view_image` — parity stub for the C's visual-observation tool.
///
/// The C advertises `view_image` unconditionally in the frozen tool table
/// (`agent_tools_prompt_intro`, between `read` and `more`), but only serves it
/// when the engine was given a vision encoder via `ds4-agent --vision FILE`.
/// plank passes a null `vision_path`, so it is always in the second case and
/// returns the C's own refusal byte-for-byte. This is parity behavior, not a
/// placeholder: an agent run without `--vision` gets exactly this string.
///
/// Wiring the real path means giving the engine a `vision_path`, encoding the
/// image with `ds4_engine_vision_encode_file`, and appending it through
/// `ds4_chat_append_multimodal_message` — which also drags in the rule that
/// image-conditioned KV must never reach the text-keyed disk cache.
fn tool_view_image(call: &ToolCall) -> String {
    let path = call.arg_value("path").unwrap_or("").trim();
    if path.is_empty() {
        return "Tool error: view_image requires path\n".to_string();
    }
    "Tool error: view_image requires ds4-agent --vision FILE\n".to_string()
}

fn tool_recall(ctx: &mut ToolContext, call: &ToolCall) -> String {
    use std::fmt::Write as _;
    if !crate::settings::active().tools.recall {
        return "Tool error: unknown tool: recall\n".to_string();
    }
    let query = call.arg_value("query").unwrap_or("").trim().to_string();
    if query.is_empty() {
        return "Tool error: recall requires a non-empty 'query'\n".to_string();
    }
    let mut out = String::new();
    let project = crate::session::project_key(&ctx.cwd);
    // Saved sessions, scoped to the current project. Best-effort: an index
    // build failure (e.g. an ambiguous session prefix) skips the saved-session
    // search but never blocks the current-transcript search below.
    let hits = match crate::session::SessionStore::open(crate::session::SessionStore::default_dir())
    {
        Ok(store) => match crate::sessionindex::build(&store, &crate::sessionindex::index_dir()) {
            Ok(_) => crate::sessionindex::search(
                &query,
                Some(&project),
                false,
                &crate::sessionindex::index_dir(),
            ),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    // The current session's pre-compaction portion: search the live transcript
    // directly (it is not yet in the index).
    let mut current: Vec<(String, String)> = Vec::new();
    for m in &ctx.current_transcript {
        if let Some(pos) = m.text.find(&query) {
            let start = pos.saturating_sub(40);
            let end = (pos + query.len() + 40).min(m.text.len());
            let mut snippet = String::new();
            if start > 0 {
                snippet.push('…');
            }
            snippet.push_str(&m.text[start..end]);
            if end < m.text.len() {
                snippet.push('…');
            }
            current.push(("current session".to_string(), snippet));
        }
    }
    if hits.is_empty() && current.is_empty() {
        let _ = writeln!(out, "No sessions match {query:?}.");
        return out;
    }
    for hit in hits {
        let _ = writeln!(out, "{} — {} — {}", hit.session_id, hit.title, hit.snippet);
    }
    for (label, snippet) in current {
        let _ = writeln!(out, "{label} — {snippet}");
    }
    // Bound the result through the M4 spill policy.
    let policy = crate::spill::SpillPolicy {
        max_bytes: crate::settings::active().tools.spill_max_bytes,
        preview_bytes: crate::settings::active().tools.spill_preview_bytes,
    };
    let (preview, spilled) = crate::spill::apply(&policy, &ctx.session_id, "recall", out);
    ctx.spill = spilled;
    preview
}

/// Implements the `run_code` tool (M10): a small script of named operations
/// (read/glob/edit/bash), one per line, each dispatched through the existing
/// tool path so the consent and sandbox checks apply — a binding that
/// shortcuts them would be a hole straight through every guard. Off by default
/// (`tools.runCode`); when off, the tool dispatches as unknown. Output is
/// bounded through the M4 spill policy.
fn tool_run_code(ctx: &mut ToolContext, call: &ToolCall) -> String {
    use std::fmt::Write as _;
    if !crate::settings::active().tools.run_code {
        return "Tool error: unknown tool: run_code\n".to_string();
    }
    let script = call.arg_value("script").unwrap_or("").trim();
    if script.is_empty() {
        return "Tool error: run_code requires a non-empty 'script'\n".to_string();
    }
    let mut out = String::new();
    for (i, line) in script.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (op, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        let (name, args): (&str, Vec<(&str, &str)>) = match op {
            "read" => ("read", vec![("path", rest)]),
            "glob" => ("glob", vec![("pattern", rest)]),
            "edit" => {
                let mut parts = rest.splitn(3, char::is_whitespace);
                let path = parts.next().unwrap_or("");
                let old = parts.next().unwrap_or("");
                let new = parts.next().unwrap_or("");
                ("edit", vec![("path", path), ("old", old), ("new", new)])
            }
            "bash" => ("bash", vec![("command", rest)]),
            _ => {
                let _ = writeln!(out, "Step {}: unknown operation {op:?}", i + 1);
                continue;
            }
        };
        let tool_call = ToolCall {
            name: name.to_string(),
            args: args
                .iter()
                .map(|(n, v)| crate::dsml::ToolArg {
                    name: (*n).to_string(),
                    value: (*v).to_string(),
                    is_string: true,
                })
                .collect(),
        };
        let res = dispatch(&tool_call, ctx);
        let _ = writeln!(out, "Step {} ({name}):\n{}", i + 1, res.output);
    }
    // Bound through the M4 spill policy like any other tool result.
    let policy = crate::spill::SpillPolicy {
        max_bytes: crate::settings::active().tools.spill_max_bytes,
        preview_bytes: crate::settings::active().tools.spill_preview_bytes,
    };
    let (preview, spilled) = crate::spill::apply(&policy, &ctx.session_id, "run_code", out);
    ctx.spill = spilled;
    preview
}

#[cfg(test)]
pub(crate) fn test_call(name: &str, args: &[(&str, &str)]) -> ToolCall {
    ToolCall {
        name: name.to_string(),
        args: args
            .iter()
            .map(|(n, v)| crate::dsml::ToolArg {
                name: (*n).to_string(),
                value: (*v).to_string(),
                is_string: true,
            })
            .collect(),
    }
}

#[cfg(test)]
pub(crate) fn test_ctx() -> (ToolContext, PathBuf) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "plank_tools_test_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    (ToolContext::new(&dir), dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_helpers_defaults() {
        assert_eq!(parse_timeout(None), 3600);
        assert_eq!(parse_timeout(Some("0")), 3600);
        assert_eq!(parse_timeout(Some("0.5")), 1);
        assert_eq!(parse_timeout(Some("999999")), 86400);
        assert_eq!(parse_int_default(Some("7"), 1, 1, 5), 5);
        assert_eq!(parse_int_default(Some("junk"), 9, 0, 100), 9);
        assert!(parse_bool_default(Some("YES"), false));
        assert!(!parse_bool_default(Some("0"), true));
        assert!(parse_bool_default(Some("maybe"), true));
    }

    #[test]
    fn dispatch_unknown_tool_errors() {
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(&test_call("frobnicate", &[]), &mut ctx);
        assert!(res.is_error);
        assert_eq!(res.output, "Tool error: unknown tool: frobnicate\n");
        std::fs::remove_dir_all(dir).ok();
    }

    /// `view_image` is advertised in the frozen prompt, so dispatch must route
    /// it rather than answer "unknown tool" — and the refusal it gives without
    /// a vision encoder is the C's string byte-for-byte, since that is exactly
    /// what `ds4-agent` prints when run without `--vision`.
    #[test]
    fn view_image_is_routable_and_refuses_with_the_c_string() {
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(
            &test_call("view_image", &[("path", "/tmp/x.png")]),
            &mut ctx,
        );
        assert_eq!(
            res.output,
            "Tool error: view_image requires ds4-agent --vision FILE\n"
        );
        // A missing path is refused before the encoder is even considered.
        let res = dispatch(&test_call("view_image", &[]), &mut ctx);
        assert_eq!(res.output, "Tool error: view_image requires path\n");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn plan_mode_gates_mutating_tools_and_exits_on_approval() {
        let (mut ctx, dir) = test_ctx();
        // Entering plan mode turns on the read-only gate.
        let res = dispatch(&test_call("EnterPlanMode", &[]), &mut ctx);
        assert!(!res.is_error);
        assert!(ctx.plan_mode);
        // A mutating tool is now refused with a plan-mode error.
        let res = dispatch(
            &test_call("write", &[("path", "x.txt"), ("content", "hi")]),
            &mut ctx,
        );
        assert!(res.is_error);
        assert!(res.output.contains("plan mode is active"));
        // A read-only tool still works (list of the scratch dir).
        let res = dispatch(&test_call("list", &[]), &mut ctx);
        assert!(!res.is_error, "read-only tool blocked: {}", res.output);
        // ExitPlanMode requires a plan.
        let res = dispatch(&test_call("ExitPlanMode", &[]), &mut ctx);
        assert!(res.is_error);
        assert!(ctx.plan_mode, "gate must stay on without a plan");
        // With a plan and no asker (non-interactive), it auto-approves.
        let res = dispatch(
            &test_call("ExitPlanMode", &[("plan", "do the thing")]),
            &mut ctx,
        );
        assert!(!res.is_error);
        assert!(!ctx.plan_mode, "gate must lift after approval");
        // Now the mutating tool is allowed again.
        let res = dispatch(
            &test_call("write", &[("path", "x.txt"), ("content", "hi")]),
            &mut ctx,
        );
        assert!(!res.is_error, "write still blocked: {}", res.output);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn exit_plan_mode_errors_when_not_planning() {
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(&test_call("ExitPlanMode", &[("plan", "p")]), &mut ctx);
        assert!(res.is_error);
        assert!(res.output.contains("plan mode is not active"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dispatch_all_frames_results() {
        let (mut ctx, dir) = test_ctx();
        let out = dispatch_all(&[test_call("nope", &[])], &mut ctx);
        assert!(out.starts_with("Tool result 1 (nope):\n"));
        assert_eq!(
            dispatch_all(&[], &mut ctx),
            "Tool error: empty tool call block\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn deadline_notice_is_off_by_default() {
        // `tools.callTimeoutSec` defaults to 0 (off), so parity is untouched
        // until a user opts in: no deadline notice on an ordinary dispatch.
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(&test_call("nope", &[]), &mut ctx);
        assert!(!res.output.contains("[deadline]"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn run_code_executes_a_script_by_default_and_can_be_disabled() {
        // `tools.runCode` defaults to true. Turning it off makes the call
        // dispatch as unknown, matching the prompt where it is unadvertised.
        let mut off = crate::settings::Settings::default();
        off.tools.run_code = false;
        crate::settings::install_for_test(off);
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(
            &test_call("run_code", &[("script", "read x.txt")]),
            &mut ctx,
        );
        assert!(res.is_error);
        assert!(res.output.contains("unknown tool: run_code"));
        // At the default, a script of named operations executes through the
        // existing dispatch path (so consent/sandbox checks apply) and collects
        // outputs.
        crate::settings::install_for_test(crate::settings::Settings::default());
        let (mut ctx, dir2) = test_ctx();
        let f = dir2.join("x.txt");
        std::fs::write(&f, "hello").expect("write");
        let res = dispatch(
            &test_call("run_code", &[("script", "read x.txt")]),
            &mut ctx,
        );
        assert!(
            !res.is_error,
            "run_code dispatches when enabled: {}",
            res.output
        );
        assert!(res.output.contains("hello"), "{}", res.output);
        // An empty script is rejected: a documented criterion the model will
        // not produce on its own.
        let empty = dispatch(&test_call("run_code", &[("script", "   ")]), &mut ctx);
        assert_eq!(
            empty.output,
            "Tool error: run_code requires a non-empty 'script'\n"
        );
        // An unrecognised operation reports and continues rather than aborting
        // the rest of the script.
        let mixed = dispatch(
            &test_call(
                "run_code",
                &[("script", "frobnicate something\nread x.txt")],
            ),
            &mut ctx,
        );
        assert!(
            mixed
                .output
                .contains("Step 1: unknown operation \"frobnicate\""),
            "{}",
            mixed.output
        );
        assert!(
            mixed.output.contains("Step 2 (read):") && mixed.output.contains("hello"),
            "a bad step must not abort the script: {}",
            mixed.output
        );
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&dir2).ok();
    }

    /// The M10 security assertion: `run_code` binds operations *through*
    /// `dispatch`, so a `bash` step inside a script must hit exactly the same
    /// sandbox refusal a bare `bash` tool call does. If a step ever succeeded
    /// here, it would be a hole straight through every guard in `sandbox.rs`.
    /// Requires /usr/bin/sandbox-exec, so macOS only.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_bash_step_inside_run_code_is_sandboxed_like_a_bare_call() {
        crate::settings::install_for_test(crate::settings::Settings::default());

        // The scratch dir lives under temp_dir(), which the sandbox profile
        // always allows, so the escape target must sit outside both cwd and
        // temp — the same reasoning as `bash_sandbox_blocks_writes_outside_cwd`.
        let home = std::env::var("HOME").expect("HOME set");
        let outside =
            std::path::Path::new(&home).join(format!(".plank-runcode-test-{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        let escape = format!("echo escape > '{}/escape.txt'", outside.display());

        // Bare `bash` tool call: refused by the sandbox.
        let (mut ctx, dir) = test_ctx();
        ctx.sandbox.enabled = true;
        let bare = dispatch(&test_call("bash", &[("command", &escape)]), &mut ctx);
        assert!(
            bare.output.contains("[sandbox blocked:"),
            "bare call must be blocked: {}",
            bare.output
        );

        // The same command as a `run_code` step: identically refused.
        let (mut ctx2, dir2) = test_ctx();
        ctx2.sandbox.enabled = true;
        let script = format!("bash {escape}");
        let via_run_code = dispatch(&test_call("run_code", &[("script", &script)]), &mut ctx2);
        assert!(
            via_run_code.output.contains("[sandbox blocked:"),
            "run_code must not bypass the sandbox: {}",
            via_run_code.output
        );
        assert!(
            !via_run_code.output.contains("exit_status=0\n"),
            "the escaping step must not succeed: {}",
            via_run_code.output
        );
        assert!(
            !outside.join("escape.txt").exists(),
            "the write must not have landed"
        );

        std::fs::remove_dir_all(outside).ok();
        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(dir2).ok();
    }

    #[test]
    fn recall_dispatches_by_default_and_can_be_disabled() {
        // `tools.recall` defaults to true: the tool dispatches and searches the
        // current transcript's pre-compaction portion. Turning it off makes the
        // call dispatch as unknown, matching the prompt where it is unadvertised.
        let mut off = crate::settings::Settings::default();
        off.tools.recall = false;
        crate::settings::install_for_test(off);
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(&test_call("recall", &[("query", "needle")]), &mut ctx);
        assert!(res.is_error);
        assert!(res.output.contains("unknown tool: recall"));
        // Back at the default, it dispatches.
        crate::settings::install_for_test(crate::settings::Settings::default());
        let (mut ctx, dir2) = test_ctx();
        ctx.current_transcript = vec![crate::session::Message::user("a needle here")];
        let res = dispatch(&test_call("recall", &[("query", "needle")]), &mut ctx);
        assert!(
            !res.is_error,
            "recall dispatches when enabled: {}",
            res.output
        );
        assert!(res.output.contains("current session"), "{}", res.output);
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&dir2).ok();
    }

    #[test]
    fn recall_rejects_an_empty_query_and_reports_no_match() {
        // Both are documented M8 pass criteria that cannot be reached through
        // the model: the request itself quotes the query, so the current
        // transcript always self-matches and `No sessions match` never fires
        // in a live session.
        crate::settings::install_for_test(crate::settings::Settings::default());

        let (mut ctx, dir) = test_ctx();
        let res = dispatch(&test_call("recall", &[("query", "   ")]), &mut ctx);
        assert!(res.is_error, "a blank query is an error: {}", res.output);
        assert_eq!(
            res.output,
            "Tool error: recall requires a non-empty 'query'\n"
        );

        let (mut ctx, dir2) = test_ctx();
        ctx.current_transcript = vec![crate::session::Message::user("nothing relevant here")];
        let res = dispatch(
            &test_call("recall", &[("query", "qqzzxxnomatchtoken")]),
            &mut ctx,
        );
        assert!(
            res.output
                .contains("No sessions match \"qqzzxxnomatchtoken\"."),
            "no-match message: {}",
            res.output
        );
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&dir2).ok();
    }

    #[test]
    fn dispatch_spills_oversized_results_and_more_continues() {
        // A 5 MB-style oversized result (here a 1000-byte read under a 100-byte
        // cap) yields a bounded preview plus a locator, and the generalised
        // `more` continues the spill by id.
        let mut s = crate::settings::Settings::default();
        s.tools.spill_max_bytes = 100;
        s.tools.spill_preview_bytes = 50;
        crate::settings::install_for_test(s);
        let (mut ctx, dir) = test_ctx();
        // Dispatch spills through the real `~/.plank/spill`, so use a session
        // id unique to this test run: a fixed one collides with other tests and
        // leaves blobs behind that shift the next run's spill numbering.
        ctx.session_id = format!(
            "plank-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let big = dir.join("big.txt");
        std::fs::write(&big, "x".repeat(1000)).expect("write");
        let res = dispatch(&test_call("read", &[("path", "big.txt")]), &mut ctx);
        assert!(
            res.output.contains("[Output truncated at 50 bytes of"),
            "preview + locator: {}",
            res.output
        );
        assert!(ctx.spill.is_some(), "spill state set for `more`");
        let more = dispatch(&test_call("more", &[("count", "100")]), &mut ctx);
        assert!(
            more.output.contains("continue_offset="),
            "more continues the spill: {}",
            more.output
        );
        std::fs::remove_dir_all(crate::spill::spill_dir().join(&ctx.session_id)).ok();
        std::fs::remove_dir_all(dir).ok();
    }
}
