//! Interactive REPL and headless front-ends over the agent turn loop.
//!
//! Port of the "Interactive Runtime Loop" section of `ds4_agent.c`. Like the
//! C, the TUI runs each turn on a worker thread (see `crate::worker`) while
//! the UI thread keeps handling input — the next prompt stays editable and
//! queueable during generation. The plain line REPL (piped stdin) stays a
//! synchronous inline loop: without a live terminal there is no input to
//! multiplex.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};

use crate::compact;
use crate::config::{AgentConfig, slash_command_known};
use crate::context::{ContextContent, ContextTokens};
use crate::dsml::ToolCall;
use crate::editor::{History, LineBuffer, default_history_path};
use crate::engine::{Engine, EngineEvent};
use crate::render::{RenderOptions, TokenRenderer};
use crate::session::{Message, Session, SessionStore};
use crate::status::{self, Status, WorkerState};
use crate::sysprompt::{self, SystemPromptReminder};
use crate::tools::{ToolContext, dispatch_all};
use crate::trace::Trace;
use crate::tui::{self, OutputLog};
use crate::viz::{RenderSink, StreamRenderer};
use crate::worker::{self, ChannelSink, TurnShared, UiEvent};

/// Stdout writer that flushes after every write so tokens appear as streamed.
#[derive(Debug)]
struct FlushingStdout;

impl Write for FlushingStdout {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut out = std::io::stdout();
        let n = out.write(buf)?;
        out.flush()?;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stdout().flush()
    }
}

/// Routes viz output into the markdown token renderer.
struct TerminalSink {
    renderer: TokenRenderer<FlushingStdout>,
}

impl RenderSink for TerminalSink {
    fn visible_text(&mut self, text: &str) {
        self.renderer.set_in_think(false);
        self.renderer.write(text);
    }
    fn think_text(&mut self, text: &str) {
        self.renderer.set_in_think(true);
        self.renderer.write(text);
    }
    fn error_text(&mut self, text: &str) {
        self.renderer.set_in_think(false);
        self.renderer.color("\x1b[1;31m");
        self.renderer.plain(text);
        self.renderer.color(ANSI_RESET);
    }
}

/// Wraps a `/btw` side question in the reference agent's system-reminder
/// framing: a separate lightweight answer over the shared context, no tools,
/// single response, and nothing enters the main conversation.
fn btw_user_message(question: &str) -> String {
    format!(
        "<system-reminder>This is a side question from the user. You must answer this question directly in a single response.\n\
         \n\
         IMPORTANT CONTEXT:\n\
         - You are a separate, lightweight agent spawned to answer this one question\n\
         - The main conversation is NOT interrupted - this exchange will not become part of it\n\
         - You share the conversation context but are a completely separate instance\n\
         - Do NOT reference being interrupted or what you were \"previously doing\" - that framing is incorrect\n\
         \n\
         CRITICAL CONSTRAINTS:\n\
         - You have NO tools available - you cannot read files, run commands, search, or take any actions\n\
         - This is a one-off response - there will be no follow-up turns\n\
         - You can ONLY provide information based on what you already know from the conversation context\n\
         - NEVER say things like \"Let me try...\", \"I'll now...\", \"Let me check...\", or promise to take any action\n\
         - If you don't know the answer, say so - do not offer to look it up or investigate\n\
         \n\
         Simply answer the question with the information you have.</system-reminder>\n\
         \n\
         {question}"
    )
}

/// Builds the model-visible payload for a failed generation pass, matching
/// the C worker loop: a preflight failure is fed back verbatim, a DSML parse
/// failure gets the C's `invalid DSML tool call: ` prefix plus the syntax
/// reminder so the model can correct its markup.
fn tool_error_payload(preflight: bool, err: &str) -> String {
    if preflight {
        format!("Tool error: {err}\n")
    } else {
        format!(
            "Tool error: invalid DSML tool call: {err}\n{}",
            sysprompt::dsml_syntax_reminder()
        )
    }
}

/// Builds the mid-stream edit preflight hook for a [`StreamRenderer`]: it
/// validates an `edit` call's `old` selector against the file on disk the
/// moment that parameter closes (the C's `agent_stream_preflight_closed_param`).
/// Captures only the working directory, so the live `ToolContext` stays free
/// for tool dispatch.
fn edit_preflight(
    ctx: &ToolContext,
) -> impl FnMut(&ToolCall) -> Result<(), String> + 'static + use<> {
    let ctx = ToolContext::new(ctx.cwd.clone());
    move |call| crate::tools::edit::preflight_edit_old(&ctx, call)
}

/// Parses a `/btw <question>` line, returning the question. Accepts a
/// whitespace or `:` separator, mirroring `OpenClaw`'s `isBtwCommand`
/// matcher; returns `None` for other input or an empty question.
fn btw_question(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("/btw")?;
    let rest = if let Some(r) = rest.strip_prefix(':') {
        r
    } else if rest.starts_with(char::is_whitespace) {
        rest
    } else {
        return None; // "/btwfoo" is not a btw command
    };
    let q = rest.trim();
    if q.is_empty() { None } else { Some(q) }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Renders the session transcript as plain text for the engine.
fn render_transcript(session: &Session, system: &str) -> String {
    use std::fmt::Write as _;
    let mut out = format!("[system]\n{system}\n");
    for m in &session.transcript {
        let tag = match m.role {
            crate::session::Role::User => "user",
            crate::session::Role::Assistant => "assistant",
        };
        let _ = write!(out, "[{tag}]\n{}\n", m.text);
    }
    out
}

/// ANSI reset used by the slash-command reports.
const ANSI_RESET: &str = "\x1b[0m";

/// Image pasting is feature-gated off until the model's handling of
/// image-file references is understood (`--features images` re-enables it).
/// The code stays compiled either way; this constant kills every runtime
/// path: clipboard probing, paste capture, and attachment injection.
const IMAGES_ENABLED: bool = cfg!(feature = "images");

/// Renders the `/mcp` server report following Claude Code's layout: a header
/// with the server count, then one `name · status · N tools` line each.
fn render_mcp_report(servers: &[crate::tools::mcp::McpServer], color: bool) -> String {
    use std::fmt::Write as _;
    let (green, red, reset) = if color {
        ("\x1b[38;5;42m", "\x1b[38;5;204m", ANSI_RESET)
    } else {
        ("", "", "")
    };
    let mut out = String::from("Manage MCP servers\n");
    if servers.is_empty() {
        out.push_str("no servers configured (checked ./.mcp.json and ~/.plank/.mcp.json)\n");
        return out;
    }
    let plural = if servers.len() == 1 { "" } else { "s" };
    let _ = writeln!(out, "{} server{plural}\n", servers.len());
    for s in servers {
        if s.alive() {
            let plural = if s.tools.len() == 1 { "" } else { "s" };
            let _ = writeln!(
                out,
                "  {} · {green}✔ connected{reset} · {} tool{plural}",
                s.name,
                s.tools.len()
            );
        } else {
            let _ = writeln!(out, "  {} · {red}✘ failed{reset}", s.name);
        }
    }
    out
}

/// Shared turn state for the interactive and headless front-ends.
struct Agent<'a> {
    engine: Box<dyn Engine>,
    cfg: &'a AgentConfig,
    session: Session,
    store: SessionStore,
    tool_ctx: ToolContext,
    system: String,
    reminder: SystemPromptReminder,
    trace: Trace,
    power_percent: i32,
    color: bool,
    show_footer: bool,
    /// True when the line editor renders its own resting footer, so the turn
    /// loop must not print a second one after generation.
    editor_owns_footer: bool,
    /// KV position reported by the engine after the last generation; 0 when
    /// no generation has run against the current transcript. Anchors the
    /// `/context` report to the real context usage.
    last_ctx_used: i32,
    /// Context content collected at session start (git, AGENTS.md, date).
    context_content: ContextContent,
    /// Skills loaded from ~/.plank/skills overlaid by ./.plank/skills.
    skills: Vec<crate::skills::Skill>,
    /// Named in-session rollback points (`/checkpoint`, `/rollback`); dropped
    /// when the session is replaced.
    checkpoints: crate::checkpoint::CheckpointStore,
}

/// Default number of user turns replayed by `/history`.
const HISTORY_DEFAULT_TURNS: usize = 3;
/// Sessions shown by the /resume picker.
const RESUME_LIST_LIMIT: usize = 10;
/// Maximum user turns `/history` accepts.
const HISTORY_MAX_TURNS: usize = 200;
/// Name of the auto-checkpoint saved before a `/rollback`, so a rollback is
/// itself undoable via `/rollback pre-rollback`.
const PRE_ROLLBACK_CHECKPOINT: &str = "pre-rollback";

impl Agent<'_> {
    /// Wraps a debug/status message in the thinking gray on color terminals.
    fn debug_line(&self, text: &str) -> String {
        if self.color {
            format!("\x1b[38;5;238m{text}{ANSI_RESET}")
        } else {
            text.to_owned()
        }
    }

    /// Streams one generation pass: paints the live status bar for prefill and
    /// generation, and routes model text through the viz + markdown pipeline.
    #[allow(clippy::type_complexity)]
    fn stream_generation(
        &mut self,
        prompt_text: &str,
        turn_start: Instant,
    ) -> Result<
        (
            StreamRenderer<TerminalSink>,
            String,
            crate::engine::GenerationStats,
        ),
        String,
    > {
        let sink = TerminalSink {
            renderer: TokenRenderer::new(
                FlushingStdout,
                RenderOptions {
                    use_color: self.color,
                    format_thinking: true,
                    format_markdown: true,
                },
            ),
        };
        let mut stream = StreamRenderer::new(sink);
        stream.set_preflight(edit_preflight(&self.tool_ctx));
        // With thinking enabled, the chat template opens `<think>` in the
        // prefill prefix, so generation streams thinking content first; start
        // the renderer inside the think block so it renders gray until `</think>`.
        if !matches!(
            self.cfg.generation.think_mode,
            crate::engine::ThinkMode::Off
        ) {
            stream.begin_in_think();
        }
        let mut assistant_text = String::new();
        let ctx_size = self.engine.ctx_size();
        let power = self.power_percent;
        let prompt_tokens = self.engine.count_tokens(prompt_text);
        let mut bar = crate::statusbar::StatusBar::new(self.show_footer && self.color, self.color);
        let verb = status::random_verb_index();
        // Set when a mid-stream preflight fails: stops the engine early, but
        // is not a user interrupt — the caller feeds the error to the model.
        let preflight_stop = AtomicBool::new(false);
        // Mirrors the C's worker greedy flag: argmax sampling while the
        // stream renderer is inside a DSML tool-call stanza.
        let greedy = AtomicBool::new(false);
        let stats = self
            .engine
            .generate(
                prompt_text,
                &self.cfg.generation,
                &|| preflight_stop.load(Ordering::Relaxed) || crate::interrupt::pending(),
                &|| greedy.load(Ordering::Relaxed),
                &mut |ev| match ev {
                    EngineEvent::Text(t) => {
                        // Model output has started: drop the prefill bar so the
                        // text streams cleanly from column zero.
                        bar.clear();
                        assistant_text.push_str(&t);
                        stream.push(&t);
                        greedy.store(stream.wants_greedy_sampling(), Ordering::Relaxed);
                        if stream.preflight_error().is_some() {
                            preflight_stop.store(true, Ordering::Relaxed);
                        }
                    }
                    EngineEvent::Prefill(p) => {
                        bar.show(&Status {
                            state: WorkerState::Prefill,
                            prefill_done: p.done,
                            prefill_total: p.total,
                            prefill_label: verb,
                            prefill_tps: p.tps,
                            elapsed_secs: turn_start.elapsed().as_secs_f64(),
                            ctx_used: prompt_tokens,
                            ctx_size,
                            power_percent: power,
                            ..Status::default()
                        });
                    }
                },
            )
            .map_err(|e| e.to_string())?;
        stream.finish();
        bar.clear();
        self.last_ctx_used = stats.ctx_used;
        Ok((stream, assistant_text, stats))
    }

    /// Runs one model turn: stream text, execute tool calls, repeat until
    /// a turn produces no tool calls. Compacts first when context is tight.
    fn run_turn(&mut self) -> Result<(), String> {
        self.maybe_compact()?;
        self.maybe_append_system_prompt_reminder();
        // One clock for the whole turn: elapsed time accumulates across the
        // generate → tools → generate loop instead of restarting per pass.
        let turn_start = Instant::now();
        // Stop hooks run at most once per turn, so a hook that always exits 2
        // cannot loop the model forever.
        let mut stop_hook_ran = false;
        loop {
            let prompt_text = render_transcript(&self.session, &self.system);
            let (stream, assistant_text, stats) =
                self.stream_generation(&prompt_text, turn_start)?;

            self.session.push(Message::assistant(assistant_text));
            let st = Status {
                state: if stats.interrupted {
                    WorkerState::Stopped
                } else {
                    WorkerState::Idle
                },
                ctx_used: stats.ctx_used,
                ctx_size: self.engine.ctx_size(),
                generated: stats.generated,
                gen_tps: stats.tps,
                power_percent: self.power_percent,
                ..Status::default()
            };
            // A preflight stop reads as an engine interrupt, but it is a tool
            // error to feed back to the model, not a user abort.
            let preflight_error = stream.preflight_error().map(str::to_owned);
            if stats.interrupted && preflight_error.is_none() {
                crate::interrupt::clear();
                let mut renderer = stream.into_sink().renderer;
                renderer.finish();
                if !renderer.last_output_newline() {
                    println!();
                }
                if self.show_footer && !self.editor_owns_footer {
                    print_footer(&st, self.color);
                }
                return Ok(());
            }
            let finished = stream.finished();
            if let Some(err) = preflight_error.as_deref().or(finished.error) {
                let payload = tool_error_payload(preflight_error.is_some(), err);
                self.session.push(Message::user(format!(
                    "<tool_result>{payload}</tool_result>"
                )));
                continue;
            }
            if !finished.calls.is_empty() {
                let observations = dispatch_all(finished.calls, &mut self.tool_ctx);
                let mut renderer = stream.into_sink().renderer;
                renderer.finish();
                for warning in self.tool_ctx.hook_warnings.drain(..) {
                    let line = if self.color {
                        format!("\x1b[38;5;238m{warning}{ANSI_RESET}")
                    } else {
                        warning
                    };
                    println!("{line}");
                }
                self.session.push(Message::user(format!(
                    "<tool_result>{observations}</tool_result>"
                )));
                continue;
            }
            let mut renderer = stream.into_sink().renderer;
            renderer.finish();
            if !renderer.last_output_newline() {
                println!();
            }
            // Stop hooks: exit 2 feeds stderr to the model and the turn
            // continues (at most once).
            if !stop_hook_ran && let Some(feedback) = self.run_stop_hooks(&mut |w| println!("{w}"))
            {
                stop_hook_ran = true;
                self.session.push(Message::user(format!(
                    "<tool_result>Stop hook feedback:\n{feedback}</tool_result>"
                )));
                continue;
            }
            if self.show_footer && !self.editor_owns_footer {
                print_footer(&st, self.color);
            }
            return Ok(());
        }
    }

    /// Runs the Stop hooks; returns the model-visible feedback of the first
    /// exit-2 hook, `None` when the turn may conclude. `warn` receives
    /// user-only lines from other nonzero exits.
    fn run_stop_hooks(&mut self, warn: &mut dyn FnMut(String)) -> Option<String> {
        if self.tool_ctx.hooks.stop.is_empty() {
            return None;
        }
        let input = crate::hooks::tool_event_input("Stop", "", "{}", None, &self.tool_ctx.cwd);
        let out =
            crate::hooks::run_event(&self.tool_ctx.hooks.stop, "", &input, &self.tool_ctx.cwd);
        for w in out.warnings {
            warn(w);
        }
        out.block
    }

    /// Re-injects the trusted system prompt shape after enough context has
    /// passed since it was last seen, mirroring the C's pressure policy.
    fn maybe_append_system_prompt_reminder(&mut self) {
        let rendered = render_transcript(&self.session, &self.system);
        let pos = self.engine.count_tokens(&rendered);
        if !self.reminder.should_remind(pos) {
            return;
        }
        println!(
            "{}",
            self.debug_line("Re-injecting system prompt reminder...")
        );
        self.trace.line(&format!(
            "system prompt reminder injected at transcript={pos}"
        ));
        let mut text = sysprompt::build_system_prompt_reminder(&self.tool_ctx.mcp);
        if !self.cfg.system.is_empty() {
            text.push_str("\nAdditional system instructions reminder:\n");
            text.push_str(&self.cfg.system);
            text.push_str("\n[End additional system instructions reminder.]\n\n");
        }
        self.session.push(Message::user(text));
    }

    /// Compacts the transcript when the rendered context is nearly full.
    fn maybe_compact(&mut self) -> Result<(), String> {
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        if !compact::should_compact(self.engine.ctx_size(), used) {
            return Ok(());
        }
        // Cheapest step first: clear old tool-result bodies (no model
        // round-trip) and only fall back to full summarization if still tight.
        if let Some(cleared) = self.try_microcompact() {
            println!(
                "{}",
                self.debug_line(&format!(
                    "microcompacted: cleared {cleared} old tool result(s)"
                ))
            );
            return Ok(());
        }
        self.compact("low context")
    }

    /// Runs microcompact; returns the cleared count when it freed enough
    /// context to skip full compaction, `None` when full compaction is still
    /// needed (any clearing done is kept — it only helps the summary pass).
    fn try_microcompact(&mut self) -> Option<usize> {
        let cleared = compact::microcompact(&mut self.session.transcript);
        if cleared == 0 {
            return None;
        }
        self.last_ctx_used = 0;
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        (!compact::should_compact(self.engine.ctx_size(), used)).then_some(cleared)
    }

    /// Rebuilds the transcript after a summarization pass: extracted summary
    /// + verbatim tail + budgeted re-injection of recently read files.
    fn rebuild_after_compact(&mut self, raw_summary: &str) {
        let summary = compact::extract_summary(raw_summary);
        let budget = compact::tail_budget(self.engine.ctx_size());
        let mut tail_start = self.session.transcript.len();
        let mut tail_tokens = 0;
        while tail_start > 0 {
            let m = &self.session.transcript[tail_start - 1];
            tail_tokens += self.engine.count_tokens(&m.text);
            if tail_tokens > budget {
                break;
            }
            tail_start -= 1;
        }
        let tail: Vec<Message> = self.session.transcript[tail_start..].to_vec();
        self.session.transcript = Vec::new();
        self.session.push(Message::user(format!(
            "<tool_result>Compacted session summary:\n{summary}</tool_result>"
        )));
        self.session.transcript.extend(tail);
        let reinject = compact::build_reinjection(
            &self.tool_ctx.recent_reads,
            compact::reinject_budget(self.engine.ctx_size()),
            &mut |s| self.engine.count_tokens(s),
        );
        if let Some(block) = reinject {
            self.session.push(Message::user(block));
        }
        self.last_ctx_used = 0;
    }

    /// Performs the compaction exchange and rebuilds the transcript as
    /// summary + recent verbatim tail.
    fn compact(&mut self, reason: &str) -> Result<(), String> {
        print!("{}", compact::banner(reason, self.color));
        let mut prompt_text = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(prompt_text, "[user]\n{}\n", compact::make_prompt(reason));
        }
        let mut summary = String::new();
        self.engine
            .generate(
                &prompt_text,
                &self.cfg.generation,
                &|| false,
                &|| false,
                &mut |ev| {
                    if let EngineEvent::Text(t) = ev {
                        summary.push_str(&t);
                    }
                },
            )
            .map_err(|e| e.to_string())?;
        if self.color {
            print!("\x1b[0m");
        }

        self.rebuild_after_compact(&summary);
        println!("{}", self.debug_line("context compacted"));
        Ok(())
    }

    /// Renders the `/context` usage breakdown with Claude Code's layout: a
    /// 20-column cell grid (1k tokens per cell, coarser for large contexts
    /// so the grid stays within half a typical screen) beside the model and
    /// totals, then the estimated usage per category.
    #[allow(clippy::too_many_lines)]
    fn render_context_report(&self, color: bool) -> String {
        use std::fmt::Write as _;
        /// Glyph for an unused context cell in the grid.
        const FREE_CELL: char = '⛶';
        /// Grid width in cells.
        const GRID_COLS: usize = 20;
        /// Maximum grid height in rows.
        const MAX_GRID_ROWS: usize = 16;
        /// Category colors matching Claude Code: violet, cyan, purple, gray.
        const COL_SYSTEM: &str = "\x1b[38;5;105m";
        const COL_MCP: &str = "\x1b[38;5;44m";
        const COL_MSG: &str = "\x1b[38;5;134m";
        const COL_CONTEXT: &str = "\x1b[38;5;208m";
        const COL_MEMORY: &str = "\x1b[38;5;114m";
        const COL_FREE: &str = "\x1b[38;5;240m";
        let paint = |col: &'static str| if color { col } else { "" };
        let reset = if color { ANSI_RESET } else { "" };
        let ctx_size = self.engine.ctx_size().max(1);
        let mut schemas = String::new();
        crate::tools::mcp::append_tool_schemas(&mut schemas, &self.tool_ctx.mcp);
        let mcp_tokens = if schemas.is_empty() {
            0
        } else {
            self.engine.count_tokens(&schemas)
        };
        // MCP tool schemas are embedded in the composed system prompt; split
        // them out so the two categories don't double-count.
        // The system prompt includes: tools prompt + user system text
        let mut system_tokens = (self.engine.count_tokens(&self.system) - mcp_tokens).max(0);
        let mut mcp_tokens = mcp_tokens;
        // AGENTS.md tokens from the context collected at session start.
        let context_tokens =
            ContextTokens::count(&self.context_content, |s| self.engine.count_tokens(s));
        // Message tokens: all transcript messages (user and assistant)
        let raw_message_tokens: i32 = self
            .session
            .transcript
            .iter()
            .map(|m| self.engine.count_tokens(&m.text))
            .sum();
        // AGENTS.md gets its own category; git and date context stay grouped
        // under Messages (they are part of the injected first user message).
        let agents_md_tokens = context_tokens.agents_md;
        let memory_tokens = context_tokens.memory;
        let mut message_tokens = raw_message_tokens - agents_md_tokens - memory_tokens;

        let estimated =
            system_tokens + mcp_tokens + message_tokens + agents_md_tokens + memory_tokens;
        if self.last_ctx_used > estimated && estimated > 0 {
            let scale = |t: i32| {
                i32::try_from(i64::from(t) * i64::from(self.last_ctx_used) / i64::from(estimated))
                    .unwrap_or(t)
            };
            system_tokens = scale(system_tokens);
            mcp_tokens = scale(mcp_tokens);
            message_tokens = scale(message_tokens);
        }

        let used = (system_tokens + mcp_tokens + message_tokens + agents_md_tokens + memory_tokens)
            .min(ctx_size);
        let free = ctx_size - used;
        let pct = |n: i32| f64::from(n) * 100.0 / f64::from(ctx_size);

        // Categories are told apart by color; the glyph of each cell shows
        // how full that cell is (see `fill_glyph`).
        let mut categories = vec![
            ("System prompt", system_tokens, COL_SYSTEM),
            ("MCP tools", mcp_tokens, COL_MCP),
        ];

        if agents_md_tokens > 0 {
            categories.push(("AGENTS.md", agents_md_tokens, COL_CONTEXT));
        }

        if memory_tokens > 0 {
            categories.push(("Memory", memory_tokens, COL_MEMORY));
        }

        categories.push(("Messages", message_tokens, COL_MSG));

        // Glyph for a cell by its fill fraction: <25%, <50%, <75%, full.
        let fill_glyph = |frac: f64| -> char {
            if frac < 0.25 {
                '⛀'
            } else if frac < 0.5 {
                '⛂'
            } else if frac < 0.75 {
                '⛁'
            } else {
                '⛃'
            }
        };

        // Adaptive density: 1k tokens per cell, coarsened (in 1k steps) so the
        // grid never exceeds half a typical 24-row screen. Every non-empty
        // category shows at least one cell; free space takes what remains.
        #[allow(clippy::cast_sign_loss)]
        let ctx = ctx_size as usize;
        let tokens_per_cell = ctx
            .div_ceil(GRID_COLS * MAX_GRID_ROWS)
            .div_ceil(1000)
            .max(1)
            * 1000;
        let total_cells = ctx.div_ceil(tokens_per_cell);
        let mut cells: Vec<(char, &'static str)> = Vec::with_capacity(total_cells);
        for &(_, tokens, col) in &categories {
            if tokens <= 0 || cells.len() == total_cells {
                continue;
            }
            // Whole cells render full; the trailing remainder renders with a
            // glyph matching its fill fraction.
            #[allow(clippy::cast_sign_loss)]
            let tokens = tokens as usize;
            let full = (tokens / tokens_per_cell).min(total_cells - cells.len());
            cells.extend(std::iter::repeat_n(('⛃', col), full));
            let rem = tokens % tokens_per_cell;
            if rem > 0 && cells.len() < total_cells {
                #[allow(clippy::cast_precision_loss)]
                cells.push((fill_glyph(rem as f64 / tokens_per_cell as f64), col));
            }
        }
        cells.truncate(total_cells);
        cells.resize(total_cells, (FREE_CELL, COL_FREE));
        let grid_rows = total_cells.div_ceil(GRID_COLS);

        // Right-hand column: model line, totals, then the category legend.
        let model = self.engine.model_name();
        let mut right: Vec<String> = Vec::new();
        if !model.is_empty() {
            right.push(model);
        }
        right.push(format!(
            "{}/{} tokens ({:.0}%)",
            status::format_ctx_size(used),
            status::format_ctx_size(ctx_size),
            pct(used)
        ));
        right.push(String::new());
        right.push("Estimated usage by category".to_owned());
        for &(label, tokens, col) in &categories {
            right.push(format!(
                "{}⛃{reset} {label}: {} tokens ({:.1}%)",
                paint(col),
                status::format_ctx_size(tokens),
                pct(tokens)
            ));
        }
        right.push(format!(
            "{}{FREE_CELL}{reset} Free space: {} ({:.1}%)",
            paint(COL_FREE),
            status::format_ctx_size(free),
            pct(free)
        ));
        right.push(format!(
            "1 cell = {} tokens",
            status::format_ctx_size(i32::try_from(tokens_per_cell).unwrap_or(i32::MAX))
        ));

        let mut out = String::from("Context Usage\n");
        let rows = right.len().max(grid_rows);
        for row in 0..rows {
            out.push_str("  ");
            if row < grid_rows {
                let start = row * GRID_COLS;
                let end = (start + GRID_COLS).min(total_cells);
                for &(glyph, col) in &cells[start..end] {
                    out.push_str(paint(col));
                    out.push(glyph);
                    out.push_str(reset);
                    out.push(' ');
                }
                out.push_str(&" ".repeat(2 * (start + GRID_COLS - end)));
            } else {
                out.push_str(&" ".repeat(2 * GRID_COLS));
            }
            if let Some(text) = right.get(row) {
                let _ = write!(out, "   {text}");
            }
            out.push('\n');
        }
        out
    }

    /// Runs the /init command: prompts the model to create AGENTS.md
    fn run_init(&mut self) {
        println!("Initializing AGENTS.md...");
        println!("The model will now analyze the codebase and generate documentation.\n");

        let prompt = concat!(
            "Analyze this codebase and create an AGENTS.md file for future agent sessions.\n\n",
            "Include:\n",
            "1. Build, lint, and test commands (especially non-standard ones)\n",
            "2. High-level architecture and structure\n",
            "3. Required setup or environment variables\n",
            "4. Non-obvious gotchas or workflow quirks\n\n",
            "Exclude:\n",
            "- File-by-file listings Claude can discover\n",
            "- Standard language conventions\n",
            "- Generic advice\n",
            "- Information from README unless essential\n\n",
            "Preface with:\n",
            "```",
            "# AGENTS.md\n\n",
            "This file provides guidance to the agent when working with code in this repository.",
            "```",
            "\n\n",
            "Write the AGENTS.md file to the current directory."
        );

        self.session.push(Message::user(prompt));
        if let Err(e) = self.run_turn() {
            println!("/init failed: {e}");
        }
    }

    /// Runs the /init command in TUI mode.
    fn tui_run_init(
        &mut self,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
        input: &mut TuiInput,
        btw: &mut BtwPanel,
    ) {
        log.push_plain("Initializing AGENTS.md...");
        log.push_plain("The model will now analyze the codebase and generate documentation.\n");

        let prompt = concat!(
            "Analyze this codebase and create an AGENTS.md file for future agent sessions.\n\n",
            "Include:\n",
            "1. Build, lint, and test commands (especially non-standard ones)\n",
            "2. High-level architecture and structure\n",
            "3. Required setup or environment variables\n",
            "4. Non-obvious gotchas or workflow quirks\n\n",
            "Exclude:\n",
            "- File-by-file listings Claude can discover\n",
            "- Standard language conventions\n",
            "- Generic advice\n",
            "- Information from README unless essential\n\n",
            "Preface with:\n",
            "```",
            "# AGENTS.md\n\n",
            "This file provides guidance to the agent when working with code in this repository.",
            "```",
            "\n\n",
            "Write the AGENTS.md file to the current directory."
        );

        log.push_spans(tui::user_echo_spans(prompt));
        self.session.push(Message::user(prompt));
        if let Err(e) = self.tui_turn(terminal, log, view, input, btw) {
            log.push_plain(format!("/init failed: {e}"));
        }
    }

    /// Handles a slash command; returns false when the REPL should exit.
    #[allow(clippy::too_many_lines)]
    fn slash(&mut self, input: &str) -> Result<bool, String> {
        let mut parts = input.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or(input);
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            "/init" => {
                self.run_init();
                return Ok(true);
            }
            "/quit" | "/exit" => return Ok(false),
            "/new" | "/clear" => {
                self.session = Session::new();
                self.reminder = SystemPromptReminder::new();
                self.context_content = ContextContent::new();
                let combined = self.context_content.combined();
                self.session.push(Message::user(combined));
                self.last_ctx_used = 0;
                self.checkpoints.clear();
                println!("started a new session");
            }
            "/help" => print!("{}", crate::config::usage()),
            "/checkpoint" => {
                if arg.is_empty() {
                    print!(
                        "{}",
                        crate::checkpoint::render_list(&self.checkpoints, now_secs(), self.color)
                    );
                } else {
                    println!("{}", self.checkpoint_create(arg));
                }
            }
            "/rollback" => {
                if arg.is_empty() {
                    println!("usage: /rollback <name> (see /checkpoint for the list)");
                } else {
                    match self.rollback_to(arg) {
                        Ok(msg) => println!("{msg}"),
                        Err(e) => println!("{e}"),
                    }
                }
            }
            "/save" => match self.store.save(&mut self.session) {
                Ok(id) => println!("saved session {}", &id[..8]),
                Err(e) => println!("save failed: {e}"),
            },
            "/list" => match self.store.list() {
                Ok(entries) => print!(
                    "{}",
                    crate::session::render_session_list(&entries, now_secs(), self.color)
                ),
                Err(e) => println!("list failed: {e}"),
            },
            "/switch" => match self.store.load(arg) {
                Ok(s) => {
                    print!(
                        "{}",
                        crate::session::render_history(&s.transcript, 6, self.color)
                    );
                    self.session = s;
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                }
                Err(e) => println!("switch failed: {e}"),
            },
            "/del" => match self.store.delete(arg) {
                Ok(id) => println!("deleted session {}", &id[..8]),
                Err(e) => println!("delete failed: {e}"),
            },
            "/resume" => match self.resume_pick(arg) {
                Ok(None) => match self.store.list() {
                    Ok(entries) => print!(
                        "{}",
                        crate::session::render_resume_list(
                            &entries,
                            now_secs(),
                            self.color,
                            RESUME_LIST_LIMIT
                        )
                    ),
                    Err(e) => println!("resume failed: {e}"),
                },
                Ok(Some(s)) => {
                    print!(
                        "{}",
                        crate::session::render_history(&s.transcript, 6, self.color)
                    );
                    self.session = s;
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                }
                Err(e) => println!("resume failed: {e}"),
            },
            "/tag" => {
                if arg.is_empty() {
                    if self.session.tag.is_empty() {
                        println!("no tag set; usage: /tag <text> (\"/tag -\" clears)");
                    } else {
                        println!("tag: {}", self.session.tag);
                    }
                } else {
                    match self.set_tag(arg) {
                        Ok(msg) => println!("{msg}"),
                        Err(e) => println!("tag failed: {e}"),
                    }
                }
            }
            "/history" => {
                let turns = if arg.is_empty() {
                    HISTORY_DEFAULT_TURNS
                } else {
                    arg.parse::<usize>()
                        .unwrap_or(HISTORY_DEFAULT_TURNS)
                        .clamp(1, HISTORY_MAX_TURNS)
                };
                print!(
                    "{}",
                    crate::session::render_history(&self.session.transcript, turns, self.color)
                );
            }
            "/power" => match crate::config::parse_power_percent(arg) {
                Some(power) => {
                    // No GPU backend yet: record and show it in the footer,
                    // like the C's deferred worker_request_power.
                    self.power_percent = power;
                    println!("power limit set to {power}%");
                }
                None => println!("usage: /power <1..100>"),
            },
            "/strip" => {
                if arg.is_empty() {
                    println!("usage: /strip <sha-prefix>");
                } else {
                    // Plank sessions store no engine KV payload to strip; keep
                    // the C's success framing with zero tokens reclaimed.
                    match self.store.find(arg) {
                        Ok((sha, _)) => {
                            println!("stripped session {} (0 tokens)", &sha[..8]);
                        }
                        Err(e) => println!("strip failed: {e}"),
                    }
                }
            }
            "/mcp" => print!("{}", render_mcp_report(&self.tool_ctx.mcp, self.color)),
            "/context" => print!("{}", self.render_context_report(self.color)),
            "/compact" => self.compact("user request")?,
            "/skills" => print!("{}", crate::skills::render_list(&self.skills)),
            "/hooks" => print!("{}", crate::hooks::render_list(&self.tool_ctx.hooks)),
            "/btw" => {
                if arg.is_empty() {
                    println!("usage: /btw <question>");
                } else {
                    self.btw_plain(arg)?;
                }
            }
            "/remember" => match remember_from_arg(&self.tool_ctx.cwd, arg) {
                Ok(path) => println!(
                    "{}",
                    self.debug_line(&format!("[saved to {}]", path.display()))
                ),
                Err(e) => println!("{e}\nusage: /remember [user] <text> (default scope: project)"),
            },
            "/repro" => match self.write_repro(arg) {
                Ok(path) => println!(
                    "{}",
                    self.debug_line(&format!("[repro written to {}]", path.display()))
                ),
                Err(e) => println!("repro failed: {e}"),
            },
            "/subagent" => {
                if arg.is_empty() {
                    println!("usage: /subagent <task>");
                } else {
                    println!("{}", self.debug_line("[subagent started]"));
                    let fork_at = self.begin_subagent_fork(arg);
                    // Restore the transcript even when the turn errored.
                    let turn = self.run_turn();
                    let reported = self.finish_subagent_fork(fork_at, arg);
                    turn?;
                    let trailer = if reported {
                        "[subagent report added to the conversation]"
                    } else {
                        "[subagent produced no report — nothing added]"
                    };
                    println!("{}", self.debug_line(trailer));
                }
            }
            _ if slash_command_known(cmd) => println!("{cmd}: not implemented yet"),
            _ => {
                if let Some(message) = self.skill_message(cmd, arg) {
                    print!("{}", status::format_user_prompt_echo(input, self.color));
                    self.session.push(Message::user(message));
                    self.run_turn()?;
                } else {
                    println!("unknown command: {cmd}");
                }
            }
        }
        Ok(true)
    }

    /// Runs a `/btw` side question in the plain REPL: one generation pass
    /// over the shared context plus the framed question, tools denied,
    /// nothing pushed to the session. The next real turn's KV sync reuses
    /// the still-matching prefix and re-prefills past the divergence, so the
    /// side question rolls back automatically.
    /// Resolves a `/resume` argument: `Ok(None)` for an empty argument (show
    /// the picker), otherwise the loaded session — a small number picks from
    /// the recency-sorted listing, anything else is a sha prefix.
    fn resume_pick(&self, arg: &str) -> Result<Option<Session>, String> {
        let arg = arg.trim();
        if arg.is_empty() {
            return Ok(None);
        }
        if let Ok(n) = arg.parse::<usize>() {
            let entries = self.store.list().map_err(|e| e.to_string())?;
            let entry = entries
                .get(n.wrapping_sub(1))
                .ok_or_else(|| format!("no session number {n} (see /resume)"))?;
            return self
                .store
                .load(&entry.id)
                .map(Some)
                .map_err(|e| e.to_string());
        }
        self.store.load(arg).map(Some).map_err(|e| e.to_string())
    }

    /// Resumes a session named on the command line (`plank /resume [prefix]`)
    /// before the interactive loop starts. An empty `arg` resumes the most
    /// recent session; otherwise it is a number from the listing or a sha
    /// prefix. Only loads the session — each front-end renders the recovered
    /// history itself (see [`resumed_history`]), since the TUI's alternate
    /// screen would wipe anything printed here.
    fn resume_from_cli(&mut self, arg: &str) -> Result<(), String> {
        let session = if arg.trim().is_empty() {
            let entries = self.store.list().map_err(|e| e.to_string())?;
            let entry = entries
                .first()
                .ok_or_else(|| "no saved sessions to resume".to_string())?;
            self.store.load(&entry.id).map_err(|e| e.to_string())?
        } else {
            self.resume_pick(arg)?
                .ok_or_else(|| "no such session".to_string())?
        };
        self.session = session;
        self.last_ctx_used = 0;
        Ok(())
    }

    /// Recent-history text for a just-resumed session (empty when the current
    /// session was not loaded from disk), plus a `[resumed …]` trailer, for a
    /// front-end to display at startup.
    fn resumed_history(&self) -> Option<String> {
        use std::fmt::Write as _;
        if self.session.id.is_empty() {
            return None;
        }
        let mut out = crate::session::render_history(&self.session.transcript, 6, self.color);
        let short = crate::session::display_id(&self.session.id);
        let _ = write!(
            out,
            "{}",
            self.debug_line(&format!("[resumed session {short}]"))
        );
        Some(out)
    }

    /// Captures a named checkpoint: the current transcript plus the engine KV
    /// snapshot (when the engine supports it). Returns a status line.
    fn checkpoint_create(&mut self, name: &str) -> String {
        let kv = self.engine.snapshot_kv();
        let had_kv = kv.is_some();
        let replaced = self.checkpoints.save(name, &self.session, kv);
        let verb = if replaced { "updated" } else { "saved" };
        let note = if had_kv {
            " (with engine KV)"
        } else {
            " (transcript only)"
        };
        format!("checkpoint {verb}: {name}{note}")
    }

    /// Rolls back to a named checkpoint: the current tail is saved first as
    /// `pre-rollback` (so the rollback is undoable), then the transcript is
    /// restored verbatim and, when the checkpoint carries engine KV, the
    /// session KV is restored so the next turn skips re-prefill.
    fn rollback_to(&mut self, name: &str) -> Result<String, String> {
        let Some(cp) = self.checkpoints.get(name).cloned() else {
            return Err(format!("no checkpoint named {name} (see /checkpoint)"));
        };
        // Snapshot the current tail before discarding it.
        let tail_kv = self.engine.snapshot_kv();
        self.checkpoints
            .save(PRE_ROLLBACK_CHECKPOINT, &self.session, tail_kv);
        crate::checkpoint::restore_transcript(&mut self.session, &cp);
        self.last_ctx_used = 0;
        let note = match &cp.kv {
            Some(bytes) if self.engine.restore_kv(bytes).is_ok() => {
                " (engine KV restored, zero re-prefill)"
            }
            _ => " (transcript restored, re-prefill on next turn)",
        };
        Ok(format!(
            "rolled back to {name}{note}; tail saved as \"{PRE_ROLLBACK_CHECKPOINT}\""
        ))
    }

    /// Sets (or with `-` clears) the session tag, re-saving immediately when
    /// the session was already saved so listings pick it up.
    fn set_tag(&mut self, arg: &str) -> Result<String, String> {
        let tag = if arg == "-" { "" } else { arg.trim() };
        tag.clone_into(&mut self.session.tag);
        self.session.dirty = true;
        let mut msg = if tag.is_empty() {
            "tag cleared".to_string()
        } else {
            format!("tag set: {tag}")
        };
        if !self.session.id.is_empty() {
            self.store
                .save(&mut self.session)
                .map_err(|e| e.to_string())?;
            msg.push_str(" (saved)");
        }
        Ok(msg)
    }

    /// Saves the session at exit and returns `(id, path)` so the caller can
    /// tell the user how to resume it. Returns `None` when there is nothing
    /// worth saving (no user turn) or the save fails.
    fn save_for_exit(&mut self) -> Option<(String, std::path::PathBuf)> {
        let id = self.store.save(&mut self.session).ok()?;
        let path = self
            .store
            .find(&id)
            .map_or_else(|_| self.store.dir().join(format!("{id}.kv")), |(_, p)| p);
        Some((id, path))
    }

    /// At session end, saves the transcript and prints where it landed and how
    /// to resume it. A session with no user turn is silently skipped.
    fn report_session_on_exit(&mut self) {
        let Some((id, path)) = self.save_for_exit() else {
            return;
        };
        let short = crate::session::display_id(&id);
        let (bold, dim, reset) = if self.color {
            ("\x1b[1m", "\x1b[38;5;238m", ANSI_RESET)
        } else {
            ("", "", "")
        };
        println!();
        println!("{bold}Session saved{reset} {dim}{}{reset}", path.display());
        println!("Resume it later with:  {bold}plank{reset}  then  {bold}/resume {short}{reset}");
    }

    /// Writes a `/repro` diagnostic dump — the exact rendered engine input
    /// plus the runtime knobs that shape generation — to `~/.plank/repro/`.
    /// `note` is an optional free-text description of the bug. Read-only: the
    /// live session is untouched.
    fn write_repro(&self, note: &str) -> Result<std::path::PathBuf, String> {
        let rendered = render_transcript(&self.session, &self.system);
        let version = crate::logo::version_label();
        let date = crate::context::current_local_iso_date();
        let meta = crate::repro::Meta {
            version: &version,
            date: &date,
            ctx_size: self.engine.ctx_size(),
            transcript_tokens: self.engine.count_tokens(&rendered),
            last_ctx_used: self.last_ctx_used,
            power_percent: self.power_percent,
            session_id: &self.session.id,
            session_tag: &self.session.tag,
            note: note.trim(),
        };
        let report = crate::repro::build_report(&meta, self.cfg, &rendered);
        crate::repro::save(&self.tool_ctx.cwd, now_secs(), &report)
    }

    /// Starts a `/subagent` fork: appends the framed task to the live
    /// transcript and returns the pre-fork length for later truncation. The
    /// fork inherits the parent transcript prefix, so the engine's per-turn
    /// sync reuses the parent KV cache.
    fn begin_subagent_fork(&mut self, task: &str) -> usize {
        let fork_at = self.session.transcript.len();
        self.session
            .push(Message::user(crate::agents::task_message(task)));
        fork_at
    }

    /// Ends a `/subagent` fork: truncates the sidechain back out of the
    /// transcript and pushes only the framed final report. Returns false when
    /// the sidechain produced no report (e.g. interrupted before any output);
    /// the transcript is still restored.
    fn finish_subagent_fork(&mut self, fork_at: usize, task: &str) -> bool {
        let report = self.session.transcript[fork_at..]
            .iter()
            .rev()
            .find(|m| {
                matches!(m.role, crate::session::Role::Assistant) && !m.text.trim().is_empty()
            })
            .map(|m| m.text.clone());
        self.session.transcript.truncate(fork_at);
        match report {
            Some(report) => {
                self.session
                    .push(Message::user(crate::agents::report_message(task, &report)));
                true
            }
            None => false,
        }
    }

    fn btw_plain(&mut self, question: &str) -> Result<(), String> {
        let mut prompt_text = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(prompt_text, "[user]\n{}\n", btw_user_message(question));
        }
        let saved_ctx = self.last_ctx_used;
        let (stream, _text, _stats) = self.stream_generation(&prompt_text, Instant::now())?;
        let tried_tool = !stream.finished().calls.is_empty() || stream.finished().error.is_some();
        let mut renderer = stream.into_sink().renderer;
        renderer.finish();
        if !renderer.last_output_newline() {
            println!();
        }
        if tried_tool {
            println!(
                "(the model tried to call a tool; tools are disabled during /btw — ask in the main conversation)"
            );
        }
        println!(
            "{}",
            self.debug_line("[btw — not part of the conversation]")
        );
        self.last_ctx_used = saved_ctx;
        Ok(())
    }

    /// Resolves `/name args` against the loaded skills, rendering the
    /// user-turn preamble on a match.
    fn skill_message(&self, cmd: &str, arg: &str) -> Option<String> {
        let name = cmd.strip_prefix('/')?;
        let skill = self.skills.iter().find(|s| s.name == name)?;
        Some(crate::skills::render(skill, arg))
    }
}

/// Parses `/remember [user] <text>` and appends to the right memory scope:
/// a leading `user` word selects the user file, everything else lands in the
/// project file.
fn remember_from_arg(cwd: &std::path::Path, arg: &str) -> Result<std::path::PathBuf, String> {
    let arg = arg.trim();
    let (scope, text) = match arg.split_once(char::is_whitespace) {
        Some(("user", rest)) => (crate::memory::Scope::User, rest),
        _ => (crate::memory::Scope::Project, arg),
    };
    crate::memory::remember(scope, cwd, text, &crate::context::current_local_iso_date())
}

/// The `/btw` side panel: `Some` while it splits the screen (main 60% / btw
/// 40%). Owned by [`Agent::tui_loop`] so it persists across turn boundaries —
/// a finished main task never closes it; only Esc does.
type BtwPanel = Option<(OutputLog, tui::OutputView)>;

/// Result of one TUI generation pass.
struct TurnOutput {
    interrupted: bool,
    /// A priority `/btw` stopped this main pass; the caller discards the
    /// partial output, answers the side question, and re-runs the pass.
    preempted: bool,
    assistant_text: String,
    calls: Vec<ToolCall>,
    error: Option<String>,
}

/// Interactive input state for the ratatui UI.
struct TuiInput {
    buf: LineBuffer,
    history: History,
    hist_idx: Option<usize>,
    stash: String,
}

impl TuiInput {
    fn new() -> Self {
        Self {
            buf: LineBuffer::new(),
            history: History::new(512),
            hist_idx: None,
            stash: String::new(),
        }
    }

    /// Display column of the cursor within the input text.
    fn cursor_col(&self) -> u16 {
        let text = self.buf.text();
        let bytes = self.buf.cursor().min(text.len());
        u16::try_from(text[..bytes].chars().count()).unwrap_or(u16::MAX)
    }

    /// Moves through history like the line editor (dir -1 = older).
    fn history_move(&mut self, dir: i32) {
        if self.history.is_empty() {
            return;
        }
        let len = self.history.len();
        let new_index = match (self.hist_idx, dir) {
            (None, d) if d < 0 => {
                self.stash = self.buf.text().to_owned();
                Some(len - 1)
            }
            (None, _) => None,
            (Some(0), d) if d < 0 => Some(0),
            (Some(i), d) if d < 0 => Some(i - 1),
            (Some(i), _) if i + 1 < len => Some(i + 1),
            (Some(_), _) => {
                self.buf.set_text(std::mem::take(&mut self.stash));
                self.hist_idx = None;
                return;
            }
        };
        self.hist_idx = new_index;
        if let Some(i) = new_index {
            let entry = self.history.get(i).unwrap_or_default().to_owned();
            self.buf.set_text(entry);
        }
    }
}

impl Agent<'_> {
    /// Runs the full-screen ratatui interactive session.
    ///
    /// # Errors
    /// Returns an error string on unrecoverable terminal or engine failure.
    fn run_tui(&mut self) -> Result<(), String> {
        let mut terminal = ratatui::init();
        // Capture the mouse so wheel events scroll the output buffer instead
        // of being translated by the terminal into arrow keys (history moves),
        // and drags select text for copying. Bracketed paste makes Cmd-V
        // arrive as a single Paste event instead of a burst of key presses.
        let _ = ratatui::crossterm::execute!(
            std::io::stdout(),
            EnableMouseCapture,
            EnableBracketedPaste
        );
        let result = self.tui_loop(&mut terminal);
        let _ = ratatui::crossterm::execute!(
            std::io::stdout(),
            DisableBracketedPaste,
            DisableMouseCapture
        );
        ratatui::restore();
        result
    }

    #[allow(clippy::too_many_lines)]
    fn tui_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<(), String> {
        let mut input = TuiInput::new();
        let hist_path = default_history_path();
        input.history.load(&hist_path).ok();
        let mut log = OutputLog::new();
        for line in tui::ansi_to_lines(&crate::logo::art(crate::logo::DEFAULT_WIDTH * 144 / 100)) {
            log.push_spans(line.spans);
        }
        log.push_plain(format!(
            "plank {} 🪵 Agent, context {} tokens",
            crate::logo::version_label(),
            status::format_ctx_size(self.engine.ctx_size())
        ));
        log.push_plain("Type a message, or /help for commands. Ctrl-D to quit.");
        log.push_plain(String::new());

        // A `plank /resume` startup shows the recovered conversation so far.
        if let Some(history) = self.resumed_history() {
            log.push_ansi(&history);
        }

        self.tui_warm(terminal, &mut log)?;

        let mut view = tui::OutputView::default();
        // The `/btw` side panel, owned here so it outlives any single turn:
        // once opened it stays until the user presses Esc, even after the main
        // task finishes and control returns to this idle loop.
        let mut btw_panel: BtwPanel = None;
        if let Some(initial) = self.cfg.prompt.as_deref().filter(|p| !p.is_empty()) {
            log.push_spans(tui::user_echo_spans(initial));
            self.session.push(Message::user(initial));
            self.tui_turn(terminal, &mut log, &mut view, &mut input, &mut btw_panel)?;
        }

        // Endpoints of a mouse drag selection over the output area, in screen
        // cells (anchor, current). Copied to the clipboard on button release.
        let mut selection: Option<((u16, u16), (u16, u16))> = None;
        // Images pasted (clipboard or file path) awaiting the next submit;
        // attached to the message as file references the model's tools can
        // read. Always empty while IMAGES_ENABLED is off.
        let mut attachments: Vec<crate::imagepaste::PastedImage> = Vec::new();
        // Clipboard-image hint, re-probed every few seconds (the probe shells
        // out to osascript, so it must not run on every 200ms poll tick).
        let mut clip_has_image = IMAGES_ENABLED && crate::imagepaste::clipboard_has_image();
        let mut clip_checked = Instant::now();
        loop {
            if IMAGES_ENABLED && clip_checked.elapsed() >= Duration::from_secs(3) {
                clip_has_image = crate::imagepaste::clipboard_has_image();
                clip_checked = Instant::now();
            }
            let mut status = self.idle_status_text();
            if clip_has_image {
                status.push_str(" | 📷 image in clipboard (Cmd-V attaches)");
            }
            terminal
                .draw(|f| {
                    // A `/btw` panel left open from an earlier turn keeps the
                    // split view even while idle; text selection falls back to
                    // the single-column path (no panel).
                    if let Some((btw_log, btw_view)) = btw_panel.as_mut() {
                        tui::draw_btw_split(
                            f,
                            &log,
                            btw_log,
                            btw_view,
                            Some(input.buf.text()),
                            input.cursor_col(),
                            &status,
                            &mut view,
                        );
                    } else {
                        tui::draw(
                            f,
                            &log,
                            Some(input.buf.text()),
                            input.cursor_col(),
                            &status,
                            &mut view,
                            selection.map(|(a, b)| tui::normalize_selection(a, b)),
                        );
                    }
                })
                .map_err(|e| e.to_string())?;

            if !event::poll(Duration::from_millis(200)).map_err(|e| e.to_string())? {
                continue;
            }
            let ev = event::read().map_err(|e| e.to_string())?;
            if let Event::Mouse(m) = &ev {
                match m.kind {
                    MouseEventKind::ScrollUp => {
                        selection = None;
                        view.follow = false;
                        view.top = view.top.saturating_sub(3);
                    }
                    MouseEventKind::ScrollDown => {
                        selection = None;
                        // Clamped by draw, which re-enters follow mode at the bottom.
                        view.top = view.top.saturating_add(3);
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        selection = Some(((m.column, m.row), (m.column, m.row)));
                    }
                    MouseEventKind::Drag(MouseButton::Left) => {
                        if let Some((_, end)) = &mut selection {
                            *end = (m.column, m.row);
                        }
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        if let Some((a, b)) = selection.filter(|(a, b)| a != b) {
                            // Redraw and read the cells inside the same frame:
                            // after a draw() the terminal's current buffer is
                            // the cleared next-frame one, so extraction must
                            // happen while the frame content is still present.
                            let sel = tui::normalize_selection(a, b);
                            let mut text = String::new();
                            let _ = terminal.draw(|f| {
                                tui::draw(
                                    f,
                                    &log,
                                    Some(input.buf.text()),
                                    input.cursor_col(),
                                    &status,
                                    &mut view,
                                    Some(sel),
                                );
                                // The output area is everything above the
                                // input and status rows.
                                let area = f.area();
                                let area = ratatui::layout::Rect::new(
                                    0,
                                    0,
                                    area.width,
                                    area.height.saturating_sub(2),
                                );
                                text = tui::selection_text(f.buffer_mut(), area, sel);
                            });
                            if !text.trim().is_empty() {
                                tui::copy_to_clipboard(&text);
                            }
                        } else {
                            selection = None;
                        }
                    }
                    _ => {}
                }
                continue;
            }
            if let Event::Paste(pasted) = &ev {
                input.hist_idx = None;
                // An empty bracketed paste means the clipboard holds an image
                // (macOS pastes no text for image content); pasted text that is
                // an image file path attaches that file.
                if IMAGES_ENABLED {
                    if pasted.trim().is_empty() {
                        match crate::imagepaste::from_clipboard() {
                            Some(img) => {
                                log.push_dim(format!(
                                    "[image #{} attached: {}]",
                                    attachments.len() + 1,
                                    img.describe()
                                ));
                                attachments.push(img);
                            }
                            None => log.push_dim("[clipboard has no image to paste]"),
                        }
                        continue;
                    }
                    if let Some(img) = crate::imagepaste::from_path_text(pasted) {
                        log.push_dim(format!(
                            "[image #{} attached: {}]",
                            attachments.len() + 1,
                            img.describe()
                        ));
                        attachments.push(img);
                        continue;
                    }
                }
                // The line editor is single-line; fold pasted newlines into
                // spaces so the paste stays editable.
                input
                    .buf
                    .insert(pasted.replace("\r\n", "\n").replace(['\n', '\r'], " "));
                continue;
            }
            let Event::Key(key) = ev else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            // Any keystroke dismisses the mouse selection highlight (the text
            // was already copied on mouse release).
            selection = None;
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            match key.code {
                KeyCode::Char('c') if ctrl => {
                    if !input.buf.text().is_empty() {
                        input.buf.clear();
                    } else if attachments.is_empty() {
                        log.push_spans(quit_hint_spans());
                    } else {
                        attachments.clear();
                        log.push_dim("[image attachments removed]");
                    }
                }
                KeyCode::Char('d') if ctrl => {
                    if input.buf.text().is_empty() {
                        break;
                    }
                    input.buf.delete();
                }
                KeyCode::Char('u') if ctrl => input.buf.kill_to_start(),
                KeyCode::Char('k') if ctrl => input.buf.kill_to_end(),
                KeyCode::Char('w') if ctrl => input.buf.delete_prev_word(),
                KeyCode::Char('a') if ctrl => input.buf.move_home(),
                KeyCode::Char('e') if ctrl => input.buf.move_end(),
                KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
                    input.hist_idx = None;
                    input.buf.insert(c.to_string());
                }
                KeyCode::Backspace => {
                    input.buf.backspace();
                }
                KeyCode::Delete => {
                    input.buf.delete();
                }
                KeyCode::Left => {
                    input.buf.move_left();
                }
                KeyCode::Right => {
                    input.buf.move_right();
                }
                KeyCode::Home => input.buf.move_home(),
                KeyCode::End => input.buf.move_end(),
                KeyCode::Up => input.history_move(-1),
                KeyCode::Down => input.history_move(1),
                // Esc while idle dismisses a `/btw` panel left open from an
                // earlier turn (the only way it ever closes).
                KeyCode::Esc if btw_panel.is_some() => btw_panel = None,
                KeyCode::Enter => {
                    let line = input.buf.text().trim().to_owned();
                    input.buf.clear();
                    input.hist_idx = None;
                    view.follow = true;
                    if line.is_empty() && attachments.is_empty() {
                        continue;
                    }
                    if !line.is_empty() {
                        input.history.add(&line);
                        input.history.save(&hist_path).ok();
                    }
                    if let Some(cmd) = line.strip_prefix('!') {
                        let cmd = cmd.trim().to_owned();
                        if cmd.is_empty() {
                            log.push_dim("usage: !<shell command>");
                            continue;
                        }
                        log.push_spans(tui::user_echo_spans(&line));
                        Self::tui_bang(
                            &self.tool_ctx.cwd.clone(),
                            &cmd,
                            &mut log,
                            terminal,
                            &mut view,
                        );
                    } else if line.starts_with('/') {
                        if !self.tui_slash(
                            &line,
                            &mut log,
                            terminal,
                            &mut view,
                            &mut input,
                            &mut btw_panel,
                        ) {
                            break;
                        }
                    } else {
                        // The engine is text-only: attach pasted images as
                        // cached-file references the model can open with its
                        // read/bash tools instead of inline content blocks.
                        let mut message = line.clone();
                        for (i, img) in attachments.drain(..).enumerate() {
                            use std::fmt::Write as _;
                            let _ = write!(
                                message,
                                "\n[Attached image #{}: {}{}. Use your tools to view it.]",
                                i + 1,
                                img.describe(),
                                img.source_path.as_deref().map_or(String::new(), |p| {
                                    format!(", original: {}", p.display())
                                })
                            );
                        }
                        let echo = if line.is_empty() { &message } else { &line };
                        log.push_spans(tui::user_echo_spans(echo));
                        self.session.push(Message::user(&message));
                        self.tui_turn(terminal, &mut log, &mut view, &mut input, &mut btw_panel)?;
                    }
                }
                _ => {}
            }
        }
        input.history.save(&hist_path).ok();
        Ok(())
    }

    /// Runs a `!` immediate shell command: output lands only in the TUI log,
    /// never in the conversation, and the model is not consulted. The frame
    /// keeps redrawing while the command runs so Esc/Ctrl-C can kill it.
    fn tui_bang(
        cwd: &std::path::Path,
        cmd: &str,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
    ) {
        let start = Instant::now();
        let mut interrupt = || {
            let status = format!("! {cmd} ({}s, Esc to stop)", start.elapsed().as_secs());
            let _ = terminal.draw(|f| tui::draw(f, log, None, 0, &status, view, None));
            while event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(k)) = event::read()
                    && k.kind == KeyEventKind::Press
                    && (matches!(k.code, KeyCode::Esc)
                        || (matches!(k.code, KeyCode::Char('c'))
                            && k.modifiers.contains(KeyModifiers::CONTROL)))
                {
                    return true;
                }
            }
            false
        };
        match crate::tools::bash::run_immediate(cwd, cmd, &mut interrupt) {
            Ok(out) => {
                for line in out.stdout.lines().chain(out.stderr.lines()) {
                    log.push_dim(line.to_owned());
                }
                if out.interrupted {
                    log.push_dim("[interrupted]");
                } else if out.exit_code != 0 {
                    log.push_dim(format!("[exit code: {}]", out.exit_code));
                }
            }
            Err(e) => log.push_dim(format!("!{cmd}: {e}")),
        }
    }

    /// Idle status line (plain text; the TUI styles the bar itself).
    /// Disk checkpoint path for the system-prompt KV cache.
    fn sysprompt_checkpoint(&self) -> std::path::PathBuf {
        self.store.dir().join("sysprompt.kv")
    }

    /// Warms the system-prompt KV cache at startup, drawing prefill progress.
    fn tui_warm(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
        log: &mut OutputLog,
    ) -> Result<(), String> {
        let checkpoint = self.sysprompt_checkpoint();
        let system = self.system.clone();
        let ctx_size = self.engine.ctx_size();
        let power = self.power_percent;
        let mut announced = false;
        let verb = status::random_verb_index();
        let start = Instant::now();
        let mut view = tui::OutputView::default();
        self.engine
            .warm_system_prompt(&system, Some(&checkpoint), &mut |ev| {
                if let EngineEvent::Prefill(p) = ev {
                    if !announced {
                        announced = true;
                        log.push_spans(vec![ratatui::text::Span::styled(
                            "Updating system prompt cache...",
                            ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
                        )]);
                    }
                    let st = Status {
                        state: WorkerState::Prefill,
                        prefill_done: p.done,
                        prefill_total: p.total,
                        prefill_label: verb,
                        prefill_tps: p.tps,
                        elapsed_secs: start.elapsed().as_secs_f64(),
                        ctx_size,
                        power_percent: power,
                        ..Status::default()
                    };
                    let line = status::build_status_text(&st, false);
                    let _ = terminal.draw(|f| tui::draw(f, log, None, 0, &line, &mut view, None));
                }
            })
            .map_err(|e| e.to_string())?;
        // The cache note is transient: remove it once the warm-up finishes.
        if announced {
            log.pop_line();
            let status = self.idle_status_text();
            let _ = terminal.draw(|f| tui::draw(f, log, None, 0, &status, &mut view, None));
        }
        Ok(())
    }

    /// Warms the system-prompt KV cache for non-TUI runs (stderr message).
    fn warm_plain(&mut self) -> Result<(), String> {
        let checkpoint = self.sysprompt_checkpoint();
        let system = self.system.clone();
        let mut announced = false;
        let color = self.color;
        self.engine
            .warm_system_prompt(&system, Some(&checkpoint), &mut |ev| {
                if matches!(ev, EngineEvent::Prefill(_)) && !announced {
                    announced = true;
                    if color {
                        eprintln!("\x1b[33mUpdating system prompt cache...{ANSI_RESET}");
                    } else {
                        eprintln!("Updating system prompt cache...");
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        // Erase the transient cache note once the warm-up finishes.
        if announced && color {
            eprint!("\x1b[A\x1b[2K\r");
        }
        Ok(())
    }

    fn idle_status_text(&mut self) -> String {
        let rendered = render_transcript(&self.session, &self.system);
        let st = Status {
            state: WorkerState::Idle,
            ctx_used: self.engine.count_tokens(&rendered),
            ctx_size: self.engine.ctx_size(),
            power_percent: self.power_percent,
            ..Status::default()
        };
        status::build_status_text(&st, false)
    }

    /// One TUI turn: runs the generate → tools loop on a worker thread while
    /// the UI thread keeps the terminal live (typing, scrolling, interrupts),
    /// then feeds user lines queued during the turn into follow-up turns.
    fn tui_turn(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
        log: &mut OutputLog,
        view: &mut tui::OutputView,
        input: &mut TuiInput,
        btw: &mut BtwPanel,
    ) -> Result<(), String> {
        // The first iteration runs the main turn; later iterations run either
        // a follow-up turn (leftover queued user lines) or a btw-only drain
        // (side questions queued after the worker's final boundary).
        let mut run_main = true;
        let mut carry_btw: Vec<String> = Vec::new();
        loop {
            let shared = TurnShared::default();
            for q in carry_btw.drain(..) {
                let _ = shared.push_btw(q);
            }
            if run_main {
                run_worker_ui(terminal, log, view, input, btw, &shared, |tx| {
                    self.worker_turn(&tx, &shared)
                })??;
            } else {
                run_worker_ui(terminal, log, view, input, btw, &shared, |tx| {
                    self.drain_btw(&tx, &shared);
                })?;
            }
            // Lines typed while busy that no tool round drained become the
            // next turn's user message(s), as if resubmitted by hand.
            let leftover = shared.take_queued();
            carry_btw = shared.take_btw();
            if leftover.is_empty() && carry_btw.is_empty() {
                return Ok(());
            }
            run_main = !leftover.is_empty();
            for line in leftover {
                self.session.push(Message::user(line));
            }
        }
    }

    /// Worker-side turn loop (the C's `worker_run_turn`): generate, dispatch
    /// tools, drain queued user lines between rounds, repeat until settled.
    /// Runs on the worker thread and talks to the UI only through `tx`.
    fn worker_turn(&mut self, tx: &Sender<UiEvent>, shared: &TurnShared) -> Result<(), String> {
        let mut note = |s: String| {
            let _ = tx.send(UiEvent::Dim(s));
        };
        self.maybe_compact_notify(&mut note)?;
        self.maybe_reminder_notify(&mut note);
        // One clock for the whole turn: elapsed time accumulates across the
        // generate → tools → generate loop instead of restarting per pass.
        let turn_start = Instant::now();
        // Stop hooks run at most once per turn, so a hook that always exits 2
        // cannot loop the model forever.
        let mut stop_hook_ran = false;
        loop {
            let prompt = render_transcript(&self.session, &self.system);
            let out = self.worker_generate(tx, shared, &prompt, turn_start, true)?;
            // A priority `/btw` stopped this pass: nothing was committed, so
            // roll back the partial output, answer the side question(s) now,
            // and re-run the same step from the last committed boundary.
            if out.preempted {
                let _ = tx.send(UiEvent::MainRollback);
                self.drain_btw(tx, shared);
                continue;
            }
            self.session.push(Message::assistant(out.assistant_text));
            let _ = tx.send(UiEvent::EndLine);
            if out.interrupted {
                crate::interrupt::clear();
                let _ = tx.send(UiEvent::Dim("[interrupted]".to_owned()));
                // Drain point 3 (BTW-DESIGN §4.4): the user asked mid-turn;
                // answer even though the main turn was interrupted.
                self.drain_btw(tx, shared);
                return Ok(());
            }
            // Side questions answer at every generation boundary, before the
            // next tool dispatch (BTW-DESIGN §4.4 drain points 1 and 2).
            self.drain_btw(tx, shared);
            if let Some(payload) = out.error {
                self.session.push(Message::user(format!(
                    "<tool_result>{payload}</tool_result>"
                )));
                self.drain_queued(shared, tx);
                continue;
            }
            if !out.calls.is_empty() {
                let observations = dispatch_all(&out.calls, &mut self.tool_ctx);
                for warning in self.tool_ctx.hook_warnings.drain(..) {
                    let _ = tx.send(UiEvent::Dim(warning));
                }
                self.session.push(Message::user(format!(
                    "<tool_result>{observations}</tool_result>"
                )));
                for line in observations.lines() {
                    let _ = tx.send(UiEvent::Dim(line.to_owned()));
                }
                self.drain_queued(shared, tx);
                continue;
            }
            // Stop hooks: exit 2 feeds stderr to the model and the turn
            // continues (at most once).
            if !stop_hook_ran {
                let mut warnings = Vec::new();
                let feedback = self.run_stop_hooks(&mut |w| warnings.push(w));
                for w in warnings {
                    let _ = tx.send(UiEvent::Dim(w));
                }
                if let Some(feedback) = feedback {
                    stop_hook_ran = true;
                    let _ = tx.send(UiEvent::Dim("[Stop hook] continuing the turn".to_owned()));
                    self.session.push(Message::user(format!(
                        "<tool_result>Stop hook feedback:\n{feedback}</tool_result>"
                    )));
                    continue;
                }
            }
            return Ok(());
        }
    }

    /// Answers queued `/btw` side questions FIFO at a generation boundary
    /// (worker thread). Each answer is one tool-free pass over the live
    /// transcript plus the framed question; nothing enters the session and
    /// `last_ctx_used` is restored, so the side exchange is rolled back by
    /// the next real pass's prefix sync. An interrupt during an answer
    /// flushes the rest of the queue (the user is saying "stop the asides");
    /// a failed answer is logged and the queue continues — side questions
    /// must never abort the main turn.
    ///
    /// While answering, `BtwBegin`/`BtwEnd` bracket the render events so the
    /// UI opens a side panel (main conversation 60%, `/btw` 40%). The drain
    /// answers every queued question FIFO and then **returns** — it does not
    /// wait for the panel to be dismissed, so the main task resumes as soon as
    /// the answer is done. `BtwEnd` only stops routing to the panel; the UI
    /// keeps it on screen (frozen, readable) until the user presses Esc.
    fn drain_btw(&mut self, tx: &Sender<UiEvent>, shared: &TurnShared) {
        let Some(mut question) = shared.pop_btw() else {
            return;
        };
        let _ = tx.send(UiEvent::BtwBegin);
        // A stale interrupt (e.g. the preempt path) must not cancel the answer
        // before the user has seen anything.
        shared.interrupt.store(false, Ordering::Relaxed);
        loop {
            let _ = tx.send(UiEvent::UserEcho(format!("/btw {question}")));
            let _ = tx.send(UiEvent::Dim("[btw]".to_owned()));
            let saved_ctx = self.last_ctx_used;
            let mut prompt = render_transcript(&self.session, &self.system);
            {
                use std::fmt::Write as _;
                let _ = write!(prompt, "[user]\n{}\n", btw_user_message(&question));
            }
            match self.worker_generate(tx, shared, &prompt, Instant::now(), false) {
                Ok(out) => {
                    let _ = tx.send(UiEvent::EndLine);
                    self.last_ctx_used = saved_ctx;
                    if out.interrupted {
                        // Esc during a streaming answer: cancel it and flush
                        // the rest of the queue.
                        crate::interrupt::clear();
                        let _ = tx.send(UiEvent::Dim("[interrupted]".to_owned()));
                        let cleared = shared.clear_btw();
                        if cleared > 0 {
                            let _ =
                                tx.send(UiEvent::Dim(format!("[btw queue cleared: {cleared}]")));
                        }
                        break;
                    }
                    if !out.calls.is_empty() || out.error.is_some() {
                        let _ = tx.send(UiEvent::Dim(
                            "(the model tried to call a tool; tools are disabled during /btw — ask in the main conversation)"
                                .to_owned(),
                        ));
                    }
                    let _ = tx.send(UiEvent::Dim(
                        "[btw — not part of the conversation]".to_owned(),
                    ));
                }
                Err(e) => {
                    let _ = tx.send(UiEvent::Dim(format!("/btw failed: {e}")));
                    self.last_ctx_used = saved_ctx;
                }
            }
            let Some(next) = shared.pop_btw() else {
                break;
            };
            question = next;
        }
        // Consume any cancelling Esc so the resumed main task is not itself
        // interrupted by it.
        shared.interrupt.store(false, Ordering::Relaxed);
        crate::interrupt::clear();
        let _ = tx.send(UiEvent::BtwEnd);
    }

    /// Moves user lines queued during the turn into the transcript between
    /// tool rounds, mirroring the C's `queued_user_drain`.
    fn drain_queued(&mut self, shared: &TurnShared, tx: &Sender<UiEvent>) {
        for line in shared.take_queued() {
            let _ = tx.send(UiEvent::Dim("[queued message joined the turn]".to_owned()));
            self.session.push(Message::user(line));
        }
    }

    /// Streams one generation pass on the worker thread, forwarding rendered
    /// output and status snapshots to the UI over `tx`.
    ///
    /// `turn_start` is when the user submitted the prompt: the status bar's
    /// elapsed time counts the whole turn (all generation passes and tool
    /// runs), not just this pass. Tokens/s stays per-pass.
    ///
    /// `is_main` marks a main-task pass (vs. a `/btw` side answer): only main
    /// passes send a `MainCheckpoint` and honor the priority-`/btw` preempt
    /// flag, so a side answer is never interrupted by a queued side question.
    #[allow(clippy::too_many_lines)]
    fn worker_generate(
        &mut self,
        tx: &Sender<UiEvent>,
        shared: &TurnShared,
        prompt: &str,
        turn_start: Instant,
        is_main: bool,
    ) -> Result<TurnOutput, String> {
        // Snapshot the main log before streaming so a preempt can roll back
        // this pass's partial output before it re-runs.
        if is_main {
            let _ = tx.send(UiEvent::MainCheckpoint);
        }
        let mut stream = StreamRenderer::new(ChannelSink(tx.clone()));
        stream.set_preflight(edit_preflight(&self.tool_ctx));
        if !matches!(
            self.cfg.generation.think_mode,
            crate::engine::ThinkMode::Off
        ) {
            stream.begin_in_think();
        }
        // Set when a mid-stream preflight fails: stops the engine early, but
        // is not a user interrupt — the turn loop feeds the error to the model.
        let preflight_stop = AtomicBool::new(false);
        // Mirrors the C's worker greedy flag: argmax sampling while the
        // stream renderer is inside a DSML tool-call stanza.
        let greedy = AtomicBool::new(false);
        let ctx_size = self.engine.ctx_size();
        let power = self.power_percent;
        // Prompt tokens already in context; generated tokens add onto this so
        // the ctx gauge moves while the model streams.
        let prompt_tokens = self.engine.count_tokens(prompt);
        let mut assistant_text = String::new();
        let mut gen_count = 0;
        let verb = status::random_verb_index();
        let start = Instant::now();

        let result = self.engine.generate(
            prompt,
            &self.cfg.generation,
            &|| {
                shared.interrupt.load(Ordering::Relaxed)
                    || (is_main && shared.preempt.load(Ordering::Relaxed))
                    || preflight_stop.load(Ordering::Relaxed)
                    || crate::interrupt::pending()
            },
            &|| greedy.load(Ordering::Relaxed),
            &mut |ev| {
                let status = match ev {
                    EngineEvent::Text(t) => {
                        assistant_text.push_str(&t);
                        stream.push(&t);
                        greedy.store(stream.wants_greedy_sampling(), Ordering::Relaxed);
                        if stream.preflight_error().is_some() {
                            preflight_stop.store(true, Ordering::Relaxed);
                        }
                        gen_count += 1;
                        let secs = start.elapsed().as_secs_f64();
                        Status {
                            state: WorkerState::Generating,
                            generated: gen_count,
                            prefill_label: verb,
                            gen_tps: if secs > 0.0 {
                                f64::from(gen_count) / secs
                            } else {
                                0.0
                            },
                            elapsed_secs: turn_start.elapsed().as_secs_f64(),
                            ctx_used: prompt_tokens + gen_count,
                            ctx_size,
                            power_percent: power,
                            greedy_sampling: greedy.load(Ordering::Relaxed),
                            ..Status::default()
                        }
                    }
                    EngineEvent::Prefill(p) => Status {
                        state: WorkerState::Prefill,
                        prefill_done: p.done,
                        prefill_total: p.total,
                        prefill_label: verb,
                        prefill_tps: p.tps,
                        elapsed_secs: turn_start.elapsed().as_secs_f64(),
                        ctx_used: prompt_tokens,
                        ctx_size,
                        power_percent: power,
                        ..Status::default()
                    },
                };
                let _ = tx.send(UiEvent::Status(status));
            },
        );

        let stats = result.map_err(|e| e.to_string())?;
        self.last_ctx_used = stats.ctx_used;
        stream.finish();
        let finished = stream.finished();
        let calls = finished.calls.to_vec();
        // A preflight stop reads as an engine interrupt, but it is a tool
        // error to feed back to the model, not a user abort.
        let preflight_error = stream.preflight_error();
        let error = preflight_error
            .map(|e| tool_error_payload(true, e))
            .or_else(|| finished.error.map(|e| tool_error_payload(false, e)));
        let user_interrupt = shared.interrupt.load(Ordering::Relaxed);
        // A real interrupt (Esc) takes precedence; only otherwise is a stopped
        // main pass a priority-`/btw` preempt. Preempt is not an error, so a
        // preflight failure never counts as one.
        let preempted = is_main
            && !user_interrupt
            && shared.preempt.load(Ordering::Relaxed)
            && preflight_error.is_none();
        if preempted {
            shared.preempt.store(false, Ordering::Relaxed);
        }
        let interrupted =
            (stats.interrupted || user_interrupt) && !preempted && preflight_error.is_none();
        // Consume the interrupt so a queued follow-up turn starts clean.
        shared.interrupt.store(false, Ordering::Relaxed);
        Ok(TurnOutput {
            interrupted,
            preempted,
            assistant_text,
            calls,
            error,
        })
    }

    /// Compacts before a TUI turn when context is tight; progress lines go to
    /// `note` (the TUI log, or the worker→UI channel during a turn).
    fn maybe_compact_notify(&mut self, note: &mut dyn FnMut(String)) -> Result<(), String> {
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        if !compact::should_compact(self.engine.ctx_size(), used) {
            return Ok(());
        }
        // Cheapest step first: clear old tool-result bodies (no model
        // round-trip) and only fall back to full summarization if still tight.
        if let Some(cleared) = self.try_microcompact() {
            note(format!(
                "microcompacted: cleared {cleared} old tool result(s)"
            ));
            return Ok(());
        }
        self.do_compact_notify("low context", note)
    }

    /// Performs a compaction pass and rebuilds the transcript.
    fn do_compact_notify(
        &mut self,
        reason: &str,
        note: &mut dyn FnMut(String),
    ) -> Result<(), String> {
        note(format!(
            "COMPACTING {reason}: summarizing durable task state..."
        ));
        let mut prompt = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(prompt, "[user]\n{}\n", compact::make_prompt(reason));
        }
        let mut summary = String::new();
        self.engine
            .generate(
                &prompt,
                &self.cfg.generation,
                &|| false,
                &|| false,
                &mut |ev| {
                    if let EngineEvent::Text(t) = ev {
                        summary.push_str(&t);
                    }
                },
            )
            .map_err(|e| e.to_string())?;
        self.rebuild_after_compact(&summary);
        note("context compacted".to_owned());
        Ok(())
    }

    /// Re-injects the system-prompt reminder in the TUI when due.
    fn maybe_reminder_notify(&mut self, note: &mut dyn FnMut(String)) {
        let rendered = render_transcript(&self.session, &self.system);
        let pos = self.engine.count_tokens(&rendered);
        if !self.reminder.should_remind(pos) {
            return;
        }
        note("Re-injecting system prompt reminder...".to_owned());
        self.trace.line(&format!(
            "system prompt reminder injected at transcript={pos}"
        ));
        let mut text = sysprompt::build_system_prompt_reminder(&self.tool_ctx.mcp);
        if !self.cfg.system.is_empty() {
            text.push_str("\nAdditional system instructions reminder:\n");
            text.push_str(&self.cfg.system);
            text.push_str("\n[End additional system instructions reminder.]\n\n");
        }
        self.session.push(Message::user(text));
    }

    /// Runs a `/btw` side question typed while the agent is idle. It reuses
    /// the same `drain_btw` path as a mid-turn `/btw`, so the answer streams
    /// into the (persistent) side panel and the panel stays open afterwards —
    /// dismissed only by Esc — exactly like the busy-time case.
    fn tui_btw(
        &mut self,
        question: &str,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
        input: &mut TuiInput,
        btw: &mut BtwPanel,
    ) -> Result<(), String> {
        let shared = TurnShared::default();
        shared.push_btw(question.to_owned());
        run_worker_ui(terminal, log, view, input, btw, &shared, |tx| {
            self.drain_btw(&tx, &shared);
        })?;
        Ok(())
    }

    /// Handles a slash command in the TUI; returns false to quit.
    #[allow(clippy::too_many_lines)]
    fn tui_slash(
        &mut self,
        line: &str,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
        input: &mut TuiInput,
        btw: &mut BtwPanel,
    ) -> bool {
        let mut parts = line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or(line);
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            "/quit" | "/exit" => return false,
            "/new" | "/clear" => {
                self.session = Session::new();
                self.reminder = SystemPromptReminder::new();
                self.context_content = ContextContent::new();
                let combined = self.context_content.combined();
                self.session.push(Message::user(combined));
                self.last_ctx_used = 0;
                self.checkpoints.clear();
                log.push_plain("started a new session");
            }
            "/checkpoint" => {
                if arg.is_empty() {
                    log.push_ansi(&crate::checkpoint::render_list(
                        &self.checkpoints,
                        now_secs(),
                        true,
                    ));
                } else {
                    log.push_plain(self.checkpoint_create(arg));
                }
            }
            "/rollback" => {
                if arg.is_empty() {
                    log.push_plain("usage: /rollback <name> (see /checkpoint for the list)");
                } else {
                    match self.rollback_to(arg) {
                        Ok(msg) => log.push_plain(msg),
                        Err(e) => log.push_plain(e),
                    }
                }
            }
            "/help" => {
                for line in crate::config::usage().lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/mcp" => log.push_ansi(&render_mcp_report(&self.tool_ctx.mcp, true)),
            "/context" => log.push_ansi(&self.render_context_report(true)),
            "/init" => self.tui_run_init(log, terminal, view, input, btw),
            "/compact" => {
                let result = {
                    let mut note = |s: String| log.push_dim(s);
                    self.do_compact_notify("user request", &mut note)
                };
                if let Err(e) = result {
                    log.push_plain(format!("compact failed: {e}"));
                }
            }
            "/save" => match self.store.save(&mut self.session) {
                Ok(id) => log.push_plain(format!("saved session {}", &id[..8])),
                Err(e) => log.push_plain(format!("save failed: {e}")),
            },
            "/list" => match self.store.list() {
                Ok(entries) => {
                    for line in
                        crate::session::render_session_list(&entries, now_secs(), false).lines()
                    {
                        log.push_plain(line.to_owned());
                    }
                }
                Err(e) => log.push_plain(format!("list failed: {e}")),
            },
            "/switch" => match self.store.load(arg) {
                Ok(s) => {
                    for line in crate::session::render_history(&s.transcript, 6, false).lines() {
                        log.push_plain(line.to_owned());
                    }
                    self.session = s;
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                }
                Err(e) => log.push_plain(format!("switch failed: {e}")),
            },
            "/del" => match self.store.delete(arg) {
                Ok(id) => log.push_plain(format!("deleted session {}", &id[..8])),
                Err(e) => log.push_plain(format!("delete failed: {e}")),
            },
            "/resume" => match self.resume_pick(arg) {
                Ok(None) => match self.store.list() {
                    Ok(entries) => log.push_ansi(&crate::session::render_resume_list(
                        &entries,
                        now_secs(),
                        true,
                        RESUME_LIST_LIMIT,
                    )),
                    Err(e) => log.push_plain(format!("resume failed: {e}")),
                },
                Ok(Some(s)) => {
                    log.push_ansi(&crate::session::render_history(&s.transcript, 6, true));
                    self.session = s;
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                }
                Err(e) => log.push_plain(format!("resume failed: {e}")),
            },
            "/tag" => {
                if arg.is_empty() {
                    if self.session.tag.is_empty() {
                        log.push_plain("no tag set; usage: /tag <text> (\"/tag -\" clears)");
                    } else {
                        log.push_plain(format!("tag: {}", self.session.tag));
                    }
                } else {
                    match self.set_tag(arg) {
                        Ok(msg) => log.push_plain(msg),
                        Err(e) => log.push_plain(format!("tag failed: {e}")),
                    }
                }
            }
            "/history" => {
                let turns = if arg.is_empty() {
                    HISTORY_DEFAULT_TURNS
                } else {
                    arg.parse::<usize>()
                        .unwrap_or(HISTORY_DEFAULT_TURNS)
                        .clamp(1, HISTORY_MAX_TURNS)
                };
                for line in
                    crate::session::render_history(&self.session.transcript, turns, false).lines()
                {
                    log.push_plain(line.to_owned());
                }
            }
            "/power" => match crate::config::parse_power_percent(arg) {
                Some(power) => {
                    self.power_percent = power;
                    log.push_plain(format!("power limit set to {power}%"));
                }
                None => log.push_plain("usage: /power <1..100>"),
            },
            "/strip" => {
                if arg.is_empty() {
                    log.push_plain("usage: /strip <sha-prefix>");
                } else {
                    match self.store.find(arg) {
                        Ok((sha, _)) => {
                            log.push_plain(format!("stripped session {} (0 tokens)", &sha[..8]));
                        }
                        Err(e) => log.push_plain(format!("strip failed: {e}")),
                    }
                }
            }
            "/skills" => {
                for line in crate::skills::render_list(&self.skills).lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/hooks" => {
                for line in crate::hooks::render_list(&self.tool_ctx.hooks).lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/btw" => {
                if arg.is_empty() {
                    log.push_plain("usage: /btw <question>");
                } else if let Err(e) = self.tui_btw(arg, log, terminal, view, input, btw) {
                    log.push_plain(format!("/btw failed: {e}"));
                }
            }
            "/remember" => match remember_from_arg(&self.tool_ctx.cwd, arg) {
                Ok(path) => log.push_dim(format!("[saved to {}]", path.display())),
                Err(e) => {
                    log.push_plain(e);
                    log.push_plain("usage: /remember [user] <text> (default scope: project)");
                }
            },
            "/repro" => match self.write_repro(arg) {
                Ok(path) => log.push_dim(format!("[repro written to {}]", path.display())),
                Err(e) => log.push_plain(format!("repro failed: {e}")),
            },
            "/subagent" => {
                if arg.is_empty() {
                    log.push_plain("usage: /subagent <task>");
                } else {
                    log.push_dim("[subagent started]");
                    let fork_at = self.begin_subagent_fork(arg);
                    if let Err(e) = self.tui_turn(terminal, log, view, input, btw) {
                        // Restore the transcript even when the turn errored.
                        self.finish_subagent_fork(fork_at, arg);
                        log.push_plain(format!("/subagent failed: {e}"));
                    } else if self.finish_subagent_fork(fork_at, arg) {
                        log.push_dim("[subagent report added to the conversation]");
                    } else {
                        log.push_dim("[subagent produced no report — nothing added]");
                    }
                }
            }
            _ if slash_command_known(cmd) => {
                log.push_plain(format!("{cmd}: not implemented yet"));
            }
            _ => {
                if let Some(message) = self.skill_message(cmd, arg) {
                    log.push_spans(tui::user_echo_spans(line));
                    self.session.push(Message::user(message));
                    if let Err(e) = self.tui_turn(terminal, log, view, input, btw) {
                        log.push_plain(format!("{cmd} failed: {e}"));
                    }
                } else {
                    log.push_plain(format!("unknown command: {cmd}"));
                }
            }
        }
        true
    }
}

/// Runs `job` on a scoped worker thread while the UI thread keeps the
/// terminal live (the C's worker/UI split). The worker owns the agent for
/// the duration of the job and reports through the channel; the UI applies
/// events to the log, redraws, and keeps the prompt editable.
#[allow(clippy::too_many_arguments)]
fn run_worker_ui<T: Send>(
    terminal: &mut ratatui::DefaultTerminal,
    log: &mut OutputLog,
    view: &mut tui::OutputView,
    input: &mut TuiInput,
    btw: &mut BtwPanel,
    shared: &TurnShared,
    job: impl FnOnce(Sender<UiEvent>) -> T + Send,
) -> Result<T, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|s| {
        let handle = s.spawn(move || job(tx));
        let ui = busy_ui_loop(terminal, log, view, input, btw, &rx, shared, || {
            handle.is_finished()
        });
        // On a UI error (terminal gone) the worker must still be stopped and
        // joined before the scope can end.
        if ui.is_err() {
            shared.interrupt.store(true, Ordering::Relaxed);
        }
        let out = handle
            .join()
            .map_err(|_| "worker thread panicked".to_owned());
        ui?;
        out
    })
}

/// Handles Esc / Ctrl-C during a worker job, with meaning that depends on the
/// `/btw` panel state:
/// - a side answer is **streaming** (`btw_active`): cancel it (interrupt) and
///   flag the panel to close when its `BtwEnd` arrives;
/// - the panel is **visible but frozen** (main task running behind it): just
///   dismiss the panel, leaving the main task running;
/// - **no panel**: interrupt the main task, as before.
fn close_or_interrupt(
    shared: &TurnShared,
    btw: &mut Option<(OutputLog, tui::OutputView)>,
    btw_active: bool,
    close_panel_on_end: &mut bool,
) {
    if btw_active {
        shared.interrupt.store(true, Ordering::Relaxed);
        *close_panel_on_end = true;
    } else if btw.is_some() {
        *btw = None;
    } else {
        shared.interrupt.store(true, Ordering::Relaxed);
    }
}

/// UI-thread event loop while a worker job runs: applies streamed render
/// events to the log, keeps the prompt editable (Enter queues the line for
/// the worker), scrolls, and maps Esc/Ctrl-C to a worker interrupt.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn busy_ui_loop(
    terminal: &mut ratatui::DefaultTerminal,
    log: &mut OutputLog,
    view: &mut tui::OutputView,
    input: &mut TuiInput,
    btw: &mut BtwPanel,
    rx: &Receiver<UiEvent>,
    shared: &TurnShared,
    done: impl Fn() -> bool,
) -> Result<(), String> {
    let mut status_line = String::new();
    // The `/btw` side panel (`btw`) is owned by `tui_loop`, so it survives
    // across turns: it opens on the first BtwBegin and stays up — even after
    // the answer finishes, the main task resumes, and the whole turn ends —
    // until the user dismisses it with Esc. `btw_active` is true only while a
    // side answer is actually streaming: it gates whether render events go to
    // the panel or to the main log, which is what lets the main task keep
    // rendering on the left while a finished answer stays frozen on the right.
    let mut btw_active = false;
    // Set when Esc is pressed mid-answer: the panel is torn down once the
    // cancelled answer's BtwEnd arrives (so late btw tokens don't leak).
    let mut close_panel_on_end = false;
    // Main-log length at the start of the current main pass; a preempting
    // `/btw` truncates back to it so the discarded partial output does not
    // duplicate when the pass re-runs.
    let mut main_checkpoint = 0usize;
    loop {
        while let Ok(ev) = rx.try_recv() {
            match ev {
                UiEvent::Status(st) => status_line = status::build_status_text(&st, false),
                UiEvent::BtwBegin => {
                    if btw.is_none() {
                        *btw = Some((OutputLog::new(), tui::OutputView::default()));
                    }
                    btw_active = true;
                }
                UiEvent::BtwEnd => {
                    btw_active = false;
                    if close_panel_on_end {
                        *btw = None;
                        close_panel_on_end = false;
                    }
                }
                // Checkpoint/rollback always act on the main log, regardless
                // of a live side panel (a preempt fires only in a main pass).
                UiEvent::MainCheckpoint => main_checkpoint = log.checkpoint(),
                UiEvent::MainRollback => {
                    log.truncate_to(main_checkpoint);
                    view.follow = true;
                }
                // Route to the panel only while an answer is streaming; once
                // it finishes the main task's output goes to the main log even
                // though the (frozen) panel is still visible.
                ev => {
                    if let (true, Some((btw_log, _))) = (btw_active, btw.as_mut()) {
                        worker::apply(btw_log, ev);
                    } else {
                        worker::apply(log, ev);
                    }
                }
            }
        }
        // Check before drawing: anything sent after this point survives in
        // the channel and is drained below (the sender is gone once the
        // worker returns).
        let finished = done();
        terminal
            .draw(|f| {
                if let Some((btw_log, btw_view)) = btw.as_mut() {
                    tui::draw_btw_split(
                        f,
                        log,
                        btw_log,
                        btw_view,
                        Some(input.buf.text()),
                        input.cursor_col(),
                        &status_line,
                        view,
                    );
                } else {
                    tui::draw(
                        f,
                        log,
                        Some(input.buf.text()),
                        input.cursor_col(),
                        &status_line,
                        view,
                        None,
                    );
                }
            })
            .map_err(|e| e.to_string())?;
        if finished {
            // The worker is done (turn over); drain the tail in order. The
            // panel is discarded when this function returns, so late btw
            // events just stop mattering.
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    UiEvent::Status(_) | UiEvent::MainCheckpoint => {}
                    UiEvent::BtwBegin => btw_active = true,
                    UiEvent::BtwEnd => btw_active = false,
                    UiEvent::MainRollback => log.truncate_to(main_checkpoint),
                    ev => {
                        if let (true, Some((btw_log, _))) = (btw_active, btw.as_mut()) {
                            worker::apply(btw_log, ev);
                        } else {
                            worker::apply(log, ev);
                        }
                    }
                }
            }
            return Ok(());
        }
        if !event::poll(Duration::from_millis(100)).map_err(|e| e.to_string())? {
            continue;
        }
        match event::read().map_err(|e| e.to_string())? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                match key.code {
                    KeyCode::Esc => {
                        close_or_interrupt(shared, btw, btw_active, &mut close_panel_on_end);
                    }
                    KeyCode::Char('c') if ctrl => {
                        // Ctrl-C clears a partly-typed line first; on an empty
                        // line it acts like Esc (cancel answer / close panel /
                        // interrupt the model).
                        if input.buf.text().is_empty() {
                            close_or_interrupt(shared, btw, btw_active, &mut close_panel_on_end);
                        } else {
                            input.buf.clear();
                        }
                    }
                    KeyCode::Enter => {
                        let line = input.buf.text().trim().to_owned();
                        input.buf.clear();
                        input.hist_idx = None;
                        if line.is_empty() {
                        } else if btw_question(&line).is_some() {
                            // A `/btw` gets priority: it preempts the running
                            // main pass so the side question is answered now,
                            // then the pass re-runs. Only when a side answer is
                            // already streaming (btw_active) is the main task
                            // paused already, so the question just joins the
                            // FIFO queue; a merely-visible frozen panel does
                            // not — the main task is running behind it, so a
                            // new `/btw` preempts it.
                            let question = btw_question(&line).unwrap_or_default().to_owned();
                            input.history.add(&line);
                            if let Some(dropped) = shared.push_btw(question) {
                                log.push_dim(format!(
                                    "[btw queue full — dropped oldest: {dropped}]"
                                ));
                            }
                            if btw_active {
                                log.push_dim("[/btw — answers next in the panel]");
                            } else {
                                shared.preempt.store(true, Ordering::Relaxed);
                                log.push_dim("[/btw — pausing the task to answer now]");
                            }
                            view.follow = true;
                        } else if line.starts_with('/') || line.starts_with('!') {
                            log.push_dim("[commands don't queue — wait for the model to finish]");
                        } else {
                            input.history.add(&line);
                            log.push_spans(tui::user_echo_spans(&line));
                            log.push_dim("[queued — joins the conversation at the next step]");
                            shared.push_queued(line);
                            view.follow = true;
                        }
                    }
                    KeyCode::Char('u') if ctrl => input.buf.kill_to_start(),
                    KeyCode::Char('k') if ctrl => input.buf.kill_to_end(),
                    KeyCode::Char('w') if ctrl => input.buf.delete_prev_word(),
                    KeyCode::Char('a') if ctrl => input.buf.move_home(),
                    KeyCode::Char('e') if ctrl => input.buf.move_end(),
                    KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
                        input.hist_idx = None;
                        input.buf.insert(c.to_string());
                    }
                    KeyCode::Backspace => {
                        input.buf.backspace();
                    }
                    KeyCode::Delete => {
                        input.buf.delete();
                    }
                    KeyCode::Left => {
                        input.buf.move_left();
                    }
                    KeyCode::Right => {
                        input.buf.move_right();
                    }
                    KeyCode::Home => input.buf.move_home(),
                    // End resumes scroll-follow on an empty line, otherwise
                    // it is the usual end-of-line motion.
                    KeyCode::End => {
                        if input.buf.text().is_empty() {
                            view.follow = true;
                        } else {
                            input.buf.move_end();
                        }
                    }
                    KeyCode::Up => input.history_move(-1),
                    KeyCode::Down => input.history_move(1),
                    _ => {}
                }
            }
            Event::Paste(pasted) => {
                input.hist_idx = None;
                // The line editor is single-line; fold pasted newlines into
                // spaces so the paste stays editable.
                input
                    .buf
                    .insert(pasted.replace("\r\n", "\n").replace(['\n', '\r'], " "));
            }
            Event::Mouse(m) => match m.kind {
                MouseEventKind::ScrollUp => {
                    view.follow = false;
                    view.top = view.top.saturating_sub(3);
                }
                MouseEventKind::ScrollDown => {
                    // Clamped by draw, which re-enters follow mode at the bottom.
                    view.top = view.top.saturating_add(3);
                }
                _ => {}
            },
            _ => {}
        }
    }
}

fn print_footer(st: &Status, color: bool) {
    let line = status::build_status_text(st, color);
    if color {
        println!(
            "{}{line}{}",
            status::STATUS_STYLE_START,
            status::STATUS_STYLE_END
        );
    } else {
        println!("{line}");
    }
}

fn new_agent(
    engine: Box<dyn Engine>,
    cfg: &AgentConfig,
    show_footer: bool,
) -> Result<Agent<'_>, String> {
    let store = SessionStore::open(SessionStore::default_dir()).map_err(|e| e.to_string())?;
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let mut trace = Trace::open(cfg.trace_path.as_deref()).map_err(|e| e.to_string())?;
    let mut session = Session::new();
    // Collect context at session start
    let context_content = ContextContent::new();
    // Inject context into the session transcript
    let combined = context_content.combined();
    trace.text("context", &combined);
    session.push(Message::user(combined));
    let mut tool_ctx = ToolContext::new(cwd);
    // Start MCP servers before composing the system prompt so their tool
    // schemas land in it, like agent_worker_init.
    tool_ctx.mcp = crate::tools::mcp::load_and_start(cfg.mcp_config_path.as_deref());
    tool_ctx.hooks = crate::hooks::load_default(&tool_ctx.cwd);
    tool_ctx.sandbox = crate::sandbox::load_default(&tool_ctx.cwd);
    if let Some(enabled) = cfg.sandbox_override {
        tool_ctx.sandbox.enabled = enabled;
    }
    if show_footer {
        // Interactive approval for web access, like agent_web_confirm;
        // headless runs keep the auto-deny default.
        tool_ctx.web_confirm = Some(Box::new(|message: &str| {
            print!("{message}");
            let _ = std::io::stdout().flush();
            let mut answer = String::new();
            if std::io::stdin().read_line(&mut answer).is_err() {
                return false;
            }
            matches!(answer.trim(), "y" | "Y" | "yes")
        }));
    }
    let system = sysprompt::build_system_prompt(&cfg.system, &tool_ctx.mcp);
    let skills = crate::skills::load_default(&tool_ctx.cwd);
    Ok(Agent {
        engine,
        cfg,
        session,
        store,
        tool_ctx,
        system,
        reminder: SystemPromptReminder::new(),
        power_percent: 0,
        trace,
        color: std::io::stdout().is_terminal(),
        show_footer,
        editor_owns_footer: false,
        last_ctx_used: 0,
        context_content,
        skills,
        checkpoints: crate::checkpoint::CheckpointStore::new(),
    })
}

/// Runs the interactive REPL until the user exits.
///
/// # Errors
/// Returns an error string on unrecoverable I/O or engine failure.
pub fn run_interactive(engine: Box<dyn Engine>, cfg: &AgentConfig) -> Result<(), String> {
    let mut agent = new_agent(engine, cfg, true)?;

    // `plank /resume [prefix]` loads a prior session before the loop starts.
    if let Some(arg) = &cfg.resume {
        agent.resume_from_cli(arg)?;
    }

    // A real terminal gets the full-screen ratatui UI (works cleanly in Warp
    // and other block terminals via the alternate screen). Piped input falls
    // back to the plain line REPL.
    let result = if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        agent.run_tui()
    } else {
        run_plain_flow(&mut agent, cfg)
    };
    // Whatever happened, save the session and tell the user how to resume it.
    agent.report_session_on_exit();
    result
}

/// Plain-REPL session flow: warm the cache, run the one-shot `-p` prompt if
/// any, then read lines until EOF.
fn run_plain_flow(agent: &mut Agent<'_>, cfg: &AgentConfig) -> Result<(), String> {
    agent.warm_plain()?;
    if let Some(history) = agent.resumed_history() {
        print!("{history}");
    }
    if let Some(initial) = cfg.prompt.as_deref().filter(|p| !p.is_empty()) {
        print!("{}", status::format_user_prompt_echo(initial, agent.color));
        agent.session.push(Message::user(initial));
        agent.run_turn()?;
    }
    run_repl_plain(agent)
}

/// Yellow hint shown when Ctrl-C is pressed on an empty idle prompt.
fn quit_hint_spans() -> Vec<ratatui::text::Span<'static>> {
    vec![ratatui::text::Span::styled(
        "Press Ctrl-D to quit.",
        ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
    )]
}

/// Plain line-based REPL used when stdin is not a terminal.
fn run_repl_plain(agent: &mut Agent<'_>) -> Result<(), String> {
    let stdin = std::io::stdin();
    loop {
        print!("{}", status::prompt_text());
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        let n = stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(()); // EOF
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input.starts_with('/') {
            if !agent.slash(input)? {
                return Ok(());
            }
            continue;
        }
        if let Some(cmd) = input.strip_prefix('!') {
            let cmd = cmd.trim();
            if cmd.is_empty() {
                println!("usage: !<shell command>");
                continue;
            }
            match crate::tools::bash::run_immediate(&agent.tool_ctx.cwd, cmd, &mut || {
                crate::interrupt::pending()
            }) {
                Ok(out) => {
                    print!("{}", out.stdout);
                    eprint!("{}", out.stderr);
                    if out.interrupted {
                        crate::interrupt::clear();
                        println!("[interrupted]");
                    } else if out.exit_code != 0 {
                        println!("[exit code: {}]", out.exit_code);
                    }
                }
                Err(e) => println!("!{cmd}: {e}"),
            }
            continue;
        }
        print!("{}", status::format_user_prompt_echo(input, agent.color));
        agent.session.push(Message::user(input));
        agent.run_turn()?;
    }
}

/// Runs headless mode: one-shot with `-p`, else a stdin-driven protocol.
///
/// # Errors
/// Returns an error string on unrecoverable I/O or engine failure.
pub fn run_non_interactive(engine: Box<dyn Engine>, cfg: &AgentConfig) -> Result<(), String> {
    let mut agent = new_agent(engine, cfg, false)?;
    agent.warm_plain()?;
    if let Some(prompt) = cfg.prompt.as_deref() {
        agent.session.push(Message::user(prompt));
        return agent.run_turn();
    }
    // Stdin protocol, like the C: announce readiness on stderr, collect bytes
    // until stdin has been quiet for 200 ms, submit that buffer as one prompt,
    // repeat until EOF. (The C also queues input arriving mid-generation; the
    // synchronous port reads between turns instead.)
    let mut eof = false;
    while !eof {
        eprintln!("+DWARFSTAR_WAITING");
        let Some(prompt) = read_quiet_batched(&mut eof).map_err(|e| e.to_string())? else {
            break;
        };
        if prompt.trim().is_empty() {
            continue;
        }
        agent.session.push(Message::user(prompt.trim_end()));
        agent.run_turn()?;
    }
    Ok(())
}

/// Reads one stdin batch: bytes accumulated until a 200 ms quiet window.
///
/// Returns `None` at EOF with nothing buffered; sets `eof` once stdin closes.
fn read_quiet_batched(eof: &mut bool) -> std::io::Result<Option<String>> {
    use std::io::Read as _;
    const QUIET_MS: i32 = 200;
    let mut buf = Vec::new();
    loop {
        let timeout = if buf.is_empty() { -1 } else { QUIET_MS };
        let mut pfd = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pfd points to a valid pollfd for the duration of the call.
        let rc = unsafe { libc::poll(&raw mut pfd, 1, timeout) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if rc == 0 {
            // Quiet window elapsed with data buffered: submit it.
            break;
        }
        let mut chunk = [0_u8; 4096];
        let n = std::io::stdin().read(&mut chunk)?;
        if n == 0 {
            *eof = true;
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    if buf.is_empty() {
        return Ok(None);
    }
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineError, EngineEvent, GenerationStats};

    /// Engine that plays back canned replies in order. Records the prompt of
    /// every generate call in `prompts` (shared, so tests can inspect it
    /// after the engine moves into the Agent) and reports the pass at index
    /// `interrupt_at` as user-interrupted.
    #[derive(Debug, Default)]
    struct ScriptedEngine {
        replies: Vec<String>,
        next: usize,
        prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        interrupt_at: Option<usize>,
    }

    impl Engine for ScriptedEngine {
        fn generate(
            &mut self,
            transcript: &str,
            _opts: &crate::engine::GenerationOptions,
            _interrupt: &dyn Fn() -> bool,
            _greedy: &dyn Fn() -> bool,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<GenerationStats, EngineError> {
            self.prompts.lock().unwrap().push(transcript.to_owned());
            let interrupted = self.interrupt_at == Some(self.next);
            let reply = self.replies.get(self.next).cloned().unwrap_or_default();
            self.next += 1;
            // Stream in small chunks to exercise partial-marker handling.
            let mut i = 0;
            while i < reply.len() {
                let mut end = (i + 7).min(reply.len());
                while !reply.is_char_boundary(end) {
                    end += 1;
                }
                on_event(EngineEvent::Text(reply[i..end].to_string()));
                i = end;
            }
            Ok(GenerationStats {
                interrupted,
                ..GenerationStats::default()
            })
        }
        fn ctx_size(&self) -> i32 {
            100_000
        }
    }

    /// Builds an Agent over a scripted engine with the standard test fields.
    fn test_agent<'a>(
        dir: &std::path::Path,
        engine: ScriptedEngine,
        cfg: &'a crate::config::AgentConfig,
    ) -> Agent<'a> {
        Agent {
            engine: Box::new(engine),
            cfg,
            session: Session::new(),
            store: SessionStore::open(dir).unwrap(),
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[]),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
        }
    }

    fn test_cfg() -> crate::config::AgentConfig {
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        cfg
    }

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-ui-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn btw_question_parses_with_boundaries() {
        assert_eq!(btw_question("/btw what is x?"), Some("what is x?"));
        assert_eq!(btw_question("/btw  why?"), Some("why?"));
        assert_eq!(btw_question("/btw: colon form"), Some("colon form"));
        assert_eq!(btw_question("/btwfoo nope"), None);
        assert_eq!(btw_question("/side why?"), None);
        assert_eq!(btw_question("/btw"), None);
        assert_eq!(btw_question("/btw   "), None);
        assert_eq!(btw_question("plain text"), None);
    }

    #[test]
    fn btw_drain_leaves_transcript_untouched() {
        let dir = scratch_dir("btw-clean");
        let prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
        let engine = ScriptedEngine {
            replies: vec!["It is 42.\n".to_string()],
            prompts: prompts.clone(),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main question"));
        agent.session.push(Message::assistant("main answer"));
        agent.last_ctx_used = 1234;
        let before = agent.session.transcript.clone();

        let shared = TurnShared::default();
        shared.push_btw("what was the answer?".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        agent.drain_btw(&tx, &shared);
        drop(tx);

        assert_eq!(agent.session.transcript.len(), before.len());
        assert_eq!(agent.last_ctx_used, 1234);
        let recorded = prompts.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert!(recorded[0].contains("side question"), "framing missing");
        assert!(recorded[0].contains("what was the answer?"));
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::UserEcho(t) if t == "/btw what was the answer?"))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::Dim(t) if t == "[btw]"))
        );
        assert!(
            events.iter().any(
                |e| matches!(e, UiEvent::Dim(t) if t.contains("not part of the conversation"))
            )
        );
        // The panel is bracketed exactly once, BtwBegin before the echo and
        // BtwEnd after the trailer, so the UI splits and tears down cleanly.
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, UiEvent::BtwBegin))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, UiEvent::BtwEnd))
                .count(),
            1
        );
        let begin = events
            .iter()
            .position(|e| matches!(e, UiEvent::BtwBegin))
            .unwrap();
        let end = events
            .iter()
            .position(|e| matches!(e, UiEvent::BtwEnd))
            .unwrap();
        let echo = events
            .iter()
            .position(|e| matches!(e, UiEvent::UserEcho(_)))
            .unwrap();
        assert!(begin < echo && echo < end, "panel must bracket the answer");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn btw_denies_tools() {
        let dir = scratch_dir("btw-tools");
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo nope</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![stanza.to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main"));
        let before = agent.session.transcript.len();

        let shared = TurnShared::default();
        shared.push_btw("run something".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        agent.drain_btw(&tx, &shared);
        drop(tx);

        // No dispatch and no tool result: transcript untouched.
        assert_eq!(agent.session.transcript.len(), before);
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events.iter().any(
                |e| matches!(e, UiEvent::Dim(t) if t.contains("tools are disabled during /btw"))
            )
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn btw_answers_at_mid_turn_boundary_and_stays_out_of_main_prompts() {
        let dir = scratch_dir("btw-boundary");
        let stanza = concat!(
            "Working.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo hi</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
        let engine = ScriptedEngine {
            replies: vec![
                stanza.to_string(),
                "The answer is 7.\n".to_string(), // side answer at the boundary
                "Done.\n".to_string(),            // main continuation
            ],
            prompts: prompts.clone(),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("do the task"));

        let shared = TurnShared::default();
        shared.push_btw("what is 3+4?".to_owned());
        let (tx, _rx) = std::sync::mpsc::channel();
        agent.worker_turn(&tx, &shared).unwrap();
        drop(tx);

        let recorded = prompts.lock().unwrap();
        assert_eq!(recorded.len(), 3, "main, side, main continuation");
        // The side prompt sees the completed pass (stanza already in the
        // transcript) but runs before the tool dispatch.
        assert!(recorded[1].contains("what is 3+4?"));
        assert!(recorded[1].contains("Working."));
        assert!(!recorded[1].contains("<tool_result>"));
        // The main continuation never sees the side exchange.
        assert!(recorded[2].contains("<tool_result>"));
        assert!(!recorded[2].contains("what is 3+4?"));
        assert!(!recorded[2].contains("The answer is 7."));
        // Nothing side-channel entered the session.
        let flat: String = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.as_str())
            .collect();
        assert!(!flat.contains("what is 3+4?"));
        assert!(!flat.contains("The answer is 7."));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn preempting_btw_rolls_back_the_pass_and_reruns_after_answering() {
        let dir = scratch_dir("btw-preempt");
        let prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
        let engine = ScriptedEngine {
            replies: vec![
                "PARTIAL main output that gets discarded\n".to_string(), // preempted pass
                "The answer is Rust.\n".to_string(),                     // side answer
                "Final main answer.\n".to_string(),                      // re-run pass
            ],
            prompts: prompts.clone(),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("do the task"));

        // A /btw queued with the preempt flag set (as the UI does mid-pass).
        let shared = TurnShared::default();
        shared.push_btw("what language?".to_owned());
        shared.preempt.store(true, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        agent.worker_turn(&tx, &shared).unwrap();
        drop(tx);

        // Three passes ran: preempted main, side answer, re-run main.
        let recorded = prompts.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        // Nothing was committed before the preempt, so the re-run's prompt is
        // byte-identical to the preempted pass's prompt.
        assert_eq!(recorded[0], recorded[2]);
        assert!(recorded[1].contains("side question"));
        assert!(recorded[1].contains("what language?"));

        // The discarded partial never reached the transcript; only the re-run
        // answer did, and the side exchange stayed out entirely.
        assert_eq!(agent.session.transcript.len(), 2);
        assert_eq!(
            agent.session.transcript[1].text.trim(),
            "Final main answer."
        );
        let flat: String = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.as_str())
            .collect();
        assert!(!flat.contains("PARTIAL"));
        assert!(!flat.contains("The answer is Rust."));
        assert!(!flat.contains("what language?"));

        // The UI was told to roll back the main log, and a panel bracketed
        // the priority answer. The preempt flag is consumed, so the re-run
        // (and future turns) are not stuck preempting.
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(events.iter().any(|e| matches!(e, UiEvent::MainRollback)));
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwBegin)));
        assert!(!shared.preempt.load(Ordering::Relaxed));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn drain_btw_returns_promptly_so_the_main_task_resumes() {
        let dir = scratch_dir("btw-nonblock");
        let engine = ScriptedEngine {
            replies: vec!["It is Rust.\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main"));

        let shared = TurnShared::default();
        shared.push_btw("what language?".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        // No external signal is provided: if the drain parked waiting for the
        // panel to be dismissed, this would hang. It must return on its own.
        agent.drain_btw(&tx, &shared);
        drop(tx);

        // BtwEnd only ends the active answer (the UI keeps the panel visible);
        // the drain still returns, letting the main task resume.
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwBegin)));
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwEnd)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_from_cli_loads_a_saved_session_by_prefix_and_most_recent() {
        let dir = scratch_dir("resume-cli");
        let cfg = test_cfg();

        // Save a session, capture its id.
        let mut a = test_agent(&dir, ScriptedEngine::default(), &cfg);
        a.session.push(Message::user("remember the alamo"));
        let id = a.store.save(&mut a.session).unwrap();

        // A fresh agent resumes it by sha prefix.
        let mut b = test_agent(&dir, ScriptedEngine::default(), &cfg);
        assert!(b.resumed_history().is_none(), "fresh session: no history");
        b.resume_from_cli(&id[..8]).unwrap();
        assert_eq!(b.session.id, id);
        assert!(
            b.session
                .transcript
                .iter()
                .any(|m| m.text == "remember the alamo")
        );
        let history = b.resumed_history().expect("resumed session shows history");
        assert!(history.contains("resumed session"));

        // A fresh agent with an empty arg resumes the most recent session.
        let mut c = test_agent(&dir, ScriptedEngine::default(), &cfg);
        c.resume_from_cli("").unwrap();
        assert_eq!(c.session.id, id);

        // An unknown prefix is a clean error, not a panic.
        let mut d = test_agent(&dir, ScriptedEngine::default(), &cfg);
        assert!(d.resume_from_cli("nonexistent0").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_for_exit_persists_a_used_session_and_skips_an_empty_one() {
        let dir = scratch_dir("exit-save");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);

        // An empty session (no user turn) has nothing worth saving.
        assert!(agent.save_for_exit().is_none());

        // After a real user turn it saves, returns the id + existing path, and
        // stamps the session id so a resume can find it.
        agent.session.push(Message::user("hello there"));
        let (id, path) = agent.save_for_exit().expect("used session should save");
        assert!(!id.is_empty());
        assert!(path.exists(), "session file written: {}", path.display());
        assert_eq!(agent.session.id, id);
        // The id resolves through the store, which is what `/resume <id>` uses.
        assert!(agent.store.find(&id[..8]).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn esc_cancels_a_streaming_answer_and_defers_the_panel_close() {
        let shared = TurnShared::default();
        let mut p: BtwPanel = Some((OutputLog::new(), tui::OutputView::default()));
        let mut close_pending = false;
        close_or_interrupt(&shared, &mut p, true, &mut close_pending);
        // Cancel the answer now; the panel is torn down later on its BtwEnd.
        assert!(shared.interrupt.load(Ordering::Relaxed));
        assert!(close_pending);
        assert!(p.is_some());
    }

    #[test]
    fn esc_on_a_frozen_panel_dismisses_it_without_interrupting_the_task() {
        let shared = TurnShared::default();
        let mut p: BtwPanel = Some((OutputLog::new(), tui::OutputView::default()));
        let mut close_pending = false;
        close_or_interrupt(&shared, &mut p, false, &mut close_pending);
        assert!(
            !shared.interrupt.load(Ordering::Relaxed),
            "task keeps running"
        );
        assert!(p.is_none(), "panel dismissed");
        assert!(!close_pending);
    }

    #[test]
    fn esc_with_no_panel_interrupts_the_task() {
        let shared = TurnShared::default();
        let mut p: BtwPanel = None;
        let mut close_pending = false;
        close_or_interrupt(&shared, &mut p, false, &mut close_pending);
        assert!(shared.interrupt.load(Ordering::Relaxed));
        assert!(p.is_none());
    }

    #[test]
    fn btw_interrupt_flushes_remaining_queue() {
        let dir = scratch_dir("btw-flush");
        let engine = ScriptedEngine {
            replies: vec!["partial".to_string()],
            interrupt_at: Some(0),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main"));
        agent.last_ctx_used = 77;

        let shared = TurnShared::default();
        shared.push_btw("first".to_owned());
        shared.push_btw("second".to_owned());
        shared.push_btw("third".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        agent.drain_btw(&tx, &shared);
        drop(tx);

        assert!(shared.pop_btw().is_none(), "queue must be flushed");
        assert_eq!(agent.last_ctx_used, 77);
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::Dim(t) if t == "[btw queue cleared: 2]"))
        );
        // The panel is torn down even on the interrupt path, so the split
        // never lingers after the user cancels.
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwEnd)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn malformed_stanza_feeds_c_format_tool_error() {
        let dir = std::env::temp_dir().join(format!("plank-ui-err-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptedEngine {
            replies: vec![
                // Legal opener, then a bogus tag the strict parser rejects.
                "<｜DSML｜tool_calls><b>".to_string(),
                "Understood.\n".to_string(),
            ],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[]),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
        };
        agent.session.push(Message::user("go"));
        agent.run_turn().unwrap();

        // user, assistant(bad stanza), user(tool error), assistant(final)
        let tool_result = &agent.session.transcript[2].text;
        assert!(
            tool_result.contains("Tool error: invalid DSML tool call: unexpected DSML tag: <b>\n"),
            "got: {tool_result}"
        );
        assert!(
            tool_result.contains("DSML syntax reminder:"),
            "got: {tool_result}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn subagent_fork_truncates_and_carries_only_the_report() {
        let dir = std::env::temp_dir().join(format!("plank-ui-sub-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stanza = concat!(
            "Counting.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo 42</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![stanza.to_string(), "There are 42 tests.\n".to_string()],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[]),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
        };
        agent.session.push(Message::user("hi"));
        agent.session.push(Message::assistant("hello"));

        let fork_at = agent.begin_subagent_fork("count the tests");
        assert_eq!(fork_at, 2);
        agent.run_turn().unwrap();
        // Fork grew: task, assistant(tool call), tool result, final report.
        assert!(agent.session.transcript.len() > 4);

        assert!(agent.finish_subagent_fork(fork_at, "count the tests"));
        // Parent keeps its two messages plus only the framed report.
        assert_eq!(agent.session.transcript.len(), 3);
        let report = &agent.session.transcript[2].text;
        assert!(report.contains("Subagent report:"), "got: {report}");
        assert!(report.contains("There are 42 tests."), "got: {report}");
        assert!(!report.contains("echo 42"), "sidechain leaked: {report}");

        // A fork with no assistant output restores the transcript untouched.
        let fork_at = agent.begin_subagent_fork("noop");
        assert!(!agent.finish_subagent_fork(fork_at, "noop"));
        assert_eq!(agent.session.transcript.len(), 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn turn_loop_executes_tool_calls_and_finishes() {
        let dir = std::env::temp_dir().join(format!("plank-ui-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stanza = concat!(
            "I'll run a command.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo plank-e2e</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![
                stanza.to_string(),
                "The command printed plank-e2e.\n".to_string(),
            ],
            ..ScriptedEngine::default()
        };
        let cfg = crate::config::AgentConfig::default();
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[]),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
        };
        agent.session.push(Message::user("run echo"));
        agent.run_turn().unwrap();

        // user, assistant(tool call), user(tool result), assistant(final)
        assert_eq!(agent.session.transcript.len(), 4);
        let tool_result = &agent.session.transcript[2].text;
        assert!(tool_result.contains("plank-e2e"), "got: {tool_result}");
        assert!(tool_result.starts_with("<tool_result>"));
        let last = &agent.session.transcript[3].text;
        assert!(last.contains("The command printed"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn worker_turn_drains_queued_user_between_tool_rounds() {
        let dir = std::env::temp_dir().join(format!("plank-ui-queue-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stanza = concat!(
            "Checking.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo hi</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![stanza.to_string(), "Done.\n".to_string()],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[]),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
        };
        agent.session.push(Message::user("run echo"));

        // A line "typed while busy": queued before the turn, so the first
        // tool round must drain it into the transcript.
        let shared = TurnShared::default();
        shared.push_queued("also check the docs".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        agent.worker_turn(&tx, &shared).unwrap();
        drop(tx);

        // user, assistant(tool call), user(tool result), user(queued),
        // assistant(final)
        assert_eq!(agent.session.transcript.len(), 5);
        assert!(
            agent.session.transcript[2]
                .text
                .starts_with("<tool_result>")
        );
        assert_eq!(agent.session.transcript[3].text, "also check the docs");
        assert!(agent.session.transcript[4].text.contains("Done."));
        assert!(shared.take_queued().is_empty());

        // The UI channel saw rendered text, the drain notice, and status
        // snapshots from generation.
        let events: Vec<UiEvent> = rx.try_iter().collect();
        let visible: String = events
            .iter()
            .filter_map(|e| match e {
                UiEvent::Visible(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(visible.contains("Checking"), "got: {visible}");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::Dim(t) if t.contains("queued message joined")))
        );
        assert!(events.iter().any(|e| matches!(e, UiEvent::Status(_))));
        std::fs::remove_dir_all(&dir).ok();
    }
}
