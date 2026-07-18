//! Interactive REPL and headless front-ends over the agent turn loop.
//!
//! Port of the "Interactive Runtime Loop" section of `ds4_agent.c`, adapted
//! to a synchronous turn loop: the C multiplexes a worker thread with `poll()`;
//! plank v1 runs the engine inline and streams output as it arrives.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};

use crate::compact;
use crate::config::{AgentConfig, slash_command_known};
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
}

/// Default number of user turns replayed by `/history`.
const HISTORY_DEFAULT_TURNS: usize = 3;
/// Maximum user turns `/history` accepts.
const HISTORY_MAX_TURNS: usize = 200;

impl Agent<'_> {
    /// Streams one generation pass: paints the live status bar for prefill and
    /// generation, and routes model text through the viz + markdown pipeline.
    #[allow(clippy::type_complexity)]
    fn stream_generation(
        &mut self,
        prompt_text: &str,
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
        let mut bar = crate::statusbar::StatusBar::new(self.show_footer && self.color, self.color);
        let stats = self
            .engine
            .generate(
                prompt_text,
                &self.cfg.generation,
                &crate::interrupt::pending,
                &mut |ev| match ev {
                    EngineEvent::Text(t) => {
                        // Model output has started: drop the prefill bar so the
                        // text streams cleanly from column zero.
                        bar.clear();
                        assistant_text.push_str(&t);
                        stream.push(&t);
                    }
                    EngineEvent::Prefill(p) => {
                        bar.show(&Status {
                            state: WorkerState::Prefill,
                            prefill_done: p.done,
                            prefill_total: p.total,
                            prefill_tps: p.tps,
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
        Ok((stream, assistant_text, stats))
    }

    /// Runs one model turn: stream text, execute tool calls, repeat until
    /// a turn produces no tool calls. Compacts first when context is tight.
    fn run_turn(&mut self) -> Result<(), String> {
        self.maybe_compact()?;
        self.maybe_append_system_prompt_reminder();
        loop {
            let prompt_text = render_transcript(&self.session, &self.system);
            let (stream, assistant_text, stats) = self.stream_generation(&prompt_text)?;

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
            if stats.interrupted {
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
            if let Some(err) = finished.error {
                self.session.push(Message::user(format!(
                    "<tool_result>Tool error: {err}</tool_result>"
                )));
                continue;
            }
            if !finished.calls.is_empty() {
                let observations = dispatch_all(finished.calls, &mut self.tool_ctx);
                let mut renderer = stream.into_sink().renderer;
                renderer.finish();
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
            if self.show_footer && !self.editor_owns_footer {
                print_footer(&st, self.color);
            }
            return Ok(());
        }
    }

    /// Re-injects the trusted system prompt shape after enough context has
    /// passed since it was last seen, mirroring the C's pressure policy.
    fn maybe_append_system_prompt_reminder(&mut self) {
        let rendered = render_transcript(&self.session, &self.system);
        let pos = self.engine.count_tokens(&rendered);
        if !self.reminder.should_remind(pos) {
            return;
        }
        println!("Re-injecting system prompt reminder...");
        self.trace.line(&format!(
            "system prompt reminder injected at transcript={pos}"
        ));
        let mut text = sysprompt::build_system_prompt_reminder();
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
        self.compact("low context")
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
            .generate(&prompt_text, &self.cfg.generation, &|| false, &mut |ev| {
                if let EngineEvent::Text(t) = ev {
                    summary.push_str(&t);
                }
            })
            .map_err(|e| e.to_string())?;
        if self.color {
            print!("\x1b[0m");
        }

        // Keep a verbatim tail within budget, starting at a user boundary.
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
        println!("context compacted");
        Ok(())
    }

    /// Handles a slash command; returns false when the REPL should exit.
    fn slash(&mut self, input: &str) -> Result<bool, String> {
        let mut parts = input.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or(input);
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            "/quit" | "/exit" => return Ok(false),
            "/new" => {
                self.session = Session::new();
                self.reminder = SystemPromptReminder::new();
                let datetime = sysprompt::datetime_context_line(std::time::SystemTime::now());
                self.session.push(Message::user(datetime));
                println!("started a new session");
            }
            "/help" => print!("{}", crate::config::usage()),
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
                }
                Err(e) => println!("switch failed: {e}"),
            },
            "/del" => match self.store.delete(arg) {
                Ok(id) => println!("deleted session {}", &id[..8]),
                Err(e) => println!("delete failed: {e}"),
            },
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
            "/compact" => self.compact("user request")?,
            _ if slash_command_known(cmd) => println!("{cmd}: not implemented yet"),
            _ => println!("unknown command: {cmd}"),
        }
        Ok(true)
    }
}

/// Result of one TUI generation pass.
struct TurnOutput {
    interrupted: bool,
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
        let result = self.tui_loop(&mut terminal);
        ratatui::restore();
        result
    }

    fn tui_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<(), String> {
        let mut input = TuiInput::new();
        let hist_path = default_history_path();
        input.history.load(&hist_path).ok();
        let mut log = OutputLog::new();
        for line in tui::ansi_to_lines(&crate::logo::art(crate::logo::DEFAULT_WIDTH)) {
            log.push_spans(line.spans);
        }
        log.push_plain(format!(
            "plank 🪵 Agent, context {} tokens",
            status::format_ctx_size(self.engine.ctx_size())
        ));
        log.push_plain("Type a message, or /help for commands. Ctrl-D to quit.");
        log.push_plain(String::new());

        self.tui_warm(terminal, &mut log)?;

        if let Some(initial) = self.cfg.prompt.as_deref().filter(|p| !p.is_empty()) {
            log.push_spans(tui::user_echo_spans(initial));
            self.session.push(Message::user(initial));
            self.tui_turn(terminal, &mut log)?;
        }

        loop {
            let status = self.idle_status_text();
            terminal
                .draw(|f| tui::draw(f, &log, input.buf.text(), input.cursor_col(), &status))
                .map_err(|e| e.to_string())?;

            if !event::poll(Duration::from_millis(200)).map_err(|e| e.to_string())? {
                continue;
            }
            let Event::Key(key) = event::read().map_err(|e| e.to_string())? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            match key.code {
                KeyCode::Char('c') if ctrl => {
                    if input.buf.text().is_empty() {
                        break;
                    }
                    input.buf.clear();
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
                KeyCode::Enter => {
                    let line = input.buf.text().trim().to_owned();
                    input.buf.clear();
                    input.hist_idx = None;
                    if line.is_empty() {
                        continue;
                    }
                    input.history.add(&line);
                    input.history.save(&hist_path).ok();
                    if line.starts_with('/') {
                        if !self.tui_slash(&line, &mut log) {
                            break;
                        }
                    } else {
                        log.push_spans(tui::user_echo_spans(&line));
                        self.session.push(Message::user(&line));
                        self.tui_turn(terminal, &mut log)?;
                    }
                }
                _ => {}
            }
        }
        input.history.save(&hist_path).ok();
        Ok(())
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
        self.engine
            .warm_system_prompt(&system, Some(&checkpoint), &mut |ev| {
                if let EngineEvent::Prefill(p) = ev {
                    if !announced {
                        announced = true;
                        log.push_plain("Updating system prompt cache...");
                    }
                    let st = Status {
                        state: WorkerState::Prefill,
                        prefill_done: p.done,
                        prefill_total: p.total,
                        prefill_tps: p.tps,
                        ctx_size,
                        power_percent: power,
                        ..Status::default()
                    };
                    let line = status::build_status_text(&st, false);
                    let _ = terminal.draw(|f| tui::draw(f, log, "", 0, &line));
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Warms the system-prompt KV cache for non-TUI runs (stderr message).
    fn warm_plain(&mut self) -> Result<(), String> {
        let checkpoint = self.sysprompt_checkpoint();
        let system = self.system.clone();
        let mut announced = false;
        self.engine
            .warm_system_prompt(&system, Some(&checkpoint), &mut |ev| {
                if matches!(ev, EngineEvent::Prefill(_)) && !announced {
                    announced = true;
                    eprintln!("Updating system prompt cache...");
                }
            })
            .map_err(|e| e.to_string())?;
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

    /// One TUI turn: generate, run any tool calls, repeat until settled.
    fn tui_turn(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
        log: &mut OutputLog,
    ) -> Result<(), String> {
        self.tui_maybe_compact(log)?;
        self.tui_maybe_reminder(log);
        loop {
            let prompt = render_transcript(&self.session, &self.system);
            let out = self.tui_generate(terminal, log, &prompt)?;
            self.session.push(Message::assistant(out.assistant_text));
            log.end_line();
            if out.interrupted {
                crate::interrupt::clear();
                log.push_plain("[interrupted]");
                return Ok(());
            }
            if let Some(err) = out.error {
                self.session.push(Message::user(format!(
                    "<tool_result>Tool error: {err}</tool_result>"
                )));
                continue;
            }
            if !out.calls.is_empty() {
                let observations = dispatch_all(&out.calls, &mut self.tool_ctx);
                self.session.push(Message::user(format!(
                    "<tool_result>{observations}</tool_result>"
                )));
                for line in observations.lines() {
                    log.push_plain(line.to_owned());
                }
                continue;
            }
            return Ok(());
        }
    }

    /// Streams one generation pass into `log`, drawing each update.
    fn tui_generate(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
        log: &mut OutputLog,
        prompt: &str,
    ) -> Result<TurnOutput, String> {
        let mut stream = StreamRenderer::new(std::mem::take(log));
        if !matches!(
            self.cfg.generation.think_mode,
            crate::engine::ThinkMode::Off
        ) {
            stream.begin_in_think();
        }
        let interrupt_flag = AtomicBool::new(false);
        let ctx_size = self.engine.ctx_size();
        let power = self.power_percent;
        let mut assistant_text = String::new();
        let mut gen_count = 0;
        let start = Instant::now();

        let result = self.engine.generate(
            prompt,
            &self.cfg.generation,
            &|| interrupt_flag.load(Ordering::Relaxed) || crate::interrupt::pending(),
            &mut |ev| {
                let status = match ev {
                    EngineEvent::Text(t) => {
                        assistant_text.push_str(&t);
                        stream.push(&t);
                        gen_count += 1;
                        let secs = start.elapsed().as_secs_f64();
                        Status {
                            state: WorkerState::Generating,
                            generated: gen_count,
                            gen_tps: if secs > 0.0 {
                                f64::from(gen_count) / secs
                            } else {
                                0.0
                            },
                            ctx_size,
                            power_percent: power,
                            ..Status::default()
                        }
                    }
                    EngineEvent::Prefill(p) => Status {
                        state: WorkerState::Prefill,
                        prefill_done: p.done,
                        prefill_total: p.total,
                        prefill_tps: p.tps,
                        ctx_size,
                        power_percent: power,
                        ..Status::default()
                    },
                };
                // Poll for Ctrl-C / Esc so generation stays interruptible.
                if event::poll(Duration::ZERO).unwrap_or(false)
                    && let Ok(Event::Key(k)) = event::read()
                    && k.kind == KeyEventKind::Press
                    && (matches!(k.code, KeyCode::Esc)
                        || (matches!(k.code, KeyCode::Char('c'))
                            && k.modifiers.contains(KeyModifiers::CONTROL)))
                {
                    interrupt_flag.store(true, Ordering::Relaxed);
                }
                let line = status::build_status_text(&status, false);
                let _ = terminal.draw(|f| tui::draw(f, stream.sink(), "", 0, &line));
            },
        );

        let stats = result.map_err(|e| e.to_string())?;
        stream.finish();
        let finished = stream.finished();
        let calls = finished.calls.to_vec();
        let error = finished.error.map(str::to_owned);
        let interrupted = stats.interrupted || interrupt_flag.load(Ordering::Relaxed);
        *log = stream.into_sink();
        Ok(TurnOutput {
            interrupted,
            assistant_text,
            calls,
            error,
        })
    }

    /// Compacts before a TUI turn when context is tight, logging progress.
    fn tui_maybe_compact(&mut self, log: &mut OutputLog) -> Result<(), String> {
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        if !compact::should_compact(self.engine.ctx_size(), used) {
            return Ok(());
        }
        self.tui_do_compact("low context", log)
    }

    /// Performs a compaction pass and rebuilds the transcript, logging progress.
    fn tui_do_compact(&mut self, reason: &str, log: &mut OutputLog) -> Result<(), String> {
        log.push_plain(format!(
            "COMPACTING {reason}: summarizing durable task state..."
        ));
        let mut prompt = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(prompt, "[user]\n{}\n", compact::make_prompt(reason));
        }
        let mut summary = String::new();
        self.engine
            .generate(&prompt, &self.cfg.generation, &|| false, &mut |ev| {
                if let EngineEvent::Text(t) = ev {
                    summary.push_str(&t);
                }
            })
            .map_err(|e| e.to_string())?;
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
        log.push_plain("context compacted");
        Ok(())
    }

    /// Re-injects the system-prompt reminder in the TUI when due.
    fn tui_maybe_reminder(&mut self, log: &mut OutputLog) {
        let rendered = render_transcript(&self.session, &self.system);
        let pos = self.engine.count_tokens(&rendered);
        if !self.reminder.should_remind(pos) {
            return;
        }
        log.push_plain("Re-injecting system prompt reminder...");
        self.trace.line(&format!(
            "system prompt reminder injected at transcript={pos}"
        ));
        let mut text = sysprompt::build_system_prompt_reminder();
        if !self.cfg.system.is_empty() {
            text.push_str("\nAdditional system instructions reminder:\n");
            text.push_str(&self.cfg.system);
            text.push_str("\n[End additional system instructions reminder.]\n\n");
        }
        self.session.push(Message::user(text));
    }

    /// Handles a slash command in the TUI; returns false to quit.
    fn tui_slash(&mut self, input: &str, log: &mut OutputLog) -> bool {
        let mut parts = input.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or(input);
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            "/quit" | "/exit" => return false,
            "/new" => {
                self.session = Session::new();
                self.reminder = SystemPromptReminder::new();
                let datetime = sysprompt::datetime_context_line(std::time::SystemTime::now());
                self.session.push(Message::user(datetime));
                log.push_plain("started a new session");
            }
            "/help" => {
                for line in crate::config::usage().lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/compact" => {
                if let Err(e) = self.tui_do_compact("user request", log) {
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
                }
                Err(e) => log.push_plain(format!("switch failed: {e}")),
            },
            "/del" => match self.store.delete(arg) {
                Ok(id) => log.push_plain(format!("deleted session {}", &id[..8])),
                Err(e) => log.push_plain(format!("delete failed: {e}")),
            },
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
            _ if slash_command_known(cmd) => {
                log.push_plain(format!("{cmd}: not implemented yet"));
            }
            _ => log.push_plain(format!("unknown command: {cmd}")),
        }
        true
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
    // Inject the session-start datetime context once, like the C worker.
    let datetime = sysprompt::datetime_context_line(std::time::SystemTime::now());
    trace.text("datetime-context", &datetime);
    session.push(Message::user(datetime));
    let mut tool_ctx = ToolContext::new(cwd);
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
    Ok(Agent {
        engine,
        cfg,
        session,
        store,
        tool_ctx,
        system: sysprompt::build_system_prompt(&cfg.system),
        reminder: SystemPromptReminder::new(),
        power_percent: 0,
        trace,
        color: std::io::stdout().is_terminal(),
        show_footer,
        editor_owns_footer: false,
    })
}

/// Runs the interactive REPL until the user exits.
///
/// # Errors
/// Returns an error string on unrecoverable I/O or engine failure.
pub fn run_interactive(engine: Box<dyn Engine>, cfg: &AgentConfig) -> Result<(), String> {
    let mut agent = new_agent(engine, cfg, true)?;

    // A real terminal gets the full-screen ratatui UI (works cleanly in Warp
    // and other block terminals via the alternate screen). Piped input falls
    // back to the plain line REPL.
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        agent.run_tui()
    } else {
        agent.warm_plain()?;
        if let Some(initial) = cfg.prompt.as_deref().filter(|p| !p.is_empty()) {
            print!("{}", status::format_user_prompt_echo(initial, agent.color));
            agent.session.push(Message::user(initial));
            agent.run_turn()?;
        }
        run_repl_plain(&mut agent)
    }
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

    /// Engine that plays back canned replies in order.
    #[derive(Debug)]
    struct ScriptedEngine {
        replies: Vec<String>,
        next: usize,
    }

    impl Engine for ScriptedEngine {
        fn generate(
            &mut self,
            _transcript: &str,
            _opts: &crate::engine::GenerationOptions,
            _interrupt: &dyn Fn() -> bool,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<GenerationStats, EngineError> {
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
            Ok(GenerationStats::default())
        }
        fn ctx_size(&self) -> i32 {
            100_000
        }
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
            next: 0,
        };
        let cfg = crate::config::AgentConfig::default();
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt(""),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
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
}
