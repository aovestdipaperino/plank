//! Interactive REPL and headless front-ends over the agent turn loop.
//!
//! Port of the "Interactive Runtime Loop" section of `ds4_agent.c`, adapted
//! to a synchronous turn loop: the C multiplexes a worker thread with `poll()`;
//! plank v1 runs the engine inline and streams output as it arrives.

use std::io::{BufRead, IsTerminal, Write};

use crate::compact;
use crate::config::{AgentConfig, slash_command_known};
use crate::engine::{Engine, EngineEvent};
use crate::render::{RenderOptions, TokenRenderer};
use crate::session::{Message, Session, SessionStore};
use crate::status::{self, Status, WorkerState};
use crate::sysprompt::{self, SystemPromptReminder};
use crate::tools::{ToolContext, dispatch_all};
use crate::trace::Trace;
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

    if let Some(initial) = cfg.prompt.as_deref().filter(|p| !p.is_empty()) {
        print!("{}", status::format_user_prompt_echo(initial, agent.color));
        agent.session.push(Message::user(initial));
        agent.run_turn()?;
    }

    if std::io::stdin().is_terminal() {
        run_repl_editor(&mut agent)
    } else {
        run_repl_plain(&mut agent)
    }
}

/// Idle footer line reflecting current context pressure.
fn idle_footer(agent: &mut Agent<'_>) -> String {
    let rendered = render_transcript(&agent.session, &agent.system);
    let st = Status {
        state: WorkerState::Idle,
        ctx_used: agent.engine.count_tokens(&rendered),
        ctx_size: agent.engine.ctx_size(),
        power_percent: agent.power_percent,
        ..Status::default()
    };
    status::build_status_text(&st, agent.color)
}

/// Full-featured REPL over the raw-mode line editor (TTY only).
fn run_repl_editor(agent: &mut Agent<'_>) -> Result<(), String> {
    // The editor paints its own resting footer at the prompt, so the turn loop
    // must not print a second footer after each generation.
    agent.editor_owns_footer = true;
    let mut editor = crate::editor::Editor::new();
    let hist_path = crate::editor::default_history_path();
    editor.history_mut().load(&hist_path).ok();
    let cache_dir = agent.store.dir().to_path_buf();
    editor.set_completion(Box::new(move |line: &str| {
        if let Some(prefix) = line.strip_prefix("/switch ") {
            if let Ok(store) = SessionStore::open(&cache_dir) {
                return store.complete(prefix.trim()).unwrap_or_default();
            }
            return Vec::new();
        }
        if line.starts_with('/') && !line.contains(' ') {
            const CMDS: [&str; 12] = [
                "/help", "/save", "/compact", "/list", "/quit", "/exit", "/new", "/power",
                "/switch", "/del", "/strip", "/history",
            ];
            return CMDS
                .iter()
                .filter(|c| c.starts_with(line))
                .map(|c| (*c).to_string())
                .collect();
        }
        Vec::new()
    }));

    loop {
        let footer = idle_footer(agent);
        let outcome = editor
            .read_line(status::prompt_text(), &footer)
            .map_err(|e| e.to_string())?;
        editor.restore_terminal();
        let line = match outcome {
            crate::editor::ReadOutcome::Line(l) => l,
            crate::editor::ReadOutcome::Interrupted => continue,
            crate::editor::ReadOutcome::Eof => break,
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        editor.history_mut().add(input);
        editor.history().save(&hist_path).ok();
        if input.starts_with('/') {
            if !agent.slash(input)? {
                break;
            }
            continue;
        }
        print!("{}", status::format_user_prompt_echo(input, agent.color));
        agent.session.push(Message::user(input));
        agent.run_turn()?;
    }
    editor.restore_terminal();
    Ok(())
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
