//! MCP client: external tools from stdio MCP servers.
//!
//! Port of the "MCP Client" section of `ds4_agent.c` (mcp-support branch).
//! Servers listed in a `.mcp.json` file are spawned as long-lived
//! subprocesses speaking newline-delimited JSON-RPC 2.0 on stdin/stdout (the
//! MCP stdio transport). The client is intentionally synchronous: one tool
//! call blocks the worker for one round trip, exactly like every other tool.
//!
//! Tool names are namespaced `mcp__<server>__<tool>` so they never collide
//! across servers or with native tools. Tools listed in a server's
//! `primaryTools` get their full JSON schema in the system prompt; the rest
//! appear in a compact directory and are described on demand via the
//! `mcp_describe` tool.

use std::fmt::Write as _;
use std::io::Write as _;
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use crate::dsml::ToolCall;

/// Timeout for one MCP request round trip, in seconds.
const MCP_TIMEOUT_SEC: u64 = 30;

// ============================================================================
// Minimal JSON value (port of agent_json)
// ============================================================================

/// A parsed JSON value, mirroring the C `agent_json` tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// JSON `null`.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON number (always stored as f64, like the C port).
    Num(f64),
    /// JSON string.
    Str(String),
    /// JSON array.
    Arr(Vec<Json>),
    /// JSON object with insertion-ordered keys.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Looks up an object member by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Returns the string value of an object member, or the default.
    #[must_use]
    pub fn str_or<'a>(&'a self, key: &str, def: &'a str) -> &'a str {
        match self.get(key) {
            Some(Json::Str(s)) => s,
            _ => def,
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl JsonParser<'_> {
    fn peek(&self) -> u8 {
        *self.bytes.get(self.pos).unwrap_or(&0)
    }

    fn bump(&mut self) -> u8 {
        let c = self.peek();
        if c != 0 {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }

    fn parse_string_raw(&mut self) -> Option<String> {
        if self.bump() != b'"' {
            return None;
        }
        let mut out = String::new();
        loop {
            match self.bump() {
                0 => return None,
                b'"' => return Some(out),
                b'\\' => match self.bump() {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let mut cp: u32 = 0;
                        for _ in 0..4 {
                            let h = self.bump();
                            cp = (cp << 4)
                                | u32::from(match h {
                                    b'0'..=b'9' => h - b'0',
                                    b'a'..=b'f' => h - b'a' + 10,
                                    b'A'..=b'F' => h - b'A' + 10,
                                    _ => return None,
                                });
                        }
                        // Like the C port, only the basic multilingual plane is
                        // handled; unpaired surrogates fall back to U+FFFD.
                        out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                    }
                    _ => return None,
                },
                c => {
                    // Raw bytes pass through; multi-byte UTF-8 is copied as-is.
                    let start = self.pos - 1;
                    let len = utf8_len(c);
                    let end = (start + len).min(self.bytes.len());
                    out.push_str(&String::from_utf8_lossy(&self.bytes[start..end]));
                    self.pos = end;
                }
            }
        }
    }

    fn parse_value(&mut self) -> Option<Json> {
        self.skip_ws();
        match self.peek() {
            b'"' => self.parse_string_raw().map(Json::Str),
            b'n' if self.eat("null") => Some(Json::Null),
            b't' if self.eat("true") => Some(Json::Bool(true)),
            b'f' if self.eat("false") => Some(Json::Bool(false)),
            b'-' | b'0'..=b'9' => {
                let start = self.pos;
                if self.peek() == b'-' {
                    self.pos += 1;
                }
                while matches!(self.peek(), b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
                    self.pos += 1;
                }
                let text = std::str::from_utf8(&self.bytes[start..self.pos]).ok()?;
                text.parse::<f64>().ok().map(Json::Num)
            }
            b'[' => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_ws();
                if self.peek() == b']' {
                    self.pos += 1;
                    return Some(Json::Arr(items));
                }
                loop {
                    items.push(self.parse_value()?);
                    self.skip_ws();
                    match self.bump() {
                        b',' => {}
                        b']' => return Some(Json::Arr(items)),
                        _ => return None,
                    }
                }
            }
            b'{' => {
                self.pos += 1;
                let mut members = Vec::new();
                self.skip_ws();
                if self.peek() == b'}' {
                    self.pos += 1;
                    return Some(Json::Obj(members));
                }
                loop {
                    self.skip_ws();
                    let key = self.parse_string_raw()?;
                    self.skip_ws();
                    if self.bump() != b':' {
                        return None;
                    }
                    let val = self.parse_value()?;
                    members.push((key, val));
                    self.skip_ws();
                    match self.bump() {
                        b',' => {}
                        b'}' => return Some(Json::Obj(members)),
                        _ => return None,
                    }
                }
            }
            _ => None,
        }
    }
}

const fn utf8_len(first: u8) -> usize {
    match first {
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

/// Parses one complete JSON document; `None` on any syntax error.
#[must_use]
pub fn json_parse(text: &str) -> Option<Json> {
    JsonParser {
        bytes: text.as_bytes(),
        pos: 0,
    }
    .parse_value()
}

/// Appends `s` as a JSON string literal (quotes and escapes included).
pub fn json_escape(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Serializes a [`Json`] value back to compact JSON text.
///
/// Used to re-embed `inputSchema` in the tools prompt, mirroring
/// `agent_json_write`.
pub fn json_write(out: &mut String, v: &Json) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            #[allow(clippy::float_cmp)] // exact integrality check, like the C port
            if n.round() == *n && n.abs() < 1e15 {
                let _ = write!(out, "{n:.0}");
            } else {
                let _ = write!(out, "{n}");
            }
        }
        Json::Str(s) => json_escape(out, s),
        Json::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json_write(out, item);
            }
            out.push(']');
        }
        Json::Obj(members) => {
            out.push('{');
            for (i, (k, val)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json_escape(out, k);
                out.push(':');
                json_write(out, val);
            }
            out.push('}');
        }
    }
}

// ============================================================================
// MCP server process and protocol
// ============================================================================

/// One tool advertised by an MCP server.
#[derive(Debug, Clone)]
pub struct McpTool {
    /// Tool name as advertised by the server (un-namespaced).
    pub name: String,
    /// Tool description from `tools/list`.
    pub description: String,
    /// Raw compact JSON text of `inputSchema`; a default object if absent.
    pub schema_json: String,
    /// Full schema in the system prompt vs directory entry.
    pub primary: bool,
}

/// A resource advertised by an MCP server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpResource {
    /// The resource URI, used verbatim in the `{server}:{uri}` token.
    pub uri: String,
    /// Human-readable label; empty when the server gave none.
    pub name: String,
}

/// Extracts the `resources` array from a `resources/list` result.
fn parse_resources(root: &Json) -> Vec<McpResource> {
    let Some(Json::Arr(list)) = root.get("resources") else {
        return Vec::new();
    };
    list.iter()
        .filter_map(|r| {
            let uri = r.str_or("uri", "");
            if uri.is_empty() {
                return None;
            }
            Some(McpResource {
                uri: uri.to_string(),
                name: r.str_or("name", "").to_string(),
            })
        })
        .collect()
}

/// Builds `{server}:{uri}` completion candidates for one server's resources.
fn resource_candidates_from(
    server: &str,
    resources: &[McpResource],
) -> Vec<crate::complete::Candidate> {
    resources
        .iter()
        .map(|r| crate::complete::Candidate {
            text: format!("{server}:{}", r.uri),
            kind: crate::complete::Kind::Resource,
            demoted: false,
        })
        .collect()
}

/// Collects completion candidates for every live server's resources.
#[must_use]
pub fn resource_candidates(servers: &[McpServer]) -> Vec<crate::complete::Candidate> {
    servers
        .iter()
        .filter(|s| s.alive())
        .flat_map(|s| resource_candidates_from(&s.name, s.resources()))
        .collect()
}

/// Config for one server parsed from `.mcp.json`, before it is started.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Server name (the key in `mcpServers`); never contains `__`.
    pub name: String,
    /// Executable to spawn.
    pub command: String,
    /// Argv tail.
    pub args: Vec<String>,
    /// Extra environment `(key, value)` pairs.
    pub env: Vec<(String, String)>,
    /// Tool names granted a full schema in the prompt; `None` = all primary.
    pub primary_tools: Option<Vec<String>>,
}

/// A live MCP server subprocess with its advertised tools.
pub struct McpServer {
    /// Server name used in `mcp__<server>__<tool>`.
    pub name: String,
    /// Tools advertised at handshake time.
    pub tools: Vec<McpTool>,
    /// Free-text usage instructions from the initialize response; empty when
    /// the server provided none.
    pub instructions: String,
    /// Resources advertised at handshake time.
    resources: Vec<McpResource>,
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    alive: bool,
    next_id: i64,
    rbuf: Vec<u8>,
}

impl std::fmt::Debug for McpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServer")
            .field("name", &self.name)
            .field("alive", &self.alive)
            .field("tools", &self.tools.len())
            .finish_non_exhaustive()
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        // Mirror agent_mcp_server_close: SIGTERM, up to 1s grace, then SIGKILL.
        if self.alive {
            #[allow(clippy::cast_possible_wrap)]
            let pid = self.child.id() as libc::pid_t;
            unsafe { libc::kill(pid, libc::SIGTERM) };
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(1) {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

impl McpServer {
    /// Spawns the server process with stdin/stdout piped and stderr dropped.
    ///
    /// # Errors
    /// Returns a message when the executable cannot be started.
    pub fn spawn(cfg: &McpServerConfig) -> Result<Self, String> {
        let mut child = Command::new(&cfg.command)
            .args(&cfg.args)
            .envs(cfg.env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr to /dev/null so stdout carries only JSON-RPC.
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|e| format!("failed to start {}: {e}", cfg.command))?;
        let stdin = child.stdin.take().ok_or("no stdin pipe")?;
        let stdout = child.stdout.take().ok_or("no stdout pipe")?;
        Ok(Self {
            name: cfg.name.clone(),
            tools: Vec::new(),
            instructions: String::new(),
            resources: Vec::new(),
            child,
            stdin,
            stdout,
            alive: true,
            next_id: 0,
            rbuf: Vec::new(),
        })
    }

    /// True when the transport has not failed.
    #[must_use]
    pub fn alive(&self) -> bool {
        self.alive
    }

    /// Resources advertised at handshake time.
    #[must_use]
    pub fn resources(&self) -> &[McpResource] {
        &self.resources
    }

    /// Reads one newline-delimited message, blocking up to `deadline`.
    ///
    /// Returns `None` (and marks the server dead) on timeout, EOF, or error.
    fn read_line(&mut self, deadline: Instant) -> Option<String> {
        loop {
            if let Some(nl) = self.rbuf.iter().position(|&b| b == b'\n') {
                let line = String::from_utf8_lossy(&self.rbuf[..nl]).into_owned();
                self.rbuf.drain(..=nl);
                return Some(line);
            }
            let remaining = deadline.checked_duration_since(Instant::now())?;
            let mut pfd = libc::pollfd {
                fd: self.stdout.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            #[allow(clippy::cast_possible_truncation)]
            let timeout_ms = (remaining.as_millis() as libc::c_int).saturating_add(1);
            let pr = unsafe { libc::poll(&raw mut pfd, 1, timeout_ms) };
            if pr <= 0 {
                self.alive = false;
                return None;
            }
            let mut chunk = [0u8; 4096];
            let n = std::io::Read::read(&mut self.stdout, &mut chunk).unwrap_or(0);
            if n == 0 {
                self.alive = false;
                return None;
            }
            self.rbuf.extend_from_slice(&chunk[..n]);
        }
    }

    fn write_line(&mut self, line: &str) -> bool {
        let ok = self
            .stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .is_ok();
        if !ok {
            self.alive = false;
        }
        ok
    }

    /// Sends a JSON-RPC request and waits for the matching reply.
    ///
    /// Notifications and replies to older calls are discarded. Returns the
    /// `result` value on success.
    ///
    /// # Errors
    /// Returns a message on transport failure or a JSON-RPC `error` reply.
    fn request(&mut self, method: &str, params_json: &str) -> Result<Json, String> {
        self.next_id += 1;
        let id = self.next_id;
        let mut req = format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":");
        json_escape(&mut req, method);
        req.push_str(",\"params\":");
        req.push_str(params_json);
        req.push('}');
        if !self.write_line(&req) {
            return Err("failed to write to server".to_string());
        }

        // The deadline covers the whole exchange, not each line: a server
        // that streams notifications forever must still answer in time.
        let deadline = Instant::now() + Duration::from_secs(MCP_TIMEOUT_SEC);
        loop {
            let Some(line) = self.read_line(deadline) else {
                return Err("no response from server (timeout or closed pipe)".to_string());
            };
            // Ignore unparsable/log lines on stdout.
            let Some(resp) = json_parse(&line) else {
                continue;
            };
            #[allow(clippy::cast_possible_truncation)]
            let matches_id = matches!(resp.get("id"), Some(Json::Num(n)) if *n as i64 == id);
            if !matches_id {
                continue; // a notification or a reply to an older call
            }
            if let Some(error) = resp.get("error") {
                return Err(error.str_or("message", "MCP error").to_string());
            }
            return Ok(resp.get("result").cloned().unwrap_or(Json::Null));
        }
    }

    fn notify(&mut self, method: &str) -> bool {
        self.write_line(&format!(
            "{{\"jsonrpc\":\"2.0\",\"method\":\"{method}\",\"params\":{{}}}}"
        ))
    }

    /// Full startup handshake: initialize, initialized, `tools/list`.
    ///
    /// # Errors
    /// Returns a message on any protocol failure; the caller drops the server
    /// rather than exposing broken tools to the model.
    pub fn handshake(&mut self, primary_tools: Option<&[String]>) -> Result<(), String> {
        let init_params = "{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\
             \"clientInfo\":{\"name\":\"plank\",\"version\":\"1.0\"}}";
        let init_result = self.request("initialize", init_params)?;
        // Servers may declare free-text usage guidance for their tools; it is
        // injected into the system prompt alongside the schemas.
        self.instructions = init_result.str_or("instructions", "").trim().to_string();
        if !self.notify("notifications/initialized") {
            return Err("failed to send notifications/initialized".to_string());
        }
        let result = self.request("tools/list", "{}")?;
        if let Some(Json::Arr(tools)) = result.get("tools") {
            for t in tools {
                let Some(Json::Str(name)) = t.get("name") else {
                    continue;
                };
                let primary = primary_tools.is_none_or(|list| list.iter().any(|p| p == name));
                let mut schema_json = String::new();
                match t.get("inputSchema") {
                    Some(schema) => json_write(&mut schema_json, schema),
                    None => schema_json.push_str("{\"type\":\"object\",\"properties\":{}}"),
                }
                self.tools.push(McpTool {
                    name: name.clone(),
                    description: t.str_or("description", "").to_string(),
                    schema_json,
                    primary,
                });
            }
        }
        // Resources are optional. Only ask a server that advertised the
        // capability: one that silently ignores unknown methods (rather than
        // replying -32601) would stall the whole handshake for
        // `MCP_TIMEOUT_SEC` and then be marked dead, costing us all of its
        // tools for a mere completion nicety.
        let advertises_resources = init_result
            .get("capabilities")
            .and_then(|c| c.get("resources"))
            .is_some();
        if advertises_resources && let Ok(res) = self.request("resources/list", "{}") {
            self.resources = parse_resources(&res);
        }
        Ok(())
    }

    fn find_tool(&self, name: &str) -> Option<&McpTool> {
        self.tools.iter().find(|t| t.name == name)
    }
}

// ============================================================================
// Config loading
// ============================================================================

/// Parses a `.mcp.json` file's `mcpServers` object into server configs.
///
/// Returns an empty list (not an error) if the file is missing; prints a
/// warning and returns an empty list if it exists but is malformed. Server
/// names containing `__` are rejected up front: tool names are advertised as
/// `mcp__<server>__<tool>` and split at the first `__` on dispatch, so such a
/// name could never be routed back.
#[must_use]
pub fn config_load(path: &Path) -> Vec<McpServerConfig> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Some(root) = json_parse(&text) else {
        eprintln!(
            "plank: {}: invalid JSON, ignoring MCP config",
            path.display()
        );
        return Vec::new();
    };
    let mut out = Vec::new();
    let Some(Json::Obj(servers)) = root.get("mcpServers") else {
        return out;
    };
    for (name, sv) in servers {
        let command = sv.str_or("command", "");
        if command.is_empty() {
            eprintln!("plank: MCP server \"{name}\" has no command, skipping");
            continue;
        }
        if name.contains("__") {
            eprintln!("plank: MCP server name \"{name}\" must not contain \"__\", skipping");
            continue;
        }
        let mut cfg = McpServerConfig {
            name: name.clone(),
            command: command.to_string(),
            args: Vec::new(),
            env: Vec::new(),
            primary_tools: None,
        };
        if let Some(Json::Arr(args)) = sv.get("args") {
            for a in args {
                if let Json::Str(s) = a {
                    cfg.args.push(s.clone());
                }
            }
        }
        if let Some(Json::Obj(env)) = sv.get("env") {
            for (k, v) in env {
                if let Json::Str(s) = v {
                    cfg.env.push((k.clone(), s.clone()));
                }
            }
        }
        // Optional prompt-size control: tools named here get their full
        // schema in the system prompt; the rest only appear in a compact
        // directory and are described on demand via mcp_describe. An empty
        // list means every tool is directory-only.
        if let Some(Json::Arr(primary)) = sv.get("primaryTools") {
            cfg.primary_tools = Some(
                primary
                    .iter()
                    .filter_map(|p| match p {
                        Json::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
            );
        }
        out.push(cfg);
    }
    out
}

/// Global MCP config location: `.mcp.json` inside plank's home directory.
///
/// `~/.plank` is the same directory the model and kv-cache live in.
#[must_use]
pub fn global_config_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".plank").join(".mcp.json"))
}

/// Merges the global and local MCP configs, local overriding by server name.
///
/// Hierarchical like Claude Code's user vs project scope: servers from
/// `~/.plank/.mcp.json` apply everywhere, and a same-named entry in the local
/// file (`./.mcp.json`, or the `--mcp-config` path when given) replaces the
/// global definition entirely. Servers only named locally are added.
#[must_use]
pub fn config_load_hierarchy(local_path: Option<&Path>) -> Vec<McpServerConfig> {
    let global = match global_config_path() {
        Some(path) => config_load(&path),
        None => Vec::new(),
    };
    let default = Path::new(".mcp.json");
    let local = config_load(local_path.unwrap_or(default));
    merge_configs(global, local)
}

/// Overlays `local` server configs onto `global`, matching by server name.
fn merge_configs(
    global: Vec<McpServerConfig>,
    local: Vec<McpServerConfig>,
) -> Vec<McpServerConfig> {
    let mut merged = global;
    for entry in local {
        match merged.iter_mut().find(|m| m.name == entry.name) {
            Some(slot) => *slot = entry,
            None => merged.push(entry),
        }
    }
    merged
}

/// Loads the config hierarchy, spawns and handshakes each server.
///
/// Mirrors `agent_mcp_load_and_start`: a server that cannot start or
/// handshake is reported to stderr and skipped rather than aborting the
/// whole agent. `path` overrides the local config location; the global
/// `~/.plank/.mcp.json` always applies underneath (see
/// [`config_load_hierarchy`]).
#[must_use]
pub fn load_and_start(path: Option<&Path>) -> Vec<McpServer> {
    start_servers(config_load_hierarchy(path))
}

/// Spawns and handshakes each configured server, dropping failures.
fn start_servers(configs: Vec<McpServerConfig>) -> Vec<McpServer> {
    let mut servers = Vec::new();
    for cfg in configs {
        let started = McpServer::spawn(&cfg).and_then(|mut s| {
            s.handshake(cfg.primary_tools.as_deref())?;
            Ok(s)
        });
        match started {
            Ok(s) => servers.push(s),
            Err(err) => eprintln!("plank: MCP server \"{}\" unavailable: {err}", cfg.name),
        }
    }
    servers
}

// ============================================================================
// Prompt rendering
// ============================================================================

fn append_one_schema(out: &mut String, server_name: &str, t: &McpTool) {
    out.push_str("{\n  \"type\": \"function\",\n  \"function\": {\n    \"name\": \"mcp__");
    out.push_str(server_name);
    out.push_str("__");
    out.push_str(&t.name);
    out.push_str("\",\n    \"description\": ");
    json_escape(out, &t.description);
    out.push_str(",\n    \"parameters\": ");
    out.push_str(&t.schema_json);
    out.push_str("\n  }\n}\n\n");
}

/// Trims a description to its first line for the directory, capped at 120
/// characters, so a secondary tool costs roughly one prompt line.
fn append_short_description(out: &mut String, desc: &str) {
    let first_line = desc.split('\n').next().unwrap_or("");
    let max = 120;
    let cut: String = first_line.chars().take(max).collect();
    let truncated = cut.len() < desc.len();
    out.push_str(&cut);
    if truncated {
        out.push_str("...");
    }
}

/// Renders MCP tools for the system prompt.
///
/// Primary tools get their full function-call JSON schema like the native
/// tools; secondary tools only get a one-line directory entry, and the model
/// fetches their schema on demand with `mcp_describe`.
pub fn append_tool_schemas(out: &mut String, servers: &[McpServer]) {
    for s in servers {
        for t in &s.tools {
            if t.primary {
                append_one_schema(out, &s.name, t);
            }
        }
    }

    let have_secondary = servers.iter().any(|s| s.tools.iter().any(|t| !t.primary));
    if !have_secondary {
        return;
    }

    out.push_str(
        "### MCP Tool Directory\n\n\
         These additional tools exist but their parameter schemas are not \
         loaded. Before the first use of one, call mcp_describe with its full \
         name to get the schema:\n\n\
         {\n\
         \x20 \"type\": \"function\",\n\
         \x20 \"function\": {\n\
         \x20   \"name\": \"mcp_describe\",\n\
         \x20   \"description\": \"Return the full parameter schema of directory MCP tools. Accepts one or more space-separated tool names.\",\n\
         \x20   \"parameters\": {\n\
         \x20     \"type\": \"object\",\n\
         \x20     \"properties\": {\n\
         \x20       \"tools\": {\"type\": \"string\"}\n\
         \x20     },\n\
         \x20     \"required\": [\"tools\"]\n\
         \x20   }\n\
         \x20 }\n\
         }\n\n\
         Directory:\n",
    );
    for s in servers {
        for t in &s.tools {
            if t.primary {
                continue;
            }
            let _ = write!(out, "- mcp__{}__{}: ", s.name, t.name);
            append_short_description(out, &t.description);
            out.push('\n');
        }
    }
    out.push('\n');
}

/// Appends the `# MCP Server Instructions` block: one `## <server>` section
/// per connected server that provided a non-empty `instructions` field in
/// its initialize response. Emits nothing when no server did.
pub fn append_server_instructions(out: &mut String, servers: &[McpServer]) {
    let mut wrote_header = false;
    for s in servers {
        if s.instructions.is_empty() {
            continue;
        }
        if !wrote_header {
            out.push_str("\n# MCP Server Instructions\n\n");
            wrote_header = true;
        }
        let _ = writeln!(out, "## {}\n{}\n", s.name, s.instructions);
    }
}

// ============================================================================
// Tool dispatch
// ============================================================================

/// Converts DSML call arguments into a JSON `arguments` object.
///
/// Per the tools-prompt contract, `string=true` args are raw text needing
/// JSON escaping and `string=false` args are already-valid JSON literals
/// (numbers, booleans, objects, arrays) that are emitted verbatim.
#[must_use]
pub fn args_to_json(call: &ToolCall) -> String {
    let mut out = String::from("{");
    for (i, arg) in call.args.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        json_escape(&mut out, &arg.name);
        out.push(':');
        if arg.is_string {
            json_escape(&mut out, &arg.value);
        } else if arg.value.is_empty() {
            out.push_str("null");
        } else {
            out.push_str(&arg.value);
        }
    }
    out.push('}');
    out
}

/// Flattens a `tools/call` result's `content[]` into plain text.
fn append_content(out: &mut String, result: &Json) {
    let Some(Json::Arr(content)) = result.get("content") else {
        return;
    };
    for item in content {
        let Some(Json::Str(text)) = item.get("text") else {
            continue;
        };
        out.push_str(text);
        if !text.ends_with('\n') {
            out.push('\n');
        }
    }
}

fn split_name(full_name: &str) -> Option<(&str, &str)> {
    let rest = full_name.strip_prefix("mcp__")?;
    let sep = rest.find("__")?;
    Some((&rest[..sep], &rest[sep + 2..]))
}

/// Executes one `mcp__<server>__<tool>` call, mirroring `agent_tool_mcp_call`.
pub fn tool_mcp_call(servers: &mut [McpServer], call: &ToolCall) -> String {
    let Some((server_name, tool_name)) = split_name(&call.name) else {
        return "Tool error: malformed mcp tool name, expected mcp__server__tool\n".to_string();
    };
    let Some(server) = servers
        .iter_mut()
        .find(|s| s.name == server_name && s.alive)
    else {
        return "Tool error: mcp server not available\n".to_string();
    };
    if server.find_tool(tool_name).is_none() {
        return format!("Tool error: unknown mcp tool: {}\n", call.name);
    }

    let mut params = String::from("{\"name\":");
    json_escape(&mut params, tool_name);
    params.push_str(",\"arguments\":");
    params.push_str(&args_to_json(call));
    params.push('}');

    let result = match server.request("tools/call", &params) {
        Ok(result) => result,
        Err(err) => return format!("Tool error: mcp call failed: {err}\n"),
    };

    let mut out = String::new();
    if matches!(result.get("isError"), Some(Json::Bool(true))) {
        out.push_str("Tool error: ");
    }
    append_content(&mut out, &result);
    if out.is_empty() {
        out.push_str("(no output)\n");
    }
    out
}

/// `mcp_describe`: returns the full schema of directory (secondary) tools.
///
/// Mirrors `agent_tool_mcp_describe` so the model can invoke directory tools
/// without carrying every schema in the system prompt.
#[must_use]
pub fn tool_mcp_describe(servers: &[McpServer], call: &ToolCall) -> String {
    let Some(names) = call.arg_value("tools").filter(|n| !n.is_empty()) else {
        return "Tool error: mcp_describe requires tools\n".to_string();
    };

    let mut out = String::new();
    let mut found = 0usize;
    for name in names
        .split([' ', '\t', '\n', ','])
        .filter(|n| !n.is_empty())
    {
        let resolved = split_name(name).and_then(|(server_name, tool_name)| {
            servers
                .iter()
                .find(|s| s.name == server_name)
                .and_then(|s| s.find_tool(tool_name).map(|t| (s, t)))
        });
        match resolved {
            Some((s, t)) => {
                append_one_schema(&mut out, &s.name, t);
                found += 1;
            }
            None => {
                let _ = writeln!(out, "Tool error: unknown mcp tool: {name}");
            }
        }
    }
    if found == 0 && out.is_empty() {
        out.push_str("Tool error: mcp_describe found no tool names\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsml::ToolArg;

    fn write_temp_config(contents: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "plank_mcp_test_{}_{}.json",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, contents).expect("write test config");
        path
    }

    /// Builds an in-memory two-tool server (one primary, one directory-only)
    /// for prompt-rendering and `mcp_describe` tests; the subprocess is inert.
    fn make_split_server() -> McpServer {
        let cfg = McpServerConfig {
            name: "demo".to_string(),
            command: "cat".to_string(),
            args: Vec::new(),
            env: Vec::new(),
            primary_tools: None,
        };
        let mut s = McpServer::spawn(&cfg).expect("spawn cat");
        s.tools = vec![
            McpTool {
                name: "alpha".to_string(),
                description: "Primary tool.".to_string(),
                schema_json: "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}"
                    .to_string(),
                primary: true,
            },
            McpTool {
                name: "omega".to_string(),
                description: "Secondary tool.\nSecond line not shown.".to_string(),
                schema_json: "{\"type\":\"object\",\"properties\":{\"o\":{\"type\":\"number\"}}}"
                    .to_string(),
                primary: false,
            },
        ];
        s
    }

    #[test]
    fn json_round_trip() {
        let text = "{\"name\":\"echo\",\"description\":\"say hi\",\"n\":3,\"ok\":true,\
                    \"tags\":[\"a\",\"b\"],\"nested\":{\"x\":1}}";
        let v = json_parse(text).expect("parse");
        assert_eq!(v.str_or("name", ""), "echo");
        assert_eq!(v.get("n"), Some(&Json::Num(3.0)));
        assert_eq!(v.get("ok"), Some(&Json::Bool(true)));
        let Some(Json::Arr(tags)) = v.get("tags") else {
            panic!("tags not an array")
        };
        assert_eq!(tags[1], Json::Str("b".to_string()));
        let x = v.get("nested").and_then(|n| n.get("x"));
        assert_eq!(x, Some(&Json::Num(1.0)));
    }

    #[test]
    fn json_escape_round_trips_through_writer() {
        let mut out = String::new();
        json_escape(&mut out, "line\n\"quoted\"\ttab");
        let v = json_parse(&out).expect("parse");
        assert_eq!(v, Json::Str("line\n\"quoted\"\ttab".to_string()));
    }

    #[test]
    fn json_write_re_emits_schema() {
        let schema = "{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}";
        let v = json_parse(schema).expect("parse");
        let mut out = String::new();
        json_write(&mut out, &v);
        let v2 = json_parse(&out).expect("reparse");
        let q = v2.get("properties").and_then(|p| p.get("q"));
        assert_eq!(q.map(|q| q.str_or("type", "")), Some("string"));
    }

    #[test]
    fn args_to_json_mixes_string_and_raw() {
        let call = ToolCall {
            name: "mcp__demo__x".to_string(),
            args: vec![
                ToolArg {
                    name: "query".to_string(),
                    value: "hello \"world\"".to_string(),
                    is_string: true,
                },
                ToolArg {
                    name: "limit".to_string(),
                    value: "5".to_string(),
                    is_string: false,
                },
            ],
        };
        let json = args_to_json(&call);
        let v = json_parse(&json).expect("parse");
        assert_eq!(v.str_or("query", ""), "hello \"world\"");
        assert_eq!(v.get("limit"), Some(&Json::Num(5.0)));
    }

    #[test]
    fn config_load_parses_servers() {
        let path = write_temp_config(
            "{\"mcpServers\":{\"demo\":{\"command\":\"echo\",\"args\":[\"hi\"],\
             \"env\":{\"FOO\":\"bar\"}}}}",
        );
        let list = config_load(&path);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "demo");
        assert_eq!(list[0].command, "echo");
        assert_eq!(list[0].args, vec!["hi".to_string()]);
        assert_eq!(list[0].env, vec![("FOO".to_string(), "bar".to_string())]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn config_load_missing_file_is_not_an_error() {
        let list = config_load(Path::new("/tmp/plank_mcp_test_does_not_exist.json"));
        assert!(list.is_empty());
    }

    #[test]
    fn config_load_rejects_double_underscore_names() {
        let path = write_temp_config("{\"mcpServers\":{\"bad__name\":{\"command\":\"echo\"}}}");
        assert!(config_load(&path).is_empty());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn hierarchy_local_overrides_global_by_server_name() {
        let global = write_temp_config(
            "{\"mcpServers\":{\"shared\":{\"command\":\"global-cmd\"},\
             \"only_global\":{\"command\":\"g\"}}}",
        );
        let local = write_temp_config(
            "{\"mcpServers\":{\"shared\":{\"command\":\"local-cmd\"},\
             \"only_local\":{\"command\":\"l\"}}}",
        );
        let merged = merge_configs(config_load(&global), config_load(&local));
        let by_name = |n: &str| {
            merged
                .iter()
                .find(|c| c.name == n)
                .map(|c| c.command.clone())
        };
        assert_eq!(by_name("shared").as_deref(), Some("local-cmd"));
        assert_eq!(by_name("only_global").as_deref(), Some("g"));
        assert_eq!(by_name("only_local").as_deref(), Some("l"));
        std::fs::remove_file(global).ok();
        std::fs::remove_file(local).ok();
    }

    #[test]
    fn config_load_parses_primary_tools() {
        let path = write_temp_config(
            "{\"mcpServers\":{\"demo\":{\"command\":\"echo\",\
             \"primaryTools\":[\"alpha\",\"beta\"]}}}",
        );
        let list = config_load(&path);
        assert_eq!(
            list[0].primary_tools,
            Some(vec!["alpha".to_string(), "beta".to_string()])
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn json_string_larger_than_128kb_survives_round_trip() {
        let n = 200 * 1024;
        let big = "a".repeat(n);
        let mut escaped = String::new();
        json_escape(&mut escaped, &big);
        let v = json_parse(&escaped).expect("parse");
        assert_eq!(v, Json::Str(big));
    }

    #[test]
    fn tool_schemas_split_primary_and_directory() {
        let s = make_split_server();
        let mut out = String::new();
        append_tool_schemas(&mut out, std::slice::from_ref(&s));

        assert!(out.contains("\"name\": \"mcp__demo__alpha\""));
        assert!(out.contains("MCP Tool Directory"));
        assert!(out.contains("\"name\": \"mcp_describe\""));
        assert!(out.contains("- mcp__demo__omega: Secondary tool...."));
        // The secondary tool's schema and description tail stay out.
        assert!(!out.contains("\"name\": \"mcp__demo__omega\""));
        assert!(!out.contains("Second line"));
    }

    #[test]
    fn describe_returns_directory_tool_schema() {
        let s = make_split_server();
        let servers = [s];
        let call = ToolCall {
            name: "mcp_describe".to_string(),
            args: vec![ToolArg {
                name: "tools".to_string(),
                value: "mcp__demo__omega mcp__demo__nope".to_string(),
                is_string: true,
            }],
        };
        let out = tool_mcp_describe(&servers, &call);
        assert!(out.contains("\"name\": \"mcp__demo__omega\""));
        assert!(out.contains("\"o\":{\"type\":\"number\"}"));
        assert!(out.contains("unknown mcp tool: mcp__demo__nope"));
    }

    #[test]
    fn server_instructions_render_as_prompt_block() {
        let mut s = make_split_server();
        let mut out = String::new();
        append_server_instructions(&mut out, std::slice::from_ref(&s));
        assert!(out.is_empty(), "no instructions -> no block: {out:?}");
        s.instructions = "Use alpha before omega.".to_string();
        append_server_instructions(&mut out, std::slice::from_ref(&s));
        assert!(
            out.starts_with("\n# MCP Server Instructions\n\n"),
            "{out:?}"
        );
        assert!(
            out.contains("## demo\nUse alpha before omega.\n"),
            "{out:?}"
        );
    }

    #[test]
    fn end_to_end_against_scripted_stdio_server() {
        // A tiny shell MCP server: answers initialize, ignores the
        // notification, lists one echo tool, and echoes tool call arguments.
        let script = r#"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *'"initialize"'*)
      printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":${id:-0},\"result\":{\"protocolVersion\":\"2024-11-05\",\"instructions\":\"Prefer the echo tool for text round-trips.\"}}" ;;
    *'"tools/list"'*)
      printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":${id:-0},\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo text.\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}}]}}" ;;
    *'"tools/call"'*)
      printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":${id:-0},\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"echoed: hi\"}]}}" ;;
    *)
      # Unknown methods (e.g. resources/list) get a JSON-RPC error reply,
      # matching a real server that doesn't implement the method — this
      # must not hang or take down the connection.
      printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":${id:-0},\"error\":{\"message\":\"method not found\"}}" ;;
  esac
done
"#;
        let path = write_temp_config(&format!(
            "{{\"mcpServers\":{{\"demo\":{{\"command\":\"sh\",\"args\":[\"-c\",{}]}}}}}}",
            {
                let mut esc = String::new();
                json_escape(&mut esc, script);
                esc
            }
        ));
        // start_servers + config_load keeps the test hermetic: the merged
        // hierarchy would also pick up the user's real ~/.plank/.mcp.json.
        let mut servers = start_servers(config_load(&path));
        std::fs::remove_file(&path).ok();
        assert_eq!(servers.len(), 1, "server should start and handshake");
        assert_eq!(servers[0].tools.len(), 1);
        assert_eq!(servers[0].tools[0].name, "echo");
        assert!(servers[0].tools[0].primary);
        assert_eq!(
            servers[0].instructions,
            "Prefer the echo tool for text round-trips."
        );

        let call = ToolCall {
            name: "mcp__demo__echo".to_string(),
            args: vec![ToolArg {
                name: "text".to_string(),
                value: "hi".to_string(),
                is_string: true,
            }],
        };
        let out = tool_mcp_call(&mut servers, &call);
        assert_eq!(out, "echoed: hi\n");
    }

    #[test]
    fn a_server_without_a_resources_capability_is_never_asked_for_resources() {
        // Regression: a server that silently ignores unknown methods would
        // stall the handshake for the full 30s timeout on `resources/list` and
        // then be marked dead, losing every one of its tools.
        let log = std::env::temp_dir().join(format!("plank-mcp-methods-{}", std::process::id()));
        std::fs::remove_file(&log).ok();
        let script = format!(
            r#"
while IFS= read -r line; do
  printf '%s\n' "$line" >> {log}
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *'"initialize"'*)
      printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":${{id:-0}},\"result\":{{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{{\"tools\":{{}}}}}}}}" ;;
    *'"tools/list"'*)
      printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":${{id:-0}},\"result\":{{\"tools\":[{{\"name\":\"echo\",\"description\":\"Echo.\",\"inputSchema\":{{\"type\":\"object\"}}}}]}}}}" ;;
    *)
      : ;;  # silently ignore anything else
  esac
done
"#,
            log = log.display()
        );
        let path = write_temp_config(&format!(
            "{{\"mcpServers\":{{\"quiet\":{{\"command\":\"sh\",\"args\":[\"-c\",{}]}}}}}}",
            {
                let mut esc = String::new();
                json_escape(&mut esc, &script);
                esc
            }
        ));
        let started = Instant::now();
        let servers = start_servers(config_load(&path));
        let elapsed = started.elapsed();
        std::fs::remove_file(&path).ok();
        let seen = std::fs::read_to_string(&log).unwrap_or_default();
        std::fs::remove_file(&log).ok();

        assert!(
            elapsed < Duration::from_secs(MCP_TIMEOUT_SEC),
            "handshake must not stall on resources/list: {elapsed:?}"
        );
        assert!(
            !seen.contains("resources/list"),
            "resources/list must not be sent: {seen}"
        );
        assert_eq!(servers.len(), 1, "the server must survive");
        assert_eq!(servers[0].tools.len(), 1, "its tools must survive");
        assert!(servers[0].resources.is_empty());
    }

    #[test]
    fn parses_a_resources_list_response() {
        let json = r#"{"resources":[
            {"uri":"file:///a.txt","name":"A"},
            {"uri":"note://b","name":"B"}
        ]}"#;
        let root = json_parse(json).expect("parses");
        let got = parse_resources(&root);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].uri, "file:///a.txt");
        assert_eq!(got[0].name, "A");
        assert_eq!(got[1].uri, "note://b");
    }

    #[test]
    fn a_missing_resources_key_yields_none() {
        let root = json_parse("{}").expect("parses");
        assert!(parse_resources(&root).is_empty());
    }

    #[test]
    fn resource_candidates_are_server_qualified() {
        let r = vec![McpResource {
            uri: "note://b".to_string(),
            name: "B".to_string(),
        }];
        let c = resource_candidates_from("tolaria", &r);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].text, "tolaria:note://b");
        assert_eq!(c[0].kind, crate::complete::Kind::Resource);
    }
}
