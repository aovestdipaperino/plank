//! `plank remote <url>`: the interactive remote-control *client* (issue #25).
//!
//! Connects to a running plank instance's remote-control WebSocket (started
//! with `--control`), authenticates with the shared bearer token, and becomes
//! another front-end over the same [`crate::worker::UiEvent`] stream: mirrored
//! server output streams to the terminal while typed lines are sent as
//! `prompt` / `command` / `btw` frames and Ctrl-C as `interrupt`.
//!
//! Transport is the same blocking [`tungstenite`] client the server speaks (no
//! async runtime). Because a single [`tungstenite::WebSocket`] cannot be read
//! and written from two threads at once, the client owns the socket on the main
//! thread and polls it with a short read timeout (the server's `mirror_loop`
//! idiom); stdin is read on a helper thread and delivered over a channel. Only
//! `ws://` is supported directly — reach a remote box through the SSH tunnel the
//! server prints, so the client always talks to a loopback port.

use std::io::Write;
use std::net::TcpStream;
use std::time::Duration;

use tungstenite::Message;
use tungstenite::http::Uri;

use super::control::{ClientFrame, ClientMsg, ServerFrame, ServerMsg};

/// How often the client loop wakes to poll the socket and flush outbound input.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// A connected, authenticated remote-control client session.
///
/// Owns the WebSocket; read it with [`RemoteClient::poll`] and drive it with
/// [`RemoteClient::send`]. The read timeout is set so `poll` returns `Ok(None)`
/// promptly when no frame is pending, letting the caller interleave input.
pub struct RemoteClient {
    ws: tungstenite::WebSocket<TcpStream>,
    /// Session id assigned by the server in its `hello` frame.
    pub session_id: u64,
    /// Whether this session was granted control on connect.
    pub controller: bool,
    /// Server-reported plank version.
    pub plank_version: String,
}

impl std::fmt::Debug for RemoteClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteClient")
            .field("session_id", &self.session_id)
            .field("controller", &self.controller)
            .field("plank_version", &self.plank_version)
            .finish_non_exhaustive()
    }
}

/// Parses a remote-control URL into a `(uri, host, port)` triple.
///
/// Accepts `ws://host:port[/path]` (or a bare `host:port`, defaulted to `ws`).
/// `wss://` is rejected: TLS is out of scope; tunnel to a loopback `ws://` port.
fn parse_ws_url(url: &str) -> Result<(Uri, String, u16), String> {
    let normalized = if url.contains("://") {
        url.to_owned()
    } else {
        format!("ws://{url}")
    };
    let (scheme, _) = normalized
        .split_once("://")
        .ok_or_else(|| format!("invalid remote URL: {url}"))?;
    match scheme {
        "ws" => {}
        "wss" => {
            return Err(
                "wss:// (TLS) is not supported; tunnel to a loopback ws:// port with SSH"
                    .to_owned(),
            );
        }
        other => return Err(format!("unsupported remote URL scheme: {other}")),
    }
    let uri: Uri = normalized
        .parse()
        .map_err(|e| format!("invalid remote URL {url}: {e}"))?;
    let host = uri
        .host()
        .ok_or_else(|| format!("remote URL has no host: {url}"))?
        .to_owned();
    let port = uri.port_u16().unwrap_or(80);
    Ok((uri, host, port))
}

impl RemoteClient {
    /// Connects to `url`, performs the WebSocket handshake, authenticates with
    /// `token`, and consumes the `hello` frame. `resume_from` requests scrollback
    /// replay newer than the given bus id.
    ///
    /// # Errors
    /// Returns a message on URL, connection, handshake, or auth failure (the
    /// server closes the socket on a bad token, surfaced here as an error).
    pub fn connect(url: &str, token: &str, resume_from: Option<u64>) -> Result<Self, String> {
        let (uri, host, port) = parse_ws_url(url)?;
        let stream =
            TcpStream::connect((host.as_str(), port)).map_err(|e| format!("connect: {e}"))?;
        stream
            .set_read_timeout(Some(POLL_INTERVAL))
            .map_err(|e| e.to_string())?;
        let (mut ws, _resp) =
            tungstenite::client(uri, stream).map_err(|e| format!("ws handshake: {e}"))?;

        // Mandatory auth first frame.
        send_frame(
            &mut ws,
            &ClientFrame::new(ClientMsg::Auth {
                token: token.to_owned(),
                resume_from,
            }),
        )?;

        // The server replies `hello` on success or closes the socket on a bad
        // token. Block (bounded) for the first server frame.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if std::time::Instant::now() > deadline {
                return Err("timed out waiting for server hello".to_owned());
            }
            match ws.read() {
                Ok(Message::Text(txt)) => {
                    let frame: ServerFrame =
                        serde_json::from_str(&txt).map_err(|e| e.to_string())?;
                    match frame.msg {
                        ServerMsg::Hello {
                            session_id,
                            controller,
                            plank_version,
                            ..
                        } => {
                            return Ok(Self {
                                ws,
                                session_id,
                                controller,
                                plank_version,
                            });
                        }
                        other => return Err(format!("expected hello, got {other:?}")),
                    }
                }
                Ok(Message::Close(_))
                | Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => {
                    return Err("authentication rejected by server".to_owned());
                }
                Ok(_) => {}
                Err(tungstenite::Error::Io(e))
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) => {}
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    /// Sends one client message.
    ///
    /// # Errors
    /// Returns a message if the socket write fails.
    pub fn send(&mut self, msg: ClientMsg) -> Result<(), String> {
        send_frame(&mut self.ws, &ClientFrame::new(msg))
    }

    /// Polls for one server frame. Returns `Ok(None)` when nothing is pending
    /// within the read timeout, `Ok(Some(frame))` for a received frame.
    ///
    /// # Errors
    /// Returns a message when the connection is closed or a socket error occurs.
    pub fn poll(&mut self) -> Result<Option<ServerFrame>, String> {
        match self.ws.read() {
            Ok(Message::Text(txt)) => {
                let frame: ServerFrame = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
                Ok(Some(frame))
            }
            Ok(Message::Close(_)) => Err("server closed the connection".to_owned()),
            Ok(Message::Ping(p)) => {
                let _ = self.ws.send(Message::Pong(p));
                Ok(None)
            }
            Ok(_) => Ok(None),
            Err(tungstenite::Error::Io(e))
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                Ok(None)
            }
            Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => {
                Err("server closed the connection".to_owned())
            }
            Err(e) => Err(e.to_string()),
        }
    }
}

fn send_frame(
    ws: &mut tungstenite::WebSocket<TcpStream>,
    frame: &ClientFrame,
) -> Result<(), String> {
    let json = frame.to_json().map_err(|e| e.to_string())?;
    ws.send(Message::Text(json)).map_err(|e| e.to_string())?;
    ws.flush().map_err(|e| e.to_string())
}

/// Classifies a typed line into the client frame to send: `/btw <q>` → a `btw`
/// side question (allowed even from a mirror), any other `/…` → a `command`
/// (routed through the server's slash dispatcher), plain text → a `prompt`.
#[must_use]
pub fn classify_line(line: &str) -> ClientMsg {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("/btw") {
        let q = rest.trim();
        if !q.is_empty() {
            return ClientMsg::Btw { text: q.to_owned() };
        }
    }
    if trimmed.starts_with('/') {
        ClientMsg::Command {
            text: trimmed.to_owned(),
        }
    } else {
        ClientMsg::Prompt {
            text: trimmed.to_owned(),
        }
    }
}

/// Renders one server message to a terminal line/segment on stdout. Mirrors the
/// text-bearing events; status footers and structural markers are omitted (the
/// client is a plain stream, not a full-screen UI).
fn render_server_msg(msg: &ServerMsg) {
    let mut out = std::io::stdout();
    match msg {
        ServerMsg::Visible { text }
        | ServerMsg::Think { text }
        | ServerMsg::Tool { text }
        | ServerMsg::Error { text }
        | ServerMsg::Dim { text }
        | ServerMsg::Plain { text }
        | ServerMsg::UserEcho { text } => {
            let _ = write!(out, "{text}");
        }
        ServerMsg::EndLine => {
            let _ = writeln!(out);
        }
        ServerMsg::Snapshot { scrollback, .. } => {
            for entry in scrollback {
                render_server_msg(&entry.msg);
            }
        }
        ServerMsg::ControlDenied { reason } => {
            let _ = writeln!(out, "[control denied: {reason}]");
        }
        ServerMsg::Bye { reason } => {
            let _ = writeln!(out, "[server: {reason}]");
        }
        // Hello is consumed during connect; Pong/Status/Btw markers are silent.
        _ => {}
    }
    let _ = out.flush();
}

/// Runs the interactive `plank remote <url>` client to completion: streams
/// mirrored output to the terminal and forwards stdin lines + Ctrl-C until the
/// server closes the connection or stdin reaches EOF.
///
/// The bearer token comes from `token` (the `--token` flag) or, when that is
/// `None`, the `PLANK_REMOTE_TOKEN` environment variable.
///
/// # Errors
/// Returns a message on connection/auth failure. A normal server close or stdin
/// EOF is a clean exit (`Ok`).
pub fn run(url: &str, token: Option<String>) -> Result<(), String> {
    let token = token
        .or_else(|| {
            std::env::var("PLANK_REMOTE_TOKEN")
                .ok()
                .filter(|t| !t.is_empty())
        })
        .ok_or_else(|| {
            "no token: pass --token <t> or set PLANK_REMOTE_TOKEN (see the server's startup log)"
                .to_owned()
        })?;

    let mut client = RemoteClient::connect(url, &token, None)?;
    eprintln!(
        "plank: connected to remote plank {} (session {}, {})",
        client.plank_version,
        client.session_id,
        if client.controller {
            "controller"
        } else {
            "mirror"
        }
    );

    // Ctrl-C sends an `interrupt` frame rather than killing the client.
    crate::interrupt::install();

    // stdin on a helper thread → channel; the main thread owns the socket.
    let (line_tx, line_rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        loop {
            let mut line = String::new();
            match lock.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if line_tx.send(line).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let mut stdin_open = true;
    loop {
        // Outbound: interrupt, then any pending typed line.
        if crate::interrupt::pending() {
            crate::interrupt::clear();
            client.send(ClientMsg::Interrupt)?;
        }
        if stdin_open {
            match line_rx.try_recv() {
                Ok(line) => {
                    if line.trim().is_empty() {
                        // skip blank input
                    } else {
                        client.send(classify_line(&line))?;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => stdin_open = false,
            }
        }

        // Inbound: one poll (blocks up to the read timeout).
        match client.poll() {
            Ok(Some(frame)) => render_server_msg(&frame.msg),
            Ok(None) => {}
            Err(_) => return Ok(()), // server closed; clean exit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_prompt_command_and_btw() {
        assert_eq!(
            classify_line("hello there"),
            ClientMsg::Prompt {
                text: "hello there".into()
            }
        );
        assert_eq!(
            classify_line("/help"),
            ClientMsg::Command {
                text: "/help".into()
            }
        );
        assert_eq!(
            classify_line("/btw   why?"),
            ClientMsg::Btw {
                text: "why?".into()
            }
        );
        // A bare `/btw` with no question is treated as a command, not a btw.
        assert_eq!(
            classify_line("/btw"),
            ClientMsg::Command {
                text: "/btw".into()
            }
        );
    }

    #[test]
    fn parse_ws_url_variants() {
        assert!(parse_ws_url("ws://127.0.0.1:9000/").is_ok());
        assert!(parse_ws_url("127.0.0.1:9000").is_ok()); // scheme defaulted
        assert!(parse_ws_url("wss://box:9000").is_err()); // TLS unsupported
        assert!(parse_ws_url("http://box:9000").is_err()); // wrong scheme
    }

    /// End-to-end round-trip against a real loopback `RemoteServer`: the client
    /// authenticates (bad token is rejected), sends a `prompt` that lands in the
    /// server's shared queue, and receives a live event mirrored from the bus.
    #[test]
    fn client_round_trip_against_loopback_server() {
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        use crate::remote::control::RemoteServer;
        use crate::worker::{BroadcastBus, TurnShared, UiEvent};

        let server = RemoteServer::start(
            "127.0.0.1:0",
            "tok".to_owned(),
            false,
            false,
            Arc::new(BroadcastBus::new()),
            Arc::new(TurnShared::default()),
        )
        .expect("server binds");
        let url = format!("ws://{}/", server.local_addr);

        // A bad token is rejected during connect.
        assert!(RemoteClient::connect(&url, "wrong", None).is_err());

        // A good token yields a controller session (headless auto-grants).
        let mut client = RemoteClient::connect(&url, "tok", None).expect("auth ok");
        assert!(client.controller, "headless first client controls");

        // A sent prompt reaches the server's shared queue.
        client
            .send(ClientMsg::Prompt {
                text: "drive me".into(),
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut queued = server.state.shared.take_queued();
        while queued.is_empty() {
            assert!(Instant::now() < deadline, "prompt never queued");
            std::thread::sleep(Duration::from_millis(10));
            queued = server.state.shared.take_queued();
        }
        assert_eq!(queued, vec!["drive me"]);

        // A live bus event is mirrored to the client.
        server
            .state
            .bus
            .broadcast(UiEvent::Visible("streamed".into()));
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut got = String::new();
        while !got.contains("streamed") {
            assert!(Instant::now() < deadline, "mirrored event never arrived");
            if let Some(frame) = client.poll().unwrap()
                && let ServerMsg::Visible { text } = frame.msg
            {
                got.push_str(&text);
            }
        }

        // The interrupt frame is accepted and drives the shared flag.
        client.send(ClientMsg::Interrupt).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while !server.state.shared.interrupt.load(Ordering::Relaxed) {
            assert!(Instant::now() < deadline, "interrupt never propagated");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
