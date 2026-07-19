//! End-to-end tests for the flavor-(a) `RemoteDs4Engine` client (issue #26).
//!
//! A hand-rolled `std::net::TcpListener` mock speaks the minimal HTTP/1.1 +
//! SSE the client needs, so the whole remote turn loop is exercised in CI with
//! no model and no async runtime. This is the remote analogue of how
//! `EchoEngine` exercises the local turn loop.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use plank::engine::{Engine, EngineEvent, GenerationOptions};
use plank::remote::ds4_client::RemoteDs4Engine;

/// Reads one HTTP request off `stream`, returning (method, path, body).
fn read_request(reader: &mut BufReader<&std::net::TcpStream>) -> Option<(String, String, String)> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).ok()? == 0 {
        return None;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(v) = trimmed.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }
    Some((method, path, String::from_utf8_lossy(&body).into_owned()))
}

fn write_json(stream: &mut std::net::TcpStream, body: &str) {
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
}

/// Starts the mock server on an ephemeral port; returns `(base_url, shutdown)`.
/// `slow` inserts a tiny delay between SSE frames so an interrupt can win.
fn start_mock(slow: bool) -> (String, Arc<AtomicBool>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);
    listener.set_nonblocking(true).unwrap();

    thread::spawn(move || {
        while !stop_thread.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut reader = BufReader::new(&stream);
                    let Some((_method, path, _body)) = read_request(&mut reader) else {
                        continue;
                    };
                    drop(reader);
                    if path == "/info" {
                        write_json(
                            &mut stream,
                            r#"{"model_name":"mock-ds4","ctx_size":4096,"protocol_version":1}"#,
                        );
                    } else if path == "/tokenize" {
                        write_json(&mut stream, r#"{"n_tokens":123}"#);
                    } else if path == "/generate" || path == "/warm" {
                        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
                        let _ = stream.write_all(head.as_bytes());
                        let frames = [
                            r#"{"type":"prefill","done":5,"total":5,"tps":10.0}"#.to_string(),
                            r#"{"type":"text","s":"hello "}"#.to_string(),
                            r#"{"type":"text","s":"world"}"#.to_string(),
                            r#"{"type":"done","stats":{"generated":2,"tps":4.0,"ctx_used":42,"interrupted":false}}"#.to_string(),
                        ];
                        for f in frames {
                            let _ = stream.write_all(format!("data: {f}\n\n").as_bytes());
                            let _ = stream.flush();
                            if slow {
                                thread::sleep(std::time::Duration::from_millis(20));
                            }
                        }
                    } else {
                        // DELETE cancel and anything else: bare 200.
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
                        );
                    }
                    let _ = stream.flush();
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    });

    (format!("http://127.0.0.1:{port}"), stop)
}

#[test]
fn remote_ds4_generate_end_to_end() {
    let (base, stop) = start_mock(false);
    let mut engine = RemoteDs4Engine::connect(&base, None).expect("connect");
    assert_eq!(engine.model_name(), "mock-ds4");
    assert_eq!(engine.ctx_size(), 4096);

    let mut text = String::new();
    let mut prefill_seen = false;
    let stats = engine
        .generate(
            plank::engine::Prompt::Flat("[system]\nx\n[user]\nhi\n"),
            &GenerationOptions::default(),
            &|| false,
            &|| false,
            &mut |ev| match ev {
                EngineEvent::Text(s) => text.push_str(&s),
                EngineEvent::Prefill(_) => prefill_seen = true,
            },
        )
        .expect("generate");

    assert!(prefill_seen);
    assert_eq!(text, "hello world");
    assert_eq!(stats.generated, 2);
    assert_eq!(stats.ctx_used, 42);
    assert!(!stats.interrupted);
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn remote_ds4_interrupt_returns_interrupted() {
    let (base, stop) = start_mock(true);
    let mut engine = RemoteDs4Engine::connect(&base, None).expect("connect");
    let stats = engine
        .generate(
            plank::engine::Prompt::Flat("hi"),
            &GenerationOptions::default(),
            &|| true, // interrupt fires immediately
            &|| false,
            &mut |_ev| {},
        )
        .expect("generate");
    assert!(stats.interrupted);
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn client_against_real_serve_with_echo_engine() {
    use plank::engine::EchoEngine;
    use plank::serve::ServeConfig;

    // Grab an ephemeral port, release it, then let `serve` re-bind it.
    let port = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let listen = format!("127.0.0.1:{port}");
    thread::spawn(move || {
        let _ = plank::serve::run(
            Box::new(EchoEngine::new(4096)),
            &ServeConfig {
                listen,
                token: Some("sekret".to_string()),
            },
        );
    });
    // Wait for the listener to come up.
    thread::sleep(std::time::Duration::from_millis(150));

    let base = format!("http://127.0.0.1:{port}");
    // Wrong/missing token must be rejected at connect (/info -> 401).
    assert!(RemoteDs4Engine::connect(&base, None).is_err());

    let mut engine =
        RemoteDs4Engine::connect(&base, Some("sekret".to_string())).expect("authorized connect");
    assert_eq!(engine.model_name(), ""); // EchoEngine has no name
    assert_eq!(engine.ctx_size(), 4096);

    let mut text = String::new();
    let stats = engine
        .generate(
            plank::engine::Prompt::Flat("[user]\nhello\n"),
            &GenerationOptions::default(),
            &|| false,
            &|| false,
            &mut |ev| {
                if let EngineEvent::Text(s) = ev {
                    text.push_str(&s);
                }
            },
        )
        .expect("generate");
    assert!(text.contains("echo engine"), "got: {text:?}");
    assert!(stats.generated > 0);
}

#[test]
fn remote_ds4_count_tokens_uses_server() {
    let (base, stop) = start_mock(false);
    let engine = RemoteDs4Engine::connect(&base, None).expect("connect");
    assert_eq!(engine.count_tokens("anything"), 123);
    // Second call is served from the memo (same value).
    assert_eq!(engine.count_tokens("anything"), 123);
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn remote_ds4_count_tokens_degrades_after_server_gone() {
    // Connect to a live mock, then shut it down: a subsequent /tokenize must
    // fall back to len()/4 (design constraint 8), never panic or error.
    let (base, stop) = start_mock(false);
    let engine = RemoteDs4Engine::connect(&base, None).expect("connect");
    stop.store(true, Ordering::Relaxed);
    thread::sleep(std::time::Duration::from_millis(80)); // let the listener close
    let text = "abcdefgh"; // 8 bytes -> 2 via len()/4
    assert_eq!(engine.count_tokens(text), 2);
}
