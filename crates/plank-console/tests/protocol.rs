// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use plank_console::registry::{Server, ServerEvent};

/// Sends a handshake to the control port and returns the reply line.
fn hello(port: u16, line: &str) -> String {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.write_all(format!("{line}\n").as_bytes()).unwrap();
    let mut r = BufReader::new(s);
    let mut reply = String::new();
    r.read_line(&mut reply).unwrap();
    reply.trim_end().to_string()
}

/// Drains events for up to a second, until `f` finds what it wants.
fn wait_for<T>(server: &Server, mut f: impl FnMut(&ServerEvent) -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        while let Ok(ev) = server.events().try_recv() {
            if let Some(v) = f(&ev) {
                return v;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for event");
}

#[test]
fn hello_allocates_a_data_port_and_opens_a_session() {
    let server = Server::bind(0).unwrap();
    let reply = hello(server.control_port(), "HELLO alpha");
    let port: u16 = reply.strip_prefix("PORT ").unwrap().parse().unwrap();
    assert_ne!(port, server.control_port());

    let (name, ev_port) = wait_for(&server, |ev| match ev {
        ServerEvent::Opened { name, port, .. } => Some((name.clone(), *port)),
        _ => None,
    });
    assert_eq!(name, "alpha");
    assert_eq!(ev_port, port);

    // The advertised port really accepts a stream.
    let mut data = TcpStream::connect(("127.0.0.1", port)).unwrap();
    data.write_all(b"tokens").unwrap();
    let got = wait_for(&server, |ev| match ev {
        ServerEvent::Bytes { data, .. } => Some(data.clone()),
        _ => None,
    });
    assert_eq!(got, b"tokens");
}

#[test]
fn the_same_name_returns_the_same_port_and_reconnects() {
    let server = Server::bind(0).unwrap();
    let first = hello(server.control_port(), "HELLO beta");
    let mut data = TcpStream::connect((
        "127.0.0.1",
        first.strip_prefix("PORT ").unwrap().parse::<u16>().unwrap(),
    ))
    .unwrap();
    data.write_all(b"a").unwrap();
    drop(data);
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Disconnected { .. }).then_some(())
    });

    let second = hello(server.control_port(), "HELLO beta");
    assert_eq!(first, second, "a known name must keep its port");
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Reconnected { .. }).then_some(())
    });
}

#[test]
fn a_dropped_data_socket_leaves_the_session_listening() {
    let server = Server::bind(0).unwrap();
    let reply = hello(server.control_port(), "HELLO gamma");
    let port: u16 = reply.strip_prefix("PORT ").unwrap().parse().unwrap();

    let data = TcpStream::connect(("127.0.0.1", port)).unwrap();
    drop(data);
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Disconnected { .. }).then_some(())
    });

    let mut again = TcpStream::connect(("127.0.0.1", port)).unwrap();
    again.write_all(b"back").unwrap();
    let got = wait_for(&server, |ev| match ev {
        ServerEvent::Bytes { data, .. } => Some(data.clone()),
        _ => None,
    });
    assert_eq!(got, b"back");
}

#[test]
fn a_second_live_writer_is_refused() {
    let server = Server::bind(0).unwrap();
    let reply = hello(server.control_port(), "HELLO delta");
    let port: u16 = reply.strip_prefix("PORT ").unwrap().parse().unwrap();

    let mut first = TcpStream::connect(("127.0.0.1", port)).unwrap();
    first.write_all(b"x").unwrap();
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Bytes { .. }).then_some(())
    });

    let mut second = TcpStream::connect(("127.0.0.1", port)).unwrap();
    second
        .set_read_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    let mut buf = Vec::new();
    second.read_to_end(&mut buf).unwrap();
    assert!(
        String::from_utf8_lossy(&buf).starts_with("ERR"),
        "a duplicate writer must be closed with a banner, got {buf:?}"
    );
}

#[test]
fn bad_names_are_refused_and_the_server_stays_up() {
    let server = Server::bind(0).unwrap();
    assert_eq!(hello(server.control_port(), "HELLO "), "ERR bad name");
    assert_eq!(
        hello(server.control_port(), &format!("HELLO {}", "x".repeat(65))),
        "ERR bad name"
    );
    assert!(hello(server.control_port(), "HELLO ok").starts_with("PORT "));
}

#[test]
fn a_non_hello_first_line_becomes_an_anonymous_session() {
    let server = Server::bind(0).unwrap();
    let mut s = TcpStream::connect(("127.0.0.1", server.control_port())).unwrap();
    s.write_all(b"just some tokens\n").unwrap();

    let name = wait_for(&server, |ev| match ev {
        ServerEvent::Opened { name, .. } => Some(name.clone()),
        _ => None,
    });
    assert!(name.starts_with("anon-"), "got {name}");

    let got = wait_for(&server, |ev| match ev {
        ServerEvent::Bytes { data, .. } => Some(data.clone()),
        _ => None,
    });
    assert_eq!(got, b"just some tokens\n");
}

#[test]
fn a_reconnect_clears_idle_since_so_the_session_never_reaps_while_reused() {
    let mut server = Server::bind(0).unwrap();
    let reply = hello(server.control_port(), "HELLO epsilon");
    let port: u16 = reply.strip_prefix("PORT ").unwrap().parse().unwrap();

    // Attach and detach once so idle_since gets set.
    let data = TcpStream::connect(("127.0.0.1", port)).unwrap();
    drop(data);
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Disconnected { .. }).then_some(())
    });

    // Reconnecting via HELLO must clear idle_since immediately, even though
    // no new data socket has attached yet.
    let second = hello(server.control_port(), "HELLO epsilon");
    assert_eq!(reply, second);
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Reconnected { .. }).then_some(())
    });

    // A zero-duration TTL would reap anything with idle_since still set;
    // since the reconnect cleared it, the session must survive. Poll for a
    // bounded window: since nothing should ever arrive, we can only prove
    // absence by waiting out a deadline, not by asserting after one recv.
    server.reap(Duration::from_secs(0));
    let deadline = Instant::now() + Duration::from_millis(300);
    while Instant::now() < deadline {
        if let Ok(ev) = server.events().try_recv() {
            assert!(
                !matches!(ev, ServerEvent::Closed { .. }),
                "reconnected session must not be reaped, got {ev:?}"
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    // And the data port still works.
    let mut again = TcpStream::connect(("127.0.0.1", port)).unwrap();
    again.write_all(b"still-here").unwrap();
    let got = wait_for(&server, |ev| match ev {
        ServerEvent::Bytes { data, .. } => Some(data.clone()),
        _ => None,
    });
    assert_eq!(got, b"still-here");
}

#[test]
fn reaping_sends_closed_and_releases_the_session_listener_port() {
    let mut server = Server::bind(0).unwrap();
    let reply = hello(server.control_port(), "HELLO zeta");
    let port: u16 = reply.strip_prefix("PORT ").unwrap().parse().unwrap();

    // Never attach a data socket; idle_since was set the instant the
    // session was created, so a zero-duration TTL makes it reapable right
    // away.
    server.reap(Duration::from_secs(0));

    let closed_id = wait_for(&server, |ev| match ev {
        ServerEvent::Closed { id } => Some(*id),
        _ => None,
    });
    assert!(closed_id > 0);

    // The port must actually be free: a fresh bind on the same port must
    // succeed. The listener thread may take a moment after `reap` returns
    // to notice the shutdown flag and drop its `TcpListener`, so poll a
    // real condition against a deadline rather than assuming it happened
    // synchronously.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => break,
            Err(e) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
                let _ = e;
            }
            Err(e) => panic!("port {port} was not released after reap: {e}"),
        }
    }
}

#[test]
fn an_anonymous_session_becomes_reapable_after_its_connection_ends() {
    let mut server = Server::bind(0).unwrap();
    let mut s = TcpStream::connect(("127.0.0.1", server.control_port())).unwrap();
    s.write_all(b"raw bytes\n").unwrap();

    let id = wait_for(&server, |ev| match ev {
        ServerEvent::Opened { id, .. } => Some(*id),
        _ => None,
    });

    drop(s);
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Disconnected { id: eid } if *eid == id).then_some(())
    });

    // Now that the connection has ended, a zero-duration TTL must reap it.
    server.reap(Duration::from_secs(0));
    let closed_id = wait_for(&server, |ev| match ev {
        ServerEvent::Closed { id } => Some(*id),
        _ => None,
    });
    assert_eq!(closed_id, id);
}
