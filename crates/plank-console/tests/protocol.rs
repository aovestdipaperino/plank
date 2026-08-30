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
    let port: u16 = first.strip_prefix("PORT ").unwrap().parse().unwrap();
    let mut data = TcpStream::connect(("127.0.0.1", port)).unwrap();
    data.write_all(b"a").unwrap();
    drop(data);
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Disconnected { .. }).then_some(())
    });

    let second = hello(server.control_port(), "HELLO beta");
    assert_eq!(first, second, "a known name must keep its port");

    // `Attached` is the sole source of truth for "connected" (defect 1),
    // so it only fires once the client's data socket actually reattaches
    // — not on the HELLO handshake alone.
    let mut data = TcpStream::connect(("127.0.0.1", port)).unwrap();
    data.write_all(b"b").unwrap();
    let reattached = wait_for(&server, |ev| match ev {
        ServerEvent::Attached { reattached, .. } => Some(*reattached),
        _ => None,
    });
    assert!(reattached, "a repeat attach must report reattached: true");
}

/// Regression test for defect 1: the *ordinary first connection* to a
/// brand-new session — `HELLO` followed by dialing the data port, with no
/// prior handshake reconnect at all — must report `Attached {reattached:
/// false}`, which is what the UI now uses to mark the session connected.
/// Before the fix, nothing at all told the UI a fresh session's data
/// socket had attached; the window stayed titled `[disconnected]` for its
/// entire first connection.
#[test]
fn the_ordinary_first_attach_reports_attached_not_reattached() {
    let server = Server::bind(0).unwrap();
    let reply = hello(server.control_port(), "HELLO first-timer");
    let port: u16 = reply.strip_prefix("PORT ").unwrap().parse().unwrap();

    let mut data = TcpStream::connect(("127.0.0.1", port)).unwrap();
    data.write_all(b"hi").unwrap();

    let reattached = wait_for(&server, |ev| match ev {
        ServerEvent::Attached { reattached, .. } => Some(*reattached),
        _ => None,
    });
    assert!(
        !reattached,
        "a brand-new session's first attach must not report reattached: true"
    );
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
    // no new data socket has attached yet. This is purely server-side
    // bookkeeping now — no event is sent for a HELLO reconnect by itself,
    // since `Attached` (sent only once a data socket actually attaches) is
    // the sole source of truth for the UI's "connected" state.
    let second = hello(server.control_port(), "HELLO epsilon");
    assert_eq!(reply, second);

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

/// Regression test for defect 2: closing a window must not just forget the
/// UI-side session but actually tear the server-side session down — its
/// listener, port and accept thread released — rather than leaving it
/// live and unreachable forever. `Console::forget_closed_windows` isn't
/// unit-testable without a TTY (`Application::new` needs a real terminal),
/// so this exercises the piece it delegates to: `Server::close_session`,
/// the same shutdown mechanism `reap` uses.
#[test]
fn close_session_releases_the_port_even_though_the_session_is_still_live() {
    let mut server = Server::bind(0).unwrap();
    let reply = hello(server.control_port(), "HELLO closed-window");
    let port: u16 = reply.strip_prefix("PORT ").unwrap().parse().unwrap();

    // `wait_for` drains events looking for a match and discards anything
    // that doesn't match along the way, so the id must be captured before
    // any other `wait_for` call that would otherwise eat and discard the
    // `Opened` event underneath it.
    let id = wait_for(&server, |ev| match ev {
        ServerEvent::Opened { id, .. } => Some(*id),
        _ => None,
    });

    // A data socket is attached and streaming — unlike `reap`, which only
    // ever targets an idle session, this must work on a *live* one, because
    // that's exactly the scenario the leak happens in: the user closes the
    // window while the stream is still connected.
    let mut data = TcpStream::connect(("127.0.0.1", port)).unwrap();
    data.write_all(b"still streaming").unwrap();
    wait_for(&server, |ev| {
        matches!(ev, ServerEvent::Bytes { .. }).then_some(())
    });
    assert_eq!(server.live_count(), 1);

    server.close_session(id);

    assert_eq!(
        server.live_count(),
        0,
        "the session must be gone, not merely idle"
    );

    // The port must actually be free: a fresh bind on the same port must
    // succeed once the accept thread notices the shutdown flag.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => break,
            Err(e) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
                let _ = e;
            }
            Err(e) => panic!("port {port} was not released after close_session: {e}"),
        }
    }

    // A fresh HELLO for the same name must be treated as brand new (a new
    // port, a new session) rather than reusing the torn-down one.
    let reopened = hello(server.control_port(), "HELLO closed-window");
    let reopened_port: u16 = reopened.strip_prefix("PORT ").unwrap().parse().unwrap();
    assert_ne!(
        reopened_port, port,
        "closing a session must not leave it reusable under its old identity"
    );
}

/// Regression test for defect 3: a client that closes its data socket and
/// immediately redials must end the session attached with exactly one
/// writer and no spurious `Disconnected` chasing the new attach — not two
/// interleaved writers and a title flipping to `[disconnected]` on a
/// connected session. The exact race (an old `LiveGuard::drop` racing a
/// fresh attach) has a narrow window, so this drives many rapid
/// close-then-redial cycles under real concurrency (no `sleep`-based
/// synchronisation — every wait below is on a real channel event with a
/// deadline) to make the window likely to be hit if the guard's teardown
/// is not made safe against a newer attachment already owning the session.
#[test]
fn close_then_immediate_redial_ends_up_attached_with_one_writer_and_no_spurious_disconnect() {
    let server = Server::bind(0).unwrap();
    let reply = hello(server.control_port(), "HELLO race");
    let port: u16 = reply.strip_prefix("PORT ").unwrap().parse().unwrap();
    // Captured once: `wait_for` drains and discards non-matching events, so
    // this must happen before any other `wait_for` call in the loop below
    // or it would eat the `Opened` event out from under us.
    let id = wait_for(&server, |ev| match ev {
        ServerEvent::Opened { id, .. } => Some(*id),
        _ => None,
    });

    for i in 0..20 {
        let mut first = TcpStream::connect(("127.0.0.1", port)).unwrap();
        first.write_all(b"x").unwrap();
        wait_for(&server, |ev| {
            matches!(ev, ServerEvent::Bytes { .. }).then_some(())
        });

        // Close and redial back-to-back, with no synchronisation between
        // them: the old connection's EOF is observed asynchronously by its
        // pump thread, so the redial races that teardown.
        drop(first);
        let mut second = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let marker = format!("iter-{i}");
        second.write_all(marker.as_bytes()).unwrap();

        // The redial must not be rejected as "already attached" — a stale
        // `live` flag (not yet cleared by the outrun old guard) would
        // reject a client that has every right to reconnect.
        second
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let mut banner = [0u8; 64];
        match second.read(&mut banner) {
            Ok(0) | Err(_) => {}
            Ok(n) => {
                let text = String::from_utf8_lossy(&banner[..n]);
                assert!(
                    !text.starts_with("ERR"),
                    "redial rejected on iteration {i}: {text}"
                );
            }
        }

        // Proof the new writer really attached, captured by draining every
        // event (not discarding non-matches, unlike `wait_for` — a
        // discarded event is exactly how an earlier version of this test
        // silently swallowed the very `Disconnected` it meant to catch):
        // collect until the marker's `Bytes` shows up, then keep draining a
        // short bounded window past it (a real deadline, not a sleep-based
        // guess) so a `Disconnected` that arrives shortly after the attach
        // is caught too. A stale guard tearing down the new attachment
        // would send exactly that spurious `Disconnected` — prematurely
        // marking a connected session disconnected (and, in the real UI,
        // flushing its pipeline mid-stream) — or, if the race instead
        // manifests as a torn read on the live flag, would show up as
        // interleaved bytes from both writers.
        let mut seen = Vec::new();
        let mut saw_marker = false;
        let find_marker_deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < find_marker_deadline && !saw_marker {
            if let Ok(ev) = server.events().try_recv() {
                saw_marker = matches!(&ev, ServerEvent::Bytes { data, .. } if data == marker.as_bytes());
                seen.push(ev);
            }
        }
        assert!(saw_marker, "iteration {i}: never saw the marker bytes");
        // Keep draining a short bounded window past the marker: a stray
        // `Disconnected` racing in shortly after the attach is exactly what
        // this test exists to catch, and it can arrive a beat later than
        // the marker itself.
        let settle_deadline = Instant::now() + Duration::from_millis(50);
        while Instant::now() < settle_deadline {
            if let Ok(ev) = server.events().try_recv() {
                seen.push(ev);
            }
        }
        for ev in &seen {
            if let ServerEvent::Bytes { data, .. } = ev
                && data != marker.as_bytes()
                && !data.is_empty()
            {
                panic!(
                    "iteration {i}: saw interleaved bytes {data:?} alongside marker \
                     {marker:?} (two writers attached at once)"
                );
            }
        }
        // One `Disconnected` for the *old* attachment ending is expected
        // and legitimate — it may land anywhere up to and including
        // alongside the new `Attached`, since the old connection's EOF is
        // observed on its own thread, independently of the new accept. What
        // must never happen is a `Disconnected` *after* the new attachment
        // is established: that would be the stale guard reaching back and
        // tearing down an attachment it does not own.
        let attached_at = seen.iter().position(
            |ev| matches!(ev, ServerEvent::Attached { id: eid, reattached: true } if *eid == id),
        );
        if let Some(attached_at) = attached_at {
            assert!(
                !seen[attached_at + 1..]
                    .iter()
                    .any(|ev| matches!(ev, ServerEvent::Disconnected { .. })),
                "iteration {i}: spurious Disconnected after the redial's Attached \
                 (old guard tore down the new attachment): {seen:?}"
            );
        }

        assert_eq!(
            server.live_count(),
            1,
            "iteration {i}: exactly one writer must be attached"
        );

        drop(second);
        wait_for(&server, |ev| {
            matches!(ev, ServerEvent::Disconnected { .. }).then_some(())
        });
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
