//! Feasibility-stage tests for the WASM plugin host (`docs/WASM-PLUGINS.md`).
//!
//! These run only with `--features plugins` **and** only when the guest in
//! `spike/abi-guest` has been built (`spike/build-guest.sh`). Both conditions
//! are normally false — CI's default job compiles no wasmtime and builds no
//! guest — so the suite skips rather than fails, and says why.

#![cfg(feature = "plugins")]

use plank::wasmhost::{ABI_VERSION, WasmError, host};

/// The built guest, or `None` when nobody has built it. Returning `None`
/// rather than panicking is deliberate: a missing cross-target artifact is the
/// normal state of a checkout, not a failure.
fn guest() -> Option<Vec<u8>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/spike/abi-guest/target/wasm32-unknown-unknown/release/plank_abi_guest.wasm"
    );
    std::fs::read(path).ok()
}

macro_rules! guest_or_skip {
    () => {
        match guest() {
            Some(w) => w,
            None => {
                eprintln!("skipping: run spike/build-guest.sh first");
                return;
            }
        }
    };
}

/// The handshake is the whole point of the spike: Extism gives no static type
/// checking, so a guest must assert the ABI it speaks before plank asks it for
/// anything else.
#[test]
fn a_guest_completes_the_abi_handshake() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    assert!(h.is_live(), "the feature is on, so the host must be real");
    let loaded = h.load("abi-guest", &wasm, &[]).expect("handshake");
    assert_eq!(loaded.abi, ABI_VERSION);
    assert_eq!(loaded.source, "abi-guest");
}

/// Bytes in, bytes out — Extism's whole convention, and the thing every
/// surface's payload encoding will sit on top of.
#[test]
fn a_payload_round_trips_through_the_guest() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load("abi-guest", &wasm, &[]).expect("handshake");
    let out = h.call("abi-guest", "echo", b"round trip").expect("echo");
    assert_eq!(out, b"round trip");
}

/// Not a plugin at all: refused at load, and named as a load failure rather
/// than blamed on the ABI.
#[test]
fn a_module_that_is_not_wasm_is_a_load_error() {
    let mut h = host(None);
    let err = h.load("junk", b"not a wasm module", &[]).unwrap_err();
    assert!(
        matches!(err, WasmError::Load(_)),
        "expected a load error, got {err:?}"
    );
}

/// The load-bearing claim of the whole design: a plugin that breaks degrades
/// its own feature and nothing else. A guest that never returns must be
/// stopped by the host's deadline, and the host must still be usable
/// afterwards — proven by loading and calling a fresh instance.
#[test]
fn a_runaway_guest_is_stopped_and_the_host_survives() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load("abi-guest", &wasm, &[]).expect("handshake");

    let started = std::time::Instant::now();
    let err = h.call("abi-guest", "spin", b"").unwrap_err();
    let elapsed = started.elapsed();

    assert!(
        matches!(err, WasmError::Trap(_)),
        "an infinite loop must surface as a trap, got {err:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "the deadline did not fire: {elapsed:?}"
    );

    // The session is intact: a fresh plugin loads and answers.
    let mut h2 = host(None);
    h2.load("abi-guest", &wasm, &[])
        .expect("handshake after a trap");
    assert_eq!(
        h2.call("abi-guest", "echo", b"still here").expect("echo"),
        b"still here"
    );
}

/// Phase 1 end to end, with the runtime present: a plugin directory carrying a
/// real module is discovered, judged by the trust store, approved, and loaded.
///
/// This is the join the unit tests cannot make — `wasmreg` is compiled without
/// the runtime and always sees `NoWasmHost`, so "trust said yes and the module
/// actually ran" is only observable here.
#[test]
fn a_trusted_component_discovers_and_loads_end_to_end() {
    use plank::plugins::{Origin, PluginSet, load_plugin};
    use plank::wasmreg::{Registry, TrustStore, discover};

    let wasm = guest_or_skip!();
    let root = std::env::temp_dir().join(format!("plank-wasm-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let plugin_dir = root.join("demo");
    std::fs::create_dir_all(plugin_dir.join(".plank-plugin")).unwrap();
    std::fs::create_dir_all(plugin_dir.join("wasm")).unwrap();
    std::fs::write(plugin_dir.join("wasm").join("demo.wasm"), &wasm).unwrap();
    std::fs::write(
        plugin_dir.join(".plank-plugin").join("plugin.json"),
        r#"{
          "name": "demo",
          "wasm": [{
            "id": "dev.plank.demo",
            "module": "demo.wasm",
            "surfaces": ["observer"],
            "capabilities": ["state"]
          }]
        }"#,
    )
    .unwrap();

    let set = PluginSet {
        plugins: vec![load_plugin(&plugin_dir, Origin::UserScan).expect("a plugin")],
        ..PluginSet::default()
    };
    let found = discover(&set);
    assert_eq!(found.components.len(), 1, "{:?}", found.warnings);

    let project = root.join("repo");
    let mut trust = TrustStore::ephemeral();
    // Untrusted first: the module is never handed to the runtime.
    let mut cold = host(None);
    let held = Registry::build(&found, &trust, &project, &mut *cold);
    assert!(
        held.loaded.is_empty(),
        "an unapproved component must not run"
    );
    assert_eq!(held.held.len(), 1);

    // Approve it at its real hash, then it loads — ABI handshake included.
    let sha = plank::wasmreg::module_sha256(&found.components[0].path).expect("hash");
    trust
        .approve(&found.components[0], &sha, &project)
        .expect("approve");
    let mut live = host(None);
    let reg = Registry::build(&found, &trust, &project, &mut *live);
    assert!(reg.held.is_empty(), "trust approved it");
    assert_eq!(reg.loaded.len(), 1, "{:?}", reg.warnings);
    assert!(reg.is_live("dev.plank.demo"));

    let _ = std::fs::remove_dir_all(&root);
}

/// Builds a plugin directory carrying the guest, claiming `command`.
fn command_plugin(tag: &str, wasm: &[u8]) -> (std::path::PathBuf, plank::plugins::PluginSet) {
    use plank::plugins::{Origin, PluginSet, load_plugin};

    let root = std::env::temp_dir().join(format!("plank-wasm-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("demo");
    std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
    std::fs::create_dir_all(dir.join("wasm")).unwrap();
    std::fs::write(dir.join("wasm").join("demo.wasm"), wasm).unwrap();
    std::fs::write(
        dir.join(".plank-plugin").join("plugin.json"),
        r#"{
          "name": "demo",
          "wasm": [{
            "id": "dev.plank.demo",
            "module": "demo.wasm",
            "surfaces": ["command"],
            "capabilities": ["print"]
          }]
        }"#,
    )
    .unwrap();
    let set = PluginSet {
        plugins: vec![load_plugin(&dir, Origin::UserScan).expect("a plugin")],
        ..PluginSet::default()
    };
    (root, set)
}

/// The `command` surface end to end: specs are read once at load, the command
/// runs, and its reply reaches the host as a structured outcome.
#[test]
fn a_command_component_registers_and_runs() {
    use plank::wasmreg::Session;

    let wasm = guest_or_skip!();
    let (root, plugins) = command_plugin("cmd", &wasm);
    let project = root.join("repo");

    let mut session = Session::default();
    session.activate(&plugins, &project);
    // Ephemeral trust with no home: nothing is approved, so nothing loads.
    assert!(session.registry.loaded.is_empty());
    assert_eq!(session.registry.held.len(), 1);
    assert!(session.registry.commands().is_empty());

    // Approving it loads it *and* registers its commands, without a restart.
    let name = session
        .approve("dev.plank.demo", &project)
        .expect("approve");
    assert_eq!(name, "dev.plank.demo");
    let commands = session.registry.commands();
    assert_eq!(commands.len(), 1, "{commands:?}");
    assert_eq!(commands[0].1.name, "greet");
    assert_eq!(commands[0].1.desc, "say hello from wasm");

    // No argument: it prints and asks for nothing else.
    let out = session
        .registry
        .run_command(&mut *session.host, "greet", "")
        .expect("run");
    assert_eq!(out.print, vec!["hello from wasm".to_string()]);
    assert_eq!(out.prompt, None);

    // With an argument: it prints and submits a prompt.
    let out = session
        .registry
        .run_command(&mut *session.host, "/greet", "ada")
        .expect("run");
    assert_eq!(out.print, vec!["greeting ada".to_string()]);
    assert_eq!(out.prompt.as_deref(), Some("say hello to ada"));

    // A name nobody contributes is not an error state, just a miss.
    assert!(
        session
            .registry
            .run_command(&mut *session.host, "nope", "")
            .is_err()
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// Claiming a surface you did not implement is a load-time error. Without this
/// the component would sit in the slash menu and fail only once a user picked
/// it — a bug reported as "your menu is broken", not "my plugin is".
#[test]
fn a_component_claiming_command_without_the_exports_is_refused() {
    use plank::plugins::{Origin, PluginSet, load_plugin};
    use plank::wasmreg::{Registry, TrustStore, discover, module_sha256};

    // The minimal guest asserts the ABI and implements no surface at all.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/spike/min-guest/target/wasm32-unknown-unknown/release/plank_min_guest.wasm"
    );
    let Ok(wasm) = std::fs::read(path) else {
        eprintln!("skipping: run spike/build-guest.sh first");
        return;
    };

    let root = std::env::temp_dir().join(format!("plank-wasm-noexp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("demo");
    std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
    std::fs::create_dir_all(dir.join("wasm")).unwrap();
    std::fs::write(dir.join("wasm").join("demo.wasm"), &wasm).unwrap();
    std::fs::write(
        dir.join(".plank-plugin").join("plugin.json"),
        r#"{"name": "demo", "wasm": [{"id": "dev.plank.liar", "module": "demo.wasm",
            "surfaces": ["command"], "capabilities": []}]}"#,
    )
    .unwrap();

    let set = PluginSet {
        plugins: vec![load_plugin(&dir, Origin::UserScan).expect("a plugin")],
        ..PluginSet::default()
    };
    let found = discover(&set);
    let project = root.join("repo");
    let mut trust = TrustStore::ephemeral();
    let sha = module_sha256(&found.components[0].path).expect("hash");
    trust.approve(&found.components[0], &sha, &project).unwrap();

    let mut host = host(None);
    let reg = Registry::build(&found, &trust, &project, &mut *host);
    assert!(
        reg.loaded.is_empty(),
        "a component that cannot serve its surface must not be admitted"
    );
    assert!(
        reg.warnings.iter().any(|w| w.contains("command_specs")
            && w.contains("command_run")
            && w.contains("dev.plank.liar")),
        "the refusal must name the component and the missing exports: {:?}",
        reg.warnings
    );
    // And it contributes nothing to the menu, which is the user-visible point.
    assert!(reg.commands().is_empty());
    let _ = std::fs::remove_dir_all(&root);
}

/// Builds a plugin whose single component declares `caps`, and returns the
/// temp root plus the loaded set.
fn caps_plugin(
    tag: &str,
    wasm: &[u8],
    caps: &str,
) -> (std::path::PathBuf, plank::plugins::PluginSet) {
    use plank::plugins::{Origin, PluginSet, load_plugin};

    let root = std::env::temp_dir().join(format!("plank-wasm-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("demo");
    std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
    std::fs::create_dir_all(dir.join("wasm")).unwrap();
    std::fs::write(dir.join("wasm").join("demo.wasm"), wasm).unwrap();
    std::fs::write(
        dir.join(".plank-plugin").join("plugin.json"),
        format!(
            r#"{{"name": "demo", "wasm": [{{"id": "dev.plank.demo", "module": "demo.wasm",
                "surfaces": ["observer"], "capabilities": [{caps}]}}]}}"#
        ),
    )
    .unwrap();
    let set = PluginSet {
        plugins: vec![load_plugin(&dir, Origin::UserScan).expect("a plugin")],
        ..PluginSet::default()
    };
    (root, set)
}

/// Loads a component with the given capabilities and returns a session ready
/// to call it, plus the temp root to clean up.
fn approved_session(
    tag: &str,
    wasm: &[u8],
    caps: &str,
    home: Option<&std::path::Path>,
) -> (std::path::PathBuf, plank::wasmreg::Session) {
    let (root, plugins) = caps_plugin(tag, wasm, caps);
    let project = root.join("repo");
    let mut session = plank::wasmreg::Session::new(home);
    session.activate(&plugins, &project);
    session
        .approve("dev.plank.demo", &project)
        .expect("approve");
    (root, session)
}

/// A granted capability works and its output reaches the host's sink; the same
/// call without the grant is refused *by name*, because the name is the thing
/// the user can change.
#[test]
fn the_print_capability_is_honored_only_when_granted() {
    let wasm = guest_or_skip!();

    let (root, mut session) = approved_session("cap-print", &wasm, r#""print""#, None);
    let reply = session
        .host
        .call("dev.plank.demo", "cap_print", b"hello\nthere")
        .expect("call");
    assert_eq!(
        String::from_utf8_lossy(&reply),
        "",
        "an empty reply means the host accepted it"
    );
    assert_eq!(
        session.host.drain_printed(),
        vec!["hello".to_string(), "there".to_string()],
        "a multi-line print becomes multiple scrollback lines"
    );
    assert!(
        session.host.drain_printed().is_empty(),
        "draining twice yields nothing"
    );
    let _ = std::fs::remove_dir_all(&root);

    let (root, mut session) = approved_session("cap-noprint", &wasm, "", None);
    let reply = session
        .host
        .call("dev.plank.demo", "cap_print", b"hello")
        .expect("call");
    let reply = String::from_utf8_lossy(&reply);
    assert!(reply.contains("'print' capability"), "{reply}");
    assert!(reply.contains("dev.plank.demo"), "{reply}");
    assert!(
        session.host.drain_printed().is_empty(),
        "a refused print must not reach scrollback"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// State persists across calls and lands in the component's own directory —
/// the point of the capability being that it needs no filesystem grant.
#[test]
fn the_state_capability_persists_across_calls_in_its_own_directory() {
    let wasm = guest_or_skip!();
    let home = std::env::temp_dir().join(format!("plank-wasm-home-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    std::fs::create_dir_all(&home).unwrap();

    let (root, mut session) = approved_session("cap-state", &wasm, r#""state""#, Some(&home));
    for expected in ["1", "2", "3"] {
        let reply = session
            .host
            .call("dev.plank.demo", "cap_bump", b"")
            .expect("call");
        assert_eq!(String::from_utf8_lossy(&reply), expected);
    }
    let stored = home
        .join("plugins")
        .join("dev.plank.demo")
        .join("state")
        .join("counter");
    assert_eq!(std::fs::read_to_string(&stored).unwrap(), "3");

    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&home);
}

/// Without the grant, a read is empty and a write is refused — so the counter
/// never advances and the component is told why.
#[test]
fn state_without_the_grant_reads_empty_and_refuses_to_write() {
    let wasm = guest_or_skip!();
    let home = std::env::temp_dir().join(format!("plank-wasm-nohome-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    std::fs::create_dir_all(&home).unwrap();

    let (root, mut session) = approved_session("cap-nostate", &wasm, r#""print""#, Some(&home));
    let reply = session
        .host
        .call("dev.plank.demo", "cap_bump", b"")
        .expect("call");
    let reply = String::from_utf8_lossy(&reply);
    assert!(reply.contains("'state' capability"), "{reply}");
    assert!(
        !home.join("plugins").join("dev.plank.demo").exists(),
        "a refused write must create nothing"
    );

    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&home);
}

/// Loads a component subscribing to `events` with the given capabilities.
fn subscribed_session(
    tag: &str,
    wasm: &[u8],
    events: &str,
    home: Option<&std::path::Path>,
) -> (std::path::PathBuf, plank::wasmreg::Session) {
    use plank::plugins::{Origin, PluginSet, load_plugin};

    let root = std::env::temp_dir().join(format!("plank-wasm-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("demo");
    std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
    std::fs::create_dir_all(dir.join("wasm")).unwrap();
    std::fs::write(dir.join("wasm").join("demo.wasm"), wasm).unwrap();
    std::fs::write(
        dir.join(".plank-plugin").join("plugin.json"),
        format!(
            r#"{{"name": "demo", "wasm": [{{"id": "dev.plank.demo", "module": "demo.wasm",
                "surfaces": ["observer"], "capabilities": ["state"], "events": [{events}]}}]}}"#
        ),
    )
    .unwrap();
    let set = PluginSet {
        plugins: vec![load_plugin(&dir, Origin::UserScan).expect("a plugin")],
        ..PluginSet::default()
    };
    let project = root.join("repo");
    let mut session = plank::wasmreg::Session::new(home);
    session.activate(&set, &project);
    session
        .approve("dev.plank.demo", &project)
        .expect("approve");
    (root, session)
}

/// A subscriber sees the events it asked for, in the payload shape the bus
/// promises — and nothing else.
#[test]
fn a_subscriber_receives_only_its_events() {
    use plank::wasmevents::{Event, EventKind};

    let wasm = guest_or_skip!();
    let home = std::env::temp_dir().join(format!("plank-wasm-evhome-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    std::fs::create_dir_all(&home).unwrap();

    let (root, mut session) =
        subscribed_session("ev-some", &wasm, r#""pre_tool_use""#, Some(&home));

    let subscribed = Event::new(EventKind::PreToolUse, vec![("name", "bash".to_string())]);
    let out = session.registry.dispatch(&mut *session.host, &subscribed);
    assert!(out.warnings.is_empty(), "{:?}", out.warnings);

    // Not subscribed: the component is never called, so its own tally of what
    // it saw does not grow.
    let unsubscribed = Event::new(EventKind::TurnEnd, vec![("turn", "1".to_string())]);
    let out = session.registry.dispatch(&mut *session.host, &unsubscribed);
    assert!(out.is_quiet(), "an unsubscribed event must cost nothing");

    let tally = std::fs::read_to_string(
        home.join("plugins")
            .join("dev.plank.demo")
            .join("state")
            .join("events"),
    )
    .unwrap();
    assert_eq!(
        tally, "pre_tool_use;",
        "the handler ran exactly once, for the event it subscribed to"
    );

    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&home);
}

/// A veto event's block reaches the caller with the component named, so the
/// user can tell which component refused their tool call.
#[test]
fn a_subscriber_can_veto_a_veto_class_event() {
    use plank::wasmevents::{Event, EventKind};

    let wasm = guest_or_skip!();
    let (root, mut session) = subscribed_session("ev-veto", &wasm, r#""pre_tool_use""#, None);

    let event = Event::new(
        EventKind::PreToolUse,
        vec![
            ("name", "bash".to_string()),
            ("args", "forbidden".to_string()),
        ],
    );
    let out = session.registry.dispatch(&mut *session.host, &event);
    let (id, reason) = out.blocked.expect("blocked");
    assert_eq!(id, "dev.plank.demo");
    assert!(reason.contains("refuses anything forbidden"), "{reason}");
    let _ = std::fs::remove_dir_all(&root);
}

/// A transform event's replacement comes back to the host, which is what lets
/// a component rewrite what the model sees.
#[test]
fn a_subscriber_can_rewrite_a_transform_class_event() {
    use plank::wasmevents::{Event, EventKind};

    let wasm = guest_or_skip!();
    let (root, mut session) = subscribed_session("ev-xform", &wasm, r#""post_tool_use""#, None);

    let event = Event::new(
        EventKind::PostToolUse,
        vec![
            ("name", "bash".to_string()),
            ("output", "please rewrite this".to_string()),
        ],
    );
    let out = session.registry.dispatch(&mut *session.host, &event);
    assert_eq!(out.replaced.as_deref(), Some("rewritten by the fixture"));
    assert!(out.blocked.is_none());
    let _ = std::fs::remove_dir_all(&root);
}

/// The class is enforced host-side: the same fixture that vetoes a veto event
/// cannot veto a notify one, no matter what it replies.
#[test]
fn a_notify_event_ignores_a_block_reply() {
    use plank::wasmevents::{Event, EventKind};

    let wasm = guest_or_skip!();
    let (root, mut session) = subscribed_session("ev-notify", &wasm, r#""turn_end""#, None);

    let event = Event::new(EventKind::TurnEnd, vec![("turn", "forbidden".to_string())]);
    let out = session.registry.dispatch(&mut *session.host, &event);
    assert!(
        out.blocked.is_none(),
        "a notify event must not be blockable: {:?}",
        out.blocked
    );
    assert!(out.replaced.is_none());
    let _ = std::fs::remove_dir_all(&root);
}

/// A component claiming `observer` without a handler is refused, the same way
/// a `command` component without its exports is.
#[test]
fn an_observer_without_on_event_is_refused() {
    use plank::plugins::{Origin, PluginSet, load_plugin};
    use plank::wasmreg::{Registry, TrustStore, discover, module_sha256};

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/spike/min-guest/target/wasm32-unknown-unknown/release/plank_min_guest.wasm"
    );
    let Ok(wasm) = std::fs::read(path) else {
        eprintln!("skipping: run spike/build-guest.sh first");
        return;
    };

    let root = std::env::temp_dir().join(format!("plank-wasm-noobs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("demo");
    std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
    std::fs::create_dir_all(dir.join("wasm")).unwrap();
    std::fs::write(dir.join("wasm").join("demo.wasm"), &wasm).unwrap();
    std::fs::write(
        dir.join(".plank-plugin").join("plugin.json"),
        r#"{"name": "demo", "wasm": [{"id": "dev.plank.deaf", "module": "demo.wasm",
            "surfaces": ["observer"], "events": ["turn_end"]}]}"#,
    )
    .unwrap();

    let set = PluginSet {
        plugins: vec![load_plugin(&dir, Origin::UserScan).expect("a plugin")],
        ..PluginSet::default()
    };
    let found = discover(&set);
    let project = root.join("repo");
    let mut trust = TrustStore::ephemeral();
    let sha = module_sha256(&found.components[0].path).expect("hash");
    trust.approve(&found.components[0], &sha, &project).unwrap();

    let mut host = host(None);
    let reg = Registry::build(&found, &trust, &project, &mut *host);
    assert!(reg.loaded.is_empty(), "{:?}", reg.warnings);
    assert!(
        reg.warnings
            .iter()
            .any(|w| w.contains("on_event") && w.contains("dev.plank.deaf")),
        "{:?}",
        reg.warnings
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// The `tool` surface end to end: specs are read at load, the exposed name is
/// the bare one when nothing contests it, and a call returns the observation
/// the model will see.
#[test]
fn a_tool_component_registers_and_runs() {
    use plank::plugins::{Origin, PluginSet, load_plugin};
    use plank::wasmreg::Session;

    let wasm = guest_or_skip!();
    let root = std::env::temp_dir().join(format!("plank-wasm-tool-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("demo");
    std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
    std::fs::create_dir_all(dir.join("wasm")).unwrap();
    std::fs::write(dir.join("wasm").join("demo.wasm"), &wasm).unwrap();
    std::fs::write(
        dir.join(".plank-plugin").join("plugin.json"),
        r#"{"name": "demo", "wasm": [{"id": "dev.plank.demo", "module": "demo.wasm",
            "surfaces": ["tool"], "capabilities": []}]}"#,
    )
    .unwrap();

    let set = PluginSet {
        plugins: vec![load_plugin(&dir, Origin::UserScan).expect("a plugin")],
        ..PluginSet::default()
    };
    let project = root.join("repo");
    let mut session = Session::new(None);
    session.activate(&set, &project);
    session
        .approve("dev.plank.demo", &project)
        .expect("approve");

    let tools = session.registry.tools();
    assert_eq!(tools.len(), 1, "{tools:?}");
    assert_eq!(tools[0].name, "wordcount");
    assert_eq!(tools[0].exposed, "wordcount", "uncontested keeps its name");
    assert!(
        tools[0].schema.contains("\"required\":[\"text\"]"),
        "{}",
        tools[0].schema
    );

    let out = session
        .registry
        .run_tool(
            &mut *session.host,
            "wordcount",
            r#"{"text":"one two three"}"#,
        )
        .expect("run");
    assert_eq!(out, "3 words\n");
    let _ = std::fs::remove_dir_all(&root);
}

/// A component cannot take over a built-in. A tool named `bash` is exposed
/// qualified, and the qualification is reported rather than silent.
#[test]
fn a_contested_tool_name_is_qualified_and_reported() {
    use plank::wasmreg::{Loaded, Registry, WasmTool};

    // Built directly: what is under test is the resolution rule, and driving it
    // through a guest would only prove the guest can be renamed too.
    let tool = |name: &str, component: &str| WasmTool {
        component: component.to_string(),
        name: name.to_string(),
        exposed: name.to_string(),
        description: String::new(),
        schema: String::new(),
    };
    let mut reg = Registry::with_loaded(vec![Loaded {
        component: plank::wasmreg::WasmComponent {
            plugin: "a".to_string(),
            origin: plank::plugins::Origin::UserScan,
            path: std::path::PathBuf::from("/x"),
            manifest: plank::wasmreg::WasmManifest {
                id: "dev.plank.a".to_string(),
                abi: 1,
                module: "a.wasm".to_string(),
                surfaces: vec![plank::wasmreg::Surface::Tool],
                capabilities: vec![],
                events: vec![],
            },
        },
        strikes: 0,
        tools: vec![tool("bash", "dev.plank.a"), tool("unique", "dev.plank.a")],
        commands: vec![],
    }]);

    let warnings = reg.resolve_tool_names(&["bash".to_string(), "read".to_string()]);
    let tools = reg.tools();
    let bash = tools.iter().find(|t| t.name == "bash").unwrap();
    let unique = tools.iter().find(|t| t.name == "unique").unwrap();
    assert_eq!(
        bash.exposed, "wasm__a__bash",
        "a built-in must never be shadowed"
    );
    assert_eq!(
        unique.exposed, "unique",
        "an uncontested name is left alone"
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("contested") && w.contains("wasm__a__bash")),
        "{warnings:?}"
    );
}

/// The `segment` surface: a cell renders from the payload the host supplies,
/// and — the part that matters for a status bar — a second call inside the
/// throttle window does not reach the guest at all.
#[test]
fn a_segment_component_renders_and_is_throttled() {
    use plank::plugins::{Origin, PluginSet, load_plugin};
    use plank::wasmreg::{SEGMENT_MIN_INTERVAL_MS, Session};

    let wasm = guest_or_skip!();
    let root = std::env::temp_dir().join(format!("plank-wasm-seg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("demo");
    std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
    std::fs::create_dir_all(dir.join("wasm")).unwrap();
    std::fs::write(dir.join("wasm").join("demo.wasm"), &wasm).unwrap();
    std::fs::write(
        dir.join(".plank-plugin").join("plugin.json"),
        r#"{"name": "demo", "wasm": [{"id": "dev.plank.demo", "module": "demo.wasm",
            "surfaces": ["segment"], "capabilities": []}]}"#,
    )
    .unwrap();

    let set = PluginSet {
        plugins: vec![load_plugin(&dir, Origin::UserScan).expect("a plugin")],
        ..PluginSet::default()
    };
    let project = root.join("repo");
    let mut session = Session::new(None);
    session.activate(&set, &project);
    session
        .approve("dev.plank.demo", &project)
        .expect("approve");

    assert!(
        session.registry.segments().is_empty(),
        "nothing rendered yet"
    );

    let rendered = session.registry.refresh_segments(
        &mut *session.host,
        r#"{"cwd": "/x", "messages": 7, "ctx_size": 1024}"#,
        10_000,
    );
    assert!(rendered);
    let cells = session.registry.segments();
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].text, "msgs 7", "the payload reached the guest");
    assert_eq!(cells[0].priority, 200);

    // Inside the window: not re-rendered, and the previous cell survives.
    let again = session.registry.refresh_segments(
        &mut *session.host,
        r#"{"cwd": "/x", "messages": 99, "ctx_size": 1024}"#,
        10_000 + SEGMENT_MIN_INTERVAL_MS - 1,
    );
    assert!(
        !again,
        "a call inside the throttle window must not reach the guest"
    );
    assert_eq!(session.registry.segments()[0].text, "msgs 7");

    // Past it: rendered again, with the new facts.
    assert!(session.registry.refresh_segments(
        &mut *session.host,
        r#"{"cwd": "/x", "messages": 99, "ctx_size": 1024}"#,
        10_000 + SEGMENT_MIN_INTERVAL_MS,
    ));
    assert_eq!(session.registry.segments()[0].text, "msgs 99");
    let _ = std::fs::remove_dir_all(&root);
}
