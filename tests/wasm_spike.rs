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
    let mut h = host();
    assert!(h.is_live(), "the feature is on, so the host must be real");
    let loaded = h.load("abi-guest", &wasm).expect("handshake");
    assert_eq!(loaded.abi, ABI_VERSION);
    assert_eq!(loaded.source, "abi-guest");
}

/// Bytes in, bytes out — Extism's whole convention, and the thing every
/// surface's payload encoding will sit on top of.
#[test]
fn a_payload_round_trips_through_the_guest() {
    let wasm = guest_or_skip!();
    let mut h = host();
    h.load("abi-guest", &wasm).expect("handshake");
    let out = h.call("echo", b"round trip").expect("echo");
    assert_eq!(out, b"round trip");
}

/// Not a plugin at all: refused at load, and named as a load failure rather
/// than blamed on the ABI.
#[test]
fn a_module_that_is_not_wasm_is_a_load_error() {
    let mut h = host();
    let err = h.load("junk", b"not a wasm module").unwrap_err();
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
    let mut h = host();
    h.load("abi-guest", &wasm).expect("handshake");

    let started = std::time::Instant::now();
    let err = h.call("spin", b"").unwrap_err();
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
    let mut h2 = host();
    h2.load("abi-guest", &wasm).expect("handshake after a trap");
    assert_eq!(h2.call("echo", b"still here").expect("echo"), b"still here");
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
    let mut cold = host();
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
    let mut live = host();
    let reg = Registry::build(&found, &trust, &project, &mut *live);
    assert!(reg.held.is_empty(), "trust approved it");
    assert_eq!(reg.loaded.len(), 1, "{:?}", reg.warnings);
    assert!(reg.is_live("dev.plank.demo"));

    let _ = std::fs::remove_dir_all(&root);
}
