// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Capability host functions: what a WASM component can do to the outside
//! world (`docs/WASM-PLUGINS.md`, "Capabilities").
//!
//! Three are implemented so far — `log`, `print` and `state` — chosen because
//! none of them needs a trust story beyond the one the registry already has.
//! `fs`, `net` and `exec` are the ones that undo the sandbox and are not wired.
//!
//! Everything here is compiled unconditionally and none of it mentions the
//! runtime. That is deliberate: the interesting parts — whether a grant is
//! honored, where a key is allowed to write, what a rejected call says — are
//! pure functions over a directory, and they are worth testing without a
//! wasm32 toolchain in the loop. The Extism glue that exposes them to a guest
//! lives in [`crate::wasmhost`] and is a thin shell over this.
//!
//! ## Why a grant is checked at call time
//!
//! Extism resolves a guest's imports at instantiation, so a component
//! importing a function plank does not provide fails to *load* with a message
//! about a missing import. That is a true statement about the module and a
//! useless one for a user, who asked why their plugin stopped working. Every
//! host function is therefore always provided, and each checks its own grant
//! when called — so the answer is "this component was not granted `print`",
//! naming the grant the user can actually give it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Scrollback lines a component asked plank to print, and the log lines it
/// wrote, waiting to be drained by whoever called it.
///
/// A host function cannot reach the UI directly: `print` may be called from
/// inside a guest call that is itself running on the UI thread, and re-entering
/// the renderer from there is how a redraw ends up interleaved with the frame
/// that triggered it. Lines are buffered instead and drained at the call
/// boundary, where the caller already owns the screen.
#[derive(Debug, Default)]
pub struct CapSink {
    /// Lines destined for scrollback, in the order they were printed.
    printed: Vec<String>,
}

impl CapSink {
    /// Records a line for the caller to flush.
    pub fn print(&mut self, line: impl Into<String>) {
        self.printed.push(line.into());
    }

    /// Takes everything buffered, leaving the sink empty.
    #[must_use]
    pub fn drain(&mut self) -> Vec<String> {
        std::mem::take(&mut self.printed)
    }

    /// Whether anything is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.printed.is_empty()
    }
}

/// A shared sink handle. Cloned into each plugin's host functions.
pub type SharedSink = Arc<Mutex<CapSink>>;

/// What a component was granted, and where its private state lives.
///
/// One per loaded component. Held by the runtime and consulted on every host
/// call, so a grant that was never given cannot be reached even by a component
/// that knows the function's name.
#[derive(Debug, Clone)]
pub struct Grants {
    /// Component id, for messages and for the state directory's name.
    pub id: String,
    /// Capability labels the user approved, as [`crate::wasmreg::Capability`]
    /// spells them. Strings rather than the enum so this layer and the runtime
    /// stay free of the registry's types.
    pub granted: Vec<String>,
    /// Root under which this component's `state` capability may write.
    /// `None` disables `state` outright — a plank with no home has nowhere
    /// private to put it, and inventing a location would be worse.
    pub state_root: Option<PathBuf>,
}

impl Grants {
    /// Whether `capability` was granted. `log` is always allowed: it reaches
    /// nothing but plank's own debug file, which is why the manifest parser
    /// adds it for free.
    #[must_use]
    pub fn allows(&self, capability: &str) -> bool {
        capability == "log" || self.granted.iter().any(|g| g == capability)
    }

    /// The message a rejected host call returns to the guest.
    ///
    /// Names the grant rather than the function, because that is the thing a
    /// user can change: "add `print` to this component's capabilities" is
    /// actionable, "`plank_print` failed" is not.
    #[must_use]
    pub fn refusal(&self, capability: &str) -> String {
        format!(
            "'{}' was not granted the '{capability}' capability",
            self.id
        )
    }
}

/// Sanitizes a `state` key into a single file name.
///
/// A key is a component's own business, but it must never escape the
/// component's directory, so anything that could traverse or nest is folded
/// away rather than rejected: a component storing `../../etc/passwd` gets a
/// file called `.._.._etc_passwd`, which is harmless and still round-trips to
/// the same key. Rejecting instead would mean a component's storage silently
/// stops working on a key it has always used.
#[must_use]
pub fn state_file_name(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for c in key.chars() {
        match c {
            '/' | '\\' | ':' | '\0' => out.push('_'),
            c if c.is_control() => out.push('_'),
            c => out.push(c),
        }
    }
    // An empty or dot-only key would name the directory itself.
    if out.is_empty() || out.chars().all(|c| c == '.') {
        out.insert_str(0, "key");
    }
    out
}

/// Reads a component's stored value for `key`.
///
/// # Errors
/// Returns a message when `state` was not granted. A key that was never set
/// reads as an empty value, not an error: a component asking for state it has
/// not written yet is the ordinary first-run case.
pub fn state_get(grants: &Grants, key: &str) -> Result<Vec<u8>, String> {
    if !grants.allows("state") {
        return Err(grants.refusal("state"));
    }
    let Some(dir) = state_dir(grants) else {
        return Ok(Vec::new());
    };
    Ok(std::fs::read(dir.join(state_file_name(key))).unwrap_or_default())
}

/// Stores `value` under `key` for this component.
///
/// # Errors
/// Returns a message when `state` was not granted, when there is nowhere to
/// store it, or when the write fails.
pub fn state_set(grants: &Grants, key: &str, value: &[u8]) -> Result<(), String> {
    if !grants.allows("state") {
        return Err(grants.refusal("state"));
    }
    let dir = state_dir(grants).ok_or_else(|| {
        format!(
            "'{}' has nowhere to store state (no plank home directory)",
            grants.id
        )
    })?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(state_file_name(key)), value).map_err(|e| e.to_string())
}

/// A component's private state directory: `<state_root>/<id>/state`.
///
/// The id is sanitized the same way a key is — it comes from a manifest, and a
/// manifest is not a thing to trust with a path.
fn state_dir(grants: &Grants) -> Option<PathBuf> {
    Some(
        grants
            .state_root
            .as_ref()?
            .join(state_file_name(&grants.id))
            .join("state"),
    )
}

/// Writes a component's line to plank's error log, which is where a plugin's
/// own diagnostics belong: never the transcript, which the model reads.
///
/// # Errors
/// Never — `log` is always allowed. The `Result` exists so every host function
/// in this module has one shape.
pub fn log(grants: &Grants, level: &str, message: &str) -> Result<(), String> {
    crate::errlog::log_error(
        &format!("wasm:{}:{level}", grants.id),
        message.trim_end_matches('\n'),
    );
    Ok(())
}

/// Buffers a line for scrollback.
///
/// # Errors
/// Returns a message when `print` was not granted.
pub fn print(grants: &Grants, sink: &SharedSink, text: &str) -> Result<(), String> {
    if !grants.allows("print") {
        return Err(grants.refusal("print"));
    }
    let mut sink = sink.lock().map_err(|_| "print sink poisoned".to_string())?;
    // Multi-line output arrives as one call; scrollback is line-oriented, so
    // it is split here rather than leaving embedded newlines to the renderer.
    for line in text.split('\n') {
        sink.print(line);
    }
    Ok(())
}

/// Every component's grants for this session, keyed by id.
pub type GrantTable = BTreeMap<String, Grants>;

/// Builds a component's grants.
#[must_use]
pub fn grants_for(id: &str, granted: &[&str], home: Option<&Path>) -> Grants {
    Grants {
        id: id.to_string(),
        granted: granted.iter().map(|s| (*s).to_string()).collect(),
        state_root: home.map(|h| h.join("plugins")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-caps-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The grant check is the sandbox, so it is checked on the way in to every
    /// capability rather than trusted from the manifest.
    #[test]
    fn an_ungranted_capability_is_refused_by_name() {
        let g = grants_for("dev.plank.demo", &["state"], None);
        let sink: SharedSink = SharedSink::default();
        let err = print(&g, &sink, "hello").unwrap_err();
        assert!(err.contains("dev.plank.demo"), "{err}");
        assert!(
            err.contains("'print' capability"),
            "the refusal must name the grant a user can give: {err}"
        );
        assert!(sink.lock().unwrap().is_empty(), "nothing was buffered");
    }

    /// `log` needs no grant: it reaches plank's own debug file and nothing else.
    #[test]
    fn log_is_always_allowed() {
        let g = grants_for("dev.plank.demo", &[], None);
        assert!(g.allows("log"));
        assert!(!g.allows("print"));
        assert!(log(&g, "info", "a line").is_ok());
    }

    #[test]
    fn print_buffers_lines_and_splits_them() {
        let g = grants_for("dev.plank.demo", &["print"], None);
        let sink: SharedSink = SharedSink::default();
        print(&g, &sink, "one\ntwo").unwrap();
        print(&g, &sink, "three").unwrap();
        let lines = sink.lock().unwrap().drain();
        assert_eq!(lines, vec!["one", "two", "three"]);
        assert!(
            sink.lock().unwrap().is_empty(),
            "a drain leaves the sink empty"
        );
    }

    #[test]
    fn state_round_trips_and_starts_empty() {
        let home = temp_dir("state");
        let g = grants_for("dev.plank.demo", &["state"], Some(&home));
        assert_eq!(
            state_get(&g, "counter").unwrap(),
            Vec::<u8>::new(),
            "an unset key is empty, not an error"
        );
        state_set(&g, "counter", b"7").unwrap();
        assert_eq!(state_get(&g, "counter").unwrap(), b"7");
        // It lands under the component's own directory, nowhere else.
        assert!(
            home.join("plugins")
                .join("dev.plank.demo")
                .join("state")
                .join("counter")
                .exists()
        );
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A key is the component's business, but it must not become a path. The
    /// traversal is folded away rather than rejected, so a component's storage
    /// never silently stops working on a key it has always used.
    #[test]
    fn a_state_key_cannot_escape_its_directory() {
        let home = temp_dir("state-escape");
        let g = grants_for("dev.plank.demo", &["state"], Some(&home));
        state_set(&g, "../../escaped", b"x").unwrap();
        assert!(
            !home.join("escaped").exists() && !home.parent().unwrap().join("escaped").exists(),
            "a key traversed out of its directory"
        );
        assert_eq!(state_get(&g, "../../escaped").unwrap(), b"x");
        assert_eq!(state_file_name("../../escaped"), ".._.._escaped");
        assert_eq!(state_file_name(""), "key");
        assert_eq!(state_file_name(".."), "key..");
        let _ = std::fs::remove_dir_all(&home);
    }

    /// With no home there is nowhere private to write. Reads degrade to empty
    /// and writes say why, rather than inventing a location.
    #[test]
    fn state_without_a_home_reads_empty_and_refuses_to_write() {
        let g = grants_for("dev.plank.demo", &["state"], None);
        assert_eq!(state_get(&g, "k").unwrap(), Vec::<u8>::new());
        let err = state_set(&g, "k", b"v").unwrap_err();
        assert!(err.contains("nowhere to store state"), "{err}");
    }
}
