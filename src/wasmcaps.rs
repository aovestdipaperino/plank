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
    /// Whether the user has sound on this session. Held here rather than read
    /// at the call: the capability answers "may this component ask for a
    /// noise", and this answers "does this terminal make one".
    pub sound: crate::arcade::Sound,
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

/// Largest single `state` value a component may store, in bytes.
///
/// `state` is for a component's own small persistent facts — a high score, a
/// cursor, a settings blob — not a cache. A megabyte is a generous ceiling for
/// any of those and small enough that no single write can fill a disk.
pub const STATE_MAX_VALUE_BYTES: usize = 1024 * 1024;

/// Longest `state` key, in bytes.
///
/// A key becomes a file name, and file systems have their own opinion about
/// long ones: `NAME_MAX` is 255 on APFS, ext4 and most everything else. The
/// cap sits exactly there so a key the quota accepts is a key the disk accepts
/// too, rather than one that passes here and dies with "file name too long".
pub const STATE_MAX_KEY_BYTES: usize = 255;

/// Most keys one component may hold at once.
pub const STATE_MAX_KEYS: usize = 256;

/// Combined size of everything one component has stored, in bytes.
///
/// The per-value cap bounds one write; this bounds the component. Sixteen
/// megabytes is sixteen maximal values, which is more than any plugin has
/// asked for and far less than the model file sitting next to it.
pub const STATE_MAX_TOTAL_BYTES: u64 = 16 * 1024 * 1024;

/// Stores `value` under `key` for this component.
///
/// Every quota is checked before anything touches the disk, so a refused write
/// leaves the component's state exactly as it was — including the value the
/// key held before, which an over-quota rewrite must not destroy.
///
/// # Errors
/// Returns a message when `state` was not granted, when there is nowhere to
/// store it, when the key or value exceeds [`STATE_MAX_KEY_BYTES`] or
/// [`STATE_MAX_VALUE_BYTES`], when the component already holds
/// [`STATE_MAX_KEYS`] keys or would exceed [`STATE_MAX_TOTAL_BYTES`], or when
/// the write fails.
pub fn state_set(grants: &Grants, key: &str, value: &[u8]) -> Result<(), String> {
    if !grants.allows("state") {
        return Err(grants.refusal("state"));
    }
    if key.len() > STATE_MAX_KEY_BYTES {
        return Err(format!(
            "'{}' state key is {} bytes, more than the {STATE_MAX_KEY_BYTES}-byte limit",
            grants.id,
            key.len()
        ));
    }
    if value.len() > STATE_MAX_VALUE_BYTES {
        return Err(format!(
            "'{}' state value for '{key}' is {} bytes, more than the \
             {STATE_MAX_VALUE_BYTES}-byte limit",
            grants.id,
            value.len()
        ));
    }
    let dir = state_dir(grants).ok_or_else(|| {
        format!(
            "'{}' has nowhere to store state (no plank home directory)",
            grants.id
        )
    })?;
    let file = dir.join(state_file_name(key));
    let (keys, total) = state_usage(&dir, &file);
    if keys >= STATE_MAX_KEYS {
        return Err(format!(
            "'{}' already holds {STATE_MAX_KEYS} state keys, the limit; '{key}' was not stored",
            grants.id
        ));
    }
    if total + value.len() as u64 > STATE_MAX_TOTAL_BYTES {
        return Err(format!(
            "'{}' state would grow to {} bytes, more than the {STATE_MAX_TOTAL_BYTES}-byte \
             limit; '{key}' was not stored",
            grants.id,
            total + value.len() as u64
        ));
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(file, value).map_err(|e| e.to_string())
}

/// How many keys a component holds and how many bytes they add up to,
/// *excluding* the file at `except` — the key about to be rewritten, whose old
/// value is being replaced rather than added to.
///
/// A scan rather than a running count: the directory is the truth, it is
/// small by construction (at most [`STATE_MAX_KEYS`] entries), and a counter
/// would drift the first time a user deleted a file by hand.
fn state_usage(dir: &Path, except: &Path) -> (usize, u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (0, 0);
    };
    entries
        .flatten()
        .filter(|e| e.path() != except)
        .filter_map(|e| e.metadata().ok())
        .filter(std::fs::Metadata::is_file)
        .fold((0, 0), |(n, bytes), m| (n + 1, bytes + m.len()))
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

/// Plays a sound cue on the component's behalf.
///
/// A guest cannot shell out and cannot reach an audio device, so the one way
/// it makes a noise is to ask. The cue set is [`crate::arcade::Cue`] exactly —
/// three blips with no pitch and no length (see `arcade::Sound`) — so a
/// component gets the same repertoire as a built-in face and no more.
///
/// Whether anything is actually audible is the *user's* setting, not the
/// component's: `Sound` is off unless the arcade command asked for it, and a
/// component cannot turn it on. That is deliberate — a plugin that could make
/// the terminal beep unbidden is a plugin that will.
///
/// # Errors
/// Returns a message when `sound` was not granted, or when the cue is not one
/// of the three.
pub fn sound(grants: &Grants, sound: crate::arcade::Sound, cue: &str) -> Result<(), String> {
    if !grants.allows("sound") {
        return Err(grants.refusal("sound"));
    }
    let cue = match cue {
        "hit" => crate::arcade::Cue::Hit,
        "lose" => crate::arcade::Cue::Lose,
        "win" => crate::arcade::Cue::Win,
        other => return Err(format!("unknown sound cue '{other}'")),
    };
    sound.play(cue);
    Ok(())
}

/// The `notify` capability: a desktop notification.
///
/// Declared since the first design and reaching nothing until now, which is the
/// worst state for a grant to be in — a user approving `notify` was approving
/// something that did not exist.
///
/// Two limits the guest cannot argue with. The component's id prefixes the
/// title, because a notification that does not say which plugin produced it is
/// one the user cannot act on or switch off. And the text is truncated: a
/// notification is a line, and a component that pastes a kilobyte into one is
/// abusing the shared attention of every other plugin.
///
/// Respects the user's own notification setting rather than routing around it.
/// A plugin is not more entitled to interrupt than plank is.
///
/// # Errors
/// Returns the refusal message when the grant is absent.
pub fn notify(grants: &Grants, title: &str, body: &str) -> Result<(), String> {
    /// Longest title and body a component may push. A notification is a line.
    const CAP: usize = 200;

    if !grants.allows("notify") {
        return Err(grants.refusal("notify"));
    }
    let clip = |text: &str| -> String {
        let mut out: String = text.chars().take(CAP).collect();
        if out.chars().count() < text.chars().count() {
            out.push('…');
        }
        // A newline in a notification title tears the layout on some servers,
        // so it is folded rather than trusted — same rule as a status cell.
        out.replace(['\n', '\r'], " ")
    };
    let title = clip(title);
    let title = if title.trim().is_empty() {
        grants.id.clone()
    } else {
        format!("{}: {title}", grants.id)
    };
    crate::notify::notify(&title, &clip(body));
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
        sound: crate::arcade::Sound::default(),
        state_root: home.map(|h| h.join("plugins")),
    }
}

#[cfg(test)]
mod tests {

    /// `notify` refuses without the grant, and shapes what it accepts with it.
    #[test]
    fn notify_requires_the_grant_and_bounds_what_it_sends() {
        let denied = grants_for("dev.plank.demo", &[], None);
        let err = notify(&denied, "hi", "there").expect_err("must refuse");
        assert!(err.contains("notify"), "{err}");

        // With the grant it succeeds. The delivery itself is a no-op when the
        // user has notifications off or is not on macOS, which is the same
        // path plank's own notifications take — a plugin is not more entitled
        // to interrupt than plank is.
        let allowed = grants_for("dev.plank.demo", &["notify"], None);
        assert!(notify(&allowed, "hi", "there").is_ok());
        // Empty title still works: the component id stands in, because a
        // notification that does not say who sent it cannot be acted on.
        assert!(notify(&allowed, "", "body").is_ok());
        // Absurd lengths are accepted and clipped rather than refused: failing
        // a component's frame over a long string would be worse than trimming.
        let long = "x".repeat(10_000);
        assert!(notify(&allowed, &long, &long).is_ok());
    }
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

    /// A component may ask for a noise only if it was granted `sound`, and
    /// only for a cue that exists — the set a built-in face has, and no more.
    #[test]
    fn sound_needs_the_grant_and_a_known_cue() {
        let ungranted = grants_for("dev.plank.demo", &[], None);
        assert!(
            sound(&ungranted, crate::arcade::Sound::default(), "hit")
                .unwrap_err()
                .contains("'sound' capability")
        );

        let granted = grants_for("dev.plank.demo", &["sound"], None);
        // Silent by default — `Sound::default()` is off — so this plays
        // nothing and still reports success: the grant was honored.
        assert!(sound(&granted, crate::arcade::Sound::default(), "hit").is_ok());
        assert!(sound(&granted, crate::arcade::Sound::default(), "win").is_ok());
        let err = sound(&granted, crate::arcade::Sound::default(), "explode").unwrap_err();
        assert!(err.contains("unknown sound cue 'explode'"), "{err}");
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

    /// A value over the per-value cap is refused, by size, and nothing lands.
    #[test]
    fn state_refuses_an_oversized_value() {
        let home = temp_dir("state-big-value");
        let g = grants_for("dev.plank.demo", &["state"], Some(&home));
        let big = vec![0u8; STATE_MAX_VALUE_BYTES + 1];
        let err = state_set(&g, "blob", &big).unwrap_err();
        assert!(err.contains("dev.plank.demo"), "{err}");
        assert!(err.contains("value"), "{err}");
        assert_eq!(state_get(&g, "blob").unwrap(), Vec::<u8>::new());
        // Exactly at the cap is fine.
        state_set(&g, "blob", &big[..STATE_MAX_VALUE_BYTES]).unwrap();
        assert_eq!(state_get(&g, "blob").unwrap().len(), STATE_MAX_VALUE_BYTES);
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A key longer than the cap is refused, by length, and nothing lands.
    #[test]
    fn state_refuses_an_overlong_key() {
        let home = temp_dir("state-long-key");
        let g = grants_for("dev.plank.demo", &["state"], Some(&home));
        let long = "k".repeat(STATE_MAX_KEY_BYTES + 1);
        let err = state_set(&g, &long, b"v").unwrap_err();
        assert!(err.contains("key"), "{err}");
        assert!(
            !home
                .join("plugins")
                .join("dev.plank.demo")
                .join("state")
                .join(&long)
                .exists()
        );
        state_set(&g, &"k".repeat(STATE_MAX_KEY_BYTES), b"v").unwrap();
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A component may hold only so many keys. Overwriting one it already
    /// holds does not count as a new one.
    #[test]
    fn state_caps_the_number_of_keys() {
        let home = temp_dir("state-key-count");
        let g = grants_for("dev.plank.demo", &["state"], Some(&home));
        for i in 0..STATE_MAX_KEYS {
            state_set(&g, &format!("k{i}"), b"v").unwrap();
        }
        let err = state_set(&g, "one-too-many", b"v").unwrap_err();
        assert!(err.contains("keys"), "{err}");
        assert_eq!(state_get(&g, "one-too-many").unwrap(), Vec::<u8>::new());
        // An existing key still updates.
        state_set(&g, "k0", b"updated").unwrap();
        assert_eq!(state_get(&g, "k0").unwrap(), b"updated");
        let _ = std::fs::remove_dir_all(&home);
    }

    /// The combined size of everything stored is bounded. Replacing a key
    /// counts the new value in place of the old, not on top of it.
    #[test]
    fn state_caps_the_combined_size() {
        let home = temp_dir("state-total");
        let g = grants_for("dev.plank.demo", &["state"], Some(&home));
        let chunk = vec![0u8; STATE_MAX_VALUE_BYTES];
        let fill = usize::try_from(STATE_MAX_TOTAL_BYTES).unwrap() / STATE_MAX_VALUE_BYTES;
        for i in 0..fill {
            state_set(&g, &format!("c{i}"), &chunk).unwrap();
        }
        let err = state_set(&g, "spill", b"x").unwrap_err();
        assert!(err.contains("bytes"), "{err}");
        assert_eq!(state_get(&g, "spill").unwrap(), Vec::<u8>::new());
        // Shrinking an existing key frees room, and rewriting one at the same
        // size is not double-counted.
        state_set(&g, "c0", &chunk).unwrap();
        state_set(&g, "c0", b"tiny").unwrap();
        state_set(&g, "spill", b"x").unwrap();
        let _ = std::fs::remove_dir_all(&home);
    }
}
