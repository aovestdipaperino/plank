// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! WASM plugin host — feasibility stage (`docs/WASM-PLUGINS.md`).
//!
//! This is the trait boundary the design calls for and nothing more: no
//! surfaces, no event bus, no manifest, no capabilities. What it exists to
//! prove is that the *shape* holds — that plank can talk to a plugin runtime
//! through one trait whose no-op implementation is always available, exactly
//! as [`crate::engine::Engine`] is always satisfiable by `EchoEngine`.
//!
//! [`NoWasmHost`] is that no-op, and it is what the whole of plank sees when
//! the `plugins` feature is off. [`ExtismHost`] is the real implementation,
//! compiled only with the feature. No caller may name either type: callers
//! take `&mut dyn WasmHost`, so turning the feature off cannot fail to
//! compile something.
//!
//! ## What the ABI handshake is for
//!
//! Extism will happily let a guest export `frame_step` with the wrong payload
//! shape and only fail deep inside a frame, so the guest is made to *assert*
//! what shape it speaks: a mandatory `plank_abi` export returning the ABI
//! major it was built against. A mismatch is a load error, never a warning.
//! This substitutes for the static type checking the Component Model would
//! have given us, and it is the reason the design can afford Extism's
//! bytes-in/bytes-out convention.

/// The plugin ABI major version this build of plank implements.
///
/// Bumped only when an export signature or a payload shape changes
/// incompatibly; adding an event, an optional payload field or a capability
/// does not bump it. A guest whose `plank_abi` disagrees is refused.
pub const ABI_VERSION: u32 = 1;

/// Why a plugin could not be loaded or called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmError {
    /// The `plugins` feature is not compiled in.
    Unsupported,
    /// The module would not load, or is not a plugin at all.
    Load(String),
    /// The mandatory `plank_abi` export is missing, unreadable, or disagrees
    /// with [`ABI_VERSION`]. Carries what the guest claimed, when it said
    /// anything at all.
    Abi(String),
    /// The guest trapped, ran out of fuel, or blew its wall-clock deadline.
    /// A plugin that breaks must degrade its own feature and nothing else, so
    /// every caller treats this as "disable the plugin", never "fail the
    /// session".
    Trap(String),
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "this build has no WASM plugin support"),
            Self::Load(m) => write!(f, "plugin failed to load: {m}"),
            Self::Abi(m) => write!(f, "plugin ABI mismatch: {m}"),
            Self::Trap(m) => write!(f, "plugin trapped: {m}"),
        }
    }
}

/// A loaded plugin's identity, as far as the feasibility stage cares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedPlugin {
    /// Where the module came from, for diagnostics.
    pub source: String,
    /// The ABI major the guest asserted.
    pub abi: u32,
}

/// What plank talks to instead of a plugin runtime.
///
/// Deliberately narrow at this stage. Every method that can fail returns
/// [`WasmError`] rather than panicking or propagating a runtime type, so the
/// runtime never appears in a signature outside this module and swapping it
/// stays a one-file change.
/// `Send` is required, not incidental: [`crate::tools::ToolContext`] owns a
/// host and is moved into the scoped thread that drives a generation pass.
/// `Sync` deliberately is *not* — the design forbids re-entrant calls into one
/// plugin, and plank serializes per instance, so shared access is exactly the
/// thing that must stay impossible.
pub trait WasmHost: std::fmt::Debug + Send {
    /// Loads a module under `source`, granting exactly `granted`, and
    /// completes the ABI handshake.
    ///
    /// `granted` is capability *labels* rather than the registry's enum, so
    /// this layer stays free of the registry's types — the dependency runs one
    /// way, registry to host.
    ///
    /// # Errors
    /// [`WasmError::Load`] when the bytes are not a loadable module,
    /// [`WasmError::Abi`] when `plank_abi` is absent or disagrees.
    fn load(
        &mut self,
        source: &str,
        wasm: &[u8],
        granted: &[&str],
    ) -> Result<LoadedPlugin, WasmError>;

    /// Takes the scrollback lines components printed since the last drain.
    ///
    /// A host function cannot reach the UI directly — `print` may be called
    /// from inside a guest call that is itself running on the UI thread — so
    /// lines are buffered and collected here, at the call boundary, where the
    /// caller already owns the screen.
    fn drain_printed(&mut self) -> Vec<String> {
        Vec::new()
    }

    /// Calls an export on the plugin loaded under `id`, with a byte payload,
    /// returning its byte reply.
    ///
    /// `id` rather than an opaque handle: the registry already keys everything
    /// on the component id, and a second identity to keep in sync would only be
    /// a second thing that can drift.
    ///
    /// # Errors
    /// [`WasmError::Trap`] when the guest traps or exceeds its budget,
    /// [`WasmError::Load`] when nothing is loaded under `id`.
    fn call(&mut self, id: &str, export: &str, input: &[u8]) -> Result<Vec<u8>, WasmError>;

    /// Whether an export exists on the plugin loaded under `id`.
    ///
    /// A surface's contract is "these exports exist", and claiming a surface
    /// you did not implement is a load-time error rather than a mystery at the
    /// first call. The no-op answers `false` for everything, which is honest:
    /// it has nothing loaded.
    fn has_export(&self, _id: &str, _export: &str) -> bool {
        false
    }

    /// Whether this host can actually run plugins. `false` for the no-op, and
    /// the one thing a caller may branch on: a `/plugins` listing says "not
    /// supported in this build" rather than "no plugins installed".
    fn is_live(&self) -> bool {
        false
    }
}

/// The always-available host: refuses everything, cheerfully.
///
/// Not a stub in the apologetic sense — it is the correct implementation for a
/// build without the feature, and it keeps every call site free of `#[cfg]`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoWasmHost;

impl WasmHost for NoWasmHost {
    fn load(
        &mut self,
        _source: &str,
        _wasm: &[u8],
        _granted: &[&str],
    ) -> Result<LoadedPlugin, WasmError> {
        Err(WasmError::Unsupported)
    }

    fn call(&mut self, _id: &str, _export: &str, _input: &[u8]) -> Result<Vec<u8>, WasmError> {
        Err(WasmError::Unsupported)
    }
}

/// The host plank uses in this build: Extism when the feature is on, the no-op
/// otherwise. Returned boxed so the choice is invisible to callers.
///
/// `home` is the plank home directory, the root of the `state` capability's
/// per-component storage. `None` leaves `state` unavailable rather than
/// inventing a location for it.
#[must_use]
pub fn host(home: Option<&std::path::Path>) -> Box<dyn WasmHost + Send> {
    #[cfg(feature = "plugins")]
    {
        Box::new(ExtismHost::new(home))
    }
    #[cfg(not(feature = "plugins"))]
    {
        let _ = home;
        Box::new(NoWasmHost)
    }
}

#[cfg(feature = "plugins")]
pub use extism_host::ExtismHost;

#[cfg(feature = "plugins")]
mod extism_host {
    use std::sync::{Arc, Mutex};

    use super::{ABI_VERSION, LoadedPlugin, WasmError, WasmHost};
    use crate::wasmcaps::{Grants, SharedSink, grants_for};

    /// Wall-clock backstop for a single call, in milliseconds.
    ///
    /// Fuel alone is not enough: it meters guest instructions and cannot see
    /// time spent inside a host function, so a guest that stalls in a granted
    /// capability would run forever on an unexhausted fuel budget. This is the
    /// *outer* bound — a real per-surface budget (a frame step gets ~2 ms) is
    /// the frame surface's business, and will be far tighter than this.
    pub(super) const CALL_TIMEOUT_MS: u64 = 1_000;

    /// Per-surface soft budget, in milliseconds, keyed by export prefix.
    ///
    /// The wall-clock timeout above is one number for every call, which is
    /// wrong in both directions: a second is absurdly generous for a frame step
    /// that has 33 ms of frame to fit inside, and stingy for a tool the user is
    /// already waiting on the model for. These are the numbers that actually
    /// describe each surface.
    ///
    /// These are the *targets*. Enforcement is at [`OVERRUN_FACTOR`] times them,
    /// because the measurement is wall-clock and a wall-clock measurement on a
    /// busy machine measures the machine. A frame steps around twenty times a
    /// second and three consecutive failures disable a component, so enforcing
    /// at the target exactly would strike out a perfectly good game on a loaded
    /// laptop — which is the failure this project has already recorded once, in
    /// a test that asserted 10ms and passed alone while failing under load.
    ///
    /// This is **not** fuel metering. Fuel meters guest instructions and needs
    /// wasmtime configuration Extism does not expose; these are wall-clock
    /// measurements taken host-side, which is strictly weaker — a guest can
    /// still burn its whole budget inside one host call — but it is the half
    /// that stops a plugin ruining the frame rate.
    /// How far over its target a call may go before it counts as broken rather
    /// than merely unlucky.
    ///
    /// Four times, so a frame step has to take over 200ms — unambiguous
    /// breakage, not a busy machine — before it costs a strike.
    pub(super) const OVERRUN_FACTOR: u128 = 4;

    pub(super) fn surface_budget_ms(export: &str) -> u64 {
        match export {
            // A frame at 30fps has 33ms for everything, and plank has to draw
            // too. The built-in arcade holds itself to the same order.
            e if e.starts_with("frame_step") => 50,
            // The status bar repaints many times a second while the user types,
            // and input has to feel immediate or the component feels broken —
            // same number, and for the same reason: both are in the way of a
            // keystroke.
            "segment_render" | "frame_key" | "frame_mouse" => 20,
            // The user is already waiting on the model, so this one may think.
            "tool_call" => CALL_TIMEOUT_MS,
            // Everything else — open, close, commands, events — is occasional.
            _ => 200,
        }
    }

    /// What a host function needs to answer a call: who is asking, and where
    /// its output goes.
    #[derive(Debug, Clone)]
    struct CallCtx {
        grants: Grants,
        sink: SharedSink,
    }

    extism::host_fn!(plank_log(ctx: CallCtx; level: String, message: String) -> String {
        let ctx = ctx.get()?;
        let ctx = ctx.lock().unwrap();
        Ok(match crate::wasmcaps::log(&ctx.grants, &level, &message) {
            Ok(()) => String::new(),
            Err(e) => e,
        })
    });

    extism::host_fn!(plank_print(ctx: CallCtx; text: String) -> String {
        let ctx = ctx.get()?;
        let ctx = ctx.lock().unwrap();
        Ok(match crate::wasmcaps::print(&ctx.grants, &ctx.sink, &text) {
            Ok(()) => String::new(),
            Err(e) => e,
        })
    });

    extism::host_fn!(plank_sound(ctx: CallCtx; cue: String) -> String {
        let ctx = ctx.get()?;
        let ctx = ctx.lock().unwrap();
        Ok(match crate::wasmcaps::sound(&ctx.grants, ctx.grants.sound, &cue) {
            Ok(()) => String::new(),
            Err(e) => e,
        })
    });

    extism::host_fn!(plank_notify(ctx: CallCtx; title: String, body: String) -> String {
        let ctx = ctx.get()?;
        let ctx = ctx.lock().unwrap();
        Ok(match crate::wasmcaps::notify(&ctx.grants, &title, &body) {
            Ok(()) => String::new(),
            Err(e) => e,
        })
    });

    extism::host_fn!(plank_state_get(ctx: CallCtx; key: String) -> Vec<u8> {
        let ctx = ctx.get()?;
        let ctx = ctx.lock().unwrap();
        // A refusal reads as empty rather than trapping the guest: `state_get`
        // already has "nothing stored" as a legitimate answer, and a component
        // that was denied the grant is in exactly that position.
        Ok(crate::wasmcaps::state_get(&ctx.grants, &key).unwrap_or_default())
    });

    extism::host_fn!(plank_state_set(ctx: CallCtx; key: String, value: Vec<u8>) -> String {
        let ctx = ctx.get()?;
        let ctx = ctx.lock().unwrap();
        Ok(match crate::wasmcaps::state_set(&ctx.grants, &key, &value) {
            Ok(()) => String::new(),
            Err(e) => e,
        })
    });

    /// Extism-backed host holding every loaded plugin, keyed by component id.
    ///
    /// One instance per session. Extism plugins are not `Sync` and plank
    /// serializes per instance anyway (the design forbids re-entrant calls into
    /// one plugin), so a plain map behind `&mut self` is the whole story.
    pub struct ExtismHost {
        plugins: std::collections::HashMap<String, extism::Plugin>,
        /// Where the `state` capability may write, if anywhere.
        home: Option<std::path::PathBuf>,
        /// Shared by every plugin's host functions; drained at call boundaries.
        sink: SharedSink,
    }

    impl ExtismHost {
        /// A host whose components store `state` under `home`.
        pub fn new(home: Option<&std::path::Path>) -> Self {
            Self {
                plugins: std::collections::HashMap::new(),
                home: home.map(std::path::Path::to_path_buf),
                sink: Arc::new(Mutex::new(crate::wasmcaps::CapSink::default())),
            }
        }
    }

    impl std::fmt::Debug for ExtismHost {
        // Hand-written because `extism::Plugin` is not `Debug`. A count is also
        // the only useful thing to say about it: dumping every loaded module's
        // internals into a log would bury the fields that matter.
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ExtismHost")
                .field("loaded", &self.plugins.len())
                .field("home", &self.home)
                .field("pending_prints", &self.sink.lock().map(|s| s.is_empty()))
                .finish()
        }
    }

    impl WasmHost for ExtismHost {
        fn load(
            &mut self,
            source: &str,
            wasm: &[u8],
            granted: &[&str],
        ) -> Result<LoadedPlugin, WasmError> {
            let manifest = extism::Manifest::new([extism::Wasm::data(wasm.to_vec())])
                .with_timeout(std::time::Duration::from_millis(CALL_TIMEOUT_MS));

            // Every host function is provided to every component, and each
            // checks its own grant when called. Providing only the granted ones
            // would make an ungranted import a *load* failure whose message is
            // about a missing wasm import — true, and useless to the user who
            // wants to know which capability to add. See `wasmcaps`.
            let ctx = extism::UserData::new(CallCtx {
                grants: grants_for(source, granted, self.home.as_deref()),
                sink: Arc::clone(&self.sink),
            });
            let functions = [
                extism::Function::new(
                    "plank_log",
                    [extism::PTR, extism::PTR],
                    [extism::PTR],
                    ctx.clone(),
                    plank_log,
                ),
                extism::Function::new(
                    "plank_print",
                    [extism::PTR],
                    [extism::PTR],
                    ctx.clone(),
                    plank_print,
                ),
                extism::Function::new(
                    "plank_sound",
                    [extism::PTR],
                    [extism::PTR],
                    ctx.clone(),
                    plank_sound,
                ),
                extism::Function::new(
                    "plank_notify",
                    [extism::PTR, extism::PTR],
                    [extism::PTR],
                    ctx.clone(),
                    plank_notify,
                ),
                extism::Function::new(
                    "plank_state_get",
                    [extism::PTR],
                    [extism::PTR],
                    ctx.clone(),
                    plank_state_get,
                ),
                extism::Function::new(
                    "plank_state_set",
                    [extism::PTR, extism::PTR],
                    [extism::PTR],
                    ctx,
                    plank_state_set,
                ),
            ];

            let mut plugin = extism::Plugin::new(&manifest, functions, true)
                .map_err(|e| WasmError::Load(format!("{source}: {e}")))?;

            // The handshake, before anything else is asked of the guest: a
            // module that cannot say what ABI it speaks is not a plugin.
            let claimed = plugin
                .call::<&[u8], &[u8]>("plank_abi", &[])
                .map_err(|e| WasmError::Abi(format!("{source}: no usable plank_abi export: {e}")))
                .and_then(|out| {
                    std::str::from_utf8(out)
                        .ok()
                        .and_then(|s| s.trim().parse::<u32>().ok())
                        .ok_or_else(|| {
                            WasmError::Abi(format!(
                                "{source}: plank_abi returned unparseable bytes"
                            ))
                        })
                })?;
            if claimed != ABI_VERSION {
                return Err(WasmError::Abi(format!(
                    "{source}: plugin speaks ABI {claimed}, this plank implements {ABI_VERSION}"
                )));
            }
            self.plugins.insert(source.to_string(), plugin);
            Ok(LoadedPlugin {
                source: source.to_string(),
                abi: claimed,
            })
        }

        fn call(&mut self, id: &str, export: &str, input: &[u8]) -> Result<Vec<u8>, WasmError> {
            let plugin = self
                .plugins
                .get_mut(id)
                .ok_or_else(|| WasmError::Load(format!("nothing loaded under '{id}'")))?;
            let started = std::time::Instant::now();
            let out = plugin
                .call::<&[u8], &[u8]>(export, input)
                .map(<[u8]>::to_vec)
                .map_err(|e| WasmError::Trap(format!("{id}:{export}: {e}")))?;
            // Measured after the call rather than enforced during it: Extism's
            // timeout is fixed at instantiation, so this is the only place the
            // budget can differ per surface.
            let elapsed = started.elapsed().as_millis();
            let budget = u128::from(surface_budget_ms(export));
            let ceiling = budget * OVERRUN_FACTOR;
            if elapsed > ceiling {
                return Err(WasmError::Trap(format!(
                    "{id}:{export}: took {elapsed}ms, more than {OVERRUN_FACTOR}x the {budget}ms \
                     budget for this surface"
                )));
            }
            Ok(out)
        }

        fn has_export(&self, id: &str, export: &str) -> bool {
            self.plugins
                .get(id)
                .is_some_and(|p| p.function_exists(export))
        }

        fn drain_printed(&mut self) -> Vec<String> {
            self.sink.lock().map(|mut s| s.drain()).unwrap_or_default()
        }

        fn is_live(&self) -> bool {
            true
        }
    }
}

#[cfg(test)]
mod tests {

    /// The budgets have to differ by surface, or the feature is just the old
    /// single timeout with more words.
    #[cfg(feature = "plugins")]
    #[test]
    fn surface_budgets_differ_and_favour_the_right_calls() {
        // Enforcement is deliberately looser than the target: a wall-clock
        // measurement on a busy machine measures the machine, and a frame that
        // steps twenty times a second would otherwise strike out on a loaded
        // laptop rather than because it is broken.
        const _: () = assert!(super::extism_host::OVERRUN_FACTOR > 1);

        use super::extism_host::surface_budget_ms as budget;

        // A frame step must be far tighter than a tool call: one has 33ms of
        // frame to fit inside, the other has a user already waiting on a model.
        assert!(budget("frame_step") < budget("tool_call"));
        assert!(
            budget("frame_step_text") == budget("frame_step"),
            "both draw"
        );
        // Input has to feel immediate.
        assert!(budget("frame_key") <= budget("frame_step"));
        assert!(budget("frame_mouse") <= budget("frame_step"));
        // The status bar repaints many times a second, competing with typing.
        assert!(budget("segment_render") < budget("frame_step"));
        // Anything unrecognised gets the occasional-call default, which must be
        // neither the tightest nor the most generous.
        let other = budget("command_run");
        assert!(other > budget("segment_render"));
        assert!(other < budget("tool_call"));
        // And nothing exceeds the hard wall-clock backstop.
        for export in [
            "frame_step",
            "frame_step_text",
            "frame_key",
            "frame_mouse",
            "segment_render",
            "tool_call",
            "command_run",
            "on_event",
        ] {
            assert!(
                budget(export) <= super::extism_host::CALL_TIMEOUT_MS,
                "{export} claims more than the outer bound"
            );
        }
    }
    use super::*;

    /// The point of the no-op is that it is always constructible and never
    /// panics, so no call site needs a `#[cfg]`.
    #[test]
    fn the_noop_host_refuses_everything_without_panicking() {
        let mut h = NoWasmHost;
        assert_eq!(h.load("x.wasm", b"\0asm", &[]), Err(WasmError::Unsupported));
        assert!(h.drain_printed().is_empty());
        assert_eq!(h.call("any", "plank_abi", b""), Err(WasmError::Unsupported));
        assert!(!h.has_export("any", "plank_abi"));
        assert!(!h.is_live(), "the no-op must not claim it can run plugins");
    }

    /// `host()` compiles and returns something usable either way; only
    /// `is_live` differs, which is exactly the one fact a caller may branch on.
    #[test]
    fn the_selected_host_matches_the_feature() {
        let h = host(None);
        assert_eq!(h.is_live(), cfg!(feature = "plugins"));
    }
}
