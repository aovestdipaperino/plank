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
pub trait WasmHost {
    /// Loads a module and completes the ABI handshake.
    ///
    /// # Errors
    /// [`WasmError::Load`] when the bytes are not a loadable module,
    /// [`WasmError::Abi`] when `plank_abi` is absent or disagrees.
    fn load(&mut self, source: &str, wasm: &[u8]) -> Result<LoadedPlugin, WasmError>;

    /// Calls an export with a byte payload, returning its byte reply.
    ///
    /// # Errors
    /// [`WasmError::Trap`] when the guest traps or exceeds its budget.
    fn call(&mut self, export: &str, input: &[u8]) -> Result<Vec<u8>, WasmError>;

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
    fn load(&mut self, _source: &str, _wasm: &[u8]) -> Result<LoadedPlugin, WasmError> {
        Err(WasmError::Unsupported)
    }

    fn call(&mut self, _export: &str, _input: &[u8]) -> Result<Vec<u8>, WasmError> {
        Err(WasmError::Unsupported)
    }
}

/// The host plank uses in this build: Extism when the feature is on, the no-op
/// otherwise. Returned boxed so the choice is invisible to callers.
#[must_use]
pub fn host() -> Box<dyn WasmHost> {
    #[cfg(feature = "plugins")]
    {
        Box::new(ExtismHost::default())
    }
    #[cfg(not(feature = "plugins"))]
    {
        Box::new(NoWasmHost)
    }
}

#[cfg(feature = "plugins")]
pub use extism_host::ExtismHost;

#[cfg(feature = "plugins")]
mod extism_host {
    use super::{ABI_VERSION, LoadedPlugin, WasmError, WasmHost};

    /// Wall-clock backstop for a single call, in milliseconds.
    ///
    /// Fuel alone is not enough: it meters guest instructions and cannot see
    /// time spent inside a host function, so a guest that stalls in a granted
    /// capability would run forever on an unexhausted fuel budget. This is the
    /// *outer* bound — a real per-surface budget (a frame step gets ~2 ms) is
    /// Phase 2's business, and will be far tighter than this.
    const CALL_TIMEOUT_MS: u64 = 1_000;

    /// Extism-backed host. One instance holds at most one plugin at this
    /// stage; the registry that holds many is Phase 1.
    #[derive(Default)]
    pub struct ExtismHost {
        plugin: Option<extism::Plugin>,
    }

    impl std::fmt::Debug for ExtismHost {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ExtismHost")
                .field("loaded", &self.plugin.is_some())
                .finish()
        }
    }

    impl WasmHost for ExtismHost {
        fn load(&mut self, source: &str, wasm: &[u8]) -> Result<LoadedPlugin, WasmError> {
            let manifest = extism::Manifest::new([extism::Wasm::data(wasm.to_vec())])
                .with_timeout(std::time::Duration::from_millis(CALL_TIMEOUT_MS));
            let mut plugin = extism::Plugin::new(&manifest, [], true)
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
            self.plugin = Some(plugin);
            Ok(LoadedPlugin {
                source: source.to_string(),
                abi: claimed,
            })
        }

        fn call(&mut self, export: &str, input: &[u8]) -> Result<Vec<u8>, WasmError> {
            let plugin = self
                .plugin
                .as_mut()
                .ok_or_else(|| WasmError::Load("no plugin loaded".to_string()))?;
            plugin
                .call::<&[u8], &[u8]>(export, input)
                .map(<[u8]>::to_vec)
                .map_err(|e| WasmError::Trap(format!("{export}: {e}")))
        }

        fn is_live(&self) -> bool {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the no-op is that it is always constructible and never
    /// panics, so no call site needs a `#[cfg]`.
    #[test]
    fn the_noop_host_refuses_everything_without_panicking() {
        let mut h = NoWasmHost;
        assert_eq!(h.load("x.wasm", b"\0asm"), Err(WasmError::Unsupported));
        assert_eq!(h.call("plank_abi", b""), Err(WasmError::Unsupported));
        assert!(!h.is_live(), "the no-op must not claim it can run plugins");
    }

    /// `host()` compiles and returns something usable either way; only
    /// `is_live` differs, which is exactly the one fact a caller may branch on.
    #[test]
    fn the_selected_host_matches_the_feature() {
        let h = host();
        assert_eq!(h.is_live(), cfg!(feature = "plugins"));
    }
}
