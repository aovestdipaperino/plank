// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! WASM component discovery, trust and registry (`docs/WASM-PLUGINS.md`,
//! Phase 1).
//!
//! A WASM component is not a plugin of its own: it is one more **component
//! kind** inside the plugin directories [`crate::plugins`] already loads,
//! alongside skills, agents, templates, hooks and MCP servers. That decision
//! is recorded in the design doc; the consequence here is that this module
//! never scans the filesystem for plugins. It is handed a loaded
//! [`crate::plugins::PluginSet`] and looks *inside* it.
//!
//! Everything in this module is compiled unconditionally, with no dependency
//! on the `plugins` feature. Discovery, manifest parsing and trust are pure
//! logic and are worth testing in the default build; only *running* a module
//! needs the runtime, and that goes through [`crate::wasmhost::WasmHost`],
//! whose no-op refuses cleanly. A build without the feature therefore still
//! lists what is installed and still explains why it is not running.
//!
//! ## Trust, and why project-local is the sharp edge
//!
//! Sandboxing is the containment story; trust is the provenance one. The rule
//! is that a component's SHA-256 **is** its identity — the same discipline
//! `session.rs` applies to KV blobs — so changed bytes are a different
//! component and re-prompt, and a capability set that grows re-prompts even
//! when the bytes are known.
//!
//! Project-local components (`./.plank/plugins/`) are default-deny, while
//! project-local *skills* are not. That asymmetry is deliberate and is the one
//! thing in this module most likely to look like a bug: cloning a repo already
//! activates its skills and MCP servers, which is tolerable, and would activate
//! a `.wasm` holding `exec`, which is not. The question trust answers is what
//! the code can reach, not where its directory sits.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::plugins::{Origin, Plugin, PluginSet};
use crate::tools::mcp::{Json, json_parse};

/// A surface a component claims: the contract of exports plank will call.
///
/// `panel` is deliberately absent — cut from v1 as the one surface with no
/// consumer (see the design doc's *Decisions*). Adding a variant later is
/// additive and needs no ABI bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Surface {
    /// Owns the whole screen: screensavers and games.
    Frame,
    /// Owns a status-bar cell.
    Segment,
    /// Owns a model-facing tool.
    Tool,
    /// Owns a slash command.
    Command,
    /// Owns nothing; observes events.
    Observer,
}

impl Surface {
    /// The manifest spelling.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "frame" => Self::Frame,
            "segment" => Self::Segment,
            "tool" => Self::Tool,
            "command" => Self::Command,
            "observer" => Self::Observer,
            _ => return None,
        })
    }

    /// The manifest spelling, for listings and diagnostics.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Frame => "frame",
            Self::Segment => "segment",
            Self::Tool => "tool",
            Self::Command => "command",
            Self::Observer => "observer",
        }
    }
}

/// A capability grant: everything a component can do to the outside world is
/// one of these, and nothing is granted by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Capability {
    /// Debug log only. The one capability that is always available, because it
    /// reaches nothing the user can see and nothing outside plank.
    Log,
    /// Writes scrollback lines.
    Print,
    /// Desktop or terminal notification.
    Notify,
    /// Per-component KV store. The only persistence most components need, and
    /// it needs no filesystem grant.
    State,
    /// An explicit path list. Never `/`.
    Fs,
    /// An explicit host list.
    Net,
    /// Shell. See [`Capability::undoes_the_sandbox`].
    Exec,
    /// Submits prompts to the model.
    Agent,
    /// Reads transcript turns.
    Session,
    /// Plays a sound cue.
    Sound,
}

impl Capability {
    /// The manifest spelling.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "log" => Self::Log,
            "print" => Self::Print,
            "notify" => Self::Notify,
            "state" => Self::State,
            "fs" => Self::Fs,
            "net" => Self::Net,
            "exec" => Self::Exec,
            "agent" => Self::Agent,
            "session" => Self::Session,
            "sound" => Self::Sound,
            _ => return None,
        })
    }

    /// The manifest spelling, for listings and prompts.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Print => "print",
            Self::Notify => "notify",
            Self::State => "state",
            Self::Fs => "fs",
            Self::Net => "net",
            Self::Exec => "exec",
            Self::Agent => "agent",
            Self::Session => "session",
            Self::Sound => "sound",
        }
    }

    /// Whether granting this means the component is not really sandboxed.
    ///
    /// `exec` hands out a shell and `net` hands out the network; a listing that
    /// renders them like `sound` is lying by omission. Kept as a predicate
    /// rather than a comment so the listing and the install prompt cannot
    /// disagree about which grants are the dangerous ones.
    #[must_use]
    pub fn undoes_the_sandbox(self) -> bool {
        matches!(self, Self::Exec | Self::Net | Self::Fs)
    }

    /// True when a host function actually stands behind this grant.
    ///
    /// `notify`, `agent` and `session` parse, appear in the approval prompt and
    /// reach nothing; `fs`, `net` and `exec` are the three left out on purpose.
    /// Approving a capability that does not exist is worse than refusing it —
    /// the user has consented to something, and nothing tells them it was
    /// nothing — so the loader warns rather than staying silent.
    #[must_use]
    pub fn is_wired(self) -> bool {
        matches!(self, Self::Log | Self::Print | Self::State | Self::Sound)
    }

    /// Why an unwired capability is unwired, for the load-time warning.
    #[must_use]
    pub fn unwired_reason(self) -> &'static str {
        if self.undoes_the_sandbox() {
            "deliberately unimplemented: it would undo the sandbox"
        } else {
            "declared but not implemented yet: the grant reaches no host function"
        }
    }
}

/// One mouse event for [`Session::frame_mouse`].
///
/// Cell coordinates, plus the frame's size in the same units, so a component
/// can hit-test without having to remember what it was last stepped with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMouse {
    /// `down`, `up`, `drag`, `move`, `scroll_up` or `scroll_down`.
    pub kind: &'static str,
    /// Column, 0-based, within the frame.
    pub x: u16,
    /// Row, 0-based, within the frame.
    pub y: u16,
    /// Frame width in cells.
    pub w: u16,
    /// Frame height in cells.
    pub h: u16,
}

/// Reads an `Outcome` reply from a frame call.
///
/// Shared by [`Session::frame_key`] and [`Session::frame_mouse`] so the two
/// cannot disagree about what closes a window. An unreadable reply is `Stay`: a
/// component that fumbles one reply should not have its window yanked out from
/// under the user.
fn parse_frame_outcome(bytes: &[u8]) -> FrameOutcome {
    let text = String::from_utf8_lossy(bytes);
    let Some(Json::Obj(fields)) = json_parse(&text) else {
        return FrameOutcome::Stay;
    };
    match fields.iter().find(|(k, _)| k == "close") {
        Some((_, Json::Str(line))) => FrameOutcome::Close(Some(line.clone())),
        Some((_, Json::Bool(true) | Json::Obj(_) | Json::Null)) => FrameOutcome::Close(None),
        _ => FrameOutcome::Stay,
    }
}

/// What kind of frame a component contributes.
///
/// A plugin that draws is one of two things, and they want opposite defaults:
/// a screensaver appears on its own and must be dismissable by anything, while
/// a game is started deliberately and owns its keys until it is quit. Making
/// that a declared *kind* rather than an activation flag means a manifest says
/// what the thing **is**, and the host derives when it may open from that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameKind {
    /// Ambient: joins the idle rotation beside the built-in faces, and can
    /// also be opened on demand. Any key dismisses it.
    Screensaver,
    /// A game: opened deliberately, never by the idle timer. The default,
    /// because a component that takes the whole terminal unasked is a surprise
    /// and a surprise should be opted into.
    #[default]
    Arcade,
}

impl FrameKind {
    /// The manifest spelling.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "screensaver" => Self::Screensaver,
            "arcade" | "game" => Self::Arcade,
            _ => return None,
        })
    }

    /// The manifest spelling, for listings.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Screensaver => "screensaver",
            Self::Arcade => "arcade",
        }
    }

    /// Whether the idle rotation may pick this.
    #[must_use]
    pub fn idle_eligible(self) -> bool {
        matches!(self, Self::Screensaver)
    }
}

/// What a plugin's manifest declares about one WASM component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmManifest {
    /// Reverse-DNS component id, globally unique and the trust store's key.
    pub id: String,
    /// ABI major the component claims, cross-checked against the module's own
    /// `plank_abi` export at load.
    pub abi: u32,
    /// Module file, relative to the plugin's `wasm/` directory.
    pub module: String,
    /// Surfaces claimed, deduplicated and sorted.
    pub surfaces: Vec<Surface>,
    /// Capabilities requested, deduplicated and sorted.
    pub capabilities: Vec<Capability>,
    /// What kind of frame this is. Meaningless for other surfaces, and
    /// defaulted rather than required so a manifest that claims no frame need
    /// not mention it.
    pub kind: FrameKind,
    /// Whether the transcript stays dimly visible under a `frame`.
    pub veiled: bool,
    /// Smallest area the component will draw into; below it the frame refuses
    /// to open rather than drawing into a space it cannot use.
    pub min_size: (u16, u16),
    /// The faces this component offers, in manifest order.
    ///
    /// Declared rather than inferred from the commands, because the two are
    /// different questions: a command is something a user may type, a face is
    /// something the *rotation* may pick and the settings picker must be able
    /// to enumerate. A component with no `frames` entry offers exactly one,
    /// named after the component, which is the single-face case written short.
    pub frames: Vec<String>,
    /// Events subscribed to, deduplicated and sorted. plank calls a component's
    /// handler only for these, so an unsubscribed event costs nothing.
    pub events: Vec<crate::wasmevents::EventKind>,
}

/// One discovered component: its manifest, where it came from, and the plugin
/// that contributed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmComponent {
    /// Contributing plugin's name, for `<plugin>:<id>` addressing and blame.
    pub plugin: String,
    /// Where the contributing plugin was activated from.
    pub origin: Origin,
    /// Absolute path of the `.wasm` module.
    pub path: PathBuf,
    /// The manifest section that declared it.
    pub manifest: WasmManifest,
}

impl WasmComponent {
    /// Whether this component asks for anything that undoes the sandbox.
    #[must_use]
    pub fn is_privileged(&self) -> bool {
        self.manifest
            .capabilities
            .iter()
            .any(|c| Capability::undoes_the_sandbox(*c))
    }
}

/// Everything discovered across a plugin set, plus what went wrong.
#[derive(Debug, Clone, Default)]
pub struct WasmSet {
    /// Components in plugin load order.
    pub components: Vec<WasmComponent>,
    /// Non-fatal complaints. A malformed component is skipped and named, never
    /// fatal: one bad manifest must not cost the user their other plugins.
    pub warnings: Vec<String>,
}

/// Reads the `wasm` array out of one plugin's manifest.
///
/// The design doc sketches a standalone `plugin.toml`; this uses the plugin
/// manifest that already exists (`.plank-plugin/plugin.json`) with a `wasm`
/// key, because a WASM component is a component of a plugin and not a plugin
/// of its own. One manifest, one parser, one place a user edits.
fn parse_manifest_section(text: &str) -> (Vec<WasmManifest>, Vec<String>) {
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    let Some(Json::Obj(top)) = json_parse(text) else {
        return (out, warnings);
    };
    let Some((_, entries)) = top.iter().find(|(k, _)| k == "wasm") else {
        return (out, warnings);
    };
    let entries = match entries {
        Json::Arr(items) => items.clone(),
        // A lone object is the shape a user writes first; accept it rather
        // than making them learn that one component still needs brackets.
        other @ Json::Obj(_) => vec![other.clone()],
        _ => {
            warnings.push("\"wasm\" must be an object or an array of objects".to_string());
            return (out, warnings);
        }
    };

    for entry in entries {
        let Json::Obj(fields) = entry else {
            warnings.push("every \"wasm\" entry must be an object".to_string());
            continue;
        };
        if let Some(m) = parse_one(&fields, &mut warnings) {
            out.push(m);
        }
    }
    (out, warnings)
}

/// Parses a component's capability grants, naming every one that is misspelled
/// or that reaches no host function.
///
/// Split out of [`parse_one`] to keep it under the function-length lint, and
/// because "which grants are real" is its own question.
fn parse_capabilities(id: &str, names: &[String], warnings: &mut Vec<String>) -> Vec<Capability> {
    let mut capabilities = Vec::new();
    for c in names {
        match Capability::parse(c) {
            Some(cap) => {
                if !cap.is_wired() {
                    warnings.push(format!(
                        "wasm component '{id}': capability '{c}' is {}",
                        cap.unwired_reason()
                    ));
                }
                capabilities.push(cap);
            }
            None => warnings.push(format!("wasm component '{id}': unknown capability '{c}'")),
        }
    }
    // `log` reaches nothing outside plank's own debug file, so it is always
    // present rather than something every author must remember to ask for.
    capabilities.push(Capability::Log);
    capabilities.sort_unstable();
    capabilities.dedup();
    capabilities
}

/// One entry of the `wasm` array. `None` — with a warning already pushed — for
/// anything malformed: a component that vanishes silently presents as "my
/// plugin does nothing", the least debuggable failure this system has.
fn parse_one(fields: &[(String, Json)], warnings: &mut Vec<String>) -> Option<WasmManifest> {
    let get_str = |key: &str| {
        fields.iter().find(|(k, _)| k == key).and_then(|(_, v)| {
            if let Json::Str(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
    };
    let list = |key: &str| -> Vec<String> {
        fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| match v {
                Json::Arr(items) => items
                    .iter()
                    .filter_map(|i| {
                        if let Json::Str(s) = i {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .collect(),
                Json::Str(s) => vec![s.clone()],
                _ => Vec::new(),
            })
            .unwrap_or_default()
    };

    let Some(id) = get_str("id").filter(|s| !s.is_empty()) else {
        warnings.push("a \"wasm\" entry has no \"id\"".to_string());
        return None;
    };
    let Some(module) = get_str("module").filter(|s| !s.is_empty()) else {
        warnings.push(format!("wasm component '{id}' has no \"module\""));
        return None;
    };
    // A module path is a file name under the plugin's `wasm/` directory, never
    // a traversal: a manifest must not be able to name `../../../etc/anything`,
    // and the check is here rather than at load so the component is refused
    // before it is ever counted as present.
    if module.contains('/') || module.contains('\\') || module.contains("..") {
        warnings.push(format!(
            "wasm component '{id}': \"module\" must be a bare file name under wasm/"
        ));
        return None;
    }
    let abi = fields
        .iter()
        .find(|(k, _)| k == "abi")
        .and_then(|(_, v)| match v {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Json::Num(n) if *n >= 0.0 => Some(*n as u32),
            Json::Str(s) => s.parse().ok(),
            _ => None,
        })
        .unwrap_or(crate::wasmhost::ABI_VERSION);

    let mut surfaces = Vec::new();
    for s in list("surfaces") {
        match Surface::parse(&s) {
            Some(surface) => surfaces.push(surface),
            // Named rather than silently dropped: a typo'd surface would
            // otherwise present as a component that loads and never runs.
            None => warnings.push(format!("wasm component '{id}': unknown surface '{s}'")),
        }
    }
    surfaces.sort_unstable();
    surfaces.dedup();
    if surfaces.is_empty() {
        warnings.push(format!("wasm component '{id}' claims no surfaces"));
        return None;
    }

    let frame = parse_frame_fields(fields, &id, warnings)?;

    let mut events = Vec::new();
    for e in list("events") {
        match crate::wasmevents::EventKind::parse(&e) {
            Some(kind) => events.push(kind),
            // A typo'd subscription is a handler that never fires, which
            // presents as "my component ignores everything".
            None => warnings.push(format!("wasm component '{id}': unknown event '{e}'")),
        }
    }
    events.sort_unstable();
    events.dedup();

    let capabilities = parse_capabilities(&id, &list("capabilities"), warnings);

    Some(WasmManifest {
        id,
        abi,
        module,
        surfaces,
        capabilities,
        kind: frame.kind,
        veiled: frame.veiled,
        min_size: frame.min_size,
        frames: frame.frames,
        events,
    })
}

/// The `frame`-only manifest fields, as one value.
///
/// A struct rather than a tuple because four fields of three different shapes
/// is exactly where a tuple stops documenting itself — `(FrameKind, bool,
/// (u16, u16), Vec<String>)` says nothing about which bool or which pair.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameFields {
    kind: FrameKind,
    veiled: bool,
    min_size: (u16, u16),
    frames: Vec<String>,
}

/// The `frame`-only manifest fields. Split out to keep [`parse_one`] readable,
/// and because they are the only fields that mean nothing to every other
/// surface.
///
/// Returns `None` — with a warning already pushed — only for an activation
/// this build does not know: that one is worth refusing the component over,
/// since guessing between "manual" and "idle" is the difference between a
/// screensaver that never appears and one that appears unasked.
fn parse_frame_fields(
    fields: &[(String, Json)],
    id: &str,
    warnings: &mut Vec<String>,
) -> Option<FrameFields> {
    let kind = match fields.iter().find(|(k, _)| k == "kind") {
        Some((_, Json::Str(s))) => {
            let Some(parsed) = FrameKind::parse(s) else {
                warnings.push(format!("wasm component '{id}': unknown frame kind '{s}'"));
                return None;
            };
            parsed
        }
        _ => FrameKind::default(),
    };
    let veiled = matches!(
        fields.iter().find(|(k, _)| k == "veiled"),
        Some((_, Json::Bool(true)))
    );
    let frames: Vec<String> = fields
        .iter()
        .find(|(k, _)| k == "frames")
        .map(|(_, v)| match v {
            Json::Arr(items) => items
                .iter()
                .filter_map(|i| match i {
                    Json::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            Json::Str(s) => vec![s.clone()],
            _ => Vec::new(),
        })
        .unwrap_or_default();
    let min_size = fields
        .iter()
        .find(|(k, _)| k == "min_size")
        .and_then(|(_, v)| match v {
            Json::Obj(dims) => {
                let dim = |key: &str| match dims.iter().find(|(k, _)| k == key) {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    Some((_, Json::Num(n))) if *n >= 0.0 => Some(*n as u16),
                    _ => None,
                };
                Some((dim("w").unwrap_or(0), dim("h").unwrap_or(0)))
            }
            _ => None,
        })
        .unwrap_or((0, 0));
    Some(FrameFields {
        kind,
        veiled,
        min_size,
        frames,
    })
}

/// Discovers every WASM component contributed by `set`.
///
/// Components are keyed by id across the whole set: two plugins declaring the
/// same id is a collision, and the *later* plugin loses, matching how the
/// plugin loader resolves contributions by load order. It warns rather than
/// failing, for the same reason it does there.
#[must_use]
pub fn discover(set: &PluginSet) -> WasmSet {
    let mut out = WasmSet::default();
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for plugin in &set.plugins {
        for m in discover_in(plugin, &mut out.warnings) {
            if let Some(first) = seen.get(&m.manifest.id) {
                out.warnings.push(format!(
                    "wasm component '{}' is declared by both '{first}' and '{}'; keeping {first}'s",
                    m.manifest.id, plugin.name
                ));
                continue;
            }
            seen.insert(m.manifest.id.clone(), plugin.name.clone());
            out.components.push(m);
        }
    }
    out
}

/// The per-plugin half of [`discover`].
fn discover_in(plugin: &Plugin, warnings: &mut Vec<String>) -> Vec<WasmComponent> {
    let Some(manifest_path) = crate::plugins::manifest_path(&plugin.root) else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(&manifest_path) else {
        return Vec::new();
    };
    let (manifests, complaints) = parse_manifest_section(&text);
    warnings.extend(
        complaints
            .into_iter()
            .map(|w| format!("plugin '{}': {w}", plugin.name)),
    );

    let mut out = Vec::new();
    for manifest in manifests {
        let path = plugin.root.join("wasm").join(&manifest.module);
        if !path.is_file() {
            warnings.push(format!(
                "plugin '{}': wasm component '{}' names a missing module wasm/{}",
                plugin.name, manifest.id, manifest.module
            ));
            continue;
        }
        out.push(WasmComponent {
            plugin: plugin.name.clone(),
            origin: plugin.origin,
            path,
            manifest,
        });
    }
    out
}

/// One recorded trust decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustEntry {
    /// SHA-256 of the module bytes the user approved. The hash *is* the
    /// identity; a different hash is a different component.
    pub sha256: String,
    /// Capabilities approved, sorted. A superset request re-prompts.
    pub granted: Vec<Capability>,
    /// Project directories this component was approved for. Empty for a
    /// user-global component; a project-local one is approved per repo, so
    /// cloning a second repo that ships the same bytes still asks.
    pub projects: Vec<PathBuf>,
    /// The publisher key id whose signature was accepted for these bytes, when
    /// the install was signed. This is what makes a later update quiet: new
    /// bytes signed by *this* publisher are the same provenance, whereas new
    /// bytes signed by anyone else are not.
    pub publisher: Option<String>,
}

/// What a component's signature says, computed before [`TrustStore::evaluate`]
/// so that function stays pure and testable.
///
/// See [`crate::wasmsig`] for the format. Signing is provenance, not
/// containment: the only thing a good signature buys is that an *update* from a
/// publisher already trusted for this component skips the re-prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SigStatus {
    /// No `.minisig` beside the module. The ordinary case, and not a problem:
    /// an unsigned plugin falls back to first-use hash approval.
    Absent,
    /// Verified against a publisher key the store holds. Carries the key id as
    /// minisign spells it, which is what gets recorded and shown.
    Valid(String),
    /// A signature by a key the store does not know. Not a failure — it just
    /// buys nothing, so trust falls back to the hash.
    UnknownKey(String),
    /// Present and did not verify. The one case that is actively wrong.
    Invalid(String),
}

/// What should happen to a component before it is loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Approved: load it.
    Trusted,
    /// Never seen. Ask, listing surfaces and capabilities.
    Unknown,
    /// Known id, different bytes. Ask again, saying so.
    Changed,
    /// Known bytes, but it now asks for capabilities that were not granted.
    /// Carries exactly the new ones, so the prompt can name them.
    Widened(Vec<Capability>),
    /// Known and approved, but not for *this* project directory.
    ProjectUnapproved,
    /// A signature was presented and did not verify. Refused rather than
    /// prompted: every other verdict means "we do not know about this", and
    /// this one means someone changed bytes that were signed.
    BadSignature(String),
}

impl Decision {
    /// Whether the component may load without asking the user first.
    #[must_use]
    pub fn is_trusted(&self) -> bool {
        matches!(self, Self::Trusted)
    }

    /// One line for a prompt or a listing.
    #[must_use]
    pub fn reason(&self) -> String {
        match self {
            Self::Trusted => "approved".to_string(),
            Self::Unknown => "not seen before".to_string(),
            Self::Changed => "its bytes changed since you approved it".to_string(),
            Self::Widened(caps) => format!(
                "it now also asks for: {}",
                caps.iter()
                    .map(|c| Capability::label(*c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::ProjectUnapproved => {
                "it is project-local and this project has not approved it".to_string()
            }
            Self::BadSignature(why) => format!("its signature did not verify: {why}"),
        }
    }
}

/// Recorded trust decisions, one file under the plank home.
#[derive(Debug, Clone, Default)]
pub struct TrustStore {
    /// Where it persists. `None` for an in-memory store (tests, no HOME).
    path: Option<PathBuf>,
    entries: BTreeMap<String, TrustEntry>,
    /// Accepted publisher keys, display id to the base64 key line. Held beside
    /// the per-component entries rather than inside them: a key is trusted for
    /// everything it signs, which is the whole point of having one.
    publishers: BTreeMap<String, String>,
}

impl TrustStore {
    /// Loads the store from `<home>/plugins/trust.json`, or an empty one.
    ///
    /// A corrupt file yields an empty store rather than an error: the failure
    /// mode of "we forgot your approvals and will ask again" is strictly safer
    /// than either trusting unparsed bytes or refusing to start.
    #[must_use]
    pub fn load(home: &Path) -> Self {
        let path = home.join("plugins").join("trust.json");
        let mut store = Self {
            path: Some(path.clone()),
            entries: BTreeMap::new(),
            publishers: BTreeMap::new(),
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return store;
        };
        let Some(Json::Obj(top)) = json_parse(&text) else {
            return store;
        };
        for (id, value) in top {
            if id == "@publishers" {
                if let Json::Obj(keys) = &value {
                    for (key_id, encoded) in keys {
                        if let Json::Str(encoded) = encoded {
                            store.publishers.insert(key_id.clone(), encoded.clone());
                        }
                    }
                }
                continue;
            }
            let Json::Obj(fields) = value else { continue };
            let field = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v);
            let Some(Json::Str(sha256)) = field("sha256") else {
                continue;
            };
            let granted = match field("granted") {
                Some(Json::Arr(items)) => items
                    .iter()
                    .filter_map(|i| match i {
                        Json::Str(s) => Capability::parse(s),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let projects = match field("projects") {
                Some(Json::Arr(items)) => items
                    .iter()
                    .filter_map(|i| match i {
                        Json::Str(s) => Some(PathBuf::from(s)),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let publisher = match field("publisher") {
                Some(Json::Str(p)) => Some(p.clone()),
                _ => None,
            };
            let mut granted: Vec<Capability> = granted;
            granted.sort_unstable();
            granted.dedup();
            store.entries.insert(
                id,
                TrustEntry {
                    sha256: sha256.clone(),
                    granted,
                    projects,
                    publisher,
                },
            );
        }
        store
    }

    /// An in-memory store that persists nothing. For tests and for a plank
    /// with no home directory, where asking every launch beats pretending an
    /// approval was recorded.
    #[must_use]
    pub fn ephemeral() -> Self {
        Self::default()
    }

    /// The recorded decision for `id`, if any.
    #[must_use]
    pub fn entry(&self, id: &str) -> Option<&TrustEntry> {
        self.entries.get(id)
    }

    /// Publisher keys the user has accepted: display id to the base64 key line.
    #[must_use]
    pub fn publishers(&self) -> &BTreeMap<String, String> {
        &self.publishers
    }

    /// Records a publisher key, so signatures from it count from now on.
    ///
    /// # Errors
    /// Returns the IO error message when the store cannot be written.
    pub fn add_publisher(
        &mut self,
        key: &crate::wasmsig::PublicKey,
        encoded: &str,
    ) -> Result<(), String> {
        self.publishers
            .insert(key.key_id_hex(), encoded.trim().to_string());
        self.persist()
    }

    /// Judges `component`, whose module hashes to `sha256`, in `project`.
    ///
    /// Checks in the order a user would want them explained: unknown before
    /// changed, changed before widened, and the project question last, since
    /// it is the only one whose answer is per-repo rather than per-artifact.
    #[must_use]
    pub fn evaluate(
        &self,
        component: &WasmComponent,
        sha256: &str,
        project: &Path,
        sig: &SigStatus,
    ) -> Decision {
        // Before anything else, including "we have never seen this": a
        // signature that does not verify is the only verdict here that means
        // something is wrong rather than merely unknown, and prompting the user
        // to approve it would be asking them to wave through the one case the
        // machine is certain about.
        if let SigStatus::Invalid(why) = sig {
            return Decision::BadSignature(why.clone());
        }
        let Some(entry) = self.entries.get(&component.manifest.id) else {
            // A valid signature does not auto-approve a first install: the
            // design buys quiet *updates*, not unattended first contact. The
            // user still sees the capabilities once.
            return Decision::Unknown;
        };
        if entry.sha256 != sha256 {
            // The one thing a signature buys. New bytes carrying a good
            // signature from the same publisher this component was approved
            // under are the same provenance, so they do not re-prompt — but
            // they still fall through to the capability check below, because a
            // publisher is not authorised to widen what their plugin may do.
            let same_publisher = matches!(
                (sig, entry.publisher.as_deref()),
                (SigStatus::Valid(signer), Some(known)) if signer == known
            );
            if !same_publisher {
                return Decision::Changed;
            }
        }
        let new: Vec<Capability> = component
            .manifest
            .capabilities
            .iter()
            .filter(|c| !entry.granted.contains(c))
            .copied()
            .collect();
        if !new.is_empty() {
            return Decision::Widened(new);
        }
        // Project-local is default-deny even when the bytes and grants are
        // known: approving a component in one repo is not approving every repo
        // that ships the same file. A user-scanned or --plugin-dir component
        // was named by the user directly and needs no per-project answer.
        if component.origin == Origin::ProjectScan && !entry.projects.iter().any(|p| p == project) {
            return Decision::ProjectUnapproved;
        }
        Decision::Trusted
    }

    /// Records approval of `component` at `sha256`, for `project` when the
    /// component is project-local. Persists immediately when the store is
    /// file-backed; a write failure is reported, never silent.
    ///
    /// # Errors
    /// Returns the underlying IO error message when the store cannot be
    /// written, so the caller can tell the user their answer will not stick.
    pub fn approve(
        &mut self,
        component: &WasmComponent,
        sha256: &str,
        project: &Path,
        sig: &SigStatus,
    ) -> Result<(), String> {
        let entry = self
            .entries
            .entry(component.manifest.id.clone())
            .or_insert_with(|| TrustEntry {
                sha256: sha256.to_string(),
                granted: Vec::new(),
                projects: Vec::new(),
                publisher: None,
            });
        // A re-approval after a change replaces the identity outright: the old
        // hash approved different bytes and must not linger beside the new one.
        entry.sha256 = sha256.to_string();
        entry.granted.clone_from(&component.manifest.capabilities);
        // Recorded on approval, and cleared when an install is unsigned or
        // signed by an unknown key: an entry claiming a publisher it was not
        // actually approved under would make the *next* update quiet on false
        // grounds, which is the one way this feature could weaken trust.
        entry.publisher = match sig {
            SigStatus::Valid(signer) => Some(signer.clone()),
            _ => None,
        };
        if component.origin == Origin::ProjectScan && !entry.projects.iter().any(|p| p == project) {
            entry.projects.push(project.to_path_buf());
        }
        self.persist()
    }

    /// Serializes the store. Written by hand for the same reason the plugin
    /// manifest is read by hand: this is a flat map of flat records, and the
    /// file is one a user may have to read or delete themselves.
    fn persist(&self) -> Result<(), String> {
        use std::fmt::Write as _;

        let Some(path) = &self.path else {
            return Ok(());
        };
        let mut out = String::from("{\n");
        // Under a reserved key rather than a second file: one file a user can
        // read, edit or delete. `@publishers` cannot collide with a component
        // id, which is reverse-DNS and never starts with `@`.
        if !self.publishers.is_empty() {
            let keys: Vec<String> = self
                .publishers
                .iter()
                .map(|(id, encoded)| format!("    {}: {}", json_str(id), json_str(encoded)))
                .collect();
            let _ = write!(out, "  \"@publishers\": {{\n{}\n  }},\n", keys.join(",\n"));
        }
        for (i, (id, e)) in self.entries.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            let granted: Vec<String> = e
                .granted
                .iter()
                .map(|c| json_str(Capability::label(*c)))
                .collect();
            let projects: Vec<String> = e
                .projects
                .iter()
                .map(|p| json_str(&p.display().to_string()))
                .collect();
            // `publisher` is written only when there is one, so an unsigned
            // install's record stays exactly the shape it has always been and
            // an older plank reading this file is unaffected.
            let publisher = e.publisher.as_ref().map_or_else(String::new, |p| {
                format!(",\n    \"publisher\": {}", json_str(p))
            });
            let _ = write!(
                out,
                "  {}: {{\n    \"sha256\": {},\n    \"granted\": [{}],\n    \"projects\": [{}]{}\n  }}",
                json_str(id),
                json_str(&e.sha256),
                granted.join(", "),
                projects.join(", "),
                publisher,
            );
        }
        out.push_str("\n}\n");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, out).map_err(|e| e.to_string())
    }
}

/// Minimal JSON string escaping for the trust file.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// One slash command a `command` component contributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    /// Name without the leading slash, as the component spells it.
    pub name: String,
    /// `<plugin>:<name>`, always available.
    ///
    /// The plugin loader's convention for every contributed entry: an alias
    /// that is always addressable, plus the bare name when nothing else claims
    /// it. A component offering several frames therefore gets `/arcade:matrix`
    /// and `/arcade:breakout` for free, which is the whole reason `/frame <id>
    /// <face>` does not have to be how anyone opens one.
    pub alias: String,
    /// Argument hint for the menu; empty when it takes none.
    pub args: String,
    /// One-line description.
    pub desc: String,
}

/// What a `command_run` asked plank to do.
///
/// Every field is optional and they compose: a command may print, *and* leave
/// text in the input box, *and* submit a prompt. Unknown fields are ignored, as
/// the payload-evolution rule requires.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CmdOutput {
    /// Lines to write to scrollback.
    pub print: Vec<String>,
    /// Text to place in the input box, replacing what is there.
    pub inject: Option<String>,
    /// Text to submit to the model as if typed.
    pub prompt: Option<String>,
    /// Open the calling component's own frame, with this string as its `arg`.
    ///
    /// Deliberately not "open any component's frame": a command opening
    /// someone else's window is a capability, and this is a convenience. It is
    /// how one module serving many frames — every arcade face in one `.wasm` —
    /// gives each face its own slash command.
    pub open: Option<String>,
}

/// Parses the `command_specs` reply.
fn parse_command_specs(bytes: &[u8]) -> Option<Vec<CommandSpec>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let Json::Arr(items) = json_parse(text)? else {
        return None;
    };
    let mut out = Vec::new();
    for item in items {
        let Json::Obj(fields) = item else { continue };
        let get = |key: &str| match fields.iter().find(|(k, _)| k == key) {
            Some((_, Json::Str(s))) => s.clone(),
            _ => String::new(),
        };
        let name = get("name");
        // A command with no name cannot be typed, so it is not a command.
        // Leading slashes are tolerated and stripped: authors write both.
        let name = name.trim_start_matches('/').to_string();
        if name.is_empty() {
            continue;
        }
        out.push(CommandSpec {
            alias: String::new(),
            name,
            args: get("args"),
            desc: get("desc"),
        });
    }
    Some(out)
}

/// Parses the `command_run` reply. A malformed reply is an empty outcome
/// rather than an error: the command ran, it just asked for nothing.
fn parse_cmd_output(bytes: &[u8]) -> CmdOutput {
    let mut out = CmdOutput::default();
    let Ok(text) = std::str::from_utf8(bytes) else {
        return out;
    };
    let Some(Json::Obj(fields)) = json_parse(text) else {
        return out;
    };
    for (key, value) in fields {
        match (key.as_str(), value) {
            ("print", Json::Arr(items)) => {
                out.print = items
                    .iter()
                    .filter_map(|i| match i {
                        Json::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
            }
            ("print", Json::Str(s)) => out.print = vec![s],
            ("inject", Json::Str(s)) => out.inject = Some(s),
            ("prompt", Json::Str(s)) => out.prompt = Some(s),
            ("open", Json::Str(s)) => out.open = Some(s),
            _ => {}
        }
    }
    out
}

/// One model-facing tool a `tool` component contributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmTool {
    /// Component that owns it, for dispatch and for blame.
    pub component: String,
    /// The tool's own name, as the component spells it.
    pub name: String,
    /// The name the *model* sees. Bare when nothing else claims it, and
    /// `wasm__<component>__<tool>` when something does — the precedent every
    /// other contributed registry in plank already follows, and the shape MCP
    /// namespaces with, so a user learns one convention rather than two.
    pub exposed: String,
    /// One-line description for the tools prompt.
    pub description: String,
    /// The parameter JSON Schema, re-serialized compactly. Held as text
    /// because plank never interprets it: it goes into the system prompt and
    /// the model is the only reader.
    pub schema: String,
}

/// Parses the `tool_specs` reply.
fn parse_tool_specs(component: &str, bytes: &[u8]) -> Option<Vec<WasmTool>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let Json::Arr(items) = json_parse(text)? else {
        return None;
    };
    let mut out = Vec::new();
    for item in items {
        let Json::Obj(fields) = item else { continue };
        let get = |key: &str| match fields.iter().find(|(k, _)| k == key) {
            Some((_, Json::Str(s))) => s.clone(),
            _ => String::new(),
        };
        let name = get("name");
        // A tool the model cannot name is not a tool. The character set is the
        // model's business, not ours, but a name with whitespace could not be
        // written as a DSML element at all.
        if name.is_empty() || name.split_whitespace().count() != 1 {
            continue;
        }
        let mut schema = String::new();
        match fields.iter().find(|(k, _)| k == "parameters") {
            Some((_, value @ Json::Obj(_))) => crate::tools::mcp::json_write(&mut schema, value),
            // A tool with no schema takes no parameters; saying so explicitly
            // beats emitting nothing, which reads to the model as "unknown".
            _ => schema.push_str(r#"{"type":"object","properties":{}}"#),
        }
        out.push(WasmTool {
            component: component.to_string(),
            name: name.clone(),
            exposed: name,
            description: get("description"),
            schema,
        });
    }
    Some(out)
}

/// One rendered status-bar cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Component that produced it.
    pub component: String,
    /// The text to show. Empty means "nothing to say right now", which is a
    /// segment's ordinary state and not a failure.
    pub text: String,
    /// Elision order when the bar overflows: higher survives longer. Plugin
    /// segments compete in the same order as the built-in ones.
    pub priority: u8,
}

/// How often a `segment` component may actually be called.
///
/// The design says "once per status-bar repaint". That is the wrong cadence
/// for this front end: the TUI repaints on every keystroke and every throbber
/// tick, so a call there costs fuel dozens of times a second and runs a guest
/// on the UI thread while the user is typing. A status cell's data changes on
/// the order of seconds, so it is rendered at most this often and read from
/// cache in between — the same reasoning that cut `token_batch` from v1.
pub const SEGMENT_MIN_INTERVAL_MS: u64 = 1_000;

/// Parses a `segment_render` reply.
fn parse_segment(component: &str, bytes: &[u8]) -> Option<Segment> {
    let text = std::str::from_utf8(bytes).ok()?;
    let Json::Obj(fields) = json_parse(text)? else {
        return None;
    };
    let get = |key: &str| match fields.iter().find(|(k, _)| k == key) {
        Some((_, Json::Str(s))) => s.clone(),
        _ => String::new(),
    };
    let priority = match fields.iter().find(|(k, _)| k == "priority") {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some((_, Json::Num(n))) if *n >= 0.0 => (*n as u32).min(255) as u8,
        _ => 128,
    };
    Some(Segment {
        component: component.to_string(),
        // A cell is one line by construction: a newline in it would tear the
        // bar apart, so it is folded rather than trusted.
        text: get("text").replace(['\n', '\r'], " "),
        priority,
    })
}

/// One screensaver face a component contributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaceRef {
    /// Component that draws it.
    pub component: String,
    /// The face's own name, as `frame_open`'s `arg`.
    pub face: String,
    /// `<plugin>:<face>` — what a user pins and what a picker lists.
    pub address: String,
}

/// A `frame` component that currently owns the screen.
///
/// Mirrors `arcade::Arcade`'s role rather than its shape: the host owns the
/// clock, the area and the seed, and the component owns only what it paints.
/// Everything time-related is passed *in* — a guest has no ambient clock and
/// no ambient randomness, the same discipline `arcade::Rng` imposes and for
/// the same reason: a seeded frame can be replayed exactly.
#[derive(Debug)]
pub struct OpenFrame {
    /// Component driving it.
    pub id: String,
    /// Opened by the idle rotation rather than by a command, which is what
    /// makes it dismissable by any activity: a screensaver a person has to
    /// press the right key to escape is not a screensaver.
    pub screensaver: bool,
    /// Whether the transcript stays dimly visible underneath.
    pub veiled: bool,
    /// The last frame it painted, kept so a repaint that arrives between steps
    /// has something to draw.
    pub last: crate::wasmglyph::GlyphFrame,
}

/// Longest `dt` a single step will carry, in milliseconds.
///
/// The same clamp `arcade::MAX_STEP_MS` applies to the built-in faces, and for
/// the same reason: the UI loop measures real elapsed time, so the first step
/// after a suspended terminal would otherwise teleport a simulation.
pub const MAX_STEP_MS: u64 = 100;

/// What a key press asks the caller to do with the open frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameOutcome {
    /// Stay open; the component absorbed the key.
    Stay,
    /// Close, optionally leaving this line in the scrollback.
    Close(Option<String>),
}

/// A session's WASM state:/// A session's WASM state:/// A session's WASM state:/// A session's WASM state: the admitted components and the runtime they live
/// in, kept together because neither is useful alone.
///
/// Defaults to "nothing discovered, nothing runnable", so a session that never
/// calls [`Session::activate`] behaves exactly as it did before this existed.
#[derive(Debug)]
pub struct Session {
    /// What was admitted, held back, or complained about.
    pub registry: Registry,
    /// The runtime. The no-op unless the `plugins` feature is on.
    pub host: Box<dyn crate::wasmhost::WasmHost + Send>,
    /// A frame the user asked to open, waiting for the UI loop to pick it up.
    ///
    /// The slash handler and the loop that owns the open frame are different
    /// functions with different borrows, and threading an `&mut Option<..>`
    /// through the handler's dozen parameters to connect them would be worse
    /// than a one-field handoff — the same shape `pending_aside` already uses
    /// for `/btw`.
    pub pending_open: Option<(String, String)>,
    /// The plank home: where trust is recorded and where the `state`
    /// capability writes. Held once, here, because the host and the trust
    /// store both need it and two callers passing the same value into one
    /// object is how they drift.
    home: Option<PathBuf>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Session {
    /// A session whose components store `state` under `home`.
    ///
    /// The default has no home, so `state` is unavailable — right for a test
    /// and for a plank that cannot find one, and wrong to paper over by
    /// guessing a directory.
    #[must_use]
    pub fn new(home: Option<&Path>) -> Self {
        Self {
            registry: Registry::default(),
            host: crate::wasmhost::host(home),
            pending_open: None,
            home: home.map(Path::to_path_buf),
        }
    }
}

impl Session {
    /// Discovers, judges and loads every component in `plugins`.
    ///
    /// Called once at startup, after the plugin set is known. Returns the
    /// warnings the caller should show; held components are not warnings —
    /// they are in [`Registry::held`] with a reason, because "we did not run
    /// this" is a different message from "this is broken".
    pub fn activate(&mut self, plugins: &PluginSet, project: &Path) -> Vec<String> {
        let found = discover(plugins);
        let trust = self
            .home
            .as_deref()
            .map_or_else(TrustStore::ephemeral, TrustStore::load);
        self.registry = Registry::build(&found, &trust, project, &mut *self.host);
        self.registry.warnings.clone()
    }

    /// Frame components that a slash command may open, as `(id, description)`.
    #[must_use]
    pub fn openable_frames(&self) -> Vec<(&str, &str)> {
        self.registry
            .loaded
            .iter()
            // Every frame component is openable on demand, whichever kind it
            // is: a screensaver the user wants to look at now is a reasonable
            // thing to ask for, and the kind only decides whether the *idle
            // timer* may put it up unasked.
            .filter(|l| {
                l.strikes < STRIKE_LIMIT && l.component.manifest.surfaces.contains(&Surface::Frame)
            })
            .map(|l| {
                (
                    l.component.manifest.id.as_str(),
                    l.component.plugin.as_str(),
                )
            })
            .collect()
    }

    /// Every screensaver face installed.
    ///
    /// The address is `<plugin>:<face>` — what a user writes in
    /// `ui.screensaverFace` to pin one, and what a settings picker lists.
    #[must_use]
    pub fn screensaver_faces(&self) -> Vec<FaceRef> {
        self.registry
            .loaded
            .iter()
            .filter(|l| {
                l.strikes < STRIKE_LIMIT
                    && l.component.manifest.surfaces.contains(&Surface::Frame)
                    && l.component.manifest.kind.idle_eligible()
            })
            .flat_map(|l| {
                let id = l.component.manifest.id.clone();
                let plugin = l.component.plugin.clone();
                // A component with no declared faces offers one, named after
                // the plugin: the single-face case, written short.
                let faces = if l.component.manifest.frames.is_empty() {
                    vec![plugin.clone()]
                } else {
                    l.component.manifest.frames.clone()
                };
                faces.into_iter().map(move |face| FaceRef {
                    address: format!("{plugin}:{face}"),
                    component: id.clone(),
                    face,
                })
            })
            .collect()
    }

    /// Resolves `<plugin>:<face>` to the component and face it names.
    #[must_use]
    pub fn resolve_screensaver_face(&self, address: &str) -> Option<(String, String)> {
        self.screensaver_faces()
            .into_iter()
            .find(|f| f.address == address)
            .map(|f| (f.component, f.face))
    }

    /// Which face the random rotation should put up for `seed`.
    ///
    /// `Some((id, face))` is a component's face; `None` means "use a built-in
    /// face". Every face gets one slot, plugin or not — the unit is a *face*,
    /// not a plugin, so a component offering three does not get one third the
    /// weight of the rain.
    ///
    /// Only reached when the user's face setting is `random`. A pinned face —
    /// built-in or plugin — is not a rotation and is not decided here.
    #[must_use]
    pub fn pick_idle_face(&self, seed: u64, builtin_faces: usize) -> Option<(String, String)> {
        let faces = self.screensaver_faces();
        let slots = builtin_faces + faces.len();
        if slots == 0 {
            return None;
        }
        let pick = usize::try_from(seed % slots as u64).unwrap_or(0);
        pick.checked_sub(builtin_faces)
            .and_then(|i| faces.get(i))
            .map(|f| (f.component.clone(), f.face.clone()))
    }

    /// Frame components the idle rotation may pick.
    /// Frame components the idle rotation may pick.
    #[must_use]
    pub fn idle_frames(&self) -> Vec<&str> {
        self.registry
            .loaded
            .iter()
            .filter(|l| {
                l.strikes < STRIKE_LIMIT
                    && l.component.manifest.surfaces.contains(&Surface::Frame)
                    && l.component.manifest.kind.idle_eligible()
            })
            .map(|l| l.component.manifest.id.as_str())
            .collect()
    }

    /// Opens `id` for a `w`×`h` area, seeded with `seed`.
    ///
    /// # Errors
    /// Returns a message when no live component owns `id`, when the area is
    /// below the component's declared minimum, or when `frame_open` traps.
    /// `arg` selects *which* frame a component offers, for a component that
    /// offers several — one module holding every arcade face, rather than one
    /// module per face. It is passed through untouched; a component with a
    /// single frame ignores it.
    pub fn open_frame(
        &mut self,
        id: &str,
        arg: &str,
        w: u16,
        h: u16,
        seed: u64,
    ) -> Result<OpenFrame, String> {
        let manifest = self
            .registry
            .loaded
            .iter()
            .find(|l| l.component.manifest.id == id && l.strikes < STRIKE_LIMIT)
            .map(|l| l.component.manifest.clone())
            .ok_or_else(|| format!("no wasm frame '{id}'"))?;
        let (min_w, min_h) = manifest.min_size;
        if w < min_w || h < min_h {
            return Err(format!(
                "{id} needs at least {min_w}x{min_h}; this terminal offers {w}x{h}"
            ));
        }
        let payload = format!(
            "{{\"w\": {w}, \"h\": {h}, \"seed\": {seed}, \"arg\": {}}}",
            json_str(arg)
        );
        if let Err(e) = self.host.call(id, "frame_open", payload.as_bytes()) {
            let disabled = self.registry.strike(id);
            return Err(if disabled {
                format!("{id}: {e} (disabled for this session)")
            } else {
                format!("{id}: {e}")
            });
        }
        Ok(OpenFrame {
            id: id.to_string(),
            screensaver: false,
            veiled: manifest.veiled,
            last: crate::wasmglyph::GlyphFrame::default(),
        })
    }

    /// Advances an open frame by `dt_ms` and decodes what it painted.
    ///
    /// A step that traps or returns an undecodable buffer closes the frame:
    /// there is no sensible way to keep a full-screen component on screen when
    /// it can no longer say what to draw, and leaving the last frame up
    /// forever would look like a hang.
    ///
    /// # Errors
    /// Returns the message to show after closing.
    pub fn step_frame(
        &mut self,
        frame: &mut OpenFrame,
        dt_ms: u64,
        w: u16,
        h: u16,
        now_ms: u64,
    ) -> Result<(), String> {
        let payload = format!(
            "{{\"dt_ms\": {}, \"w\": {w}, \"h\": {h}, \"now_ms\": {now_ms}}}",
            dt_ms.min(MAX_STEP_MS)
        );
        // A component exports one or the other. `frame_step` is the packed
        // buffer — one memcpy out of linear memory — and is what a component
        // graduates to; `frame_step_text` returns JSON rows and is the shape an
        // author writes first. Packed wins when a module somehow has both, so
        // adding the fallback cannot slow an existing component down.
        let text_only = !self.host.has_export(&frame.id, "frame_step")
            && self.host.has_export(&frame.id, "frame_step_text");
        let export = if text_only {
            "frame_step_text"
        } else {
            "frame_step"
        };
        let bytes = self
            .host
            .call(&frame.id, export, payload.as_bytes())
            .map_err(|e| {
                self.registry.strike(&frame.id);
                format!("{}: {e}", frame.id)
            })?;
        let decoded = if text_only {
            crate::wasmglyph::decode_text(&bytes, w, h)
        } else {
            crate::wasmglyph::decode(&bytes)
        };
        match decoded {
            Ok(decoded) => {
                frame.last = decoded;
                Ok(())
            }
            Err(e) => {
                self.registry.strike(&frame.id);
                Err(format!("{}: {e}", frame.id))
            }
        }
    }

    /// Delivers a key. `code` is the key's name (`"left"`, `"q"`, `"space"`).
    ///
    /// # Errors
    /// Returns the message to show when the call traps; the caller closes.
    pub fn frame_key(&mut self, frame: &OpenFrame, code: &str) -> Result<FrameOutcome, String> {
        let payload = format!("{{\"code\": {}}}", json_str(code));
        let bytes = self
            .host
            .call(&frame.id, "frame_key", payload.as_bytes())
            .map_err(|e| {
                self.registry.strike(&frame.id);
                format!("{}: {e}", frame.id)
            })?;
        Ok(parse_frame_outcome(&bytes))
    }

    /// Delivers a mouse event to an open frame.
    ///
    /// `frame_mouse` is optional — a keyboard-only game is a perfectly good
    /// component — so a module that does not export it stays open and is not
    /// struck: not implementing an optional export is not a failure. Without
    /// this the host enabled mouse reporting for every frame and then dropped
    /// every event, which is why all four ported games had their mouse
    /// handlers removed whole rather than left half-wired.
    ///
    /// # Errors
    /// Returns a message when the call itself fails, having struck the
    /// component.
    pub fn frame_mouse(
        &mut self,
        frame: &OpenFrame,
        mouse: &FrameMouse,
    ) -> Result<FrameOutcome, String> {
        if !self.host.has_export(&frame.id, "frame_mouse") {
            return Ok(FrameOutcome::Stay);
        }
        let payload = format!(
            "{{\"kind\": {}, \"x\": {}, \"y\": {}, \"w\": {}, \"h\": {}}}",
            json_str(mouse.kind),
            mouse.x,
            mouse.y,
            mouse.w,
            mouse.h
        );
        let bytes = self
            .host
            .call(&frame.id, "frame_mouse", payload.as_bytes())
            .map_err(|e| {
                self.registry.strike(&frame.id);
                format!("{}: {e}", frame.id)
            })?;
        Ok(parse_frame_outcome(&bytes))
    }

    /// Closes a frame, returning any scrollback line it asked to leave behind.
    pub fn close_frame(&mut self, frame: &OpenFrame) -> Option<String> {
        let bytes = self.host.call(&frame.id, "frame_close", b"").ok()?;
        let text = String::from_utf8_lossy(&bytes);
        let Some(Json::Obj(fields)) = json_parse(&text) else {
            return None;
        };
        match fields.iter().find(|(k, _)| k == "scrollback") {
            Some((_, Json::Str(line))) if !line.is_empty() => Some(line.clone()),
            _ => None,
        }
    }

    /// Approves a held component and loads it, without waiting for a restart.
    ///
    /// # Errors
    /// Returns a message when nothing is held under `id`, when the approval
    /// cannot be recorded, or when the module then fails to load.
    pub fn approve(&mut self, id: &str, project: &Path) -> Result<String, String> {
        let idx = self
            .registry
            .held
            .iter()
            .position(|(c, _)| c.manifest.id == id)
            .ok_or_else(|| format!("no held wasm component '{id}'"))?;
        let (component, _) = self.registry.held[idx].clone();
        let sha = module_sha256(&component.path)
            .ok_or_else(|| format!("cannot hash {}", component.path.display()))?;
        let mut trust = self
            .home
            .as_deref()
            .map_or_else(TrustStore::ephemeral, TrustStore::load);
        // The publisher recorded with the approval is whoever signed the bytes
        // being approved right now, not whoever signed them last time.
        let sig = signature_status(&component.path, &trust);
        trust.approve(&component, &sha, project, &sig)?;

        // Re-run the admission path for this one component rather than
        // hand-rolling a second load: the export cross-check and the
        // command-spec fetch must not have two implementations that can drift.
        let one = WasmSet {
            components: vec![component],
            warnings: Vec::new(),
        };
        let fresh = Registry::build(&one, &trust, project, &mut *self.host);
        if let Some(l) = fresh.loaded.into_iter().next() {
            self.registry.held.remove(idx);
            let name = l.component.manifest.id.clone();
            self.registry.loaded.push(l);
            return Ok(name);
        }
        Err(fresh
            .warnings
            .first()
            .cloned()
            .unwrap_or_else(|| format!("{id} was approved but would not load")))
    }
}

/// A component that was admitted, and the host handle it loaded into.
#[derive(Debug)]
pub struct Loaded {
    /// What was loaded.
    pub component: WasmComponent,
    /// Consecutive failures. At [`STRIKE_LIMIT`] the component is disabled for
    /// the rest of the session.
    pub strikes: u8,
    /// Model-facing tools this component contributes, read from `tool_specs`
    /// once at load. Once at load is not an optimization here but a
    /// requirement: the tool list is part of the system prompt, and a prompt
    /// that changed mid-session would invalidate the Tier 1 KV checkpoint
    /// every time it changed.
    pub tools: Vec<WasmTool>,
    /// Slash commands this component contributes, read from `command_specs`
    /// once at load. Read once rather than per keystroke: the menu is rebuilt
    /// on every input change, and a WASM call per frame to re-derive a list
    /// that cannot change is the kind of cost that turns a feature into a
    /// regression.
    pub commands: Vec<CommandSpec>,
}

/// How many times a component may trap before it is disabled for the session.
///
/// A plugin that breaks should degrade its own feature and nothing else, but a
/// plugin that breaks *every* frame degrades the session by being asked at all.
pub const STRIKE_LIMIT: u8 = 3;

/// The session's admitted components, and why the others were not.
#[derive(Debug, Default)]
pub struct Registry {
    /// Admitted components, in discovery order.
    pub loaded: Vec<Loaded>,
    /// A frame a component's own command asked to open, as `(id, face)`.
    /// Drained by [`Session::take_pending_frame`].
    pending_frame: Option<(String, String)>,
    /// Last rendered status-bar cells, highest priority first.
    segments: Vec<Segment>,
    /// When they were last rendered, in milliseconds since an arbitrary epoch
    /// the caller supplies. `None` until the first render.
    segments_rendered_ms: Option<u64>,
    /// Components held back, each with the decision that held it.
    pub held: Vec<(WasmComponent, Decision)>,
    /// Non-fatal complaints, including load failures.
    pub warnings: Vec<String>,
}

impl Registry {
    /// Builds the session registry: hashes each component, asks `trust` about
    /// it, and loads what is approved through `host`.
    ///
    /// `host` is `&mut dyn` so this compiles and behaves identically with the
    /// `plugins` feature off — every load fails as unsupported, every component
    /// lands in `warnings`, and discovery and trust still ran. That is the
    /// point: a build without the runtime can still tell a user what they have
    /// installed and why none of it is running.
    #[must_use]
    pub fn build(
        set: &WasmSet,
        trust: &TrustStore,
        project: &Path,
        host: &mut dyn crate::wasmhost::WasmHost,
    ) -> Self {
        let mut out = Self {
            warnings: set.warnings.clone(),
            ..Self::default()
        };
        for component in &set.components {
            let Some(sha) = module_sha256(&component.path) else {
                out.warnings.push(format!(
                    "wasm component '{}': cannot hash {}",
                    component.manifest.id,
                    component.path.display()
                ));
                continue;
            };
            let decision = judge(component, &sha, project, trust);
            if !decision.is_trusted() {
                out.held.push((component.clone(), decision));
                continue;
            }
            let Ok(bytes) = std::fs::read(&component.path) else {
                out.warnings.push(format!(
                    "wasm component '{}': cannot read {}",
                    component.manifest.id,
                    component.path.display()
                ));
                continue;
            };
            // Exactly the capabilities this component declared — which, by the
            // time it reaches here, the user has approved. The runtime never
            // sees the manifest, so the grant set is the whole of what it
            // knows about what this component may reach.
            let granted: Vec<&str> = component
                .manifest
                .capabilities
                .iter()
                .map(|c| c.label())
                .collect();
            if let Err(e) = host.load(&component.manifest.id, &bytes, &granted) {
                out.warnings
                    .push(format!("wasm component '{}': {e}", component.manifest.id));
                continue;
            }
            // Claiming a surface you did not implement is a load-time error,
            // not a mystery at the first call: a `command` component whose
            // exports are missing would otherwise sit in the slash menu and
            // fail only once a user picked it.
            let missing = missing_exports(component, host);
            if !missing.is_empty() {
                out.warnings.push(format!(
                    "wasm component '{}' claims surfaces it does not implement (missing {})",
                    component.manifest.id,
                    missing.join(", ")
                ));
                continue;
            }
            let tools = if component.manifest.surfaces.contains(&Surface::Tool) {
                match host.call(&component.manifest.id, "tool_specs", b"") {
                    Ok(bytes) => {
                        parse_tool_specs(&component.manifest.id, &bytes).unwrap_or_else(|| {
                            out.warnings.push(format!(
                                "wasm component '{}': tool_specs returned unreadable JSON",
                                component.manifest.id
                            ));
                            Vec::new()
                        })
                    }
                    Err(e) => {
                        out.warnings
                            .push(format!("wasm component '{}': {e}", component.manifest.id));
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };
            let commands = if component.manifest.surfaces.contains(&Surface::Command) {
                match host.call(&component.manifest.id, "command_specs", b"") {
                    Ok(bytes) => parse_command_specs(&bytes).map_or_else(
                        || {
                            out.warnings.push(format!(
                                "wasm component '{}': command_specs returned unreadable JSON",
                                component.manifest.id
                            ));
                            Vec::new()
                        },
                        |mut specs| {
                            // Stamped here rather than in the parser: the alias
                            // is `<plugin>:<name>`, and only the load site knows
                            // which plugin contributed this component.
                            for spec in &mut specs {
                                spec.alias = format!("{}:{}", component.plugin, spec.name);
                            }
                            specs
                        },
                    ),
                    Err(e) => {
                        out.warnings
                            .push(format!("wasm component '{}': {e}", component.manifest.id));
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };
            out.loaded.push(Loaded {
                component: component.clone(),
                strikes: 0,
                tools,
                commands,
            });
        }
        out
    }

    /// Takes a frame-open request a component's command left behind.
    pub fn take_pending_frame(&mut self) -> Option<(String, String)> {
        self.pending_frame.take()
    }

    /// A registry holding exactly `loaded` and nothing else.
    ///
    /// For callers that build a registry directly rather than admitting one —
    /// tests, and anything that wants to reason about a fixed component set.
    /// The segment cache stays private because it must stay consistent with
    /// its own throttle, so it is not something a caller may seed.
    #[must_use]
    pub fn with_loaded(loaded: Vec<Loaded>) -> Self {
        Self {
            loaded,
            ..Self::default()
        }
    }

    /// The status-bar cells as last rendered, highest priority first.
    ///
    /// Cheap and side-effect free: the bar reads this on every repaint, and
    /// nothing here calls a guest.
    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Re-renders the status cells if enough time has passed, returning
    /// whether anything was rendered.
    ///
    /// Self-throttling on purpose: callers should call this wherever they
    /// naturally can — a turn boundary, an idle tick — without having to own
    /// the decision about how often a guest may run. `now_ms` is supplied
    /// rather than read here so the throttle is testable without sleeping.
    pub fn refresh_segments(
        &mut self,
        host: &mut dyn crate::wasmhost::WasmHost,
        status_json: &str,
        now_ms: u64,
    ) -> bool {
        if let Some(last) = self.segments_rendered_ms
            && now_ms.saturating_sub(last) < SEGMENT_MIN_INTERVAL_MS
        {
            return false;
        }
        let ids: Vec<String> = self
            .loaded
            .iter()
            .filter(|l| {
                l.strikes < STRIKE_LIMIT
                    && l.component.manifest.surfaces.contains(&Surface::Segment)
            })
            .map(|l| l.component.manifest.id.clone())
            .collect();
        self.segments_rendered_ms = Some(now_ms);
        if ids.is_empty() {
            return false;
        }
        let mut rendered = Vec::new();
        for id in ids {
            match host.call(&id, "segment_render", status_json.as_bytes()) {
                // An unreadable reply drops the cell for this round rather
                // than striking the component: a segment that says nothing is
                // its ordinary state, and the bar must never be a place where
                // a component gets disabled for being quiet.
                Ok(bytes) => {
                    if let Some(seg) = parse_segment(&id, &bytes).filter(|s| !s.text.is_empty()) {
                        rendered.push(seg);
                    }
                }
                Err(e) => {
                    // A trap *is* worth a strike: it means the guest broke,
                    // and it will break again on the next repaint.
                    let disabled = self.strike(&id);
                    self.warnings.push(if disabled {
                        format!(
                            "{id}: {e} (disabled for this session after {STRIKE_LIMIT} failures)"
                        )
                    } else {
                        format!("{id}: {e}")
                    });
                }
            }
        }
        rendered.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.component.cmp(&b.component))
        });
        self.segments = rendered;
        true
    }

    /// Every model-facing tool contributed this session, under the name the
    /// model sees.
    #[must_use]
    pub fn tools(&self) -> Vec<&WasmTool> {
        self.loaded
            .iter()
            .filter(|l| l.strikes < STRIKE_LIMIT)
            .flat_map(|l| l.tools.iter())
            .collect()
    }

    /// Resolves collisions among contributed tool names against `taken` — the
    /// names already claimed by built-ins and MCP.
    ///
    /// A contested tool falls back to `wasm__<component>__<tool>`; an
    /// uncontested one keeps its bare name. Built-ins always win, so no
    /// component can quietly replace `bash`, and two components claiming one
    /// name both get qualified rather than one silently losing.
    pub fn resolve_tool_names(&mut self, taken: &[String]) -> Vec<String> {
        let mut warnings = Vec::new();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for l in &self.loaded {
            for t in &l.tools {
                *counts.entry(t.name.clone()).or_default() += 1;
            }
        }
        for l in &mut self.loaded {
            for t in &mut l.tools {
                let contested =
                    taken.contains(&t.name) || counts.get(&t.name).is_some_and(|n| *n > 1);
                if contested {
                    let short = t.component.rsplit('.').next().unwrap_or(&t.component);
                    t.exposed = format!("wasm__{short}__{}", t.name);
                    warnings.push(format!(
                        "wasm tool '{}' from '{}' is contested; exposed as '{}'",
                        t.name, t.component, t.exposed
                    ));
                }
            }
        }
        warnings
    }

    /// Runs a model-facing tool by the name the model used.
    ///
    /// # Errors
    /// Returns a message when no live component owns `exposed`, or when the
    /// call traps — which also costs the component a strike.
    pub fn run_tool(
        &mut self,
        host: &mut dyn crate::wasmhost::WasmHost,
        exposed: &str,
        args_json: &str,
    ) -> Result<String, String> {
        let (id, name) = self
            .tools()
            .into_iter()
            .find(|t| t.exposed == exposed)
            .map(|t| (t.component.clone(), t.name.clone()))
            .ok_or_else(|| format!("no wasm tool '{exposed}'"))?;
        let payload = format!("{{\"name\": {}, \"args\": {args_json}}}", json_str(&name));
        match host.call(&id, "tool_call", payload.as_bytes()) {
            Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
            Err(e) => {
                let disabled = self.strike(&id);
                Err(if disabled {
                    format!("{id}: {e} (disabled for this session after {STRIKE_LIMIT} failures)")
                } else {
                    format!("{id}: {e}")
                })
            }
        }
    }

    /// Every slash command contributed this session, as `(component id, spec)`.
    #[must_use]
    pub fn commands(&self) -> Vec<(&str, &CommandSpec)> {
        self.loaded
            .iter()
            .filter(|l| l.strikes < STRIKE_LIMIT)
            .flat_map(|l| {
                l.commands
                    .iter()
                    .map(move |c| (l.component.manifest.id.as_str(), c))
            })
            .collect()
    }

    /// Runs `name` against the component that contributes it.
    ///
    /// A trap costs the component a strike and returns the message for the
    /// user; it never propagates as a session error. That is the whole
    /// containment claim, applied at the one place a user can trigger a call
    /// on purpose.
    ///
    /// # Errors
    /// Returns a human-readable message when no live component owns `name`, or
    /// when the call traps.
    pub fn run_command(
        &mut self,
        host: &mut dyn crate::wasmhost::WasmHost,
        name: &str,
        args: &str,
    ) -> Result<CmdOutput, String> {
        let name = name.trim_start_matches('/');
        // Either spelling resolves: the alias is always addressable, the bare
        // name only when the menu offered it. Matching both here rather than
        // making the caller choose keeps `/arcade:matrix` and `/matrix`
        // indistinguishable once one of them has been typed.
        let (id, name) = self
            .commands()
            .into_iter()
            .find(|(_, c)| c.alias == name || c.name == name)
            .map(|(id, c)| (id.to_string(), c.name.clone()))
            .ok_or_else(|| format!("no wasm command '{name}'"))?;
        let payload = format!(
            "{{\"name\": {}, \"args\": {}}}",
            json_str(&name),
            json_str(args)
        );
        match host.call(&id, "command_run", payload.as_bytes()) {
            Ok(bytes) => {
                let mut out = parse_cmd_output(&bytes);
                // The component may only open *its own* frame, so the id is
                // taken from who was called rather than from the reply.
                if let Some(face) = &out.open {
                    self.pending_frame = Some((id.clone(), face.clone()));
                }
                // Anything the component printed through the `print` capability
                // happened *during* the call, so it precedes whatever the reply
                // asks to print. Draining here rather than in the UI keeps both
                // front ends from having to remember to do it.
                let mut printed = host.drain_printed();
                printed.append(&mut out.print);
                out.print = printed;
                Ok(out)
            }
            Err(e) => {
                let disabled = self.strike(&id);
                Err(if disabled {
                    format!("{id}: {e} (disabled for this session after {STRIKE_LIMIT} failures)")
                } else {
                    format!("{id}: {e}")
                })
            }
        }
    }

    /// Dispatches `event` to every live subscriber, in load order.
    ///
    /// Transform replacements chain: the second subscriber sees what the first
    /// produced, which is what makes a redactor and a summarizer composable
    /// rather than mutually exclusive. A veto stops the chain — once an action
    /// is refused, asking the rest to weigh in is asking about something that
    /// will not happen.
    ///
    /// A trap costs a strike and becomes a warning. It never propagates: an
    /// observer that breaks must not take the turn with it.
    pub fn dispatch(
        &mut self,
        host: &mut dyn crate::wasmhost::WasmHost,
        event: &crate::wasmevents::Event<'_>,
    ) -> crate::wasmevents::Outcome {
        use crate::wasmevents::Reply;

        let mut outcome = crate::wasmevents::Outcome::default();
        let subscribers: Vec<String> = self
            .loaded
            .iter()
            .filter(|l| {
                l.strikes < STRIKE_LIMIT && l.component.manifest.events.contains(&event.kind)
            })
            .map(|l| l.component.manifest.id.clone())
            .collect();
        if subscribers.is_empty() {
            return outcome;
        }

        let class = event.kind.class();
        let field = crate::wasmevents::Event::transform_field(event.kind);
        // The payload is rebuilt per subscriber only when a replacement has
        // actually changed it; otherwise every subscriber sees the same bytes.
        let mut payload = event.to_json();
        for id in subscribers {
            match host.call(&id, "on_event", payload.as_bytes()) {
                Ok(bytes) => {
                    let reply = Reply::parse(&bytes).constrained_to(class);
                    if let Some(text) = reply.replace
                        && let Some(field) = field
                    {
                        let mut next = event.clone();
                        for (key, value) in &mut next.fields {
                            if *key == field {
                                value.clone_from(&text);
                            }
                        }
                        payload = next.to_json();
                        outcome.replaced = Some(text);
                    }
                    if let Some(reason) = reply.block {
                        outcome.blocked = Some((id, reason));
                        break;
                    }
                }
                Err(e) => {
                    let disabled = self.strike(&id);
                    outcome.warnings.push(if disabled {
                        format!(
                            "{id}: {e} (disabled for this session after {STRIKE_LIMIT} failures)"
                        )
                    } else {
                        format!("{id}: {e}")
                    });
                }
            }
        }
        // Whatever the handlers printed, collected once at the end regardless
        // of class: the sink is shared, so draining per subscriber would only
        // interleave the same lines differently, and a notify subscriber is
        // exactly the kind that has nothing to say *except* by printing.
        outcome.printed = host.drain_printed();
        outcome
    }

    /// Records a failure against `id`, returning whether that strike disabled
    /// it. Disabling is sticky for the session; nothing resets it but a
    /// restart, because a component that failed three times has earned the
    /// doubt.
    pub fn strike(&mut self, id: &str) -> bool {
        let Some(l) = self
            .loaded
            .iter_mut()
            .find(|l| l.component.manifest.id == id)
        else {
            return false;
        };
        l.strikes = l.strikes.saturating_add(1);
        l.strikes >= STRIKE_LIMIT
    }

    /// Whether `id` is still allowed to run.
    #[must_use]
    pub fn is_live(&self, id: &str) -> bool {
        self.loaded
            .iter()
            .any(|l| l.component.manifest.id == id && l.strikes < STRIKE_LIMIT)
    }
}

/// Renders the held-components section of `/plugins`: what was found, why it
/// is not running, and the exact command that would approve it.
///
/// Held components are not warnings. "We did not run this" is a different
/// message from "this is broken", and collapsing the two is how a security
/// decision gets scrolled past.
#[must_use]
pub fn render_held(registry: &Registry) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    if registry.held.is_empty() {
        return out;
    }
    out.push_str("\nwasm components awaiting approval:\n");
    for (component, decision) in &registry.held {
        let _ = writeln!(
            out,
            "  {} ({}) — {}",
            component.manifest.id,
            component.origin.label(),
            decision.reason()
        );
        let caps: Vec<&str> = component
            .manifest
            .capabilities
            .iter()
            .map(|c| c.label())
            .collect();
        let _ = write!(out, "    wants: {}", caps.join(", "));
        if component.is_privileged() {
            out.push_str("  ⚠ this is not a sandboxed component");
        }
        let _ = writeln!(
            out,
            "\n    approve with: /plugins trust {}",
            component.manifest.id
        );
    }
    out
}

/// Renders the WASM section of the `/plugins` listing.
///
/// Surfaces and capabilities are both shown, and a component asking for
/// anything that undoes the sandbox is marked: a listing that renders `exec`
/// the way it renders `sound` is lying by omission. Trust verdicts are not
/// shown here — they need the session's project directory, and the listing is
/// a pure function of the plugin set.
#[must_use]
pub fn render_components(set: &WasmSet) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    if set.components.is_empty() && set.warnings.is_empty() {
        return out;
    }
    out.push_str("\nwasm components:\n");
    for c in &set.components {
        let mut surfaces: Vec<String> = c
            .manifest
            .surfaces
            .iter()
            .map(|s| s.label().to_string())
            .collect();
        // A frame's kind is the thing a user actually wants to know about it —
        // whether it can appear on its own — so it is shown as part of the
        // surface rather than buried.
        if c.manifest.surfaces.contains(&Surface::Frame) {
            for entry in &mut surfaces {
                if entry == "frame" {
                    *entry = format!("frame: {}", c.manifest.kind.label());
                }
            }
        }
        let _ = write!(
            out,
            "  {} ({}) [{}]",
            c.manifest.id,
            c.plugin,
            surfaces.join(", ")
        );
        if c.is_privileged() {
            out.push_str(" ⚠ not sandboxed");
        }
        out.push('\n');
        let caps: Vec<&str> = c.manifest.capabilities.iter().map(|c| c.label()).collect();
        let _ = writeln!(out, "    capabilities: {}", caps.join(", "));
    }
    for w in &set.warnings {
        let _ = writeln!(out, "  warning: {w}");
    }
    out
}

/// Exports a component's claimed surfaces require but the module does not have.
///
/// Only the surfaces implemented so far are checked. A surface plank cannot yet
/// drive is not a reason to refuse a module that is otherwise sound — the
/// author may be ahead of the host, which the payload-evolution rule expects.
fn missing_exports(
    component: &WasmComponent,
    host: &dyn crate::wasmhost::WasmHost,
) -> Vec<&'static str> {
    let id = &component.manifest.id;
    let mut missing = Vec::new();
    if component.manifest.surfaces.contains(&Surface::Command) {
        for export in ["command_specs", "command_run"] {
            if !host.has_export(id, export) {
                missing.push(export);
            }
        }
    }
    if component.manifest.surfaces.contains(&Surface::Frame) {
        // `frame_mouse` is deliberately absent from this list: a component may
        // implement it, and one that does not simply never sees a mouse event.
        // Requiring it would refuse a perfectly good keyboard-only game.
        // `frame_mouse` is optional (see above), and a component satisfies the
        // step requirement with either the packed export or the text one — the
        // text form exists precisely so a first plugin need not pack buffers,
        // and demanding the packed export alongside it would defeat that.
        if !host.has_export(id, "frame_step") && !host.has_export(id, "frame_step_text") {
            missing.push("frame_step");
        }
        for export in ["frame_open", "frame_key", "frame_close"] {
            if !host.has_export(id, export) {
                missing.push(export);
            }
        }
    }
    if component.manifest.surfaces.contains(&Surface::Segment)
        && !host.has_export(id, "segment_render")
    {
        missing.push("segment_render");
    }
    if component.manifest.surfaces.contains(&Surface::Tool) {
        for export in ["tool_specs", "tool_call"] {
            if !host.has_export(id, export) {
                missing.push(export);
            }
        }
    }
    // `observer` owns nothing but its handler, so a component claiming it
    // without `on_event` claims nothing at all. A component of any surface that
    // *subscribes* needs the same export, which is why this is keyed on the
    // subscription list and not only on the surface.
    let observes = component.manifest.surfaces.contains(&Surface::Observer)
        || !component.manifest.events.is_empty();
    if observes && !host.has_export(id, "on_event") {
        missing.push("on_event");
    }
    missing
}

/// Reads a component's signature and judges it against the trust store.
///
/// Split from `build` only to keep that function under the length lint; the
/// pairing is the point, since a decision made without looking for a signature
/// would never let a signed update through.
fn judge(component: &WasmComponent, sha256: &str, project: &Path, trust: &TrustStore) -> Decision {
    let sig = signature_status(&component.path, trust);
    trust.evaluate(component, sha256, project, &sig)
}

/// Reads `<module>.minisig` beside a module and verifies it against the
/// publisher keys `trust` holds.
///
/// The signature file sits next to the module and is named after it, the way
/// minisign writes it by default, so a publisher signs with
/// `minisign -S -m wasm/thing.wasm` and ships the `.minisig` alongside — no
/// packaging step that could get the pairing wrong.
///
/// Absence is [`SigStatus::Absent`], not an error: unsigned is the ordinary
/// case. A signature by a key the store does not hold is `UnknownKey` rather
/// than `Invalid` — it proves nothing to *this* user, but nothing is wrong.
#[must_use]
pub fn signature_status(module: &Path, trust: &TrustStore) -> SigStatus {
    let mut sig_path = module.as_os_str().to_os_string();
    sig_path.push(".minisig");
    let Ok(sig_text) = std::fs::read_to_string(PathBuf::from(sig_path)) else {
        return SigStatus::Absent;
    };
    let sig = match crate::wasmsig::Signature::parse(&sig_text) {
        Ok(sig) => sig,
        // A `.minisig` that will not even parse is malformed rather than
        // forged, but it is still a signature the publisher meant to be
        // checked, so it fails rather than being ignored.
        Err(e) => return SigStatus::Invalid(e.to_string()),
    };
    let Ok(bytes) = std::fs::read(module) else {
        return SigStatus::Invalid("cannot read the module to verify it".to_string());
    };
    // Only keys the user has accepted count. Trying every key would let an
    // attacker who can write the trust file also choose the verdict.
    for (id, encoded) in trust.publishers() {
        let Ok(key) = crate::wasmsig::PublicKey::parse(encoded) else {
            continue;
        };
        if key.key_id != sig.key_id() {
            continue;
        }
        return match crate::wasmsig::verify(&bytes, &sig, &key) {
            Ok(()) => SigStatus::Valid(id.clone()),
            Err(e) => SigStatus::Invalid(e.to_string()),
        };
    }
    SigStatus::UnknownKey(crate::wasmsig::key_id_display(sig.key_id()))
}

/// SHA-256 of a module's bytes, via the same `shasum` path the image cache
/// uses — no crypto dependency, and a subprocess per component at load time is
/// invisible next to compiling one.
///
/// Public because the trust store's caller needs the same hash to record an
/// approval, and a second implementation is a second thing that can disagree.
#[must_use]
pub fn module_sha256(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    crate::imagepaste::sha256_hex(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-wasmreg-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Builds a plugin directory with a manifest and a module file.
    fn plugin_dir(root: &Path, name: &str, manifest: &str, modules: &[&str]) -> Plugin {
        let dir = root.join(name);
        std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
        std::fs::create_dir_all(dir.join("wasm")).unwrap();
        std::fs::write(dir.join(".plank-plugin").join("plugin.json"), manifest).unwrap();
        for m in modules {
            std::fs::write(dir.join("wasm").join(m), b"\0asm fake").unwrap();
        }
        crate::plugins::load_plugin(&dir, Origin::UserScan).expect("a plugin")
    }

    fn set_of(plugins: Vec<Plugin>) -> PluginSet {
        PluginSet {
            plugins,
            ..PluginSet::default()
        }
    }

    const FULL: &str = r#"{
      "name": "demo",
      "wasm": [{
        "id": "dev.plank.demo",
        "module": "demo.wasm",
        "surfaces": ["frame", "command"],
        "capabilities": ["state", "sound"]
      }]
    }"#;

    /// The `Outcome` reply is read the same way for keys and mouse events, and
    /// an unreadable one keeps the window open.
    #[test]
    fn a_frame_outcome_is_read_the_same_for_every_input() {
        assert_eq!(parse_frame_outcome(br"{}"), FrameOutcome::Stay);
        assert_eq!(
            parse_frame_outcome(br#"{"stay": true}"#),
            FrameOutcome::Stay
        );
        assert_eq!(
            parse_frame_outcome(br#"{"close": "final score 12"}"#),
            FrameOutcome::Close(Some("final score 12".to_string()))
        );
        assert_eq!(
            parse_frame_outcome(br#"{"close": true}"#),
            FrameOutcome::Close(None)
        );
        assert_eq!(
            parse_frame_outcome(br#"{"close": {}}"#),
            FrameOutcome::Close(None)
        );
        // Garbage keeps the window: a component that fumbles one reply should
        // not have its window yanked out from under the user.
        assert_eq!(parse_frame_outcome(b"not json at all"), FrameOutcome::Stay);
        assert_eq!(parse_frame_outcome(b""), FrameOutcome::Stay);
    }

    /// The one thing a signature buys: an update from the publisher this
    /// component was approved under does not re-prompt.
    #[test]
    fn a_signed_update_from_the_same_publisher_is_quiet() {
        let mut trust = TrustStore::ephemeral();
        let c = component(Origin::UserScan, vec![]);
        let project = Path::new("/repo");
        let signed = SigStatus::Valid("E171F5A2933C03AE".to_string());

        // First install still asks, signature or not: the design buys quiet
        // updates, not unattended first contact.
        assert_eq!(
            trust.evaluate(&c, "hash-a", project, &signed),
            Decision::Unknown
        );
        trust.approve(&c, "hash-a", project, &signed).unwrap();
        assert_eq!(
            trust.entry(&c.manifest.id).unwrap().publisher.as_deref(),
            Some("E171F5A2933C03AE"),
            "the approval must record who signed it"
        );

        // New bytes, same publisher: trusted without asking.
        assert_eq!(
            trust.evaluate(&c, "hash-b", project, &signed),
            Decision::Trusted
        );
        // New bytes, *different* publisher: this is the case the feature exists
        // to catch, and it must fall back to asking.
        let other = SigStatus::Valid("0000000000000000".to_string());
        assert_eq!(
            trust.evaluate(&c, "hash-b", project, &other),
            Decision::Changed
        );
        // New bytes, no signature at all: also asks. A previously signed
        // component going unsigned is not a quiet update.
        assert_eq!(
            trust.evaluate(&c, "hash-b", project, &SigStatus::Absent),
            Decision::Changed
        );
    }

    /// A valid signature does not authorise new capabilities. Provenance says
    /// who built it, not what it may do.
    #[test]
    fn a_signature_never_widens_capabilities() {
        let mut trust = TrustStore::ephemeral();
        let project = Path::new("/repo");
        let signed = SigStatus::Valid("E171F5A2933C03AE".to_string());
        let plain = component(Origin::UserScan, vec![]);
        trust.approve(&plain, "hash-a", project, &signed).unwrap();

        // Same publisher, new bytes, and now it wants `exec`.
        let greedy = component(Origin::UserScan, vec![Capability::Exec]);
        assert_eq!(
            trust.evaluate(&greedy, "hash-b", project, &signed),
            Decision::Widened(vec![Capability::Exec]),
            "a signed update that widens must still prompt"
        );
    }

    /// A signature that does not verify is refused rather than prompted, and
    /// before every other verdict: it is the only one where the machine is
    /// certain something is wrong.
    #[test]
    fn a_bad_signature_is_refused_not_prompted() {
        let trust = TrustStore::ephemeral();
        let c = component(Origin::UserScan, vec![]);
        let bad = SigStatus::Invalid("signature does not match the artifact".to_string());
        let decision = trust.evaluate(&c, "hash-a", Path::new("/repo"), &bad);
        assert!(
            matches!(decision, Decision::BadSignature(_)),
            "unknown-but-signed-badly must not present as merely unknown: {decision:?}"
        );
        assert!(!decision.is_trusted());
        assert!(
            decision.reason().contains("did not verify"),
            "{}",
            decision.reason()
        );
    }

    /// An unknown signing key proves nothing but breaks nothing: trust falls
    /// back to the hash, exactly as for an unsigned plugin.
    #[test]
    fn an_unknown_signing_key_falls_back_to_the_hash() {
        let mut trust = TrustStore::ephemeral();
        let c = component(Origin::UserScan, vec![]);
        let project = Path::new("/repo");
        let unknown = SigStatus::UnknownKey("DEADBEEFDEADBEEF".to_string());
        assert_eq!(
            trust.evaluate(&c, "hash-a", project, &unknown),
            Decision::Unknown
        );
        trust.approve(&c, "hash-a", project, &unknown).unwrap();
        // No publisher recorded, so a later update cannot be waved through by
        // claiming to come from one.
        assert!(trust.entry(&c.manifest.id).unwrap().publisher.is_none());
        assert_eq!(
            trust.evaluate(&c, "hash-a", project, &unknown),
            Decision::Trusted
        );
        assert_eq!(
            trust.evaluate(&c, "hash-b", project, &unknown),
            Decision::Changed
        );
    }

    /// `signature_status` against real files on disk: a real `.minisig` from
    /// minisign 0.12, beside a real module, resolved through a real store.
    ///
    /// The unit tests above pass a `SigStatus` in directly, so this is the only
    /// thing covering the part that reads the filesystem — the file naming, and
    /// the rule that only keys the user accepted are tried.
    #[test]
    fn signature_status_reads_a_real_minisig_from_disk() {
        let root = std::env::temp_dir().join(format!("plank-sigstatus-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let module = root.join("thing.wasm");
        std::fs::write(
            &module,
            include_bytes!("../tests/fixtures/minisign/artifact.wasm"),
        )
        .unwrap();

        // No signature file: absent, which is the ordinary case.
        let empty = TrustStore::ephemeral();
        assert_eq!(signature_status(&module, &empty), SigStatus::Absent);

        // The signature minisign wrote, named the way minisign names it.
        std::fs::write(
            root.join("thing.wasm.minisig"),
            include_str!("../tests/fixtures/minisign/artifact.wasm.minisig"),
        )
        .unwrap();
        // Still unknown until the user has accepted the key: an untrusted
        // signature must not verify to anything, or writing a key into the file
        // would be enough to choose the verdict.
        assert!(matches!(
            signature_status(&module, &empty),
            SigStatus::UnknownKey(id) if id == "E171F5A2933C03AE"
        ));

        let key =
            crate::wasmsig::PublicKey::parse(include_str!("../tests/fixtures/minisign/pub.key"))
                .unwrap();
        let encoded = include_str!("../tests/fixtures/minisign/pub.key")
            .lines()
            .next_back()
            .unwrap();
        let mut trust = TrustStore::ephemeral();
        trust.add_publisher(&key, encoded).unwrap();
        assert_eq!(
            signature_status(&module, &trust),
            SigStatus::Valid("E171F5A2933C03AE".to_string())
        );

        // Change one byte of the module and the same signature must fail.
        std::fs::write(&module, b"tampered\n").unwrap();
        assert!(
            matches!(signature_status(&module, &trust), SigStatus::Invalid(_)),
            "a modified module with a good signature file must be Invalid"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Publisher keys survive a round trip through the file, beside the
    /// component entries rather than in a second file.
    #[test]
    fn publishers_persist_alongside_the_entries() {
        let home = std::env::temp_dir().join(format!("plank-pub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let key =
            crate::wasmsig::PublicKey::parse(include_str!("../tests/fixtures/minisign/pub.key"))
                .expect("fixture key parses");
        let encoded = include_str!("../tests/fixtures/minisign/pub.key")
            .lines()
            .next_back()
            .unwrap()
            .to_string();

        let mut trust = TrustStore::load(&home);
        trust.add_publisher(&key, &encoded).expect("persisted");
        let c = component(Origin::UserScan, vec![]);
        trust
            .approve(
                &c,
                "hash-a",
                Path::new("/repo"),
                &SigStatus::Valid(key.key_id_hex()),
            )
            .expect("approved");

        let reloaded = TrustStore::load(&home);
        assert_eq!(
            reloaded.publishers().get(&key.key_id_hex()),
            Some(&encoded),
            "the publisher key did not survive: {:?}",
            reloaded.publishers()
        );
        assert_eq!(
            reloaded.entry(&c.manifest.id).unwrap().publisher.as_deref(),
            Some(key.key_id_hex().as_str())
        );
        // And the reserved key is not mistaken for a component.
        assert!(reloaded.entry("@publishers").is_none());
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A grant with no host function behind it is named at load time.
    ///
    /// Silence here means the user approves a capability that reaches nothing:
    /// they have consented to something and nothing tells them it was nothing.
    /// The two reasons are distinguished because they mean different things to
    /// an author — one is "not yet", the other is "not ever, by design".
    #[test]
    fn an_unwired_capability_grant_is_warned_about() {
        let text = r#"{
          "name": "demo",
          "wasm": [{
            "id": "dev.plank.demo",
            "module": "demo.wasm",
            "surfaces": ["tool"],
            "capabilities": ["state", "notify", "exec"]
          }]
        }"#;
        let (manifests, warnings) = parse_manifest_section(text);
        assert_eq!(manifests.len(), 1, "the component still loads");
        // Granted anyway: refusing here would change what the approval prompt
        // shows, and the manifest is the author's statement of intent.
        assert!(manifests[0].capabilities.contains(&Capability::Notify));

        let joined = warnings.join("\n");
        assert!(
            joined.contains("'notify' is declared but not implemented yet"),
            "{joined}"
        );
        assert!(
            joined.contains("'exec' is deliberately unimplemented"),
            "{joined}"
        );
        // `state` is wired, so it must not be warned about — a warning on every
        // grant would be noise nobody reads.
        assert!(!joined.contains("'state'"), "{joined}");
    }

    /// Every capability is either wired or has a reason. A new variant added
    /// without a host function must show up as unwired rather than defaulting
    /// to "fine".
    #[test]
    fn wired_capabilities_are_exactly_the_ones_with_host_functions() {
        for c in [
            Capability::Log,
            Capability::Print,
            Capability::State,
            Capability::Sound,
        ] {
            assert!(c.is_wired(), "{c:?}");
        }
        for c in [
            Capability::Notify,
            Capability::Agent,
            Capability::Session,
            Capability::Fs,
            Capability::Net,
            Capability::Exec,
        ] {
            assert!(!c.is_wired(), "{c:?}");
        }
    }

    /// The manifests plank actually ships must parse with this parser.
    ///
    /// They are hand-written JSON in `guests/*/plugin.json`, assembled into a
    /// plugin directory by `guests/package.sh` and published by the release
    /// workflow. Nothing else checks them: a typo would ship a plugin whose
    /// component silently fails to appear, which is the least debuggable
    /// failure this system has.
    #[test]
    fn the_shipped_guest_manifests_parse_clean() {
        for (guest, id, module) in [
            (
                "screensavers",
                "dev.plank.screensavers",
                "screensavers.wasm",
            ),
            ("arcades", "dev.plank.arcades", "arcades.wasm"),
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("guests")
                .join(guest)
                .join("plugin.json");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let (manifests, warnings) = parse_manifest_section(&text);
            assert!(warnings.is_empty(), "{guest}: {warnings:?}");
            assert_eq!(manifests.len(), 1, "{guest}");
            let m = &manifests[0];
            assert_eq!(m.id, id, "{guest}");
            // `module` is what the host opens, and package.sh copies the built
            // artifact to exactly this name — a mismatch is a plugin that
            // loads and then cannot find its own code.
            assert_eq!(m.module, module, "{guest}");
            assert!(m.surfaces.contains(&Surface::Frame), "{guest}");
            assert!(m.surfaces.contains(&Surface::Command), "{guest}");
            // Both guests import only `plank_abi`, so they ask for nothing.
            // A grant added here without a matching host call would prompt the
            // user to approve a capability that reaches no code.
            assert_eq!(
                m.capabilities,
                vec![Capability::Log],
                "{guest} should grant nothing beyond the implicit log"
            );
        }
    }

    #[test]
    fn a_manifest_section_parses_surfaces_and_capabilities() {
        let (manifests, warnings) = parse_manifest_section(FULL);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(manifests.len(), 1);
        let m = &manifests[0];
        assert_eq!(m.id, "dev.plank.demo");
        assert_eq!(
            m.surfaces,
            vec![Surface::Frame, Surface::Command].tap_sorted()
        );
        // `log` is always present without being asked for.
        assert!(m.capabilities.contains(&Capability::Log));
        assert!(m.capabilities.contains(&Capability::State));
        assert_eq!(m.abi, crate::wasmhost::ABI_VERSION, "abi defaults");
    }

    /// Every malformed shape is skipped and *named*. A component that silently
    /// vanishes presents as "my plugin does nothing", which is the least
    /// debuggable failure this system can have.
    #[test]
    fn malformed_entries_are_skipped_with_a_reason() {
        let text = r#"{"wasm": [
          {"module": "a.wasm", "surfaces": ["frame"]},
          {"id": "b", "surfaces": ["frame"]},
          {"id": "c", "module": "../escape.wasm", "surfaces": ["frame"]},
          {"id": "d", "module": "d.wasm", "surfaces": []},
          {"id": "e", "module": "e.wasm", "surfaces": ["frame", "panel"]}
        ]}"#;
        let (manifests, warnings) = parse_manifest_section(text);
        // Only 'e' survives — with a warning about `panel`, which was cut.
        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].id, "e");
        let joined = warnings.join("\n");
        for expected in [
            "no \"id\"",
            "no \"module\"",
            "bare file name",
            "no surfaces",
        ] {
            assert!(joined.contains(expected), "missing {expected}: {joined}");
        }
        assert!(joined.contains("unknown surface 'panel'"), "{joined}");
    }

    /// A manifest naming a module that is not there is the commonest packaging
    /// mistake, and it must not present as "no components".
    #[test]
    fn a_missing_module_file_is_reported_not_silently_dropped() {
        let root = temp_dir("missing-module");
        let p = plugin_dir(&root, "demo", FULL, &[]);
        let set = discover(&set_of(vec![p]));
        assert!(set.components.is_empty());
        assert!(
            set.warnings.iter().any(|w| w.contains("missing module")),
            "{:?}",
            set.warnings
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discovery_finds_a_component_and_records_its_origin() {
        let root = temp_dir("discover");
        let p = plugin_dir(&root, "demo", FULL, &["demo.wasm"]);
        let set = discover(&set_of(vec![p]));
        assert_eq!(set.components.len(), 1, "{:?}", set.warnings);
        let c = &set.components[0];
        assert_eq!(c.manifest.id, "dev.plank.demo");
        assert_eq!(c.plugin, "demo");
        assert_eq!(c.origin, Origin::UserScan);
        assert!(!c.is_privileged(), "state+sound is not privileged");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Two plugins claiming one id: first wins, and the loser is named. Same
    /// rule the plugin loader uses for every other contribution kind.
    #[test]
    fn a_duplicate_id_keeps_the_first_and_warns() {
        let root = temp_dir("dup-id");
        // Distinct plugin *names*, not just distinct directories: the name
        // comes from the manifest, and the warning has to name the plugins the
        // user would recognize.
        let a = plugin_dir(
            &root,
            "alpha",
            &FULL.replace("\"demo\"", "\"alpha\""),
            &["demo.wasm"],
        );
        let b = plugin_dir(
            &root,
            "beta",
            &FULL.replace("\"demo\"", "\"beta\""),
            &["demo.wasm"],
        );
        let set = discover(&set_of(vec![a, b]));
        assert_eq!(set.components.len(), 1);
        assert_eq!(set.components[0].plugin, "alpha");
        assert!(
            set.warnings.iter().any(|w| w.contains("both 'alpha'")),
            "{:?}",
            set.warnings
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    fn component(origin: Origin, caps: Vec<Capability>) -> WasmComponent {
        WasmComponent {
            plugin: "demo".to_string(),
            origin,
            path: PathBuf::from("/nowhere/demo.wasm"),
            manifest: WasmManifest {
                id: "dev.plank.demo".to_string(),
                abi: 1,
                module: "demo.wasm".to_string(),
                surfaces: vec![Surface::Frame],
                capabilities: caps,
                kind: FrameKind::default(),
                frames: Vec::new(),
                veiled: false,
                min_size: (0, 0),
                events: Vec::new(),
            },
        }
    }

    /// The four ways trust can withhold a component, each distinguishable —
    /// "it didn't load" cannot tell them apart from the outside, and each has a
    /// different answer.
    #[test]
    fn trust_distinguishes_unknown_changed_widened_and_project() {
        let project = Path::new("/repo");
        let mut trust = TrustStore::ephemeral();
        let c = component(Origin::UserScan, vec![Capability::State]);

        assert_eq!(
            trust.evaluate(&c, "hash-a", project, &SigStatus::Absent),
            Decision::Unknown
        );

        trust
            .approve(&c, "hash-a", project, &SigStatus::Absent)
            .unwrap();
        assert_eq!(
            trust.evaluate(&c, "hash-a", project, &SigStatus::Absent),
            Decision::Trusted
        );

        // Same id, different bytes.
        assert_eq!(
            trust.evaluate(&c, "hash-b", project, &SigStatus::Absent),
            Decision::Changed
        );

        // Same bytes, a new capability.
        let greedy = component(Origin::UserScan, vec![Capability::State, Capability::Exec]);
        assert_eq!(
            trust.evaluate(&greedy, "hash-a", project, &SigStatus::Absent),
            Decision::Widened(vec![Capability::Exec])
        );
        assert!(greedy.is_privileged());
    }

    /// The asymmetry that will look like a bug: approving a project-local
    /// component in one repo does not approve it in the next, even though the
    /// bytes and the grants are identical.
    #[test]
    fn a_project_local_component_is_approved_per_repo() {
        let mut trust = TrustStore::ephemeral();
        let c = component(Origin::ProjectScan, vec![Capability::State]);
        trust
            .approve(&c, "hash-a", Path::new("/repo-one"), &SigStatus::Absent)
            .unwrap();

        assert_eq!(
            trust.evaluate(&c, "hash-a", Path::new("/repo-one"), &SigStatus::Absent),
            Decision::Trusted
        );
        assert_eq!(
            trust.evaluate(&c, "hash-a", Path::new("/repo-two"), &SigStatus::Absent),
            Decision::ProjectUnapproved
        );

        // A user-scanned component of the same bytes needs no per-repo answer:
        // the user named that directory themselves.
        let user = component(Origin::UserScan, vec![Capability::State]);
        assert_eq!(
            trust.evaluate(&user, "hash-a", Path::new("/repo-two"), &SigStatus::Absent),
            Decision::Trusted
        );
    }

    #[test]
    fn the_trust_store_round_trips_through_disk() {
        let home = temp_dir("trust-disk");
        let c = component(
            Origin::ProjectScan,
            vec![Capability::State, Capability::Net],
        );
        {
            let mut trust = TrustStore::load(&home);
            trust
                .approve(&c, "hash-a", Path::new("/repo"), &SigStatus::Absent)
                .unwrap();
        }
        let reloaded = TrustStore::load(&home);
        let entry = reloaded.entry("dev.plank.demo").expect("recorded");
        assert_eq!(entry.sha256, "hash-a");
        assert!(entry.granted.contains(&Capability::Net));
        assert_eq!(entry.projects, vec![PathBuf::from("/repo")]);
        assert_eq!(
            reloaded.evaluate(&c, "hash-a", Path::new("/repo"), &SigStatus::Absent),
            Decision::Trusted
        );
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A corrupt trust file must lose approvals, never grant them: asking
    /// again is the only safe failure mode.
    #[test]
    fn a_corrupt_trust_file_forgets_rather_than_trusts() {
        let home = temp_dir("trust-corrupt");
        std::fs::create_dir_all(home.join("plugins")).unwrap();
        std::fs::write(home.join("plugins").join("trust.json"), b"{not json").unwrap();
        let trust = TrustStore::load(&home);
        let c = component(Origin::UserScan, vec![]);
        assert_eq!(
            trust.evaluate(&c, "hash-a", Path::new("/repo"), &SigStatus::Absent),
            Decision::Unknown
        );
        let _ = std::fs::remove_dir_all(&home);
    }

    /// With the runtime absent, discovery and trust still run and the registry
    /// still explains itself — the whole point of keeping this module free of
    /// the feature gate.
    #[test]
    fn an_untrusted_component_is_held_before_the_runtime_is_asked() {
        let root = temp_dir("registry-held");
        let p = plugin_dir(&root, "demo", FULL, &["demo.wasm"]);
        let set = discover(&set_of(vec![p]));
        let trust = TrustStore::ephemeral();
        let mut host = crate::wasmhost::NoWasmHost;
        let reg = Registry::build(&set, &trust, Path::new("/repo"), &mut host);

        assert!(reg.loaded.is_empty());
        assert_eq!(reg.held.len(), 1);
        assert_eq!(reg.held[0].1, Decision::Unknown);
        assert!(
            reg.held[0].1.reason().contains("not seen before"),
            "a held component must say why"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Approved but unrunnable: with no runtime the load fails, and it fails as
    /// a *warning* naming the component, not as a silent absence.
    #[test]
    fn a_trusted_component_without_a_runtime_warns_and_does_not_load() {
        let root = temp_dir("registry-noruntime");
        let p = plugin_dir(&root, "demo", FULL, &["demo.wasm"]);
        let set = discover(&set_of(vec![p]));
        let mut trust = TrustStore::ephemeral();
        let sha = module_sha256(&set.components[0].path).expect("hash");
        trust
            .approve(
                &set.components[0],
                &sha,
                Path::new("/repo"),
                &SigStatus::Absent,
            )
            .unwrap();

        let mut host = crate::wasmhost::NoWasmHost;
        let reg = Registry::build(&set, &trust, Path::new("/repo"), &mut host);
        assert!(reg.held.is_empty(), "trust said yes");
        assert!(reg.loaded.is_empty(), "but nothing can run it");
        assert!(
            reg.warnings
                .iter()
                .any(|w| w.contains("dev.plank.demo") && w.contains("no WASM plugin support")),
            "{:?}",
            reg.warnings
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Three strikes disables a component for the session, and nothing short of
    /// a restart brings it back.
    #[test]
    fn three_strikes_disable_a_component_for_the_session() {
        let mut reg = Registry {
            loaded: vec![Loaded {
                component: component(Origin::UserScan, vec![]),
                strikes: 0,
                tools: Vec::new(),
                commands: Vec::new(),
            }],
            ..Registry::default()
        };
        assert!(reg.is_live("dev.plank.demo"));
        assert!(!reg.strike("dev.plank.demo"));
        assert!(!reg.strike("dev.plank.demo"));
        assert!(reg.is_live("dev.plank.demo"), "two strikes is still alive");
        assert!(reg.strike("dev.plank.demo"), "the third disables it");
        assert!(!reg.is_live("dev.plank.demo"));
        // Further strikes are harmless and it stays disabled.
        assert!(reg.strike("dev.plank.demo"));
        assert!(!reg.is_live("dev.plank.demo"));
        assert!(!reg.strike("nobody"), "an unknown id is not a strike");
    }

    #[test]
    fn command_specs_parse_and_tolerate_a_leading_slash() {
        let specs = parse_command_specs(
            br#"[{"name": "/greet", "args": "<who>", "desc": "hi"},
                 {"name": "bare"},
                 {"desc": "nameless, so not a command"}]"#,
        )
        .expect("parsed");
        assert_eq!(specs.len(), 2, "{specs:?}");
        assert_eq!(specs[0].name, "greet", "a leading slash is stripped");
        assert_eq!(specs[1].name, "bare");
        assert!(specs[1].args.is_empty() && specs[1].desc.is_empty());
    }

    /// A command that replies with nonsense still *ran*; the outcome is empty,
    /// not an error. Erroring here would cost the component a strike for the
    /// host's inability to read a reply it did not have to send.
    #[test]
    fn a_command_reply_degrades_to_an_empty_outcome() {
        assert_eq!(parse_cmd_output(b"not json"), CmdOutput::default());
        assert_eq!(parse_cmd_output(b"{}"), CmdOutput::default());
        let out = parse_cmd_output(br#"{"print": "one line", "inject": "x", "nonsense": 1}"#);
        assert_eq!(out.print, vec!["one line".to_string()]);
        assert_eq!(out.inject.as_deref(), Some("x"));
        assert_eq!(out.prompt, None, "unknown fields are ignored, not fatal");
    }

    /// A disabled component drops out of the menu entirely. Leaving it listed
    /// would offer the user a command that is guaranteed to fail.
    #[test]
    fn a_struck_out_component_stops_contributing_commands() {
        let mut reg = Registry {
            loaded: vec![Loaded {
                component: component(Origin::UserScan, vec![]),
                strikes: 0,
                tools: Vec::new(),
                commands: vec![CommandSpec {
                    alias: "demo:greet".to_string(),
                    name: "greet".to_string(),
                    args: String::new(),
                    desc: String::new(),
                }],
            }],
            ..Registry::default()
        };
        assert_eq!(reg.commands().len(), 1);
        for _ in 0..STRIKE_LIMIT {
            reg.strike("dev.plank.demo");
        }
        assert!(
            reg.commands().is_empty(),
            "a disabled component must not be offered"
        );
    }

    /// Both spellings resolve to the same command, and the alias is what makes
    /// a component's frames addressable while a built-in still owns the bare
    /// name — `/arcade:matrix` when `/matrix` is the built-in rain.
    #[test]
    fn a_command_is_addressable_bare_and_namespaced() {
        let specs =
            parse_command_specs(br#"[{"name": "matrix", "desc": "rain"}]"#).expect("parsed");
        assert_eq!(specs[0].name, "matrix");
        assert!(
            specs[0].alias.is_empty(),
            "the alias is stamped at load, where the plugin name is known"
        );

        let reg = Registry::with_loaded(vec![Loaded {
            component: component(Origin::UserScan, vec![]),
            strikes: 0,
            tools: Vec::new(),
            commands: vec![CommandSpec {
                alias: "arcade:matrix".to_string(),
                name: "matrix".to_string(),
                args: String::new(),
                desc: "rain".to_string(),
            }],
        }]);
        let offered = reg.commands();
        assert_eq!(offered.len(), 1, "one command, two ways to say it");
        assert_eq!(offered[0].1.alias, "arcade:matrix");
        assert_eq!(offered[0].1.name, "matrix");
    }

    /// The held listing has to name the component, the reason, what it wants,
    /// and the exact command that approves it — a security decision the user
    /// cannot act on is just noise.
    #[test]
    fn the_held_listing_says_what_it_wants_and_how_to_approve_it() {
        let reg = Registry {
            held: vec![(
                component(Origin::ProjectScan, vec![Capability::Exec]),
                Decision::Unknown,
            )],
            ..Registry::default()
        };
        let out = render_held(&reg);
        assert!(out.contains("dev.plank.demo"), "{out}");
        assert!(out.contains("not seen before"), "{out}");
        assert!(out.contains("exec"), "{out}");
        assert!(out.contains("not a sandboxed component"), "{out}");
        assert!(out.contains("/plugins trust dev.plank.demo"), "{out}");
        assert!(
            render_held(&Registry::default()).is_empty(),
            "nothing held, nothing said"
        );
    }

    /// Test-only helper: sorted copy, so an expectation can be written in the
    /// order a human would list it.
    trait TapSorted {
        fn tap_sorted(self) -> Self;
    }
    impl<T: Ord> TapSorted for Vec<T> {
        fn tap_sorted(mut self) -> Self {
            self.sort_unstable();
            self
        }
    }
}
