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

    let mut capabilities = Vec::new();
    for c in list("capabilities") {
        match Capability::parse(&c) {
            Some(cap) => capabilities.push(cap),
            None => warnings.push(format!("wasm component '{id}': unknown capability '{c}'")),
        }
    }
    // `log` reaches nothing outside plank's own debug file, so it is always
    // present rather than something every author must remember to ask for.
    capabilities.push(Capability::Log);
    capabilities.sort_unstable();
    capabilities.dedup();

    Some(WasmManifest {
        id,
        abi,
        module,
        surfaces,
        capabilities,
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
        }
    }
}

/// Recorded trust decisions, one file under the plank home.
#[derive(Debug, Clone, Default)]
pub struct TrustStore {
    /// Where it persists. `None` for an in-memory store (tests, no HOME).
    path: Option<PathBuf>,
    entries: BTreeMap<String, TrustEntry>,
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
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return store;
        };
        let Some(Json::Obj(top)) = json_parse(&text) else {
            return store;
        };
        for (id, value) in top {
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
            let mut granted: Vec<Capability> = granted;
            granted.sort_unstable();
            granted.dedup();
            store.entries.insert(
                id,
                TrustEntry {
                    sha256: sha256.clone(),
                    granted,
                    projects,
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

    /// Judges `component`, whose module hashes to `sha256`, in `project`.
    ///
    /// Checks in the order a user would want them explained: unknown before
    /// changed, changed before widened, and the project question last, since
    /// it is the only one whose answer is per-repo rather than per-artifact.
    #[must_use]
    pub fn evaluate(&self, component: &WasmComponent, sha256: &str, project: &Path) -> Decision {
        let Some(entry) = self.entries.get(&component.manifest.id) else {
            return Decision::Unknown;
        };
        if entry.sha256 != sha256 {
            return Decision::Changed;
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
    ) -> Result<(), String> {
        let entry = self
            .entries
            .entry(component.manifest.id.clone())
            .or_insert_with(|| TrustEntry {
                sha256: sha256.to_string(),
                granted: Vec::new(),
                projects: Vec::new(),
            });
        // A re-approval after a change replaces the identity outright: the old
        // hash approved different bytes and must not linger beside the new one.
        entry.sha256 = sha256.to_string();
        entry.granted.clone_from(&component.manifest.capabilities);
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
            let _ = write!(
                out,
                "  {}: {{\n    \"sha256\": {},\n    \"granted\": [{}],\n    \"projects\": [{}]\n  }}",
                json_str(id),
                json_str(&e.sha256),
                granted.join(", "),
                projects.join(", "),
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
    /// Name without the leading slash.
    pub name: String,
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
            _ => {}
        }
    }
    out
}

/// A session's WASM state: the admitted components and the runtime they live
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
    pub fn activate(
        &mut self,
        plugins: &PluginSet,
        home: Option<&Path>,
        project: &Path,
    ) -> Vec<String> {
        let found = discover(plugins);
        let trust = home.map_or_else(TrustStore::ephemeral, TrustStore::load);
        self.registry = Registry::build(&found, &trust, project, &mut *self.host);
        self.registry.warnings.clone()
    }

    /// Approves a held component and loads it, without waiting for a restart.
    ///
    /// # Errors
    /// Returns a message when nothing is held under `id`, when the approval
    /// cannot be recorded, or when the module then fails to load.
    pub fn approve(
        &mut self,
        id: &str,
        home: Option<&Path>,
        project: &Path,
    ) -> Result<String, String> {
        let idx = self
            .registry
            .held
            .iter()
            .position(|(c, _)| c.manifest.id == id)
            .ok_or_else(|| format!("no held wasm component '{id}'"))?;
        let (component, _) = self.registry.held[idx].clone();
        let sha = module_sha256(&component.path)
            .ok_or_else(|| format!("cannot hash {}", component.path.display()))?;
        let mut trust = home.map_or_else(TrustStore::ephemeral, TrustStore::load);
        trust.approve(&component, &sha, project)?;

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
            let decision = trust.evaluate(component, &sha, project);
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
            let commands = if component.manifest.surfaces.contains(&Surface::Command) {
                match host.call(&component.manifest.id, "command_specs", b"") {
                    Ok(bytes) => parse_command_specs(&bytes).unwrap_or_else(|| {
                        out.warnings.push(format!(
                            "wasm component '{}': command_specs returned unreadable JSON",
                            component.manifest.id
                        ));
                        Vec::new()
                    }),
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
                commands,
            });
        }
        out
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
        let id = self
            .commands()
            .into_iter()
            .find(|(_, c)| c.name == name)
            .map(|(id, _)| id.to_string())
            .ok_or_else(|| format!("no wasm command '{name}'"))?;
        let payload = format!(
            "{{\"name\": {}, \"args\": {}}}",
            json_str(name),
            json_str(args)
        );
        match host.call(&id, "command_run", payload.as_bytes()) {
            Ok(bytes) => {
                let mut out = parse_cmd_output(&bytes);
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
        let surfaces: Vec<&str> = c.manifest.surfaces.iter().map(|s| s.label()).collect();
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
    missing
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

        assert_eq!(trust.evaluate(&c, "hash-a", project), Decision::Unknown);

        trust.approve(&c, "hash-a", project).unwrap();
        assert_eq!(trust.evaluate(&c, "hash-a", project), Decision::Trusted);

        // Same id, different bytes.
        assert_eq!(trust.evaluate(&c, "hash-b", project), Decision::Changed);

        // Same bytes, a new capability.
        let greedy = component(Origin::UserScan, vec![Capability::State, Capability::Exec]);
        assert_eq!(
            trust.evaluate(&greedy, "hash-a", project),
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
        trust.approve(&c, "hash-a", Path::new("/repo-one")).unwrap();

        assert_eq!(
            trust.evaluate(&c, "hash-a", Path::new("/repo-one")),
            Decision::Trusted
        );
        assert_eq!(
            trust.evaluate(&c, "hash-a", Path::new("/repo-two")),
            Decision::ProjectUnapproved
        );

        // A user-scanned component of the same bytes needs no per-repo answer:
        // the user named that directory themselves.
        let user = component(Origin::UserScan, vec![Capability::State]);
        assert_eq!(
            trust.evaluate(&user, "hash-a", Path::new("/repo-two")),
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
            trust.approve(&c, "hash-a", Path::new("/repo")).unwrap();
        }
        let reloaded = TrustStore::load(&home);
        let entry = reloaded.entry("dev.plank.demo").expect("recorded");
        assert_eq!(entry.sha256, "hash-a");
        assert!(entry.granted.contains(&Capability::Net));
        assert_eq!(entry.projects, vec![PathBuf::from("/repo")]);
        assert_eq!(
            reloaded.evaluate(&c, "hash-a", Path::new("/repo")),
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
            trust.evaluate(&c, "hash-a", Path::new("/repo")),
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
            .approve(&set.components[0], &sha, Path::new("/repo"))
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
                commands: vec![CommandSpec {
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
