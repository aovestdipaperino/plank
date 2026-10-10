// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Profile specs: the `profile` block of a plugin manifest.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::tools::mcp::{Json, json_parse, json_write};

/// The accent color a profile paints the UI with.
///
/// Kept as plank's own enum rather than a `ratatui::style::Color` so this
/// module stays a leaf: `src/tui.rs` converts at the point of use, and the
/// plain-stdout path emits its own ANSI from the same value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accent {
    /// An ANSI 256-color index.
    Indexed(u8),
    /// A 24-bit color.
    Rgb(u8, u8, u8),
}

/// The `profile` block of a plugin manifest, with paths already resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileSpec {
    /// Name shown in the banner, window title and status bar. Falls back to
    /// the plugin name when absent.
    pub display_name: Option<String>,
    /// PNG rendered as the startup banner; the plank logo when absent.
    pub logo: Option<PathBuf>,
    /// UI accent; the built-in green when absent.
    pub accent: Option<Accent>,
    /// `secondary`: the far end of the status verb's shimmer sweep, which
    /// ranges from [`accent`](Self::accent) to this color.
    ///
    /// `None` with an accent set derives one (`anim::derived_secondary`), so an
    /// accented profile never shimmers in the built-in olive. `None` with no
    /// accent leaves the hand-picked `status::SHIMMER_RAMP` alone, which is
    /// what keeps plank's own look unchanged.
    ///
    /// Nothing requires it to be lighter than the accent: plank's own profile
    /// uses `#444444`, so its sweep reads as a shadow crossing the word.
    pub secondary: Option<Accent>,
    /// The file holding the whole system prompt. The one required field.
    pub system_prompt: PathBuf,
    /// Builtin tools the profile allows. `None` means all of them.
    pub builtin_tools: Option<Vec<String>>,
    /// The `settings` object, re-serialized so `crate::settings` can overlay
    /// it with the same text-based path it uses for every settings file.
    pub settings_json: Option<String>,
    /// Non-fatal complaints raised while parsing.
    pub warnings: Vec<String>,
    /// `folderContext`: whether the session starts with context about the
    /// launch folder (its git status and `<folder>/.plank/MEMORY.md`).
    /// `false` unless the manifest says `true`.
    pub folder_context: bool,
    /// `agentsMd`: whether `AGENTS.md` files are read, and offered or linked
    /// at launch. `false` unless the manifest says `true`.
    pub agents_md: bool,
    /// `recommendedModel`: an engine name from the catalog that the run uses
    /// when no `--model` was given and that engine's main file is already on
    /// disk. Never downloaded and never asked about; `None` when absent or
    /// malformed.
    pub recommended_model: Option<String>,
    /// `steering`: the named direction (from `~/.plank/models/vectors.json`)
    /// and scales the run steers with when [`recommended_model`] is the
    /// engine selected, `{"direction": NAME, "ffn": F, "attn": F}` with `ffn`
    /// 1.0 and `attn` 0.0 when absent. A `--dir-steering` on the command line
    /// wins. `None` when absent or malformed, or when no `recommendedModel`
    /// names the engine it belongs to.
    ///
    /// [`recommended_model`]: Self::recommended_model
    pub steering: Option<crate::steervec::Steering>,
    /// `grids`: MCP server name to WASM frame component id, for servers
    /// allowed to hand table data to a grid component. Empty unless the
    /// manifest declares routes; a malformed entry is dropped with a
    /// warning rather than widening what gets routed.
    pub grids: BTreeMap<String, String>,
    /// `verbs` or `additionalVerbs`: the status-bar verb pools the profile
    /// replaces or extends. `None` keeps plank's own vocabulary.
    pub verbs: Option<ProfileVerbs>,
}

/// The phase keys of a per-phase `verbs` object, in the order
/// [`ProfileVerbs::pools`] is indexed (`crate::status::VerbPhase` as `u8`).
pub const VERB_PHASE_KEYS: [&str; 5] = ["thinking", "generating", "tool", "prefill", "fun"];

/// A profile's status-bar verbs, from `verbs` (replacing plank's pools) or
/// `additionalVerbs` (appended to them). The two are mutually exclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileVerbs {
    /// `true` for `verbs`, `false` for `additionalVerbs`.
    pub replace: bool,
    /// One pool per phase, indexed as [`VERB_PHASE_KEYS`]. An empty pool
    /// leaves that phase's built-in verbs alone, even under `replace`, so a
    /// per-phase object that omits a phase never leaves it with nothing to
    /// show.
    pub pools: [Vec<String>; 5],
}

impl ProfileSpec {
    /// Whether builtin tool `name` is offered under this profile.
    ///
    /// The single source of truth: both `crate::tools::dispatch` and the
    /// system-prompt schema block call this, so the model can never be told
    /// about a tool dispatch would reject.
    #[must_use]
    pub fn builtin_enabled(&self, name: &str) -> bool {
        match &self.builtin_tools {
            None => true,
            Some(allowed) => allowed.iter().any(|t| t == name),
        }
    }
}

/// The default UI accent when a profile declares none (`crate::tui`'s
/// built-in green).
pub const DEFAULT_ACCENT: Accent = Accent::Indexed(114);

/// An [`Accent`] as an RGB triple, for interpolation.
#[must_use]
pub fn accent_rgb(a: Accent) -> crate::anim::Rgb {
    match a {
        Accent::Indexed(i) => crate::anim::indexed_to_rgb(i),
        Accent::Rgb(r, g, b) => (r, g, b),
    }
}

/// How a profile paints the status verb: the colour the word rests in, and the
/// shades the highlight sweeps through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shimmer {
    /// The word's resting colour — the profile's `secondary`.
    pub rest: crate::anim::Rgb,
    /// Sweep shades, outermost column first, ending on the accent.
    pub ramp: [u8; 3],
}

/// How a profile paints the status verb, or `None` to leave the hand-picked
/// `crate::status::SHIMMER_RAMP` and the default accent in place.
///
/// The word rests in the `secondary` and the **accent is the travelling
/// highlight**: the sweep runs secondary -> accent, brightest (in the accent's
/// own colour) at its centre. That is the way round to remember — the accent is
/// the thing moving, not the background it moves over.
///
/// A profile that declares neither colour keeps the built-in ramp, which is
/// what makes this invisible to a plain run and to every profile that has not
/// opted in. Declaring only an `accent` derives the resting colour from it
/// ([`crate::anim::derived_secondary`]) so the word still reads in the
/// profile's own hue; declaring only a `secondary` rests in it and sweeps
/// toward the default accent.
#[must_use]
pub fn shimmer_ramp(accent: Option<Accent>, secondary: Option<Accent>) -> Option<Shimmer> {
    if accent.is_none() && secondary.is_none() {
        return None;
    }
    let highlight = accent_rgb(accent.unwrap_or(DEFAULT_ACCENT));
    let rest = secondary.map_or_else(|| crate::anim::derived_secondary(highlight), accent_rgb);
    Some(Shimmer {
        rest,
        ramp: crate::anim::shimmer_ramp(rest, highlight),
    })
}

/// Parses an ANSI index (`"160"`) or a hex triple (`"#c04040"`).
#[must_use]
pub fn parse_accent(s: &str) -> Option<Accent> {
    if let Some(hex) = s.strip_prefix('#') {
        // Count and index by chars, not bytes: a non-ASCII manifest value
        // (e.g. a euro sign) can be 6 bytes without being 6 hex digits, and
        // slicing by byte offset into such a string panics on a non-char
        // boundary. Collecting to a Vec<char> keeps every index valid.
        let chars: Vec<char> = hex.chars().collect();
        if chars.len() != 6 {
            return None;
        }
        let byte = |i: usize| -> Option<u8> {
            let pair: String = chars[i..i + 2].iter().collect();
            u8::from_str_radix(&pair, 16).ok()
        };
        return Some(Accent::Rgb(byte(0)?, byte(2)?, byte(4)?));
    }
    s.parse::<u8>().ok().map(Accent::Indexed)
}

/// One optional color field (`accent`, `secondary`), parsed by
/// [`parse_accent`]. An unparseable value warns — naming `fallback` so the
/// message says what happens instead — and yields `None`, never failing the
/// profile over a color.
fn color_field(
    block: &Json,
    key: &str,
    fallback: &str,
    warnings: &mut Vec<String>,
) -> Option<Accent> {
    let raw = str_field(block, key)?;
    let parsed = parse_accent(&raw);
    if parsed.is_none() {
        warnings.push(format!("profile: unrecognized {key} {raw:?}; {fallback}"));
    }
    parsed
}

/// Parses the `profile` block out of a plugin manifest.
///
/// `None` when the manifest does not parse, has no `profile` object, or that
/// object has no `systemPrompt` — a block without a prompt is a skin, and
/// activating a skin over plank's own prompt would misrepresent what is
/// running. Every other malformed field yields a warning and a default.
#[must_use]
pub fn parse(manifest_text: &str, root: &Path) -> Option<ProfileSpec> {
    let root_json = json_parse(manifest_text)?;
    let block = root_json.get("profile")?;
    if !matches!(block, Json::Obj(_)) {
        return None;
    }
    let system_prompt = str_field(block, "systemPrompt")?;
    let mut warnings = Vec::new();

    let accent = color_field(block, "accent", "using the default", &mut warnings);
    let secondary = color_field(
        block,
        "secondary",
        "deriving one from the accent",
        &mut warnings,
    );

    let builtin_tools = match block.get("tools").and_then(|t| t.get("builtin")) {
        None => None,
        Some(Json::Arr(items)) => {
            let mut bad = false;
            let names: Vec<String> = items
                .iter()
                .filter_map(|v| match v {
                    Json::Str(s) if !s.is_empty() => Some(s.clone()),
                    _ => {
                        bad = true;
                        None
                    }
                })
                .collect();
            if bad {
                warnings.push(
                    "profile: tools.builtin has a non-string or empty entry; skipping it"
                        .to_string(),
                );
            }
            warn_unknown_builtin_names(&names, &mut warnings);
            Some(names)
        }
        Some(_) => {
            // Fails closed, not open: `None` means "every builtin allowed",
            // so a malformed restriction must not silently widen access to
            // everything — an empty allow-list is the safe default here.
            warnings.push(
                "profile: tools.builtin is not an array of strings; allowing no builtins"
                    .to_string(),
            );
            Some(Vec::new())
        }
    };

    let display_name = match block.get("displayName") {
        None => None,
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        Some(_) => {
            warnings.push(
                "profile: displayName is not a non-empty string; using the default".to_string(),
            );
            None
        }
    };

    let logo = match block.get("logo") {
        None => None,
        Some(Json::Str(s)) if !s.is_empty() => Some(resolve(root, s)),
        Some(_) => {
            warnings.push("profile: logo is not a non-empty string; using the default".to_string());
            None
        }
    };

    let settings_json = match block.get("settings") {
        None => None,
        Some(s @ Json::Obj(members)) => {
            warn_refused_engine_settings(members, &mut warnings);
            let mut out = String::new();
            json_write(&mut out, s);
            Some(out)
        }
        Some(_) => {
            warnings.push("profile: settings is not an object; using the default".to_string());
            None
        }
    };

    let folder_context = bool_field(block, "folderContext", &mut warnings);
    let agents_md = bool_field(block, "agentsMd", &mut warnings);

    let recommended_model = recommended_model_field(block, &mut warnings);
    let steering = steering_field(block, recommended_model.is_some(), &mut warnings);

    let grids = grids_field(block, &mut warnings);

    let verbs = verbs_field(block, &mut warnings);

    Some(ProfileSpec {
        display_name,
        logo,
        accent,
        secondary,
        system_prompt: resolve(root, &system_prompt),
        builtin_tools,
        settings_json,
        warnings,
        folder_context,
        agents_md,
        recommended_model,
        steering,
        grids,
        verbs,
    })
}

/// `verbs` / `additionalVerbs`: either an array of strings, used for every
/// phase, or an object keyed by [`VERB_PHASE_KEYS`] with an array each.
///
/// Declaring both is a contradiction the parser will not guess at: it warns
/// and keeps plank's own verbs. A malformed value warns and is ignored; a
/// malformed entry or unknown phase key inside an otherwise-valid value warns
/// and is dropped while the rest is kept.
fn verbs_field(obj: &Json, warnings: &mut Vec<String>) -> Option<ProfileVerbs> {
    let (key, value, replace) = match (obj.get("verbs"), obj.get("additionalVerbs")) {
        (None, None) => return None,
        (Some(_), Some(_)) => {
            warnings.push(
                "profile: verbs and additionalVerbs are mutually exclusive; using the default verbs"
                    .to_string(),
            );
            return None;
        }
        (Some(v), None) => ("verbs", v, true),
        (None, Some(v)) => ("additionalVerbs", v, false),
    };
    let mut pools: [Vec<String>; 5] = Default::default();
    match value {
        Json::Arr(_) => {
            let pool = verb_list(value, key, warnings);
            for slot in &mut pools {
                slot.clone_from(&pool);
            }
        }
        Json::Obj(members) => {
            for (phase, list) in members {
                let Some(i) = VERB_PHASE_KEYS.iter().position(|k| k == phase) else {
                    warnings.push(format!(
                        "profile: {key} has unknown phase {phase:?}; skipping it"
                    ));
                    continue;
                };
                pools[i] = verb_list(list, &format!("{key}.{phase}"), warnings);
            }
        }
        _ => {
            warnings.push(format!(
                "profile: {key} is neither an array nor an object; using the default verbs"
            ));
            return None;
        }
    }
    if pools.iter().all(Vec::is_empty) {
        warnings.push(format!(
            "profile: {key} names no verbs; using the default verbs"
        ));
        return None;
    }
    Some(ProfileVerbs { replace, pools })
}

/// The non-empty strings of one verb array; anything else warns.
fn verb_list(value: &Json, what: &str, warnings: &mut Vec<String>) -> Vec<String> {
    let Json::Arr(items) = value else {
        warnings.push(format!("profile: {what} is not an array; skipping it"));
        return Vec::new();
    };
    let mut bad = false;
    let out = items
        .iter()
        .filter_map(|v| match v {
            Json::Str(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
            _ => {
                bad = true;
                None
            }
        })
        .collect();
    if bad {
        warnings.push(format!(
            "profile: {what} has a non-string or empty entry; skipping it"
        ));
    }
    out
}

/// `grids`: an object mapping MCP server name to WASM frame component id.
///
/// A non-object value warns and yields an empty map, the same fail-closed
/// shape as `tools.builtin`: a malformed restriction must never widen what
/// gets routed. Within an otherwise-valid object, an entry with a
/// non-string value, or an empty key or value, warns and is dropped while
/// the rest of the map is kept.
fn grids_field(obj: &Json, warnings: &mut Vec<String>) -> BTreeMap<String, String> {
    match obj.get("grids") {
        None => BTreeMap::new(),
        Some(Json::Obj(members)) => {
            let mut out = BTreeMap::new();
            for (key, value) in members {
                match value {
                    Json::Str(s) if !key.is_empty() && !s.is_empty() => {
                        out.insert(key.clone(), s.clone());
                    }
                    _ => warnings.push(format!(
                        "profile: grids entry {key:?} is malformed; skipping it"
                    )),
                }
            }
            out
        }
        Some(_) => {
            warnings.push(
                "profile: grids is not an object; routing no MCP servers to a grid".to_string(),
            );
            BTreeMap::new()
        }
    }
}

/// Pushes a warning for each `names` entry that matches no known builtin.
///
/// An entry matching nothing is a warning, not an error: a profile written
/// against a newer plank (one that added a builtin this binary does not have
/// yet) must still run, just with that entry silently inert.
fn warn_unknown_builtin_names(names: &[String], warnings: &mut Vec<String>) {
    let known = crate::sysprompt::known_builtin_names();
    for name in names {
        if !known.contains(name) {
            warnings.push(format!(
                "profile: tools.builtin names {name:?}, which matches no known builtin tool"
            ));
        }
    }
}

/// Pushes a warning for each `engine.*` key a profile's `settings` block
/// tries to set.
///
/// `crate::settings::Settings::overlay_from` drops `engine.*` from a
/// profile's settings layer the same way it drops it from a plugin's, but
/// silently — the caller there has no per-key warning channel to surface it
/// on. This is that surfacing: one warning per dropped key, at parse time,
/// the same shape `plugins.rs` emits for its own refused sections.
fn warn_refused_engine_settings(members: &[(String, Json)], warnings: &mut Vec<String>) {
    let Some((_, engine_value)) = members.iter().find(|(k, _)| k == "engine") else {
        return;
    };
    match engine_value {
        Json::Obj(engine_keys) if !engine_keys.is_empty() => {
            for (key, _) in engine_keys {
                warnings.push(format!(
                    "profile: settings.engine.{key} is refused (a profile may not set engine.*); set it yourself in ~/.plank/settings.json if you want it"
                ));
            }
        }
        _ => warnings.push(
            "profile: settings.engine is refused (a profile may not set engine.*); set it yourself in ~/.plank/settings.json if you want it"
                .to_string(),
        ),
    }
}

/// `recommendedModel`: an engine name, or `None` with a warning when it is
/// present but not a valid one (`crate::engines::valid_name`).
fn recommended_model_field(obj: &Json, warnings: &mut Vec<String>) -> Option<String> {
    match obj.get("recommendedModel") {
        None => None,
        Some(Json::Str(s)) if crate::engines::valid_name(s) => Some(s.clone()),
        Some(Json::Str(s)) if !s.is_empty() => {
            warnings.push(format!(
                "profile: recommendedModel {s:?} is not a valid engine name; ignoring it"
            ));
            None
        }
        Some(_) => {
            warnings.push(
                "profile: recommendedModel is not a non-empty string; ignoring it".to_string(),
            );
            None
        }
    }
}

/// `steering`: a direction and its scales (`crate::steervec::parse_steering`),
/// or `None` with a warning when it is malformed or has no `recommendedModel`
/// to belong to: a stored direction is tied to one model.
fn steering_field(
    obj: &Json,
    has_model: bool,
    warnings: &mut Vec<String>,
) -> Option<crate::steervec::Steering> {
    let value = obj.get("steering")?;
    if !has_model {
        warnings.push(
            "profile: steering needs a recommendedModel naming the engine it belongs to; \
             ignoring it"
                .to_string(),
        );
        return None;
    }
    let mut text = String::new();
    json_write(&mut text, value);
    let parsed = serde_json::from_str::<serde_json::Value>(&text)
        .map_err(|e| format!("steering: {e}"))
        .and_then(|v| crate::steervec::parse_steering(&v));
    match parsed {
        Ok(st) => Some(st),
        Err(e) => {
            warnings.push(format!("profile: {e}; ignoring it"));
            None
        }
    }
}

/// A boolean profile field: `false` when absent, and `false` with a warning
/// when it is not a boolean, so a typo never turns a context source on.
fn bool_field(obj: &Json, key: &str, warnings: &mut Vec<String>) -> bool {
    match obj.get(key) {
        None => false,
        Some(Json::Bool(b)) => *b,
        Some(_) => {
            warnings.push(format!("profile: {key} is not true or false; using false"));
            false
        }
    }
}

/// A non-empty string member, or `None`.
fn str_field(obj: &Json, key: &str) -> Option<String> {
    match obj.get(key) {
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Resolves a manifest path against the plugin root, leaving absolute paths be.
fn resolve(root: &Path, p: &str) -> PathBuf {
    let p = Path::new(p);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

/// The profile a run is operating under: the plugin name that selected it and
/// its parsed spec.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveProfile {
    /// The plugin name `--profile` named.
    pub name: String,
    /// The parsed block.
    pub spec: ProfileSpec,
    /// The contents of `spec.system_prompt`, read exactly once at
    /// activation (`resolve_and_activate_profile` in `main.rs`).
    ///
    /// Composition (`sysprompt.rs`) then uses this text and never touches
    /// the filesystem itself, so a prompt file that is deleted or edited
    /// mid-session cannot make composition fail or drift: the process reads
    /// it once, before there is a TUI to corrupt.
    pub prompt: String,
    /// The plugin manifest's `version`, empty when it declares none.
    pub version: String,
}

/// The active profile, set once at startup.
static ACTIVE: std::sync::OnceLock<ActiveProfile> = std::sync::OnceLock::new();

/// Installs the active profile. The first call wins; later calls are ignored,
/// which is what makes every reader below infallible.
pub fn install(active: ActiveProfile) {
    let _ = ACTIVE.set(active);
}

/// The active profile, or `None` for a plain plank run.
#[must_use]
pub fn active() -> Option<&'static ActiveProfile> {
    ACTIVE.get()
}

/// The active profile's plugin name, or `None`.
#[must_use]
pub fn active_name() -> Option<&'static str> {
    ACTIVE.get().map(|a| a.name.as_str())
}

/// The active profile's status-bar verbs, or `None` for plank's own.
#[must_use]
pub fn active_verbs() -> Option<&'static ProfileVerbs> {
    ACTIVE.get().and_then(|a| a.spec.verbs.as_ref())
}

/// What to call the agent in the banner, window title and status bar:
/// the profile's `displayName`, else its plugin name, else `plank`.
#[must_use]
pub fn display_name() -> &'static str {
    match ACTIVE.get() {
        Some(a) => a.spec.display_name.as_deref().unwrap_or(&a.name),
        None => "plank",
    }
}

/// The profile's name with its version, `HAL v0.3.1`, for the banners; the
/// name alone when the manifest declares no version.
#[must_use]
pub fn title() -> String {
    title_of(
        display_name(),
        ACTIVE.get().map_or("", |a| a.version.as_str()),
    )
}

/// [`title`] for an explicit name and version, so the rendering is testable
/// without the process-global active profile.
#[must_use]
pub fn title_of(name: &str, version: &str) -> String {
    let version = version.trim();
    if version.is_empty() {
        return name.to_owned();
    }
    let v = version.strip_prefix('v').unwrap_or(version);
    format!("{name} v{v}")
}

// Test-only override for the active profile, consulted before the process
// global `ACTIVE` `OnceLock`.
//
// `ACTIVE` can only be set once per process, which makes it useless for
// testing the `dispatch` guard's wiring across multiple cases in the same
// test binary: whichever test runs first would poison every test after it.
// This thread-local lets a test install (and, on drop, remove) its own
// `ProfileSpec` without touching `ACTIVE` at all. Thread-local rather than
// process-global so tests running on different threads never see each
// other's override.
#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::RefCell<Option<ProfileSpec>> = const { std::cell::RefCell::new(None) };
}

/// RAII guard that installs a `ProfileSpec` into [`TEST_OVERRIDE`] for the
/// life of a test and clears it on drop — including an unwinding drop from a
/// panicking test, so a failing test can never leak its override into the
/// next test on the same thread.
#[cfg(test)]
pub(crate) struct TestProfileGuard;

#[cfg(test)]
impl TestProfileGuard {
    /// Installs `spec` as the active profile for the current thread until
    /// the returned guard is dropped.
    pub(crate) fn install(spec: ProfileSpec) -> Self {
        TEST_OVERRIDE.with(|cell| *cell.borrow_mut() = Some(spec));
        Self
    }
}

#[cfg(test)]
impl Drop for TestProfileGuard {
    fn drop(&mut self) {
        TEST_OVERRIDE.with(|cell| *cell.borrow_mut() = None);
    }
}

/// Whether the session starts with launch-folder context (git status and
/// project memory). Always true without a profile; a profile opts in with
/// `folderContext: true`.
#[must_use]
pub fn folder_context_enabled() -> bool {
    #[cfg(test)]
    {
        if let Some(on) =
            TEST_OVERRIDE.with(|cell| cell.borrow().as_ref().map(|s| s.folder_context))
        {
            return on;
        }
    }
    ACTIVE.get().is_none_or(|a| a.spec.folder_context)
}

/// Whether `AGENTS.md` is read, offered and linked. Always true without a
/// profile; a profile opts in with `agentsMd: true`.
#[must_use]
pub fn agents_md_enabled() -> bool {
    #[cfg(test)]
    {
        if let Some(on) = TEST_OVERRIDE.with(|cell| cell.borrow().as_ref().map(|s| s.agents_md)) {
            return on;
        }
    }
    ACTIVE.get().is_none_or(|a| a.spec.agents_md)
}

/// Whether builtin tool `name` is offered under the active profile.
/// True for every tool when no profile is active.
#[must_use]
pub fn builtin_enabled(name: &str) -> bool {
    #[cfg(test)]
    {
        if let Some(enabled) = TEST_OVERRIDE.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|spec| spec.builtin_enabled(name))
        }) {
            return enabled;
        }
    }
    ACTIVE.get().is_none_or(|a| a.spec.builtin_enabled(name))
}

/// What `--profile NAME` came to.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// No `--profile` was given; run as plain plank.
    None,
    /// Activate this profile.
    Activate(Box<ActiveProfile>),
    /// A bare `--profile`: list these names and exit successfully.
    List(Vec<String>),
    /// The name matched no plugin. Carries the available profile names so the
    /// error can show them.
    NoSuchPlugin(String, Vec<String>),
    /// The name matched a plugin that declares no `profile` block.
    NotAProfile(String),
    /// `--profile ""`: an explicit empty name, distinct from the bare-flag
    /// listing request. Fatal — see finding 4 in the profile cleanup notes.
    EmptyName,
}

/// Resolves `--profile`'s argument against the loaded plugins.
///
/// `explicit_empty` distinguishes `--profile ""` (a fatal error: the user
/// supplied an empty name) from a bare `--profile` (the listing request) —
/// both otherwise parse to `requested == Some("")`.
///
/// Failure is deliberately not silent: running as plain plank when the user
/// asked for HAL is worse than not running, so every miss is a distinct
/// variant the caller turns into a fatal message.
#[must_use]
pub fn resolve_profile(
    requested: Option<&str>,
    explicit_empty: bool,
    set: &crate::plugins::PluginSet,
) -> Resolution {
    let Some(name) = requested else {
        return Resolution::None;
    };
    if name.is_empty() {
        if explicit_empty {
            return Resolution::EmptyName;
        }
        return Resolution::List(crate::plugins::profile_names(set));
    }
    let Some(plugin) = set.plugins.iter().find(|p| p.name == name) else {
        return Resolution::NoSuchPlugin(name.to_string(), crate::plugins::profile_names(set));
    };
    match &plugin.profile {
        Some(spec) => Resolution::Activate(Box::new(ActiveProfile {
            name: plugin.name.clone(),
            spec: spec.clone(),
            // Filled in by `resolve_and_activate_profile` (main.rs) once it
            // has confirmed the file is readable; this function has no
            // filesystem access and stays pure.
            prompt: String::new(),
            version: plugin.version.clone(),
        })),
        None => Resolution::NotAProfile(name.to_string()),
    }
}

/// The load-time warning for a profile prompt that never asks for the tool
/// protocol, or `None` when it does.
///
/// Without [`crate::sysprompt::TOOL_PROTOCOL_TOKEN`] the model is handed the
/// schema block but not the DSML call syntax, so it can see its tools and
/// has no way to call them. That can be a deliberate chat-only profile, so
/// it warns rather than refusing.
#[must_use]
pub fn missing_protocol_warning(name: &str, prompt: &str) -> Option<String> {
    (!prompt.contains(crate::sysprompt::TOOL_PROTOCOL_TOKEN)).then(|| {
        format!(
            "plank: profile {name:?}: the prompt has no {} token, so the model is not told how to call its tools",
            crate::sysprompt::TOOL_PROTOCOL_TOKEN
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_banner_title_carries_the_profile_version() {
        assert_eq!(title_of("HAL", "0.3.1"), "HAL v0.3.1");
        assert_eq!(title_of("HAL", "v1.2"), "HAL v1.2", "no doubled v");
        assert_eq!(title_of("HAL", "  "), "HAL", "no version, name alone");
    }

    fn verbs_of(block: &str) -> (Option<ProfileVerbs>, Vec<String>) {
        let text = format!(r#"{{"profile":{{"systemPrompt":"p.md",{block}}}}}"#);
        let spec = parse(&text, Path::new("/p")).expect("parses");
        (spec.verbs, spec.warnings)
    }

    #[test]
    fn verbs_absent_keeps_the_builtins() {
        let spec = parse(r#"{"profile":{"systemPrompt":"p.md"}}"#, Path::new("/p")).unwrap();
        assert_eq!(spec.verbs, None);
    }

    #[test]
    fn a_verbs_array_replaces_every_phase() {
        let (v, w) = verbs_of(r#""verbs":["Hexing","Cursing"]"#);
        let v = v.expect("verbs");
        assert!(v.replace && w.is_empty());
        assert!(v.pools.iter().all(|p| p == &["Hexing", "Cursing"]));
    }

    #[test]
    fn additional_verbs_by_phase_extend() {
        let (v, w) = verbs_of(r#""additionalVerbs":{"tool":["Smelting"],"fun":["Yawning"]}"#);
        let v = v.expect("verbs");
        assert!(!v.replace && w.is_empty());
        assert_eq!(v.pools[2], ["Smelting"]);
        assert_eq!(v.pools[4], ["Yawning"]);
        assert!(v.pools[0].is_empty());
    }

    #[test]
    fn verbs_and_additional_verbs_are_mutually_exclusive() {
        let (v, w) = verbs_of(r#""verbs":["A"],"additionalVerbs":["B"]"#);
        assert_eq!(v, None);
        assert!(w[0].contains("mutually exclusive"), "{w:?}");
    }

    #[test]
    fn malformed_verbs_warn_and_keep_what_is_valid() {
        let (v, w) = verbs_of(r#""verbs":{"thinking":["Brooding",3,""],"dreaming":["X"]}"#);
        assert_eq!(v.expect("verbs").pools[0], ["Brooding"]);
        assert_eq!(w.len(), 2, "{w:?}");
        let (v, w) = verbs_of(r#""verbs":"Hexing""#);
        assert_eq!(v, None);
        assert_eq!(w.len(), 1);
        let (v, w) = verbs_of(r#""additionalVerbs":[]"#);
        assert_eq!(v, None);
        assert!(w[0].contains("names no verbs"));
    }

    #[test]
    fn folder_context_and_agents_md_default_to_false() {
        let root = Path::new("/p");
        let spec = parse(r#"{"profile":{"systemPrompt":"p.md"}}"#, root).expect("parses");
        assert!(!spec.folder_context && !spec.agents_md);
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","folderContext":true,"agentsMd":true}}"#,
            root,
        )
        .expect("parses");
        assert!(spec.folder_context && spec.agents_md);
        assert!(spec.warnings.is_empty(), "{:?}", spec.warnings);
    }

    #[test]
    fn no_grids_means_empty_with_no_warning() {
        let spec =
            parse(r#"{"profile":{"systemPrompt":"p.md"}}"#, Path::new("/p")).expect("parses");
        assert!(spec.grids.is_empty());
        assert!(spec.warnings.is_empty(), "{:?}", spec.warnings);
    }

    #[test]
    fn a_valid_grids_map_parses() {
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","grids":{"chatbgt":"dev.plank.csvedit"}}}"#,
            Path::new("/p"),
        )
        .expect("parses");
        assert_eq!(
            spec.grids.get("chatbgt").map(String::as_str),
            Some("dev.plank.csvedit")
        );
        assert!(spec.warnings.is_empty(), "{:?}", spec.warnings);
    }

    #[test]
    fn a_non_object_grids_warns_and_yields_empty() {
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","grids":"chatbgt"}}"#,
            Path::new("/p"),
        )
        .expect("parses");
        assert!(spec.grids.is_empty());
        assert_eq!(spec.warnings.len(), 1, "{:?}", spec.warnings);
        assert!(spec.warnings[0].contains("grids"));
    }

    #[test]
    fn a_non_string_grids_value_warns_and_is_dropped_while_others_stay() {
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","grids":{"chatbgt":"dev.plank.csvedit","other":7}}}"#,
            Path::new("/p"),
        )
        .expect("parses");
        assert_eq!(spec.grids.len(), 1);
        assert_eq!(
            spec.grids.get("chatbgt").map(String::as_str),
            Some("dev.plank.csvedit")
        );
        assert_eq!(spec.warnings.len(), 1, "{:?}", spec.warnings);
        assert!(spec.warnings[0].contains("other"));
    }

    #[test]
    fn an_empty_grids_key_or_value_is_dropped() {
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","grids":{"":"dev.plank.csvedit","chatbgt":""}}}"#,
            Path::new("/p"),
        )
        .expect("parses");
        assert!(spec.grids.is_empty());
        assert_eq!(spec.warnings.len(), 2, "{:?}", spec.warnings);
    }

    #[test]
    fn the_chatbgt_fixture_manifest_parses_its_grids_route() {
        let text = std::fs::read_to_string("tests/fixtures/profiles/chatbgt/plugin.json")
            .expect("fixture readable");
        let spec = parse(&text, Path::new("tests/fixtures/profiles/chatbgt")).expect("parses");
        assert_eq!(spec.grids.len(), 1);
        assert_eq!(
            spec.grids.get("chatbgt").map(String::as_str),
            Some("dev.plank.csvedit")
        );
    }

    #[test]
    fn recommended_model_is_an_optional_engine_name() {
        let root = Path::new("/p");
        let spec = parse(r#"{"profile":{"systemPrompt":"p.md"}}"#, root).expect("parses");
        assert_eq!(spec.recommended_model, None);
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","recommendedModel":"qwen"}}"#,
            root,
        )
        .expect("parses");
        assert_eq!(spec.recommended_model.as_deref(), Some("qwen"));
        assert!(spec.warnings.is_empty(), "{:?}", spec.warnings);
    }

    /// A manifest whose profile block holds `systemPrompt` plus `extra`.
    fn manifest(extra: &str) -> String {
        format!(r#"{{"profile":{{"systemPrompt":"p.md",{extra}}}}}"#)
    }

    #[test]
    fn steering_pairs_a_direction_with_the_recommended_model() {
        let root = Path::new("/p");
        let spec = parse(
            &manifest(r#""recommendedModel":"ds4vision","steering":{"direction":"abliterated","attn":1,"ffn":0}"#),
            root,
        )
        .unwrap();
        assert!(spec.warnings.is_empty(), "{:?}", spec.warnings);
        let st = spec.steering.unwrap();
        assert_eq!(st.direction, "abliterated");
        assert!((st.attn - 1.0).abs() < f32::EPSILON && st.ffn.abs() < f32::EPSILON);
        // `ffn` defaults to 1.0.
        let spec = parse(
            &manifest(r#""recommendedModel":"ds4vision","steering":{"direction":"heretic"}"#),
            root,
        )
        .unwrap();
        assert!((spec.steering.unwrap().ffn - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn steering_without_a_model_or_malformed_warns_and_is_ignored() {
        let root = Path::new("/p");
        for extra in [
            r#""steering":{"direction":"heretic"}"#,
            r#""recommendedModel":"ds4vision","steering":{"ffn":2}"#,
            r#""recommendedModel":"ds4vision","steering":"heretic""#,
        ] {
            let spec = parse(&manifest(extra), root).unwrap();
            assert_eq!(spec.steering, None, "{extra}");
            assert_eq!(spec.warnings.len(), 1, "{extra}: {:?}", spec.warnings);
        }
    }

    #[test]
    fn a_malformed_recommended_model_warns_and_is_ignored() {
        for bad in [r#""""#, "7", "true", r#""Qwen 3""#] {
            let text =
                format!(r#"{{"profile":{{"systemPrompt":"p.md","recommendedModel":{bad}}}}}"#);
            let spec = parse(&text, Path::new("/p")).expect("parses");
            assert_eq!(spec.recommended_model, None, "{bad}");
            assert_eq!(spec.warnings.len(), 1, "{bad}: {:?}", spec.warnings);
            assert!(spec.warnings[0].contains("recommendedModel"), "{bad}");
        }
    }

    #[test]
    fn a_non_boolean_context_flag_warns_and_stays_off() {
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","folderContext":"yes","agentsMd":1}}"#,
            Path::new("/p"),
        )
        .expect("parses");
        assert!(!spec.folder_context && !spec.agents_md);
        assert_eq!(spec.warnings.len(), 2, "{:?}", spec.warnings);
        assert!(spec.warnings[0].contains("folderContext"));
    }

    #[test]
    fn the_context_switches_follow_the_profile_and_stay_on_without_one() {
        assert!(
            folder_context_enabled() && agents_md_enabled(),
            "no profile: unchanged"
        );
        let spec = parse(
            r#"{"profile":{"systemPrompt":"p.md","agentsMd":true}}"#,
            Path::new("/p"),
        )
        .expect("parses");
        let _guard = TestProfileGuard::install(spec);
        assert!(!folder_context_enabled());
        assert!(agents_md_enabled());
    }

    #[test]
    fn a_prompt_without_the_protocol_token_warns_once_by_name() {
        let w = missing_protocol_warning("chatbgt", "You are ChatBGT.\n")
            .expect("no token, so a warning");
        assert!(w.contains("\"chatbgt\""), "{w}");
        assert!(w.contains(crate::sysprompt::TOOL_PROTOCOL_TOKEN), "{w}");
        let with = format!(
            "You are ChatBGT.\n\n{}\n",
            crate::sysprompt::TOOL_PROTOCOL_TOKEN
        );
        assert_eq!(missing_protocol_warning("chatbgt", &with), None);
    }

    const FULL: &str = r#"{
      "name": "hal",
      "profile": {
        "displayName": "HAL",
        "logo": "assets/hal.png",
        "accent": "160",
        "systemPrompt": "prompt.md",
        "tools": { "builtin": ["bash", "ask"] },
        "settings": { "ui": { "showThinking": false } }
      }
    }"#;

    #[test]
    fn parses_every_field_and_resolves_paths_against_root() {
        let root = Path::new("/plugins/hal");
        let spec = parse(FULL, root).expect("has a profile block");
        assert_eq!(spec.display_name.as_deref(), Some("HAL"));
        assert_eq!(
            spec.logo.as_deref(),
            Some(Path::new("/plugins/hal/assets/hal.png"))
        );
        assert_eq!(spec.accent, Some(Accent::Indexed(160)));
        assert_eq!(spec.system_prompt, Path::new("/plugins/hal/prompt.md"));
        assert_eq!(
            spec.builtin_tools.as_deref(),
            Some(&["bash".to_string(), "ask".to_string()][..])
        );
        assert!(spec.settings_json.is_some());
        assert_eq!(spec.warnings, [] as [std::string::String; 0]);
    }

    #[test]
    fn a_profiles_engine_settings_are_dropped_loudly() {
        // Finding 5: `Settings::overlay_from` drops `engine.*` from a
        // profile's settings layer the same way it drops a plugin's, but
        // silently. This is where that becomes audible: one warning per
        // dropped key, mirroring `settings_audit_warnings`'s per-key shape
        // for a plugin's refused sections.
        let text = r#"{
          "profile": {
            "systemPrompt": "prompt.md",
            "settings": { "engine": { "model": "evil.gguf", "threads": 99 }, "ui": { "showThinking": false } }
          }
        }"#;
        let spec = parse(text, Path::new("/p")).expect("still parses as a profile");
        assert!(
            spec.warnings
                .iter()
                .any(|w| w.contains("engine.model") && w.contains("refused")),
            "expected a warning naming engine.model, got: {:?}",
            spec.warnings
        );
        assert!(
            spec.warnings
                .iter()
                .any(|w| w.contains("engine.threads") && w.contains("refused")),
            "expected a warning naming engine.threads, got: {:?}",
            spec.warnings
        );
        // The refusal itself (dropping the key) is `settings.rs`'s job, not
        // this parser's — it still serializes the block as given.
        assert!(spec.settings_json.unwrap().contains("\"engine\""));
    }

    #[test]
    fn settings_without_an_engine_block_warns_about_nothing() {
        let text = r#"{
          "profile": {
            "systemPrompt": "prompt.md",
            "settings": { "ui": { "showThinking": false } }
          }
        }"#;
        let spec = parse(text, Path::new("/p")).expect("still parses as a profile");
        assert!(
            spec.warnings.is_empty(),
            "no engine block, no warnings expected: {:?}",
            spec.warnings
        );
    }

    #[test]
    fn a_manifest_without_a_profile_block_is_not_a_profile() {
        let text = r#"{ "name": "plain", "description": "no profile" }"#;
        assert!(parse(text, Path::new("/p")).is_none());
    }

    #[test]
    fn a_profile_block_without_a_system_prompt_is_not_a_profile() {
        // systemPrompt is the identity; without it the block is a skin, and
        // silently activating a skin over the plank prompt would surprise.
        let text = r#"{ "profile": { "displayName": "X" } }"#;
        assert!(parse(text, Path::new("/p")).is_none());
    }

    #[test]
    fn unparseable_json_is_not_a_profile() {
        assert!(parse("{ not json", Path::new("/p")).is_none());
    }

    #[test]
    fn an_allow_list_entry_matching_no_builtin_warns_but_still_loads() {
        // Finding 3: a name a newer plank added (or a plain typo) must not
        // fail the whole profile — the design spec requires it to run, just
        // with that entry inert, and to say so through the warning channel.
        let text = r#"{
          "profile": {
            "systemPrompt": "prompt.md",
            "tools": { "builtin": ["bash", "not_a_real_tool"] }
          }
        }"#;
        let spec = parse(text, Path::new("/p")).expect("still parses as a profile");
        assert_eq!(
            spec.builtin_tools.as_deref(),
            Some(&["bash".to_string(), "not_a_real_tool".to_string()][..]),
            "the unknown entry stays in the allow-list, it is not dropped"
        );
        assert!(
            spec.warnings.iter().any(|w| w.contains("not_a_real_tool")),
            "expected a warning naming the unknown entry, got: {:?}",
            spec.warnings
        );
        assert!(spec.builtin_enabled("bash"));
    }

    #[test]
    fn a_recognized_native_extra_in_the_allow_list_warns_about_nothing() {
        // Companion: `glob` is a real builtin (a native extra, not one of
        // the twelve C-parsed schemas) so it must not trip the finding-3
        // warning the way a genuinely unknown name does.
        let text = r#"{
          "profile": {
            "systemPrompt": "prompt.md",
            "tools": { "builtin": ["glob", "ask"] }
          }
        }"#;
        let spec = parse(text, Path::new("/p")).expect("still parses as a profile");
        assert!(
            spec.warnings.is_empty(),
            "glob and ask are real builtins; got warnings: {:?}",
            spec.warnings
        );
    }

    #[test]
    fn a_bad_accent_warns_and_leaves_the_default() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "accent": "chartreuse" } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.accent, None);
        assert_eq!(spec.warnings.len(), 1);
        assert!(spec.warnings[0].contains("accent"));
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "futureThing": 3 } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.warnings, [] as [std::string::String; 0]);
    }

    #[test]
    fn a_profile_with_neither_accent_nor_secondary_keeps_the_builtin_ramp() {
        // The guard on plank's own look: nothing declared, nothing generated.
        assert_eq!(shimmer_ramp(None, None), None);
    }

    #[test]
    fn the_word_rests_in_the_secondary_and_the_accent_travels() {
        // The role assignment, pinned: EAP rests cream and its dark red is the
        // highlight sweeping across. Getting this backwards is exactly the
        // mistake this pair of colours cannot reveal by eye in a test.
        let s = shimmer_ramp(
            Some(Accent::Rgb(0x7a, 0x1f, 0x2b)),
            Some(Accent::Rgb(231, 229, 199)),
        )
        .expect("a declared pair generates a shimmer");
        assert_eq!(s.rest, (231, 229, 199), "the word rests in the secondary");
        assert_eq!(
            s.ramp[2],
            crate::anim::cube_index_rgb((0x7a, 0x1f, 0x2b)),
            "the sweep centre is the accent"
        );
    }

    #[test]
    fn an_accent_alone_derives_the_resting_colour() {
        let accent = (0xc0, 0x40, 0x40);
        let derived = shimmer_ramp(Some(Accent::Rgb(accent.0, accent.1, accent.2)), None)
            .expect("an accent alone still generates a shimmer");
        assert_eq!(derived.rest, crate::anim::derived_secondary(accent));
        // The accent is still what the sweep lands on.
        assert_eq!(derived.ramp[2], crate::anim::cube_index_rgb(accent));
        // And it is not the built-in olive, which is the point.
        assert_ne!(derived.ramp, crate::status::SHIMMER_RAMP);
    }

    #[test]
    fn a_secondary_alone_sweeps_toward_the_default_accent() {
        let a = shimmer_ramp(None, Some(Accent::Rgb(0x44, 0x44, 0x44))).expect("shimmer");
        let b = shimmer_ramp(Some(DEFAULT_ACCENT), Some(Accent::Rgb(0x44, 0x44, 0x44)))
            .expect("shimmer");
        assert_eq!(a, b);
        assert_eq!(a.rest, (0x44, 0x44, 0x44));
    }

    #[test]
    fn a_bad_secondary_warns_and_falls_back_to_the_derived_one() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "accent": "160",
                        "secondary": "chartreuse" } }"#;
        let spec = parse(text, Path::new("/tmp")).expect("spec");
        assert_eq!(spec.accent, Some(Accent::Indexed(160)));
        assert_eq!(spec.secondary, None);
        assert!(spec.warnings.iter().any(|w| w.contains("secondary")));
        // The accent still drives a shimmer, so a typo degrades to a derived
        // resting colour rather than back to the built-in olive.
        assert!(shimmer_ramp(spec.accent, spec.secondary).is_some());
    }

    #[test]
    fn secondary_parses_the_two_documented_profile_values() {
        let text = r##"{ "profile": { "systemPrompt": "p.md", "secondary": "#e7e5c7" } }"##;
        let spec = parse(text, Path::new("/tmp")).expect("spec");
        assert_eq!(spec.secondary, Some(Accent::Rgb(231, 229, 199)));

        let text = r##"{ "profile": { "systemPrompt": "p.md", "secondary": "#444444" } }"##;
        let spec = parse(text, Path::new("/tmp")).expect("spec");
        assert_eq!(spec.secondary, Some(Accent::Rgb(68, 68, 68)));
    }

    #[test]
    fn accent_parses_an_ansi_index_and_a_hex_triple() {
        assert_eq!(parse_accent("0"), Some(Accent::Indexed(0)));
        assert_eq!(parse_accent("255"), Some(Accent::Indexed(255)));
        assert_eq!(parse_accent("#c04040"), Some(Accent::Rgb(0xc0, 0x40, 0x40)));
        assert_eq!(parse_accent("#C04040"), Some(Accent::Rgb(0xc0, 0x40, 0x40)));
        assert_eq!(parse_accent("256"), None);
        assert_eq!(parse_accent("#c040"), None);
        assert_eq!(parse_accent(""), None);
    }

    #[test]
    fn an_absent_allow_list_enables_every_builtin() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert!(spec.builtin_enabled("read"));
        assert!(spec.builtin_enabled("bash"));
    }

    #[test]
    fn an_allow_list_enables_only_what_it_names() {
        let spec = parse(FULL, Path::new("/p")).expect("a profile");
        assert!(spec.builtin_enabled("bash"));
        assert!(spec.builtin_enabled("ask"));
        assert!(!spec.builtin_enabled("read"));
    }

    #[test]
    fn an_absolute_path_in_the_manifest_is_used_as_is() {
        let text = r#"{ "profile": { "systemPrompt": "/etc/p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert_eq!(spec.system_prompt, Path::new("/etc/p.md"));
    }

    // FINDING 1: parse_accent must never panic, no matter the byte/char shape
    // of its input.

    #[test]
    fn accent_does_not_panic_on_a_six_byte_non_ascii_string() {
        // "#\u{20ac}000" is a '#' followed by 6 bytes (the euro sign is 3
        // bytes, "000" is 3 more), so the old byte-length check let a
        // char-boundary slice through.
        assert_eq!(parse_accent("#\u{20ac}000"), None);
    }

    #[test]
    fn accent_does_not_panic_on_a_multi_byte_char_of_non_six_byte_length() {
        assert_eq!(parse_accent("#\u{1f600}"), None);
        assert_eq!(parse_accent("#\u{1f600}\u{1f600}"), None);
    }

    #[test]
    fn accent_does_not_panic_on_an_empty_string() {
        assert_eq!(parse_accent(""), None);
        assert_eq!(parse_accent("#"), None);
    }

    // FINDING 2: malformed-but-present optional fields warn instead of
    // silently defaulting; absent fields stay silent.

    #[test]
    fn a_wrong_type_display_name_warns_and_defaults() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "displayName": 3 } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.display_name, None);
        assert!(spec.warnings.iter().any(|w| w.contains("displayName")));
    }

    #[test]
    fn an_absent_display_name_is_silent() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert_eq!(spec.display_name, None);
        assert_eq!(spec.warnings, [] as [std::string::String; 0]);
    }

    #[test]
    fn a_wrong_type_logo_warns_and_defaults() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "logo": 3 } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.logo, None);
        assert!(spec.warnings.iter().any(|w| w.contains("logo")));
    }

    #[test]
    fn an_absent_logo_is_silent() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert_eq!(spec.warnings, [] as [std::string::String; 0]);
    }

    #[test]
    fn a_wrong_type_settings_warns_and_defaults() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "settings": "nope" } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(spec.settings_json, None);
        assert!(spec.warnings.iter().any(|w| w.contains("settings")));
    }

    #[test]
    fn an_absent_settings_is_silent() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert_eq!(spec.warnings, [] as [std::string::String; 0]);
    }

    #[test]
    fn a_wrong_type_tools_builtin_warns_and_fails_closed() {
        let text = r#"{ "profile": { "systemPrompt": "p.md", "tools": { "builtin": "bash" } } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        // Fails closed: an empty allow-list, not None (which would mean
        // "everything allowed") — a malformed restriction must restrict.
        assert_eq!(spec.builtin_tools.as_deref(), Some(&[][..]));
        assert!(!spec.builtin_enabled("bash"));
        assert!(!spec.builtin_enabled("read"));
        assert!(spec.warnings.iter().any(|w| w.contains("tools.builtin")));
    }

    #[test]
    fn an_absent_tools_builtin_is_silent() {
        let text = r#"{ "profile": { "systemPrompt": "p.md" } }"#;
        let spec = parse(text, Path::new("/p")).expect("a profile");
        assert_eq!(spec.builtin_tools, None);
        assert_eq!(spec.warnings, [] as [std::string::String; 0]);
    }

    #[test]
    fn tools_builtin_array_with_bad_elements_warns_and_skips_them() {
        let text = r#"{ "profile": { "systemPrompt": "p.md",
            "tools": { "builtin": ["bash", 3, "", "read"] } } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert_eq!(
            spec.builtin_tools.as_deref(),
            Some(&["bash".to_string(), "read".to_string()][..])
        );
        assert!(spec.warnings.iter().any(|w| w.contains("tools.builtin")));
    }

    // FINDING 3: warnings from independent malformed fields all accumulate.

    #[test]
    fn multiple_malformed_fields_accumulate_warnings_and_parse_still_succeeds() {
        let text = r#"{ "profile": {
            "systemPrompt": "p.md",
            "accent": "chartreuse",
            "displayName": 3,
            "tools": { "builtin": "bash" }
        } }"#;
        let spec = parse(text, Path::new("/p")).expect("still a profile");
        assert!(spec.warnings.iter().any(|w| w.contains("accent")));
        assert!(spec.warnings.iter().any(|w| w.contains("displayName")));
        assert!(spec.warnings.iter().any(|w| w.contains("tools.builtin")));
        assert_eq!(spec.warnings.len(), 3);
    }

    fn spec_named(tools: Option<Vec<String>>) -> ProfileSpec {
        ProfileSpec {
            display_name: Some("HAL".to_string()),
            logo: None,
            accent: None,
            secondary: None,
            system_prompt: PathBuf::from("/p/prompt.md"),
            builtin_tools: tools,
            settings_json: None,
            warnings: Vec::new(),
            folder_context: false,
            agents_md: false,
            recommended_model: None,
            steering: None,
            grids: BTreeMap::new(),
            verbs: None,
        }
    }

    #[test]
    fn the_predicate_follows_the_installed_spec() {
        // Exercised directly on the spec rather than through the global, so it
        // does not depend on test ordering within the process.
        let spec = spec_named(Some(vec!["bash".to_string()]));
        assert!(spec.builtin_enabled("bash"));
        assert!(!spec.builtin_enabled("read"));
    }

    #[test]
    fn no_flag_resolves_to_no_profile() {
        let set = crate::plugins::PluginSet::default();
        assert!(matches!(
            resolve_profile(None, false, &set),
            Resolution::None
        ));
    }

    #[test]
    fn a_bare_flag_lists_the_available_profiles() {
        let set = crate::plugins::PluginSet::default();
        match resolve_profile(Some(""), false, &set) {
            Resolution::List(names) => assert_eq!(names, [] as [std::string::String; 0]),
            other => panic!("expected a listing, got {other:?}"),
        }
    }

    #[test]
    fn an_explicit_empty_name_is_fatal_not_a_listing() {
        // Finding 4: `--profile ""` must not collide with the bare-flag
        // listing sentinel — both parse `requested` to `Some("")`, so only
        // `explicit_empty` tells them apart.
        let set = crate::plugins::PluginSet::default();
        assert!(matches!(
            resolve_profile(Some(""), true, &set),
            Resolution::EmptyName
        ));
    }

    #[test]
    fn an_unknown_name_reports_what_is_available() {
        let set = crate::plugins::PluginSet::default();
        match resolve_profile(Some("nope"), false, &set) {
            Resolution::NoSuchPlugin(name, available) => {
                assert_eq!(name, "nope");
                assert_eq!(available, [] as [std::string::String; 0]);
            }
            other => panic!("expected NoSuchPlugin, got {other:?}"),
        }
    }

    #[test]
    fn a_plugin_without_a_profile_block_is_refused_by_name() {
        let (dir, set) = set_with_plugin("plain", r#"{"name":"plain"}"#);
        let _ = &dir;
        match resolve_profile(Some("plain"), false, &set) {
            Resolution::NotAProfile(name) => assert_eq!(name, "plain"),
            other => panic!("expected NotAProfile, got {other:?}"),
        }
    }

    #[test]
    fn a_profile_bearing_plugin_activates() {
        let (dir, set) = set_with_plugin(
            "hal",
            r#"{"name":"hal","profile":{"systemPrompt":"prompt.md","displayName":"HAL"}}"#,
        );
        let _ = &dir;
        match resolve_profile(Some("hal"), false, &set) {
            Resolution::Activate(a) => {
                assert_eq!(a.name, "hal");
                assert_eq!(a.spec.display_name.as_deref(), Some("HAL"));
            }
            other => panic!("expected Activate, got {other:?}"),
        }
    }

    /// A one-plugin `PluginSet` on disk, in a unique scratch directory under
    /// the OS temp dir (mirroring `src/plugins.rs`'s own test helper, since
    /// this project has no `tempfile` dev-dependency). The directory is
    /// returned so the caller can keep it alive for the duration of the
    /// assertion, though nothing here deletes it early.
    fn set_with_plugin(name: &str, manifest: &str) -> (PathBuf, crate::plugins::PluginSet) {
        let base = std::env::temp_dir().join(format!(
            "plank-profile-resolve-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join(name);
        std::fs::create_dir_all(dir.join(".plank-plugin")).expect("mkdir");
        std::fs::write(dir.join(".plank-plugin").join("plugin.json"), manifest).expect("write");
        std::fs::write(dir.join("prompt.md"), "You are a test agent.\n").expect("write prompt");
        let set = crate::plugins::load_in_with(None, &base, std::slice::from_ref(&dir));
        (base, set)
    }

    /// The shipped `chatbgt` profile must stay parseable and warning-free. A
    /// typo in a builtin name or an `engine.*` key in its settings would
    /// otherwise only be noticed by someone running it.
    #[test]
    fn the_shipped_chatbgt_profile_parses_without_warnings() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/profiles/chatbgt");
        let manifest = root.join(".plank-plugin").join("plugin.json");
        let text = std::fs::read_to_string(&manifest).expect("chatbgt manifest is readable");
        let spec = parse(&text, &root).expect("chatbgt declares a profile");
        assert_eq!(spec.display_name.as_deref(), Some("ChatBGT"));
        assert!(spec.system_prompt.is_file(), "prompt file is missing");
        assert!(spec.logo.is_none(), "chatbgt exercises the logo fallback");
        assert!(spec.warnings.is_empty(), "{:?}", spec.warnings);
    }
}
