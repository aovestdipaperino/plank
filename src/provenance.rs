// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Provenance of effective configuration: which layer each setting came from.
//!
//! plank layers settings from several sources — built-in defaults, plugin
//! `settings.json` files, `~/.plank/settings.json`, `./.plank/settings.json`,
//! and CLI flags that shadow `engine.*` keys. `/config --resolved` (and the
//! `--dump-config` CLI flag) answer "why is this setting what it is" by
//! reporting, per effective key, the winning origin and every lower layer that
//! also set it (shadowed).
//!
//! The provenance is a side table keyed by setting path (`section.key`, the
//! same addressing `/config` and `configform::FIELDS` use); the hot-path
//! values stay plain fields on `Settings`/`AgentConfig`, so no existing
//! signature changes.

/// Where an effective setting or entry came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    /// Built-in default; no file or flag set it.
    Default,
    /// `~/.plank/settings.json`.
    UserSettings,
    /// `./.plank/settings.json`.
    ProjectSettings,
    /// A plugin's `settings.json` (or a plugin-contributed entry). The string
    /// is the plugin name when known, empty for a generic plugin layer.
    Plugin(String),
    /// A CLI flag.
    Cli,
    /// An environment variable.
    Env,
}

impl Origin {
    /// Human-readable label for the resolved dump.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Origin::Default => "default".to_string(),
            Origin::UserSettings => "~/.plank/settings.json".to_string(),
            Origin::ProjectSettings => "./.plank/settings.json".to_string(),
            Origin::Plugin(name) => {
                if name.is_empty() {
                    "plugin settings".to_string()
                } else {
                    format!("plugin {name}")
                }
            }
            Origin::Cli => "CLI flag".to_string(),
            Origin::Env => "environment".to_string(),
        }
    }
}

/// Provenance of one effective key: the winning origin plus every lower layer
/// that also set it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// The layer that won.
    pub origin: Origin,
    /// Lower layers that also set the key, in increasing precedence order.
    pub shadowed: Vec<Origin>,
}

impl Provenance {
    /// A fresh provenance whose winner is `origin`.
    #[must_use]
    pub fn new(origin: Origin) -> Self {
        Self {
            origin,
            shadowed: Vec::new(),
        }
    }

    /// Records that `origin` set the key. If a different origin already won,
    /// the previous winner is demoted to shadowed. Overlay runs low-to-high,
    /// so the shadowed list accumulates in increasing precedence order.
    pub fn note(&mut self, origin: Origin) {
        if self.origin == origin {
            return;
        }
        if !self.shadowed.contains(&self.origin) {
            self.shadowed.push(self.origin.clone());
        }
        self.origin = origin;
    }
}

/// One bare name claimed by a plugin: which plugin won the bare name (if any)
/// and which plugins lost it, each loser still reachable as `<plugin>:<name>`.
///
/// `winner` is `None` when no plugin holds the bare name — either a non-plugin
/// (user/project) entry holds it, or two plugins collided and both were aliased.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimInfo {
    /// The component the entry belongs to: `skills`, `agents` or `templates`.
    pub component: String,
    /// The bare name that was claimed.
    pub name: String,
    /// The plugin that won the bare name, or `None` when no plugin holds it.
    pub winner: Option<String>,
    /// Plugins that lost the bare name, each still reachable as
    /// `<plugin>:<name>`.
    pub shadowed: Vec<String>,
}

/// Claiming provenance for plugin-contributed skills/templates/agents,
/// populated at session construction. `/config --resolved` reads it to show
/// which plugin won each bare name and which are shadowed.
static CLAIMS: std::sync::RwLock<Vec<ClaimInfo>> = std::sync::RwLock::new(Vec::new());

/// Appends claiming provenance for one component (skills/templates/agents).
pub fn add_claims(claims: Vec<ClaimInfo>) {
    if let Ok(mut slot) = CLAIMS.write() {
        slot.extend(claims);
    }
}

/// The accumulated claiming provenance, for `/config --resolved`.
#[must_use]
pub fn claims() -> Vec<ClaimInfo> {
    CLAIMS.read().map(|slot| slot.clone()).unwrap_or_default()
}

/// Renders the resolved configuration: every effective settings key with its
/// value, the winning origin, and the shadowed candidates beneath it.
///
/// CLI overrides (recorded on `cfg.cli_provenance`) beat the file provenance
/// recorded on `settings.provenance`; anything neither set is a default.
#[must_use]
pub fn render_resolved(
    settings: &crate::settings::Settings,
    cfg: &crate::config::AgentConfig,
) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    // The editable fields, with their effective values.
    let mut covered = std::collections::BTreeSet::new();
    for field in crate::configform::FIELDS {
        let key = format!("{}.{}", field.section, field.key);
        covered.insert(key.clone());
        let value = effective_value(settings, cfg, field.id);
        let (origin, shadowed) = provenance_of(&key, settings, cfg);
        let _ = writeln!(out, "{key} = {value}        <- {}", origin.label());
        for s in shadowed {
            let _ = writeln!(
                out,
                "                              (shadowed: {})",
                s.label()
            );
        }
    }
    // Keys a settings file can set but `/config` cannot edit (kvcache.*,
    // worktree.*, update.check, pluginConfig.*): show the origin, no value.
    for (key, p) in &settings.provenance {
        if covered.contains(key) {
            continue;
        }
        let _ = writeln!(out, "{key}        <- {}", p.origin.label());
        for s in &p.shadowed {
            let _ = writeln!(
                out,
                "                              (shadowed: {})",
                s.label()
            );
        }
    }
    // Plugin-contributed names: which plugin won each bare name, and which are
    // shadowed (still reachable as `<plugin>:<name>`).
    for claim in claims() {
        let _ = writeln!(
            out,
            "{}.{}        <- {}",
            claim.component,
            claim.name,
            claim.winner.as_deref().map_or_else(
                || "no plugin holds the bare name".to_string(),
                |w| format!("plugin {w}")
            )
        );
        for s in &claim.shadowed {
            let _ = writeln!(
                out,
                "                              (shadowed: plugin {s}, still reachable as {s}:{})",
                claim.name
            );
        }
    }
    out
}

/// The winning origin and shadowed layers for one settings key: CLI overrides
/// beat the file provenance; anything neither set is a default.
fn provenance_of(
    key: &str,
    settings: &crate::settings::Settings,
    cfg: &crate::config::AgentConfig,
) -> (Origin, Vec<Origin>) {
    if let Some(o) = cfg.cli_provenance.get(key) {
        let mut shadowed = Vec::new();
        if let Some(p) = settings.provenance.get(key) {
            shadowed.push(p.origin.clone());
            shadowed.extend(p.shadowed.iter().cloned());
        }
        (o.clone(), shadowed)
    } else if let Some(p) = settings.provenance.get(key) {
        (p.origin.clone(), p.shadowed.clone())
    } else {
        (Origin::Default, Vec::new())
    }
}

/// The effective value of a field: for keys a CLI flag can override, the
/// resolved `AgentConfig` value (which applies defaults and flags); for
/// everything else, the `Settings` field as [`configform::display`] renders it.
fn effective_value(
    settings: &crate::settings::Settings,
    cfg: &crate::config::AgentConfig,
    id: crate::configform::FieldId,
) -> String {
    use crate::configform::FieldId;
    match id {
        FieldId::EngineModel => cfg
            .model_path
            .as_ref()
            .map_or_else(|| "(unset)".to_string(), |p| p.display().to_string()),
        FieldId::EngineThreads => {
            if cfg.n_threads == 0 {
                "(unset)".to_string()
            } else {
                cfg.n_threads.to_string()
            }
        }
        FieldId::EngineBackend => cfg.backend.map_or_else(
            || "(unset)".to_string(),
            |b| format!("{b:?}").to_lowercase(),
        ),
        FieldId::EnginePower => {
            if cfg.power_percent == 0 {
                "(unset)".to_string()
            } else {
                cfg.power_percent.to_string()
            }
        }
        FieldId::EngineCtx => cfg.generation.ctx_size.to_string(),
        FieldId::SafetySandbox => match cfg.sandbox_override {
            None => "(default)".to_string(),
            Some(true) => "true".to_string(),
            Some(false) => "false".to_string(),
        },
        FieldId::SafetyBtwSuspend => {
            if cfg.btw.suspend {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        _ => crate::configform::display(settings, id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_demotes_the_previous_winner_in_precedence_order() {
        let mut p = Provenance::new(Origin::Plugin("qa".to_string()));
        p.note(Origin::UserSettings);
        p.note(Origin::ProjectSettings);
        assert_eq!(p.origin, Origin::ProjectSettings);
        assert_eq!(
            p.shadowed,
            vec![Origin::Plugin("qa".to_string()), Origin::UserSettings]
        );
    }

    #[test]
    fn note_is_a_noop_for_the_same_origin() {
        let mut p = Provenance::new(Origin::UserSettings);
        p.note(Origin::UserSettings);
        assert_eq!(p.origin, Origin::UserSettings);
        assert!(p.shadowed.is_empty());
    }

    #[test]
    fn labels_are_human_readable() {
        assert_eq!(Origin::Default.label(), "default");
        assert_eq!(Origin::UserSettings.label(), "~/.plank/settings.json");
        assert_eq!(Origin::ProjectSettings.label(), "./.plank/settings.json");
        assert_eq!(Origin::Plugin("qa".to_string()).label(), "plugin qa");
        assert_eq!(Origin::Plugin(String::new()).label(), "plugin settings");
        assert_eq!(Origin::Cli.label(), "CLI flag");
        assert_eq!(Origin::Env.label(), "environment");
    }

    #[test]
    fn render_resolved_shows_cli_beating_the_file_and_the_shadowed_layers() {
        // A settings file sets `engine.ctx`; a CLI `-c` overrides it. The dump
        // must name the CLI as the winner and the file as shadowed beneath it.
        let mut settings = crate::settings::Settings::default();
        settings.overlay_from(r#"{"engine":{"ctx":262144}}"#, &Origin::UserSettings);
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.ctx_size = 262_144; // what `-c 262144` would set
        cfg.cli_provenance
            .insert("engine.ctx".to_string(), Origin::Cli);
        let out = render_resolved(&settings, &cfg);
        assert!(
            out.contains("engine.ctx = 262144        <- CLI flag"),
            "CLI must win over the file: {out}"
        );
        assert!(
            out.contains("(shadowed: ~/.plank/settings.json)"),
            "the file must be listed as shadowed: {out}"
        );
    }

    #[test]
    fn render_resolved_shows_a_default_when_nothing_set_it() {
        let settings = crate::settings::Settings::default();
        let cfg = crate::config::AgentConfig::default();
        let out = render_resolved(&settings, &cfg);
        assert!(
            out.contains("engine.ctx = 1048576        <- default"),
            "an untouched key must show as default: {out}"
        );
    }

    #[test]
    fn render_resolved_shows_the_claiming_rule() {
        // A bare name won by one plugin, with another shadowed and still
        // reachable as `<plugin>:<name>`.
        add_claims(vec![ClaimInfo {
            component: "skills".to_string(),
            name: "review".to_string(),
            winner: Some("rev".to_string()),
            shadowed: vec!["qa".to_string()],
        }]);
        let settings = crate::settings::Settings::default();
        let cfg = crate::config::AgentConfig::default();
        let out = render_resolved(&settings, &cfg);
        assert!(
            out.contains("skills.review        <- plugin rev"),
            "the winning plugin must be named: {out}"
        );
        assert!(
            out.contains("(shadowed: plugin qa, still reachable as qa:review)"),
            "the loser must be named with its qualified alias: {out}"
        );
    }
}
