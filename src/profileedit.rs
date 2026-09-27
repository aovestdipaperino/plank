// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `/edit-profile`: the running profile's files in one editable buffer.
//!
//! The same shape as `/memory` (`memory::combine`/`memory::apply`): a header
//! comment, then one marked section per file, split back on save. The header
//! says where every profile field comes from, so the user can see what the
//! manifest sets and what falls back to a default before touching anything.
//!
//! Saving is all-or-nothing on validation: the edited manifest must still
//! parse into a `profile` block with a `systemPrompt`, and the prompt must not
//! be empty, before a single byte is written. That is the same gate
//! `/install-profile` applies, so an edit cannot leave on disk a profile the
//! next launch would refuse.
//!
//! A saved change takes effect only at the next launch (the profile is set
//! once at startup), so the front end offers a restart; [`restart_args`] and
//! [`exec_restart`] turn that into plank re-executing itself on the same
//! session. Everything here is pure except the file writes and the `exec`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::plugins::{Origin, Plugin};
use crate::profile::{Accent, ProfileSpec};

/// One editable file of a profile, as its section is named in the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The `plugin.json` the loader read, in either spelling.
    Manifest,
    /// The `systemPrompt` file.
    Prompt,
    /// The plugin root's `settings.json`.
    Settings,
    /// The plugin root's `.mcp.json`.
    Mcp,
}

impl Kind {
    /// The name used in the section markers.
    #[must_use]
    pub fn marker_name(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::Prompt => "prompt",
            Self::Settings => "settings",
            Self::Mcp => "mcp",
        }
    }

    /// The inverse of [`Kind::marker_name`].
    #[must_use]
    pub fn from_marker_name(name: &str) -> Option<Self> {
        match name {
            "manifest" => Some(Self::Manifest),
            "prompt" => Some(Self::Prompt),
            "settings" => Some(Self::Settings),
            "mcp" => Some(Self::Mcp),
            _ => None,
        }
    }
}

/// A section of the buffer and the file it is read from and written to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Which piece of the profile this is.
    pub kind: Kind,
    /// Computed from the profile, never from the buffer.
    pub path: PathBuf,
}

const BEGIN: &str = "<!-- plank-profile: begin ";
const END: &str = "<!-- plank-profile: end ";
const HEADER: &str = "<!-- plank profile ";

/// The files the buffer shows, in order: the manifest and the prompt always,
/// `settings.json` and `.mcp.json` only when they exist, so saving never
/// creates a file the profile did not already have.
#[must_use]
pub fn sections(plugin: &Plugin, spec: &ProfileSpec) -> Vec<Section> {
    let mut out = Vec::new();
    if let Some(path) = crate::plugins::manifest_path(&plugin.root) {
        out.push(Section {
            kind: Kind::Manifest,
            path,
        });
    }
    out.push(Section {
        kind: Kind::Prompt,
        path: spec.system_prompt.clone(),
    });
    for (kind, name) in [(Kind::Settings, "settings.json"), (Kind::Mcp, ".mcp.json")] {
        let path = plugin.root.join(name);
        if path.is_file() {
            out.push(Section { kind, path });
        }
    }
    out
}

/// The profile as the manifest on disk describes it now, or `fallback` (the
/// spec activated at startup) when the manifest no longer parses.
///
/// Read fresh so a second `/edit-profile` in the same session shows what the
/// first one wrote, not what plank started with.
#[must_use]
pub fn current_spec(plugin: &Plugin, fallback: &ProfileSpec) -> ProfileSpec {
    crate::plugins::manifest_path(&plugin.root)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|text| crate::profile::parse(&text, &plugin.root))
        .unwrap_or_else(|| fallback.clone())
}

/// The header comment: where the profile was loaded from, then one row per
/// field with its value and its source.
fn header(plugin: &Plugin, spec: &ProfileSpec) -> String {
    let mut out = format!(
        "{HEADER}\"{}\", loaded from {} {}\n",
        plugin.name,
        plugin.origin.label(),
        plugin.root.display()
    );
    if plugin.origin == Origin::Profile {
        out.push_str(
            "     This is the installed copy under ~/.plank/profiles, not the source it was installed from.\n",
        );
    }
    let accent = spec.accent.map(|a| match a {
        Accent::Indexed(i) => i.to_string(),
        Accent::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
    });
    let rows: [(&str, String, &str); 6] = [
        match &spec.display_name {
            Some(d) => ("displayName", d.clone(), "manifest"),
            None => (
                "displayName",
                plugin.name.clone(),
                "default (the plugin name)",
            ),
        },
        match accent {
            Some(a) => ("accent", a, "manifest"),
            None => ("accent", "114".to_owned(), "default (plank green)"),
        },
        match &spec.logo {
            Some(p) => (
                "logo",
                short(p, &plugin.root),
                "manifest (a PNG: replace the file to change it)",
            ),
            None => ("logo", "plank's logo".to_owned(), "default"),
        },
        (
            "systemPrompt",
            short(&spec.system_prompt, &plugin.root),
            "manifest",
        ),
        match &spec.builtin_tools {
            Some(t) if t.is_empty() => ("tools", "none".to_owned(), "manifest (tools.builtin)"),
            Some(t) => ("tools", t.join(" "), "manifest (tools.builtin)"),
            None => ("tools", "all builtins".to_owned(), "default"),
        },
        match spec.settings_json.as_deref().map(settings_keys) {
            Some(keys) if !keys.is_empty() => {
                ("settings", keys.join(", "), "manifest (profile.settings)")
            }
            _ => ("settings", "none".to_owned(), "default"),
        },
    ];
    for (field, value, source) in rows {
        let _ = writeln!(out, "     {field:<12} {value:<28} {source}");
    }
    out.push_str("     Edit inside the sections; save to write, quit to discard. -->\n");
    out
}

/// `path` relative to the plugin `root` when it lies inside it, else whole:
/// the header's first line already names the root, and a full path would
/// push the source column off the screen.
fn short(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// The dotted keys a settings object sets, leaves only, in document order.
fn settings_keys(json: &str) -> Vec<String> {
    fn walk(prefix: &str, value: &crate::tools::mcp::Json, out: &mut Vec<String>) {
        match value {
            crate::tools::mcp::Json::Obj(members) if !members.is_empty() => {
                for (k, v) in members {
                    let key = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    walk(&key, v, out);
                }
            }
            _ if !prefix.is_empty() => out.push(prefix.to_owned()),
            _ => {}
        }
    }
    let mut out = Vec::new();
    if let Some(root) = crate::tools::mcp::json_parse(json) {
        walk("", &root, &mut out);
    }
    out
}

/// The whole buffer: [`header`], then each of [`sections`] between markers.
///
/// # Errors
///
/// A file that exists but cannot be read. Rendering it as empty would let a
/// save write the editor's blank over contents that were never loaded.
pub fn render(plugin: &Plugin, spec: &ProfileSpec) -> Result<String, String> {
    let mut out = header(plugin, spec);
    for s in sections(plugin, spec) {
        let body = std::fs::read_to_string(&s.path)
            .map_err(|e| format!("cannot read {}: {e}", s.path.display()))?;
        let _ = writeln!(
            out,
            "\n{BEGIN}{} {} -->",
            s.kind.marker_name(),
            s.path.display()
        );
        out.push_str(body.trim_end_matches('\n'));
        if !body.trim_end_matches('\n').is_empty() {
            out.push('\n');
        }
        let _ = writeln!(out, "{END}{} -->", s.kind.marker_name());
    }
    Ok(out)
}

/// Splits an edited buffer into `(kind, body)` pairs.
///
/// Stricter than `memory::split`: a profile file is structured text, so
/// anything typed outside a section, a duplicated or unknown section, or an
/// unclosed one is an error rather than something to guess a home for.
fn split(edited: &str) -> Result<Vec<(Kind, String)>, String> {
    let mut sections: Vec<(Kind, String)> = Vec::new();
    let mut open: Option<Kind> = None;
    let mut in_header = false;
    for line in edited.lines() {
        let t = line.trim();
        if open.is_none() && (in_header || t.starts_with(HEADER)) {
            in_header = !t.ends_with("-->");
            continue;
        }
        if let Some(rest) = t.strip_prefix(BEGIN) {
            let name = rest.split_whitespace().next().unwrap_or("");
            if let Some(k) = open {
                return Err(format!(
                    "section {} begins inside section {}",
                    name,
                    k.marker_name()
                ));
            }
            let kind =
                Kind::from_marker_name(name).ok_or_else(|| format!("unknown section {name:?}"))?;
            if sections.iter().any(|(k, _)| *k == kind) {
                return Err(format!("section {name} appears twice"));
            }
            sections.push((kind, String::new()));
            open = Some(kind);
            continue;
        }
        if let Some(rest) = t.strip_prefix(END) {
            let name = rest.split_whitespace().next().unwrap_or("");
            match open {
                Some(k) if k.marker_name() == name => open = None,
                Some(k) => {
                    return Err(format!(
                        "section {} is closed by an end marker for {name}",
                        k.marker_name()
                    ));
                }
                None => return Err(format!("end marker for {name} with no section open")),
            }
            continue;
        }
        match (open, sections.last_mut()) {
            (Some(_), Some((_, body))) => {
                body.push_str(line);
                body.push('\n');
            }
            _ if t.is_empty() => {}
            _ => return Err(format!("text outside a section: {t:?}")),
        }
    }
    if let Some(k) = open {
        return Err(format!("section {} is never closed", k.marker_name()));
    }
    Ok(sections)
}

/// What [`apply`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applied {
    /// Whether any file was written.
    pub changed: bool,
    /// One line per file, for the log.
    pub lines: Vec<String>,
}

/// Writes an edited buffer back to the profile's files.
///
/// The whole buffer is validated before anything is written, and only files
/// whose text changed are rewritten, each atomically. The path in a begin
/// marker is ignored: every section goes to the path [`sections`] computes,
/// so a hand-edited marker cannot redirect a write.
///
/// # Errors
///
/// Markup errors, a missing or unexpected section, a manifest that no longer
/// parses into a `profile` block with a `systemPrompt`, an empty prompt, an
/// unreadable file, or the first failed write.
pub fn apply(plugin: &Plugin, spec: &ProfileSpec, edited: &str) -> Result<Applied, String> {
    let expected = sections(plugin, spec);
    let parsed = split(edited)?;
    let body = |kind: Kind| {
        parsed
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, b)| b.as_str())
    };
    for s in &expected {
        if body(s.kind).is_none() {
            return Err(format!("section {} is missing", s.kind.marker_name()));
        }
    }
    if let Some((k, _)) = parsed
        .iter()
        .find(|(k, _)| !expected.iter().any(|s| s.kind == *k))
    {
        return Err(format!("section {} has no file behind it", k.marker_name()));
    }
    let new_spec = match body(Kind::Manifest) {
        Some(text) => Some(crate::profile::parse(text, &plugin.root).ok_or(
            "the manifest must be valid JSON with a profile block that names a systemPrompt",
        )?),
        None => None,
    };
    if body(Kind::Prompt).is_some_and(|p| p.trim().is_empty()) {
        return Err("the prompt is empty".to_owned());
    }
    let mut applied = Applied {
        changed: false,
        lines: Vec::new(),
    };
    for s in &expected {
        let new = body(s.kind).unwrap_or_default();
        let current = std::fs::read_to_string(&s.path)
            .map_err(|e| format!("cannot read {}: {e}", s.path.display()))?;
        if current.trim_end_matches('\n') == new.trim_end_matches('\n') {
            applied
                .lines
                .push(format!("{}: unchanged", s.path.display()));
            continue;
        }
        crate::memory::write_atomic(&s.path, new.as_bytes())
            .map_err(|e| format!("cannot write {}: {e}", s.path.display()))?;
        applied.changed = true;
        applied.lines.push(format!("wrote {}", s.path.display()));
    }
    if let Some(new_spec) = new_spec.filter(|n| n.system_prompt != spec.system_prompt) {
        applied.lines.push(format!(
            "systemPrompt now names {}; the prompt section was written to {}",
            new_spec.system_prompt.display(),
            spec.system_prompt.display()
        ));
    }
    Ok(applied)
}

/// The directory plank was launched from and its arguments, recorded before
/// `--chdir` or `--worktree` moves the process.
#[derive(Debug, Clone)]
struct Launch {
    dir: PathBuf,
    args: Vec<String>,
}

static LAUNCH: OnceLock<Launch> = OnceLock::new();

/// Records the launch directory and arguments (without `argv[0]`). Call once,
/// first thing in `main`; later calls are ignored.
pub fn record_launch(dir: PathBuf, args: Vec<String>) {
    let _ = LAUNCH.set(Launch { dir, args });
}

/// A restart the user asked for: the saved session to resume and the
/// directory it was running in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Restart {
    /// The id `plank /resume` takes.
    pub session: String,
    /// The session's working directory, which a worktree may have moved.
    pub cwd: PathBuf,
}

static PENDING: std::sync::Mutex<Option<Restart>> = std::sync::Mutex::new(None);

/// Records that the user chose to restart into `restart`. The TUI then quits
/// normally; `ui::run_interactive` collects it with [`take_restart`] once the
/// agent has been dropped.
pub fn request_restart(restart: Restart) {
    if let Ok(mut slot) = PENDING.lock() {
        *slot = Some(restart);
    }
}

/// Takes the restart [`request_restart`] recorded, if any.
#[must_use]
pub fn take_restart() -> Option<Restart> {
    PENDING.lock().ok().and_then(|mut slot| slot.take())
}

/// The arguments that restart plank on `session`.
///
/// `original` is the launch argument list. One-shot or directory-moving
/// options are dropped (`/resume`, `--worktree`, `--worktree-pr`, `-p`,
/// `--prompt`, `--chdir`), and `--chdir <session_cwd>` plus `/resume
/// <session>` are appended. A relative `--plugin-dir` was resolved after the
/// original `--chdir`, so it is made absolute against that directory (or the
/// launch directory when there was none) before the new `--chdir` could
/// change what it means. Everything else is kept in order.
#[must_use]
pub fn restart_args(
    original: &[String],
    session: &str,
    launch_dir: &Path,
    session_cwd: &Path,
) -> Vec<String> {
    let old_chdir = original
        .iter()
        .position(|a| a == "--chdir")
        .and_then(|i| original.get(i + 1));
    let base = old_chdir.map_or_else(|| launch_dir.to_path_buf(), |d| launch_dir.join(d));
    let mut out = Vec::new();
    let mut i = 0;
    while i < original.len() {
        let arg = original[i].as_str();
        match arg {
            "/resume" => {
                // Mirrors the parser: the next token is the prefix unless it
                // is a flag.
                if original.get(i + 1).is_some_and(|n| !n.starts_with('-')) {
                    i += 1;
                }
            }
            "--worktree" | "--worktree-pr" | "-p" | "--prompt" | "--chdir" => i += 1,
            "--plugin-dir" => {
                out.push(arg.to_owned());
                if let Some(v) = original.get(i + 1) {
                    out.push(base.join(v).to_string_lossy().into_owned());
                    i += 1;
                }
            }
            _ => out.push(arg.to_owned()),
        }
        i += 1;
    }
    out.push("--chdir".to_owned());
    out.push(session_cwd.to_string_lossy().into_owned());
    out.push("/resume".to_owned());
    out.push(session.to_owned());
    out
}

/// Re-executes plank on `restart`. Returns only on failure, with the reason.
///
/// Called after the TUI has quit normally, so the terminal is restored, the
/// session saved, and the `Agent` dropped (which stops MCP servers and bash
/// jobs: `exec` would not). The engine's instance lock is on a descriptor it
/// marks close-on-exec, so the new image can take it.
#[must_use]
pub fn exec_restart(restart: &Restart) -> String {
    use std::os::unix::process::CommandExt as _;
    let Some(launch) = LAUNCH.get() else {
        return "the launch arguments were not recorded".to_owned();
    };
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => return format!("cannot find the plank executable: {e}"),
    };
    if let Err(e) = std::env::set_current_dir(&launch.dir) {
        return format!("cannot return to {}: {e}", launch.dir.display());
    }
    let args = restart_args(&launch.args, &restart.session, &launch.dir, &restart.cwd);
    std::process::Command::new(exe)
        .args(args)
        .exec()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("plank-profileedit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn write(dir: &Path, rel: &str, text: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(path, text).expect("write");
    }

    const MANIFEST: &str = r##"{
  "name": "hal",
  "profile": {
    "displayName": "HAL",
    "accent": "#d0021b",
    "systemPrompt": "prompt.md",
    "tools": { "builtin": ["read", "ask"] },
    "settings": { "ui": { "showThinking": false } }
  }
}
"##;

    /// A profile plugin on disk, loaded the way the plugin scan loads one.
    fn profile(name: &str, manifest: &str, origin: Origin) -> (Plugin, ProfileSpec) {
        let dir = scratch(name).join("hal");
        write(&dir, ".plank-plugin/plugin.json", manifest);
        write(&dir, "prompt.md", "You are HAL.\n");
        let plugin = crate::plugins::load_plugin(&dir, origin).expect("loads");
        let spec = plugin.profile.clone().expect("has a profile");
        (plugin, spec)
    }

    #[test]
    fn the_header_says_where_each_field_comes_from() {
        let (plugin, spec) = profile("header", MANIFEST, Origin::CliDir);
        let text = render(&plugin, &spec).expect("renders");
        let row = |field: &str| {
            text.lines()
                .find(|l| l.trim_start().starts_with(field))
                .unwrap_or_else(|| panic!("no {field} row in:\n{text}"))
                .to_owned()
        };
        assert!(text.starts_with("<!-- plank profile \"hal\", loaded from --plugin-dir "));
        assert!(row("displayName").contains("HAL") && row("displayName").ends_with("manifest"));
        assert!(row("accent").contains("#d0021b"));
        assert!(row("logo").contains("plank's logo") && row("logo").ends_with("default"));
        assert!(row("tools").contains("read ask"));
        assert!(row("systemPrompt").contains(" prompt.md "), "{text}");
        assert!(row("settings").contains("ui.showThinking"));
        assert!(!text.contains("installed copy"));
    }

    #[test]
    fn unset_fields_show_their_defaults() {
        let (plugin, spec) = profile(
            "defaults",
            r#"{"name":"hal","profile":{"systemPrompt":"prompt.md"}}"#,
            Origin::CliDir,
        );
        let text = render(&plugin, &spec).expect("renders");
        assert!(text.contains("default (the plugin name)"), "{text}");
        assert!(text.contains("default (plank green)"), "{text}");
        assert!(text.contains("all builtins"), "{text}");
    }

    #[test]
    fn an_installed_profile_says_it_is_the_installed_copy() {
        let (plugin, spec) = profile("installed", MANIFEST, Origin::Profile);
        assert!(
            render(&plugin, &spec)
                .expect("renders")
                .contains("installed copy")
        );
    }

    #[test]
    fn optional_sections_appear_only_when_their_files_exist() {
        let (plugin, spec) = profile("optional", MANIFEST, Origin::CliDir);
        let kinds = |p: &Plugin| {
            sections(p, &spec)
                .iter()
                .map(|s| s.kind)
                .collect::<Vec<_>>()
        };
        assert_eq!(kinds(&plugin), [Kind::Manifest, Kind::Prompt]);
        write(&plugin.root, ".mcp.json", "{}\n");
        write(&plugin.root, "settings.json", "{}\n");
        assert_eq!(
            kinds(&plugin),
            [Kind::Manifest, Kind::Prompt, Kind::Settings, Kind::Mcp]
        );
    }

    #[test]
    fn an_unedited_buffer_writes_nothing() {
        let (plugin, spec) = profile("roundtrip", MANIFEST, Origin::CliDir);
        let text = render(&plugin, &spec).expect("renders");
        let applied = apply(&plugin, &spec, &text).expect("applies");
        assert!(!applied.changed, "{:?}", applied.lines);
        assert!(applied.lines.iter().all(|l| l.ends_with("unchanged")));
    }

    #[test]
    fn an_edited_prompt_writes_only_the_prompt() {
        let (plugin, spec) = profile("prompt-edit", MANIFEST, Origin::CliDir);
        let manifest = crate::plugins::manifest_path(&plugin.root).expect("manifest");
        let before = std::fs::metadata(&manifest).and_then(|m| m.modified()).ok();
        let text = render(&plugin, &spec)
            .expect("renders")
            .replace("You are HAL.", "You are HAL, terse.");
        let applied = apply(&plugin, &spec, &text).expect("applies");
        assert!(applied.changed);
        assert_eq!(
            std::fs::read_to_string(&spec.system_prompt).expect("read"),
            "You are HAL, terse.\n"
        );
        assert_eq!(
            std::fs::metadata(&manifest).and_then(|m| m.modified()).ok(),
            before
        );
    }

    #[test]
    fn a_hand_edited_marker_path_cannot_redirect_a_write() {
        let (plugin, spec) = profile("redirect", MANIFEST, Origin::CliDir);
        let decoy = plugin.root.join("decoy.md");
        let text = render(&plugin, &spec)
            .expect("renders")
            .replace(
                &format!("begin prompt {}", spec.system_prompt.display()),
                &format!("begin prompt {}", decoy.display()),
            )
            .replace("You are HAL.", "Edited.");
        apply(&plugin, &spec, &text).expect("applies");
        assert!(!decoy.exists());
        assert_eq!(
            std::fs::read_to_string(&spec.system_prompt).expect("read"),
            "Edited.\n"
        );
    }

    #[test]
    fn an_invalid_profile_writes_nothing() {
        let (plugin, spec) = profile("invalid", MANIFEST, Origin::CliDir);
        let text = render(&plugin, &spec).expect("renders");
        let broken_json = text
            .replace("\"name\": \"hal\",", "\"name\": \"hal\"")
            .replace("You are HAL.", "Changed prompt.");
        let no_block = text.replace("\"profile\"", "\"notprofile\"");
        let empty_prompt = text.replace("You are HAL.\n", "\n");
        for edited in [broken_json, no_block, empty_prompt] {
            assert!(apply(&plugin, &spec, &edited).is_err());
        }
        assert_eq!(
            std::fs::read_to_string(&spec.system_prompt).expect("read"),
            "You are HAL.\n"
        );
    }

    #[test]
    fn malformed_markup_is_refused() {
        let (plugin, spec) = profile("markup", MANIFEST, Origin::CliDir);
        let text = render(&plugin, &spec).expect("renders");
        let prompt_start = text
            .find("\n<!-- plank-profile: begin prompt")
            .expect("prompt");
        let missing = text[..prompt_start].to_owned();
        let duplicated = format!("{text}{}", &text[prompt_start..]);
        let stray = format!("{text}\nstray words\n");
        let unclosed = text.replace("<!-- plank-profile: end prompt -->\n", "");
        for edited in [missing, duplicated, stray, unclosed] {
            assert!(apply(&plugin, &spec, &edited).is_err(), "{edited}");
        }
    }

    #[test]
    fn a_requested_restart_is_taken_once() {
        let r = Restart {
            session: "brave-curie".to_owned(),
            cwd: PathBuf::from("/w"),
        };
        request_restart(r.clone());
        assert_eq!(take_restart(), Some(r));
        assert_eq!(take_restart(), None);
    }

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn restart_resumes_the_session_and_drops_one_shot_options() {
        let out = restart_args(
            &args(&[
                "--plugin-dir",
                "examples/profiles/hal",
                "--profile",
                "hal",
                "/resume",
                "old",
                "--worktree",
                "feat",
                "--worktree-pr",
                "12",
                "-p",
                "hello",
                "-c",
                "8192",
            ]),
            "brave-curie",
            Path::new("/launch"),
            Path::new("/launch/.claude/worktrees/feat"),
        );
        assert_eq!(
            out,
            args(&[
                "--plugin-dir",
                "/launch/examples/profiles/hal",
                "--profile",
                "hal",
                "-c",
                "8192",
                "--chdir",
                "/launch/.claude/worktrees/feat",
                "/resume",
                "brave-curie",
            ])
        );
    }

    #[test]
    fn a_bare_resume_is_dropped_without_eating_the_next_flag() {
        let out = restart_args(
            &args(&["/resume", "--profile", "hal"]),
            "id",
            Path::new("/l"),
            Path::new("/l"),
        );
        assert_eq!(
            out,
            args(&["--profile", "hal", "--chdir", "/l", "/resume", "id"])
        );
    }

    #[test]
    fn a_relative_plugin_dir_resolves_against_the_original_chdir() {
        let out = restart_args(
            &args(&["--chdir", "proj", "--plugin-dir", "hal", "--profile", "hal"]),
            "id",
            Path::new("/l"),
            Path::new("/l/proj"),
        );
        assert_eq!(
            out,
            args(&[
                "--plugin-dir",
                "/l/proj/hal",
                "--profile",
                "hal",
                "--chdir",
                "/l/proj",
                "/resume",
                "id",
            ])
        );
        let absolute = restart_args(
            &args(&["--plugin-dir", "/abs/hal"]),
            "id",
            Path::new("/l"),
            Path::new("/l"),
        );
        assert_eq!(absolute[1], "/abs/hal");
    }
}
