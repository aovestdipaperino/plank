// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `--dump-profiles` and `--dump-engines`: read-only renderings of the two
//! configurations plank resolves at launch but never otherwise shows.
//!
//! Both are diagnostics in the shape of [`crate::provenance::render_resolved`]
//! (which backs `--dump-config`): they print and exit, start no session, and
//! touch no state. Every entry point takes its root explicitly so the tests
//! run against a scratch directory rather than the user's real `~/.plank`.

use std::fmt::Write as _;
use std::path::Path;

/// The key column's width: the longest key any row prints (`recommendedModel`,
/// 16) plus a two-space gutter, so even that key leaves a gap before its value
/// instead of butting up against the single separating space and reading as
/// misaligned against the shorter keys around it.
const KEY_WIDTH: usize = 18;

/// One `key: value` line, indented under an entry. Keys pad to [`KEY_WIDTH`] so
/// every value starts in the same column.
fn row(out: &mut String, key: &str, value: &str) {
    let _ = writeln!(out, "    {key:<KEY_WIDTH$} {value}");
}

/// Resets SGR attributes after a colored span.
const RESET: &str = "\x1b[0m";

/// The SGR foreground escape for an [`crate::profile::Accent`], so a colored
/// terminal can render the profile's header in its own accent.
fn accent_ansi(a: crate::profile::Accent) -> String {
    match a {
        crate::profile::Accent::Indexed(i) => format!("\x1b[38;5;{i}m"),
        crate::profile::Accent::Rgb(r, g, b) => format!("\x1b[38;2;{r};{g};{b}m"),
    }
}

/// An [`crate::profile::Accent`] as it would be written in a manifest, so the
/// dump reads back as something you could paste into a `profile` block.
fn accent_text(a: crate::profile::Accent) -> String {
    match a {
        crate::profile::Accent::Indexed(i) => i.to_string(),
        crate::profile::Accent::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
    }
}

/// Every directory under `home`'s profiles root, with the fields each one
/// resolves to.
///
/// Reads manifests only: it loads no prompt and activates nothing.
///
/// Deliberately enumerates *directories* rather than calling
/// [`crate::profiles::names`]. That function lists only what
/// [`crate::profile::parse`] accepts, so listing and activation can never
/// disagree — the right rule for choosing a profile, and the wrong one for a
/// diagnostic, where a directory that is silently not a profile is exactly
/// what the user is trying to understand. Such a directory is reported with
/// the reason instead of being skipped.
///
/// `color` enables ANSI: the header (name and version) is painted in the
/// profile's accent when it declares one. The caller passes whether stdout is a
/// terminal, so a redirected dump stays plain text.
#[must_use]
pub fn render_profiles_in(home: &Path, color: bool) -> String {
    let mut out = String::new();
    let dir = crate::profiles::dir(home);
    let _ = writeln!(out, "profiles in {}", dir.display());

    let mut dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        // `.replacing` is `claudeplugin::replace_profile`'s staging area, not
        // a profile. Skipping dot-directories keeps plank's own bookkeeping
        // out of a listing meant to show what you can launch.
        .filter(|p| {
            p.file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|n| !n.starts_with('.'))
        })
        .collect();
    dirs.sort();

    if dirs.is_empty() {
        let _ = writeln!(out, "  (none installed)");
        return out;
    }

    for path in dirs {
        render_one_profile(&mut out, &path, color);
    }
    out
}

/// Renders one profile directory: a blank line, the accent-painted header, and
/// either the field rows or the reason the directory is not an activatable
/// profile. Split from [`render_profiles_in`] so the per-profile body stays one
/// readable unit rather than a loop long enough to lint.
fn render_one_profile(out: &mut String, path: &Path, color: bool) {
    let _ = writeln!(out);
    let name = path
        .file_name()
        .map_or_else(|| "?".to_string(), |n| n.to_string_lossy().into_owned());
    let version = crate::profiles::version_of(path).unwrap_or_else(|| "-".to_string());

    // Parse before printing the header, so it can be painted in the profile's
    // own accent. The manifest is read once; its error cases are still reported
    // below, after the header, exactly as before.
    let manifest = read_manifest(path);
    let spec = manifest
        .as_ref()
        .and_then(|(text, root)| crate::profile::parse(text, root));
    let header = format!("  {name}  v{version}");
    match spec.as_ref().and_then(|s| s.accent).filter(|_| color) {
        Some(accent) => {
            let _ = writeln!(out, "{}{header}{RESET}", accent_ansi(accent));
        }
        None => {
            let _ = writeln!(out, "{header}");
        }
    }

    if manifest.is_none() {
        row(out, "error", "no readable plugin.json");
        return;
    }
    let Some(spec) = spec else {
        // A `profile` block without `systemPrompt` is a skin, not a profile, and
        // is deliberately not activatable. Say so rather than printing nothing.
        row(out, "error", "no profile block, or no systemPrompt");
        return;
    };

    if let Some(d) = &spec.display_name {
        row(out, "displayName", d);
    }
    row(out, "systemPrompt", &spec.system_prompt.display().to_string());
    row(
        out,
        "accent",
        &spec
            .accent
            .map_or_else(|| "(default green)".to_string(), accent_text),
    );
    row(
        out,
        "secondary",
        &spec.secondary.map_or_else(
            || {
                if spec.accent.is_some() {
                    "(derived from accent)".to_string()
                } else {
                    "(built-in ramp)".to_string()
                }
            },
            accent_text,
        ),
    );
    if let Some(l) = &spec.logo {
        row(out, "logo", &l.display().to_string());
    }
    row(out, "folderContext", &spec.folder_context.to_string());
    row(out, "agentsMd", &spec.agents_md.to_string());
    if let Some(m) = &spec.recommended_model {
        row(out, "recommendedModel", m);
    }
    row(
        out,
        "tools.builtin",
        &spec
            .builtin_tools
            .as_ref()
            .map_or_else(|| "(all)".to_string(), |t| t.join(", ")),
    );
    if !spec.grids.is_empty() {
        let grids: Vec<String> = spec
            .grids
            .iter()
            .map(|(server, id)| format!("{server} -> {id}"))
            .collect();
        row(out, "grids", &grids.join(", "));
    }
    if let Some(source) = read_source(path) {
        row(out, "source", &source);
    }
    for w in &spec.warnings {
        row(out, "warning", w);
    }
}

/// A profile directory's manifest text and the root its paths resolve against.
///
/// Locates the manifest with [`crate::plugins::manifest_path`], the same
/// reader discovery uses, so the dump cannot find a file that activation would
/// not, or miss one it would.
fn read_manifest(dir: &Path) -> Option<(String, std::path::PathBuf)> {
    let manifest = crate::plugins::manifest_path(dir)?;
    let text = std::fs::read_to_string(manifest).ok()?;
    Some((text, dir.to_path_buf()))
}

/// The `.plank-source` an install recorded, if any.
fn read_source(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join(".plank-source"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The engine catalog as resolved for this machine: every engine, its roles,
/// and whether each role's file is actually on disk.
///
/// The on-disk column is the point. The catalog says what an engine *is*; only
/// the filesystem says whether it can run, and that gap — a declared engine
/// whose main file was never downloaded — is what `recommendedModel` silently
/// steps around (`engines::choose_with_recommendation_in` never downloads).
#[must_use]
pub fn render_engines_in(root: &Path) -> String {
    let mut warn = Vec::new();
    let catalog = crate::engines::load_in(root, &mut warn);
    render_catalog(root, &catalog, &warn)
}

/// [`render_engines_in`] against an already-loaded catalog, so the tests can
/// hand in one they built rather than writing catalog files to disk.
#[must_use]
pub fn render_catalog(
    root: &Path,
    catalog: &crate::engines::Catalog,
    warnings: &[String],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "engines for {}", root.display());
    let _ = writeln!(out, "  catalog version  {}", catalog.version);
    let _ = writeln!(out, "  default          {}", catalog.default_name());

    if catalog.engines.is_empty() {
        let _ = writeln!(out, "\n  (no engines)");
        return out;
    }

    let default = catalog.default_name().to_string();
    for (name, entry) in &catalog.engines {
        let _ = writeln!(out);
        let marker = if *name == default { "  (default)" } else { "" };
        let version = if entry.version == 0 {
            "local".to_string()
        } else {
            format!("v{}", entry.version)
        };
        let _ = writeln!(out, "  {name}  {version}{marker}");

        for role in crate::engines::ROLES {
            let (path, managed) = if let Some(p) = entry.paths.get(role) {
                (Some(p.clone()), false)
            } else if entry.files.contains_key(role) {
                let id = crate::manifest::EngineId::new(name);
                (
                    id.and_then(|id| crate::manifest::local_path_for_in(root, id, role)),
                    true,
                )
            } else {
                (None, false)
            };
            let Some(path) = path else {
                continue;
            };
            let state = if path.exists() {
                "on disk"
            } else {
                "not downloaded"
            };
            let kind = if managed { "managed" } else { "local" };
            row(
                &mut out,
                role,
                &format!("{} [{kind}, {state}]", path.display()),
            );
            if let Some(url) = entry.path_urls.get(role) {
                row(&mut out, "", &format!("url {url}"));
            }
        }
    }

    for w in warnings {
        let _ = writeln!(out, "\n  warning: {w}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh scratch root for one test.
    ///
    /// Tagged by pid *and* an atomic counter: a name shared between tests has
    /// caused parallel-run flakes in this repo before, and every test here
    /// writes into its root.
    fn temp_root() -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("plank-dump-test-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    /// A profile directory with `manifest` as its `.plank-plugin/plugin.json`.
    fn profile(home: &Path, name: &str, manifest: &str) {
        let dir = crate::profiles::dir(home).join(name).join(".plank-plugin");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("plugin.json"), manifest).expect("write");
    }

    #[test]
    fn an_empty_profiles_dir_says_so_rather_than_printing_nothing() {
        let tmp = temp_root();
        let out = render_profiles_in(tmp.as_path(), false);
        assert!(out.contains("(none installed)"), "{out}");
    }

    #[test]
    fn a_profile_dump_shows_both_shimmer_colors() {
        let tmp = temp_root();
        profile(
            tmp.as_path(),
            "hal",
            r##"{"name":"hal","version":"0.3.3","profile":{
                "displayName":"HAL","systemPrompt":"prompt.md",
                "accent":"#ffffff","secondary":"#d0021b"}}"##,
        );
        let out = render_profiles_in(tmp.as_path(), false);
        assert!(out.contains("hal  v0.3.3"), "{out}");
        assert!(out.contains("HAL"), "{out}");
        assert!(out.contains("#ffffff"), "{out}");
        assert!(out.contains("#d0021b"), "{out}");
    }

    #[test]
    fn the_header_wears_the_accent_only_when_color_is_on() {
        let tmp = temp_root();
        profile(
            tmp.as_path(),
            "red",
            r##"{"name":"red","version":"1.2.3","profile":{
                "systemPrompt":"p.md","accent":"#ff2b2b"}}"##,
        );
        // Colored: the header sits inside the accent's truecolor escape, then a
        // reset. The name and version stay intact inside the span.
        let colored = render_profiles_in(tmp.as_path(), true);
        assert!(
            colored.contains("\x1b[38;2;255;43;43m  red  v1.2.3\x1b[0m"),
            "{colored:?}"
        );
        // Plain: no escapes at all, whatever the accent.
        let plain = render_profiles_in(tmp.as_path(), false);
        assert!(!plain.contains('\x1b'), "{plain:?}");
        assert!(plain.contains("  red  v1.2.3"), "{plain}");
    }

    #[test]
    fn a_header_without_an_accent_is_never_colored() {
        let tmp = temp_root();
        profile(
            tmp.as_path(),
            "plainprofile",
            r#"{"name":"plainprofile","profile":{"systemPrompt":"p.md"}}"#,
        );
        // Color is on, but with no accent declared there is nothing to paint.
        let out = render_profiles_in(tmp.as_path(), true);
        assert!(!out.contains('\x1b'), "{out:?}");
    }

    #[test]
    fn every_value_starts_in_the_same_column_even_after_the_longest_key() {
        let tmp = temp_root();
        profile(
            tmp.as_path(),
            "wide",
            r#"{"name":"wide","profile":{"systemPrompt":"p.md",
                "displayName":"Wide","recommendedModel":"some-engine"}}"#,
        );
        let out = render_profiles_in(tmp.as_path(), false);
        // The column where a value begins: index of the value after the key
        // pad. `recommendedModel` is the longest key any row prints, so if it
        // aligns with a short key like `displayName`, every row does.
        let value_col = |key: &str| {
            let line = out
                .lines()
                .find(|l| l.trim_start().starts_with(key))
                .unwrap_or_else(|| panic!("no {key} row in\n{out}"));
            let after = &line[line.find(key).unwrap() + key.len()..];
            let spaces = after.len() - after.trim_start().len();
            (line.find(key).unwrap() + key.len() + spaces, spaces)
        };
        let (short_col, _) = value_col("displayName");
        let (long_col, long_gutter) = value_col("recommendedModel");
        assert_eq!(short_col, long_col, "values misaligned:\n{out}");
        // The longest key must still leave a gutter, not butt up against its
        // value with the single separating space.
        assert!(long_gutter >= 2, "longest key has no gutter:\n{out}");
    }

    #[test]
    fn a_profile_without_a_secondary_says_where_its_shimmer_comes_from() {
        let tmp = temp_root();
        profile(
            tmp.as_path(),
            "accented",
            r#"{"name":"accented","profile":{"systemPrompt":"p.md","accent":"160"}}"#,
        );
        profile(
            tmp.as_path(),
            "plain",
            r#"{"name":"plain","profile":{"systemPrompt":"p.md"}}"#,
        );
        let out = render_profiles_in(tmp.as_path(), false);
        // The distinction the flag exists to make visible: an accent alone
        // derives its far end, nothing declared keeps the built-in ramp.
        assert!(out.contains("(derived from accent)"), "{out}");
        assert!(out.contains("(built-in ramp)"), "{out}");
    }

    #[test]
    fn plank_s_own_staging_directory_is_not_listed_as_a_profile() {
        let tmp = temp_root();
        std::fs::create_dir_all(crate::profiles::dir(tmp.as_path()).join(".replacing"))
            .expect("mkdir");
        profile(
            tmp.as_path(),
            "real",
            r#"{"name":"real","profile":{"systemPrompt":"p.md"}}"#,
        );
        let out = render_profiles_in(tmp.as_path(), false);
        assert!(out.contains("real"), "{out}");
        assert!(!out.contains(".replacing"), "{out}");
    }

    #[test]
    fn a_skin_without_a_system_prompt_is_reported_not_skipped() {
        let tmp = temp_root();
        profile(
            tmp.as_path(),
            "skin",
            r#"{"name":"skin","profile":{"accent":"160"}}"#,
        );
        let out = render_profiles_in(tmp.as_path(), false);
        assert!(out.contains("skin"), "{out}");
        assert!(
            out.contains("no profile block, or no systemPrompt"),
            "{out}"
        );
    }

    #[test]
    fn a_malformed_color_surfaces_as_a_warning_row() {
        let tmp = temp_root();
        profile(
            tmp.as_path(),
            "typo",
            r#"{"name":"typo","profile":{"systemPrompt":"p.md","secondary":"chartreuse"}}"#,
        );
        let out = render_profiles_in(tmp.as_path(), false);
        assert!(out.contains("warning"), "{out}");
        assert!(out.contains("secondary"), "{out}");
    }

    #[test]
    fn the_engine_dump_marks_the_default_and_whether_files_are_present() {
        let tmp = temp_root();
        let mut warn = Vec::new();
        let catalog = crate::engines::load_in(tmp.as_path(), &mut warn);
        let out = render_catalog(tmp.as_path(), &catalog, &warn);
        let default = catalog.default_name();
        assert!(out.contains("catalog version"), "{out}");
        assert!(out.contains(default), "{out}");
        assert!(out.contains("(default)"), "{out}");
        // Nothing is downloaded into a scratch root, so every managed role
        // must report as absent rather than silently looking installed.
        assert!(out.contains("not downloaded"), "{out}");
        assert!(!out.contains("on disk"), "{out}");
    }

    #[test]
    fn an_empty_catalog_says_so() {
        let tmp = temp_root();
        let catalog = crate::engines::Catalog::default();
        let out = render_catalog(tmp.as_path(), &catalog, &[]);
        assert!(out.contains("(no engines)"), "{out}");
    }
}
