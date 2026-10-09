//! Profiles: plugin folders with a `profile` block, installed under
//! `~/.plank/profiles/<name>`.
//!
//! A profile folder holds `.plank-plugin/plugin.json` (or the Claude Code
//! spelling `.claude-plugin/plugin.json`) declaring a `profile` object, plus
//! whatever it names: a system prompt, a logo, skills. Installing copies the
//! folder whole and records where it came from in [`SOURCE_FILE`], which is
//! how plank's `--profile <source>` finds an installed copy again.

use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use crate::Error;

/// The file an install writes into the installed folder, naming its source.
pub const SOURCE_FILE: &str = ".plank-source";

/// The installed-profiles root for the home directory `home`:
/// `<home>/.plank/profiles`, as plank's `profiles::dir` has it.
#[must_use]
pub fn root_in(home: &Path) -> PathBuf {
    home.join(".plank").join("profiles")
}

/// [`root_in`] for `$HOME` (or `.` without one).
#[must_use]
pub fn root() -> PathBuf {
    root_in(&std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from))
}

/// A steering direction and the scales to run it at: what a profile's
/// `steering` block names.
#[derive(Debug, Clone, PartialEq)]
pub struct Steering {
    /// The direction's name (`--dir-steering`), looked up for the model in
    /// `~/.plank/models/vectors.json`.
    pub direction: String,
    /// FFN scale (`--dir-steering-ffn`); 1.0 when the block gives none, like
    /// the C.
    pub ffn: f32,
    /// Attention scale (`--dir-steering-attn`); 0.0 when the block gives none.
    pub attn: f32,
    /// `"from": "user"` (the default) or `"all"`: whether the FFN edit starts
    /// at the user's first message or at the first prompt token. `user`
    /// requires `attn` to be 0, since only the FFN scale can be switched on a
    /// live session, so a block with an attention scale defaults to `all`.
    pub from_user: bool,
}

/// Reads a `steering` block: `{"direction": NAME, "ffn": F, "attn": F,
/// "from": "user"|"all"}`, only `direction` required. The scales are
/// range-checked like the command-line flags; the old `file` key is refused
/// with a pointer to its replacement rather than silently ignored.
///
/// # Errors
/// Describes the first problem found, prefixed `steering:`.
pub fn parse_steering(v: &Value) -> Result<Steering, Error> {
    let bad = |m: String| Error::msg(m);
    if !v.is_object() {
        return Err(bad("steering: must be an object".to_string()));
    }
    if v.get("file").is_some() {
        return Err(bad(
            "steering: `file` was replaced by `direction`, a name stored for the model in \
             ~/.plank/models/vectors.json (`pt vectorize … -n NAME`)"
                .to_string(),
        ));
    }
    let direction = v
        .get("direction")
        .and_then(Value::as_str)
        .filter(|d| !d.trim().is_empty())
        .ok_or_else(|| bad("steering: needs a `direction` name".to_string()))?;
    let scale = |key: &str, default: f32| -> Result<f32, Error> {
        let Some(x) = v.get(key) else {
            return Ok(default);
        };
        let n = x
            .as_f64()
            .ok_or_else(|| bad(format!("steering: `{key}` must be a number")))?;
        if !(-100.0..=100.0).contains(&n) {
            return Err(bad(format!("steering: `{key}` must be within -100..100")));
        }
        #[allow(clippy::cast_possible_truncation, reason = "range-checked above")]
        Ok(n as f32)
    };
    let (ffn, attn) = (scale("ffn", 1.0)?, scale("attn", 0.0)?);
    let from_user = match v.get("from").and_then(Value::as_str) {
        // The default defers the FFN edit, which an attention edit rules out.
        None => attn == 0.0,
        Some("all") => false,
        Some("user") if attn != 0.0 => {
            return Err(bad(
                "steering: `from: user` defers only the FFN edit, so `attn` must be 0".to_string(),
            ));
        }
        Some("user") => true,
        Some(other) => {
            return Err(bad(format!(
                "steering: `from` must be \"all\" or \"user\", not `{other}`"
            )));
        }
    };
    Ok(Steering {
        direction: direction.to_string(),
        ffn,
        attn,
        from_user,
    })
}

/// The manifest of a profile folder, the parts an install cares about.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// The plugin `name`, which is also the installed folder's name.
    pub name: String,
    /// The plugin `version`, as written.
    pub version: Option<String>,
    /// `profile.recommendedModel`.
    pub recommended_model: Option<String>,
    /// `profile.steering`, when present and valid.
    pub steering: Option<Steering>,
}

/// The manifest file of the plugin folder `dir`, either spelling.
#[must_use]
pub fn manifest_path(dir: &Path) -> Option<PathBuf> {
    [".plank-plugin", ".claude-plugin"]
        .iter()
        .map(|d| dir.join(d).join("plugin.json"))
        .find(|p| p.is_file())
}

/// The profile folder a path points into: the folder itself, its manifest
/// directory (`.plank-plugin`), or its `plugin.json`.
#[must_use]
pub fn folder_of(path: &Path) -> Option<PathBuf> {
    let mut dir = if path.is_file() {
        path.parent()?.to_path_buf()
    } else {
        path.to_path_buf()
    };
    if matches!(
        dir.file_name().and_then(|n| n.to_str()),
        Some(".plank-plugin" | ".claude-plugin")
    ) {
        dir = dir.parent()?.to_path_buf();
    }
    manifest_path(&dir).map(|_| dir)
}

/// Reads the manifest of the profile folder `dir`.
///
/// # Errors
/// Fails when there is no manifest, it is not JSON, it declares no `profile`
/// object, or its name could not be a folder name.
pub fn read_manifest(dir: &Path) -> Result<Manifest, Error> {
    let path = manifest_path(dir)
        .ok_or_else(|| Error::msg(format!("{}: no .plank-plugin/plugin.json", dir.display())))?;
    let text = std::fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
    let json: Value =
        serde_json::from_str(&text).map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    let profile = json
        .get("profile")
        .filter(|p| p.is_object())
        .ok_or_else(|| Error::msg(format!("{}: declares no `profile`", path.display())))?;
    let name = json
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| dir.file_name().map(|n| n.to_string_lossy().into_owned()))
        .filter(|n| valid_name(n))
        .ok_or_else(|| {
            Error::msg(format!(
                "{}: the plugin `name` must be one folder name (letters, digits, `-`, `_`, `.`)",
                path.display()
            ))
        })?;
    Ok(Manifest {
        name,
        version: json
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_owned),
        recommended_model: profile
            .get("recommendedModel")
            .and_then(Value::as_str)
            .map(str::to_owned),
        steering: profile.get("steering").and_then(|s| parse_steering(s).ok()),
    })
}

/// Whether `name` is safe to use as one folder under the profiles root.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// What installing a profile folder would do.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// The incoming folder.
    pub source_dir: PathBuf,
    /// Its manifest.
    pub manifest: Manifest,
    /// Where it would be installed.
    pub dest: PathBuf,
    /// How that relates to what is there.
    pub state: State,
}

/// How an incoming profile relates to the installed one.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    /// Nothing of that name is installed.
    New,
    /// An identical copy is installed (the source record aside).
    Unchanged,
    /// A different copy is installed, at this version.
    Differs {
        /// The installed manifest's `version`.
        installed_version: Option<String>,
    },
}

/// Plans installing the profile folder `dir` under `root`, without writing.
///
/// # Errors
/// Fails when `dir` is not a profile folder or holds a symlink (a symlink in
/// a fetched repository could point anywhere on this machine).
pub fn plan(dir: &Path, root: &Path) -> Result<Plan, Error> {
    let manifest = read_manifest(dir)?;
    check_tree(dir)?;
    let dest = root.join(&manifest.name);
    let state = if !dest.exists() {
        State::New
    } else if same_tree(dir, &dest)? {
        State::Unchanged
    } else {
        State::Differs {
            installed_version: read_manifest(&dest).ok().and_then(|m| m.version),
        }
    };
    Ok(Plan {
        source_dir: dir.to_path_buf(),
        manifest,
        dest,
        state,
    })
}

/// Installs a [`plan`]ned profile, replacing an installed copy atomically:
/// the new folder is staged beside the old one and swapped in by rename, and
/// the old one is restored if the swap fails. `source` is written to
/// [`SOURCE_FILE`] when given. An unchanged copy only has its source record
/// refreshed.
///
/// # Errors
/// Fails when the folder cannot be copied or the swap fails.
pub fn install(plan: &Plan, source: Option<&str>) -> Result<(), Error> {
    let record = |dir: &Path| -> Result<(), Error> {
        if let Some(source) = source {
            let file = dir.join(SOURCE_FILE);
            std::fs::write(&file, format!("{source}\n")).map_err(|e| Error::io(&file, e))?;
        }
        Ok(())
    };
    if plan.state == State::Unchanged {
        return record(&plan.dest);
    }
    let root = plan
        .dest
        .parent()
        .ok_or_else(|| Error::msg("the profiles root has no parent"))?;
    std::fs::create_dir_all(root).map_err(|e| Error::io(root, e))?;
    let pid = std::process::id();
    let name = &plan.manifest.name;
    let stage = root.join(format!(".{name}.pt-new-{pid}"));
    let old = root.join(format!(".{name}.pt-old-{pid}"));
    let _ = std::fs::remove_dir_all(&stage);
    let staged = copy_tree(&plan.source_dir, &stage).and_then(|()| record(&stage));
    if let Err(e) = staged {
        let _ = std::fs::remove_dir_all(&stage);
        return Err(e);
    }
    let had_old = plan.dest.exists();
    if had_old {
        std::fs::rename(&plan.dest, &old).map_err(|e| {
            let _ = std::fs::remove_dir_all(&stage);
            Error::io(&plan.dest, e)
        })?;
    }
    if let Err(e) = std::fs::rename(&stage, &plan.dest) {
        if had_old {
            let _ = std::fs::rename(&old, &plan.dest);
        }
        let _ = std::fs::remove_dir_all(&stage);
        return Err(Error::io(&plan.dest, e));
    }
    if had_old {
        let _ = std::fs::remove_dir_all(&old);
    }
    Ok(())
}

/// Whether two folders hold the same files with the same bytes, ignoring
/// [`SOURCE_FILE`] and `.git`.
///
/// # Errors
/// Fails when either folder cannot be read.
pub fn same_tree(a: &Path, b: &Path) -> Result<bool, Error> {
    let (fa, fb) = (files(a)?, files(b)?);
    if fa != fb {
        return Ok(false);
    }
    for rel in &fa {
        let read = |root: &Path| {
            let p = root.join(rel);
            std::fs::read(&p).map_err(|e| Error::io(&p, e))
        };
        if read(a)? != read(b)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Every regular file under `dir`, relative and sorted, skipping `.git` and
/// [`SOURCE_FILE`].
fn files(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Error> {
        for entry in std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
            let entry = entry.map_err(|e| Error::io(dir, e))?;
            let path = entry.path();
            let name = entry.file_name();
            if name == ".git" || (dir == root && name == SOURCE_FILE) {
                continue;
            }
            let kind = entry.file_type().map_err(|e| Error::io(&path, e))?;
            if kind.is_dir() {
                walk(root, &path, out)?;
            } else if kind.is_file() {
                out.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out)?;
    out.sort();
    Ok(out)
}

/// Refuses a folder holding symlinks or anything but files and folders.
fn check_tree(dir: &Path) -> Result<(), Error> {
    for entry in std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let path = entry.path();
        if entry.file_name() == ".git" {
            continue;
        }
        let kind = entry.file_type().map_err(|e| Error::io(&path, e))?;
        if kind.is_dir() {
            check_tree(&path)?;
        } else if !kind.is_file() {
            return Err(Error::msg(format!(
                "{}: only plain files and folders can be installed",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Copies `from` into the new folder `to`, skipping `.git`.
fn copy_tree(from: &Path, to: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(to).map_err(|e| Error::io(to, e))?;
    for rel in files(from)? {
        debug_assert!(rel.components().all(|c| matches!(c, Component::Normal(_))));
        let dst = to.join(&rel);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        std::fs::copy(from.join(&rel), &dst).map_err(|e| Error::io(&dst, e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "plank-lib-profiles-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn profile(dir: &Path, version: &str, prompt: &str) {
        std::fs::create_dir_all(dir.join(".plank-plugin")).unwrap();
        std::fs::write(
            dir.join(".plank-plugin").join("plugin.json"),
            format!(
                r#"{{"name":"evil","version":"{version}","profile":{{"systemPrompt":"prompt.md",
                    "recommendedModel":"ds4vision","steering":{{"direction":"abliterated","ffn":1}}}}}}"#
            ),
        )
        .unwrap();
        std::fs::write(dir.join("prompt.md"), prompt).unwrap();
        std::fs::create_dir_all(dir.join("skills").join("x")).unwrap();
        std::fs::write(dir.join("skills").join("x").join("SKILL.md"), "s").unwrap();
    }

    #[test]
    fn steering_blocks_parse_with_defaults_and_refuse_files() {
        let st = parse_steering(&serde_json::json!({"direction":"h","attn":2})).unwrap();
        assert!((st.ffn - 1.0).abs() < f32::EPSILON);
        assert!(!st.from_user, "an attention scale steers every token");
        for bad in [
            serde_json::json!({"file":"/v.f32"}),
            serde_json::json!({"direction":" "}),
            serde_json::json!({"direction":"h","ffn":500}),
            serde_json::json!({"direction":"h","from":"user","attn":1}),
        ] {
            assert!(parse_steering(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_profile_folder_is_found_from_any_of_its_paths() {
        let dir = scratch("folder");
        let p = dir.join("repo").join("profiles").join("evil");
        profile(&p, "1.0", "be evil");
        for inner in [
            p.clone(),
            p.join(".plank-plugin"),
            p.join(".plank-plugin").join("plugin.json"),
        ] {
            assert_eq!(folder_of(&inner).as_deref(), Some(p.as_path()), "{inner:?}");
        }
        assert_eq!(folder_of(&dir), None);
        let m = read_manifest(&p).unwrap();
        assert_eq!(m.name, "evil");
        assert_eq!(m.recommended_model.as_deref(), Some("ds4vision"));
        assert_eq!(m.steering.unwrap().direction, "abliterated");
    }

    #[test]
    fn installing_is_new_then_unchanged_then_a_replace() {
        let dir = scratch("install");
        let root = dir.join("profiles");
        let src = dir.join("src");
        profile(&src, "1.0", "be evil");

        let p = plan(&src, &root).unwrap();
        assert_eq!(p.state, State::New);
        install(&p, Some("owner/repo:profiles/evil")).unwrap();
        let dest = root.join("evil");
        assert_eq!(
            std::fs::read_to_string(dest.join("prompt.md")).unwrap(),
            "be evil"
        );
        assert_eq!(
            std::fs::read_to_string(dest.join(SOURCE_FILE)).unwrap(),
            "owner/repo:profiles/evil\n"
        );
        assert!(dest.join("skills/x/SKILL.md").is_file());

        // The source record does not make an identical copy differ.
        assert_eq!(plan(&src, &root).unwrap().state, State::Unchanged);

        profile(&src, "2.0", "be worse");
        let p = plan(&src, &root).unwrap();
        assert_eq!(
            p.state,
            State::Differs {
                installed_version: Some("1.0".into())
            }
        );
        install(&p, None).unwrap();
        assert_eq!(
            std::fs::read_to_string(dest.join("prompt.md")).unwrap(),
            "be worse"
        );
        // Nothing staged is left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(leftovers, ["evil"]);
    }

    #[test]
    fn symlinks_and_unsafe_names_are_refused() {
        let dir = scratch("unsafe");
        let src = dir.join("src");
        profile(&src, "1.0", "x");
        std::os::unix::fs::symlink("/etc/passwd", src.join("link")).unwrap();
        assert!(plan(&src, &dir.join("profiles")).is_err());

        let bad = dir.join("bad");
        std::fs::create_dir_all(bad.join(".plank-plugin")).unwrap();
        std::fs::write(
            bad.join(".plank-plugin/plugin.json"),
            r#"{"name":"../x","profile":{}}"#,
        )
        .unwrap();
        assert!(read_manifest(&bad).is_err());
    }
}
