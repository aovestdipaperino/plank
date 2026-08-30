// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! File plumbing for the `/open` slash command.
//!
//! Every decision `/open` makes that does *not* need a terminal lives here:
//! which file to open, reading it as text, writing it back without risking the
//! original, and remembering which file the model edited last. `src/miniedit/`
//! stays a string-in/string-out editor and `src/ui.rs` stays a thin glue arm,
//! which is what makes the whole command unit-testable.

use std::path::{Path, PathBuf};

/// What bare `/open` says when nothing has been edited yet.
pub const NO_LAST_EDITED: &str = "no file edited yet this session — usage: /open <path>";

/// The largest file `/open` will load, in bytes.
///
/// `load` runs after the TUI has already been torn down for miniedit to take
/// the terminal, so a file large enough to make the read-into-`String` (and
/// the buffer it seeds) take a visible while looks like a hang with no way to
/// interrupt it — there is no running UI left to show a progress bar or take
/// Esc. 32 MiB comfortably covers any real source file or log a terminal text
/// editor is used on, while refusing the "accidentally opened a database
/// dump" case outright.
pub const MAX_OPEN_FILE_BYTES: u64 = 32 * 1024 * 1024;

/// Picks the file `/open` should edit and checks it is editable.
///
/// `arg` is the slash command's argument, empty for a bare `/open`; `last` is
/// the session's last-edited pointer; relative paths resolve against `cwd`.
///
/// The three refusals — missing, directory, and (in [`load`]) non-UTF-8 — are
/// what keep `/open` from creating a file out of a typo or handing miniedit
/// something its `String` buffer would mangle.
///
/// # Errors
/// Returns a user-facing message when there is no target or the target is not
/// an editable file.
pub fn resolve_open_target(arg: &str, last: Option<&Path>, cwd: &Path) -> Result<PathBuf, String> {
    let arg = arg.trim();
    let path = if arg.is_empty() {
        last.ok_or_else(|| NO_LAST_EDITED.to_string())?
            .to_path_buf()
    } else {
        resolve(arg, cwd)
    };
    let meta = std::fs::metadata(&path).map_err(|_| {
        format!(
            "{} does not exist — /open edits existing files",
            path.display()
        )
    })?;
    if meta.is_dir() {
        return Err(format!("{} is a directory", path.display()));
    }
    Ok(path)
}

/// Joins `path` onto `cwd` unless it is already absolute.
///
/// Mirrors `ToolContext::resolve` (`src/tools/mod.rs:243`) so a `/open`
/// argument and a tool-call path mean the same thing.
fn resolve(path: &str, cwd: &Path) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

/// Reads `path` as text.
///
/// # Errors
/// Returns a user-facing message when the file cannot be read, is larger than
/// [`MAX_OPEN_FILE_BYTES`], or is not valid UTF-8. miniedit's buffer is a
/// `String`, so a binary file is refused rather than silently mangled, and a
/// huge one is refused rather than freezing the (already torn-down) TUI.
pub fn load(path: &Path) -> Result<String, String> {
    // Check the size before reading the bytes: a metadata call is cheap even
    // when the file itself is huge.
    let len = std::fs::metadata(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?
        .len();
    if len > MAX_OPEN_FILE_BYTES {
        return Err(format!(
            "{} is too large to open ({len} bytes, limit {MAX_OPEN_FILE_BYTES})",
            path.display()
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    String::from_utf8(bytes)
        .map_err(|_| format!("{} is not text (not valid UTF-8)", path.display()))
}

/// Writes `text` to `path` without risking the original.
///
/// Writes a sibling temp file and renames it over the target, the same
/// tmp-then-rename shape as `KVCache::persist` (`src/kvcache.rs:129`), so an
/// interrupted or failed write cannot leave a truncated file. The target's
/// permissions are carried over, which matters for the executable scripts a
/// user is likely to `/open`.
///
/// `path` is canonicalized first so a symlink (a real pattern for dotfiles,
/// e.g. `~/.vimrc` pointing into a dotfiles checkout) has its *target*
/// rewritten in place rather than being replaced by a plain file: renaming a
/// temp file over the link itself would sever it. When canonicalization fails
/// (e.g. a dangling symlink) the path is used as given rather than losing the
/// user's edit.
///
/// # Errors
/// Returns a user-facing message when the temp file cannot be written or the
/// rename fails.
pub fn save(path: &Path, text: &str) -> Result<(), String> {
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let dir = target.parent().unwrap_or(Path::new("."));
    let name = target.file_name().map_or_else(
        || std::ffi::OsString::from("open"),
        std::ffi::OsStr::to_os_string,
    );
    // Sibling, not $TMPDIR: `rename` is only atomic within a filesystem.
    let mut tmp_name = name;
    tmp_name.push(format!(".plank-open.{}", std::process::id()));
    let tmp = dir.join(tmp_name);

    std::fs::write(&tmp, text.as_bytes())
        .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    if let Ok(meta) = std::fs::metadata(&target) {
        // Best-effort: a permission we cannot copy is not a reason to lose the
        // user's edit.
        let _ = std::fs::set_permissions(&tmp, meta.permissions());
    }
    if let Err(e) = std::fs::rename(&tmp, &target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("cannot replace {}: {e}", target.display()));
    }
    Ok(())
}

/// Points `last` at the file the just-finished dispatch wrote, if any.
///
/// `written` is `ToolContext::last_written`, set by every file-mutating tool.
/// This deliberately does *not* read the diff previews: creating a new file
/// pushes no preview (the streaming dim preview already showed it), so a
/// preview-driven pointer silently missed exactly the case a bare `/open` is
/// most useful for — "write a summary to status.md", then open it.
///
/// The tool resolves against the cwd at write time, so the stored path is
/// already absolute; that is what keeps it correct across an `EnterWorktree`
/// that moves the cwd mid-session.
pub fn note_written(last: &mut Option<PathBuf>, written: Option<PathBuf>) {
    if let Some(p) = written {
        *last = Some(p);
    }
}

/// Extensions `/open` hands to the browser rather than the text editor.
///
/// Deliberately short: only markup a browser renders as a *document*. An
/// `.svg` or `.json` opened from a coding agent is far more likely to be
/// something the user wants to edit than to look at.
const BROWSER_EXTENSIONS: [&str; 2] = ["html", "htm"];

/// Whether `/open` should show `path` in the browser instead of miniedit.
///
/// Extension-based, and case-insensitive because `.HTML` is still HTML.
#[must_use]
pub fn is_browser_target(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|ext| {
            BROWSER_EXTENSIONS
                .iter()
                .any(|known| ext.eq_ignore_ascii_case(known))
        })
}

/// The platform command that hands a path to the default browser.
///
/// Split out from [`open_in_browser`] so the platform choice is unit-testable
/// without actually launching anything.
#[must_use]
fn browser_command(path: &Path) -> (&'static str, Vec<std::ffi::OsString>) {
    let p = path.as_os_str().to_os_string();
    if cfg!(target_os = "macos") {
        ("open", vec![p])
    } else if cfg!(target_os = "windows") {
        // `start` is a cmd builtin, not an executable; the empty string is the
        // window title `start` would otherwise eat the path as.
        ("cmd", vec!["/C".into(), "start".into(), "".into(), p])
    } else {
        ("xdg-open", vec![p])
    }
}

/// Opens `path` in the default browser.
///
/// The child is spawned and *not* waited on: the launcher exits immediately on
/// macOS but `xdg-open` can outlive the browser it starts, and `/open` must not
/// block the session either way. Output is silenced so a launcher's chatter
/// cannot scribble over the TUI.
///
/// # Errors
/// Returns a user-facing message when the launcher cannot be spawned — the
/// common case being a headless Linux box with no `xdg-open` installed.
pub fn open_in_browser(path: &Path) -> Result<(), String> {
    let (program, args) = browser_command(path);
    std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| ())
        .map_err(|e| {
            format!(
                "cannot open {} in a browser ({program}: {e})",
                path.display()
            )
        })
}

/// The log line for a file handed to the browser.
#[must_use]
pub fn opened_in_browser_message(display: &str) -> String {
    format!("opened {display} in the default browser")
}

/// The log line for a successful write.
#[must_use]
pub fn wrote_message(display: &str, text: &str) -> String {
    let lines = text.lines().count();
    let unit = if lines == 1 { "line" } else { "lines" };
    format!("wrote {display} ({lines} {unit})")
}

/// The log line for a cancel or a no-op accept.
#[must_use]
pub fn unchanged_message(display: &str) -> String {
    format!("{display} unchanged")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-test scratch directory, cleaned up by the caller.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-openfile-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn explicit_relative_path_resolves_against_cwd() {
        let dir = scratch("explicit");
        std::fs::write(dir.join("a.txt"), "hi").unwrap();
        let got = resolve_open_target("a.txt", None, &dir).unwrap();
        assert_eq!(got, dir.join("a.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn explicit_path_wins_over_the_pointer() {
        let dir = scratch("wins");
        std::fs::write(dir.join("a.txt"), "hi").unwrap();
        std::fs::write(dir.join("b.txt"), "yo").unwrap();
        let last = dir.join("b.txt");
        let got = resolve_open_target("a.txt", Some(&last), &dir).unwrap();
        assert_eq!(got, dir.join("a.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bare_open_uses_the_pointer() {
        let dir = scratch("bare");
        let last = dir.join("b.txt");
        std::fs::write(&last, "yo").unwrap();
        let got = resolve_open_target("", Some(&last), &dir).unwrap();
        assert_eq!(got, last);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bare_open_without_a_pointer_explains_itself() {
        let dir = scratch("nopointer");
        let err = resolve_open_target("", None, &dir).unwrap_err();
        assert_eq!(err, NO_LAST_EDITED);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn whitespace_only_arg_counts_as_bare() {
        let dir = scratch("blank");
        let err = resolve_open_target("   ", None, &dir).unwrap_err();
        assert_eq!(err, NO_LAST_EDITED);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_path_is_refused_not_created() {
        let dir = scratch("missing");
        let err = resolve_open_target("nope.txt", None, &dir).unwrap_err();
        assert!(err.contains("nope.txt"), "{err}");
        assert!(err.contains("does not exist"), "{err}");
        // The refusal must not have created it.
        assert!(!dir.join("nope.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn directory_is_refused() {
        let dir = scratch("dir");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let err = resolve_open_target("sub", None, &dir).unwrap_err();
        assert!(err.contains("is a directory"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_refuses_a_file_above_the_size_limit() {
        let dir = scratch("huge");
        let path = dir.join("huge.txt");
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(MAX_OPEN_FILE_BYTES + 1).unwrap();
        drop(f);
        let err = load(&path).unwrap_err();
        assert!(err.contains("too large"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_refuses_non_utf8() {
        let dir = scratch("binary");
        let path = dir.join("b.bin");
        std::fs::write(&path, [0xff, 0xfe, 0x00]).unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.contains("not text"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_round_trips_and_leaves_no_temp_file() {
        let dir = scratch("save");
        let path = dir.join("a.txt");
        std::fs::write(&path, "old\n").unwrap();
        save(&path, "new\ncontent\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new\ncontent\n");
        // A stray sibling temp file would be mistaken for a real file by
        // anything globbing the directory.
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.txt".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn save_through_a_symlink_updates_the_target_and_keeps_the_link() {
        let dir = scratch("symlink");
        let target = dir.join("real.txt");
        std::fs::write(&target, "old\n").unwrap();
        let link = dir.join("link.txt");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        save(&link, "new\n").unwrap();

        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new\n");
        let link_meta = std::fs::symlink_metadata(&link).unwrap();
        assert!(
            link_meta.file_type().is_symlink(),
            "save must not replace the symlink with a regular file"
        );
        assert_eq!(std::fs::read_link(&link).unwrap(), target);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn save_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("perms");
        let path = dir.join("x.sh");
        std::fs::write(&path, "old\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        save(&path, "new\n").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "an executable script must stay executable");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn note_written_takes_the_path_the_tool_resolved() {
        let mut last = None;
        note_written(&mut last, Some(PathBuf::from("/work/b.rs")));
        assert_eq!(last, Some(PathBuf::from("/work/b.rs")));
    }

    #[test]
    fn note_written_follows_a_newly_created_file() {
        // The regression: `write` creating a file pushes no diff card, but the
        // new file is precisely what a bare `/open` should open.
        let mut last = Some(PathBuf::from("/work/old.rs"));
        note_written(&mut last, Some(PathBuf::from("/work/status.md")));
        assert_eq!(last, Some(PathBuf::from("/work/status.md")));
    }

    #[test]
    fn note_written_leaves_the_pointer_alone_when_nothing_was_written() {
        let mut last = Some(PathBuf::from("/work/a.rs"));
        note_written(&mut last, None);
        assert_eq!(last, Some(PathBuf::from("/work/a.rs")));
    }

    #[test]
    fn wrote_message_counts_lines_not_newlines() {
        // "a\nb\n" is two lines, not three: the trailing newline terminates
        // the second line rather than starting a third.
        assert_eq!(wrote_message("a.txt", "a\nb\n"), "wrote a.txt (2 lines)");
        assert_eq!(wrote_message("a.txt", "a\n"), "wrote a.txt (1 line)");
        assert_eq!(wrote_message("a.txt", ""), "wrote a.txt (0 lines)");
    }

    #[test]
    fn html_files_go_to_the_browser() {
        assert!(is_browser_target(Path::new("/work/report.html")));
        assert!(is_browser_target(Path::new("/work/report.htm")));
        // A browser renders it, but from a coding agent it is far more likely
        // to be something the user wants to edit.
        assert!(!is_browser_target(Path::new("/work/logo.svg")));
        assert!(!is_browser_target(Path::new("/work/data.json")));
        assert!(!is_browser_target(Path::new("/work/notes.md")));
        assert!(!is_browser_target(Path::new("/work/src/ui.rs")));
    }

    #[test]
    fn the_extension_match_is_case_insensitive() {
        assert!(is_browser_target(Path::new("/work/REPORT.HTML")));
        assert!(is_browser_target(Path::new("/work/Report.Htm")));
    }

    #[test]
    fn an_extensionless_or_dotfile_name_is_not_html() {
        // `.html` as the whole file name is an extensionless dotfile, not an
        // HTML document; neither is a file that merely contains the word.
        assert!(!is_browser_target(Path::new("/work/.html")));
        assert!(!is_browser_target(Path::new("/work/README")));
        assert!(!is_browser_target(Path::new("/work/html")));
        assert!(!is_browser_target(Path::new("/work/index.html.bak")));
    }

    #[test]
    fn the_browser_command_passes_the_path_as_one_argument() {
        // A path with spaces must survive as a single argv entry rather than
        // being re-split by a shell -- which is why this spawns a program
        // directly instead of going through `sh -c`.
        let path = Path::new("/work/my report.html");
        let (program, args) = browser_command(path);
        assert!(!program.is_empty());
        assert_eq!(
            args.last().map(std::ffi::OsString::as_os_str),
            Some(path.as_os_str())
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_uses_the_open_launcher() {
        let (program, args) = browser_command(Path::new("/work/r.html"));
        assert_eq!(program, "open");
        assert_eq!(args.len(), 1);
    }

    #[test]
    fn opened_in_browser_message_names_the_file() {
        assert_eq!(
            opened_in_browser_message("/work/r.html"),
            "opened /work/r.html in the default browser"
        );
    }

    #[test]
    fn unchanged_message_names_the_file() {
        assert_eq!(unchanged_message("a.txt"), "a.txt unchanged");
    }
}
