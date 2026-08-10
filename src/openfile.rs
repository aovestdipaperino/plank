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

use crate::tools::diff::EditPreview;

/// What bare `/open` says when nothing has been edited yet.
pub const NO_LAST_EDITED: &str = "no file edited yet this session — usage: /open <path>";

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
/// Returns a user-facing message when the file cannot be read or is not valid
/// UTF-8. miniedit's buffer is a `String`, so a binary file is refused rather
/// than silently mangled.
pub fn load(path: &Path) -> Result<String, String> {
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
/// # Errors
/// Returns a user-facing message when the temp file cannot be written or the
/// rename fails.
pub fn save(path: &Path, text: &str) -> Result<(), String> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().map_or_else(
        || std::ffi::OsString::from("open"),
        std::ffi::OsStr::to_os_string,
    );
    // Sibling, not $TMPDIR: `rename` is only atomic within a filesystem.
    let mut tmp_name = name;
    tmp_name.push(format!(".plank-open.{}", std::process::id()));
    let tmp = dir.join(tmp_name);

    std::fs::write(&tmp, text.as_bytes())
        .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    if let Ok(meta) = std::fs::metadata(path) {
        // Best-effort: a permission we cannot copy is not a reason to lose the
        // user's edit.
        let _ = std::fs::set_permissions(&tmp, meta.permissions());
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("cannot replace {}: {e}", path.display()));
    }
    Ok(())
}

/// Points `last` at the file the just-finished dispatch edited, if any.
///
/// Both `edit` and `write` push an [`EditPreview`], so the previews the UI
/// already drains after every dispatch are a complete record of what changed —
/// no extra plumbing through `ToolContext` is needed. The final preview wins:
/// "the last file edited" is the last one in dispatch order.
///
/// Paths are resolved against `cwd` here rather than stored as the raw relative
/// string, because `EnterWorktree` moves the cwd mid-session.
pub fn note_edited(last: &mut Option<PathBuf>, previews: &[EditPreview], cwd: &Path) {
    if let Some(p) = previews.last() {
        *last = Some(resolve(&p.path, cwd));
    }
}

/// What `/open` should do once the editor hands the terminal back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AfterEdit {
    /// Write nothing: the user cancelled, or accepted without changing a byte.
    Unchanged,
    /// Write this text back to the file.
    Save(String),
}

/// Decides whether an editor session's result needs writing.
///
/// `edited` is what the editor returned — `None` on cancel. Accepting a
/// byte-identical buffer is deliberately *not* a write: rewriting the file
/// would bump its mtime and make every watcher downstream rebuild for nothing.
///
/// Separated from the UI arm so the decision is unit-testable; the arm itself
/// needs a live terminal and cannot be.
#[must_use]
pub fn after_edit(edited: Option<String>, initial: &str) -> AfterEdit {
    match edited {
        Some(text) if text != initial => AfterEdit::Save(text),
        _ => AfterEdit::Unchanged,
    }
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

    /// Builds a preview carrying just the path; the diff rows are irrelevant
    /// to the pointer bookkeeping.
    fn preview(path: &str) -> EditPreview {
        EditPreview {
            path: path.to_string(),
            created: false,
            added: 1,
            removed: 0,
            bytes: None,
            rows: Vec::new(),
        }
    }

    #[test]
    fn note_edited_takes_the_last_preview() {
        let cwd = Path::new("/work");
        let mut last = None;
        note_edited(&mut last, &[preview("a.rs"), preview("b.rs")], cwd);
        assert_eq!(last, Some(PathBuf::from("/work/b.rs")));
    }

    #[test]
    fn note_edited_resolves_relative_paths_eagerly() {
        // `EnterWorktree` moves the cwd mid-session, so a stored relative path
        // would later resolve to the worktree copy — a different file than the
        // one that was edited.
        let mut last = None;
        note_edited(&mut last, &[preview("src/ui.rs")], Path::new("/work"));
        assert_eq!(last, Some(PathBuf::from("/work/src/ui.rs")));
        note_edited(&mut last, &[preview("src/ui.rs")], Path::new("/work/.wt/x"));
        assert_eq!(last, Some(PathBuf::from("/work/.wt/x/src/ui.rs")));
    }

    #[test]
    fn note_edited_keeps_an_absolute_path_as_is() {
        let mut last = None;
        note_edited(&mut last, &[preview("/etc/hosts")], Path::new("/work"));
        assert_eq!(last, Some(PathBuf::from("/etc/hosts")));
    }

    #[test]
    fn note_edited_leaves_the_pointer_alone_when_nothing_changed() {
        let mut last = Some(PathBuf::from("/work/a.rs"));
        note_edited(&mut last, &[], Path::new("/work"));
        assert_eq!(last, Some(PathBuf::from("/work/a.rs")));
    }

    #[test]
    fn cancelling_writes_nothing() {
        assert_eq!(after_edit(None, "hello\n"), AfterEdit::Unchanged);
    }

    #[test]
    fn accepting_without_editing_writes_nothing() {
        // Rewriting a byte-identical file would bump its mtime and make every
        // watcher downstream rebuild for nothing.
        assert_eq!(
            after_edit(Some("hello\n".to_string()), "hello\n"),
            AfterEdit::Unchanged
        );
    }

    /// Regression: file mode inserts a trailing newline the raw file text on
    /// disk may not have, so comparing an accept against the raw `initial`
    /// text made a true no-op accept look like an edit and rewrite the file.
    /// `after_edit` must be compared against the same seeded text the editor
    /// started from, not the raw bytes.
    #[test]
    fn accepting_an_untouched_file_without_a_trailing_newline_writes_nothing() {
        let initial = "hello"; // no trailing newline, as read from disk
        let seed = crate::miniedit::file_seed_text(initial);
        assert_eq!(seed, "hello\n");
        // The editor returns exactly what it was seeded with: the user typed
        // nothing and just pressed Ctrl+S. `tui_open` must compare against
        // the seeded text, not the raw `initial` it read from disk.
        assert_eq!(after_edit(Some(seed.clone()), &seed), AfterEdit::Unchanged);
    }

    #[test]
    fn accepting_an_edit_asks_for_a_save() {
        assert_eq!(
            after_edit(Some("bye\n".to_string()), "hello\n"),
            AfterEdit::Save("bye\n".to_string())
        );
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
    fn unchanged_message_names_the_file() {
        assert_eq!(unchanged_message("a.txt"), "a.txt unchanged");
    }
}
