//! Where plank keeps its per-user state.
//!
//! Normally that is `~/.plank`. When a machine has no `~/.plank` but does have
//! a shared [`SHARED_HOME`] directory, plank uses that instead, so several
//! accounts on the same box can share one set of models, sessions and plugins.
//! The shared directory is never *created*: a fresh install with neither
//! directory present still lands in `~/.plank`, so nothing is written outside
//! the user's own home unless an administrator put it there first.
//!
//! Every consumer of the user-scoped plank directory resolves it here, so the
//! containment check, the Seatbelt profile and the model downloader can never
//! disagree about which directory they mean. Project-scoped `./.plank` is a
//! different thing and stays where it is.

use std::path::{Path, PathBuf};

/// The machine-wide plank home, used only when the user has no `~/.plank`.
pub const SHARED_HOME: &str = "/Users/.plank";

/// The plank home for `home`, applying the shared-directory fallback.
///
/// The fallback is considered only when `home` really is this process's
/// `$HOME`; a caller that injects some other root (a test, a worktree copy)
/// always gets `home/.plank` so the machine's own state cannot leak in.
#[must_use]
pub fn plank_home_in(home: impl AsRef<Path>) -> PathBuf {
    let home = home.as_ref();
    resolve(home, Path::new(SHARED_HOME), is_real_home(home))
}

/// The plank home for this process: `~/.plank`, the shared directory, or
/// `./.plank` when `HOME` is unset.
#[must_use]
pub fn plank_home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from(".").join(".plank"), plank_home_in)
}

/// The plank home for this process, or `None` when `HOME` is unset.
#[must_use]
pub fn plank_home_opt() -> Option<PathBuf> {
    std::env::var_os("HOME").map(plank_home_in)
}

/// The resolution itself, with both the shared directory and the "this really
/// is our HOME" answer injected so tests can exercise every branch.
fn resolve(home: &Path, shared: &Path, allow_shared: bool) -> PathBuf {
    let primary = home.join(".plank");
    if primary.is_dir() {
        return primary;
    }
    if allow_shared && shared.is_dir() {
        return shared.to_path_buf();
    }
    primary
}

fn is_real_home(home: &Path) -> bool {
    std::env::var_os("HOME").is_some_and(|h| Path::new(&h) == home)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-home-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir scratch");
        dir
    }

    #[test]
    fn the_users_own_directory_wins() {
        let dir = scratch("own");
        let home = dir.join("home");
        let shared = dir.join("shared");
        std::fs::create_dir_all(home.join(".plank")).expect("mkdir");
        std::fs::create_dir_all(&shared).expect("mkdir");
        assert_eq!(resolve(&home, &shared, true), home.join(".plank"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_shared_directory_stands_in_for_a_missing_one() {
        let dir = scratch("shared");
        let home = dir.join("home");
        let shared = dir.join("shared");
        std::fs::create_dir_all(&home).expect("mkdir");
        std::fs::create_dir_all(&shared).expect("mkdir");
        assert_eq!(resolve(&home, &shared, true), shared);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_injected_root_never_reaches_the_shared_directory() {
        let dir = scratch("injected");
        let home = dir.join("home");
        let shared = dir.join("shared");
        std::fs::create_dir_all(&home).expect("mkdir");
        std::fs::create_dir_all(&shared).expect("mkdir");
        assert_eq!(resolve(&home, &shared, false), home.join(".plank"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn with_neither_present_the_home_directory_is_still_the_answer() {
        let dir = scratch("neither");
        let home = dir.join("home");
        let resolved = resolve(&home, &dir.join("shared"), true);
        assert_eq!(resolved, home.join(".plank"));
        assert!(!resolved.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
