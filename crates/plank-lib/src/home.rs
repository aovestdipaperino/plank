//! Where plank keeps its per-user state.

use std::path::{Path, PathBuf};

/// The machine-wide plank home, used only when the user has no `~/.plank`.
pub const SHARED_HOME: &str = "/Users/.plank";

/// The plank home: `~/.plank`, else [`SHARED_HOME`] when only that exists,
/// else `~/.plank` (or `./.plank` without `HOME`). Mirrors plank's own
/// `home::plank_home`, so both programs agree on which directory they mean.
#[must_use]
pub fn plank_dir() -> PathBuf {
    let Some(home) = std::env::var_os("HOME") else {
        return PathBuf::from(".").join(".plank");
    };
    let primary = PathBuf::from(home).join(".plank");
    let shared = Path::new(SHARED_HOME);
    if !primary.is_dir() && shared.is_dir() {
        shared.to_path_buf()
    } else {
        primary
    }
}
