// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Durable per-user consent for the web tools.
//!
//! The web approval gate normally asks once per session (see
//! [`tools::web`](crate::tools::web)). When the user answers "Always allow" the
//! choice is recorded here as an empty marker file under `~/.plank`, so future
//! sessions skip the prompt entirely. Deleting the file revokes consent.

use std::path::PathBuf;

/// Marker file name under `~/.plank` recording standing web consent.
const WEB_CONSENT_FILE: &str = "web-consent";

/// Path to the web-consent marker, or `None` when `$HOME` is unset.
#[must_use]
pub fn web_consent_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(web_consent_path_in(home))
}

/// [`web_consent_path`] under an explicit home directory.
///
/// Tests use this rather than pointing `$HOME` at a scratch directory. `HOME`
/// is process-global, so mutating it races every other test in the binary —
/// and `home::plank_home_in` deliberately honours the shared `/Users/.plank`
/// fallback when the root it is handed *is* the live `$HOME`, so a test that
/// sets `HOME` and then resolves this path can land on the user's real
/// consent marker and revoke it.
#[must_use]
pub fn web_consent_path_in(home: impl AsRef<std::path::Path>) -> PathBuf {
    crate::home::plank_home_in(home).join(WEB_CONSENT_FILE)
}

/// True when the user has previously granted standing web consent.
#[must_use]
pub fn web_consent_granted() -> bool {
    web_consent_path().is_some_and(|p| p.exists())
}

/// Records standing web consent so future sessions do not prompt.
///
/// # Errors
/// Returns a message if the marker file could not be created.
pub fn grant_web_consent() -> Result<(), String> {
    let path = web_consent_path().ok_or_else(|| "HOME is not set".to_string())?;
    write_marker(&path)
}

/// The body of [`grant_web_consent`], with the destination already resolved.
fn write_marker(path: &std::path::Path) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
    }
    std::fs::write(path, b"").map_err(|e| format!("failed to write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grant_then_detect_with_scoped_home() {
        // A scratch home, resolved explicitly — never via `$HOME`, which this
        // test used to set. See `web_consent_path_in` for why that was wrong.
        let home = std::env::temp_dir().join(format!("plank-consent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();

        let path = web_consent_path_in(&home);
        assert_eq!(path, home.join(".plank").join(WEB_CONSENT_FILE));
        assert!(!path.exists(), "a fresh home has granted nothing");

        write_marker(&path).unwrap();
        assert!(path.exists(), "the grant is the marker file");

        // Revoked once the marker is removed.
        std::fs::remove_file(&path).unwrap();
        assert!(!path.exists());

        let _ = std::fs::remove_dir_all(&home);
    }

    /// The public pair reads the same marker the explicit-root pair writes:
    /// `web_consent_granted` is exactly "does `web_consent_path` exist", so
    /// the scoped test above covers the real code path and not a parallel one.
    #[test]
    fn granted_is_the_existence_of_the_resolved_marker() {
        let expected = std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .map(web_consent_path_in);
        assert_eq!(web_consent_path(), expected);
        assert_eq!(
            web_consent_granted(),
            web_consent_path().is_some_and(|p| p.exists())
        );
    }
}
