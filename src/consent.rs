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
    Some(PathBuf::from(home).join(".plank").join(WEB_CONSENT_FILE))
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
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
    }
    std::fs::write(&path, b"").map_err(|e| format!("failed to write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grant_then_detect_with_scoped_home() {
        // Point HOME at a scratch dir so the real consent file is untouched.
        let tmp = std::env::temp_dir().join(format!("plank-consent-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        // SAFETY: single-threaded test; restored before returning.
        let prev = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", &tmp) };

        let path = web_consent_path().unwrap();
        std::fs::remove_file(&path).ok();
        assert!(!web_consent_granted());
        grant_web_consent().unwrap();
        assert!(web_consent_granted());
        assert_eq!(path, tmp.join(".plank").join(WEB_CONSENT_FILE));

        // Revoked once the marker is removed.
        std::fs::remove_file(&path).unwrap();
        assert!(!web_consent_granted());

        match prev {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        std::fs::remove_dir_all(&tmp).ok();
    }
}
