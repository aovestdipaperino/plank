//! `repo:path` references to a file or folder in a repository, and fetching
//! them.
//!
//! The repository is a local checkout (any existing directory), a GitHub
//! `owner/repo` shorthand, or a git URL; the path is relative to its root and
//! may not leave it. A remote is cloned shallowly into a temporary directory
//! that is removed when the [`Checkout`] is dropped.

use std::path::{Component, Path, PathBuf};
use std::process::Command;

use crate::Error;

/// Where a repository lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Repo {
    /// A directory on this machine.
    Local(PathBuf),
    /// A GitHub repository, `owner/repo`.
    GitHub(String),
    /// Any other git remote.
    Url(String),
}

/// A path inside a repository: `repo:path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoPath {
    /// The repository.
    pub repo: Repo,
    /// The path inside it, relative, `/`-separated, never leaving the root.
    pub path: String,
}

impl RepoPath {
    /// Parses `repo:path`. The split is at the last `:`, so a URL's own
    /// colons stay with the repository; a leading `/` on the path is allowed
    /// (`owner/repo:/profiles/evil`).
    ///
    /// # Errors
    /// Fails when there is no `:`, the path is empty or climbs out with `..`,
    /// or the repository is none of the accepted forms.
    pub fn parse(spec: &str) -> Result<Self, Error> {
        let (repo, path) = spec.rsplit_once(':').ok_or_else(|| {
            Error::msg(format!(
                "`{spec}`: expected repo:path, e.g. owner/repo:profiles/evil"
            ))
        })?;
        let path = path.trim().trim_matches('/');
        if path.is_empty()
            || !Path::new(path)
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(Error::msg(format!(
                "`{spec}`: the path must be inside the repository"
            )));
        }
        let local = expand_tilde(repo);
        let repo = if local.is_dir() {
            Repo::Local(local)
        } else if repo.contains("://") || repo.starts_with("git@") {
            Repo::Url(repo.to_owned())
        } else if is_github(repo) {
            Repo::GitHub(repo.trim_end_matches(".git").to_owned())
        } else {
            return Err(Error::msg(format!(
                "`{repo}` is not a local folder, a GitHub owner/repo, or a git URL"
            )));
        };
        Ok(Self {
            repo,
            path: path.to_owned(),
        })
    }

    /// Makes the repository available: a local one as it is, a remote one
    /// cloned with `git clone --depth 1`.
    ///
    /// # Errors
    /// Fails when git cannot run or the clone fails (the message carries
    /// git's own reason).
    pub fn fetch(&self) -> Result<Checkout, Error> {
        let url = match &self.repo {
            Repo::Local(dir) => {
                return Ok(Checkout {
                    root: dir.clone(),
                    temp: None,
                });
            }
            Repo::GitHub(repo) => format!("https://github.com/{repo}.git"),
            Repo::Url(url) => url.clone(),
        };
        let temp = std::env::temp_dir().join(format!(
            "plank-fetch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&temp).map_err(|e| Error::io(&temp, e))?;
        let root = temp.join("repo");
        // A missing or private repository makes git ask for a username;
        // fail instead of waiting on a prompt the user did not expect.
        let out = Command::new("git")
            .env("GIT_TERMINAL_PROMPT", "0")
            .args(["clone", "--quiet", "--depth", "1", "--"])
            .arg(&url)
            .arg(&root)
            .output();
        let fail = |why: String| {
            let _ = std::fs::remove_dir_all(&temp);
            Error::msg(why)
        };
        let out = out.map_err(|e| fail(format!("cannot run git: {e}")))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(fail(format!("cannot clone {url}: {}", stderr.trim())));
        }
        Ok(Checkout {
            root,
            temp: Some(temp),
        })
    }

    /// The source plank records for something installed from `rel` in this
    /// repository, in the form its `--profile` takes: `owner/repo:rel` for
    /// GitHub, the canonical folder for a local repository, `url:rel` else.
    #[must_use]
    pub fn record(&self, rel: &Path) -> String {
        let rel = rel.to_string_lossy().replace('\\', "/");
        match &self.repo {
            Repo::GitHub(repo) => format!("{repo}:{rel}"),
            Repo::Url(url) => format!("{url}:{rel}"),
            Repo::Local(dir) => {
                let full = dir.join(&rel);
                full.canonicalize()
                    .unwrap_or(full)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    /// The repository as written, for messages.
    #[must_use]
    pub fn repo_label(&self) -> String {
        match &self.repo {
            Repo::Local(dir) => dir.display().to_string(),
            Repo::GitHub(r) | Repo::Url(r) => r.clone(),
        }
    }
}

/// A fetched repository; a temporary clone is removed on drop.
#[derive(Debug)]
pub struct Checkout {
    root: PathBuf,
    temp: Option<PathBuf>,
}

impl Checkout {
    /// The repository's root folder.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `path` inside the checkout, which must exist.
    ///
    /// # Errors
    /// Fails when nothing is at that path.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, Error> {
        let full = self.root.join(path);
        if full.exists() {
            Ok(full)
        } else {
            Err(Error::msg(format!("`{path}` is not in the repository")))
        }
    }
}

impl Drop for Checkout {
    fn drop(&mut self) {
        if let Some(temp) = &self.temp {
            let _ = std::fs::remove_dir_all(temp);
        }
    }
}

fn is_github(repo: &str) -> bool {
    let ok = |s: &str| {
        !s.is_empty()
            && !s.starts_with('.')
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    matches!(repo.split_once('/'), Some((o, r)) if ok(o) && ok(r))
}

fn expand_tilde(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_paths_parse_in_every_accepted_form() {
        let p = RepoPath::parse("aovestdipaperino/plank-profiles:/profiles/evil").unwrap();
        assert_eq!(
            p.repo,
            Repo::GitHub("aovestdipaperino/plank-profiles".into())
        );
        assert_eq!(p.path, "profiles/evil");
        assert_eq!(
            p.record(Path::new("profiles/evil")),
            "aovestdipaperino/plank-profiles:profiles/evil"
        );

        let p = RepoPath::parse("https://git.example.com/a/b.git:vectors.json").unwrap();
        assert_eq!(p.repo, Repo::Url("https://git.example.com/a/b.git".into()));
        assert_eq!(p.path, "vectors.json");

        let here = std::env::temp_dir();
        let p = RepoPath::parse(&format!("{}:x/y.json", here.display())).unwrap();
        assert_eq!(p.repo, Repo::Local(here));
    }

    #[test]
    fn paths_that_leave_the_repository_are_refused() {
        for bad in [
            "o/r:",
            "o/r:../x",
            "o/r:a/../../b",
            "no-colon",
            "not a repo:x",
        ] {
            assert!(RepoPath::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_local_repository_is_used_in_place() {
        let dir = std::env::temp_dir().join(format!("plank-lib-src-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("f.json"), "[]").unwrap();
        let p = RepoPath::parse(&format!("{}:sub/f.json", dir.display())).unwrap();
        let checkout = p.fetch().unwrap();
        assert_eq!(checkout.root(), dir);
        assert!(checkout.resolve("sub/f.json").is_ok());
        assert!(checkout.resolve("missing").is_err());
        drop(checkout);
        assert!(dir.is_dir(), "a local repository is never removed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
