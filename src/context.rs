//! Context tracking for agent: git status, AGENTS.md discovery, datetime.
//!
//! Provides context content and token-counting by category for the /context report.

use std::fmt::Write as _;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum characters for git status output before truncating.
const MAX_STATUS_CHARS: usize = 2000;

/// Categorized context with content and token counts.
///
/// **Cache-boundary rule** (docs/SYSTEM-PROMPT.md): everything collected here
/// is per-session volatile and is injected as the session's *first user
/// message*, never into the system prompt — the system prompt is what the
/// `sysprompt.kv` KV snapshot fingerprints, and volatile bytes there would
/// force a rebuild every launch.
#[derive(Debug, Default, Clone)]
pub struct ContextContent {
    /// Git context content (if in a git repo).
    pub git_content: Option<String>,
    /// AGENTS.md content (if found).
    pub agents_md_content: Option<String>,
    /// Current date context line.
    pub date_content: String,
}

impl ContextContent {
    /// Collects all context content at session start.
    #[must_use]
    pub fn new() -> Self {
        let git_content = fetch_git_context();
        let agents_md_content = discover_agents_md_files();
        let date_content = date_context_line();

        Self {
            git_content,
            agents_md_content,
            date_content,
        }
    }

    /// Returns the combined context content as a single string.
    #[must_use]
    pub fn combined(&self) -> String {
        let mut out = String::new();

        if let Some(git) = &self.git_content {
            out.push_str(git);
            out.push('\n');
        }

        if let Some(agents_md) = &self.agents_md_content {
            out.push_str("Agent instructions:\n\n");
            out.push_str(agents_md);
            out.push('\n');
        }

        out.push_str(&self.date_content);

        out
    }
}

/// Token counts by context category.
#[derive(Debug, Default, Clone, Copy)]
pub struct ContextTokens {
    /// Tokens from git context (status, branch, commits).
    pub git: i32,
    /// Tokens from AGENTS.md files.
    pub agents_md: i32,
    /// Tokens from date context line.
    pub date: i32,
    /// Total context tokens.
    pub total: i32,
}

impl ContextTokens {
    /// Counts tokens for each context category using the given token counter.
    pub fn count<F>(content: &ContextContent, mut counter: F) -> Self
    where
        F: FnMut(&str) -> i32,
    {
        let git = content.git_content.as_deref().map_or(0, &mut counter);
        let agents_md = content.agents_md_content.as_deref().map_or(0, &mut counter);
        let date = counter(&content.date_content);
        let total = git + agents_md + date;

        Self {
            git,
            agents_md,
            date,
            total,
        }
    }
}

/// Fetches git context from the current directory.
fn fetch_git_context() -> Option<String> {
    if !is_inside_git_worktree() {
        return None;
    }

    // Fan the five independent git commands out in parallel so session-start
    // latency is the slowest single command, not the sum. The composed block
    // stays byte-identical to the sequential version.
    let (branch, main_branch, user_name, status, recent_commits) = std::thread::scope(|s| {
        let branch = s.spawn(git_current_branch);
        let main_branch = s.spawn(git_main_branch);
        let user_name = s.spawn(git_user_name);
        let status = s.spawn(git_status);
        let recent_commits = s.spawn(git_recent_commits);
        (
            branch.join().ok().flatten(),
            main_branch.join().ok().flatten(),
            user_name.join().ok().flatten(),
            status.join().ok().flatten(),
            recent_commits.join().ok().flatten(),
        )
    });
    let branch = branch?;

    let mut out = String::from(
        "This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation.\n\n",
    );

    let _ = writeln!(out, "Current branch: {branch}");
    if let Some(main) = main_branch {
        let _ = writeln!(
            out,
            "Main branch (you will usually use this for PRs): {main}"
        );
    }
    if let Some(user) = user_name {
        let _ = writeln!(out, "Git user: {user}");
    }

    out.push_str("Status:\n");
    if let Some(status) = status {
        out.push_str(if status.is_empty() {
            "(clean)"
        } else {
            &status
        });
    } else {
        out.push_str("(clean)");
    }
    out.push('\n');

    if let Some(commits) = recent_commits {
        let _ = writeln!(out, "Recent commits:\n{commits}");
    }

    Some(out)
}

/// Checks if the current directory is inside a git worktree.
fn is_inside_git_worktree() -> bool {
    Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).trim() == "true")
}

/// Gets the current git branch name.
fn git_current_branch() -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.is_empty() || s == "HEAD" {
                None
            } else {
                Some(s)
            }
        })
}

/// Gets the default/main branch name.
fn git_main_branch() -> Option<String> {
    Command::new("git")
        .args(["symbolic-ref", "refs/remotes/origin/HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            let s = String::from_utf8_lossy(&output.stdout);
            s.strip_prefix("refs/remotes/origin/")
                .map(|branch| branch.trim().to_string())
        })
        .or_else(|| {
            Command::new("git")
                .args(["config", "init.defaultBranch"])
                .output()
                .ok()
                .map(|output| {
                    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if s.is_empty() { "main".to_string() } else { s }
                })
        })
}

/// Gets the git user name from config.
fn git_user_name() -> Option<String> {
    Command::new("git")
        .args(["config", "user.name"])
        .output()
        .ok()
        .and_then(|output| {
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        })
}

/// Gets short git status output, truncated if too long.
fn git_status() -> Option<String> {
    Command::new("git")
        .args(["--no-optional-locks", "status", "--short"])
        .output()
        .ok()
        .map(|output| {
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.len() > MAX_STATUS_CHARS {
                format!(
                    "{}\n... (truncated because it exceeds 2k characters. If you need more information, run \"git status\" using BashTool)",
                    &s[..MAX_STATUS_CHARS]
                )
            } else {
                s
            }
        })
}

/// Gets recent commit log (last 5 commits, one-line format).
fn git_recent_commits() -> Option<String> {
    Command::new("git")
        .args(["--no-optional-locks", "log", "--oneline", "-n", "5"])
        .output()
        .ok()
        .and_then(|output| {
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        })
}

/// Discovers AGENTS.md files from the current directory upward.
///
/// CLAUDE.md is treated as a synonym: in each directory it is used as a
/// fallback when AGENTS.md is missing (AGENTS.md wins when both exist).
fn discover_agents_md_files() -> Option<String> {
    let mut contents = Vec::new();
    let mut current_dir = std::env::current_dir().ok()?;

    loop {
        for name in ["AGENTS.md", "CLAUDE.md"] {
            let path = current_dir.join(name);
            if let Ok(content) = std::fs::read_to_string(&path) {
                let header = format!("\n---\n# From: {}\n---\n", path.display());
                contents.push(header);
                contents.push(content);
                break;
            }
        }

        // At the filesystem root `parent()` is None; stop walking without
        // discarding what was already collected.
        let Some(parent) = current_dir.parent() else {
            break;
        };
        current_dir = parent.to_path_buf();
    }

    if contents.is_empty() {
        None
    } else {
        Some(contents.join("\n"))
    }
}

/// Gets the current local ISO date string.
#[must_use]
pub fn current_local_iso_date() -> String {
    format_local_date().unwrap_or_else(|()| "date unavailable".to_string())
}

/// Formats the current date as local ISO format (YYYY-MM-DD).
fn format_local_date() -> Result<String, ()> {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ())?
        .as_secs();
    let t: libc::time_t = i64::try_from(secs).map_err(|_| ())?;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };

    if unsafe { libc::localtime_r(&raw const t, &raw mut tm) }.is_null() {
        return Err(());
    }

    let mut buf = [0u8; 16];
    let fmt = c"%Y-%m-%d";

    let n = unsafe {
        libc::strftime(
            buf.as_mut_ptr().cast::<libc::c_char>(),
            buf.len(),
            fmt.as_ptr(),
            &raw const tm,
        )
    };

    if n == 0 {
        return Err(());
    }

    Ok(String::from_utf8_lossy(&buf[..n]).into_owned())
}

/// Formats the current date as a context line.
#[must_use]
pub fn date_context_line() -> String {
    format!("Today's date is {}.", current_local_iso_date())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_context_line_has_expected_prefix() {
        let line = date_context_line();
        assert!(line.starts_with("Today's date is "));
        assert!(line.ends_with('.'));
    }

    #[test]
    fn context_content_combines_parts() {
        let content = ContextContent {
            git_content: Some("Git info".to_string()),
            agents_md_content: Some("AGENTS.md content".to_string()),
            date_content: "Today's date is 2026-01-01.".to_string(),
        };

        let combined = content.combined();
        assert!(combined.contains("Git info"));
        assert!(combined.contains("AGENTS.md content"));
        assert!(combined.contains("Today's date is 2026-01-01."));
    }

    #[test]
    fn context_tokens_counts_categories() {
        let content = ContextContent {
            git_content: Some("git".to_string()),
            agents_md_content: Some("agents".to_string()),
            date_content: "date".to_string(),
        };

        let tokens = ContextTokens::count(&content, |s| i32::try_from(s.len()).unwrap());
        assert_eq!(tokens.git, 3);
        assert_eq!(tokens.agents_md, 6);
        assert_eq!(tokens.date, 4);
        assert_eq!(tokens.total, 13);
    }

    #[test]
    fn context_tokens_handles_none() {
        let content = ContextContent {
            git_content: None,
            agents_md_content: None,
            date_content: "date".to_string(),
        };

        let tokens = ContextTokens::count(&content, |s| i32::try_from(s.len()).unwrap());
        assert_eq!(tokens.git, 0);
        assert_eq!(tokens.agents_md, 0);
        assert_eq!(tokens.date, 4);
        assert_eq!(tokens.total, 4);
    }
}
