// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

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
    /// The model-visible sub-agent roster, when any definition is dispatchable
    /// by the model. Lives here rather than in the `agent` tool's schema so
    /// editing a definition does not invalidate the fingerprinted system prompt
    /// — a 1M-token reprefill — and instead only rebuilds the far smaller
    /// Tier-2 project cache, which [`Self::stable_hash`] already keys.
    pub agents_content: Option<String>,
    /// Persistent memory section (if any memory file has content).
    pub memory_content: Option<String>,
    /// Current date context line.
    pub date_content: String,
}

impl ContextContent {
    /// Collects all context content at session start.
    #[must_use]
    pub fn new() -> Self {
        let git_content = fetch_git_context();
        let agents_md_content = discover_agents_md_files();
        let memory_content = std::env::current_dir()
            .ok()
            .and_then(|cwd| crate::memory::load_default(&cwd));
        let agents_content = std::env::current_dir()
            .ok()
            .and_then(|cwd| agent_roster_context(&crate::agents::load_default(&cwd)));
        let date_content = date_context_line();

        Self {
            git_content,
            agents_md_content,
            agents_content,
            memory_content,
            date_content,
        }
    }

    /// Tier 2 (project-stable) context: the discovered `AGENTS.md`/`CLAUDE.md`
    /// set and persistent memory. These rarely change, so they form a reusable
    /// KV prefix keyed by [`Self::stable_hash`] (`project.kv`, issue #60). Empty
    /// when nothing stable was discovered.
    #[must_use]
    pub fn stable_context(&self) -> String {
        let mut out = String::new();

        if let Some(agents_md) = &self.agents_md_content {
            out.push_str("Agent instructions:\n\n");
            out.push_str(agents_md);
            out.push('\n');
        }

        if let Some(memory) = &self.memory_content {
            out.push_str(memory);
            out.push('\n');
        }

        // Tier 2 rather than Tier 3: definitions are on-disk files that change
        // only when edited, so the roster is cacheable — and an edit costs a
        // project-tier rebuild instead of a system-prompt one.
        if let Some(agents) = &self.agents_content {
            out.push_str(agents);
            out.push('\n');
        }

        out
    }

    /// Tier 3 (session-volatile) context: git status and the date line. These
    /// always differ (git status per session, date per day), so this unit is
    /// prefill-only and never cached — it sits *after* the stable prefix so it
    /// only ever costs its own small prefill (issue #60). Kept as a single
    /// cleanly-identifiable block for context warming (issue #63).
    #[must_use]
    pub fn volatile_context(&self) -> String {
        let mut out = String::new();

        if let Some(git) = &self.git_content {
            out.push_str(git);
            out.push('\n');
        }

        out.push_str(&self.date_content);

        out
    }

    /// Content hash of the Tier 2 (stable) context, used as the volatile part of
    /// the `project.kv` fingerprint key (`fp2 = sha(fp1 ‖ stable-hash)`). `None`
    /// when there is no stable content to cache. Pure function of the discovered
    /// AGENTS.md/CLAUDE.md set and memory, so it is deterministic and
    /// order-stable given the same inputs.
    #[must_use]
    pub fn stable_hash(&self) -> Option<String> {
        let stable = self.stable_context();
        if stable.is_empty() {
            None
        } else {
            Some(crate::session::sha1_hex(stable.as_bytes()))
        }
    }

    /// Returns the combined context content as a single string, ordered
    /// **stable-then-volatile** (issue #60): the reusable project instructions
    /// first, then the always-changing git status and date. This ordering is
    /// what lets the stable prefix stay a valid, cacheable KV tier while the
    /// volatile bytes sit after it. It is not parity-constrained (the C
    /// reference does not fix this first-user-message ordering; see
    /// `tests/c_parity.rs`).
    #[must_use]
    pub fn combined(&self) -> String {
        let mut out = self.stable_context();
        out.push_str(&self.volatile_context());
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
    /// Tokens from persistent memory files.
    pub memory: i32,
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
        let memory = content.memory_content.as_deref().map_or(0, &mut counter);
        let date = counter(&content.date_content);
        let total = git + agents_md + memory + date;

        Self {
            git,
            agents_md,
            memory,
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

/// Renders the model-visible sub-agent roster for the session context, or `None`
/// when the model may not select any definition.
///
/// Filtered exactly as the tool schema used to be: a definition the model must
/// not reach for — `auto: false`, `agents.autoRoute` off, or a missing API-key
/// variable — is absent here too, so the roster never advertises something that
/// would fail on dispatch. `/agent` still lists those, with the reason.
///
/// Names only need to be recognisable, not enumerated in a schema: an unmatched
/// name runs a general-purpose sub-agent and says so, so the model self-corrects
/// from the note rather than from JSON validation.
#[must_use]
pub fn agent_roster_context(defs: &[crate::agents::AgentDef]) -> Option<String> {
    use std::fmt::Write as _;
    let auto_route = crate::settings::active().agents.auto_route;
    let visible = crate::agents::model_visible(defs, auto_route);
    if visible.is_empty() {
        return None;
    }
    let mut out =
        String::from("Configured sub-agents, selectable by passing `name` to the `agent` tool:\n");
    for d in visible {
        let _ = write!(out, "  {}", d.name);
        if d.description.is_empty() {
            out.push_str(": (no description)");
        } else {
            let _ = write!(out, ": {}", d.description);
        }
        out.push('\n');
    }
    Some(out)
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
            memory_content: Some("Persistent memory: prefers tabs".to_string()),
            agents_content: None,
            date_content: "Today's date is 2026-01-01.".to_string(),
        };

        let combined = content.combined();
        assert!(combined.contains("Git info"));
        assert!(combined.contains("AGENTS.md content"));
        assert!(combined.contains("Persistent memory: prefers tabs"));
        assert!(combined.contains("Today's date is 2026-01-01."));
    }

    #[test]
    fn combined_is_stable_then_volatile() {
        // Issue #60: the stable project instructions must precede the volatile
        // git/date block, and `combined` must be exactly their concatenation.
        let content = ContextContent {
            git_content: Some("GIT-STATUS".to_string()),
            agents_md_content: Some("AGENTS".to_string()),
            memory_content: Some("MEM".to_string()),
            agents_content: None,
            date_content: "DATE".to_string(),
        };

        let combined = content.combined();
        assert_eq!(
            combined,
            format!("{}{}", content.stable_context(), content.volatile_context())
        );

        // Stable (AGENTS/MEM) appears before volatile (git/date).
        let agents_at = combined.find("AGENTS").unwrap();
        let git_at = combined.find("GIT-STATUS").unwrap();
        let date_at = combined.find("DATE").unwrap();
        assert!(agents_at < git_at, "stable must precede volatile");
        assert!(git_at < date_at, "git precedes date within volatile");
    }

    #[test]
    fn stable_and_volatile_partition_by_volatility() {
        let content = ContextContent {
            git_content: Some("GIT-STATUS".to_string()),
            agents_md_content: Some("AGENTS".to_string()),
            memory_content: Some("MEM".to_string()),
            agents_content: None,
            date_content: "DATE".to_string(),
        };
        let stable = content.stable_context();
        let volatile = content.volatile_context();

        assert!(stable.contains("AGENTS") && stable.contains("MEM"));
        assert!(!stable.contains("GIT-STATUS") && !stable.contains("DATE"));
        assert!(volatile.contains("GIT-STATUS") && volatile.contains("DATE"));
        assert!(!volatile.contains("AGENTS") && !volatile.contains("MEM"));
    }

    #[test]
    fn stable_hash_is_deterministic_and_content_keyed() {
        let a = ContextContent {
            git_content: Some("differs-every-session".to_string()),
            agents_md_content: Some("AGENTS".to_string()),
            memory_content: Some("MEM".to_string()),
            agents_content: None,
            date_content: "differs-every-day".to_string(),
        };
        // Same stable content, different volatile content → same stable hash.
        let b = ContextContent {
            git_content: Some("other-git".to_string()),
            agents_md_content: Some("AGENTS".to_string()),
            memory_content: Some("MEM".to_string()),
            agents_content: None,
            date_content: "other-date".to_string(),
        };
        assert_eq!(a.stable_hash(), b.stable_hash());
        assert!(a.stable_hash().is_some());

        // Changed stable content → changed hash.
        let c = ContextContent {
            agents_md_content: Some("AGENTS-CHANGED".to_string()),
            ..a.clone()
        };
        assert_ne!(a.stable_hash(), c.stable_hash());
    }

    #[test]
    fn stable_hash_none_without_stable_content() {
        let content = ContextContent {
            git_content: Some("git".to_string()),
            agents_md_content: None,
            memory_content: None,
            agents_content: None,
            date_content: "date".to_string(),
        };
        assert_eq!(content.stable_hash(), None);
    }

    #[test]
    fn context_tokens_counts_categories() {
        let content = ContextContent {
            git_content: Some("git".to_string()),
            agents_md_content: Some("agents".to_string()),
            memory_content: Some("mem".to_string()),
            agents_content: None,
            date_content: "date".to_string(),
        };

        let tokens = ContextTokens::count(&content, |s| i32::try_from(s.len()).unwrap());
        assert_eq!(tokens.git, 3);
        assert_eq!(tokens.agents_md, 6);
        assert_eq!(tokens.memory, 3);
        assert_eq!(tokens.date, 4);
        assert_eq!(tokens.total, 16);
    }

    #[test]
    fn context_tokens_handles_none() {
        let content = ContextContent {
            git_content: None,
            agents_md_content: None,
            memory_content: None,
            agents_content: None,
            date_content: "date".to_string(),
        };

        let tokens = ContextTokens::count(&content, |s| i32::try_from(s.len()).unwrap());
        assert_eq!(tokens.git, 0);
        assert_eq!(tokens.agents_md, 0);
        assert_eq!(tokens.date, 4);
        assert_eq!(tokens.total, 4);
    }

    /// The point of moving the roster here: editing a definition must cost the
    /// Tier-2 project cache, not the fingerprinted system prompt. So the roster
    /// has to be inside `stable_context` (and therefore inside `stable_hash`),
    /// and absent from the prompt entirely.
    #[test]
    fn the_roster_keys_the_project_tier_not_the_system_prompt() {
        use crate::agents::AgentDef;
        let def = |name: &str, desc: &str| AgentDef {
            name: name.to_string(),
            description: desc.to_string(),
            body: "Body.".to_string(),
            path: std::path::PathBuf::from(format!("/tmp/{name}.md")),
            engine: None,
            auto: true,
        };
        crate::settings::install_for_test(crate::settings::Settings::default());

        let one = ContextContent {
            agents_content: agent_roster_context(&[def("reviewer", "reviews diffs")]),
            ..ContextContent::default()
        };
        let two = ContextContent {
            agents_content: agent_roster_context(&[def("reviewer", "reviews diffs BUT EDITED")]),
            ..ContextContent::default()
        };

        assert!(
            one.stable_context().contains("reviewer: reviews diffs"),
            "the roster rides in the stable tier: {}",
            one.stable_context()
        );
        assert!(
            !one.volatile_context().contains("reviewer"),
            "not in the volatile tier, which is never cached"
        );
        assert_ne!(
            one.stable_hash(),
            two.stable_hash(),
            "editing a description must rekey project.kv"
        );

        // And the expensive prefix is untouched by any of it.
        let prompt = crate::sysprompt::build_system_prompt("", &[], true);
        assert!(
            !prompt.contains("reviewer"),
            "roster stayed out of the prompt"
        );
    }

    /// A definition the model must not select is withheld here too, so the
    /// roster never advertises something that would fail on dispatch.
    #[test]
    fn the_roster_applies_the_model_visibility_gates() {
        use crate::agents::AgentDef;
        let mut manual = AgentDef {
            name: "manual-only".to_string(),
            description: "only by hand".to_string(),
            body: String::new(),
            path: std::path::PathBuf::from("/tmp/m.md"),
            engine: None,
            auto: false,
        };
        crate::settings::install_for_test(crate::settings::Settings::default());
        assert!(
            agent_roster_context(std::slice::from_ref(&manual)).is_none(),
            "auto: false is withheld"
        );
        manual.auto = true;
        assert!(agent_roster_context(std::slice::from_ref(&manual)).is_some());

        let mut settings = crate::settings::Settings::default();
        settings.agents.auto_route = false;
        crate::settings::install_for_test(settings);
        assert!(
            agent_roster_context(std::slice::from_ref(&manual)).is_none(),
            "autoRoute off withholds everything"
        );
        crate::settings::install_for_test(crate::settings::Settings::default());
    }
}
