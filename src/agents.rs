// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Single-subagent sidechain support (issue #10, reduced scope).
//!
//! `/subagent <task>` runs a delegated task as a *fork* of the current
//! conversation: the framed task is appended to the live transcript, the
//! normal turn loop runs (tools included), and afterwards the fork is
//! truncated so only the subagent's final report — framed by
//! [`report_message`] — enters the parent conversation. Because the fork
//! shares the parent transcript prefix, the engine's per-turn common-prefix
//! sync reuses the parent KV cache on the way in and rolls the sidechain
//! back on the next real turn.
//!
//! One built-in general-purpose subagent, plus *named* agent definitions
//! loaded from `~/.plank/agents/*.md` overlaid by `./.plank/agents/*.md`
//! (issue #19). A named definition supplies extra instructions that frame the
//! subagent's turn. Parallel/team orchestration remains out of scope (blocked
//! on per-session KV save/restore; see the tracking issue).

use std::path::{Path, PathBuf};

/// One loaded named agent definition.
#[derive(Debug, Clone)]
pub struct AgentDef {
    /// Definition name; matched as the first token of `/subagent <name> …`.
    /// Defaults to the file stem.
    pub name: String,
    /// One-line description shown by `/agent`.
    pub description: String,
    /// Markdown body used as the subagent's instructions (frontmatter stripped).
    pub body: String,
    /// File the definition was loaded from.
    pub path: PathBuf,
}

/// Splits leading `---` frontmatter from an agent `.md`; returns (frontmatter
/// fields, body). Mirrors the skill loader's parser.
fn split_frontmatter(text: &str) -> (Vec<(String, String)>, String) {
    let Some(rest) = text.strip_prefix("---\n") else {
        return (Vec::new(), text.to_string());
    };
    let Some(end) = rest.find("\n---") else {
        return (Vec::new(), text.to_string());
    };
    let head = &rest[..end];
    let mut body = &rest[end + "\n---".len()..];
    if let Some(b) = body.strip_prefix('\n') {
        body = b;
    }
    let fields = head
        .lines()
        .filter_map(|line| {
            let (k, v) = line.split_once(':')?;
            Some((k.trim().to_ascii_lowercase(), v.trim().to_string()))
        })
        .collect();
    (fields, body.to_string())
}

/// Loads one definition from `path`; `None` when missing or unusable.
fn load_def(path: &Path) -> Option<AgentDef> {
    let text = std::fs::read_to_string(path).ok()?;
    let (fields, body) = split_frontmatter(&text);
    let get = |key: &str| {
        fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    };
    let mut name = get("name");
    if name.is_empty() {
        name = path.file_stem()?.to_string_lossy().into_owned();
    }
    // The name is matched as a bare `/subagent` argument token: reject anything
    // with whitespace or a slash that could never be typed as one token.
    if name.is_empty() || name.contains(char::is_whitespace) || name.contains('/') {
        return None;
    }
    if body.trim().is_empty() {
        return None;
    }
    Some(AgentDef {
        name,
        description: get("description"),
        body,
        path: path.to_path_buf(),
    })
}

/// Loads `<root>/*.md`, sorted by name for stable listings.
fn load_dir(root: &Path) -> Vec<AgentDef> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut defs: Vec<AgentDef> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .filter_map(|p| load_def(&p))
        .collect();
    defs.sort_by(|a, b| a.name.cmp(&b.name));
    defs
}

/// Loads definitions from the given roots in order; a later root's definition
/// replaces an earlier one with the same name (project overrides global).
#[must_use]
pub fn load_from(roots: &[PathBuf]) -> Vec<AgentDef> {
    let mut merged: Vec<AgentDef> = Vec::new();
    for root in roots {
        for def in load_dir(root) {
            if let Some(existing) = merged.iter_mut().find(|d| d.name == def.name) {
                *existing = def;
            } else {
                merged.push(def);
            }
        }
    }
    merged.sort_by(|a, b| a.name.cmp(&b.name));
    merged
}

/// Loads definitions from the default hierarchy: `~/.plank/agents` overlaid by
/// `<cwd>/.plank/agents`.
#[must_use]
pub fn load_default(cwd: &Path) -> Vec<AgentDef> {
    let mut roots = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".plank").join("agents"));
    }
    roots.push(cwd.join(".plank").join("agents"));
    load_from(&roots)
}

/// Resolves a `/subagent` argument against the loaded definitions. When the
/// first token names a known definition, returns that definition and the rest
/// as the task. Otherwise returns `None` and the whole argument (today's
/// general-purpose behavior).
#[must_use]
pub fn resolve<'a>(defs: &'a [AgentDef], arg: &'a str) -> (Option<&'a AgentDef>, &'a str) {
    let arg = arg.trim();
    let (first, rest) = match arg.split_once(char::is_whitespace) {
        Some((f, r)) => (f, r.trim()),
        None => (arg, ""),
    };
    match defs.iter().find(|d| d.name == first) {
        Some(def) => (Some(def), rest),
        None => (None, arg),
    }
}

/// Renders the `/agent` listing.
#[must_use]
pub fn render_list(defs: &[AgentDef]) -> String {
    if defs.is_empty() {
        return "no agent definitions found (checked ~/.plank/agents and ./.plank/agents)\n"
            .to_string();
    }
    let mut out = String::from("Agents (dispatch with /subagent <name> <task>):\n");
    for d in defs {
        out.push_str("  ");
        out.push_str(&d.name);
        if !d.description.is_empty() {
            out.push_str(" — ");
            out.push_str(&d.description);
        }
        out.push('\n');
    }
    out
}

/// Frames the delegated task as the sidechain's user turn. `instructions`, when
/// present, is a named definition's body prepended as the subagent's persona.
#[must_use]
pub fn task_message(instructions: Option<&str>, task: &str) -> String {
    let mut out = String::from(
        "<system-reminder>\n\
         You are now acting as a subagent, handling a task delegated from the \
         main conversation. Complete the task using your tools, then end with \
         a final report of your results — only that report is carried back \
         into the main conversation; everything else is discarded.\n\
         </system-reminder>\n\n",
    );
    if let Some(instructions) = instructions {
        let instructions = instructions.trim();
        if !instructions.is_empty() {
            out.push_str("Instructions:\n");
            out.push_str(instructions);
            out.push_str("\n\n");
        }
    }
    out.push_str("Task: ");
    out.push_str(task.trim());
    out
}

/// Frames the subagent's final report for the parent conversation.
#[must_use]
pub fn report_message(task: &str, report: &str) -> String {
    format!(
        "<system-reminder>\n\
         A subagent completed the delegated task: {}\n\
         Its final report follows. This is background context from a sidechain \
         run; do not respond to it directly unless the user asks.\n\
         </system-reminder>\n\n\
         Subagent report:\n{}",
        task.trim(),
        report.trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_and_report_framing() {
        let task = task_message(None, "count the tests\n");
        assert!(task.starts_with("<system-reminder>\n"));
        assert!(task.ends_with("Task: count the tests"));
        assert!(!task.contains("Instructions:"));
        let report = report_message("count the tests", "There are 42.\n");
        assert!(report.contains("completed the delegated task: count the tests"));
        assert!(report.ends_with("Subagent report:\nThere are 42."));
    }

    #[test]
    fn task_message_embeds_instructions() {
        let task = task_message(Some("  Be terse.\n"), "count the tests");
        assert!(task.contains("Instructions:\nBe terse.\n\nTask: count the tests"));
        // An empty/whitespace body adds no Instructions block.
        assert!(!task_message(Some("   "), "do it").contains("Instructions:"));
    }

    fn write_def(root: &Path, file: &str, content: &str) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join(file), content).unwrap();
    }

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("plank-agents-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn loads_frontmatter_and_body_sorted() {
        let root = temp_root("load");
        write_def(
            &root,
            "reviewer.md",
            "---\nname: reviewer\ndescription: Reviews code\n---\nYou are a strict reviewer.\n",
        );
        // No frontmatter: name defaults to the file stem.
        write_def(&root, "bare.md", "Just a body.\n");
        // Empty body is rejected.
        write_def(&root, "empty.md", "---\nname: empty\n---\n   \n");
        // Non-markdown files are ignored.
        write_def(&root, "notes.txt", "ignore me\n");
        let defs = load_from(std::slice::from_ref(&root));
        assert_eq!(defs.len(), 2, "{defs:?}");
        assert_eq!(defs[0].name, "bare");
        assert_eq!(defs[1].name, "reviewer");
        assert_eq!(defs[1].description, "Reviews code");
        assert_eq!(defs[1].body, "You are a strict reviewer.\n");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn project_overrides_global_by_name() {
        let global = temp_root("global");
        let project = temp_root("project");
        write_def(&global, "reviewer.md", "global body\n");
        write_def(&global, "only-global.md", "global-only body\n");
        write_def(&project, "reviewer.md", "project body\n");
        let defs = load_from(&[global.clone(), project.clone()]);
        let reviewer = defs.iter().find(|d| d.name == "reviewer").unwrap();
        assert_eq!(reviewer.body, "project body\n");
        assert!(defs.iter().any(|d| d.name == "only-global"));
        std::fs::remove_dir_all(&global).ok();
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn listing_shows_name_and_description() {
        let root = temp_root("list");
        write_def(
            &root,
            "reviewer.md",
            "---\ndescription: Reviews code\n---\nbody\n",
        );
        let defs = load_from(std::slice::from_ref(&root));
        let list = render_list(&defs);
        assert!(list.contains("reviewer — Reviews code"), "{list}");
        assert!(render_list(&[]).contains("no agent definitions found"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolve_name_vs_freeform() {
        let defs = vec![AgentDef {
            name: "reviewer".into(),
            description: String::new(),
            body: "Be strict.".into(),
            path: PathBuf::new(),
        }];
        // First token names a definition: rest is the task.
        let (def, task) = resolve(&defs, "reviewer check the diff");
        assert_eq!(def.unwrap().name, "reviewer");
        assert_eq!(task, "check the diff");
        // Unknown first token: whole argument is a freeform task.
        let (def, task) = resolve(&defs, "count the tests");
        assert!(def.is_none());
        assert_eq!(task, "count the tests");
        // Bare name with no task resolves the definition and an empty task.
        let (def, task) = resolve(&defs, "reviewer");
        assert_eq!(def.unwrap().name, "reviewer");
        assert_eq!(task, "");
    }
}
