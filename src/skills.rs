// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! User-defined skills: markdown prompt templates exposed as slash commands.
//!
//! A skill is a directory containing a `SKILL.md` file with optional YAML-ish
//! frontmatter (`name`, `description`, `argument-hint`) followed by the prompt
//! body. Invoking `/<name> [args]` injects the body — with `$ARGUMENTS`
//! substituted — as a user-turn preamble and runs a normal turn.
//!
//! Discovery mirrors the `.mcp.json` layering: the global `~/.plank/skills/`
//! directory is loaded first, then the project's `./.plank/skills/`, and a
//! project skill overrides a global one with the same name.

use std::path::{Path, PathBuf};

/// One loaded skill.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Slash-command name (no leading `/`); defaults to the directory name.
    pub name: String,
    /// One-line description shown by `/skills`.
    pub description: String,
    /// Hint describing what to pass as arguments, shown by `/skills`.
    pub argument_hint: String,
    /// Markdown prompt body (frontmatter stripped).
    pub body: String,
    /// Directory the skill was loaded from.
    pub dir: PathBuf,
}

/// Splits leading `---` frontmatter from a SKILL.md; returns (frontmatter
/// lines, body). Files without frontmatter yield an empty first element.
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

/// Loads one skill from `dir/SKILL.md`; `None` when missing or unusable.
fn load_skill(dir: &Path) -> Option<Skill> {
    let text = std::fs::read_to_string(dir.join("SKILL.md")).ok()?;
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
        name = dir.file_name()?.to_string_lossy().into_owned();
    }
    // The name becomes a slash command: reject anything unroutable.
    if name.is_empty() || name.contains(char::is_whitespace) || name.contains('/') {
        return None;
    }
    if body.trim().is_empty() {
        return None;
    }
    Some(Skill {
        name,
        description: get("description"),
        argument_hint: get("argument-hint"),
        body,
        dir: dir.to_path_buf(),
    })
}

/// Loads skills from `<root>/*/SKILL.md`, sorted by name for stable listings.
fn load_dir(root: &Path) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut skills: Vec<Skill> = entries
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .filter_map(|e| load_skill(&e.path()))
        .collect();
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Loads skills from the given roots in order; a later root's skill replaces
/// an earlier one with the same name (project overrides global).
#[must_use]
pub fn load_from(roots: &[PathBuf]) -> Vec<Skill> {
    let mut merged: Vec<Skill> = Vec::new();
    for root in roots {
        for skill in load_dir(root) {
            if let Some(existing) = merged.iter_mut().find(|s| s.name == skill.name) {
                *existing = skill;
            } else {
                merged.push(skill);
            }
        }
    }
    merged.sort_by(|a, b| a.name.cmp(&b.name));
    merged
}

/// Loads skills from the default hierarchy: `~/.plank/skills` overlaid by
/// `<cwd>/.plank/skills`.
#[must_use]
pub fn load_default(cwd: &Path) -> Vec<Skill> {
    let mut roots = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".plank").join("skills"));
    }
    roots.push(cwd.join(".plank").join("skills"));
    load_from(&roots)
}

/// Renders a skill invocation into the user-turn preamble: `$ARGUMENTS` is
/// substituted with `args`; when the body has no placeholder and arguments
/// were given, they are appended as a trailing paragraph so they are never
/// silently dropped.
#[must_use]
pub fn render(skill: &Skill, args: &str) -> String {
    let args = args.trim();
    if skill.body.contains("$ARGUMENTS") {
        return skill.body.replace("$ARGUMENTS", args);
    }
    if args.is_empty() {
        skill.body.clone()
    } else {
        format!("{}\n\nArguments: {args}\n", skill.body.trim_end())
    }
}

/// Renders the `/skills` listing.
#[must_use]
pub fn render_list(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return "no skills found (checked ~/.plank/skills and ./.plank/skills)\n".to_string();
    }
    let mut out = String::from("Skills (invoke with /<name> [arguments]):\n");
    for s in skills {
        out.push_str("  /");
        out.push_str(&s.name);
        if !s.argument_hint.is_empty() {
            out.push(' ');
            out.push_str(&s.argument_hint);
        }
        if !s.description.is_empty() {
            out.push_str(" — ");
            out.push_str(&s.description);
        }
        out.push('\n');
    }
    out
}

/// Renders the model-facing skill list for the `skill` tool's enumerate case:
/// one `name — description` per line, or a clear no-skills message.
#[must_use]
pub fn render_names(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return "No skills are installed (checked ~/.plank/skills and ./.plank/skills).\n"
            .to_string();
    }
    let mut out = String::from("Available skills (call skill with name set to one of):\n");
    for s in skills {
        out.push_str("- ");
        out.push_str(&s.name);
        if !s.description.is_empty() {
            out.push_str(" — ");
            out.push_str(&s.description);
        }
        out.push('\n');
    }
    out
}

/// `skill` tool: lets the model invoke a skill by name, mirroring what the
/// user's `/name args` slash command produces (issue #36).
///
/// A missing/empty `name` enumerates the installed skills. An unknown name
/// lists the available ones so a near miss self-corrects. The rendered text is
/// returned as the tool result, so it lands in the transcript as guidance the
/// model then follows.
pub fn tool_skill(
    skills: &[Skill],
    invocations: &mut usize,
    cap: usize,
    call: &crate::dsml::ToolCall,
) -> String {
    let name = call.arg_value("name").unwrap_or("").trim();
    if name.is_empty() {
        return render_names(skills);
    }
    *invocations += 1;
    if *invocations > cap {
        return format!(
            "Tool error: skill invocation limit ({cap}) reached this turn; \
             refusing to expand another skill to avoid a loop\n"
        );
    }
    let args = call.arg_value("args").unwrap_or("");
    if let Some(skill) = skills.iter().find(|s| s.name == name) {
        render(skill, args)
    } else {
        let mut out = format!("Tool error: unknown skill: {name}\n");
        out.push_str(&render_names(skills));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_skill(root: &Path, dir_name: &str, content: &str) {
        let dir = root.join(dir_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), content).unwrap();
    }

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("plank-skills-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn loads_frontmatter_and_body() {
        let root = temp_root("load");
        write_skill(
            &root,
            "review",
            "---\nname: review\ndescription: Review code\nargument-hint: <path>\n---\nReview $ARGUMENTS carefully.\n",
        );
        write_skill(&root, "bare", "Just a body, no frontmatter.\n");
        write_skill(&root, "empty", "---\nname: empty\n---\n   \n");
        let skills = load_from(std::slice::from_ref(&root));
        assert_eq!(skills.len(), 2, "{skills:?}");
        assert_eq!(skills[0].name, "bare");
        assert_eq!(skills[1].name, "review");
        assert_eq!(skills[1].description, "Review code");
        assert_eq!(skills[1].argument_hint, "<path>");
        assert_eq!(skills[1].body, "Review $ARGUMENTS carefully.\n");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn project_overrides_global_by_name() {
        let global = temp_root("global");
        let project = temp_root("project");
        write_skill(&global, "deploy", "global body\n");
        write_skill(&global, "only-global", "global-only body\n");
        write_skill(&project, "deploy", "project body\n");
        let skills = load_from(&[global.clone(), project.clone()]);
        let deploy = skills.iter().find(|s| s.name == "deploy").unwrap();
        assert_eq!(deploy.body, "project body\n");
        assert!(skills.iter().any(|s| s.name == "only-global"));
        std::fs::remove_dir_all(&global).ok();
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn render_substitutes_or_appends_arguments() {
        let mut s = Skill {
            name: "t".into(),
            description: String::new(),
            argument_hint: String::new(),
            body: "Do the thing with $ARGUMENTS now.".into(),
            dir: PathBuf::new(),
        };
        assert_eq!(render(&s, "x y"), "Do the thing with x y now.");
        assert_eq!(render(&s, ""), "Do the thing with  now.");
        s.body = "No placeholder here.".into();
        assert_eq!(render(&s, ""), "No placeholder here.");
        assert_eq!(
            render(&s, "extra"),
            "No placeholder here.\n\nArguments: extra\n"
        );
    }

    #[test]
    fn listing_shows_hint_and_description() {
        let root = temp_root("list");
        write_skill(
            &root,
            "review",
            "---\ndescription: Review code\nargument-hint: <path>\n---\nbody\n",
        );
        let skills = load_from(std::slice::from_ref(&root));
        let list = render_list(&skills);
        assert!(list.contains("/review <path> — Review code"), "{list}");
        assert!(render_list(&[]).contains("no skills found"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn invalid_names_are_skipped() {
        let root = temp_root("invalid");
        write_skill(&root, "spacey", "---\nname: has space\n---\nbody\n");
        let skills = load_from(std::slice::from_ref(&root));
        assert!(skills.is_empty(), "{skills:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    fn skill(name: &str, body: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("does {name}"),
            argument_hint: String::new(),
            body: body.to_string(),
            dir: PathBuf::new(),
        }
    }

    fn skill_call(args: &[(&str, &str)]) -> crate::dsml::ToolCall {
        crate::dsml::ToolCall {
            name: "skill".to_string(),
            args: args
                .iter()
                .map(|(k, v)| crate::dsml::ToolArg {
                    name: (*k).to_string(),
                    value: (*v).to_string(),
                    is_string: true,
                })
                .collect(),
        }
    }

    #[test]
    fn skill_tool_renders_the_same_text_as_the_slash_command() {
        let skills = vec![skill("plan", "Plan for $ARGUMENTS now.")];
        let mut n = 0;
        let out = tool_skill(
            &skills,
            &mut n,
            8,
            &skill_call(&[("name", "plan"), ("args", "the API")]),
        );
        assert_eq!(out, render(&skills[0], "the API"));
        assert_eq!(out, "Plan for the API now.");
    }

    #[test]
    fn skill_tool_with_no_name_enumerates() {
        let skills = vec![skill("plan", "b"), skill("review", "b")];
        let mut n = 0;
        let out = tool_skill(&skills, &mut n, 8, &skill_call(&[]));
        assert!(out.contains("plan — does plan"), "{out}");
        assert!(out.contains("review — does review"), "{out}");
        assert_eq!(n, 0, "enumerate does not count against the cap");
    }

    #[test]
    fn skill_tool_unknown_name_lists_the_available_ones() {
        let skills = vec![skill("plan", "b")];
        let mut n = 0;
        let out = tool_skill(&skills, &mut n, 8, &skill_call(&[("name", "nope")]));
        assert!(
            out.starts_with("Tool error: unknown skill: nope\n"),
            "{out}"
        );
        assert!(out.contains("plan — does plan"), "{out}");
    }

    #[test]
    fn skill_tool_with_no_skills_installed_is_a_clear_message() {
        let mut n = 0;
        let out = tool_skill(&[], &mut n, 8, &skill_call(&[]));
        assert!(out.contains("No skills are installed"), "{out}");
        let out2 = tool_skill(&[], &mut n, 8, &skill_call(&[("name", "x")]));
        assert!(out2.starts_with("Tool error: unknown skill: x\n"), "{out2}");
        assert!(out2.contains("No skills are installed"), "{out2}");
    }

    #[test]
    fn skill_tool_caps_recursion_depth() {
        let skills = vec![skill("plan", "body")];
        let mut n = 0;
        for _ in 0..3 {
            let out = tool_skill(&skills, &mut n, 3, &skill_call(&[("name", "plan")]));
            assert_eq!(out, "body");
        }
        let capped = tool_skill(&skills, &mut n, 3, &skill_call(&[("name", "plan")]));
        assert!(
            capped.contains("skill invocation limit (3) reached"),
            "{capped}"
        );
    }
}
