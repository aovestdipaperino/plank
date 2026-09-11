// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! User-defined skills: markdown prompt templates exposed as slash commands.
//!
//! A skill is a directory containing a `SKILL.md` file with optional YAML-ish
//! frontmatter (`name`, `description`, `argument-hint`) followed by the prompt
//! body. Invoking `/<name> [args]` injects the body — with `$ARGUMENTS`
//! substituted — as a user-turn preamble and runs a normal turn.
//!
//! Discovery mirrors the `.mcp.json` layering: the skills compiled into the
//! binary come first, then the global `~/.plank/skills/` directory, then the
//! project's `./.plank/skills/`, each layer overriding the last by name. So a
//! `code-review` directory in a project replaces the built-in of that name
//! outright rather than colliding with it — the built-ins are defaults, not
//! reserved words.

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
    /// Directory the skill was loaded from; empty for a built-in, which has no
    /// directory because it is compiled into the binary.
    pub dir: PathBuf,
}

impl Skill {
    /// True for a skill compiled into the binary rather than read from disk.
    /// Keyed on the empty `dir` because that is what having no directory means
    /// — there is no second source of truth to drift from it.
    #[must_use]
    pub fn is_builtin(&self) -> bool {
        self.dir.as_os_str().is_empty()
    }
}

/// The skills plank ships with, as `(name, SKILL.md text)`. Each is a real
/// `SKILL.md` under `src/resources/skills/`, parsed by the same loader as a
/// user's — so a built-in cannot use a frontmatter key or body construct that
/// a user's skill could not.
const BUILTIN: &[(&str, &str)] = &[
    (
        "code-review",
        include_str!("resources/skills/code-review/SKILL.md"),
    ),
    ("debug", include_str!("resources/skills/debug/SKILL.md")),
    (
        "remember",
        include_str!("resources/skills/remember/SKILL.md"),
    ),
    (
        "skillify",
        include_str!("resources/skills/skillify/SKILL.md"),
    ),
    (
        "update-config",
        include_str!("resources/skills/update-config/SKILL.md"),
    ),
    ("verify", include_str!("resources/skills/verify/SKILL.md")),
];

/// Parses the compiled-in skills. A built-in that fails to parse is dropped
/// silently here and caught by `builtins_all_parse` in the tests instead: a
/// malformed built-in is a build-time bug, and refusing to start over one
/// would turn it into a user-facing outage.
#[must_use]
pub fn builtins() -> Vec<Skill> {
    BUILTIN
        .iter()
        .filter_map(|(name, text)| parse_skill(text, Path::new(""), name))
        .collect()
}

/// Splits leading `---` frontmatter from a SKILL.md; returns (frontmatter
/// lines, body). Files without frontmatter yield an empty first element.
pub(crate) fn split_frontmatter(text: &str) -> (Vec<(String, String)>, String) {
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
    let fallback = dir.file_name()?.to_string_lossy().into_owned();
    parse_skill(&text, dir, &fallback)
}

/// Parses one SKILL.md text. `dir` is recorded as the skill's source (empty for
/// a built-in) and `fallback` names it when the frontmatter does not.
fn parse_skill(text: &str, dir: &Path, fallback: &str) -> Option<Skill> {
    let (fields, body) = split_frontmatter(text);
    let get = |key: &str| {
        fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    };
    let mut name = get("name");
    if name.is_empty() {
        name = fallback.to_string();
    }
    // The name becomes a slash command: reject anything unroutable. A colon
    // is refused too, because `<plugin>:<name>` is how plugin entries are
    // addressed and `reconcile` relies on a bare name never containing one —
    // a skill shipping `name: other:thing` would otherwise claim a namespace
    // that belongs to the plugin `other`.
    if name.is_empty()
        || name.contains(char::is_whitespace)
        || name.contains('/')
        || name.contains(':')
    {
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
/// an earlier one with the same name (project overrides global). Disk only —
/// [`load_layered`] is the one that puts the built-ins underneath, because a
/// plugin's own skill directory is loaded through here too and must not come
/// back carrying plank's built-ins as that plugin's contributions.
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

/// The built-ins with `roots` layered on top: a user or project skill replaces
/// the built-in of the same name outright.
#[must_use]
pub fn load_layered(roots: &[PathBuf]) -> Vec<Skill> {
    let mut merged = builtins();
    for skill in load_from(roots) {
        if let Some(existing) = merged.iter_mut().find(|s| s.name == skill.name) {
            *existing = skill;
        } else {
            merged.push(skill);
        }
    }
    merged.sort_by(|a, b| a.name.cmp(&b.name));
    merged
}

/// Loads skills from the default hierarchy: the built-ins, then
/// `~/.plank/skills`, then `<cwd>/.plank/skills`.
#[must_use]
pub fn load_default(cwd: &Path) -> Vec<Skill> {
    let mut roots = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".plank").join("skills"));
    }
    roots.push(cwd.join(".plank").join("skills"));
    load_layered(&roots)
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
    let mut out = String::from(
        "Skills (invoke with /<name> [arguments]; override a built-in by name in \
         ~/.plank/skills or ./.plank/skills):\n",
    );
    // An uncontested plugin skill is registered only under its `<plugin>:<name>`
    // alias; `listing` surfaces it once, naming the plugin.
    for listed in crate::plugins::listing(skills) {
        let s = listed.entry;
        out.push_str("  /");
        out.push_str(listed.name);
        if !s.argument_hint.is_empty() {
            out.push(' ');
            out.push_str(&s.argument_hint);
        }
        if !s.description.is_empty() {
            out.push_str(" — ");
            out.push_str(&s.description);
        }
        if let Some(plugin) = listed.plugin {
            out.push_str(" [plugin ");
            out.push_str(plugin);
            out.push(']');
        } else if s.is_builtin() {
            out.push_str(" [built-in]");
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
        // Only reachable when the caller passes an explicitly empty list (a
        // sub-agent with no skills, say): the default paths always carry the
        // built-ins.
        return "No skills are installed (checked ~/.plank/skills and ./.plank/skills).\n"
            .to_string();
    }
    let mut out = String::from("Available skills (call skill with name set to one of):\n");
    // A plugin skill is registered only under its `<plugin>:<name>` alias, so
    // listing it once is already correct — no bare-name twin to suppress.
    for listed in crate::plugins::listing(skills) {
        out.push_str("- ");
        out.push_str(listed.name);
        if !listed.entry.description.is_empty() {
            out.push_str(" — ");
            out.push_str(&listed.entry.description);
        }
        if let Some(plugin) = listed.plugin {
            out.push_str(" [plugin ");
            out.push_str(plugin);
            out.push(']');
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

    /// `<plugin>:<name>` is how plugin skills are addressed, and `reconcile`
    /// assumes a bare name never contains `:`; a skill naming itself
    /// `other:thing` would otherwise squat on the plugin `other`'s namespace.
    #[test]
    fn a_colon_in_the_name_rejects_the_skill() {
        let root = temp_root("colon");
        write_skill(
            &root,
            "squat",
            "---\nname: superpowers:brainstorm\ndescription: x\n---\nbody\n",
        );
        write_skill(&root, "a:b", "body from a directory name with a colon\n");
        write_skill(&root, "fine", "---\nname: fine\n---\nbody\n");
        let skills = load_from(std::slice::from_ref(&root));
        let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["fine"], "{skills:?}");
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
    fn builtins_all_parse_and_are_routable() {
        let built = builtins();
        assert_eq!(
            built.len(),
            BUILTIN.len(),
            "a built-in failed to parse: {:?}",
            built.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        for (declared, skill) in BUILTIN.iter().map(|(n, _)| *n).zip(&built) {
            // The table name is the directory name; the frontmatter name is
            // what routes. A mismatch would make the table a lie.
            assert_eq!(declared, skill.name, "table name vs frontmatter name");
            assert!(
                !skill.description.is_empty(),
                "{} has no description",
                skill.name
            );
            assert!(
                skill.is_builtin(),
                "{} should have no directory",
                skill.name
            );
            // A skill that advertises an argument hint must place the
            // arguments itself; one that does not takes the appended-paragraph
            // fallback, which is fine.
            assert!(
                skill.argument_hint.is_empty() || skill.body.contains("$ARGUMENTS"),
                "{} hints at arguments but never uses them",
                skill.name
            );
        }
    }

    #[test]
    fn a_local_skill_replaces_a_builtin_of_the_same_name() {
        let root = temp_root("shadow-builtin");
        write_skill(
            &root,
            "debug",
            "---\nname: debug\n---\nmine, not the built-in\n",
        );
        let skills = load_layered(std::slice::from_ref(&root));
        let hits: Vec<&Skill> = skills.iter().filter(|s| s.name == "debug").collect();
        assert_eq!(hits.len(), 1, "one entry per name: {hits:?}");
        assert!(!hits[0].is_builtin());
        assert_eq!(hits[0].body, "mine, not the built-in\n");
        // The other built-ins are untouched by the override.
        assert!(skills.iter().any(|s| s.name == "verify" && s.is_builtin()));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_listing_marks_builtins() {
        let listing = render_list(&builtins());
        assert!(listing.contains("/code-review"), "{listing}");
        assert!(listing.contains("[built-in]"), "{listing}");
        // A skill read from disk is not marked.
        let local = vec![skill("mine", "body")];
        assert!(!render_list(&local).contains("[built-in]"));
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
            dir: PathBuf::from("/tmp/plank-test-skill"),
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
