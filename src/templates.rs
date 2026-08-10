// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Prompt templates: single Markdown files with `{{var}}` interpolation,
//! surfaced as `/name` slash commands (issue #67).
//!
//! A template is one `*.md` file whose stem is the command name, with the
//! same optional `---` frontmatter as a skill (`name`, `description`,
//! `argument-hint`) followed by the prompt body. `/name args...` fills the
//! body's `{{var}}` placeholders and submits the result as the user turn.
//!
//! Templates are the lighter-weight sibling of [`crate::skills`]: a skill is
//! a directory with a `SKILL.md` and a single `$ARGUMENTS` splice, a template
//! is a bare file with named holes. Discovery is identical — the global
//! `~/.plank/templates/` is loaded first, then the project's
//! `./.plank/templates/`, and a project template overrides a global one with
//! the same name. Built-in slash commands always win over a template name.
//!
//! Interpolation is deliberately minimal: `{{name}}` (whitespace around the
//! name is ignored) and nothing else — no conditionals, no expressions.
//! `{{{{` and `}}}}` are escapes for literal `{{` and `}}`. A placeholder
//! with no value is a hard error listing every missing variable rather than a
//! silent empty substitution, and naming a variable the template does not
//! declare is an error too.

use std::path::{Path, PathBuf};

/// One loaded prompt template.
#[derive(Debug, Clone)]
pub struct Template {
    /// Slash-command name (no leading `/`); defaults to the file stem.
    pub name: String,
    /// One-line description shown by `/templates`.
    pub description: String,
    /// Hint describing what to pass as arguments, shown by `/templates`.
    pub argument_hint: String,
    /// Markdown prompt body (frontmatter stripped).
    pub body: String,
    /// File the template was loaded from.
    pub path: PathBuf,
}

/// One piece of a parsed template body.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    /// Literal text, escapes already resolved.
    Literal(String),
    /// A `{{name}}` placeholder.
    Var(String),
}

/// True for a placeholder name plank is willing to route: non-empty, no
/// whitespace and no braces. Anything else keeps its literal `{{...}}` text.
fn valid_var(name: &str) -> bool {
    !name.is_empty()
        && !name.contains(char::is_whitespace)
        && !name.contains(['{', '}'])
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

/// Splits a body into literals and `{{var}}` placeholders. `{{{{` / `}}}}`
/// collapse to literal `{{` / `}}`; an unterminated or unusable placeholder is
/// left verbatim so a stray brace never eats the rest of the prompt.
fn parse_segments(body: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let mut lit = String::new();
    let mut rest = body;
    while !rest.is_empty() {
        let Some(idx) = rest.find(['{', '}']) else {
            lit.push_str(rest);
            break;
        };
        lit.push_str(&rest[..idx]);
        let tail = &rest[idx..];
        if let Some(after) = tail.strip_prefix("{{{{") {
            lit.push_str("{{");
            rest = after;
            continue;
        }
        if let Some(after) = tail.strip_prefix("}}}}") {
            lit.push_str("}}");
            rest = after;
            continue;
        }
        if let Some(after) = tail.strip_prefix("{{")
            && let Some(end) = after.find("}}")
            && valid_var(after[..end].trim())
        {
            if !lit.is_empty() {
                out.push(Segment::Literal(std::mem::take(&mut lit)));
            }
            out.push(Segment::Var(after[..end].trim().to_string()));
            rest = &after[end + 2..];
            continue;
        }
        // Not a placeholder: keep the brace and move past it.
        let ch = tail.chars().next().unwrap_or('{');
        lit.push(ch);
        rest = &tail[ch.len_utf8()..];
    }
    if !lit.is_empty() {
        out.push(Segment::Literal(lit));
    }
    out
}

/// The template's variables, in first-appearance order, deduplicated.
#[must_use]
pub fn variables(body: &str) -> Vec<String> {
    let mut vars: Vec<String> = Vec::new();
    for seg in parse_segments(body) {
        if let Segment::Var(name) = seg
            && !vars.contains(&name)
        {
            vars.push(name);
        }
    }
    vars
}

/// Loads one template from a `*.md` file; `None` when unusable.
fn load_file(path: &Path) -> Option<Template> {
    if path.extension()?.to_str()? != "md" {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    let (fields, body) = crate::skills::split_frontmatter(&text);
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
    // The name becomes a slash command: reject anything unroutable.
    if name.is_empty() || name.contains(char::is_whitespace) || name.contains('/') {
        return None;
    }
    if body.trim().is_empty() {
        return None;
    }
    Some(Template {
        name,
        description: get("description"),
        argument_hint: get("argument-hint"),
        body,
        path: path.to_path_buf(),
    })
}

/// Loads templates from `<root>/*.md`, sorted by name for stable listings.
fn load_dir(root: &Path) -> Vec<Template> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut templates: Vec<Template> = entries
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
        .filter_map(|e| load_file(&e.path()))
        .collect();
    templates.sort_by(|a, b| a.name.cmp(&b.name));
    templates
}

/// Loads templates from the given roots in order; a later root's template
/// replaces an earlier one with the same name (project overrides global).
#[must_use]
pub fn load_from(roots: &[PathBuf]) -> Vec<Template> {
    let mut merged: Vec<Template> = Vec::new();
    for root in roots {
        for tpl in load_dir(root) {
            if let Some(existing) = merged.iter_mut().find(|t| t.name == tpl.name) {
                *existing = tpl;
            } else {
                merged.push(tpl);
            }
        }
    }
    merged.sort_by(|a, b| a.name.cmp(&b.name));
    merged
}

/// Loads templates from the default hierarchy: `~/.plank/templates` overlaid
/// by `<cwd>/.plank/templates`.
#[must_use]
pub fn load_default(cwd: &Path) -> Vec<Template> {
    let mut roots = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".plank").join("templates"));
    }
    roots.push(cwd.join(".plank").join("templates"));
    load_from(&roots)
}

/// Splits an argument string into whitespace-separated words, honouring
/// double quotes so `title="two words"` and `"two words"` stay one word.
fn split_words(arg: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut started = false;
    for c in arg.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            c if c.is_whitespace() && !quoted => {
                if started {
                    words.push(std::mem::take(&mut cur));
                    started = false;
                }
            }
            c => {
                cur.push(c);
                started = true;
            }
        }
    }
    if started {
        words.push(cur);
    }
    words
}

/// Binds `arg` to the template's variables. `key=value` words bind by name;
/// remaining words fill the still-unbound variables in first-appearance
/// order, and any surplus words are appended to the last variable so a prose
/// argument is never truncated.
///
/// # Errors
/// Returns a message naming the unknown variables (a `key=value` the template
/// does not declare) or the missing ones (declared but never bound).
fn bind(tpl: &Template, arg: &str) -> Result<Vec<(String, String)>, String> {
    let vars = variables(&tpl.body);
    let mut bound: Vec<(String, String)> = Vec::new();
    let mut positional: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    for word in split_words(arg) {
        match word.split_once('=') {
            Some((k, v)) if valid_var(k) => {
                if vars.iter().any(|n| n == k) {
                    if let Some(slot) = bound.iter_mut().find(|(n, _)| n == k) {
                        slot.1 = v.to_string();
                    } else {
                        bound.push((k.to_string(), v.to_string()));
                    }
                } else if !unknown.iter().any(|n| n == k) {
                    unknown.push(k.to_string());
                }
            }
            _ => positional.push(word),
        }
    }
    if !unknown.is_empty() {
        return Err(format!(
            "/{}: unknown variable{} {} (this template takes: {})",
            tpl.name,
            if unknown.len() == 1 { "" } else { "s" },
            unknown.join(", "),
            if vars.is_empty() {
                "no variables".to_string()
            } else {
                vars.join(", ")
            }
        ));
    }
    let mut open: Vec<&String> = vars
        .iter()
        .filter(|n| !bound.iter().any(|(b, _)| &b == n))
        .collect();
    let mut words = positional.into_iter();
    while open.len() > 1 {
        let name = open.remove(0);
        let Some(w) = words.next() else { break };
        bound.push((name.clone(), w));
    }
    if let Some(name) = open.first() {
        let rest: Vec<String> = words.collect();
        if !rest.is_empty() {
            bound.push(((*name).clone(), rest.join(" ")));
        }
    }
    let missing: Vec<&str> = vars
        .iter()
        .filter(|n| !bound.iter().any(|(b, _)| b == *n))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "/{}: missing value{} for {} (usage: /{} {})",
            tpl.name,
            if missing.len() == 1 { "" } else { "s" },
            missing.join(", "),
            tpl.name,
            usage_args(tpl, &vars)
        ));
    }
    Ok(bound)
}

/// The argument spelling shown in errors and listings: the explicit
/// `argument-hint` when set, otherwise `<var> <var>` from the body.
fn usage_args(tpl: &Template, vars: &[String]) -> String {
    if !tpl.argument_hint.is_empty() {
        return tpl.argument_hint.clone();
    }
    vars.iter()
        .map(|v| format!("<{v}>"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Renders `/<name> <arg>` into the user-turn prompt.
///
/// # Errors
/// Returns a human-readable message when a variable is unknown or unbound;
/// nothing is ever substituted with an empty string silently.
pub fn render(tpl: &Template, arg: &str) -> Result<String, String> {
    let bound = bind(tpl, arg)?;
    let mut out = String::with_capacity(tpl.body.len());
    for seg in parse_segments(&tpl.body) {
        match seg {
            Segment::Literal(text) => out.push_str(&text),
            Segment::Var(name) => {
                let value = bound
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| v.as_str())
                    .unwrap_or_default();
                out.push_str(value);
            }
        }
    }
    Ok(out)
}

/// Resolves a slash command against the loaded templates. Built-in commands
/// always win, so a template can never shadow `/help` and friends; `None`
/// means "not a template, keep looking".
#[must_use]
pub fn resolve<'a>(templates: &'a [Template], cmd: &str) -> Option<&'a Template> {
    if crate::config::slash_command_known(cmd) {
        return None;
    }
    let name = cmd.strip_prefix('/')?;
    templates.iter().find(|t| t.name == name)
}

/// Renders the `/templates` listing.
#[must_use]
pub fn render_list(templates: &[Template]) -> String {
    if templates.is_empty() {
        return "no templates found (checked ~/.plank/templates and ./.plank/templates)\n"
            .to_string();
    }
    let mut out = String::from("Prompt templates (invoke with /<name> [values]):\n");
    // An uncontested plugin template holds both its bare name and its
    // `<plugin>:<name>` alias; `listing` shows it once and names the plugin.
    for listed in crate::plugins::listing(templates) {
        let t = listed.entry;
        out.push_str("  /");
        out.push_str(listed.name);
        let vars = variables(&t.body);
        let args = usage_args(t, &vars);
        if !args.is_empty() {
            out.push(' ');
            out.push_str(&args);
        }
        if !t.description.is_empty() {
            out.push_str(" — ");
            out.push_str(&t.description);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_template(root: &Path, file: &str, content: &str) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join(file), content).unwrap();
    }

    fn temp_root(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("plank-templates-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn tpl(name: &str, body: &str) -> Template {
        Template {
            name: name.to_string(),
            description: String::new(),
            argument_hint: String::new(),
            body: body.to_string(),
            path: PathBuf::new(),
        }
    }

    #[test]
    fn loads_stem_as_name_with_optional_frontmatter() {
        let root = temp_root("load");
        write_template(&root, "review.md", "Review {{path}} for {{topic}}.\n");
        write_template(
            &root,
            "ship.md",
            "---\ndescription: Ship it\nargument-hint: <version>\n---\nShip {{version}}.\n",
        );
        write_template(&root, "empty.md", "---\nname: empty\n---\n  \n");
        write_template(&root, "notes.txt", "ignored {{x}}\n");
        let templates = load_from(std::slice::from_ref(&root));
        assert_eq!(templates.len(), 2, "{templates:?}");
        assert_eq!(templates[0].name, "review");
        assert_eq!(templates[1].name, "ship");
        assert_eq!(templates[1].description, "Ship it");
        assert_eq!(templates[1].argument_hint, "<version>");
        assert_eq!(templates[1].body, "Ship {{version}}.\n");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn project_overrides_user_by_name() {
        let user = temp_root("user");
        let project = temp_root("project");
        write_template(&user, "deploy.md", "user body {{env}}\n");
        write_template(&user, "only-user.md", "user-only body\n");
        write_template(&project, "deploy.md", "project body {{env}}\n");
        let templates = load_from(&[user.clone(), project.clone()]);
        let deploy = templates.iter().find(|t| t.name == "deploy").unwrap();
        assert_eq!(deploy.body, "project body {{env}}\n");
        assert_eq!(deploy.path.parent().unwrap(), project.as_path());
        assert!(templates.iter().any(|t| t.name == "only-user"));
        std::fs::remove_dir_all(&user).ok();
        std::fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn variables_are_first_appearance_order_and_deduplicated() {
        assert_eq!(
            variables("{{b}} then {{ a }} then {{b}}"),
            vec!["b".to_string(), "a".to_string()]
        );
        assert!(variables("no holes here").is_empty());
    }

    #[test]
    fn renders_named_and_positional_arguments() {
        let t = tpl("review", "Review {{path}} for {{topic}}.");
        assert_eq!(
            render(&t, "src/ui.rs topic=races").unwrap(),
            "Review src/ui.rs for races."
        );
        assert_eq!(
            render(&t, "src/ui.rs data races everywhere").unwrap(),
            "Review src/ui.rs for data races everywhere.",
            "surplus words land in the last variable"
        );
        assert_eq!(
            render(&t, "\"src/a b.rs\" locks").unwrap(),
            "Review src/a b.rs for locks."
        );
    }

    #[test]
    fn missing_variables_are_a_clear_error() {
        let t = tpl("review", "Review {{path}} for {{topic}}.");
        let err = render(&t, "src/ui.rs").unwrap_err();
        assert!(err.contains("missing value for topic"), "{err}");
        assert!(err.contains("usage: /review <path> <topic>"), "{err}");
        let err2 = render(&t, "").unwrap_err();
        assert!(err2.contains("missing values for path, topic"), "{err2}");
    }

    #[test]
    fn unknown_variables_are_rejected() {
        let t = tpl("review", "Review {{path}}.");
        let err = render(&t, "path=x nope=1").unwrap_err();
        assert!(err.contains("unknown variable nope"), "{err}");
        assert!(err.contains("this template takes: path"), "{err}");
        let bare = tpl("bare", "no holes");
        let err2 = render(&bare, "k=v").unwrap_err();
        assert!(err2.contains("no variables"), "{err2}");
        assert_eq!(render(&bare, "").unwrap(), "no holes");
    }

    #[test]
    fn escapes_and_stray_braces_stay_literal() {
        let t = tpl(
            "json",
            "{{{{ \"a\": {{v}} }}}} and { lone } and {{ bad name }}",
        );
        assert_eq!(
            render(&t, "v=1").unwrap(),
            "{{ \"a\": 1 }} and { lone } and {{ bad name }}"
        );
        assert_eq!(variables(t.body.as_str()), vec!["v".to_string()]);
        let unterminated = tpl("u", "half {{open and done");
        assert!(variables(&unterminated.body).is_empty());
        assert_eq!(render(&unterminated, "").unwrap(), "half {{open and done");
    }

    #[test]
    fn builtins_win_over_a_template_of_the_same_name() {
        let templates = vec![tpl("help", "shadow attempt"), tpl("review", "Review.")];
        assert!(resolve(&templates, "/help").is_none());
        assert!(resolve(&templates, "/skills").is_none());
        assert_eq!(resolve(&templates, "/review").unwrap().name, "review");
        assert!(resolve(&templates, "/nope").is_none());
        assert!(resolve(&templates, "review").is_none());
    }

    #[test]
    fn listing_shows_variables_and_description() {
        let mut t = tpl("review", "Review {{path}} for {{topic}}.");
        t.description = "Review a file".to_string();
        let list = render_list(std::slice::from_ref(&t));
        assert!(
            list.contains("/review <path> <topic> — Review a file"),
            "{list}"
        );
        t.argument_hint = "<file> <concern>".to_string();
        assert!(render_list(&[t]).contains("/review <file> <concern>"));
        assert!(render_list(&[]).contains("no templates found"));
    }
}
