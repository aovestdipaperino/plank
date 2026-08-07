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

use crate::remote::provider::ProviderKind;
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
    /// Engine override; `None` runs the subagent on the parent's engine, as
    /// every definition did before cross-provider subagents.
    pub engine: Option<AgentEngine>,
    /// Whether the model may select this definition on its own initiative.
    /// Defaults true; `auto: false` makes it `/subagent`-only.
    pub auto: bool,
    /// Run this sub-agent in its own throwaway git worktree
    /// (`isolation: worktree`), so its edits cannot collide with the parent's.
    /// Defaults false, and to the `worktree.isolateAgents` setting when the
    /// frontmatter is silent — a checkout per agent is not free.
    pub isolate: bool,
}

/// Engine override for a named definition: what its sidechain runs on instead
/// of the parent's engine.
///
/// `provider: local` is deliberately distinct from *omitting* `provider:`.
/// Omitting it means "whatever the parent is", which under `--provider` is the
/// remote model; `local` means the ds4 engine specifically, and makes plank load
/// it even when the main agent is remote. Two different intentions that used to
/// be spelled the same way.
#[derive(Debug, Clone)]
pub enum AgentEngine {
    /// The local ds4 engine, whatever the main agent runs on.
    Local,
    /// A provider-backed engine.
    Provider(ProviderSpec),
}

/// Which provider and model a definition's sidechain runs on.
///
/// The key *value* is deliberately absent — only the variable's *name* is
/// configurable, so a definition file stays committable to a shared repo while
/// still selecting the right secret. One provider protocol can front several
/// endpoints that do not share credentials (two gateways, work vs. personal),
/// which a single global default cannot address.
#[derive(Debug, Clone)]
pub struct ProviderSpec {
    /// Which provider wire protocol to speak.
    pub kind: ProviderKind,
    /// Provider-side model name, e.g. `claude-opus-5`.
    pub model: String,
    /// Base URL override; `None` uses the provider default.
    pub base_url: Option<String>,
    /// Context window; `None` asks the provider at first dispatch.
    pub ctx: Option<i32>,
    /// Environment variable holding this definition's API key. Resolved at
    /// load: the frontmatter's `api-key-env:` when given, else the provider
    /// default. Never empty, so every reader has one name to consult.
    pub api_key_env: String,
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
    // Engine override: `provider:` opts in, and requires a `model:`. A
    // half-specified or unknown provider makes the definition unusable rather
    // than silently running on the parent's engine — a definition that names a
    // provider clearly means to use it.
    let engine = match get("provider").as_str() {
        "" => None,
        // No model, key or URL to give: it is this process's own engine.
        "local" => Some(AgentEngine::Local),
        provider => {
            let kind = ProviderKind::parse(provider)?;
            let model = get("model");
            if model.is_empty() {
                return None;
            }
            // An unparseable ctx is ignored, not fatal: the provider is asked
            // instead, which is the same path as omitting the key.
            let ctx = get("ctx").parse::<i32>().ok().filter(|c| *c > 0);
            // Resolve the key variable once, here, so no downstream reader has
            // to re-apply the default.
            let api_key_env = match get("api-key-env") {
                v if v.is_empty() => kind.api_key_env().to_string(),
                v => v,
            };
            Some(AgentEngine::Provider(ProviderSpec {
                kind,
                model,
                base_url: Some(get("base-url")).filter(|s| !s.is_empty()),
                ctx,
                api_key_env,
            }))
        }
    };
    Some(AgentDef {
        name,
        description: get("description"),
        body,
        path: path.to_path_buf(),
        engine,
        auto: get("auto") != "false",
        isolate: match get("isolation").as_str() {
            "worktree" => true,
            "" => crate::settings::active().worktree.isolate_agents,
            _ => false,
        },
    })
}

/// Prefixes a definition's instructions with a notice that this sub-agent is
/// running in an isolated worktree.
///
/// Without it the sub-agent would keep using absolute paths inherited from the
/// parent's message — which point at the *main* checkout — and its edits would
/// land exactly where the isolation was supposed to keep them out of.
#[must_use]
pub fn worktree_notice(instructions: Option<&str>, worktree: &std::path::Path) -> String {
    let notice = format!(
        "You are running in an isolated git worktree at {}. It is a complete checkout of the \
         repository on its own branch. Treat it as the project root: any absolute path you were \
         given refers to the main checkout, so translate it to the matching path under this \
         worktree before reading or editing. Your changes stay here and do not affect the main \
         working copy.",
        worktree.display()
    );
    match instructions {
        Some(text) if !text.trim().is_empty() => format!("{notice}\n\n{text}"),
        _ => notice,
    }
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

/// Whether a remote-backed definition's API-key variable is absent, making it
/// undispatchable. Always false for a definition with no engine override.
fn missing_key(def: &AgentDef) -> bool {
    match &def.engine {
        Some(AgentEngine::Provider(p)) => {
            !std::env::var(&p.api_key_env).is_ok_and(|v| !v.trim().is_empty())
        }
        // A local definition has no credential to be missing. Whether the local
        // engine is actually loaded is a startup question, not a roster one.
        Some(AgentEngine::Local) | None => false,
    }
}

/// The definitions the model may select on its own initiative.
///
/// Three gates, all of which must pass: the definition opted in (`auto`),
/// model-initiated routing is enabled globally (`auto_route`), and — for a
/// remote-backed definition — its own API-key variable is actually set. A
/// definition failing the last gate silently vanishing from the model's view is
/// correct; it stays listed by [`render_list`] with the reason.
///
/// `/subagent <name>` deliberately does not consult this: the gates govern
/// *model* initiative, never what the user can ask for.
#[must_use]
pub fn model_visible(defs: &[AgentDef], auto_route: bool) -> Vec<&AgentDef> {
    if !auto_route {
        return Vec::new();
    }
    defs.iter().filter(|d| d.auto && !missing_key(d)).collect()
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
    use std::fmt::Write as _;
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
        // A remote-backed definition names its engine, and — when its key
        // variable is unset — the exact variable to set. The model never sees
        // such a definition (see `model_visible`), so this listing is the only
        // place the reason is visible.
        match &d.engine {
            Some(AgentEngine::Local) => {
                let _ = write!(out, " [local]");
            }
            Some(AgentEngine::Provider(p)) => {
                let _ = write!(out, " [{} {}]", p.kind.label(), p.model);
                if missing_key(d) {
                    let _ = write!(out, " (no {})", p.api_key_env);
                }
            }
            None => {}
        }
        out.push('\n');
    }
    // The roster is only half the story: whether the *model* may reach for these
    // is a setting, so say which one and what it currently is. Both `/agent`
    // front ends render this string, so the two cannot disagree.
    let s = crate::settings::active();
    let _ = writeln!(
        out,
        "\nModel may pick these on its own: {} (/config agents.autoRoute), \
         up to {} at once (/config agents.maxParallel).",
        if s.agents.auto_route { "yes" } else { "no" },
        s.agents.max_parallel
    );
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

/// Pushed as the sidechain's last user turn when its round budget is spent.
///
/// A subagent that keeps calling tools until the budget runs out would otherwise
/// return nothing at all, discarding everything it found. Forcing a text answer
/// converts exhaustion into a usable report. Shared by the serial loop and the
/// parallel fan-out so the two cannot drift.
#[must_use]
pub fn final_round_reminder() -> String {
    "<system-reminder>\n\
     This is your final turn. Do not call any more tools — any tool call you \
     make now is discarded. Write your final report as plain text now, using \
     what you have already found.\n\
     </system-reminder>"
        .to_string()
}

/// Frames the subagent's final report for the parent conversation.
///
/// The framing tells the model to *continue from* the report, because a turn
/// runs on it as soon as it lands: the report is the delegated work coming
/// back, not background reading. It also says not to redo that work — the
/// sidechain is truncated out of the transcript, so the tool calls behind the
/// report are gone and re-running them is the tempting failure mode.
#[must_use]
pub fn report_message(task: &str, report: &str) -> String {
    format!(
        "<system-reminder>\n\
         A subagent completed the delegated task: {}\n\
         Its final report follows. Continue from it: act on what it found, or \
         answer the user with it. Do not repeat the work it already did.\n\
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
    fn engine_frontmatter_parses() {
        let root = temp_root("engine-fm");
        write_def(
            &root,
            "reviewer.md",
            "---\nname: reviewer\ndescription: reviews diffs\nprovider: anthropic\n\
             model: claude-opus-5\nbase-url: https://gw.example/v1\nctx: 200000\n\
             api-key-env: ANTHROPIC_API_KEY_WORK\n---\nBe exacting.\n",
        );
        let defs = load_from(std::slice::from_ref(&root));
        assert_eq!(defs.len(), 1, "{defs:?}");
        let Some(AgentEngine::Provider(e)) = defs[0].engine.as_ref() else {
            panic!("expected a provider spec, got {:?}", defs[0].engine);
        };
        assert_eq!(e.kind, crate::remote::provider::ProviderKind::Anthropic);
        assert_eq!(e.model, "claude-opus-5");
        assert_eq!(e.base_url.as_deref(), Some("https://gw.example/v1"));
        assert_eq!(e.ctx, Some(200_000));
        assert_eq!(
            e.api_key_env, "ANTHROPIC_API_KEY_WORK",
            "explicit override wins"
        );
        assert!(defs[0].auto, "auto defaults true");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn engine_defaults_and_omissions() {
        let root = temp_root("engine-def");
        write_def(
            &root,
            "plain.md",
            "---\nname: plain\ndescription: local\n---\nBody.\n",
        );
        write_def(
            &root,
            "minimal.md",
            "---\nname: minimal\nprovider: openai\nmodel: gpt-5\n---\nBody.\n",
        );
        write_def(
            &root,
            "opted-out.md",
            "---\nname: opted-out\nauto: false\n---\nBody.\n",
        );
        let defs = load_from(std::slice::from_ref(&root));
        let by = |n: &str| defs.iter().find(|d| d.name == n).expect("def");
        assert!(by("plain").engine.is_none(), "no provider -> no engine");
        assert!(by("plain").auto);
        let Some(AgentEngine::Provider(m)) = by("minimal").engine.as_ref() else {
            panic!("expected a provider spec");
        };
        assert!(m.base_url.is_none(), "base-url omitted -> None");
        assert!(m.ctx.is_none(), "ctx omitted -> None");
        assert_eq!(
            m.api_key_env, "OPENAI_API_KEY",
            "api-key-env omitted -> the provider default"
        );
        assert!(!by("opted-out").auto, "auto: false is honored");
        std::fs::remove_dir_all(&root).ok();
    }

    /// `provider: local` is its own thing: no model, key or URL to give, and
    /// distinct from omitting `provider:` (which means "the parent's engine").
    #[test]
    fn provider_local_parses_and_needs_nothing_else() {
        let root = temp_root("engine-local");
        write_def(
            &root,
            "cheap.md",
            "---\nname: cheap\ndescription: Runs on the local model\nprovider: local\n---\nBody.\n",
        );
        write_def(
            &root,
            "inherits.md",
            "---\nname: inherits\ndescription: Runs on whatever the parent is\n---\nBody.\n",
        );
        let defs = load_from(std::slice::from_ref(&root));
        let by = |n: &str| defs.iter().find(|d| d.name == n).expect("def");
        assert!(
            matches!(by("cheap").engine, Some(AgentEngine::Local)),
            "provider: local -> the local engine, got {:?}",
            by("cheap").engine
        );
        assert!(
            by("inherits").engine.is_none(),
            "omitting provider: still means the parent's engine"
        );
        // No credential to be missing, so it is never hidden from the model.
        assert!(!missing_key(by("cheap")));

        let listing = render_list(&defs);
        assert!(
            listing.contains("cheap — Runs on the local model [local]"),
            "{listing}"
        );
        assert!(
            !listing.contains("inherits — Runs on whatever the parent is ["),
            "a parent-engine definition names no engine: {listing}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn broken_engine_frontmatter_rejects_the_definition() {
        let root = temp_root("engine-bad");
        // A provider without a model is unusable.
        write_def(
            &root,
            "nomodel.md",
            "---\nname: nomodel\nprovider: anthropic\n---\nBody.\n",
        );
        // An unknown provider name is unusable.
        write_def(
            &root,
            "bogus.md",
            "---\nname: bogus\nprovider: gemini\nmodel: x\n---\nBody.\n",
        );
        // A non-numeric ctx is ignored, not fatal — the rest of the def stands.
        write_def(
            &root,
            "badctx.md",
            "---\nname: badctx\nprovider: openai\nmodel: gpt-5\nctx: lots\n---\nBody.\n",
        );
        let defs = load_from(std::slice::from_ref(&root));
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["badctx"], "only the recoverable one survives");
        let Some(AgentEngine::Provider(p)) = defs[0].engine.as_ref() else {
            panic!("expected a provider spec");
        };
        assert_eq!(p.ctx, None);
        std::fs::remove_dir_all(&root).ok();
    }

    /// Builds a def with an explicit, test-only key variable. Using a private
    /// variable name keeps these tests hermetic: they never depend on (or race
    /// with) a real `ANTHROPIC_API_KEY` in the developer's environment.
    fn remote_def(name: &str, key_env: &str, auto: bool) -> AgentDef {
        AgentDef {
            name: name.to_string(),
            description: String::new(),
            body: "Body.".to_string(),
            path: PathBuf::from(format!("/tmp/{name}.md")),
            engine: Some(AgentEngine::Provider(ProviderSpec {
                kind: ProviderKind::Anthropic,
                model: "m".to_string(),
                base_url: None,
                ctx: None,
                api_key_env: key_env.to_string(),
            })),
            auto,
            isolate: false,
        }
    }

    fn local_def(name: &str, auto: bool) -> AgentDef {
        AgentDef {
            name: name.to_string(),
            description: String::new(),
            body: "Body.".to_string(),
            path: PathBuf::from(format!("/tmp/{name}.md")),
            engine: None,
            auto,
            isolate: false,
        }
    }

    #[test]
    fn model_visible_applies_every_gate() {
        const KEY: &str = "PLANK_TEST_VISIBLE_KEY";
        let defs = vec![
            local_def("local", true),
            local_def("hidden", false),
            remote_def("keyed", KEY, true),
        ];
        let names = |auto_route| -> Vec<String> {
            model_visible(&defs, auto_route)
                .iter()
                .map(|d| d.name.clone())
                .collect()
        };

        unsafe { std::env::remove_var(KEY) };
        assert_eq!(names(true), vec!["local"], "no key -> remote def is hidden");

        unsafe { std::env::set_var(KEY, "sk-test") };
        assert_eq!(
            names(true),
            vec!["local", "keyed"],
            "keyed def appears once its own variable is set; auto:false never does"
        );

        assert!(
            names(false).is_empty(),
            "auto_route off withholds the whole roster"
        );
        unsafe { std::env::remove_var(KEY) };
    }

    #[test]
    fn a_definition_reads_its_own_key_variable_not_the_provider_default() {
        const PINNED: &str = "PLANK_TEST_PINNED_KEY";
        let defs = vec![remote_def("pinned", PINNED, true)];
        // The provider default being set must not satisfy a def that pinned its
        // own variable — otherwise a work/personal split silently collapses.
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "sk-default") };
        unsafe { std::env::remove_var(PINNED) };
        assert!(
            model_visible(&defs, true).is_empty(),
            "provider default does not satisfy a pinned variable"
        );
        unsafe { std::env::set_var(PINNED, "sk-test") };
        assert_eq!(model_visible(&defs, true).len(), 1, "own variable does");
        unsafe { std::env::remove_var(PINNED) };
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
    }

    #[test]
    fn render_list_names_the_engine_and_the_missing_variable() {
        const KEY: &str = "PLANK_TEST_RENDER_KEY";
        let mut def = remote_def("keyed", KEY, true);
        def.description = "needs a key".to_string();
        let defs = vec![def];

        unsafe { std::env::remove_var(KEY) };
        let out = render_list(&defs);
        assert!(out.contains("keyed — needs a key"), "{out}");
        assert!(out.contains("[anthropic m]"), "names the engine: {out}");
        assert!(
            out.contains(&format!("(no {KEY})")),
            "names the exact variable to set: {out}"
        );

        // With the key present the marker disappears; the engine label stays.
        unsafe { std::env::set_var(KEY, "sk-test") };
        let out = render_list(&defs);
        assert!(out.contains("[anthropic m]"), "{out}");
        assert!(!out.contains("(no "), "no marker once set: {out}");
        unsafe { std::env::remove_var(KEY) };

        // A local definition gets no engine label at all.
        assert!(!render_list(&[local_def("plain", true)]).contains('['));
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
            engine: None,
            auto: true,
            isolate: false,
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

    #[test]
    fn resolve_ignores_the_auto_gate() {
        // `/subagent <name>` is explicit user dispatch: it must reach a
        // definition the *model* is not allowed to select. The `auto` gate and
        // `agents.autoRoute` govern model initiative only, never what the user
        // can ask for by name.
        let defs = vec![local_def("hidden", false)];
        let (def, task) = resolve(&defs, "hidden do the thing");
        assert_eq!(
            def.expect("auto:false is still user-dispatchable").name,
            "hidden"
        );
        assert_eq!(task, "do the thing");
        // …while the model is offered nothing.
        assert!(model_visible(&defs, true).is_empty());
    }
}
