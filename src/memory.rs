// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Layered persistent memory loaded into session-start context (issue #2).
//!
//! Memory is two plain markdown files, layered like the other `.plank`
//! configs: `~/.plank/MEMORY.md` (user scope — who the user is, durable
//! preferences) and `<cwd>/.plank/MEMORY.md` (project scope — goals and
//! constraints of this checkout). Both are loaded at session start and
//! injected into the context message, so the model sees them before the
//! first user turn.
//!
//! Entries are appended with `/remember [user] <text>` as dated bullets.
//! The file template documents the four entry types worth keeping — facts
//! the model cannot re-derive from the repository:
//! `user` (who the user is), `feedback` (corrections on how to work),
//! `project` (goals/constraints not in the code), `reference` (external
//! URLs/tickets/dashboards).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Template written when a memory file is first created.
const TEMPLATE: &str = "\
# Memory

Durable notes loaded into every session start. Keep entries to facts that
cannot be re-derived from the repository. Types: [user] who the user is,
[feedback] corrections on how to work, [project] goals and constraints,
[reference] external URLs/tickets/dashboards.
";

/// Memory scope selector for [`remember`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// `~/.plank/MEMORY.md` — follows the user across projects.
    User,
    /// `<cwd>/.plank/MEMORY.md` — tied to this checkout.
    Project,
}

/// Path of the memory file for a scope; `None` when `HOME` is unset for the
/// user scope.
#[must_use]
pub fn path_for(scope: Scope, cwd: &Path) -> Option<PathBuf> {
    match scope {
        Scope::User => {
            std::env::var_os("HOME").map(|h| crate::home::plank_home_in(h).join("MEMORY.md"))
        }
        Scope::Project => Some(cwd.join(".plank").join("MEMORY.md")),
    }
}

/// As [`path_for`], but for [`Scope::User`] an explicit `user_root` (when
/// given) is used in place of resolving `HOME`: the path becomes
/// `user_root.join("MEMORY.md")`. This is what lets the mutating,
/// both-scope functions (`forget_matching_to`, `apply_verdicts_to`) be
/// exercised in tests without ever reading or writing the real
/// `~/.plank/MEMORY.md` — the `Scope::Project` branch is untouched, since
/// that scope already gets its own hermetic redirection through `cwd`.
#[must_use]
fn scoped_path_for(scope: Scope, cwd: &Path, user_root: Option<&Path>) -> Option<PathBuf> {
    match (scope, user_root) {
        (Scope::User, Some(root)) => Some(root.join("MEMORY.md")),
        _ => path_for(scope, cwd),
    }
}

/// Appends a dated bullet to the scope's memory file, creating it (with the
/// template header) on first use. Returns the file written.
///
/// # Errors
///
/// Returns a message when the file cannot be created or written.
pub fn remember(scope: Scope, cwd: &Path, text: &str, date: &str) -> Result<PathBuf, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("nothing to remember".to_string());
    }
    let Some(path) = path_for(scope, cwd) else {
        return Err("HOME is not set".to_string());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut body = match std::fs::read_to_string(&path) {
        Ok(existing) => existing,
        Err(_) => TEMPLATE.to_string(),
    };
    if !body.ends_with('\n') {
        body.push('\n');
    }
    let _ = writeln!(body, "- ({date}) {text}");
    std::fs::write(&path, body).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Per-type character budgets for the rendered memory section. These replace
/// the single file-level cap: a runaway `project` block can no longer
/// silently evict the `user` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budgets {
    /// Budget for `[user]` entries.
    pub user: usize,
    /// Budget for `[feedback]` entries.
    pub feedback: usize,
    /// Budget for `[project]` entries.
    pub project: usize,
    /// Budget for `[reference]` entries.
    pub reference: usize,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            user: 4096,
            feedback: 4096,
            project: 6144,
            reference: 2048,
        }
    }
}

impl Budgets {
    /// The budget for one kind.
    #[must_use]
    pub fn for_kind(&self, kind: Kind) -> usize {
        match kind {
            Kind::User => self.user,
            Kind::Feedback => self.feedback,
            Kind::Project => self.project,
            Kind::Reference => self.reference,
        }
    }
}

/// Chooses which entries render, per type, under the budgets.
///
/// Retracted entries are filtered out first and are never reported as
/// dropped — retraction is a model decision, not budget pressure, and the
/// bytes survive until a reconciliation pass removes the line.
///
/// Within a type, entries are ranked pinned-first, then by descending `uses`,
/// then by most recent `last_used`. Age is the last tiebreak rather than the
/// only rule, which inverts the old tail truncation: the oldest facts about a
/// user are usually the most durable ones.
///
/// Returns `(kept in file order, dropped)`.
#[must_use]
pub fn select_for_render(
    entries: &[Entry],
    meta: &MetaStore,
    budgets: &Budgets,
) -> (Vec<Entry>, Vec<Entry>) {
    let mut kept_ids: Vec<String> = Vec::new();
    let mut dropped: Vec<Entry> = Vec::new();

    for kind in Kind::ALL {
        let mut block: Vec<&Entry> = entries
            .iter()
            .filter(|e| e.kind == kind && !meta.get(&e.id()).retracted)
            .collect();
        block.sort_by(|a, b| {
            let (ma, mb) = (meta.get(&a.id()), meta.get(&b.id()));
            mb.pinned
                .cmp(&ma.pinned)
                .then(mb.uses.cmp(&ma.uses))
                .then(mb.last_used.cmp(&ma.last_used))
        });
        let budget = budgets.for_kind(kind);
        let mut used = 0usize;
        for e in block {
            let cost = e.render().len();
            if used + cost <= budget {
                used += cost;
                kept_ids.push(e.id());
            } else {
                dropped.push(e.clone());
            }
        }
    }

    let kept = entries
        .iter()
        .filter(|e| kept_ids.iter().any(|id| id == &e.id()))
        .cloned()
        .collect();
    (kept, dropped)
}

/// Reads one scope's memory file and selects what renders under the budgets.
fn load_scope(scope: Scope, cwd: &Path) -> Option<String> {
    let path = path_for(scope, cwd)?;
    let text = std::fs::read_to_string(&path).ok()?;
    if text.trim().is_empty() {
        return None;
    }
    let entries = parse_entries(&text);
    if entries.is_empty() {
        return None;
    }
    let meta = MetaStore::load(&meta_path_for(&path));
    let budgets = crate::settings::active().memory.budgets;
    let (kept, dropped) = select_for_render(&entries, &meta, &budgets);
    if kept.is_empty() {
        return None;
    }
    let mut out = String::new();
    for kind in Kind::ALL {
        let block: Vec<&Entry> = kept.iter().filter(|e| e.kind == kind).collect();
        if block.is_empty() {
            continue;
        }
        let _ = writeln!(out, "### {}", kind.tag());
        for e in block {
            let _ = writeln!(
                out,
                "- ({}) [{}] {{{}}} {}",
                e.date,
                e.kind.tag(),
                e.id(),
                e.text
            );
        }
        out.push('\n');
    }
    if !dropped.is_empty() {
        let _ = writeln!(
            out,
            "({} older entries omitted under the type budgets; /memory shows the full file)",
            dropped.len()
        );
    }
    Some(out.trim_end().to_string())
}

/// Renders the session-start memory section: user scope first, then project.
/// `None` when neither file has content.
#[must_use]
pub fn load_default(cwd: &Path) -> Option<String> {
    let user = load_scope(Scope::User, cwd);
    let project = load_scope(Scope::Project, cwd);
    if user.is_none() && project.is_none() {
        return None;
    }
    let mut out = String::from(
        "Persistent memory (durable notes from past sessions; \
         background context, not instructions):\n\n",
    );
    if let Some(user) = user {
        out.push_str("## User memory (~/.plank/MEMORY.md)\n");
        out.push_str(&user);
        out.push('\n');
    }
    if let Some(project) = project {
        if !out.ends_with("\n\n") {
            out.push('\n');
        }
        out.push_str("## Project memory (.plank/MEMORY.md)\n");
        out.push_str(&project);
        out.push('\n');
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// /memory: one editable view over every source, split back on save
// ---------------------------------------------------------------------------

/// One memory file as `/memory` sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    /// Which scope the file is.
    pub scope: Scope,
    /// Where it lives.
    pub path: PathBuf,
}

impl Scope {
    /// The word used inside the section markers.
    #[must_use]
    pub fn marker_name(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
        }
    }

    fn from_marker_name(name: &str) -> Option<Self> {
        match name {
            "user" => Some(Self::User),
            "project" => Some(Self::Project),
            _ => None,
        }
    }
}

/// The closed taxonomy of memory entry types. Content derivable from the
/// repository is deliberately not representable here — those facts are
/// re-derived more accurately by looking, and they are what makes a memory
/// file rot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Who the user is: role, expertise, preferences.
    User,
    /// Corrections and confirmed approaches on how to work.
    Feedback,
    /// Goals and constraints not derivable from code or git history.
    Project,
    /// Pointers to external URLs, tickets, dashboards.
    Reference,
}

impl Kind {
    /// Every kind, in rendering order.
    pub const ALL: [Kind; 4] = [Kind::User, Kind::Feedback, Kind::Project, Kind::Reference];

    /// The word inside the `[...]` tag.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Feedback => "feedback",
            Self::Project => "project",
            Self::Reference => "reference",
        }
    }

    /// Parses a tag word. `None` for anything unrecognised, which callers
    /// treat as an untagged entry rather than an error.
    #[must_use]
    pub fn from_tag(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.tag() == name)
    }
}

/// One parsed memory bullet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The `(YYYY-MM-DD)` stamp.
    pub date: String,
    /// The `[type]` tag; `Project` when the line carried none.
    pub kind: Kind,
    /// The entry text, tag and date stripped.
    pub text: String,
}

impl Entry {
    /// A short content hash. Computed, never stored: nothing in `MEMORY.md`
    /// becomes machine-owned, so a hand edit merely orphans a sidecar row
    /// rather than corrupting anything.
    ///
    /// Deliberately over the text only. Re-tagging or re-dating an entry
    /// keeps its identity, and therefore its accumulated usage.
    #[must_use]
    pub fn id(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.text.as_bytes());
        let digest = hasher.finalize();
        digest.iter().take(6).fold(String::new(), |mut acc, b| {
            use std::fmt::Write;
            let _ = write!(acc, "{b:02x}");
            acc
        })
    }

    /// The canonical bullet form.
    #[must_use]
    pub fn render(&self) -> String {
        format!("- ({}) [{}] {}\n", self.date, self.kind.tag(), self.text)
    }
}

/// Parses every bullet in a memory file body. Lines that are not bullets
/// (the template header, blank lines, prose) are skipped.
#[must_use]
pub fn parse_entries(body: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for line in body.lines() {
        let Some(rest) = line.trim_start().strip_prefix("- (") else {
            continue;
        };
        let Some((date, rest)) = rest.split_once(')') else {
            continue;
        };
        let rest = rest.trim_start();
        let (kind, text) = match rest.strip_prefix('[').and_then(|r| r.split_once(']')) {
            Some((tag, after)) => match Kind::from_tag(tag) {
                Some(k) => (k, after),
                None => (Kind::Project, rest),
            },
            None => (Kind::Project, rest),
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        out.push(Entry {
            date: date.trim().to_string(),
            kind,
            text: text.to_string(),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// MetaStore: per-entry sidecar with usage counters and pin state
// ---------------------------------------------------------------------------

/// Per-entry bookkeeping. Every field is advisory: the memory file renders
/// correctly with all of this at its default.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Meta {
    /// How many extraction passes judged this entry to have borne on the work.
    pub uses: u32,
    /// The date of the most recent such pass.
    pub last_used: String,
    /// Never evict, whatever the counters say. Some facts are used rarely and
    /// are catastrophic to lose.
    pub pinned: bool,
    /// Retracted by a model `forget`: hidden from rendering, bytes kept until
    /// a reconciliation pass drops the line.
    pub retracted: bool,
}

/// The sidecar for one memory file, keyed by [`Entry::id`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetaStore {
    rows: BTreeMap<String, Meta>,
}

/// The sidecar path for a memory file: the file's own path plus
/// `.meta.json`, so `/memory` never sees it as a source.
#[must_use]
pub fn meta_path_for(memory_path: &Path) -> PathBuf {
    let mut name = memory_path.as_os_str().to_os_string();
    name.push(".meta.json");
    PathBuf::from(name)
}

impl MetaStore {
    /// Reads a sidecar. Every failure — missing file, unreadable file,
    /// malformed JSON, wrong shape — yields an empty store, because the
    /// sidecar is never allowed to break memory loading.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        let Some(json) = crate::tools::mcp::json_parse(&text) else {
            return Self::default();
        };
        let crate::tools::mcp::Json::Obj(members) = json else {
            return Self::default();
        };
        let mut rows = BTreeMap::new();
        for (id, value) in members {
            let num = |k: &str| -> u32 {
                match value.get(k) {
                    Some(crate::tools::mcp::Json::Num(n)) => {
                        if *n >= 0.0 && *n <= f64::from(u32::MAX) {
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            {
                                *n as u32
                            }
                        } else if *n > 0.0 {
                            u32::MAX
                        } else {
                            0
                        }
                    }
                    _ => 0,
                }
            };
            let flag = |k: &str| matches!(value.get(k), Some(crate::tools::mcp::Json::Bool(true)));
            rows.insert(
                id,
                Meta {
                    uses: num("uses"),
                    last_used: value.str_or("last_used", "").to_string(),
                    pinned: flag("pinned"),
                    retracted: flag("retracted"),
                },
            );
        }
        Self { rows }
    }

    /// Writes the sidecar, creating the parent directory.
    ///
    /// # Errors
    ///
    /// Returns a message when the directory or file cannot be written.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        use crate::tools::mcp::json_escape;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut out = String::from("{\n");
        for (i, (id, meta)) in self.rows.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            out.push_str("  ");
            json_escape(&mut out, id);
            out.push_str(": {\"uses\": ");
            let _ = write!(out, "{}", meta.uses);
            out.push_str(", \"last_used\": ");
            json_escape(&mut out, &meta.last_used);
            let _ = write!(
                out,
                ", \"pinned\": {}, \"retracted\": {}}}",
                meta.pinned, meta.retracted
            );
        }
        out.push_str("\n}\n");
        std::fs::write(path, out).map_err(|e| e.to_string())
    }

    /// The row for an id, defaulted when absent.
    #[must_use]
    pub fn get(&self, id: &str) -> Meta {
        self.rows.get(id).cloned().unwrap_or_default()
    }

    /// Credits an entry with one use on `date`.
    pub fn bump(&mut self, id: &str, date: &str) {
        let row = self.rows.entry(id.to_string()).or_default();
        row.uses = row.uses.saturating_add(1);
        row.last_used = date.to_string();
    }

    /// Sets or clears retraction.
    pub fn set_retracted(&mut self, id: &str, value: bool) {
        self.rows.entry(id.to_string()).or_default().retracted = value;
    }

    /// Sets or clears the pin.
    pub fn set_pinned(&mut self, id: &str, value: bool) {
        self.rows.entry(id.to_string()).or_default().pinned = value;
    }

    /// Moves a row to a new id, which is what makes an `UPDATE` worth more
    /// than a delete-plus-add: an entry rephrased six times over a month
    /// keeps its accumulated usage instead of resetting to zero each time.
    pub fn carry(&mut self, old: &str, new: &str) {
        if let Some(row) = self.rows.remove(old) {
            self.rows.insert(new.to_string(), row);
        }
    }

    /// Drops rows with no corresponding live entry — the debris left behind
    /// when the user edits `MEMORY.md` by hand.
    pub fn gc(&mut self, live_ids: &[String]) {
        self.rows.retain(|id, _| live_ids.iter().any(|l| l == id));
    }
}

/// The memory sources for a checkout, user scope first. The user scope is
/// absent only when `HOME` is unset.
#[must_use]
pub fn sources_for(cwd: &Path) -> Vec<Source> {
    [Scope::User, Scope::Project]
        .into_iter()
        .filter_map(|scope| path_for(scope, cwd).map(|path| Source { scope, path }))
        .collect()
}

/// Opening line of the combined view, explaining the markup to whoever edits it.
const COMBINED_HEADER: &str = "\
<!-- plank memory: every source in one file. Edit inside the sections; each
     section is written back to the file named on its begin marker. Text you
     add between or after sections joins the section above it. Delete a
     section's markers to leave that file untouched. -->
";

/// Builds the combined `/memory` view: every source's full text (never the
/// injection-truncated form) between markers that name its scope and file.
/// A missing file appears as an empty section, so it can be created by
/// typing into it.
#[must_use]
pub fn combine(sources: &[Source]) -> String {
    let mut out = String::from(COMBINED_HEADER);
    for src in sources {
        let body = std::fs::read_to_string(&src.path).unwrap_or_default();
        let _ = write!(
            out,
            "\n<!-- plank-memory: begin {} {} -->\n",
            src.scope.marker_name(),
            src.path.display()
        );
        out.push_str(body.trim_end_matches('\n'));
        if !body.trim().is_empty() {
            out.push('\n');
        }
        let _ = writeln!(
            out,
            "<!-- plank-memory: end {} -->",
            src.scope.marker_name()
        );
    }
    out
}

/// Parses a begin marker line into its scope; the path after the scope is
/// informational and ignored, so a hand-edited path can never redirect a write.
fn parse_begin(line: &str) -> Option<Scope> {
    let rest = line.trim().strip_prefix("<!-- plank-memory: begin ")?;
    let name = rest.split_whitespace().next()?;
    Scope::from_marker_name(name)
}

fn parse_end(line: &str) -> Option<Scope> {
    let rest = line.trim().strip_prefix("<!-- plank-memory: end ")?;
    let name = rest.strip_suffix("-->")?.trim();
    Scope::from_marker_name(name)
}

/// Splits an edited combined view back into per-scope bodies.
///
/// The smart part is what happens to text outside the markers: lines between
/// or after sections are appended to the section above them (someone typing
/// a note under the project block meant it for the project file), and lines
/// before the first section, other than the header comment, go to the first
/// section. A scope whose markers were deleted is simply absent from the
/// result and its file is left alone. Returns an error only for markup that
/// cannot be interpreted: a begin without its end, or a scope opened twice.
///
/// # Errors
///
/// See above.
pub fn split(edited: &str) -> Result<Vec<(Scope, String)>, String> {
    let mut sections: Vec<(Scope, String)> = Vec::new();
    let mut open: Option<Scope> = None;
    let mut preamble = String::new();
    let mut in_header = false;
    // Blank lines right after an end marker are spacing between sections,
    // not content; they are dropped until the first real stray line.
    let mut just_closed = false;
    for line in edited.lines() {
        if let Some(scope) = parse_begin(line) {
            if open.is_some() {
                return Err(format!(
                    "memory markup: begin {} inside another section",
                    scope.marker_name()
                ));
            }
            if sections.iter().any(|(s, _)| *s == scope) {
                return Err(format!(
                    "memory markup: section {} appears twice",
                    scope.marker_name()
                ));
            }
            sections.push((scope, String::new()));
            open = Some(scope);
            continue;
        }
        if let Some(scope) = parse_end(line) {
            if open != Some(scope) {
                return Err(format!(
                    "memory markup: end {} without its begin",
                    scope.marker_name()
                ));
            }
            open = None;
            just_closed = true;
            continue;
        }
        if open.is_some() {
            if let Some((_, body)) = sections.last_mut() {
                body.push_str(line);
                body.push('\n');
            }
        } else if let Some((_, body)) = sections.last_mut() {
            // After a closed section: the note belongs to the block above.
            if just_closed && line.trim().is_empty() {
                continue;
            }
            just_closed = false;
            body.push_str(line);
            body.push('\n');
        } else {
            // Before the first section: skip the explanatory header comment,
            // keep anything the user typed.
            let t = line.trim();
            if t.starts_with("<!-- plank memory:") {
                in_header = true;
            }
            if in_header {
                if t.ends_with("-->") {
                    in_header = false;
                }
                continue;
            }
            preamble.push_str(line);
            preamble.push('\n');
        }
    }
    if let Some(scope) = open {
        return Err(format!(
            "memory markup: section {} is never closed",
            scope.marker_name()
        ));
    }
    if let Some((_, body)) = sections.first_mut().filter(|_| !preamble.trim().is_empty()) {
        *body = format!("{preamble}{body}");
    }
    for (_, body) in &mut sections {
        let trimmed = body.trim_end_matches('\n');
        let mut t = trimmed.to_owned();
        if !t.is_empty() {
            t.push('\n');
        }
        *body = t;
    }
    Ok(sections)
}

/// Writes an edited combined view back to its sources. Only files whose body
/// actually changed are written; a file is created when its section gained
/// text. Returns one line per source describing what happened.
///
/// # Errors
///
/// Markup errors from [`split`], or the first write failure.
pub fn apply(sources: &[Source], edited: &str) -> Result<Vec<String>, String> {
    let sections = split(edited)?;
    let mut report = Vec::new();
    for src in sources {
        let Some((_, body)) = sections.iter().find(|(s, _)| *s == src.scope) else {
            report.push(format!(
                "{}: markers removed, left untouched",
                src.path.display()
            ));
            continue;
        };
        let current = std::fs::read_to_string(&src.path).unwrap_or_default();
        let unchanged = current.trim_end_matches('\n') == body.trim_end_matches('\n');
        if unchanged {
            report.push(format!("{}: unchanged", src.path.display()));
            continue;
        }
        if body.is_empty() && current.is_empty() {
            continue;
        }
        if let Some(parent) = src.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&src.path, body).map_err(|e| format!("{}: {e}", src.path.display()))?;
        report.push(format!(
            "{}: wrote {} line(s)",
            src.path.display(),
            body.lines().count()
        ));
    }
    Ok(report)
}

/// Where the maintenance audit log lives. `None` when `HOME` is unset.
#[must_use]
pub fn log_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| crate::home::plank_home_in(h).join("memory-log.jsonl"))
}

/// Appends one audit line to an explicit path. Best-effort: a write failure
/// is swallowed, because failing to log must never fail the operation being
/// logged.
fn append_log_line(path: &Path, action: &str, scope: Scope, id: &str, text: &str, reason: &str) {
    use crate::tools::mcp::json_escape;
    use std::io::Write as _;
    let mut line = String::from("{\"action\": ");
    json_escape(&mut line, action);
    line.push_str(", \"scope\": ");
    json_escape(&mut line, scope.marker_name());
    line.push_str(", \"id\": ");
    json_escape(&mut line, id);
    line.push_str(", \"text\": ");
    json_escape(&mut line, text);
    line.push_str(", \"reason\": ");
    json_escape(&mut line, reason);
    line.push_str("}\n");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Records one memory change. Best-effort, as above.
pub fn log_change(action: &str, scope: Scope, id: &str, text: &str, reason: &str) {
    if let Some(path) = log_path() {
        append_log_line(&path, action, scope, id, text, reason);
    }
}

/// As [`log_change`], but `log_dest` overrides where the audit line lands:
/// `Some(path)` writes there instead of resolving `~/.plank` from `HOME`.
/// Mirrors `apply_verdicts_to`'s `log_dest`, and exists for the same reason —
/// it lets the `remember`/`forget` tools be tested without ever touching the
/// real `~/.plank` or setting `HOME`.
pub(crate) fn log_change_to(
    log_dest: Option<&Path>,
    action: &str,
    scope: Scope,
    id: &str,
    text: &str,
    reason: &str,
) {
    match log_dest {
        Some(path) => append_log_line(path, action, scope, id, text, reason),
        None => log_change(action, scope, id, text, reason),
    }
}

/// The last `limit` audit lines from an explicit path, oldest first.
#[must_use]
fn read_log_from(path: &Path, limit: usize) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let all: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = all.len().saturating_sub(limit);
    all[start..].iter().map(|s| (*s).to_string()).collect()
}

/// The last `limit` audit lines, oldest first.
#[must_use]
pub fn read_log(limit: usize) -> Vec<String> {
    log_path().map_or_else(Vec::new, |p| read_log_from(&p, limit))
}

/// One decision from the extraction pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// A new entry.
    Add {
        /// The entry text.
        text: String,
        /// Its type.
        kind: Kind,
        /// Which file it belongs in.
        scope: Scope,
    },
    /// Replace an existing entry's text in place, carrying its sidecar row.
    Update {
        /// The existing entry's id.
        id: String,
        /// The replacement text.
        text: String,
    },
    /// Remove an entry outright. The pass is the audited path, so this is a
    /// real deletion; a model `forget` sets retraction instead.
    Delete {
        /// The existing entry's id.
        id: String,
    },
    /// Credit an entry with having borne on the work.
    Used {
        /// The existing entry's id.
        id: String,
    },
}

/// Parses the pass's JSON verdict array.
///
/// # Errors
///
/// Returns a message when the text is not JSON or is not an array. A single
/// malformed element is skipped rather than failing the batch, because one
/// bad verdict should not discard a whole pass's work.
pub fn parse_verdicts(json: &str) -> Result<Vec<Verdict>, String> {
    use crate::tools::mcp::{Json, json_parse};
    let parsed = json_parse(json).ok_or_else(|| "verdicts are not valid JSON".to_string())?;
    let Json::Arr(items) = parsed else {
        return Err("verdicts must be a JSON array".to_string());
    };
    let mut out = Vec::new();
    for item in items {
        let id = item.str_or("id", "").to_string();
        let text = item.str_or("text", "").trim().to_string();
        match item.str_or("verdict", "") {
            "ADD" if !text.is_empty() => out.push(Verdict::Add {
                text,
                kind: Kind::from_tag(item.str_or("type", "project")).unwrap_or(Kind::Project),
                scope: if item.str_or("scope", "project") == "user" {
                    Scope::User
                } else {
                    Scope::Project
                },
            }),
            "UPDATE" if !id.is_empty() && !text.is_empty() => {
                out.push(Verdict::Update { id, text });
            }
            "DELETE" if !id.is_empty() => out.push(Verdict::Delete { id }),
            "USED" if !id.is_empty() => out.push(Verdict::Used { id }),
            _ => {}
        }
    }
    Ok(out)
}

/// Finds the line index holding the live entry with this id, if any. The
/// file always wins: a verdict naming an id with no matching line is simply
/// not found here, and callers skip it silently.
///
/// This also governs same-batch ordering: verdicts are applied in order
/// against `lines` as rewritten so far, so if a batch contains
/// `UPDATE{id:X}` followed by `USED{id:X}` or `DELETE{id:X}` naming the
/// *pre-update* id, the later verdict's `locate` call no longer finds `X`
/// (the line now holds the new text, with a new id) and silently no-ops.
/// That is deterministic and consistent with "the file always wins", but is
/// easy to be surprised by when triaging a batch that looks like it should
/// have applied.
fn locate(lines: &[String], id: &str) -> Option<usize> {
    lines
        .iter()
        .position(|l| parse_entries(l).first().is_some_and(|e| e.id() == id))
}

/// One audit line staged during a scope's verdict loop, flushed only once
/// the scope's file write (if any) has actually succeeded. See
/// [`apply_verdicts`].
struct PendingLog {
    action: &'static str,
    id: String,
    text: String,
    reason: &'static str,
}

/// Mutable state threaded through one scope's verdict loop by
/// [`apply_one_verdict`]: the in-progress file lines, the sidecar, and the
/// staged (not-yet-flushed) audit entries and notes.
struct ScopeState<'a> {
    lines: &'a mut Vec<String>,
    meta: &'a mut MetaStore,
    changed: &'a mut bool,
    meta_dirty: &'a mut bool,
    audit: &'a mut Vec<PendingLog>,
    notes: &'a mut Vec<String>,
}

/// Applies one verdict against `scope`'s in-progress state, staging any
/// resulting file line change, sidecar mutation, audit entry, and note.
/// Nothing here touches disk; see [`apply_verdicts`] for why.
fn apply_one_verdict(state: &mut ScopeState<'_>, scope: Scope, v: &Verdict, date: &str) {
    match v {
        Verdict::Add {
            text,
            kind,
            scope: s,
        } if *s == scope => {
            let entry = Entry {
                date: date.to_string(),
                kind: *kind,
                text: text.clone(),
            };
            if locate(state.lines, &entry.id()).is_some() {
                return; // already present; re-adding is a no-op
            }
            state.lines.push(entry.render().trim_end().to_string());
            *state.changed = true;
            state.audit.push(PendingLog {
                action: "add",
                id: entry.id(),
                text: text.clone(),
                reason: "extracted",
            });
            state.notes.push(format!("added [{}] {text}", kind.tag()));
        }
        // An `Add` destined for the *other* scope. The loop visits both
        // files, so the matching iteration writes it.
        Verdict::Add { .. } => {}
        Verdict::Update { id, text } => {
            // See the comment on `locate`: a later verdict in this same
            // batch naming the pre-update id will not resolve.
            let Some(i) = locate(state.lines, id) else {
                return;
            };
            let Some(old) = parse_entries(&state.lines[i]).into_iter().next() else {
                return;
            };
            let new = Entry {
                date: old.date.clone(),
                kind: old.kind,
                text: text.clone(),
            };
            state.lines[i] = new.render().trim_end().to_string();
            state.meta.carry(id, &new.id());
            *state.meta_dirty = true;
            *state.changed = true;
            state.audit.push(PendingLog {
                action: "update",
                id: new.id(),
                text: text.clone(),
                reason: "reconciled",
            });
            state
                .notes
                .push(format!("updated [{}] {text}", new.kind.tag()));
        }
        Verdict::Delete { id } => {
            let Some(i) = locate(state.lines, id) else {
                return;
            };
            let Some(old) = parse_entries(&state.lines[i]).into_iter().next() else {
                return;
            };
            state.lines.remove(i);
            *state.changed = true;
            state.audit.push(PendingLog {
                action: "delete",
                id: id.clone(),
                text: old.text.clone(),
                reason: "reconciled",
            });
            state
                .notes
                .push(format!("removed [{}] {}", old.kind.tag(), old.text));
        }
        Verdict::Used { id } => {
            let Some(i) = locate(state.lines, id) else {
                return;
            };
            let text = parse_entries(&state.lines[i])
                .into_iter()
                .next()
                .map_or_else(String::new, |e| e.text);
            state.meta.bump(id, date);
            *state.meta_dirty = true;
            state.audit.push(PendingLog {
                action: "used",
                id: id.clone(),
                text,
                reason: "reused",
            });
        }
    }
}

/// Applies a batch of verdicts to both scopes' files and sidecars.
///
/// Every verdict naming an id with no live entry is discarded silently: the
/// user may have edited `MEMORY.md` by hand between the pass reading it and
/// this write, and the file always wins.
///
/// Nothing durable is recorded unless it actually reached disk: audit lines
/// and sidecar (`MetaStore`) changes are staged in memory while verdicts are
/// applied, then flushed together only after a needed file write succeeds
/// (or when there was nothing to write). If the write fails, the audit log
/// and sidecar for that scope are left exactly as they were, and the loop
/// moves on to the next scope.
///
/// Returns one human-readable note per applied change; each is also written
/// to the audit log.
#[must_use]
pub fn apply_verdicts(cwd: &Path, verdicts: &[Verdict], date: &str) -> Vec<String> {
    apply_verdicts_to(cwd, verdicts, date, None, None)
}

/// As [`apply_verdicts`], but `log_path` overrides where audit lines land:
/// `Some(path)` writes there instead of resolving `~/.plank` from `HOME`,
/// which is what lets a test redirect the audit log without ever setting
/// `HOME` itself. `user_root` is the analogous override for *where the user
/// scope's `MEMORY.md` itself lives*: `Some(root)` resolves it as
/// `root.join("MEMORY.md")` instead of following `HOME`, so a test can
/// exercise the user-scope branch of this loop — including its deletions and
/// rewrites — without ever touching or setting up the real `~/.plank`.
/// `None` for either keeps production behavior.
#[must_use]
pub(crate) fn apply_verdicts_to(
    cwd: &Path,
    verdicts: &[Verdict],
    date: &str,
    log_dest: Option<&Path>,
    user_root: Option<&Path>,
) -> Vec<String> {
    let mut notes = Vec::new();
    for scope in [Scope::User, Scope::Project] {
        let Some(path) = scoped_path_for(scope, cwd, user_root) else {
            continue;
        };
        let body = std::fs::read_to_string(&path).unwrap_or_else(|_| TEMPLATE.to_string());
        let mut lines: Vec<String> = body.lines().map(str::to_string).collect();
        let mut meta = MetaStore::load(&meta_path_for(&path));
        let mut changed = false;
        let mut meta_dirty = false;
        let mut audit: Vec<PendingLog> = Vec::new();
        let mut scope_notes: Vec<String> = Vec::new();

        let mut state = ScopeState {
            lines: &mut lines,
            meta: &mut meta,
            changed: &mut changed,
            meta_dirty: &mut meta_dirty,
            audit: &mut audit,
            notes: &mut scope_notes,
        };
        for v in verdicts {
            apply_one_verdict(&mut state, scope, v, date);
        }

        let write_ok = if changed {
            let mut out = lines.join("\n");
            out.push('\n');
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(&path, out).is_ok()
        } else {
            true
        };

        if !write_ok {
            // The file write failed: nothing durable happened for this
            // scope, so neither the audit log nor the sidecar may record
            // anything either. Leave both untouched and move on.
            continue;
        }

        for entry in &audit {
            match log_dest {
                Some(p) => {
                    append_log_line(p, entry.action, scope, &entry.id, &entry.text, entry.reason);
                }
                None => log_change(entry.action, scope, &entry.id, &entry.text, entry.reason),
            }
        }
        notes.extend(scope_notes);

        if changed || meta_dirty {
            let live: Vec<String> = parse_entries(&lines.join("\n"))
                .iter()
                .map(Entry::id)
                .collect();
            meta.gc(&live);
            let _ = meta.save(&meta_path_for(&path));
        }
    }
    notes
}

/// Entries in either scope whose text contains `pattern`, case-insensitively,
/// rendered as `[kind] text`, without modifying anything. Callers use this to
/// show the user exactly what a following [`forget_matching`] call would
/// remove, before asking them to confirm it.
#[must_use]
pub fn forget_preview(cwd: &Path, pattern: &str) -> Vec<String> {
    forget_preview_to(cwd, pattern, None)
}

/// As [`forget_preview`], but with [`forget_matching_to`]'s `user_root`
/// override for where the user scope lives.
///
/// The preview takes the same override as the deletion for one reason: the
/// two must always describe the same set of entries. A preview that read the
/// real `~/.plank/MEMORY.md` while the deletion ran against a redirected root
/// would show the user one thing and remove another.
#[must_use]
pub fn forget_preview_to(cwd: &Path, pattern: &str, user_root: Option<&Path>) -> Vec<String> {
    let needle = pattern.trim().to_lowercase();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for scope in [Scope::User, Scope::Project] {
        let Some(path) = scoped_path_for(scope, cwd, user_root) else {
            continue;
        };
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        for e in parse_entries(&body) {
            if e.text.to_lowercase().contains(&needle) {
                hits.push(format!("[{}] {}", e.kind.tag(), e.text));
            }
        }
    }
    hits
}

/// Removes every entry whose text contains `pattern`, case-insensitively,
/// from both scopes. Returns the removed entries' rendered `[kind] text`
/// form.
///
/// Unlike a model `forget` ([`MetaStore::set_retracted`]), this deletes the
/// bytes outright: the user asked, so there is nothing to keep recoverable.
/// Callers are expected to confirm with the user before calling this.
///
/// # Errors
///
/// Returns a message when `pattern` is empty or a file write fails.
pub fn forget_matching(cwd: &Path, pattern: &str) -> Result<Vec<String>, String> {
    forget_matching_to(cwd, pattern, None, None)
}

/// As [`forget_matching`], but `log_dest` overrides where audit lines land,
/// same as [`apply_verdicts_to`]'s `log_dest` — it lets tests exercise this
/// without ever touching the real `~/.plank` audit log or setting `HOME`.
/// `user_root` is [`apply_verdicts_to`]'s same override for where the user
/// scope's `MEMORY.md` itself lives: without it, this function — which
/// *deletes* matching lines — would silently mutate the real
/// `~/.plank/MEMORY.md` in any test that redirects only `cwd`.
pub(crate) fn forget_matching_to(
    cwd: &Path,
    pattern: &str,
    log_dest: Option<&Path>,
    user_root: Option<&Path>,
) -> Result<Vec<String>, String> {
    let needle = pattern.trim().to_lowercase();
    if needle.is_empty() {
        return Err("give a pattern to forget".to_string());
    }
    let mut removed = Vec::new();
    for scope in [Scope::User, Scope::Project] {
        let Some(path) = scoped_path_for(scope, cwd, user_root) else {
            continue;
        };
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        let mut kept: Vec<&str> = Vec::new();
        let mut hits: Vec<Entry> = Vec::new();
        for line in body.lines() {
            match parse_entries(line).into_iter().next() {
                Some(e) if e.text.to_lowercase().contains(&needle) => hits.push(e),
                _ => kept.push(line),
            }
        }
        if hits.is_empty() {
            continue;
        }
        let mut out = kept.join("\n");
        out.push('\n');
        std::fs::write(&path, out).map_err(|e| e.to_string())?;

        for e in &hits {
            let id = e.id();
            log_change_to(log_dest, "forget", scope, &id, &e.text, "user /forget");
            removed.push(format!("[{}] {}", e.kind.tag(), e.text));
        }

        let meta_path = meta_path_for(&path);
        let mut meta = MetaStore::load(&meta_path);
        let live: Vec<String> = parse_entries(&kept.join("\n"))
            .iter()
            .map(Entry::id)
            .collect();
        meta.gc(&live);
        let _ = meta.save(&meta_path);
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(text: &str, kind: Kind) -> Entry {
        Entry {
            date: "2026-09-15".into(),
            kind,
            text: text.into(),
        }
    }

    #[test]
    fn eviction_prefers_pinned_then_most_used_then_most_recent() {
        let entries = vec![
            entry("aaaa", Kind::Project),
            entry("bbbb", Kind::Project),
            entry("cccc", Kind::Project),
        ];
        let mut meta = MetaStore::default();
        meta.set_pinned(&entries[0].id(), true); // pinned, never used
        meta.bump(&entries[1].id(), "2026-09-15"); // used once
        // entries[2] unused and unpinned — the first to go.

        let budgets = Budgets {
            project: 2 * entries[0].render().len(),
            ..Budgets::default()
        };
        let (kept, dropped) = select_for_render(&entries, &meta, &budgets);
        let kept: Vec<_> = kept.iter().map(|e| e.text.clone()).collect();
        assert_eq!(kept, vec!["aaaa".to_string(), "bbbb".to_string()]);
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].text, "cccc");
    }

    #[test]
    fn a_retracted_entry_is_not_rendered_but_is_not_dropped_data() {
        let entries = vec![entry("gone", Kind::User), entry("stays", Kind::User)];
        let mut meta = MetaStore::default();
        meta.set_retracted(&entries[0].id(), true);
        let (kept, dropped) = select_for_render(&entries, &meta, &Budgets::default());
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].text, "stays");
        assert!(dropped.is_empty(), "retraction is not a budget eviction");
    }

    #[test]
    fn a_block_within_budget_keeps_every_entry_in_file_order() {
        let entries = vec![entry("one", Kind::Feedback), entry("two", Kind::Feedback)];
        let (kept, dropped) =
            select_for_render(&entries, &MetaStore::default(), &Budgets::default());
        assert_eq!(kept, entries);
        assert!(dropped.is_empty());
    }

    fn two_sources(dir: &Path) -> Vec<Source> {
        vec![
            Source {
                scope: Scope::User,
                path: dir.join("user/MEMORY.md"),
            },
            Source {
                scope: Scope::Project,
                path: dir.join("proj/.plank/MEMORY.md"),
            },
        ]
    }

    #[test]
    fn combine_marks_each_source_and_round_trips_unchanged() {
        let dir = scratch("combine");
        let srcs = two_sources(&dir);
        std::fs::create_dir_all(dir.join("user")).unwrap();
        std::fs::write(&srcs[0].path, "# Memory\n- (2026-09-01) likes tabs\n").unwrap();
        let text = combine(&srcs);
        assert!(text.contains("<!-- plank-memory: begin user "));
        assert!(text.contains("<!-- plank-memory: begin project "));
        assert!(text.contains("- (2026-09-01) likes tabs\n<!-- plank-memory: end user -->"));
        let report = apply(&srcs, &text).unwrap();
        assert!(report[0].ends_with("unchanged"), "{report:?}");
        assert!(
            !srcs[1].path.exists(),
            "an empty untouched section creates no file"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn edits_are_routed_to_the_right_file_and_strays_join_the_section_above() {
        let dir = scratch("split");
        let srcs = two_sources(&dir);
        std::fs::create_dir_all(dir.join("user")).unwrap();
        std::fs::write(&srcs[0].path, "u1\n").unwrap();
        let mut text = combine(&srcs);
        text = text.replace("u1\n", "u1\nu2\n");
        text = text.replace(
            "<!-- plank-memory: end project -->\n",
            "<!-- plank-memory: end project -->\n\n- typed below the last block\n",
        );
        let report = apply(&srcs, &text).unwrap();
        assert_eq!(std::fs::read_to_string(&srcs[0].path).unwrap(), "u1\nu2\n");
        assert_eq!(
            std::fs::read_to_string(&srcs[1].path).unwrap(),
            "- typed below the last block\n",
            "stray text after the project block lands in the project file"
        );
        assert!(report.iter().all(|l| l.contains("wrote")), "{report:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn removed_markers_leave_that_file_alone_and_broken_markup_is_refused() {
        let dir = scratch("markers");
        let srcs = two_sources(&dir);
        std::fs::create_dir_all(dir.join("user")).unwrap();
        std::fs::write(&srcs[0].path, "keep me\n").unwrap();
        let text = combine(&srcs);
        let start = text.find("<!-- plank-memory: begin user").unwrap();
        let end = text.find("<!-- plank-memory: end user -->\n").unwrap()
            + "<!-- plank-memory: end user -->\n".len();
        let without_user = format!("{}{}", &text[..start], &text[end..]);
        let report = apply(&srcs, &without_user).unwrap();
        assert!(report[0].contains("markers removed"), "{report:?}");
        assert_eq!(std::fs::read_to_string(&srcs[0].path).unwrap(), "keep me\n");
        let unterminated = text.replace("<!-- plank-memory: end project -->\n", "");
        assert!(split(&unterminated).unwrap_err().contains("never closed"));
        let twice =
            format!("{text}<!-- plank-memory: begin user x -->\n<!-- plank-memory: end user -->\n");
        assert!(split(&twice).unwrap_err().contains("twice"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_hand_edited_marker_path_cannot_redirect_the_write() {
        let text = "<!-- plank-memory: begin project /etc/passwd -->\nx\n<!-- plank-memory: end project -->\n";
        assert_eq!(
            split(text).unwrap(),
            vec![(Scope::Project, "x\n".to_owned())]
        );
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-memory-{}-{name}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn remember_creates_template_then_appends() {
        let cwd = scratch("append");
        let path = remember(Scope::Project, &cwd, "prefers tabs", "2026-07-19").unwrap();
        remember(Scope::Project, &cwd, "ships on Fridays", "2026-07-20").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# Memory\n"));
        assert!(text.contains("- (2026-07-19) prefers tabs\n"));
        assert!(text.ends_with("- (2026-07-20) ships on Fridays\n"));
        assert!(remember(Scope::Project, &cwd, "  ", "2026-07-20").is_err());
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn load_default_renders_project_section() {
        let cwd = scratch("load");
        assert!(load_default(&cwd).is_none() || std::env::var_os("HOME").is_some());
        remember(Scope::Project, &cwd, "target is macOS only", "2026-07-19").unwrap();
        let out = load_default(&cwd).unwrap();
        assert!(out.starts_with("Persistent memory"));
        assert!(out.contains("## Project memory (.plank/MEMORY.md)"));
        assert!(out.contains("target is macOS only"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// `load_scope` now evicts per type under `Budgets::default()` instead of
    /// tail-truncating the raw file; a runaway `[project]` block reports the
    /// omission rather than silently swallowing the whole file.
    #[test]
    fn oversized_scope_is_evicted_by_budget_not_tail_truncated() {
        let cwd = scratch("trunc");
        let path = path_for(Scope::Project, &cwd).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut big = String::from("# Memory\n");
        for i in 0..2000 {
            let _ = writeln!(
                big,
                "- (2026-07-19) [project] entry number {i} with padding"
            );
        }
        std::fs::write(&path, &big).unwrap();
        let out = load_scope(Scope::Project, &cwd).unwrap();
        assert!(out.contains("### project"));
        assert!(out.contains("older entries omitted under the type budgets"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn tagged_and_untagged_entries_both_parse() {
        let body = "# Memory\n\n\
                    - (2026-09-15) [feedback] Don't force-add generated docs.\n\
                    - (2026-09-14) plain untagged entry\n\
                    not a bullet at all\n";
        let entries = parse_entries(body);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind, Kind::Feedback);
        assert_eq!(entries[0].date, "2026-09-15");
        assert_eq!(entries[0].text, "Don't force-add generated docs.");
        assert_eq!(
            entries[1].kind,
            Kind::Project,
            "untagged falls back to project"
        );
        assert_eq!(entries[1].text, "plain untagged entry");
    }

    #[test]
    fn entry_id_is_stable_over_text_and_ignores_date() {
        let a = Entry {
            date: "2026-09-15".into(),
            kind: Kind::User,
            text: "prefers tabs".into(),
        };
        let b = Entry {
            date: "2026-01-01".into(),
            kind: Kind::Project,
            text: "prefers tabs".into(),
        };
        let c = Entry {
            date: "2026-09-15".into(),
            kind: Kind::User,
            text: "prefers spaces".into(),
        };
        assert_eq!(a.id(), b.id(), "id is a hash of text only");
        assert_ne!(a.id(), c.id());
        assert_eq!(a.id().len(), 12);
    }

    #[test]
    fn render_round_trips_through_parse() {
        let e = Entry {
            date: "2026-09-15".into(),
            kind: Kind::Reference,
            text: "dashboard at example".into(),
        };
        let parsed = parse_entries(&e.render());
        assert_eq!(parsed, vec![e]);
    }

    /// The off-by-default equivalence property: turn everything off (no
    /// tags, defaults everywhere) and rendering is exactly what it was
    /// before this feature existed — every entry, nothing evicted, nothing
    /// annotated. This is what makes the feature shippable on by default.
    #[test]
    fn an_untagged_legacy_file_renders_every_entry_when_nothing_is_configured() {
        let dir = std::env::temp_dir().join(format!("plank-legacy-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        std::fs::write(
            dir.join(".plank").join("MEMORY.md"),
            "# Memory\n\n- (2026-01-01) an old untagged fact\n- (2026-01-02) another one\n",
        )
        .unwrap();

        let rendered = load_default(&dir).unwrap();
        assert!(rendered.contains("an old untagged fact"));
        assert!(rendered.contains("another one"));
        assert!(
            !rendered.contains("omitted"),
            "nothing is evicted at default budgets"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unrecognised_bracket_stays_in_the_entry_text() {
        // A bracket that is not one of the four kinds is the user's own
        // prose, not a failed tag: stripping `[WIP]` would silently delete
        // something they typed. So it is kept verbatim, and the entry reads
        // as untagged. Deliberate, and pinned here because the asymmetry
        // with a genuinely untagged line looks like an oversight otherwise.
        let entries = parse_entries("- (2026-01-01) [WIP] ship it\n");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, Kind::Project);
        assert_eq!(entries[0].text, "[WIP] ship it");
    }

    #[test]
    fn meta_store_round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("plank-meta-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("MEMORY.md");
        let mut store = MetaStore::default();
        store.bump("abc123", "2026-09-15");
        store.bump("abc123", "2026-09-16");
        store.set_retracted("dead99", true);
        store.save(&meta_path_for(&path)).unwrap();

        let reloaded = MetaStore::load(&meta_path_for(&path));
        assert_eq!(reloaded.get("abc123").uses, 2);
        assert_eq!(reloaded.get("abc123").last_used, "2026-09-16");
        assert!(reloaded.get("dead99").retracted);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_or_corrupt_sidecar_degrades_to_zeroed_counters() {
        let missing = MetaStore::load(std::path::Path::new("/nonexistent/MEMORY.md.meta.json"));
        assert_eq!(missing.get("anything").uses, 0);
        assert!(!missing.get("anything").pinned);

        let dir = std::env::temp_dir().join(format!("plank-meta-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("x.meta.json");
        std::fs::write(&bad, "{ this is not json").unwrap();
        assert_eq!(MetaStore::load(&bad).get("anything").uses, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn carry_moves_a_row_to_a_new_id_and_gc_drops_orphans() {
        let mut store = MetaStore::default();
        store.bump("old", "2026-09-15");
        store.bump("old", "2026-09-15");
        store.carry("old", "new");
        assert_eq!(store.get("new").uses, 2, "usage survives a rephrasing");
        assert_eq!(store.get("old").uses, 0);

        store.bump("orphan", "2026-09-15");
        store.gc(&["new".to_string()]);
        assert_eq!(
            store.get("orphan").uses,
            0,
            "rows with no live entry are dropped"
        );
        assert_eq!(store.get("new").uses, 2);
    }

    #[test]
    fn audit_lines_are_json_and_read_back_newest_last() {
        let dir = std::env::temp_dir().join(format!("plank-auditlog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory-log.jsonl");
        append_log_line(
            &path,
            "delete",
            Scope::Project,
            "abc123",
            "stale fact",
            "over budget",
        );
        append_log_line(
            &path,
            "add",
            Scope::User,
            "def456",
            "prefers tabs",
            "extracted",
        );

        let lines = read_log_from(&path, 10);
        assert_eq!(lines.len(), 2);
        assert!(
            lines[1].contains("\"action\": \"add\""),
            "newest last: {}",
            lines[1]
        );
        assert!(lines[0].contains("over budget"));
        assert!(
            crate::tools::mcp::json_parse(&lines[0]).is_some(),
            "each line parses as JSON"
        );

        let capped = read_log_from(&path, 1);
        assert_eq!(capped.len(), 1);
        assert!(
            capped[0].contains("\"action\": \"add\""),
            "the cap keeps the newest"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_logged_entry_survives_quotes_backslashes_and_newlines() {
        // The JSONL invariant is one object per line, so an embedded newline
        // in the entry text must be escaped rather than ending the line. And
        // json_escape emits its own surrounding quotes, so wrapping its
        // output by hand would double-quote and produce malformed JSON --
        // a bug this plan already hit once elsewhere.
        let dir = std::env::temp_dir().join(format!("plank-auditlog-esc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory-log.jsonl");
        let nasty = "he said \"hi\"\nthen C:\\path";
        append_log_line(&path, "delete", Scope::User, "abc123", nasty, "reconciled");

        let lines = read_log_from(&path, 10);
        assert_eq!(
            lines.len(),
            1,
            "an embedded newline must not split the record"
        );
        let parsed = crate::tools::mcp::json_parse(&lines[0]).expect("line parses as JSON");
        assert_eq!(
            parsed.str_or("text", ""),
            nasty,
            "the text round-trips verbatim"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verdicts_parse_from_the_pass_json() {
        let json = r#"[
            {"verdict": "ADD", "text": "prefers tabs", "type": "user", "scope": "user"},
            {"verdict": "UPDATE", "id": "abc123", "text": "prefers tabs, width 4"},
            {"verdict": "DELETE", "id": "def456"},
            {"verdict": "USED", "id": "aaa111"},
            {"verdict": "NOOP"}
        ]"#;
        let v = parse_verdicts(json).unwrap();
        assert_eq!(v.len(), 4, "NOOP is dropped, not an error");
        assert!(
            matches!(&v[0], Verdict::Add { kind: Kind::User, scope: Scope::User, text } if text == "prefers tabs")
        );
        assert!(matches!(&v[1], Verdict::Update { id, .. } if id == "abc123"));
        assert!(matches!(&v[2], Verdict::Delete { id } if id == "def456"));
        assert!(matches!(&v[3], Verdict::Used { id } if id == "aaa111"));
    }

    #[test]
    fn malformed_verdict_json_is_an_error_not_a_partial_write() {
        assert!(parse_verdicts("not json at all").is_err());
        assert!(
            parse_verdicts(r#"{"verdict": "ADD"}"#).is_err(),
            "must be an array"
        );
    }

    #[test]
    fn update_rewrites_the_line_in_place_and_carries_usage() {
        let dir = std::env::temp_dir().join(format!("plank-verdict-update-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        let path = dir.join(".plank").join("MEMORY.md");
        let user = dir.join("userhome");
        std::fs::write(&path, "# Memory\n\n- (2026-09-01) [project] old wording\n").unwrap();

        let old = Entry {
            date: "2026-09-01".into(),
            kind: Kind::Project,
            text: "old wording".into(),
        };
        let mut meta = MetaStore::default();
        meta.bump(&old.id(), "2026-09-10");
        meta.bump(&old.id(), "2026-09-11");
        meta.save(&meta_path_for(&path)).unwrap();

        let log = dir.join("audit.jsonl");
        let _ = apply_verdicts_to(
            &dir,
            &[Verdict::Update {
                id: old.id(),
                text: "new wording".into(),
            }],
            "2026-09-15",
            Some(&log),
            Some(&user),
        );

        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("new wording"));
        assert!(!body.contains("old wording"));
        assert!(
            body.contains("(2026-09-01)"),
            "the original date is preserved"
        );

        let new = Entry {
            date: "2026-09-01".into(),
            kind: Kind::Project,
            text: "new wording".into(),
        };
        let reloaded = MetaStore::load(&meta_path_for(&path));
        assert_eq!(
            reloaded.get(&new.id()).uses,
            2,
            "usage survived the rewrite"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_verdict_naming_an_unknown_id_is_discarded_silently() {
        let dir = std::env::temp_dir().join(format!("plank-verdict-ghost-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        let path = dir.join(".plank").join("MEMORY.md");
        let user = dir.join("userhome");
        std::fs::write(&path, "# Memory\n\n- (2026-09-01) [project] kept\n").unwrap();

        let log = dir.join("audit.jsonl");
        let notes = apply_verdicts_to(
            &dir,
            &[Verdict::Delete {
                id: "0000deadbeef".into(),
            }],
            "2026-09-15",
            Some(&log),
            Some(&user),
        );

        assert!(std::fs::read_to_string(&path).unwrap().contains("kept"));
        assert!(
            notes.iter().all(|n| !n.contains("0000deadbeef")),
            "no write, no note"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn update_preserves_usage_across_repeated_rewrites() {
        // Beyond the brief: rephrase the same entry twice in a row and check
        // usage keeps accumulating rather than resetting on the second carry.
        let dir =
            std::env::temp_dir().join(format!("plank-verdict-update-chain-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        let path = dir.join(".plank").join("MEMORY.md");
        let user = dir.join("userhome");
        std::fs::write(&path, "# Memory\n\n- (2026-09-01) [project] v1\n").unwrap();

        let v1 = Entry {
            date: "2026-09-01".into(),
            kind: Kind::Project,
            text: "v1".into(),
        };
        let mut meta = MetaStore::default();
        meta.bump(&v1.id(), "2026-09-05");
        meta.save(&meta_path_for(&path)).unwrap();

        let log = dir.join("audit.jsonl");
        let _ = apply_verdicts_to(
            &dir,
            &[Verdict::Update {
                id: v1.id(),
                text: "v2".into(),
            }],
            "2026-09-10",
            Some(&log),
            Some(&user),
        );
        let v2 = Entry {
            date: "2026-09-01".into(),
            kind: Kind::Project,
            text: "v2".into(),
        };
        let _ = apply_verdicts_to(
            &dir,
            &[
                Verdict::Used { id: v2.id() },
                Verdict::Update {
                    id: v2.id(),
                    text: "v3".into(),
                },
            ],
            "2026-09-12",
            Some(&log),
            Some(&user),
        );
        let v3 = Entry {
            date: "2026-09-01".into(),
            kind: Kind::Project,
            text: "v3".into(),
        };
        let reloaded = MetaStore::load(&meta_path_for(&path));
        assert_eq!(
            reloaded.get(&v3.id()).uses,
            2,
            "usage accumulated across two rewrites"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_file_write_leaves_the_sidecar_and_its_counters_untouched() {
        // Finding 1: if the memory file write fails, neither the sidecar nor
        // the audit log may record the change. We force the write to fail
        // portably by making the memory file's own path a directory, which
        // `fs::write` always refuses.
        let dir = std::env::temp_dir().join(format!(
            "plank-verdict-write-fails-{}-{}",
            std::process::id(),
            "a"
        ));
        let plank_dir = dir.join(".plank");
        let path = plank_dir.join("MEMORY.md");
        let user = dir.join("userhome");
        std::fs::create_dir_all(&path).unwrap(); // path is a directory, not a file

        // Seed the sidecar with a counter that must survive untouched.
        let existing_id = "0123456789ab";
        let mut meta = MetaStore::default();
        meta.bump(existing_id, "2026-09-01");
        meta.bump(existing_id, "2026-09-02");
        meta.save(&meta_path_for(&path)).unwrap();

        let log = dir.join("audit.jsonl");
        let notes = apply_verdicts_to(
            &dir,
            &[Verdict::Add {
                text: "should never land".into(),
                kind: Kind::Project,
                scope: Scope::Project,
            }],
            "2026-09-15",
            Some(&log),
            Some(&user),
        );

        assert!(
            notes.is_empty(),
            "no note should be produced when the write fails"
        );
        assert!(
            std::fs::read_to_string(&path).is_err(),
            "the path is still a directory: no file was ever written"
        );
        let reloaded = MetaStore::load(&meta_path_for(&path));
        assert_eq!(
            reloaded.get(existing_id).uses,
            2,
            "sidecar counters must survive a failed write untouched"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_used_verdict_produces_an_audit_log_entry() {
        // Finding 2: USED must be logged like every other verdict. The audit
        // log is redirected to a private temp file via `apply_verdicts_to`,
        // so this never touches `~/.plank`.
        let dir =
            std::env::temp_dir().join(format!("plank-verdict-used-logs-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        let path = dir.join(".plank").join("MEMORY.md");
        let user = dir.join("userhome");
        let text = "kept";
        std::fs::write(
            &path,
            format!("# Memory\n\n- (2026-09-01) [project] {text}\n"),
        )
        .unwrap();
        let id = Entry {
            date: "2026-09-01".into(),
            kind: Kind::Project,
            text: text.into(),
        }
        .id();

        let log = dir.join("audit.jsonl");
        let _ = apply_verdicts_to(
            &dir,
            &[Verdict::Used { id: id.clone() }],
            "2026-09-15",
            Some(&log),
            Some(&user),
        );

        let log_text = std::fs::read_to_string(&log).unwrap_or_default();
        let expected = format!(
            "{{\"action\": \"used\", \"scope\": \"project\", \"id\": \"{id}\", \"text\": \"{text}\", \"reason\": \"reused\"}}\n"
        );
        assert_eq!(
            log_text, expected,
            "expected exactly one used-action audit line for id {id}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_verdicts_never_touches_the_real_audit_log() {
        // The hermetic property itself: routing the audit log to an explicit
        // temp path writes there, and leaves whatever `~/.plank` holds
        // (present, absent, any size) completely unchanged. This must hold
        // without ever setting `HOME` in a test.
        let dir =
            std::env::temp_dir().join(format!("plank-verdict-hermetic-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        let path = dir.join(".plank").join("MEMORY.md");
        let user = dir.join("userhome");
        std::fs::write(
            &path,
            "# Memory\n\n- (2026-09-01) [project] hermetic fact\n",
        )
        .unwrap();
        let id = Entry {
            date: "2026-09-01".into(),
            kind: Kind::Project,
            text: "hermetic fact".into(),
        }
        .id();

        let real_before = log_path().and_then(|p| std::fs::read_to_string(&p).ok());

        let log = dir.join("audit.jsonl");
        let _ = apply_verdicts_to(
            &dir,
            &[Verdict::Used { id: id.clone() }],
            "2026-09-15",
            Some(&log),
            Some(&user),
        );

        assert!(
            std::fs::read_to_string(&log)
                .unwrap_or_default()
                .contains(&id),
            "the redirected log must have received the entry"
        );

        let real_after = log_path().and_then(|p| std::fs::read_to_string(&p).ok());
        assert_eq!(
            real_before, real_after,
            "the real ~/.plank audit log must be untouched by a redirected call"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn forget_matching_removes_case_insensitive_hits_and_leaves_the_rest() {
        let dir = std::env::temp_dir().join(format!("plank-forgetcmd-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        let path = dir.join(".plank").join("MEMORY.md");
        let user = dir.join("userhome");
        std::fs::write(
            &path,
            "# Memory\n\n\
             - (2026-09-01) [project] Ship the BETA on Friday\n\
             - (2026-09-02) [user] prefers tabs\n",
        )
        .unwrap();

        let log = dir.join("audit.jsonl");
        let removed = forget_matching_to(&dir, "beta", Some(&log), Some(&user)).unwrap();
        assert_eq!(removed.len(), 1);
        assert!(removed[0].contains("Ship the BETA"));

        let body = std::fs::read_to_string(&path).unwrap();
        assert!(!body.contains("BETA"));
        assert!(body.contains("prefers tabs"));

        let log_text = std::fs::read_to_string(&log).unwrap();
        assert!(log_text.contains("\"action\": \"forget\""));
        assert!(log_text.contains("Ship the BETA"));

        assert!(
            forget_matching_to(&dir, "nothing here", Some(&log), Some(&user))
                .unwrap()
                .is_empty()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_used_only_batch_still_persists_the_bumped_counter() {
        // Finding 3 subtlety: USED bumps the sidecar without touching
        // `lines`, so `changed` stays false for a USED-only batch — but the
        // sidecar must still be saved, or the bump is silently lost.
        let dir = std::env::temp_dir().join(format!(
            "plank-verdict-used-only-persists-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        let path = dir.join(".plank").join("MEMORY.md");
        let user = dir.join("userhome");
        std::fs::write(&path, "# Memory\n\n- (2026-09-01) [project] kept as-is\n").unwrap();
        let id = Entry {
            date: "2026-09-01".into(),
            kind: Kind::Project,
            text: "kept as-is".into(),
        }
        .id();

        let log = dir.join("audit.jsonl");
        let _ = apply_verdicts_to(
            &dir,
            &[Verdict::Used { id: id.clone() }],
            "2026-09-15",
            Some(&log),
            Some(&user),
        );

        let body_after = std::fs::read_to_string(&path).unwrap();
        assert!(
            body_after.contains("kept as-is"),
            "USED must never rewrite the line"
        );
        let reloaded = MetaStore::load(&meta_path_for(&path));
        assert_eq!(
            reloaded.get(&id).uses,
            1,
            "the bump must be persisted even though no line changed"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn forget_matching_with_an_explicit_user_root_never_touches_the_default_user_scope_location() {
        // The hermetic property for the *user-scope memory file itself*
        // (companion to `apply_verdicts_never_touches_the_real_audit_log`,
        // which only covers the audit log). `forget_matching_to` deletes
        // matching lines, so if it ever fell back to resolving `HOME` for
        // the user scope despite an explicit `user_root`, this would catch
        // it: a "beta" line planted under a decoy default-location stand-in
        // must survive completely untouched, while the same line planted
        // under the explicit `user_root` is deleted.
        let dir = std::env::temp_dir().join(format!(
            "plank-forget-user-root-hermetic-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(dir.join(".plank")).unwrap();
        let project_path = dir.join(".plank").join("MEMORY.md");
        std::fs::write(
            &project_path,
            "# Memory\n\n- (2026-09-01) [project] project scope entry\n",
        )
        .unwrap();

        let user_root = dir.join("userhome");
        std::fs::create_dir_all(&user_root).unwrap();
        let user_path = user_root.join("MEMORY.md");
        std::fs::write(
            &user_path,
            "# Memory\n\n- (2026-09-01) [user] contains beta keyword\n",
        )
        .unwrap();

        // A decoy standing in for "the default user-scope location" — never
        // passed as `user_root`, so it must be left byte-for-byte alone.
        let decoy_default = dir.join("decoy-default-userhome");
        std::fs::create_dir_all(&decoy_default).unwrap();
        let decoy_path = decoy_default.join("MEMORY.md");
        let decoy_before =
            "# Memory\n\n- (2026-09-01) [user] also contains beta keyword\n".to_string();
        std::fs::write(&decoy_path, &decoy_before).unwrap();

        // The audit log is redirected too: a test proving hermeticity must not
        // itself append to the real ~/.plank/memory-log.jsonl.
        let log = dir.join("audit.jsonl");
        let removed = forget_matching_to(&dir, "beta", Some(&log), Some(&user_root)).unwrap();

        assert_eq!(
            removed.len(),
            1,
            "only the entry under the explicit user_root is matched and removed"
        );
        assert!(removed[0].contains("contains beta keyword"));

        let user_body = std::fs::read_to_string(&user_path).unwrap();
        assert!(
            !user_body.contains("beta"),
            "the explicit user_root's file must have had the match deleted"
        );

        let decoy_after = std::fs::read_to_string(&decoy_path).unwrap();
        assert_eq!(
            decoy_after, decoy_before,
            "a location that was never passed as user_root must be left byte-for-byte untouched"
        );

        let project_body = std::fs::read_to_string(&project_path).unwrap();
        assert!(
            project_body.contains("project scope entry"),
            "the project scope is unaffected by the user-scope redirection"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
