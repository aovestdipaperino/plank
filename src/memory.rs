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
            out.push_str(&e.render());
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
}
