// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `/kvcache`: the state behind the KV lineage view.
//!
//! Mirrors the `/config` modal's shape ([`crate::configform`]): this module
//! owns rows, selection and key handling; `tui::draw_kvcache` only
//! paints what [`KvPane::rows`] returns, and the plain-stdout REPL prints
//! [`render_text`] of the same rows. Neither front end reimplements the
//! layout, so the two cannot drift.

use std::collections::HashSet;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::kvgc::{SweepPolicy, plan_sweep};
use crate::kvmeta::{KvLabel, KvMeta};
use crate::kvtree::{KvForest, KvNode};

/// Collapse key for the synthetic `(orphaned)` group header. Not a
/// fingerprint, and deliberately unrepresentable as one.
const ORPHAN_KEY: &str = "\u{0}orphans";

/// One render-ready line of the tree, plus the detail line beneath it.
#[derive(Debug, Clone)]
pub struct Row {
    /// Indent level: `0` for a root or the orphan header.
    pub depth: usize,
    /// Left label, `"<role>  <fp-or-name>"` (or `"(orphaned)"`).
    pub label: String,
    /// Second line shown under the row; empty when there is nothing to say.
    pub detail: String,
    /// Right-hand column: size, hits, age, and any pin/expiry marker.
    pub right: String,
    /// The cursor is on this row.
    pub selected: bool,
    /// This row's children are shown (always true when it has none).
    pub expanded: bool,
    /// This row has children, so collapsing it is meaningful.
    pub has_children: bool,
    /// Fingerprint this row acts on; empty for the orphan header.
    pub fingerprint: String,
}

/// What a key press asks the caller to do with the pane.
#[derive(Debug)]
pub enum Outcome {
    /// Stay open; the pane absorbed the key.
    Stay,
    /// Close the pane.
    Close,
    /// Pin the named blob.
    Pin(String),
    /// Unpin the named blob.
    Unpin(String),
    /// Delete the named blob (already confirmed).
    Delete(String),
    /// Run a sweep now.
    Sweep,
}

/// One node of the flattened tree, in the order [`KvPane::rows`] walks it.
#[derive(Debug, Clone)]
struct Flat {
    depth: usize,
    /// Collapse keys of every ancestor, outermost first.
    ancestors: Vec<String>,
    /// This row's own collapse key: the blob's fingerprint (or [`ORPHAN_KEY`]
    /// for the synthetic header). Two bodies can legitimately share a
    /// fingerprint, so they fold and pin as one row-key. That is inherent to the
    /// fingerprint-keyed [`Row`]/[`Outcome`] API rather than a bug here, and it
    /// is fail-safe: an [`Outcome::Delete`] naming such a fingerprint is refused
    /// as ambiguous rather than guessing which file to unlink.
    key: String,
    has_children: bool,
    /// Index into the flattened metadata list; `None` for the orphan header.
    meta: Option<usize>,
}

/// Interactive state over a KV cache forest.
#[derive(Debug)]
pub struct KvPane {
    /// Every node in walk order, tree shape flattened once.
    flat: Vec<Flat>,
    /// Metadata parallel to `Flat::meta`, in the same order.
    metas: Vec<KvMeta>,
    /// Metadata indices a sweep would condemn given the *on-disk* pin flags.
    /// Only a starting point: [`KvPane::sweep_now`] re-derives the verdicts
    /// whenever the pane holds unsaved pin flips.
    condemned: HashSet<usize>,
    /// The policy the verdicts are derived under, kept so a pin flip can be
    /// re-costed without rebuilding the pane and losing the fold state.
    policy: SweepPolicy,
    /// Collapse keys whose subtrees are hidden.
    collapsed: HashSet<String>,
    /// Fingerprints whose pin state the pane has flipped locally, so a second
    /// `p` press reads as the inverse of the first.
    pin_flips: HashSet<String>,
    /// Cursor over the *visible* rows.
    cursor: usize,
    /// A `d` press is awaiting confirmation.
    pending_delete: bool,
    /// Total bytes across the whole forest.
    total: u64,
    /// Bytes a sweep would reclaim.
    reclaimable: u64,
    /// The `kvcache.maxBytes` ceiling; `0` means unbounded.
    max_bytes: u64,
    /// Wall clock the ages and TTLs are measured against.
    now: u64,
}

impl KvPane {
    /// Builds the pane from a forest, flattening the tree and computing the
    /// sweep verdicts once.
    ///
    /// `policy` and `now` are only used to mark expired rows and total what a
    /// sweep would free; nothing here touches the filesystem.
    #[must_use]
    pub fn new(forest: KvForest, policy: SweepPolicy, max_bytes: u64, now: u64) -> Self {
        let mut flat: Vec<Flat> = Vec::new();
        let mut metas: Vec<KvMeta> = Vec::new();
        let total = forest.total_bytes();
        for root in forest.roots {
            push_node(&root, 0, &mut Vec::new(), &mut flat, &mut metas);
        }
        if !forest.orphans.is_empty() {
            flat.push(Flat {
                depth: 0,
                ancestors: Vec::new(),
                key: ORPHAN_KEY.to_owned(),
                has_children: true,
                meta: None,
            });
            for orphan in forest.orphans {
                push_node(
                    &orphan,
                    1,
                    &mut vec![ORPHAN_KEY.to_owned()],
                    &mut flat,
                    &mut metas,
                );
            }
        }
        // One `plan_sweep` for the whole pane: the plan names indices into
        // `metas`, which is exactly the order `flat` refers to.
        let plan = plan_sweep(&metas, &[], &policy, now);
        Self {
            flat,
            condemned: plan.doomed.iter().copied().collect(),
            policy,
            metas,
            collapsed: HashSet::new(),
            pin_flips: HashSet::new(),
            cursor: 0,
            pending_delete: false,
            total,
            reclaimable: plan.bytes,
            max_bytes,
            now,
        }
    }

    /// Whether a `d` press is waiting for a `y`.
    #[must_use]
    pub fn pending_delete(&self) -> bool {
        self.pending_delete
    }

    /// Indices into `self.flat` of the rows currently visible.
    fn visible(&self) -> Vec<usize> {
        self.flat
            .iter()
            .enumerate()
            .filter(|(_, f)| !f.ancestors.iter().any(|a| self.collapsed.contains(a)))
            .map(|(i, _)| i)
            .collect()
    }

    /// The effective pin state of a node, including the pane's local flip.
    fn pinned(&self, meta: &KvMeta) -> bool {
        meta.pinned ^ self.pin_flips.contains(&meta.fingerprint)
    }

    /// The sweep verdicts and reclaimable total under the pane's *effective*
    /// pin state, i.e. the on-disk flags with the local `p` flips applied.
    ///
    /// The construction-time snapshot cannot answer this: `plan_sweep` spares a
    /// pinned node, so pinning an expired row would leave its bytes counted as
    /// reclaimable, and unpinning a pinned-but-idle row would hide both its
    /// bytes and its `⏳` marker. Re-planning is pure and in-memory, so it is
    /// cheap enough to do per draw rather than rebuilding the pane — which would
    /// throw away the user's folds and cursor.
    /// Borrows the construction-time verdicts on the common path, so a draw
    /// that has flipped no pins allocates nothing.
    fn sweep_now(&self) -> (std::borrow::Cow<'_, HashSet<usize>>, u64) {
        use std::borrow::Cow;
        if self.pin_flips.is_empty() {
            return (Cow::Borrowed(&self.condemned), self.reclaimable);
        }
        let metas: Vec<KvMeta> = self
            .metas
            .iter()
            .map(|m| {
                let mut m = m.clone();
                m.pinned = self.pinned(&m);
                m
            })
            .collect();
        let plan = plan_sweep(&metas, &[], &self.policy, self.now);
        (Cow::Owned(plan.doomed.into_iter().collect()), plan.bytes)
    }

    /// The render rows, top to bottom, with collapsed subtrees omitted.
    #[must_use]
    pub fn rows(&self) -> Vec<Row> {
        let visible = self.visible();
        let cursor = self.cursor.min(visible.len().saturating_sub(1));
        // One re-plan for the whole draw, not one per row.
        let (condemned, _) = self.sweep_now();
        visible
            .iter()
            .enumerate()
            .map(|(pos, &i)| {
                let f = &self.flat[i];
                let selected = pos == cursor;
                let expanded = !self.collapsed.contains(&f.key);
                let Some(mi) = f.meta else {
                    return Row {
                        depth: f.depth,
                        label: "(orphaned)".to_owned(),
                        detail: String::new(),
                        right: String::new(),
                        selected,
                        expanded,
                        has_children: true,
                        fingerprint: String::new(),
                    };
                };
                let m = &self.metas[mi];
                Row {
                    depth: f.depth,
                    label: format!("{}  {}", m.role.as_str(), short_name(m)),
                    detail: detail_of(m),
                    right: self.right_of(mi, &condemned),
                    selected,
                    expanded,
                    has_children: f.has_children,
                    fingerprint: m.fingerprint.clone(),
                }
            })
            .collect()
    }

    /// The right-hand column for one metadata index, against the verdicts
    /// [`KvPane::sweep_now`] produced for this draw.
    fn right_of(&self, mi: usize, condemned: &HashSet<usize>) -> String {
        let m = &self.metas[mi];
        let mut s = format!(
            "{}  hits {}  {}",
            human_bytes(m.bytes),
            m.hits,
            relative_age(self.now.saturating_sub(m.last_used))
        );
        if self.pinned(m) {
            s.push_str(" 📌 pinned");
        } else if condemned.contains(&mi) {
            s.push_str(" ⏳ expired");
        }
        s
    }

    /// The one-line summary under the tree.
    #[must_use]
    pub fn footer(&self) -> String {
        use std::fmt::Write as _;
        let (_, reclaimable) = self.sweep_now();
        let mut s = format!(
            "total {} · {} reclaimable",
            human_bytes(self.total),
            human_bytes(reclaimable)
        );
        if self.max_bytes > 0 && self.total > self.max_bytes {
            let _ = write!(s, " · over the {} budget", human_bytes(self.max_bytes));
        }
        s
    }

    /// The fingerprint the cursor is on, if it names a real blob.
    fn selected_fp(&self) -> Option<String> {
        let visible = self.visible();
        let &i = visible.get(self.cursor.min(visible.len().saturating_sub(1)))?;
        let mi = self.flat[i].meta?;
        Some(self.metas[mi].fingerprint.clone())
    }

    /// The collapse key of the row the cursor is on.
    fn selected_key(&self) -> Option<String> {
        let visible = self.visible();
        let &i = visible.get(self.cursor.min(visible.len().saturating_sub(1)))?;
        Some(self.flat[i].key.clone())
    }

    /// Drives the pane from one key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        // A pending delete owns the keyboard: `y` confirms, everything else
        // (Esc included) cancels without also closing the pane.
        if self.pending_delete {
            self.pending_delete = false;
            if matches!(key.code, KeyCode::Char('y' | 'Y'))
                && let Some(fp) = self.selected_fp()
            {
                return Outcome::Delete(fp);
            }
            return Outcome::Stay;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let len = self.visible().len();
        self.cursor = self.cursor.min(len.saturating_sub(1));
        match key.code {
            KeyCode::Char('c') if ctrl => return Outcome::Close,
            KeyCode::Esc | KeyCode::Char('q' | 'Q') => return Outcome::Close,
            KeyCode::Up => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Down => {
                if self.cursor + 1 < len {
                    self.cursor += 1;
                }
            }
            KeyCode::Left => self.collapse_or_ascend(),
            KeyCode::Right => {
                if let Some(k) = self.selected_key() {
                    self.collapsed.remove(&k);
                }
            }
            KeyCode::Char('p' | 'P') => {
                if let Some(fp) = self.selected_fp() {
                    if self.pin_flips.contains(&fp) {
                        self.pin_flips.remove(&fp);
                    } else {
                        self.pin_flips.insert(fp.clone());
                    }
                    let now_pinned = self
                        .metas
                        .iter()
                        .find(|m| m.fingerprint == fp)
                        .is_some_and(|m| self.pinned(m));
                    return if now_pinned {
                        Outcome::Pin(fp)
                    } else {
                        Outcome::Unpin(fp)
                    };
                }
            }
            KeyCode::Char('d' | 'D') => {
                if self.selected_fp().is_some() {
                    self.pending_delete = true;
                }
            }
            KeyCode::Char('g' | 'G') => return Outcome::Sweep,
            _ => {}
        }
        Outcome::Stay
    }

    /// `Left`: collapse the selected row, or step to its parent when it is
    /// already collapsed (or has nothing to collapse).
    fn collapse_or_ascend(&mut self) {
        let visible = self.visible();
        let Some(&i) = visible.get(self.cursor) else {
            return;
        };
        let f = &self.flat[i];
        if f.has_children && !self.collapsed.contains(&f.key) {
            self.collapsed.insert(f.key.clone());
            let len = self.visible().len();
            self.cursor = self.cursor.min(len.saturating_sub(1));
            return;
        }
        // Step to the nearest visible ancestor.
        if let Some(parent) = f.ancestors.last().cloned()
            && let Some(pos) = visible
                .iter()
                .position(|&j| self.flat[j].key == parent && self.flat[j].depth < f.depth)
        {
            self.cursor = pos;
        }
    }
}

/// Flattens one node and its children into `flat`/`metas` in walk order.
fn push_node(
    node: &KvNode,
    depth: usize,
    ancestors: &mut Vec<String>,
    flat: &mut Vec<Flat>,
    metas: &mut Vec<KvMeta>,
) {
    let key = node.meta.fingerprint.clone();
    metas.push(node.meta.clone());
    flat.push(Flat {
        depth,
        ancestors: ancestors.clone(),
        key: key.clone(),
        has_children: !node.children.is_empty(),
        meta: Some(metas.len() - 1),
    });
    ancestors.push(key);
    for child in &node.children {
        push_node(child, depth + 1, ancestors, flat, metas);
    }
    ancestors.pop();
}

/// The short name a row shows: the session's name when recorded, else the
/// leading 8 characters of the fingerprint.
fn short_name(m: &KvMeta) -> String {
    if let KvLabel::Session { name, .. } = &m.label
        && !name.is_empty()
    {
        return name.clone();
    }
    m.fingerprint.chars().take(8).collect()
}

/// The second line under a row: whatever the role-specific label can say.
fn detail_of(m: &KvMeta) -> String {
    match &m.label {
        KvLabel::System {
            think_mode,
            global_mcp,
            ..
        } => format!("{think_mode} · {} global MCP tools", global_mcp.len()),
        KvLabel::Project {
            project_path,
            agents_files,
            local_mcp,
        } => format!(
            "{project_path} · {} · {} local MCP",
            agents_files.join(", "),
            local_mcp.len()
        ),
        KvLabel::Session { title, .. } => title.clone(),
        KvLabel::Unknown => String::new(),
    }
}

/// Idle time as a short phrase, e.g. `3d ago`.
fn relative_age(secs: u64) -> String {
    const MINUTE: u64 = 60;
    const HOUR: u64 = 60 * MINUTE;
    const DAY: u64 = 24 * HOUR;
    match secs {
        0..MINUTE => "just now".to_owned(),
        MINUTE..HOUR => format!("{}m ago", secs / MINUTE),
        HOUR..DAY => format!("{}h ago", secs / HOUR),
        _ => format!("{}d ago", secs / DAY),
    }
}

/// A byte count as `B` under a kibibyte, else `KB`/`MB`/`GB` to one decimal.
///
/// Integer arithmetic throughout: a float round trip would lose precision on
/// multi-gigabyte caches, which is exactly the size these get.
#[must_use]
pub fn human_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    let (unit, name) = if n < KB {
        return format!("{n} B");
    } else if n < MB {
        (KB, "KB")
    } else if n < GB {
        (MB, "MB")
    } else {
        (GB, "GB")
    };
    // Tenths, rounded down, so a size never reads larger than it is.
    let tenths = n.saturating_mul(10) / unit;
    format!("{}.{} {name}", tenths / 10, tenths % 10)
}

/// Renders the pane as a plain-text tree for the stdout REPL.
///
/// Box-drawing prefixes come from each row's depth and whether it is the last
/// of its siblings, so the text view and the TUI pane show the same shape.
/// Ends without a trailing newline: the REPL prints this with `println!`.
#[must_use]
pub fn render_text(pane: &KvPane) -> String {
    use std::fmt::Write as _;
    let rows = pane.rows();
    // For each row and each ancestor level, whether that level continues below
    // this row — which decides a `│` versus a blank in the gutter.
    let mut out = String::new();
    for (i, row) in rows.iter().enumerate() {
        let mut prefix = String::new();
        for level in 1..row.depth {
            prefix.push_str(if continues(&rows, i, level) {
                "│  "
            } else {
                "   "
            });
        }
        if row.depth > 0 {
            prefix.push_str(if continues(&rows, i, row.depth) {
                "├─ "
            } else {
                "└─ "
            });
        }
        let marker = if row.has_children && !row.expanded {
            "+ "
        } else {
            ""
        };
        // Trimmed at the end: the orphan header has an empty `right` field, so
        // an untrimmed line would carry trailing whitespace.
        let line = format!("{prefix}{marker}{}    {}", row.label, row.right);
        let _ = writeln!(out, "{}", line.trim_end());
        if !row.detail.is_empty() {
            let gutter = "   ".repeat(row.depth);
            let _ = writeln!(out, "{gutter}   {}", row.detail.trim_end());
        }
    }
    // The blank separator belongs between the tree and the footer, so an empty
    // forest must not open with one.
    if !rows.is_empty() {
        out.push('\n');
    }
    let _ = write!(out, "{}", pane.footer());
    out
}

/// Whether another row at `level` follows row `i` without leaving the subtree,
/// i.e. whether row `i` has a later sibling at that depth.
fn continues(rows: &[Row], i: usize, level: usize) -> bool {
    rows.iter()
        .skip(i + 1)
        .take_while(|r| r.depth >= level)
        .any(|r| r.depth == level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvmeta::{KvLabel, KvMeta, KvRole, META_VERSION};
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    const DAY: u64 = 86_400;
    const NOW: u64 = 1_000 * DAY;

    fn meta(role: KvRole, fp: &str, parent: Option<&str>, bytes: u64) -> KvMeta {
        KvMeta {
            version: META_VERSION,
            role,
            fingerprint: fp.into(),
            parent: parent.map(ToOwned::to_owned),
            model: "m".into(),
            created: 0,
            last_used: NOW - DAY,
            hits: 3,
            bytes,
            pinned: false,
            label: KvLabel::Unknown,
        }
    }

    fn pane() -> KvPane {
        let forest = crate::kvtree::build(vec![
            meta(KvRole::System, "a19f", None, 400 * 1024 * 1024),
            meta(KvRole::Project, "7c02", Some("a19f"), 88 * 1024 * 1024),
            meta(KvRole::Session, "cheeky-bell", Some("7c02"), 1024 * 1024),
        ]);
        KvPane::new(
            forest,
            crate::kvgc::SweepPolicy {
                ttl_session_secs: 14 * DAY,
                ttl_tier_secs: 30 * DAY,
                max_bytes: 0,
            },
            0,
            NOW,
        )
    }

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    #[test]
    fn human_bytes_reads_at_a_glance() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(human_bytes(23 * 1024 * 1024 * 1024), "23.0 GB");
    }

    #[test]
    fn rows_start_fully_expanded_and_indent_by_depth() {
        let rows = pane().rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].depth, 0);
        assert_eq!(rows[1].depth, 1);
        assert_eq!(rows[2].depth, 2);
        assert!(rows[0].label.starts_with("system"));
        assert!(rows[1].label.starts_with("project"));
        assert!(rows[2].label.starts_with("session"));
        assert!(rows[0].selected, "the first row starts selected");
    }

    #[test]
    fn collapsing_hides_a_whole_subtree() {
        let mut p = pane();
        // Left on the selected root collapses it.
        assert!(matches!(
            p.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)),
            Outcome::Stay
        ));
        let rows = p.rows();
        assert_eq!(rows.len(), 1, "children and grandchildren both hidden");
        assert!(!rows[0].expanded);
        p.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(p.rows().len(), 3);
    }

    #[test]
    fn navigation_clamps_at_both_ends() {
        let mut p = pane();
        for _ in 0..10 {
            p.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        }
        assert!(p.rows()[2].selected);
        for _ in 0..10 {
            p.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        }
        assert!(p.rows()[0].selected);
    }

    #[test]
    fn action_keys_name_the_selected_node() {
        let mut p = pane();
        assert!(matches!(p.handle_key(key('p')), Outcome::Pin(fp) if fp == "a19f"));
        // The pane flips its own copy, so a second press is the inverse.
        assert!(matches!(p.handle_key(key('p')), Outcome::Unpin(fp) if fp == "a19f"));
        p.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        // Delete asks first and only fires on confirmation.
        assert!(matches!(p.handle_key(key('d')), Outcome::Stay));
        assert!(matches!(p.handle_key(key('y')), Outcome::Delete(fp) if fp == "7c02"));
        assert!(matches!(p.handle_key(key('g')), Outcome::Sweep));
        assert!(matches!(
            p.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            Outcome::Close
        ));
    }

    #[test]
    fn a_cancelled_delete_does_not_fire() {
        let mut p = pane();
        assert!(matches!(p.handle_key(key('d')), Outcome::Stay));
        assert!(
            matches!(
                p.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
                Outcome::Stay
            ),
            "Esc cancels the confirmation rather than closing the pane"
        );
        assert!(matches!(p.handle_key(key('g')), Outcome::Sweep));
    }

    #[test]
    fn the_footer_reports_the_total_and_what_a_sweep_would_free() {
        let f = pane().footer();
        assert!(f.contains("489.0 MB"), "total: {f}");
        assert!(f.contains("0 B reclaimable"), "nothing is expired: {f}");
    }

    #[test]
    fn expired_nodes_are_marked_and_counted_as_reclaimable() {
        let mut m = meta(KvRole::Session, "old", None, 2048);
        m.last_used = NOW - 99 * DAY;
        let p = KvPane::new(
            crate::kvtree::build(vec![m]),
            crate::kvgc::SweepPolicy {
                ttl_session_secs: 14 * DAY,
                ttl_tier_secs: 30 * DAY,
                max_bytes: 0,
            },
            0,
            NOW,
        );
        assert!(p.rows()[0].right.contains("expired"));
        assert!(p.footer().contains("2.0 KB reclaimable"));
    }

    /// A pane over one long-idle blob, optionally already pinned on disk.
    fn expired_pane(pinned: bool) -> KvPane {
        let mut m = meta(KvRole::Session, "old", None, 2048);
        m.last_used = NOW - 99 * DAY;
        m.pinned = pinned;
        KvPane::new(
            crate::kvtree::build(vec![m]),
            crate::kvgc::SweepPolicy {
                ttl_session_secs: 14 * DAY,
                ttl_tier_secs: 30 * DAY,
                max_bytes: 0,
            },
            0,
            NOW,
        )
    }

    #[test]
    fn pinning_an_expired_row_takes_its_bytes_out_of_the_reclaimable_total() {
        let mut p = expired_pane(false);
        assert!(p.footer().contains("2.0 KB reclaimable"));
        assert!(matches!(p.handle_key(key('p')), Outcome::Pin(fp) if fp == "old"));
        assert!(
            p.footer().contains("0 B reclaimable"),
            "a pinned row is not reclaimable: {}",
            p.footer()
        );
        let right = &p.rows()[0].right;
        assert!(right.contains("pinned"), "{right}");
        assert!(!right.contains("expired"), "{right}");
    }

    #[test]
    fn unpinning_a_pinned_but_idle_row_puts_its_bytes_back() {
        let mut p = expired_pane(true);
        assert!(p.footer().contains("0 B reclaimable"), "{}", p.footer());
        assert!(!p.rows()[0].right.contains("expired"));
        assert!(matches!(p.handle_key(key('p')), Outcome::Unpin(fp) if fp == "old"));
        assert!(
            p.footer().contains("2.0 KB reclaimable"),
            "only the pin was sparing it: {}",
            p.footer()
        );
        let right = &p.rows()[0].right;
        assert!(right.contains("expired"), "{right}");
        assert!(!right.contains("pinned"), "{right}");
    }

    #[test]
    fn render_text_produces_a_tree_the_plain_repl_can_print() {
        let out = render_text(&pane());
        assert!(out.contains("system"));
        assert!(
            out.contains("└─") || out.contains("├─"),
            "drawn as a tree: {out}"
        );
        assert!(out.lines().count() >= 4, "rows plus a footer");
        // No trailing blank line: the REPL prints this with println!.
        assert!(!out.ends_with("\n\n"));
        // No trailing whitespace on any line either. The orphan header has an
        // empty right-hand column, which used to leave the padding behind.
        for line in out.lines() {
            assert_eq!(line, line.trim_end(), "trailing whitespace: {line:?}");
        }
    }

    #[test]
    fn render_text_on_an_empty_forest_is_just_the_footer() {
        // The blank line separates the tree from the footer, so with no tree it
        // must not appear: the output used to open with a stray empty line.
        let pane = KvPane::new(
            crate::kvtree::KvForest::default(),
            crate::kvgc::SweepPolicy {
                ttl_session_secs: 14 * DAY,
                ttl_tier_secs: 30 * DAY,
                max_bytes: 0,
            },
            0,
            NOW,
        );
        let out = render_text(&pane);
        assert!(out.starts_with("total "), "{out:?}");
        assert_eq!(out.lines().count(), 1, "{out:?}");
    }

    #[test]
    fn orphans_get_their_own_header_group() {
        let p = KvPane::new(
            crate::kvtree::build(vec![
                meta(KvRole::System, "a19f", None, 100),
                meta(KvRole::Session, "lost", Some("vanished"), 50),
            ]),
            crate::kvgc::SweepPolicy {
                ttl_session_secs: 14 * DAY,
                ttl_tier_secs: 30 * DAY,
                max_bytes: 0,
            },
            0,
            NOW,
        );
        let rows = p.rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1].label, "(orphaned)");
        assert!(rows[1].has_children);
        assert!(rows[1].fingerprint.is_empty());
        assert_eq!(rows[2].depth, 1);
    }

    #[test]
    fn the_footer_names_the_budget_when_the_cache_is_over_it() {
        let p = KvPane::new(
            crate::kvtree::build(vec![meta(KvRole::Session, "big", None, 4096)]),
            crate::kvgc::SweepPolicy {
                ttl_session_secs: 14 * DAY,
                ttl_tier_secs: 30 * DAY,
                max_bytes: 0,
            },
            1024,
            NOW,
        );
        assert!(
            p.footer().contains("over the 1.0 KB budget"),
            "{}",
            p.footer()
        );
    }
}
