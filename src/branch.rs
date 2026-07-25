// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Tree-structured sessions: in-place branching (`/tree`, `/fork`, `/clone`).
//!
//! A plank session is stored as a linear transcript, but conversations are
//! naturally a *tree*: from any earlier user message you can try a different
//! prompt without losing what you already explored. This module holds that
//! tree and the pure operations over it; [`crate::ui`] owns the commands and
//! [`crate::session`] owns persistence.
//!
//! # Model
//!
//! One [`Node`] per transcript message, each pointing at its parent. The
//! *active path* is the chain from a root down to the active node, and its
//! messages are exactly the live `Session::transcript`. Everything else in the
//! tree is an *off-path* node: a branch you forked away from, kept so `/tree`
//! can show it and a later `/fork` can return to it.
//!
//! - [`BranchTree::append`] extends the active path (what every turn does).
//! - [`BranchTree::fork_at`] moves the active node to a message's *parent*, so
//!   that message and everything after it become an off-path branch and the
//!   next prompt starts a sibling.
//! - [`BranchTree::clone_active`] duplicates the whole active path into a
//!   second, identical branch, so the original is preserved verbatim while the
//!   copy continues.
//!
//! # KV-cache discipline
//!
//! Both operations are deliberately shaped so the engine's cached KV prefix
//! survives them. A fork leaves the active transcript a strict *prefix* of what
//! it was, and a clone leaves it byte-identical; since every turn rebuilds the
//! prompt and reconciles it against the live session with a token common-prefix
//! probe (see [`crate::ds4engine`]), the shared prefix is reused and only the
//! divergent tail is re-evaluated. No KV state is copied, restored, or
//! reinterpreted here — a branch switch never hands the engine bytes captured
//! for a different token sequence. [`BranchTree::shared_prefix_len`] reports
//! how many leading messages two branches share, which is the upper bound on
//! that reuse.
//!
//! # Persistence
//!
//! [`BranchTree::into_parts`] splits the tree back into `(active path,
//! off-path nodes)`. The active path is written with the pre-existing `msg`
//! records, so a session with a single branch is byte-identical to one written
//! before branching existed, and any older session file loads as a
//! single-branch tree. Off-path nodes get extra `node` records, whose parent
//! ids refer to active-path indices or to earlier `node` ids.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::session::{Message, Role, title_clip};

/// Index of a node within a [`BranchTree`].
pub type NodeId = usize;

/// An off-path node as persisted next to the transcript.
///
/// `id` is the node's index in the serialized numbering (active-path messages
/// occupy `0..path.len()`, off-path nodes continue from there); `parent` is
/// `None` for a branch root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffNode {
    /// Serialized id of this node.
    pub id: NodeId,
    /// Serialized id of the parent node; `None` for a root.
    pub parent: Option<NodeId>,
    /// The message this node carries.
    pub message: Message,
}

/// One message in the tree, with its parent and children links.
#[derive(Debug, Clone)]
pub struct Node {
    /// Parent node, or `None` for a branch root.
    pub parent: Option<NodeId>,
    /// Children in creation order.
    pub children: Vec<NodeId>,
    /// The message this node carries.
    pub message: Message,
}

/// A session transcript as a tree of messages with one active path.
#[derive(Debug, Clone, Default)]
pub struct BranchTree {
    nodes: Vec<Node>,
    roots: Vec<NodeId>,
    active: Option<NodeId>,
}

impl BranchTree {
    /// Builds an empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuilds a tree from a persisted `(active path, off-path nodes)` pair.
    ///
    /// Off-path nodes are accepted in order; one whose parent id is unknown
    /// (a truncated or corrupt file, or a transcript rewritten by compaction)
    /// is dropped along with its descendants, rather than trusted. A tree
    /// built from an empty `off` is exactly the linear session — this is what
    /// makes every pre-branching session file load unchanged.
    #[must_use]
    pub fn from_parts(path: &[Message], off: &[OffNode]) -> Self {
        let mut tree = Self::new();
        let mut map: HashMap<NodeId, NodeId> = HashMap::new();
        for (i, m) in path.iter().enumerate() {
            let id = tree.push_node(if i == 0 { None } else { Some(i - 1) }, m.clone());
            map.insert(i, id);
        }
        tree.active = path.len().checked_sub(1);
        for node in off {
            let parent = match node.parent {
                None => None,
                Some(p) => match map.get(&p) {
                    Some(&actual) => Some(actual),
                    // Unknown parent: skip this node; its children then find
                    // no parent either and are skipped in turn.
                    None => continue,
                },
            };
            let id = tree.push_node(parent, node.message.clone());
            map.insert(node.id, id);
        }
        tree
    }

    /// Splits the tree into its persisted `(active path, off-path nodes)`
    /// form, renumbering so active-path messages occupy `0..path.len()` and
    /// every off-path node is emitted after its parent.
    #[must_use]
    pub fn into_parts(&self) -> (Vec<Message>, Vec<OffNode>) {
        let path = self.active_path();
        let mut ids: HashMap<NodeId, NodeId> = HashMap::new();
        for (i, &n) in path.iter().enumerate() {
            ids.insert(n, i);
        }
        let messages: Vec<Message> = path
            .iter()
            .map(|&n| self.nodes[n].message.clone())
            .collect();

        let mut off = Vec::new();
        let mut next = messages.len();
        // Depth-first from the roots: a parent is always visited (and so
        // numbered) before its children.
        let mut stack: Vec<NodeId> = self.roots.iter().rev().copied().collect();
        while let Some(n) = stack.pop() {
            if !ids.contains_key(&n) {
                let parent = self.nodes[n].parent.map(|p| ids[&p]);
                ids.insert(n, next);
                off.push(OffNode {
                    id: next,
                    parent,
                    message: self.nodes[n].message.clone(),
                });
                next += 1;
            }
            for &c in self.nodes[n].children.iter().rev() {
                stack.push(c);
            }
        }
        (messages, off)
    }

    fn push_node(&mut self, parent: Option<NodeId>, message: Message) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            parent,
            children: Vec::new(),
            message,
        });
        match parent {
            Some(p) => self.nodes[p].children.push(id),
            None => self.roots.push(id),
        }
        id
    }

    /// Total number of nodes across every branch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the tree holds no messages at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Borrows a node by id.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// The active node: the last message of the live transcript.
    #[must_use]
    pub fn active(&self) -> Option<NodeId> {
        self.active
    }

    /// Node ids from the active branch's root down to the active node.
    #[must_use]
    pub fn active_path(&self) -> Vec<NodeId> {
        self.path_to(self.active)
    }

    /// Node ids from a branch root down to `node` (empty when `node` is
    /// `None`).
    #[must_use]
    pub fn path_to(&self, node: Option<NodeId>) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut cur = node;
        while let Some(n) = cur {
            out.push(n);
            cur = self.nodes[n].parent;
        }
        out.reverse();
        out
    }

    /// Messages on the active path — the live transcript.
    #[must_use]
    pub fn active_messages(&self) -> Vec<Message> {
        self.active_path()
            .into_iter()
            .map(|n| self.nodes[n].message.clone())
            .collect()
    }

    /// Appends a message as a child of the active node and makes it active.
    pub fn append(&mut self, message: Message) -> NodeId {
        let id = self.push_node(self.active, message);
        self.active = Some(id);
        id
    }

    /// Number of leading messages two branches share.
    ///
    /// This is the message-level bound on KV prefix reuse when switching from
    /// the branch ending at `a` to the one ending at `b`: everything up to the
    /// divergence is the same token sequence, so the engine's common-prefix
    /// probe keeps it cached and only re-evaluates past it.
    #[must_use]
    pub fn shared_prefix_len(&self, a: NodeId, b: NodeId) -> usize {
        let pa = self.path_to(Some(a));
        let pb = self.path_to(Some(b));
        pa.iter().zip(pb.iter()).take_while(|(x, y)| x == y).count()
    }

    /// Forks at `node`: the active node becomes `node`'s parent, so `node` and
    /// its descendants stay in the tree as a sibling branch and the next
    /// appended message starts a new one.
    ///
    /// Returns the new active node (`None` when forking at a branch root, which
    /// empties the transcript).
    ///
    /// # Errors
    ///
    /// Returns an error when `node` is not a node of this tree.
    pub fn fork_at(&mut self, node: NodeId) -> Result<Option<NodeId>, String> {
        let parent = self
            .nodes
            .get(node)
            .ok_or_else(|| "no such point in the session tree".to_owned())?
            .parent;
        self.active = parent;
        Ok(parent)
    }

    /// Duplicates the active path into a second, identical branch and makes
    /// the copy active.
    ///
    /// The copy hangs off the same parent as the active path's root (a new
    /// root when the path starts at one), so the original branch is untouched
    /// and both render the same transcript. Because the copy's messages are
    /// identical, the engine's cached prefix stays valid in full.
    ///
    /// Returns the new active node, or `None` when there is nothing to clone.
    pub fn clone_active(&mut self) -> Option<NodeId> {
        let path = self.active_path();
        if path.is_empty() {
            return None;
        }
        let mut parent = self.nodes[path[0]].parent;
        for &n in &path {
            let message = self.nodes[n].message.clone();
            parent = Some(self.push_node(parent, message));
        }
        self.active = parent;
        parent
    }

    /// Leaf nodes, one per branch tip, in creation order.
    #[must_use]
    pub fn leaves(&self) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&n| self.nodes[n].children.is_empty())
            .collect()
    }

    /// Number of branches in the tree; 1 for a linear session.
    ///
    /// Every leaf is a branch tip, plus the active node when it is not one —
    /// which is exactly the state right after a `/fork`, where the active
    /// branch is a prefix of the branch it forked away from.
    #[must_use]
    pub fn branch_count(&self) -> usize {
        let leaves = self.leaves();
        let active_is_interior = self
            .active
            .is_some_and(|a| !self.nodes[a].children.is_empty());
        leaves.len() + usize::from(active_is_interior)
    }

    /// Fork points on the active path: the real (non-tool-result) user
    /// messages, oldest first. `/fork n` selects the `n`-th of these (1-based).
    #[must_use]
    pub fn fork_points(&self) -> Vec<NodeId> {
        self.active_path()
            .into_iter()
            .filter(|&n| {
                let m = &self.nodes[n].message;
                m.role == Role::User && !is_tool_user(m)
            })
            .collect()
    }
}

/// Canonicalizes a persisted `(path, off-nodes)` pair: drops off-path nodes
/// whose parents no longer exist and renumbers the rest.
///
/// Called on load and on save so a transcript rewritten out from under the
/// tree (compaction, a truncated file) can never resurrect a bogus branch.
#[must_use]
pub fn canonicalize(path: &[Message], off: &[OffNode]) -> Vec<OffNode> {
    if off.is_empty() {
        return Vec::new();
    }
    BranchTree::from_parts(path, off).into_parts().1
}

/// Mirror of `session`'s tool-result detection (that helper is crate-private
/// to its module's callers).
fn is_tool_user(m: &Message) -> bool {
    let t = m.text.trim();
    m.role == Role::User
        && (t.starts_with("<tool_result>")
            || t.starts_with("Tool:")
            || t.starts_with("Tool result"))
}

/// One-line summary of a message for the tree view.
fn summarize(m: &Message) -> String {
    let text = m.text.trim();
    let first = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let clipped = title_clip(first.trim(), 60);
    if clipped.is_empty() {
        "(empty)".to_owned()
    } else {
        clipped
    }
}

/// Renders the session tree the way `/tree` prints it.
///
/// Linear runs of messages collapse into one line each, so the output shows
/// the fork structure rather than every turn; the active branch is marked with
/// `*`. A trailing section numbers the fork points `/fork <n>` accepts.
#[must_use]
pub fn render_tree(tree: &BranchTree, color: bool) -> String {
    if tree.is_empty() {
        return "the session tree is empty; start a conversation first\n".to_owned();
    }
    let (head_on, mark_on, dim, reset) = if color {
        ("\x1b[1;97m", "\x1b[1;96m", "\x1b[90m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };
    let active: Vec<NodeId> = tree.active_path();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{head_on}session tree{reset} {dim}({} branch{}, * = active){reset}",
        tree.branch_count(),
        if tree.branch_count() == 1 { "" } else { "es" }
    );
    for root in tree_roots(tree) {
        render_run(tree, root, 0, &active, color, &mut out);
    }

    let points = tree.fork_points();
    if points.is_empty() {
        let _ = writeln!(out, "{dim}no user messages to fork from yet{reset}");
        return out;
    }
    let _ = writeln!(out, "\n{head_on}fork points on the active branch{reset}");
    for (i, &n) in points.iter().enumerate() {
        let mark = if i + 1 == points.len() { "*" } else { " " };
        let msg = tree
            .node(n)
            .map(|x| summarize(&x.message))
            .unwrap_or_default();
        let _ = writeln!(out, "{mark_on}{mark}{:>3}.{reset} {msg}", i + 1);
    }
    let _ = writeln!(
        out,
        "{dim}Use /fork <n> to branch from a point, /clone to duplicate this branch.{reset}"
    );
    out
}

/// Root ids in creation order.
fn tree_roots(tree: &BranchTree) -> Vec<NodeId> {
    (0..tree.len())
        .filter(|&n| tree.node(n).is_some_and(|x| x.parent.is_none()))
        .collect()
}

/// Prints the linear run starting at `start` as one line, then recurses into
/// the children of the node the run ends at.
fn render_run(
    tree: &BranchTree,
    start: NodeId,
    depth: usize,
    active: &[NodeId],
    color: bool,
    out: &mut String,
) {
    let (mark_on, dim, reset) = if color {
        ("\x1b[1;96m", "\x1b[90m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let mut run = vec![start];
    let mut cur = start;
    while let Some(node) = tree.node(cur) {
        if node.children.len() != 1 {
            break;
        }
        cur = node.children[0];
        run.push(cur);
    }
    let on_active = run.iter().any(|n| active.contains(n));
    let mark = if on_active { "*" } else { " " };
    // The last real user prompt in the run labels it; otherwise its first
    // message does.
    let label = run
        .iter()
        .rev()
        .find_map(|&n| {
            tree.node(n)
                .filter(|x| x.message.role == Role::User && !is_tool_user(&x.message))
                .map(|x| summarize(&x.message))
        })
        .or_else(|| tree.node(start).map(|x| summarize(&x.message)))
        .unwrap_or_default();
    let indent = "  ".repeat(depth);
    let _ = writeln!(
        out,
        "{mark_on}{mark}{reset} {indent}{dim}[{} msg{}]{reset} {label}",
        run.len(),
        if run.len() == 1 { "" } else { "s" }
    );
    let children: Vec<NodeId> = tree
        .node(cur)
        .map(|x| x.children.clone())
        .unwrap_or_default();
    for c in children {
        render_run(tree, c, depth + 1, active, color, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_of(msgs: &[Message]) -> BranchTree {
        BranchTree::from_parts(msgs, &[])
    }

    fn texts(tree: &BranchTree) -> Vec<String> {
        tree.active_messages().into_iter().map(|m| m.text).collect()
    }

    fn linear() -> BranchTree {
        tree_of(&[
            Message::user("one"),
            Message::assistant("a1"),
            Message::user("two"),
            Message::assistant("a2"),
        ])
    }

    #[test]
    fn linear_transcript_is_a_single_branch_tree() {
        let t = linear();
        assert_eq!(t.len(), 4);
        assert_eq!(t.branch_count(), 1);
        assert_eq!(t.active(), Some(3));
        assert_eq!(texts(&t), ["one", "a1", "two", "a2"]);
        assert_eq!(t.into_parts().1, Vec::new());
    }

    #[test]
    fn empty_tree_has_no_active_node() {
        let t = tree_of(&[]);
        assert!(t.is_empty());
        assert_eq!(t.active(), None);
        assert_eq!(t.branch_count(), 0);
        assert!(t.active_path().is_empty());
    }

    #[test]
    fn append_extends_the_active_path() {
        let mut t = linear();
        let id = t.append(Message::user("three"));
        assert_eq!(t.active(), Some(id));
        assert_eq!(texts(&t), ["one", "a1", "two", "a2", "three"]);
        assert_eq!(t.branch_count(), 1);
    }

    #[test]
    fn fork_moves_active_to_the_parent_and_keeps_the_old_tail() {
        let mut t = linear();
        // Fork at "two" (node 2): the transcript rewinds to before it.
        assert_eq!(t.fork_at(2).unwrap(), Some(1));
        assert_eq!(texts(&t), ["one", "a1"]);
        // The old tail survives as an off-path branch.
        assert_eq!(t.len(), 4);
        assert_eq!(t.branch_count(), 2);
        t.append(Message::user("two-prime"));
        assert_eq!(texts(&t), ["one", "a1", "two-prime"]);
        assert_eq!(t.branch_count(), 2);
    }

    #[test]
    fn fork_at_a_root_empties_the_active_path() {
        let mut t = linear();
        assert_eq!(t.fork_at(0).unwrap(), None);
        assert!(t.active_path().is_empty());
        t.append(Message::user("fresh"));
        assert_eq!(texts(&t), ["fresh"]);
        // The original branch is still a root of the tree.
        assert_eq!(t.branch_count(), 2);
    }

    #[test]
    fn fork_at_an_unknown_node_errors() {
        let mut t = linear();
        assert!(t.fork_at(99).is_err());
        assert_eq!(texts(&t), ["one", "a1", "two", "a2"]);
    }

    #[test]
    fn clone_duplicates_the_active_branch_verbatim() {
        let mut t = linear();
        let before = texts(&t);
        let tip = t.clone_active().unwrap();
        assert_eq!(t.active(), Some(tip));
        assert_eq!(texts(&t), before);
        assert_eq!(t.len(), 8);
        assert_eq!(t.branch_count(), 2);
        // Continuing the clone leaves the original untouched.
        t.append(Message::user("only-on-clone"));
        let original_tip = 3;
        let original: Vec<String> = t
            .path_to(Some(original_tip))
            .into_iter()
            .map(|n| t.node(n).unwrap().message.text.clone())
            .collect();
        assert_eq!(original, before);
    }

    #[test]
    fn clone_of_an_empty_tree_is_a_noop() {
        let mut t = tree_of(&[]);
        assert_eq!(t.clone_active(), None);
        assert!(t.is_empty());
    }

    #[test]
    fn shared_prefix_len_measures_the_common_ancestry() {
        let mut t = linear();
        t.fork_at(2).unwrap();
        let b = t.append(Message::user("two-prime"));
        // Old tip is node 3 ("a2"); the branches share "one" and "a1".
        assert_eq!(t.shared_prefix_len(3, b), 2);
        assert_eq!(t.shared_prefix_len(b, b), 3);
        assert_eq!(t.shared_prefix_len(0, 3), 1);
    }

    #[test]
    fn a_clone_shares_no_nodes_but_the_same_messages() {
        let mut t = linear();
        let tip = t.clone_active().unwrap();
        // Distinct nodes, so the paths share nothing structurally...
        assert_eq!(t.shared_prefix_len(3, tip), 0);
        // ...yet the rendered messages are identical, which is what the
        // engine's token common-prefix probe actually reuses.
        let a: Vec<String> = t
            .path_to(Some(3))
            .into_iter()
            .map(|n| t.node(n).unwrap().message.text.clone())
            .collect();
        assert_eq!(a, texts(&t));
    }

    #[test]
    fn round_trip_through_parts_preserves_the_tree() {
        let mut t = linear();
        t.fork_at(2).unwrap();
        t.append(Message::user("two-prime"));
        t.append(Message::assistant("a2-prime"));
        let (path, off) = t.into_parts();
        assert_eq!(
            path.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(),
            ["one", "a1", "two-prime", "a2-prime"]
        );
        // Off-path nodes are numbered after the active path and reference
        // their parent by the new numbering.
        assert_eq!(off.len(), 2);
        assert_eq!(off[0].id, 4);
        assert_eq!(off[0].parent, Some(1));
        assert_eq!(off[1].parent, Some(4));

        let back = BranchTree::from_parts(&path, &off);
        assert_eq!(texts(&back), texts(&t));
        assert_eq!(back.len(), t.len());
        assert_eq!(back.branch_count(), t.branch_count());
        assert_eq!(back.into_parts().1, off);
    }

    #[test]
    fn round_trip_survives_multiple_roots() {
        let mut t = linear();
        t.fork_at(0).unwrap();
        t.append(Message::user("fresh"));
        let (path, off) = t.into_parts();
        assert_eq!(path.len(), 1);
        assert_eq!(off[0].parent, None);
        let back = BranchTree::from_parts(&path, &off);
        assert_eq!(back.len(), 5);
        assert_eq!(texts(&back), ["fresh"]);
    }

    #[test]
    fn off_nodes_with_an_unknown_parent_are_dropped_with_their_descendants() {
        let path = [Message::user("one")];
        let off = [
            OffNode {
                id: 7,
                parent: Some(42),
                message: Message::user("orphan"),
            },
            OffNode {
                id: 8,
                parent: Some(7),
                message: Message::assistant("orphan child"),
            },
            OffNode {
                id: 9,
                parent: Some(0),
                message: Message::assistant("legit"),
            },
        ];
        let t = BranchTree::from_parts(&path, &off);
        assert_eq!(t.len(), 2);
        // The active node ("one") is interior now, so it is its own branch
        // alongside the one continuing into "legit".
        assert_eq!(t.branch_count(), 2);
        let kept = t.into_parts().1;
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].message.text, "legit");
    }

    #[test]
    fn canonicalize_renumbers_and_drops_orphans() {
        let path = [Message::user("one"), Message::assistant("a1")];
        let off = [OffNode {
            id: 50,
            parent: Some(0),
            message: Message::user("other"),
        }];
        let out = canonicalize(&path, &off);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 2);
        assert_eq!(out[0].parent, Some(0));
        assert!(canonicalize(&path, &[]).is_empty());
    }

    #[test]
    fn fork_points_skip_tool_results_and_assistant_turns() {
        let t = tree_of(&[
            Message::user("one"),
            Message::assistant("a1"),
            Message::user("<tool_result>x</tool_result>"),
            Message::assistant("a2"),
            Message::user("two"),
        ]);
        assert_eq!(t.fork_points(), vec![0, 4]);
    }

    #[test]
    fn render_marks_the_active_branch_and_lists_fork_points() {
        let mut t = linear();
        t.fork_at(2).unwrap();
        t.append(Message::user("two-prime"));
        let out = render_tree(&t, false);
        assert!(out.contains("2 branches"), "{out}");
        assert!(out.contains("two-prime"), "{out}");
        assert!(out.contains("fork points on the active branch"), "{out}");
        // Exactly one run line is marked active per level of the active path.
        let marked: Vec<&str> = out.lines().filter(|l| l.starts_with('*')).collect();
        assert!(!marked.is_empty(), "{out}");
        assert!(!out.contains("\x1b["), "{out}");
    }

    #[test]
    fn render_draws_the_fork_structure() {
        let mut t = tree_of(&[
            Message::user("write a parser"),
            Message::assistant("a1"),
            Message::user("use nom instead"),
            Message::assistant("a2"),
        ]);
        t.fork_at(2).unwrap();
        t.append(Message::user("hand-roll it instead"));
        t.append(Message::assistant("a3"));
        assert_eq!(
            render_tree(&t, false),
            "session tree (2 branches, * = active)\n\
             * [2 msgs] write a parser\n\
             \x20   [2 msgs] use nom instead\n\
             *   [2 msgs] hand-roll it instead\n\
             \n\
             fork points on the active branch\n\
             \x20  1. write a parser\n\
             *  2. hand-roll it instead\n\
             Use /fork <n> to branch from a point, /clone to duplicate this branch.\n"
        );
    }

    #[test]
    fn render_of_an_empty_tree_is_a_hint() {
        let out = render_tree(&BranchTree::new(), false);
        assert!(out.contains("empty"), "{out}");
    }

    #[test]
    fn render_with_color_emits_ansi() {
        let out = render_tree(&linear(), true);
        assert!(out.contains("\x1b["), "{out}");
    }
}
