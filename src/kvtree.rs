// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Building the KV lineage tree that `/kvcache` draws.
//!
//! Pure: a `Vec<KvMeta>` in, a [`KvForest`] out, no filesystem and no engine.
//! Cycles are impossible by construction — every chained fingerprint embeds its
//! parent's, so a cycle would require a hash collision — but the build is
//! written not to loop regardless, because the input is untrusted JSON.

use crate::kvmeta::KvMeta;

/// One blob and everything that extends it.
#[derive(Debug, Clone)]
pub struct KvNode {
    /// Position of this blob in the slice [`build`] was given, which is also
    /// its position in [`crate::session::SessionStore::kv_blob_nodes`] — so the
    /// index identifies the *file*, where a fingerprint does not: two bodies can
    /// legitimately carry one fingerprint, and a session body's sidecar
    /// fingerprint is its payload signature rather than anything in its name.
    pub idx: usize,
    /// This blob's metadata.
    pub meta: KvMeta,
    /// Blobs whose `parent` names this one, largest subtree first.
    pub children: Vec<KvNode>,
}

impl KvNode {
    /// Bytes held by this blob and every descendant.
    #[must_use]
    pub fn subtree_bytes(&self) -> u64 {
        self.children.iter().fold(self.meta.bytes, |acc, c| {
            acc.saturating_add(c.subtree_bytes())
        })
    }
}

/// The whole cache as a tree, plus whatever could not be attached to one.
#[derive(Debug, Clone, Default)]
pub struct KvForest {
    /// Parentless blobs — normally the system tiers, one per live prompt.
    pub roots: Vec<KvNode>,
    /// Blobs naming a parent that is not present. Rendered under their own
    /// heading rather than dropped: a blob you cannot see is a blob you cannot
    /// delete, and these are exactly the ones worth deleting.
    pub orphans: Vec<KvNode>,
}

impl KvForest {
    /// Bytes across every root and orphan.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.roots
            .iter()
            .chain(&self.orphans)
            .fold(0u64, |acc, n| acc.saturating_add(n.subtree_bytes()))
    }
}

/// Depth beyond which the build stops descending. The real tree is four tiers
/// deep; this only bounds pathological input.
const MAX_DEPTH: usize = 16;

/// Assembles `nodes` into a forest, ordering siblings largest subtree first.
///
/// Each [`KvNode`] records the index it came from, so a caller that walks the
/// forest can still name the exact file the node was read from.
///
/// Every node appears exactly once, duplicate fingerprints included; when two
/// nodes share a fingerprint only the first to be attached absorbs the children
/// keyed under it, and the later one renders childless. A node naming an absent
/// or self-referential parent becomes an orphan, as does one whose parent is
/// present but itself unreachable.
#[must_use]
pub fn build(nodes: Vec<KvMeta>) -> KvForest {
    use std::cmp::Reverse;
    use std::collections::{HashMap, HashSet};

    // Recursive, depth-bounded attach: the input is untrusted JSON, so the
    // build must terminate even on input a chained fingerprint could not
    // actually produce. `MAX_DEPTH` is the actual safeguard; recursion depth
    // is bounded by it regardless of how `children` is shaped.
    type Item = (usize, KvMeta);
    fn attach(item: Item, children: &mut HashMap<String, Vec<Item>>, depth: usize) -> KvNode {
        let (idx, m) = item;
        let mut kids: Vec<KvNode> = if depth >= MAX_DEPTH {
            Vec::new()
        } else {
            children
                .remove(&m.fingerprint)
                .unwrap_or_default()
                .into_iter()
                .map(|c| attach(c, children, depth + 1))
                .collect()
        };
        // Cached: `subtree_bytes` is O(subtree), and `sort_by_key` would
        // recompute it on every comparison.
        kids.sort_by_cached_key(|b| Reverse(b.subtree_bytes()));
        KvNode {
            idx,
            meta: m,
            children: kids,
        }
    }

    let present: HashSet<String> = nodes.iter().map(|m| m.fingerprint.clone()).collect();
    let mut children: HashMap<String, Vec<Item>> = HashMap::new();
    let mut roots: Vec<Item> = Vec::new();
    let mut orphans: Vec<Item> = Vec::new();
    for (idx, m) in nodes.into_iter().enumerate() {
        match m.parent.clone() {
            None => roots.push((idx, m)),
            Some(p) if p == m.fingerprint || !present.contains(&p) => orphans.push((idx, m)),
            Some(p) => children.entry(p).or_default().push((idx, m)),
        }
    }

    let mut root_nodes: Vec<KvNode> = roots
        .into_iter()
        .map(|m| attach(m, &mut children, 0))
        .collect();
    root_nodes.sort_by_cached_key(|b| Reverse(b.subtree_bytes()));
    let mut orphan_nodes: Vec<KvNode> = orphans
        .into_iter()
        .map(|m| attach(m, &mut children, 0))
        .collect();
    // Anything still unattached had a present parent that was itself
    // unreachable; surface it rather than dropping it.
    let stragglers: Vec<Item> = children.into_values().flatten().collect();
    orphan_nodes.extend(stragglers.into_iter().map(|(idx, m)| KvNode {
        idx,
        meta: m,
        children: Vec::new(),
    }));
    orphan_nodes.sort_by_cached_key(|b| Reverse(b.subtree_bytes()));

    KvForest {
        roots: root_nodes,
        orphans: orphan_nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvmeta::{KvLabel, KvRole, META_VERSION};

    fn meta(role: KvRole, fp: &str, parent: Option<&str>, bytes: u64) -> KvMeta {
        KvMeta {
            version: META_VERSION,
            role,
            fingerprint: fp.into(),
            parent: parent.map(ToOwned::to_owned),
            model: "m".into(),
            created: 0,
            last_used: 0,
            hits: 0,
            bytes,
            pinned: false,
            label: KvLabel::Unknown,
        }
    }

    #[test]
    fn builds_a_single_chain() {
        let f = build(vec![
            meta(KvRole::Session, "s1", Some("p1"), 400),
            meta(KvRole::System, "y1", None, 100),
            meta(KvRole::Project, "p1", Some("y1"), 200),
        ]);
        assert_eq!(f.roots.len(), 1);
        assert!(f.orphans.is_empty());
        assert_eq!(f.roots[0].meta.fingerprint, "y1");
        assert_eq!(f.roots[0].children[0].meta.fingerprint, "p1");
        assert_eq!(f.roots[0].children[0].children[0].meta.fingerprint, "s1");
        assert_eq!(f.roots[0].subtree_bytes(), 700);
        assert_eq!(f.total_bytes(), 700);
    }

    #[test]
    fn multiple_system_roots_coexist() {
        // The whole point of dropping the keep-set: two system prompts alive
        // at once, each with its own subtree.
        let f = build(vec![
            meta(KvRole::System, "y1", None, 100),
            meta(KvRole::System, "y2", None, 150),
            meta(KvRole::Project, "p1", Some("y1"), 10),
        ]);
        assert_eq!(f.roots.len(), 2);
        assert_eq!(f.total_bytes(), 260);
    }

    #[test]
    fn a_node_whose_parent_is_gone_is_an_orphan_not_a_loss() {
        let f = build(vec![
            meta(KvRole::System, "y1", None, 100),
            meta(KvRole::Session, "s1", Some("vanished"), 400),
        ]);
        assert_eq!(f.roots.len(), 1);
        assert_eq!(f.orphans.len(), 1);
        assert_eq!(f.orphans[0].meta.fingerprint, "s1");
        assert_eq!(f.total_bytes(), 500, "orphans still count toward the total");
    }

    #[test]
    fn a_self_parented_node_is_an_orphan_and_does_not_hang() {
        let f = build(vec![meta(KvRole::Project, "p1", Some("p1"), 10)]);
        assert!(f.roots.is_empty());
        assert_eq!(f.orphans.len(), 1);
    }

    #[test]
    fn an_empty_pool_builds_an_empty_forest() {
        let f = build(Vec::new());
        assert!(f.roots.is_empty() && f.orphans.is_empty());
        assert_eq!(f.total_bytes(), 0);
    }

    #[test]
    fn a_mutual_cycle_surfaces_both_nodes_and_does_not_hang() {
        // Impossible from plank's own writes, since every chained fingerprint
        // embeds its parent's, but the input is untrusted JSON. Neither node is
        // a root and neither parent is absent, so the only correct outcome is
        // that both fall out as stragglers rather than vanishing.
        let f = build(vec![
            meta(KvRole::Project, "a", Some("b"), 10),
            meta(KvRole::Project, "b", Some("a"), 20),
        ]);
        assert!(f.roots.is_empty());
        let mut fps: Vec<&str> = f
            .orphans
            .iter()
            .map(|n| n.meta.fingerprint.as_str())
            .collect();
        fps.sort_unstable();
        assert_eq!(fps, vec!["a", "b"], "never silently drop a node");
        assert_eq!(f.total_bytes(), 30);
    }

    /// Nodes in one subtree, including its own root.
    fn count(n: &KvNode) -> usize {
        1 + n.children.iter().map(count).sum::<usize>()
    }

    #[test]
    fn a_chain_deeper_than_max_depth_truncates_the_display_but_loses_nothing() {
        // Display stops descending at MAX_DEPTH; the nodes below it must still
        // be reachable, because a blob you cannot see is a blob you cannot
        // delete.
        let depth = MAX_DEPTH + 5;
        let mut nodes = vec![meta(KvRole::System, "n000", None, 1)];
        for i in 1..=depth {
            let fp = format!("n{i:03}");
            let parent = format!("n{:03}", i - 1);
            nodes.push(meta(KvRole::Session, &fp, Some(&parent), 1));
        }
        let f = build(nodes);
        assert_eq!(f.roots.len(), 1);
        // The chain under the single root is cut at MAX_DEPTH...
        let mut n = &f.roots[0];
        let mut reached = 1usize;
        while let Some(child) = n.children.first() {
            n = child;
            reached += 1;
        }
        assert_eq!(reached, MAX_DEPTH + 1, "descent stops at MAX_DEPTH");
        // ...and every node the descent did not reach is still present.
        let total: usize = f.roots.iter().chain(&f.orphans).map(count).sum();
        assert_eq!(total, depth + 1, "never silently drop a node");
        assert_eq!(f.total_bytes(), (depth + 1) as u64);
    }

    #[test]
    fn every_node_remembers_the_slice_position_it_came_from() {
        // The index is how a caller names the *file* a node was read from, so it
        // must survive the sibling sort, the root sort and the orphan path. A
        // fingerprint cannot do that job: two bodies may share one, and a
        // session body's sidecar fingerprint is its payload signature rather
        // than its file stem.
        fn walk(n: &KvNode, out: &mut Vec<usize>) {
            out.push(n.idx);
            for c in &n.children {
                walk(c, out);
            }
        }
        let nodes = vec![
            meta(KvRole::System, "y1", None, 1),
            meta(KvRole::Project, "small", Some("y1"), 10),
            meta(KvRole::Project, "big", Some("y1"), 900),
            meta(KvRole::Session, "lost", Some("gone"), 5),
        ];
        let f = build(nodes.clone());
        assert_eq!(f.roots[0].idx, 0);
        assert_eq!(f.roots[0].children[0].meta.fingerprint, "big");
        assert_eq!(f.roots[0].children[0].idx, 2, "the sort must not renumber");
        assert_eq!(f.roots[0].children[1].idx, 1);
        assert_eq!(f.orphans[0].idx, 3);
        // Every position appears exactly once across the whole forest.
        let mut all = Vec::new();
        for n in f.roots.iter().chain(&f.orphans) {
            walk(n, &mut all);
        }
        all.sort_unstable();
        assert_eq!(all, (0..nodes.len()).collect::<Vec<_>>());
    }

    #[test]
    fn children_sort_largest_first_so_the_tree_leads_with_what_costs() {
        let f = build(vec![
            meta(KvRole::System, "y1", None, 1),
            meta(KvRole::Project, "small", Some("y1"), 10),
            meta(KvRole::Project, "big", Some("y1"), 900),
        ]);
        assert_eq!(f.roots[0].children[0].meta.fingerprint, "big");
    }
}
