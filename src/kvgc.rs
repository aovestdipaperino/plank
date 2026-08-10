// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Retention policy for persisted KV blobs.
//!
//! The rule, applied to each node in order and stopping at the first match:
//!
//! 1. pinned → keep
//! 2. in the tier chain this launch is using → keep
//! 3. has a surviving child → keep
//! 4. `now - last_used` past the role's TTL → delete
//!
//! Rule 3 is evaluated against the node set as it stood **before** the sweep,
//! not a set mutating as files are unlinked, so the result does not depend on
//! directory scan order. The cost is that a parent whose last child died this
//! run is collected on the *next* run — the intended bottom-up cascade.
//!
//! This replaces the fingerprint-equality GC that kept exactly the current
//! system and project checkpoints and deleted every sibling, which made
//! switching model or reasoning level cost a full re-prefill each way.
//!
//! `kvcache.maxBytes` is not consulted here. It is an advisory figure shown in
//! `/kvcache`; making it evict would make a node's fate depend on the other
//! nodes' sizes and on visit order, which is exactly what this module avoids.

use crate::kvmeta::{KvMeta, KvRole};

/// Seconds in a day, for turning the day-valued settings into TTLs.
const SECS_PER_DAY: u64 = 86_400;

/// How long each role survives after its last use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweepPolicy {
    /// TTL for [`KvRole::Session`] payloads, in seconds.
    pub ttl_session_secs: u64,
    /// TTL for [`KvRole::System`] and [`KvRole::Project`] checkpoints, in
    /// seconds.
    pub ttl_tier_secs: u64,
}

impl SweepPolicy {
    /// The policy the `kvcache` settings block describes.
    #[must_use]
    pub fn from_settings(s: &crate::settings::KvCacheSettings) -> Self {
        Self {
            ttl_session_secs: s.ttl_session_days.saturating_mul(SECS_PER_DAY),
            ttl_tier_secs: s.ttl_tier_days.saturating_mul(SECS_PER_DAY),
        }
    }

    /// TTL for `role`.
    fn ttl(&self, role: KvRole) -> u64 {
        match role {
            KvRole::Session => self.ttl_session_secs,
            KvRole::System | KvRole::Project => self.ttl_tier_secs,
        }
    }
}

/// What a sweep would remove.
#[derive(Debug, Clone, Default)]
pub struct SweepPlan {
    /// Indices into the `nodes` slice `plan_sweep` was given, one per blob to
    /// delete.
    ///
    /// Indices, not fingerprints: two distinct bodies can carry the same
    /// fingerprint — a root `sysprompt-X.kv_raw` beside a
    /// `<proj>/project-X.kv_raw`, or the same `project-X` under two project
    /// directories — and a fingerprint-keyed verdict deletes the kept namesake
    /// along with the doomed one. The caller keeps a path list parallel to
    /// `nodes`, so index *i* names exactly one file.
    pub doomed: Vec<usize>,
    /// Bytes those blobs occupy.
    pub bytes: u64,
}

/// Decides which blobs a sweep removes. Pure: no clock, no filesystem.
///
/// `active` holds the fingerprints of every tier this launch is using,
/// including those of any secondary engine — a `provider: local` sub-agent has
/// its own system fingerprint, and omitting it is what used to delete that
/// sub-agent's checkpoint on every single run.
#[must_use]
pub fn plan_sweep(nodes: &[KvMeta], active: &[&str], policy: &SweepPolicy, now: u64) -> SweepPlan {
    use std::collections::HashSet;
    // Rule 3's input: the parent set as it stood before this sweep.
    let parents: HashSet<&str> = nodes.iter().filter_map(|m| m.parent.as_deref()).collect();
    let mut plan = SweepPlan::default();
    for (i, m) in nodes.iter().enumerate() {
        let fp = m.fingerprint.as_str();
        let keep = m.pinned
            || active.contains(&fp)
            || parents.contains(fp)
            // Strictly `<`: a zero TTL must mean "collect now", so idle time
            // equal to the TTL counts as past it.
            || now.saturating_sub(m.last_used) < policy.ttl(m.role);
        if !keep {
            plan.bytes = plan.bytes.saturating_add(m.bytes);
            plan.doomed.push(i);
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvmeta::{KvLabel, META_VERSION};

    const DAY: u64 = 86_400;
    const NOW: u64 = 1_000 * DAY;

    fn node(role: KvRole, fp: &str, parent: Option<&str>, days_idle: u64) -> KvMeta {
        KvMeta {
            version: META_VERSION,
            role,
            fingerprint: fp.into(),
            parent: parent.map(ToOwned::to_owned),
            model: "m".into(),
            created: 0,
            last_used: NOW - days_idle * DAY,
            hits: 0,
            bytes: 100,
            pinned: false,
            label: KvLabel::Unknown,
        }
    }

    fn policy() -> SweepPolicy {
        SweepPolicy {
            ttl_session_secs: 14 * DAY,
            ttl_tier_secs: 30 * DAY,
        }
    }

    /// The plan's doomed *indices* resolved back to fingerprints, so the
    /// assertions read as the policy statements they are.
    fn doomed_fps(plan: &SweepPlan, nodes: &[KvMeta]) -> Vec<String> {
        let mut d: Vec<String> = plan
            .doomed
            .iter()
            .map(|&i| nodes[i].fingerprint.clone())
            .collect();
        d.sort();
        d
    }

    fn doomed(nodes: &[KvMeta], active: &[&str]) -> Vec<String> {
        doomed_fps(&plan_sweep(nodes, active, &policy(), NOW), nodes)
    }

    #[test]
    fn a_fresh_node_survives_and_a_stale_one_does_not() {
        let nodes = vec![
            node(KvRole::Session, "fresh", None, 1),
            node(KvRole::Session, "stale", None, 20),
        ];
        assert_eq!(doomed(&nodes, &[]), vec!["stale".to_owned()]);
    }

    #[test]
    fn sessions_and_tiers_use_different_ttls() {
        // 20 days idle: past the 14-day session TTL, inside the 30-day tier one.
        let nodes = vec![
            node(KvRole::Session, "sess", None, 20),
            node(KvRole::System, "sys", None, 20),
            node(KvRole::Project, "proj", None, 20),
        ];
        assert_eq!(doomed(&nodes, &[]), vec!["sess".to_owned()]);
    }

    #[test]
    fn pinning_beats_expiry() {
        let mut n = node(KvRole::Session, "kept", None, 999);
        n.pinned = true;
        assert!(doomed(&[n], &[]).is_empty());
    }

    #[test]
    fn the_active_chain_beats_expiry() {
        let nodes = vec![node(KvRole::System, "live", None, 999)];
        assert!(doomed(&nodes, &["live"]).is_empty());
        assert_eq!(doomed(&nodes, &[]), vec!["live".to_owned()]);
    }

    #[test]
    fn an_expired_parent_with_a_live_child_stays() {
        // The rule that makes multi-sysprompt retention safe: an old system
        // prompt still under an active session must not be pulled out from
        // beneath it.
        let nodes = vec![
            node(KvRole::System, "old-sys", None, 999),
            node(KvRole::Session, "live-sess", Some("old-sys"), 1),
        ];
        assert!(doomed(&nodes, &[]).is_empty());
    }

    #[test]
    fn the_cascade_takes_one_run_per_level() {
        // Both expired. The child goes this run; the parent is protected by
        // rule 3 because it is evaluated against the pre-sweep set, and goes
        // on the next run.
        let nodes = vec![
            node(KvRole::System, "old-sys", None, 999),
            node(KvRole::Session, "old-sess", Some("old-sys"), 999),
        ];
        assert_eq!(doomed(&nodes, &[]), vec!["old-sess".to_owned()]);
        let after = vec![node(KvRole::System, "old-sys", None, 999)];
        assert_eq!(doomed(&after, &[]), vec!["old-sys".to_owned()]);
    }

    #[test]
    fn several_system_prompts_coexist_while_they_are_all_in_use() {
        // The behaviour the old keep-set GC made impossible.
        let nodes = vec![
            node(KvRole::System, "sys-a", None, 1),
            node(KvRole::System, "sys-b", None, 2),
            node(KvRole::System, "sys-c", None, 3),
        ];
        assert!(doomed(&nodes, &["sys-a"]).is_empty());
    }

    #[test]
    fn the_plan_totals_the_bytes_it_will_free() {
        let nodes = vec![
            node(KvRole::Session, "s1", None, 99),
            node(KvRole::Session, "s2", None, 99),
        ];
        assert_eq!(plan_sweep(&nodes, &[], &policy(), NOW).bytes, 200);
    }

    #[test]
    fn two_bodies_sharing_a_fingerprint_get_separate_verdicts() {
        // A root `sysprompt-dup` and a `<proj>/project-dup` collide on the
        // fingerprint. The verdict is per node, so the fresh one is not dragged
        // down by its expired namesake.
        let nodes = vec![
            node(KvRole::System, "dup", None, 1),
            node(KvRole::Project, "dup", None, 999),
        ];
        let plan = plan_sweep(&nodes, &[], &policy(), NOW);
        assert_eq!(plan.doomed, vec![1], "only the expired body");
        assert_eq!(plan.bytes, 100);
    }

    #[test]
    fn a_zero_ttl_still_spares_pinned_and_active_nodes() {
        let p = SweepPolicy {
            ttl_session_secs: 0,
            ttl_tier_secs: 0,
        };
        let mut pinned = node(KvRole::Session, "pin", None, 0);
        pinned.pinned = true;
        let nodes = vec![
            pinned,
            node(KvRole::Session, "live", None, 0),
            node(KvRole::Session, "gone", None, 0),
        ];
        let plan = plan_sweep(&nodes, &["live"], &p, NOW);
        assert_eq!(doomed_fps(&plan, &nodes), vec!["gone".to_owned()]);
    }
}
