// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Retention policy for persisted KV blobs.
//!
//! A sweep runs in two phases.
//!
//! **Phase 1 — TTL and pins.** Per node, stopping at the first match:
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
//! **Phase 2 — the `kvcache.maxBytes` ceiling.** Phase 1 alone puts no upper
//! bound on the cache: Tier 1 forks on every model, reasoning level, plank
//! version and global MCP set, Tier 2 on every `AGENTS.md` edit, and session
//! payloads run to ~1 GB each, so a steady state of tens of GB is ordinary.
//! Phase 2 therefore evicts survivors, least recently used first, until the
//! total is at or under the budget.
//!
//! `maxBytes` used to be advisory for one reason: a node's fate had to be
//! independent of directory scan order, and a size-driven rule seemed to make
//! it depend on which node was visited first. Sorting resolves that. Phase 2
//! runs only once phase 1's verdicts are fully determined, and it evicts in a
//! *globally sorted* order — ascending `last_used`, ties broken by fingerprint
//! — so the outcome is a pure function of the inputs, scan order included.
//!
//! Two asymmetries are deliberate. Phase 2 re-derives "has a surviving child"
//! against the **post-phase-1** set, where phase 1 reads the pre-sweep set:
//! otherwise a parent whose only child just expired would outlive every budget
//! forever. And `maxBytes == 0` means *unbounded*, never "evict everything" —
//! the inverse reading would wipe the whole cache for anyone who never sets the
//! key. A budget is also a target, not a licence: when every remaining node is
//! pinned, active, or a parent of a survivor, the sweep leaves the total over
//! budget rather than deleting something protected.

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
    /// Hard ceiling on total cache bytes, enforced after the TTL pass. `0`
    /// disables the budget pass entirely — it means "unbounded", never
    /// "evict everything".
    pub max_bytes: u64,
}

impl SweepPolicy {
    /// The policy the `kvcache` settings block describes.
    #[must_use]
    pub fn from_settings(s: &crate::settings::KvCacheSettings) -> Self {
        Self {
            ttl_session_secs: s.ttl_session_days.saturating_mul(SECS_PER_DAY),
            ttl_tier_secs: s.ttl_tier_days.saturating_mul(SECS_PER_DAY),
            max_bytes: s.max_bytes,
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
///
/// Two phases. Phase 1 applies the per-node TTL and pin rules (see the module
/// docs). Phase 2 then enforces `policy.max_bytes` over what phase 1 left,
/// evicting least-recently-used survivors — ties broken by fingerprint — until
/// the total fits. `max_bytes == 0` skips phase 2 entirely: it means
/// "unbounded".
///
/// The budget is a target, not a licence. Phase 2 never touches a node that is
/// pinned, active, or the parent of a phase-1 survivor, so when everything left
/// is protected the plan leaves the total over budget rather than deleting
/// something it must keep.
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

    // Phase 2: a hard ceiling. Runs only after phase 1's verdicts are fully
    // determined, so the survivor set below is a pure function of the inputs and
    // the eviction order does not depend on how the directory was scanned.
    if policy.max_bytes == 0 {
        return plan;
    }
    let doomed: HashSet<usize> = plan.doomed.iter().copied().collect();
    let mut total: u64 = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| !doomed.contains(i))
        .fold(0u64, |acc, (_, m)| acc.saturating_add(m.bytes));
    if total <= policy.max_bytes {
        return plan;
    }
    // "Has a surviving child" is re-derived against the POST-phase-1 set:
    // unlike phase 1, where reading the pre-sweep set is what makes the result
    // order-independent, here a parent whose only child just expired must become
    // evictable or it would outlive every budget forever.
    let survivor_parents: HashSet<&str> = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| !doomed.contains(i))
        .filter_map(|(_, m)| m.parent.as_deref())
        .collect();
    let mut candidates: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| !doomed.contains(i))
        .filter(|(_, m)| {
            !m.pinned
                && !active.contains(&m.fingerprint.as_str())
                && !survivor_parents.contains(m.fingerprint.as_str())
        })
        .map(|(i, _)| i)
        .collect();
    // Least-recently-used first, ties broken by fingerprint so the order is
    // total and reproducible.
    candidates.sort_by(|&a, &b| {
        nodes[a]
            .last_used
            .cmp(&nodes[b].last_used)
            .then_with(|| nodes[a].fingerprint.cmp(&nodes[b].fingerprint))
    });
    for i in candidates {
        if total <= policy.max_bytes {
            break;
        }
        total = total.saturating_sub(nodes[i].bytes);
        plan.bytes = plan.bytes.saturating_add(nodes[i].bytes);
        plan.doomed.push(i);
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
            max_bytes: 0,
        }
    }

    /// A policy with the given budget and TTLs long enough that phase 1 never
    /// fires, so a test isolates the budget pass.
    fn budget_only(max_bytes: u64) -> SweepPolicy {
        SweepPolicy {
            ttl_session_secs: 9_999 * DAY,
            ttl_tier_secs: 9_999 * DAY,
            max_bytes,
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
            max_bytes: 0,
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

    #[test]
    fn a_zero_budget_disables_the_budget_pass() {
        // 0 means unbounded, not "evict everything" — the inverse mistake would
        // wipe the whole cache on every launch for anyone who never sets it.
        let nodes = vec![
            node(KvRole::Session, "a", None, 1),
            node(KvRole::Session, "b", None, 2),
        ];
        assert!(
            plan_sweep(&nodes, &[], &budget_only(0), NOW)
                .doomed
                .is_empty()
        );
    }

    #[test]
    fn a_total_under_budget_evicts_nothing() {
        let nodes = vec![node(KvRole::Session, "a", None, 1)];
        // `node` builds 100-byte blobs.
        assert!(
            plan_sweep(&nodes, &[], &budget_only(1_000), NOW)
                .doomed
                .is_empty()
        );
    }

    #[test]
    fn the_budget_pass_evicts_least_recently_used_first_until_under_budget() {
        // Four 100-byte nodes, budget 250: evict until <= 250, so two must go,
        // and they must be the two idlest.
        let nodes = vec![
            node(KvRole::Session, "newest", None, 1),
            node(KvRole::Session, "oldest", None, 40),
            node(KvRole::Session, "middle", None, 20),
            node(KvRole::Session, "second-newest", None, 5),
        ];
        let plan = plan_sweep(&nodes, &[], &budget_only(250), NOW);
        let mut gone: Vec<&str> = plan
            .doomed
            .iter()
            .map(|&i| nodes[i].fingerprint.as_str())
            .collect();
        gone.sort_unstable();
        assert_eq!(gone, vec!["middle", "oldest"]);
        assert_eq!(plan.bytes, 200);
    }

    #[test]
    fn the_budget_pass_spares_pinned_active_and_parents() {
        // Every protection from phase 1 still holds in phase 2, so a tiny
        // budget cannot evict a pinned node, the live chain, or an ancestor of
        // a survivor — it just fails to reach the target.
        let mut pinned = node(KvRole::Session, "pinned", None, 99);
        pinned.pinned = true;
        let nodes = vec![
            pinned,
            node(KvRole::System, "live", None, 99),
            node(KvRole::System, "parent", None, 99),
            node(KvRole::Session, "child", Some("parent"), 1),
            node(KvRole::Session, "evictable", None, 99),
        ];
        let plan = plan_sweep(&nodes, &["live"], &budget_only(1), NOW);
        let mut gone: Vec<&str> = plan
            .doomed
            .iter()
            .map(|&i| nodes[i].fingerprint.as_str())
            .collect();
        gone.sort_unstable();
        assert_eq!(
            gone,
            vec!["child", "evictable"],
            "a budget is a target, not a licence to delete protected nodes"
        );
    }

    #[test]
    fn the_budget_pass_counts_only_what_phase_1_left() {
        // Phase 1 kills 200 bytes of expired sessions; the remaining 100 is
        // already under a 150 budget, so phase 2 must evict nothing more.
        let p = SweepPolicy {
            ttl_session_secs: 14 * DAY,
            ttl_tier_secs: 30 * DAY,
            max_bytes: 150,
        };
        let nodes = vec![
            node(KvRole::Session, "expired-1", None, 99),
            node(KvRole::Session, "expired-2", None, 99),
            node(KvRole::Session, "fresh", None, 1),
        ];
        let plan = plan_sweep(&nodes, &[], &p, NOW);
        let mut gone: Vec<&str> = plan
            .doomed
            .iter()
            .map(|&i| nodes[i].fingerprint.as_str())
            .collect();
        gone.sort_unstable();
        assert_eq!(gone, vec!["expired-1", "expired-2"]);
    }

    #[test]
    fn a_parent_orphaned_by_phase_1_becomes_evictable_in_phase_2() {
        // Phase 2's "has a surviving child" test reads the POST-phase-1 set,
        // unlike phase 1's, which reads the pre-sweep set. Otherwise a parent
        // whose only child just expired would be immortal under any budget.
        let p = SweepPolicy {
            ttl_session_secs: 14 * DAY,
            ttl_tier_secs: 9_999 * DAY,
            max_bytes: 1,
        };
        let nodes = vec![
            node(KvRole::System, "parent", None, 99),
            node(KvRole::Session, "dead-child", Some("parent"), 99),
        ];
        let plan = plan_sweep(&nodes, &[], &p, NOW);
        let mut gone: Vec<&str> = plan
            .doomed
            .iter()
            .map(|&i| nodes[i].fingerprint.as_str())
            .collect();
        gone.sort_unstable();
        assert_eq!(gone, vec!["dead-child", "parent"]);
    }

    #[test]
    fn the_budget_pass_is_deterministic_regardless_of_input_order() {
        // The whole reason phase 2 sorts before evicting. Two nodes with the
        // same last_used tie-break on fingerprint, so reversing the input must
        // not change the outcome.
        let a = node(KvRole::Session, "aaa", None, 50);
        let b = node(KvRole::Session, "bbb", None, 50);
        let fwd = plan_sweep(&[a.clone(), b.clone()], &[], &budget_only(100), NOW);
        let rev = plan_sweep(&[b, a], &[], &budget_only(100), NOW);
        assert_eq!(fwd.doomed.len(), 1);
        assert_eq!(rev.doomed.len(), 1);
        assert_eq!(fwd.bytes, rev.bytes);
        // Forward: index 0 is "aaa"; reversed: index 1 is "aaa". Same node.
        assert_eq!(fwd.doomed[0], 0);
        assert_eq!(rev.doomed[0], 1);
    }
}
