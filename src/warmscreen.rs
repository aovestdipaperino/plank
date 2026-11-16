// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The launch checklist: one row per KV cache tier, shown all at once while
//! the cache is (re)built.
//!
//! Pure state, no drawing: [`crate::tui::draw_warm`] renders it. Rows start as
//! the plan from [`crate::kvtier::warm_plan`] predicted, and the stage events
//! from [`crate::kvtier::warm`] move them: `Start` makes a row active (even one
//! the plan called cached, when a checkpoint turned out unusable), `Finish`
//! makes it done.

use crate::kvtier::{StageEvent, StageState, TierKind, WarmStage};

/// Where one row of the checklist stands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowState {
    /// Cached, or finished prefilling: drawn with a checkmark.
    Done,
    /// Prefilling now: drawn with a live bar.
    Active {
        /// Tokens evaluated so far in this tier's pass.
        done: i32,
        /// Tokens this tier's pass has to evaluate.
        total: i32,
        /// Current prefill rate in tokens per second.
        tps: f64,
    },
    /// Still to prefill: drawn with an empty bar.
    Waiting,
}

/// One tier (or the sub-agent's cache) in the checklist.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The tier, or `None` for the sub-agent row.
    pub kind: Option<TierKind>,
    /// What to print next to the marker.
    pub label: String,
    /// Where it stands.
    pub state: RowState,
}

/// All rows of the launch checklist plus which one is prefilling.
#[derive(Debug, Clone, Default)]
pub struct WarmScreen {
    rows: Vec<Row>,
    active: Option<usize>,
}

impl WarmScreen {
    /// Builds the rows a [`crate::kvtier::warm_plan`] predicted.
    #[must_use]
    pub fn from_plan(plan: &[WarmStage]) -> Self {
        Self {
            rows: plan
                .iter()
                .map(|s| Row {
                    kind: Some(s.kind),
                    label: s.kind.short_label().to_owned(),
                    state: Self::initial(s.state),
                })
                .collect(),
            active: None,
        }
    }

    fn initial(state: StageState) -> RowState {
        match state {
            StageState::Cached => RowState::Done,
            StageState::Pending => RowState::Waiting,
        }
    }

    /// Appends the sub-agent's cache as a final row, driven by [`Self::alt_stage`].
    pub fn push_alt(&mut self, label: &str, state: StageState) {
        self.rows.push(Row {
            kind: None,
            label: label.to_owned(),
            state: Self::initial(state),
        });
    }

    /// The rows, top to bottom.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Whether any row still has work to do.
    #[must_use]
    pub fn any_pending(&self) -> bool {
        self.rows.iter().any(|r| r.state != RowState::Done)
    }

    /// Applies a stage event from the main engine's warm walk.
    pub fn stage(&mut self, ev: StageEvent) {
        let kind = match ev {
            StageEvent::Start(k) | StageEvent::Finish(k) => k,
        };
        let idx = self.rows.iter().position(|r| r.kind == Some(kind));
        self.apply(idx, ev);
    }

    /// Applies a stage event from the sub-agent engine's walk, which always
    /// reports `System` and must reach only the alt row.
    pub fn alt_stage(&mut self, ev: StageEvent) {
        let idx = self.rows.iter().rposition(|r| r.kind.is_none());
        self.apply(idx, ev);
    }

    fn apply(&mut self, idx: Option<usize>, ev: StageEvent) {
        let Some(i) = idx else { return };
        match ev {
            StageEvent::Start(_) => {
                self.rows[i].state = RowState::Active {
                    done: 0,
                    total: 1,
                    tps: 0.0,
                };
                self.active = Some(i);
            }
            StageEvent::Finish(_) => {
                self.rows[i].state = RowState::Done;
                if self.active == Some(i) {
                    self.active = None;
                }
            }
        }
    }

    /// Records prefill progress on the row that is prefilling, if any.
    pub fn progress(&mut self, done: i32, total: i32, tps: f64) {
        if let Some(i) = self.active {
            self.rows[i].state = RowState::Active { done, total, tps };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvtier::{StageEvent, StageState, TierKind, WarmStage};

    fn plan() -> Vec<WarmStage> {
        vec![
            WarmStage {
                kind: TierKind::System,
                state: StageState::Cached,
            },
            WarmStage {
                kind: TierKind::ProjectStable,
                state: StageState::Pending,
            },
            WarmStage {
                kind: TierKind::SessionVolatile,
                state: StageState::Pending,
            },
        ]
    }

    #[test]
    fn rows_start_as_the_plan_predicted() {
        let s = WarmScreen::from_plan(&plan());
        let states: Vec<_> = s.rows().iter().map(|r| r.state).collect();
        assert_eq!(
            states,
            vec![RowState::Done, RowState::Waiting, RowState::Waiting]
        );
        assert_eq!(s.rows()[0].label, "System prompt");
        assert!(s.any_pending());
    }

    #[test]
    fn start_progress_finish_walk_a_row_through_its_states() {
        let mut s = WarmScreen::from_plan(&plan());
        s.stage(StageEvent::Start(TierKind::ProjectStable));
        s.progress(10, 40, 100.0);
        assert_eq!(
            s.rows()[1].state,
            RowState::Active {
                done: 10,
                total: 40,
                tps: 100.0
            }
        );
        assert_eq!(s.rows()[2].state, RowState::Waiting);
        s.stage(StageEvent::Finish(TierKind::ProjectStable));
        assert_eq!(s.rows()[1].state, RowState::Done);
        // Progress with no active row is ignored, not a panic.
        s.progress(1, 2, 3.0);
        assert_eq!(s.rows()[1].state, RowState::Done);
    }

    #[test]
    fn a_planned_cached_row_that_really_prefills_becomes_active() {
        let mut s = WarmScreen::from_plan(&plan());
        s.stage(StageEvent::Start(TierKind::System));
        assert!(matches!(s.rows()[0].state, RowState::Active { .. }));
    }

    #[test]
    fn the_alt_row_is_driven_separately_from_the_tier_rows() {
        let mut s = WarmScreen::from_plan(&plan());
        s.push_alt("Sub-agent system prompt", StageState::Pending);
        assert_eq!(s.rows().len(), 4);
        assert_eq!(s.rows()[3].kind, None);
        // The alt walk reports a `System` start; it must hit the alt row only.
        s.alt_stage(StageEvent::Start(TierKind::System));
        assert!(matches!(s.rows()[3].state, RowState::Active { .. }));
        assert_eq!(s.rows()[0].state, RowState::Done);
        s.alt_stage(StageEvent::Finish(TierKind::System));
        assert_eq!(s.rows()[3].state, RowState::Done);
    }

    #[test]
    fn nothing_pending_when_every_row_is_cached() {
        let all_cached: Vec<_> = plan()
            .into_iter()
            .map(|w| WarmStage {
                state: StageState::Cached,
                ..w
            })
            .collect();
        assert!(!WarmScreen::from_plan(&all_cached).any_pending());
    }
}
