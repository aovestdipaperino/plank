// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Cooperative multiplexing of several sessions on the caller's own thread.
//!
//! [`EngineHost`](crate::host::EngineHost) already interleaves sessions
//! round-robin, but it owns a dedicated GPU worker thread and an admission
//! layer, and only `plank serve --shared-engine` runs one. The interactive
//! agent drives a single [`Ds4Session`](crate::ds4engine::Ds4Session) directly
//! and has no host to borrow — the same gap that made
//! [`Ds4Session::fork`](crate::ds4engine::Ds4Session::fork) necessary
//! (`docs/SESSION-CLONE-DESIGN.md`).
//!
//! [`SliceRunner`] is the missing half: given several sessions — typically a
//! live one plus its forks — it advances each in turn by
//! `slice_tokens`, on whatever thread calls [`run`](SliceRunner::run). No
//! thread is spawned, nothing is admitted, and the caller keeps ownership.
//!
//! **This is time-slicing, not parallelism.** One Metal command queue means
//! total wall time is roughly the sum of the jobs, not the max. What it buys is
//! that every stream *advances*: an aside answers while the main task keeps
//! inching forward, and N subagents make visible progress together instead of
//! the first finishing before the second starts. Nothing finishes sooner. For
//! a fan-out it is the difference between a live dashboard and a queue; for a
//! single `/btw` it is the difference between a frozen main task and a moving
//! one.
//!
//! Prefill is non-preemptible, matching the host (design §5, §11.8): a job's
//! first slice runs its whole prefill before yielding. Jobs therefore begin
//! interleaving only once each has been prefilled.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::engine::{EngineError, EngineEvent, GenerationOptions, GenerationStats};
use crate::host::HostSession;

/// Identifies one job within a [`SliceRunner`], and tags the events it emits.
/// Assigned in insertion order, so it doubles as the dispatch index callers
/// need when rejoining results deterministically (completion order is not
/// stable under interleaving).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(pub usize);

/// What one job produced: its terminal stats, or the error that ended it.
#[derive(Debug)]
pub struct JobOutcome {
    /// The job's dispatch index.
    pub id: JobId,
    /// Terminal stats, or the backend error that stopped this job. One job
    /// failing never stops the others.
    pub result: Result<GenerationStats, EngineError>,
}

struct Job<'a> {
    session: &'a mut dyn HostSession,
    interrupt: Arc<AtomicBool>,
    result: Option<Result<GenerationStats, EngineError>>,
}

/// Round-robin driver for several concurrent generations on one thread.
///
/// Jobs are *borrowed*, not owned: a caller can lend its live session for the
/// duration of a run and keep it afterwards, which is what letting a main task
/// and an aside share the thread requires.
///
/// See the [module docs](self) for what this does and does not buy.
pub struct SliceRunner<'a> {
    slice_tokens: usize,
    jobs: Vec<Job<'a>>,
    /// Indices still runnable, in service order.
    rotation: VecDeque<usize>,
}

impl<'a> SliceRunner<'a> {
    /// Creates an empty runner that yields between jobs every `slice_tokens`
    /// generated tokens. Smaller slices interleave more smoothly and cost more
    /// switching; the host's [`DEFAULT_SLICE_TOKENS`](crate::host::DEFAULT_SLICE_TOKENS)
    /// is a reasonable default. Zero is treated as one.
    #[must_use]
    pub fn new(slice_tokens: usize) -> Self {
        Self {
            slice_tokens: slice_tokens.max(1),
            jobs: Vec::new(),
            rotation: VecDeque::new(),
        }
    }

    /// Adds a session and the transcript it should generate over. The job runs
    /// on the next [`run`](Self::run); nothing happens until then.
    ///
    /// `interrupt` is polled between slices, so setting it stops just this job
    /// and leaves its siblings running.
    pub fn add(
        &mut self,
        session: &'a mut dyn HostSession,
        transcript: String,
        opts: GenerationOptions,
        interrupt: Arc<AtomicBool>,
    ) -> JobId {
        session.begin(transcript, opts);
        let id = JobId(self.jobs.len());
        self.jobs.push(Job {
            session,
            interrupt,
            result: None,
        });
        self.rotation.push_back(id.0);
        id
    }

    /// Number of jobs added so far, finished or not.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// True when no jobs have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Drives every job to completion, interleaving them a slice at a time, and
    /// returns their outcomes **sorted by [`JobId`]** — dispatch order, not the
    /// nondeterministic completion order.
    ///
    /// `on_event` receives each job's stream tagged with its id, so a caller can
    /// route one job's tokens to a side panel and another's to the main log.
    ///
    /// A job that returns an error is recorded and dropped from the rotation;
    /// the rest keep running. Interrupting a job likewise ends only that job —
    /// its stats carry `interrupted`.
    pub fn run(&mut self, on_event: &mut dyn FnMut(JobId, EngineEvent)) -> Vec<JobOutcome> {
        while let Some(idx) = self.rotation.pop_front() {
            let job = &mut self.jobs[idx];
            let id = JobId(idx);
            let mut sink = |ev: EngineEvent| on_event(id, ev);
            match job
                .session
                .advance(self.slice_tokens, &job.interrupt, &mut sink)
            {
                // More to do: back of the queue, so every sibling gets a slice
                // before this one runs again.
                Ok(None) => self.rotation.push_back(idx),
                Ok(Some(stats)) => job.result = Some(Ok(stats)),
                Err(e) => job.result = Some(Err(e)),
            }
        }
        self.take_outcomes()
    }

    /// Collects the finished jobs' results in dispatch order. A job that never
    /// produced a result (impossible after [`run`](Self::run) returns, since it
    /// drains the rotation) is reported as an error rather than silently
    /// dropped.
    fn take_outcomes(&mut self) -> Vec<JobOutcome> {
        self.jobs
            .iter_mut()
            .enumerate()
            .map(|(idx, job)| JobOutcome {
                id: JobId(idx),
                result: job
                    .result
                    .take()
                    .unwrap_or_else(|| Err(EngineError::new("job never ran"))),
            })
            .collect()
    }
}

impl std::fmt::Debug for SliceRunner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SliceRunner")
            .field("slice_tokens", &self.slice_tokens)
            .field("jobs", &self.jobs.len())
            .field("runnable", &self.rotation.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    /// A session that emits one token per `advance` and finishes after `total`,
    /// so a test can observe the interleaving directly.
    ///
    /// The `EchoEngine` stub cannot serve here: it runs a whole generation
    /// inside a single `advance`, so every job would complete on its first
    /// slice and the rotation would look serial no matter what the runner did.
    struct StepSession {
        tag: &'static str,
        total: usize,
        emitted: usize,
        fail_at: Option<usize>,
    }

    impl StepSession {
        fn steps(tag: &'static str, total: usize) -> Self {
            Self {
                tag,
                total,
                emitted: 0,
                fail_at: None,
            }
        }

        fn failing(tag: &'static str, total: usize, fail_at: usize) -> Self {
            Self {
                tag,
                total,
                emitted: 0,
                fail_at: Some(fail_at),
            }
        }
    }

    impl HostSession for StepSession {
        fn begin(&mut self, _transcript: String, _opts: GenerationOptions) {
            self.emitted = 0;
        }

        fn advance(
            &mut self,
            _k: usize,
            interrupt: &AtomicBool,
            sink: &mut dyn FnMut(EngineEvent),
        ) -> Result<Option<GenerationStats>, EngineError> {
            if interrupt.load(Ordering::Relaxed) {
                return Ok(Some(GenerationStats {
                    generated: i32::try_from(self.emitted).unwrap_or(i32::MAX),
                    interrupted: true,
                    ..GenerationStats::default()
                }));
            }
            if self.fail_at == Some(self.emitted) {
                return Err(EngineError::new(format!("{} failed", self.tag)));
            }
            self.emitted += 1;
            sink(EngineEvent::Text(format!("{}{}", self.tag, self.emitted)));
            if self.emitted >= self.total {
                return Ok(Some(GenerationStats {
                    generated: i32::try_from(self.emitted).unwrap_or(i32::MAX),
                    ..GenerationStats::default()
                }));
            }
            Ok(None)
        }

        fn ctx_tokens(&self) -> i32 {
            i32::try_from(self.emitted).unwrap_or(i32::MAX)
        }
    }

    fn opts() -> GenerationOptions {
        GenerationOptions::default()
    }

    fn flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    fn drive(runner: &mut SliceRunner) -> (Vec<JobOutcome>, Vec<(JobId, String)>) {
        let mut log = Vec::new();
        let outcomes = runner.run(&mut |id, ev| {
            if let EngineEvent::Text(t) = ev {
                log.push((id, t));
            }
        });
        (outcomes, log)
    }

    /// The point of the whole module: two jobs advance together rather than one
    /// finishing before the other starts.
    #[test]
    fn jobs_interleave_rather_than_serialize() {
        let (mut sa, mut sb) = (StepSession::steps("a", 3), StepSession::steps("b", 3));
        let mut runner = SliceRunner::new(1);
        let a = runner.add(&mut sa, "x".into(), opts(), flag());
        let b = runner.add(&mut sb, "y".into(), opts(), flag());

        let (outcomes, log) = drive(&mut runner);
        let text: Vec<&str> = log.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(
            text,
            ["a1", "b1", "a2", "b2", "a3", "b3"],
            "the two streams alternate a slice at a time"
        );
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].id, a);
        assert_eq!(outcomes[1].id, b);
        for o in &outcomes {
            assert_eq!(o.result.as_ref().unwrap().generated, 3);
        }
    }

    /// Outcomes come back in dispatch order even though the jobs finish in a
    /// different one — what a fan-out needs to rejoin reports deterministically.
    #[test]
    fn outcomes_are_ordered_by_dispatch_not_completion() {
        let (mut sa, mut sb) = (StepSession::steps("a", 4), StepSession::steps("b", 1));
        let mut runner = SliceRunner::new(1);
        // The longer job is dispatched first, so completion order is b then a.
        runner.add(&mut sa, "x".into(), opts(), flag());
        runner.add(&mut sb, "y".into(), opts(), flag());

        let (outcomes, log) = drive(&mut runner);
        let last = log.last().expect("both jobs emitted");
        assert_eq!(last.0, JobId(0), "the longer first job finishes last");
        assert_eq!(
            outcomes.iter().map(|o| o.id).collect::<Vec<_>>(),
            [JobId(0), JobId(1)],
            "outcomes stay in dispatch order regardless"
        );
    }

    /// A short job retiring must not stall or starve the survivors.
    #[test]
    fn a_finished_job_leaves_the_rotation() {
        let (mut sa, mut sb) = (StepSession::steps("a", 1), StepSession::steps("b", 3));
        let mut runner = SliceRunner::new(1);
        runner.add(&mut sa, "x".into(), opts(), flag());
        runner.add(&mut sb, "y".into(), opts(), flag());

        let (_, log) = drive(&mut runner);
        let text: Vec<&str> = log.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(
            text,
            ["a1", "b1", "b2", "b3"],
            "b keeps running alone after a retires"
        );
    }

    /// One job's interrupt is its own: siblings run to completion.
    #[test]
    fn interrupt_stops_only_its_own_job() {
        let (mut sa, mut sb) = (StepSession::steps("a", 3), StepSession::steps("b", 2));
        let mut runner = SliceRunner::new(1);
        let stop = flag();
        stop.store(true, Ordering::Relaxed);
        runner.add(&mut sa, "x".into(), opts(), stop);
        runner.add(&mut sb, "y".into(), opts(), flag());

        let (outcomes, log) = drive(&mut runner);
        let text: Vec<&str> = log.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(text, ["b1", "b2"], "the interrupted job emits nothing");
        assert!(
            outcomes[0].result.as_ref().unwrap().interrupted,
            "the interrupted job reports it"
        );
        assert!(
            !outcomes[1].result.as_ref().unwrap().interrupted,
            "its sibling ran normally"
        );
    }

    /// A backend failure is contained: it is reported for that job only, and
    /// the others still finish.
    #[test]
    fn a_failing_job_does_not_take_down_its_siblings() {
        let (mut sa, mut sb) = (StepSession::failing("a", 5, 2), StepSession::steps("b", 3));
        let mut runner = SliceRunner::new(1);
        runner.add(&mut sa, "x".into(), opts(), flag());
        runner.add(&mut sb, "y".into(), opts(), flag());

        let (outcomes, log) = drive(&mut runner);
        assert!(
            outcomes[0].result.is_err(),
            "the failing job reports its error"
        );
        assert_eq!(
            outcomes[1].result.as_ref().unwrap().generated,
            3,
            "its sibling completed"
        );
        let b_tokens = log.iter().filter(|(id, _)| *id == JobId(1)).count();
        assert_eq!(b_tokens, 3, "the sibling's whole stream came through");
    }

    /// Every event is tagged, so a caller can route jobs to different sinks.
    #[test]
    fn events_carry_their_job_id() {
        let (mut sa, mut sb) = (StepSession::steps("a", 2), StepSession::steps("b", 2));
        let mut runner = SliceRunner::new(1);
        runner.add(&mut sa, "x".into(), opts(), flag());
        runner.add(&mut sb, "y".into(), opts(), flag());

        let (_, log) = drive(&mut runner);
        for (id, text) in &log {
            let expected = if *id == JobId(0) { "a" } else { "b" };
            assert!(
                text.starts_with(expected),
                "job {id:?} emitted {text:?}, which belongs to the other job"
            );
        }
    }

    /// The caller keeps its sessions: the runner only borrows them, so a live
    /// session lent for one run is usable again afterwards. This is what lets
    /// the main task join a rotation and then carry on.
    #[test]
    fn borrowed_sessions_return_to_the_caller() {
        let (mut sa, mut sb) = (StepSession::steps("a", 2), StepSession::steps("b", 4));
        {
            let mut runner = SliceRunner::new(1);
            runner.add(&mut sa, "x".into(), opts(), flag());
            runner.add(&mut sb, "y".into(), opts(), flag());
            drive(&mut runner);
        }
        assert_eq!(sa.ctx_tokens(), 2, "the caller still owns its session");
        assert_eq!(sb.ctx_tokens(), 4);
    }

    /// Running with nothing queued is a no-op, not a hang.
    #[test]
    fn an_empty_runner_completes_immediately() {
        let mut runner = SliceRunner::new(8);
        assert!(runner.is_empty());
        let (outcomes, log) = drive(&mut runner);
        assert!(outcomes.is_empty() && log.is_empty());
    }
}
