//! Shared reference-counted engine host: one model, many sessions (issue #28).
//!
//! Implements the design in `docs/SHARED-ENGINE-DESIGN.md`. The host owns a
//! [`SharedModel`](ModelHandle) behind an `Arc` (the immutable weights + Metal
//! queue, paid once) and hands out [`SessionHandle`]s, each backed by its own
//! private KV suffix. A single dedicated GPU worker thread owns every
//! `ds4_session_*` call, so the engine's single-threaded contract is preserved
//! *by construction* rather than by a mutex the C code was never audited
//! against (design §5). Sessions are interleaved cooperatively at K-token
//! granularity via round-robin; prefill is non-preemptible in v1.
//!
//! This module is engine-agnostic: it drives anything implementing
//! [`ModelHandle`]/[`HostSession`], so it is fully exercisable with the
//! [`crate::engine::EchoEngine`] stub and needs no model for CI (design §10).
//! The Metal-backed `Ds4Model`/`Ds4Session` implementations live in
//! `ds4engine.rs` under the `ds4_engine` cfg.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::engine::{
    EchoEngine, Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats,
};

/// Default round-robin slice granularity, in tokens (design §5, K≈8–16).
pub const DEFAULT_SLICE_TOKENS: usize = 12;

/// Default conservative admission cap on concurrently attached sessions.
pub const DEFAULT_MAX_SESSIONS: usize = 8;

/// Fallback estimate of resident KV bytes per context token, used by the
/// KV-bytes admission budget when the model does not supply a better figure
/// (design §7). Deliberately conservative; the real per-token cost depends on
/// the model's layer/head geometry and quant. Only used to *size* admission,
/// never to allocate — over-estimating simply admits fewer sessions.
pub const DEFAULT_KV_BYTES_PER_TOKEN: u64 = 128 * 1024;

/// Host tuning knobs.
#[derive(Debug, Clone, Copy)]
pub struct HostConfig {
    /// Maximum number of concurrently attached sessions (admission cap, §7).
    pub max_sessions: usize,
    /// Tokens generated per session per round-robin slice (§5).
    pub slice_tokens: usize,
    /// Idle-session KV reclamation threshold (design §7). `None` (default)
    /// disables reclamation — a strict no-op. When `Some(d)`, a session idle
    /// longer than `d` has its live KV snapshotted to disk and reclaimed, and is
    /// restored transparently on its next request.
    pub idle_reclaim: Option<Duration>,
    /// Default per-session context window, in tokens (design §7, v2). `None`
    /// (default) means each session gets the model's full `ctx_size`; a smaller
    /// value lets more clients fit. A per-`attach` request overrides this.
    /// Always clamped to `[1, model ctx_size]`.
    pub session_ctx_size: Option<i32>,
    /// Aggregate KV-bytes admission budget (design §7, v2). `None` (default)
    /// keeps admission count-only. When `Some(b)`, `attach` is rejected with a
    /// real [`EngineError`] once granting the new session's estimated KV bytes
    /// would push the host's total past `b` — bounding resident memory instead
    /// of OOM-ing.
    pub kv_budget_bytes: Option<u64>,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            max_sessions: DEFAULT_MAX_SESSIONS,
            slice_tokens: DEFAULT_SLICE_TOKENS,
            idle_reclaim: None,
            session_ctx_size: None,
            kv_budget_bytes: None,
        }
    }
}

/// Clamps a requested per-session context size into `[1, model_ctx]`.
fn clamp_ctx_size(requested: i32, model_ctx: i32) -> i32 {
    requested.clamp(1, model_ctx.max(1))
}

/// Estimated resident KV bytes for a session of `ctx_size` tokens.
fn est_kv_bytes(ctx_size: i32, per_token: u64) -> u64 {
    u64::try_from(ctx_size.max(0)).unwrap_or(0) * per_token
}

/// Per-session KV/context accounting (design §7, §9 step 5).
#[derive(Debug, Clone, Default)]
pub struct SessionAccount {
    /// Opaque host-assigned session id.
    pub id: u64,
    /// Configured per-session context window, in tokens (design §7, v2).
    pub ctx_size: i32,
    /// Resident context size in tokens (0 while reclaimed to disk).
    pub ctx_tokens: i32,
    /// True when the session's KV is snapshotted to disk and its live context
    /// reclaimed; it restores transparently on the next request.
    pub reclaimed: bool,
}

/// A cheap snapshot of host-level accounting, published by the scheduler thread
/// after every session-set change and generation completion, and read by
/// `/info`/status without touching the GPU thread (design §9 step 5).
#[derive(Debug, Clone, Default)]
pub struct HostStatus {
    /// Currently attached sessions (resident + idle-reclaimed).
    pub live_sessions: usize,
    /// Admission cap on concurrently attached sessions.
    pub max_sessions: usize,
    /// Per-session context window size in tokens.
    pub ctx_size: i32,
    /// Aggregate resident KV in tokens, summed over non-reclaimed sessions.
    pub resident_ctx_tokens: i64,
    /// Aggregate estimated KV bytes granted across attached sessions, summed
    /// over their configured `ctx_size` (design §7, v2). Independent of the
    /// reclaimed/resident distinction: the budget bounds *granted* contexts.
    pub kv_bytes: u64,
    /// Configured aggregate KV-bytes budget, if any (`--kv-budget-bytes`).
    pub kv_budget_bytes: Option<u64>,
    /// Per-session accounting, sorted by id.
    pub sessions: Vec<SessionAccount>,
}

/// The shared, immutable model: weights, tokenizer, and the Metal command
/// queue. Held behind an `Arc`; its strong count *is* the refcount the issue
/// asks for — the last drop frees the engine and Metal context (§4).
///
/// [`spawn`](ModelHandle::spawn) creates a fresh session over the shared
/// weights, positioned after the warm system-prompt prefix (§6). It runs on
/// the GPU worker thread, so implementations may freely touch the engine.
pub trait ModelHandle: Send + Sync + 'static {
    /// Creates a fresh session over the shared weights with a context window of
    /// `ctx_size` tokens (design §7, v2 per-client sizing). The host clamps
    /// `ctx_size` to `[1, self.ctx_size()]` before calling, so implementations
    /// may pass it straight to `ds4_session_create`.
    ///
    /// # Errors
    /// Returns [`EngineError`] if the session cannot be created.
    fn spawn(self: Arc<Self>, ctx_size: i32) -> Result<Box<dyn HostSession>, EngineError>;

    /// Human-readable model name for status displays.
    fn model_name(&self) -> String {
        String::new()
    }

    /// Context window size in tokens.
    fn ctx_size(&self) -> i32;

    /// Estimated resident KV bytes per context token, for the KV-bytes
    /// admission budget (design §7). The default is a conservative constant;
    /// a backend that knows its true per-token cost may override it.
    fn kv_bytes_per_token(&self) -> u64 {
        DEFAULT_KV_BYTES_PER_TOKEN
    }

    /// Approximate token count of `text` for context accounting. Reads only the
    /// immutable tokenizer, so it may run off the GPU thread (design §3).
    fn count_tokens(&self, text: &str) -> i32 {
        i32::try_from(text.len() / 4).unwrap_or(i32::MAX)
    }
}

/// A live session driven cooperatively on the GPU worker thread. All methods
/// run on that one thread, so implementations need not be `Send`/`Sync`.
pub trait HostSession {
    /// Begins a new generation over `transcript`; the following
    /// [`advance`](HostSession::advance) calls produce its tokens.
    fn begin(&mut self, transcript: String, opts: GenerationOptions);

    /// Advances the current generation by up to `k` tokens, streaming text via
    /// `sink`. `interrupt` is polled cooperatively.
    ///
    /// Returns `Ok(None)` when more tokens remain (yield to the next session),
    /// `Ok(Some(stats))` when this generation has finished (or was
    /// interrupted), and `Err` on a backend failure. The first call performs
    /// the non-preemptible suffix prefill before producing tokens (§5).
    ///
    /// # Errors
    /// Returns [`EngineError`] on a backend failure.
    fn advance(
        &mut self,
        k: usize,
        interrupt: &AtomicBool,
        sink: &mut dyn FnMut(EngineEvent),
    ) -> Result<Option<GenerationStats>, EngineError>;

    /// Current resident KV size in tokens, for host accounting (design §7).
    /// Reads only local session state on the GPU thread; defaults to 0.
    fn ctx_tokens(&self) -> i32 {
        0
    }

    /// Serializes the session's live KV to owned bytes so its context can be
    /// reclaimed while idle (design §7). Returns `None` when the backend cannot
    /// snapshot, in which case the scheduler leaves the session live (never
    /// reclaims it). Runs on the GPU thread.
    fn snapshot_bytes(&mut self) -> Option<Vec<u8>> {
        None
    }

    /// Restores KV from bytes previously produced by [`snapshot_bytes`], into a
    /// freshly spawned session, on the next request after reclamation. The bytes
    /// were read back from disk, so implementations MUST use the non-owning
    /// `restore_bytes` FFI path (FINDINGS: disk-read snapshots must not be freed
    /// by the engine). Runs on the GPU thread.
    ///
    /// # Errors
    /// Returns [`EngineError`] if the backend rejects the payload.
    fn restore_bytes(&mut self, _bytes: &[u8]) -> Result<(), EngineError> {
        Ok(())
    }
}

/// One streamed message from the GPU thread to a blocked client: either a
/// generation event or the terminal result. A single ordered channel keeps
/// streaming and completion in lockstep without a busy-wait (design §5).
enum Yield {
    Event(EngineEvent),
    Done(Result<GenerationStats, EngineError>),
}

/// Work submitted to the GPU worker thread.
enum Command {
    Attach {
        /// Requested per-session context size; `None` uses the host default
        /// (`session_ctx_size`, else the model's full `ctx_size`).
        ctx_size: Option<i32>,
        reply: Sender<Result<u64, EngineError>>,
    },
    Generate {
        id: u64,
        transcript: String,
        opts: Box<GenerationOptions>,
        interrupt: Arc<AtomicBool>,
        out: Sender<Yield>,
    },
    Detach {
        id: u64,
    },
    Shutdown,
}

/// An active generation in the round-robin rotation.
struct ActiveJob {
    id: u64,
    interrupt: Arc<AtomicBool>,
    out: Sender<Yield>,
}

/// The long-lived owner of one shared model. Created at `plank serve` startup
/// (or for the in-process multi-session case). Spawns the GPU worker thread and
/// hands out [`SessionHandle`]s via [`attach`](EngineHost::attach).
pub struct EngineHost {
    model: Arc<dyn ModelHandle>,
    cmd_tx: Sender<Command>,
    worker: Option<JoinHandle<()>>,
    status: Arc<Mutex<HostStatus>>,
}

impl std::fmt::Debug for EngineHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineHost")
            .field("model", &self.model.model_name())
            .finish_non_exhaustive()
    }
}

impl EngineHost {
    /// Builds a host over `model`, spawning the single GPU worker thread.
    ///
    /// # Panics
    /// Panics if the OS refuses to spawn the GPU worker thread.
    #[must_use]
    pub fn new(model: Arc<dyn ModelHandle>, cfg: HostConfig) -> Self {
        let (cmd_tx, cmd_rx) = channel::<Command>();
        let worker_model = Arc::clone(&model);
        let ctx_size = model.ctx_size();
        let status = Arc::new(Mutex::new(HostStatus {
            max_sessions: cfg.max_sessions,
            ctx_size,
            ..HostStatus::default()
        }));
        let worker_status = Arc::clone(&status);
        let worker = std::thread::Builder::new()
            .name("plank-gpu".to_string())
            .spawn(move || scheduler_loop(&worker_model, &cmd_rx, cfg, &worker_status))
            .expect("spawn GPU worker thread");
        Self {
            model,
            cmd_tx,
            worker: Some(worker),
            status,
        }
    }

    /// A cheap snapshot of live-session count and per-session KV accounting
    /// (design §9 step 5). Reads a mutex the scheduler thread refreshes on
    /// coarse events (attach/detach/reclaim/completion), never per token, so it
    /// does not contend the GPU hot path.
    ///
    /// # Panics
    /// Panics only if the status mutex was poisoned by a prior panic while held.
    #[must_use]
    pub fn status(&self) -> HostStatus {
        self.status.lock().unwrap().clone()
    }

    /// Attaches a new session at the host default context size, restoring the
    /// warm system-prompt prefix (§6).
    ///
    /// # Errors
    /// Returns a real (non-`unsupported`) [`EngineError`] when the admission
    /// cap or KV-bytes budget is reached or the session cannot be created
    /// (design §7).
    pub fn attach(&self) -> Result<SessionHandle, EngineError> {
        self.attach_sized(None)
    }

    /// Attaches a new session with a requested per-session context size (design
    /// §7, v2). `None` uses the host default; the value is clamped to
    /// `[1, ctx_size]`. Smaller contexts let more clients fit under the
    /// KV-bytes budget.
    ///
    /// # Errors
    /// Returns a real (non-`unsupported`) [`EngineError`] when the admission
    /// cap or KV-bytes budget is reached or the session cannot be created.
    pub fn attach_sized(&self, ctx_size: Option<i32>) -> Result<SessionHandle, EngineError> {
        let (reply, reply_rx) = channel();
        self.cmd_tx
            .send(Command::Attach { ctx_size, reply })
            .map_err(|_| EngineError::new("engine host stopped"))?;
        let id = reply_rx
            .recv()
            .map_err(|_| EngineError::new("engine host stopped"))??;
        Ok(SessionHandle {
            id,
            cmd_tx: self.cmd_tx.clone(),
        })
    }

    /// Human-readable model name.
    #[must_use]
    pub fn model_name(&self) -> String {
        self.model.model_name()
    }

    /// Context window size in tokens.
    #[must_use]
    pub fn ctx_size(&self) -> i32 {
        self.model.ctx_size()
    }

    /// Approximate token count of `text`.
    #[must_use]
    pub fn count_tokens(&self, text: &str) -> i32 {
        self.model.count_tokens(text)
    }
}

impl Drop for EngineHost {
    fn drop(&mut self) {
        // Tell the worker to exit and join it, so every session it owns (and
        // its Arc clone of the model) is dropped before the host's own model
        // reference — making the refcount teardown exact (design §4, §11.4).
        let _ = self.cmd_tx.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// A cheap handle to one attached session. Its [`generate`](Self::generate)
/// enqueues work on the GPU thread and blocks on the result channel, upholding
/// the synchronous generation contract while the FFI runs on another thread
/// (design §5). Dropping it detaches the session and frees its KV.
#[derive(Debug)]
pub struct SessionHandle {
    id: u64,
    cmd_tx: Sender<Command>,
}

impl SessionHandle {
    /// Runs one generation pass, streaming events via `on_event` and blocking
    /// until the GPU thread reports the terminal stats.
    ///
    /// `interrupt` is a shared flag the caller (e.g. a `plank serve` cancel
    /// request on another connection) can set to stop generation; it is polled
    /// cooperatively on the GPU thread between token slices.
    ///
    /// # Errors
    /// Returns [`EngineError`] on a backend failure or if the host has stopped.
    pub fn generate(
        &self,
        transcript: &str,
        opts: &GenerationOptions,
        interrupt: Arc<AtomicBool>,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        let (out, out_rx) = channel::<Yield>();
        self.cmd_tx
            .send(Command::Generate {
                id: self.id,
                transcript: transcript.to_string(),
                opts: Box::new(opts.clone()),
                interrupt,
                out,
            })
            .map_err(|_| EngineError::new("engine host stopped"))?;
        // Single ordered channel: events arrive in emission order, then Done.
        loop {
            match out_rx.recv() {
                Ok(Yield::Event(ev)) => on_event(ev),
                Ok(Yield::Done(result)) => return result,
                Err(_) => return Err(EngineError::new("engine host stopped")),
            }
        }
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Command::Detach { id: self.id });
    }
}

/// Monotonic session-id source (host-global; ids are opaque handles).
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

/// One attached session's scheduler-thread state. `session` is `None` while the
/// session is idle-reclaimed to disk; `snapshot_path` then holds its persisted
/// KV, restored on the next request (design §7).
struct SessionSlot {
    session: Option<Box<dyn HostSession>>,
    last_active: Instant,
    /// Configured per-session context window, in tokens (design §7, v2). Fixed
    /// at attach; used to re-`spawn` after idle reclamation and to bill the
    /// KV-bytes budget.
    ctx_size: i32,
    /// Estimated KV bytes this session is billed against the budget.
    kv_bytes: u64,
    ctx_tokens: i32,
    snapshot_path: Option<PathBuf>,
}

impl SessionSlot {
    fn new(session: Box<dyn HostSession>, ctx_size: i32, kv_bytes: u64) -> Self {
        let ctx_tokens = session.ctx_tokens();
        Self {
            session: Some(session),
            last_active: Instant::now(),
            ctx_size,
            kv_bytes,
            ctx_tokens,
            snapshot_path: None,
        }
    }
}

/// The single GPU worker thread: owns every session and interleaves their
/// generations round-robin at K-token granularity (design §5).
#[allow(clippy::similar_names)] // `stats`/`status` are both intrinsic here
fn scheduler_loop(
    model: &Arc<dyn ModelHandle>,
    cmd_rx: &Receiver<Command>,
    cfg: HostConfig,
    status: &Mutex<HostStatus>,
) {
    let mut sessions: HashMap<u64, SessionSlot> = HashMap::new();
    let mut rotation: VecDeque<ActiveJob> = VecDeque::new();
    let ctx_size = model.ctx_size();

    loop {
        // Drain pending commands. Block only when there is no runnable work,
        // so an idle host does not spin. When idle reclamation is enabled, wake
        // periodically to reclaim sessions that have gone idle (design §7).
        if rotation.is_empty() {
            match recv_or_reclaim(cmd_rx, cfg.idle_reclaim) {
                RecvOutcome::Command(cmd) => {
                    if !apply_command(cmd, model, &mut sessions, &mut rotation, cfg) {
                        return;
                    }
                }
                RecvOutcome::ReclaimTick => {
                    if let Some(threshold) = cfg.idle_reclaim {
                        reclaim_idle_sessions(&mut sessions, &rotation, threshold);
                    }
                }
                RecvOutcome::Disconnected => return,
            }
            publish_status(status, &sessions, cfg, ctx_size);
        }
        let mut dirty = false;
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !apply_command(cmd, model, &mut sessions, &mut rotation, cfg) {
                return;
            }
            dirty = true;
        }
        if dirty {
            publish_status(status, &sessions, cfg, ctx_size);
        }

        // One round-robin pass: give each active job a single K-token slice,
        // then yield to the next (fairness, no full starvation — design §5).
        let mut completed = false;
        for _ in 0..rotation.len() {
            let Some(job) = rotation.pop_front() else {
                break;
            };
            let Some(slot) = sessions.get_mut(&job.id) else {
                // Session detached mid-flight; drop the job.
                continue;
            };
            let Some(session) = slot.session.as_mut() else {
                // Reclaimed mid-flight (should not happen for a rotating job);
                // drop the job rather than lose its KV silently.
                continue;
            };
            let mut sink = |ev: EngineEvent| {
                let _ = job.out.send(Yield::Event(ev));
            };
            match session.advance(cfg.slice_tokens, &job.interrupt, &mut sink) {
                Ok(None) => {
                    slot.last_active = Instant::now();
                    rotation.push_back(job);
                }
                Ok(Some(stats)) => {
                    slot.ctx_tokens = session.ctx_tokens();
                    slot.last_active = Instant::now();
                    let _ = job.out.send(Yield::Done(Ok(stats)));
                    completed = true;
                }
                Err(e) => {
                    slot.last_active = Instant::now();
                    let _ = job.out.send(Yield::Done(Err(e)));
                    completed = true;
                }
            }
        }
        if completed {
            publish_status(status, &sessions, cfg, ctx_size);
        }
    }
}

/// The result of waiting for the next command with optional idle-reclaim wakeups.
enum RecvOutcome {
    Command(Command),
    ReclaimTick,
    Disconnected,
}

/// Blocks for the next command; when reclamation is enabled, wakes on a poll
/// interval to let the caller reclaim idle sessions.
fn recv_or_reclaim(cmd_rx: &Receiver<Command>, idle_reclaim: Option<Duration>) -> RecvOutcome {
    match idle_reclaim {
        Some(threshold) => match cmd_rx.recv_timeout(reclaim_poll_interval(threshold)) {
            Ok(cmd) => RecvOutcome::Command(cmd),
            Err(RecvTimeoutError::Timeout) => RecvOutcome::ReclaimTick,
            Err(RecvTimeoutError::Disconnected) => RecvOutcome::Disconnected,
        },
        None => match cmd_rx.recv() {
            Ok(cmd) => RecvOutcome::Command(cmd),
            Err(_) => RecvOutcome::Disconnected,
        },
    }
}

/// Poll cadence for idle checks: responsive relative to the threshold, but
/// bounded so we neither miss the deadline badly nor busy-wake.
fn reclaim_poll_interval(threshold: Duration) -> Duration {
    (threshold / 4)
        .max(Duration::from_millis(20))
        .min(Duration::from_millis(500))
}

/// Snapshots and reclaims the live KV of every attached session idle past
/// `threshold` and not currently in the rotation (design §7). A backend that
/// cannot snapshot (returns `None`) or a failed disk write leaves the session
/// live — reclamation is always safe to skip.
fn reclaim_idle_sessions(
    sessions: &mut HashMap<u64, SessionSlot>,
    rotation: &VecDeque<ActiveJob>,
    threshold: Duration,
) {
    let now = Instant::now();
    for (id, slot) in sessions.iter_mut() {
        if slot.session.is_none() {
            continue; // already reclaimed
        }
        if rotation.iter().any(|j| j.id == *id) {
            continue; // active; never reclaim a running session
        }
        if now.duration_since(slot.last_active) < threshold {
            continue;
        }
        let mut session = slot.session.take().expect("checked is_some above");
        match session.snapshot_bytes() {
            Some(bytes) => {
                let path = idle_snapshot_path(*id);
                if std::fs::write(&path, &bytes).is_ok() {
                    // Dropping `session` here frees its live KV (design §7).
                    slot.snapshot_path = Some(path);
                } else {
                    slot.session = Some(session); // keep live on write failure
                }
            }
            None => slot.session = Some(session), // backend can't snapshot
        }
    }
}

/// Disk location for an idle session's persisted KV snapshot. Keyed by pid and
/// session id so concurrent hosts do not collide.
fn idle_snapshot_path(id: u64) -> PathBuf {
    std::env::temp_dir().join(format!("plank-idle-{}-{id}.kv", std::process::id()))
}

/// Publishes a fresh accounting snapshot. Cheap: one mutex lock on coarse events.
#[allow(clippy::similar_names)] // `status`/`sessions` are the natural names here
fn publish_status(
    status: &Mutex<HostStatus>,
    sessions: &HashMap<u64, SessionSlot>,
    cfg: HostConfig,
    ctx_size: i32,
) {
    let mut accounts: Vec<SessionAccount> = sessions
        .iter()
        .map(|(id, slot)| SessionAccount {
            id: *id,
            ctx_size: slot.ctx_size,
            ctx_tokens: slot.ctx_tokens,
            reclaimed: slot.session.is_none(),
        })
        .collect();
    accounts.sort_by_key(|a| a.id);
    let resident: i64 = sessions
        .values()
        .filter(|s| s.session.is_some())
        .map(|s| i64::from(s.ctx_tokens))
        .sum();
    let kv_bytes: u64 = sessions.values().map(|s| s.kv_bytes).sum();
    let mut g = status.lock().unwrap();
    g.live_sessions = sessions.len();
    g.max_sessions = cfg.max_sessions;
    g.ctx_size = ctx_size;
    g.resident_ctx_tokens = resident;
    g.kv_bytes = kv_bytes;
    g.kv_budget_bytes = cfg.kv_budget_bytes;
    g.sessions = accounts;
}

/// Applies one command on the GPU thread. Returns `false` to stop the worker.
fn apply_command(
    cmd: Command,
    model: &Arc<dyn ModelHandle>,
    sessions: &mut HashMap<u64, SessionSlot>,
    rotation: &mut VecDeque<ActiveJob>,
    cfg: HostConfig,
) -> bool {
    match cmd {
        Command::Attach { ctx_size, reply } => {
            let model_ctx = model.ctx_size();
            // Requested size, else the host default, else the model's full ctx;
            // always clamped into the model's range (design §7, v2).
            let requested = ctx_size.or(cfg.session_ctx_size).unwrap_or(model_ctx);
            let ctx = clamp_ctx_size(requested, model_ctx);
            let new_bytes = est_kv_bytes(ctx, model.kv_bytes_per_token());
            let current_bytes: u64 = sessions.values().map(|s| s.kv_bytes).sum();
            let result = if sessions.len() >= cfg.max_sessions {
                Err(EngineError::new(format!(
                    "engine host at capacity: {} sessions already attached",
                    cfg.max_sessions
                )))
            } else if let Some(budget) = cfg.kv_budget_bytes
                && current_bytes.saturating_add(new_bytes) > budget
            {
                Err(EngineError::new(format!(
                    "engine host KV budget exceeded: {current_bytes} + {new_bytes} bytes \
                     would exceed the {budget}-byte budget ({} sessions attached)",
                    sessions.len()
                )))
            } else {
                Arc::clone(model).spawn(ctx).map(|session| {
                    let id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
                    sessions.insert(id, SessionSlot::new(session, ctx, new_bytes));
                    id
                })
            };
            let _ = reply.send(result);
        }
        Command::Generate {
            id,
            transcript,
            opts,
            interrupt,
            out,
        } => {
            let Some(slot) = sessions.get_mut(&id) else {
                let _ = out.send(Yield::Done(Err(EngineError::new("no such session"))));
                return true;
            };
            // Restore a reclaimed session's KV into a fresh session before use
            // (design §7). The bytes came from disk, so `restore_bytes` uses the
            // non-owning FFI path (FINDINGS double-free gotcha).
            if slot.session.is_none() {
                match restore_reclaimed(model, slot) {
                    Ok(()) => {}
                    Err(e) => {
                        let _ = out.send(Yield::Done(Err(e)));
                        return true;
                    }
                }
            }
            let session = slot.session.as_mut().expect("restored above");
            session.begin(transcript, *opts);
            slot.last_active = Instant::now();
            rotation.push_back(ActiveJob { id, interrupt, out });
        }
        Command::Detach { id } => {
            if let Some(slot) = sessions.remove(&id)
                && let Some(path) = slot.snapshot_path
            {
                let _ = std::fs::remove_file(path);
            }
            rotation.retain(|job| job.id != id);
        }
        Command::Shutdown => return false,
    }
    true
}

/// Spawns a fresh session and restores a reclaimed slot's persisted KV into it.
fn restore_reclaimed(
    model: &Arc<dyn ModelHandle>,
    slot: &mut SessionSlot,
) -> Result<(), EngineError> {
    let mut session = Arc::clone(model).spawn(slot.ctx_size)?;
    if let Some(path) = slot.snapshot_path.take() {
        let bytes =
            std::fs::read(&path).map_err(|e| EngineError::new(format!("restore snapshot: {e}")))?;
        session.restore_bytes(&bytes)?;
        let _ = std::fs::remove_file(&path);
    }
    slot.ctx_tokens = session.ctx_tokens();
    slot.session = Some(session);
    Ok(())
}

/// A [`ModelHandle`] over the stub [`EchoEngine`], so the shared engine (and
/// `plank serve --shared-engine`) runs without a model on any platform (dev/CI
/// and the in-process case). Each attached session is an independent
/// `EchoEngine`. Generation is not truly K-sliced here — the stub runs to
/// completion within a single `advance` — which is fine for a stub; the real
/// K-token interleaving lives in `Ds4HostSession` (design §5).
#[derive(Debug)]
pub struct EchoSharedModel {
    ctx_size: i32,
}

impl EchoSharedModel {
    /// Creates an echo-backed shared model with the given context size.
    #[must_use]
    pub fn new(ctx_size: i32) -> Self {
        Self { ctx_size }
    }
}

struct EchoSharedSession {
    engine: EchoEngine,
    pending: Option<(String, GenerationOptions)>,
    /// Running token count, so host accounting and idle reclamation have a
    /// non-trivial value to carry across a snapshot/restore cycle even without a
    /// real KV cache.
    ctx_tokens: i32,
}

impl ModelHandle for EchoSharedModel {
    fn spawn(self: Arc<Self>, ctx_size: i32) -> Result<Box<dyn HostSession>, EngineError> {
        Ok(Box::new(EchoSharedSession {
            engine: EchoEngine::new(ctx_size),
            pending: None,
            ctx_tokens: 0,
        }))
    }

    fn ctx_size(&self) -> i32 {
        self.ctx_size
    }
}

impl HostSession for EchoSharedSession {
    fn begin(&mut self, transcript: String, opts: GenerationOptions) {
        self.pending = Some((transcript, opts));
    }

    fn advance(
        &mut self,
        _k: usize,
        interrupt: &AtomicBool,
        sink: &mut dyn FnMut(EngineEvent),
    ) -> Result<Option<GenerationStats>, EngineError> {
        let Some((transcript, opts)) = self.pending.take() else {
            return Ok(Some(GenerationStats::default()));
        };
        let intr = || interrupt.load(Ordering::SeqCst);
        let stats = self.engine.generate(
            crate::engine::Prompt::Flat(&transcript),
            &opts,
            &intr,
            &|| false,
            sink,
        )?;
        self.ctx_tokens = self.ctx_tokens.saturating_add(stats.generated);
        Ok(Some(stats))
    }

    fn ctx_tokens(&self) -> i32 {
        self.ctx_tokens
    }

    // The stub has no real KV; a snapshot just round-trips the token count so
    // the serve-echo path can exercise idle reclamation end to end (design §10).
    fn snapshot_bytes(&mut self) -> Option<Vec<u8>> {
        Some(self.ctx_tokens.to_le_bytes().to_vec())
    }

    fn restore_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        if let Ok(arr) = <[u8; 4]>::try_from(bytes) {
            self.ctx_tokens = i32::from_le_bytes(arr);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test model over the echo engine: records, on a shared trace, which
    /// session emitted each slice so round-robin ordering can be asserted
    /// deterministically (the trace is written on the single GPU thread, in
    /// emission order).
    struct EchoModel {
        ctx_size: i32,
        tokens_per_gen: usize,
        trace: Arc<Mutex<Vec<u64>>>,
        next_session: AtomicU64,
    }

    struct EchoHostSession {
        id: u64,
        tokens_per_gen: usize,
        trace: Arc<Mutex<Vec<u64>>>,
        remaining: usize,
        generated: i32,
    }

    impl ModelHandle for EchoModel {
        fn spawn(self: Arc<Self>, _ctx_size: i32) -> Result<Box<dyn HostSession>, EngineError> {
            let id = self.next_session.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(EchoHostSession {
                id,
                tokens_per_gen: self.tokens_per_gen,
                trace: Arc::clone(&self.trace),
                remaining: 0,
                generated: 0,
            }))
        }
        fn ctx_size(&self) -> i32 {
            self.ctx_size
        }
    }

    impl HostSession for EchoHostSession {
        fn begin(&mut self, _transcript: String, opts: GenerationOptions) {
            self.remaining = if opts.n_predict > 0 {
                usize::try_from(opts.n_predict).unwrap_or(self.tokens_per_gen)
            } else {
                self.tokens_per_gen
            };
            self.generated = 0;
        }

        fn advance(
            &mut self,
            k: usize,
            interrupt: &AtomicBool,
            sink: &mut dyn FnMut(EngineEvent),
        ) -> Result<Option<GenerationStats>, EngineError> {
            if interrupt.load(Ordering::Relaxed) {
                return Ok(Some(GenerationStats {
                    generated: self.generated,
                    interrupted: true,
                    ..GenerationStats::default()
                }));
            }
            let n = k.min(self.remaining);
            if n > 0 {
                self.trace.lock().unwrap().push(self.id);
            }
            for _ in 0..n {
                sink(EngineEvent::Text("x".to_string()));
                self.generated += 1;
            }
            self.remaining -= n;
            if self.remaining == 0 {
                Ok(Some(GenerationStats {
                    generated: self.generated,
                    ..GenerationStats::default()
                }))
            } else {
                Ok(None)
            }
        }
    }

    fn echo_host(
        max_sessions: usize,
        tokens_per_gen: usize,
        slice: usize,
    ) -> (EngineHost, Arc<Mutex<Vec<u64>>>) {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(EchoModel {
            ctx_size: 4096,
            tokens_per_gen,
            trace: Arc::clone(&trace),
            next_session: AtomicU64::new(1),
        });
        let host = EngineHost::new(
            model,
            HostConfig {
                max_sessions,
                slice_tokens: slice,
                idle_reclaim: None,
                session_ctx_size: None,
                kv_budget_bytes: None,
            },
        );
        (host, trace)
    }

    #[test]
    fn host_attach_detach_refcount() {
        // The Arc strong count *is* the refcount: the model must drop exactly
        // once, after the last session and the host are gone (design §4, §10).
        let trace = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(EchoModel {
            ctx_size: 4096,
            tokens_per_gen: 4,
            trace,
            next_session: AtomicU64::new(1),
        });
        let weak = Arc::downgrade(&model);
        let host = EngineHost::new(model, HostConfig::default());

        let a = host.attach().unwrap();
        let b = host.attach().unwrap();
        let c = host.attach().unwrap();
        assert!(weak.upgrade().is_some(), "model alive while attached");
        drop(a);
        drop(b);
        drop(c);
        // Sessions drop asynchronously on the GPU thread; dropping the host
        // joins that thread, guaranteeing all session Arc clones are gone.
        drop(host);
        assert!(
            weak.upgrade().is_none(),
            "model must be freed once the last session and the host are dropped"
        );
    }

    #[test]
    fn host_admission_cap() {
        let (host, _trace) = echo_host(2, 4, 4);
        let a = host.attach().unwrap();
        let _b = host.attach().unwrap();
        let err = host.attach().expect_err("third attach must be refused");
        assert!(
            !err.is_unsupported(),
            "admission failure is a real error, not an unsupported fallback"
        );
        // Freeing a slot lets a new attach succeed (no permanent leak).
        drop(a);
        // Give the GPU thread a moment to process the Detach before re-attaching.
        std::thread::sleep(Duration::from_millis(30));
        assert!(host.attach().is_ok(), "a freed slot is reusable");
    }

    #[test]
    fn scheduler_round_robin_fairness() {
        // Two long generations must interleave: once both are runnable, each
        // advances within one rotation — neither starves the other (design §5,
        // §10). Both are long and interrupt-stopped once interleaving is
        // observed, so the test does not depend on wall-clock timing.
        let (host, trace) = echo_host(4, 1_000_000, 2);
        // Attach order fixes FIFO ids: A = 1, B = 2.
        let a = host.attach().unwrap();
        let b = host.attach().unwrap();
        let a_flag = Arc::new(AtomicBool::new(false));
        let b_flag = Arc::new(AtomicBool::new(false));

        let poll = |pred: &dyn Fn(&[u64]) -> bool| {
            for _ in 0..500 {
                if pred(&trace.lock().unwrap()) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            false
        };

        let ta = {
            let f = Arc::clone(&a_flag);
            std::thread::spawn(move || {
                a.generate("[user]\nA\n", &GenerationOptions::default(), f, &mut |_| {})
                    .unwrap()
            })
        };
        // Wait until A is actually running before admitting B, so A is FIFO-first.
        assert!(poll(&|t| t.contains(&1)), "A must start running");

        let tb = {
            let f = Arc::clone(&b_flag);
            std::thread::spawn(move || {
                b.generate("[user]\nB\n", &GenerationOptions::default(), f, &mut |_| {})
                    .unwrap()
            })
        };
        // B must get a slice while A is still running (interleaving), and A must
        // then get another slice after B — i.e. neither is starved.
        assert!(poll(&|t| t.contains(&2)), "B must get a slice while A runs");
        assert!(
            poll(&|t| {
                let first_b = t.iter().position(|&id| id == 2);
                let last_a = t.iter().rposition(|&id| id == 1);
                matches!((first_b, last_a), (Some(fb), Some(la)) if fb < la)
            }),
            "A must get a slice after B (round-robin, no starvation)"
        );

        a_flag.store(true, Ordering::Relaxed);
        b_flag.store(true, Ordering::Relaxed);
        let sa = ta.join().unwrap();
        let sb = tb.join().unwrap();
        assert!(sa.interrupted && sb.interrupted);
    }

    #[test]
    fn scheduler_per_session_interrupt() {
        // Interrupting A leaves B streaming and completing normally; A returns
        // interrupted:true (design §5, §10).
        let (host, _trace) = echo_host(4, 100_000, 4);
        let a = host.attach().unwrap();
        let b = host.attach().unwrap();

        let a_flag = Arc::new(AtomicBool::new(false));
        let a_flag2 = Arc::clone(&a_flag);
        let ta = std::thread::spawn(move || {
            a.generate(
                "[user]\nA\n",
                &GenerationOptions::default(),
                a_flag2,
                &mut |_| {},
            )
            .unwrap()
        });
        let tb = std::thread::spawn(move || {
            b.generate(
                "[user]\nB\n",
                &GenerationOptions {
                    n_predict: 5,
                    ..GenerationOptions::default()
                },
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            )
            .unwrap()
        });

        // B is short; it completes on its own.
        let sb = tb.join().unwrap();
        assert!(!sb.interrupted, "B completes normally");
        assert_eq!(sb.generated, 5);

        // Now stop the long-running A.
        a_flag.store(true, Ordering::Relaxed);
        let sa = ta.join().unwrap();
        assert!(sa.interrupted, "A must report it was interrupted");
    }

    #[test]
    fn scheduler_sync_contract() {
        // generate() blocks and returns real stats even though the FFI runs on
        // the GPU thread; the calling thread never touches the engine (§5).
        let (host, _trace) = echo_host(4, 16, 4);
        let handle = host.attach().unwrap();
        let mut count = 0;
        let stats = handle
            .generate(
                "[user]\nhi\n",
                &GenerationOptions::default(),
                Arc::new(AtomicBool::new(false)),
                &mut |ev| {
                    if matches!(ev, EngineEvent::Text(_)) {
                        count += 1;
                    }
                },
            )
            .unwrap();
        assert_eq!(stats.generated, 16, "returns real GenerationStats");
        assert_eq!(count, 16, "all streamed events were delivered in order");
        assert!(!stats.interrupted);
    }

    // --- Accounting + idle reclamation (design §7, §9 step 5) ---------------

    /// Builds a host over the real `EchoSharedModel` (which tracks `ctx_tokens`
    /// and supports snapshot/restore), so accounting and reclamation are
    /// exercised end to end without a model.
    fn shared_host(cfg: HostConfig) -> EngineHost {
        EngineHost::new(Arc::new(EchoSharedModel::new(4096)), cfg)
    }

    fn run_gen(handle: &SessionHandle) {
        handle
            .generate(
                "[user]\nhi\n",
                &GenerationOptions::default(),
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            )
            .unwrap();
    }

    /// Polls `host.status()` until `pred` holds or the budget elapses.
    fn poll_status(host: &EngineHost, pred: impl Fn(&HostStatus) -> bool) -> HostStatus {
        for _ in 0..500 {
            let st = host.status();
            if pred(&st) {
                return st;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        host.status()
    }

    #[test]
    fn status_reports_live_count_and_accounting() {
        let host = shared_host(HostConfig {
            max_sessions: 3,
            slice_tokens: 8,
            idle_reclaim: None,
            ..HostConfig::default()
        });
        // Cap is visible immediately; no sessions yet.
        let st = host.status();
        assert_eq!(st.max_sessions, 3);
        assert_eq!(st.live_sessions, 0);

        let a = host.attach().unwrap();
        let b = host.attach().unwrap();
        let st = poll_status(&host, |s| s.live_sessions == 2);
        assert_eq!(st.live_sessions, 2, "two attached sessions are counted");
        assert_eq!(
            st.resident_ctx_tokens, 0,
            "no generation yet, no resident KV"
        );

        // A generation grows that session's resident KV accounting.
        run_gen(&a);
        let st = poll_status(&host, |s| s.resident_ctx_tokens > 0);
        assert!(
            st.resident_ctx_tokens > 0,
            "resident KV tracked after a turn"
        );
        assert_eq!(st.sessions.len(), 2);
        assert!(st.sessions.iter().any(|s| s.ctx_tokens > 0 && !s.reclaimed));

        // Detach frees a slot and drops the live count.
        drop(a);
        drop(b);
        let st = poll_status(&host, |s| s.live_sessions == 0);
        assert_eq!(st.live_sessions, 0, "detach drops the live-session count");
    }

    #[test]
    fn admission_cap_surfaced_in_status_and_error() {
        let host = shared_host(HostConfig {
            max_sessions: 1,
            slice_tokens: 8,
            idle_reclaim: None,
            ..HostConfig::default()
        });
        let _a = host.attach().unwrap();
        let st = poll_status(&host, |s| s.live_sessions == 1);
        assert_eq!(st.live_sessions, 1);
        assert_eq!(st.max_sessions, 1);
        let err = host.attach().expect_err("second attach exceeds the cap");
        assert!(!err.is_unsupported(), "admission failure is a real error");
        assert!(err.to_string().contains("capacity"));
    }

    #[test]
    fn attach_honors_requested_ctx_size_and_reports_it() {
        // The echo shared model has a 4096-token ctx. A smaller per-session
        // request is honored and surfaced in status; the default attach gets
        // the model's full ctx (design §7, v2).
        let host = shared_host(HostConfig {
            max_sessions: 4,
            slice_tokens: 8,
            ..HostConfig::default()
        });
        let _small = host.attach_sized(Some(512)).unwrap();
        let _big = host.attach().unwrap(); // default → model ctx
        let st = poll_status(&host, |s| s.sessions.len() == 2);
        let sizes: Vec<i32> = st.sessions.iter().map(|s| s.ctx_size).collect();
        assert!(sizes.contains(&512), "requested per-session ctx honored");
        assert!(sizes.contains(&4096), "default attach gets the model ctx");
        // An over-large request is clamped to the model max, never exceeds it.
        let _huge = host.attach_sized(Some(999_999)).unwrap();
        let st = poll_status(&host, |s| s.sessions.len() == 3);
        assert!(
            st.sessions.iter().all(|s| s.ctx_size <= 4096),
            "requested ctx is clamped to the model maximum"
        );
    }

    #[test]
    fn kv_budget_rejects_before_max_sessions() {
        // Budget fits exactly two 4096-token sessions; a third is rejected on
        // the KV-bytes budget even though max_sessions (5) is not reached. The
        // rejection is a real error, not an `unsupported` fallback, and no OOM.
        let per_session = est_kv_bytes(4096, DEFAULT_KV_BYTES_PER_TOKEN);
        let host = shared_host(HostConfig {
            max_sessions: 5,
            slice_tokens: 8,
            kv_budget_bytes: Some(per_session * 2),
            ..HostConfig::default()
        });
        let _a = host.attach().unwrap();
        let _b = host.attach().unwrap();
        let st = poll_status(&host, |s| s.live_sessions == 2);
        assert_eq!(st.kv_bytes, per_session * 2, "aggregate KV bytes tracked");
        assert_eq!(st.kv_budget_bytes, Some(per_session * 2));

        let err = host
            .attach()
            .expect_err("third attach exceeds the KV budget");
        assert!(!err.is_unsupported(), "budget failure is a real error");
        assert!(err.to_string().contains("KV budget"));
        assert!(
            st.live_sessions < st.max_sessions,
            "rejected under the session-count cap: it was the KV budget"
        );

        // Shrinking ctx_size lets more clients in (design §7, v2): a budget
        // with headroom for two full sessions plus one tiny one admits the
        // tiny session but would still reject a third full-ctx session.
        let tiny = est_kv_bytes(1, DEFAULT_KV_BYTES_PER_TOKEN);
        let host2 = shared_host(HostConfig {
            max_sessions: 5,
            slice_tokens: 8,
            kv_budget_bytes: Some(per_session * 2 + tiny),
            ..HostConfig::default()
        });
        let _x = host2.attach().unwrap();
        let _y = host2.attach().unwrap();
        assert!(
            host2.attach_sized(Some(1)).is_ok(),
            "a tiny-ctx session still fits the residual budget"
        );
    }

    #[test]
    fn idle_reclaim_snapshots_and_restores() {
        let host = shared_host(HostConfig {
            max_sessions: 2,
            slice_tokens: 8,
            idle_reclaim: Some(Duration::from_millis(120)),
            ..HostConfig::default()
        });
        let a = host.attach().unwrap();
        run_gen(&a);
        let before = poll_status(&host, |s| s.resident_ctx_tokens > 0);
        let tokens = before.resident_ctx_tokens;
        assert!(tokens > 0);

        // After the idle threshold, the scheduler reclaims the session: it is
        // still attached (counted) but marked reclaimed with no resident KV.
        let st = poll_status(&host, |s| s.sessions.iter().any(|x| x.reclaimed));
        assert_eq!(st.live_sessions, 1, "reclaimed session is still attached");
        assert!(
            st.sessions.iter().all(|x| x.reclaimed),
            "the idle session was reclaimed to disk"
        );
        assert_eq!(st.resident_ctx_tokens, 0, "reclaimed KV is not resident");

        // Next activity transparently restores the persisted KV; the carried
        // token count comes back and the session is resident again.
        run_gen(&a);
        let st = poll_status(&host, |s| s.sessions.iter().all(|x| !x.reclaimed));
        assert!(
            st.resident_ctx_tokens >= tokens,
            "restored session's KV accounting is preserved"
        );
    }

    #[test]
    fn idle_reclaim_disabled_is_noop() {
        let host = shared_host(HostConfig {
            max_sessions: 2,
            slice_tokens: 8,
            idle_reclaim: None,
            ..HostConfig::default()
        });
        let a = host.attach().unwrap();
        run_gen(&a);
        poll_status(&host, |s| s.resident_ctx_tokens > 0);
        // Well past any plausible threshold: with reclamation off nothing is
        // ever reclaimed.
        std::thread::sleep(Duration::from_millis(150));
        let st = host.status();
        assert!(
            st.sessions.iter().all(|x| !x.reclaimed),
            "reclamation disabled: no session is ever reclaimed"
        );
        assert!(
            st.resident_ctx_tokens > 0,
            "KV stays resident when disabled"
        );
    }
}
