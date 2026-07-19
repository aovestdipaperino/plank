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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use crate::engine::{
    EchoEngine, Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats,
};

/// Default round-robin slice granularity, in tokens (design §5, K≈8–16).
pub const DEFAULT_SLICE_TOKENS: usize = 12;

/// Default conservative admission cap on concurrently attached sessions.
pub const DEFAULT_MAX_SESSIONS: usize = 8;

/// Host tuning knobs.
#[derive(Debug, Clone, Copy)]
pub struct HostConfig {
    /// Maximum number of concurrently attached sessions (admission cap, §7).
    pub max_sessions: usize,
    /// Tokens generated per session per round-robin slice (§5).
    pub slice_tokens: usize,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            max_sessions: DEFAULT_MAX_SESSIONS,
            slice_tokens: DEFAULT_SLICE_TOKENS,
        }
    }
}

/// The shared, immutable model: weights, tokenizer, and the Metal command
/// queue. Held behind an `Arc`; its strong count *is* the refcount the issue
/// asks for — the last drop frees the engine and Metal context (§4).
///
/// [`spawn`](ModelHandle::spawn) creates a fresh session over the shared
/// weights, positioned after the warm system-prompt prefix (§6). It runs on
/// the GPU worker thread, so implementations may freely touch the engine.
pub trait ModelHandle: Send + Sync + 'static {
    /// Creates a fresh session over the shared weights.
    ///
    /// # Errors
    /// Returns [`EngineError`] if the session cannot be created.
    fn spawn(self: Arc<Self>) -> Result<Box<dyn HostSession>, EngineError>;

    /// Human-readable model name for status displays.
    fn model_name(&self) -> String {
        String::new()
    }

    /// Context window size in tokens.
    fn ctx_size(&self) -> i32;

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
        let worker = std::thread::Builder::new()
            .name("plank-gpu".to_string())
            .spawn(move || scheduler_loop(&worker_model, &cmd_rx, cfg))
            .expect("spawn GPU worker thread");
        Self {
            model,
            cmd_tx,
            worker: Some(worker),
        }
    }

    /// Attaches a new session, restoring the warm system-prompt prefix (§6).
    ///
    /// # Errors
    /// Returns a real (non-`unsupported`) [`EngineError`] when the admission
    /// cap is reached or the session cannot be created (design §7).
    pub fn attach(&self) -> Result<SessionHandle, EngineError> {
        let (reply, reply_rx) = channel();
        self.cmd_tx
            .send(Command::Attach { reply })
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

/// The single GPU worker thread: owns every session and interleaves their
/// generations round-robin at K-token granularity (design §5).
fn scheduler_loop(model: &Arc<dyn ModelHandle>, cmd_rx: &Receiver<Command>, cfg: HostConfig) {
    let mut sessions: HashMap<u64, Box<dyn HostSession>> = HashMap::new();
    let mut rotation: VecDeque<ActiveJob> = VecDeque::new();

    loop {
        // Drain pending commands. Block only when there is no runnable work,
        // so an idle host does not spin.
        if rotation.is_empty() {
            match cmd_rx.recv() {
                Ok(cmd) => {
                    if !apply_command(cmd, model, &mut sessions, &mut rotation, cfg.max_sessions) {
                        return;
                    }
                }
                Err(_) => return,
            }
        }
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !apply_command(cmd, model, &mut sessions, &mut rotation, cfg.max_sessions) {
                return;
            }
        }

        // One round-robin pass: give each active job a single K-token slice,
        // then yield to the next (fairness, no full starvation — design §5).
        for _ in 0..rotation.len() {
            let Some(job) = rotation.pop_front() else {
                break;
            };
            let Some(session) = sessions.get_mut(&job.id) else {
                // Session detached mid-flight; drop the job.
                continue;
            };
            let mut sink = |ev: EngineEvent| {
                let _ = job.out.send(Yield::Event(ev));
            };
            match session.advance(cfg.slice_tokens, &job.interrupt, &mut sink) {
                Ok(None) => rotation.push_back(job),
                Ok(Some(stats)) => {
                    let _ = job.out.send(Yield::Done(Ok(stats)));
                }
                Err(e) => {
                    let _ = job.out.send(Yield::Done(Err(e)));
                }
            }
        }
    }
}

/// Applies one command on the GPU thread. Returns `false` to stop the worker.
fn apply_command(
    cmd: Command,
    model: &Arc<dyn ModelHandle>,
    sessions: &mut HashMap<u64, Box<dyn HostSession>>,
    rotation: &mut VecDeque<ActiveJob>,
    max_sessions: usize,
) -> bool {
    match cmd {
        Command::Attach { reply } => {
            let result = if sessions.len() >= max_sessions {
                Err(EngineError::new(format!(
                    "engine host at capacity: {max_sessions} sessions already attached"
                )))
            } else {
                Arc::clone(model).spawn().map(|session| {
                    let id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
                    sessions.insert(id, session);
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
            if let Some(session) = sessions.get_mut(&id) {
                session.begin(transcript, *opts);
                rotation.push_back(ActiveJob { id, interrupt, out });
            } else {
                let _ = out.send(Yield::Done(Err(EngineError::new("no such session"))));
            }
        }
        Command::Detach { id } => {
            sessions.remove(&id);
            rotation.retain(|job| job.id != id);
        }
        Command::Shutdown => return false,
    }
    true
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
}

impl ModelHandle for EchoSharedModel {
    fn spawn(self: Arc<Self>) -> Result<Box<dyn HostSession>, EngineError> {
        Ok(Box::new(EchoSharedSession {
            engine: EchoEngine::new(self.ctx_size),
            pending: None,
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
        Ok(Some(stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

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
        fn spawn(self: Arc<Self>) -> Result<Box<dyn HostSession>, EngineError> {
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
}
