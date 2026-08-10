// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Plank agent binary: entry point for all agent modes.
//!
//! This binary provides four entry points, selected by CLI arguments:
//!
//! 1. **Interactive agent** (default, TUI or plain stdout): the main agent loop
//!    that drives inference, tool dispatch, and user interaction. Uses Ratatui
//!    when both stdin and stdout are real terminals, otherwise a plain line REPL.
//! 2. **Non-interactive / headless** (`--non-interactive`): reads commands from
//!    stdin and prints structured output, for scripting and CI integration.
//! 3. **Remote server** (`plank serve`): hosts the engine over a WebSocket
//!    control interface. Supports single-tenant and shared-engine modes.
//! 4. **Remote client** (`plank remote <url>`): connects to a remote server,
//!    mirrors its output, and forwards typed lines as prompts.
//!
//! Engine selection is delegated to [`make_engine`] (local ds4 engine on macOS,
//! provider API engines, or the `EchoEngine` stub elsewhere) and [`make_host`] for
//! the shared-engine variant. Startup maintenance, RAM checks, and
//! instance-lock guarding are all handled here before handing control to the
//! UI layer; the remote-control server itself is started at runtime by `/rc`,
//! not here.

use std::io::{IsTerminal, Write as _};
use std::process::ExitCode;

use plank::config::{AgentConfig, usage};
#[cfg(not(ds4_engine))]
use plank::engine::EchoEngine;
use plank::engine::Engine;
use plank::status;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // `plank serve ...` runs the flavor-(a) host instead of the interactive
    // agent (issue #26). It reuses `make_engine`, so it hosts the real ds4
    // engine on a Metal box and the EchoEngine stub elsewhere.
    if args.first().map(String::as_str) == Some("serve") {
        return run_serve(&args[1..]);
    }

    // `plank remote <url>` runs the interactive remote-control client (issue
    // #25): it connects to another instance's `/rc` WebSocket, mirrors its
    // output, and drives it. It never loads an engine of its own.
    if args.first().map(String::as_str) == Some("remote") {
        return run_remote_client(&args[1..]);
    }

    // Settings seed the config; every CLI flag then overrides them. Read from
    // the launch directory: `--chdir` is parsed here and so cannot yet be
    // known, which is a documented limitation of project-scoped settings.
    let settings = plank::settings::Settings::load();
    plank::settings::install(settings.clone());
    let cfg = match plank::config::parse_options_with(&settings, &args) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("plank: {msg}");
            return ExitCode::from(2);
        }
    };
    if cfg.show_help {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }
    // Settings can move plank off Metal or shrink the context, and both are
    // invisible once the UI is up — you just notice it got slow. Say so.
    if let Some(note) = plank::settings::startup_note(&settings, &cfg) {
        eprintln!("{note}");
    }
    // One-shot wipe of pre-`.kv_raw` KV blobs, before any terminal setup so
    // the note prints as a plain line on every front end (TUI, plain REPL,
    // and `--non-interactive` all funnel through here). Best-effort: a store
    // that fails to open is skipped silently, and the next launch retries.
    //
    // Gated on the cache directory already existing, and deliberately not
    // moved below `run_startup_maintenance`: `SessionStore::open` does a
    // `create_dir_all`, which creates `~/.plank` as a side effect, and a
    // `~/.plank` that exists with no version marker is what
    // `upgrade::classify` reads as a major version change — so a brand-new
    // install announced "cleared the image cache" on its very first launch.
    // The migration has to stay above every terminal setup (running it inside
    // the live alternate screen garbled the warm-progress frame), so the gate
    // is the fix rather than a reorder. Nothing to migrate exists before the
    // directory does, so skipping is exact rather than merely cheap.
    let kv_dir = plank::session::SessionStore::default_dir();
    if kv_dir.is_dir()
        && let Ok(store) = plank::session::SessionStore::open(&kv_dir)
        && let Some(bytes) = store.migrate_legacy_blobs()
        && bytes > 0
    {
        #[allow(clippy::cast_precision_loss)] // GB display only; loses no meaningful precision
        let gb = bytes as f64 / 1_073_741_824.0;
        eprintln!("kvcache: migrated to the .kv_raw format, reclaimed {gb:.1} GB");
    }
    if let Some(dir) = &cfg.chdir_path
        && let Err(e) = std::env::set_current_dir(dir)
    {
        eprintln!("plank: chdir {}: {e}", dir.display());
        return ExitCode::FAILURE;
    }

    // `--worktree` runs before anything reads the working directory, because
    // the whole session — its hooks, agent definitions, and every tool's cwd —
    // is meant to live inside the worktree rather than the original checkout.
    if let Err(e) = enter_startup_worktree(&cfg, &settings) {
        eprintln!("plank: {e}");
        return ExitCode::FAILURE;
    }

    // First launch after an upgrade: the KV caches self-validate and survive,
    // but a major version change drops the image cache (see upgrade.rs).
    if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        let plank_dir = std::path::PathBuf::from(home).join(".plank");
        let t = plank::upgrade::run_startup_maintenance(&plank_dir, env!("CARGO_PKG_VERSION"));
        if t == plank::upgrade::Transition::Major {
            eprintln!("plank: major version change detected; cleared the image cache");
        }
        // Best-effort, rate-limited (once/day), offline-safe check for a newer
        // release. Never blocks startup; the notice is stashed for whichever
        // front-end comes up to render non-intrusively (issue #56).
        if settings.update.check {
            let notice = plank::upgrade::check_for_update(&plank_dir, env!("CARGO_PKG_VERSION"));
            plank::upgrade::set_update_notice(notice);
        }
    }

    plank::interrupt::install();
    let engine = match make_engine(&cfg) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("plank: {e}");
            return ExitCode::FAILURE;
        }
    };
    match run(engine.main, engine.local, &cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("plank: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Honors `--worktree` / `--worktree-pr`: creates (or resumes) the named
/// worktree and moves the process into it, parking the session for the agent.
///
/// Also sweeps worktrees abandoned by an earlier plank that was killed before
/// it could clean up — done here, once, rather than on a timer, because the
/// sweep shells out to git for every candidate and nothing about it is urgent.
///
/// # Errors
/// Returns a message when the worktree cannot be created or entered. A failure
/// here is fatal: silently continuing in the original checkout would be exactly
/// the collision `--worktree` was asked to prevent.
fn enter_startup_worktree(
    cfg: &AgentConfig,
    settings: &plank::settings::Settings,
) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    if let Some(root) = plank::worktree::canonical_git_root(&cwd) {
        let removed =
            plank::worktree::cleanup_stale_worktrees(&root, plank::worktree::STALE_AFTER, None);
        if removed > 0 {
            eprintln!("plank: removed {removed} abandoned sub-agent worktree(s)");
        }
    }
    // `--worktree-pr` on its own is meaningful: name the worktree after the PR.
    let name = match (&cfg.worktree, cfg.worktree_pr) {
        (Some(name), _) => name.clone(),
        (None, Some(pr)) => format!("pr-{pr}"),
        (None, None) => return Ok(()),
    };
    let opts = plank::worktree::CreateOptions {
        pr_number: cfg.worktree_pr,
        sparse_paths: settings.worktree.sparse_paths.clone(),
        symlink_dirs: settings.worktree.symlink_directories.clone(),
    };
    let hooks = plank::hooks::load_default(&cwd);
    let session = plank::worktree::create_for_session(&cwd, &name, &hooks, &opts)
        .map_err(|e| format!("worktree '{name}': {e}"))?;
    std::env::set_current_dir(&session.path)
        .map_err(|e| format!("chdir {}: {e}", session.path.display()))?;
    eprintln!(
        "plank: working in worktree '{name}' at {}",
        session.path.display()
    );
    plank::worktree::set_startup_session(session);
    Ok(())
}

/// Minimum physical RAM plank requires to run the model, in bytes (96 GiB).
#[cfg(ds4_engine)]
const MIN_RAM_BYTES: u64 = 96 * 1024 * 1024 * 1024;

/// Total physical RAM in bytes, via `sysctl hw.memsize`.
#[cfg(ds4_engine)]
fn total_ram_bytes() -> Option<u64> {
    let mut mem: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: hw.memsize returns a u64; `mem`/`len` are valid out-params and
    // the name is a NUL-terminated C string.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut mem).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then_some(mem)
}

/// Fails fast when another plank/ds4 instance is already running, with a clear
/// message — instead of the engine's own guard, which calls `exit(2)` deep in
/// `ds4_engine_open` (`ds4_acquire_instance_lock` in `ds4.c`) and kills the
/// process before plank can report anything useful.
///
/// It probes the *same* lock file the engine uses (`$DS4_LOCK_FILE`, default
/// `/tmp/ds4.lock`) but only **checks** it — the lock is released immediately
/// so the engine can acquire it itself moments later. (Holding it here would
/// make the engine's own in-process acquire fail, since `flock` is keyed to
/// the open file description, not the process.)
///
/// Any inability to probe (unwritable path, non-contention error) is treated
/// as "no guard" — the engine's own check still backstops it.
///
/// # Errors
/// Returns a clear message when another instance already holds the lock.
#[cfg(ds4_engine)]
fn acquire_model_lock() -> Result<(), String> {
    use plank::singleton::{LockProbe, lock_holder_pid, probe_lock};

    let path = std::env::var_os("DS4_LOCK_FILE")
        .filter(|p| !p.is_empty())
        .map_or_else(|| std::path::PathBuf::from("/tmp/ds4.lock"), Into::into);
    if probe_lock(&path) == LockProbe::Contended {
        let who = lock_holder_pid(&path).map_or_else(String::new, |pid| format!(" (PID {pid})"));
        return Err(format!(
            "another plank (ds4) instance is already running{who}. Only one instance can load the \
             ~82 GB DeepSeek V4 Flash model at a time — close the other instance and try again."
        ));
    }
    Ok(())
}

/// Refuses to run when the machine has less than [`MIN_RAM_BYTES`] of RAM.
///
/// # Errors
/// Returns an explanatory message when physical RAM is below the minimum.
#[cfg(ds4_engine)]
fn require_min_ram() -> Result<(), String> {
    if let Some(bytes) = total_ram_bytes()
        && bytes < MIN_RAM_BYTES
    {
        #[allow(clippy::cast_precision_loss)]
        let have = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        return Err(format!(
            "plank needs at least 96 GB of RAM to run DeepSeek V4 Flash; this machine has {have:.0} GB"
        ));
    }
    Ok(())
}

/// Builds the inference engine: the real ds4 engine on macOS (from `-m`, else
/// `engine.model` in settings.json, else the default `~/.plank/ds4flash.gguf`,
/// downloading it if missing), else the stub.
/// The engines a session runs on: the main one, and — only when the main engine
/// is a provider *and* a `provider: local` sub-agent definition exists — the
/// local ds4 engine held for those sidechains.
struct Engines {
    main: Box<dyn Engine>,
    /// `None` unless a definition explicitly asked for the local engine while
    /// the main agent is remote. Loading it costs the full ~82 GB residency, so
    /// it is never speculative.
    local: Option<Box<dyn Engine>>,
}

/// Whether any sub-agent definition visible from `cwd` asks for the local
/// engine explicitly (`provider: local`). Omitting `provider:` does *not* count:
/// that means "the parent's engine", whatever it happens to be.
fn wants_local_subagent() -> bool {
    let Ok(cwd) = std::env::current_dir() else {
        return false;
    };
    plank::agents::load_default(&cwd)
        .iter()
        .any(|d| matches!(d.engine, Some(plank::agents::AgentEngine::Local)))
}

fn make_engine(cfg: &AgentConfig) -> Result<Engines, String> {
    // Remote engine (flavor a, issue #26) is available on every platform and
    // takes precedence over the local selectors when `--remote` is given.
    if let Some(url) = &cfg.remote_url {
        use plank::remote::ds4_client::RemoteDs4Engine;
        eprintln!("plank: connecting to remote engine {url}...");
        plank::status::set_engine_origin(&plank::status::url_host(url));
        let engine = RemoteDs4Engine::connect(url, cfg.remote_token.clone())
            .map_err(|e| format!("remote connect: {e}"))?;
        eprintln!("plank: remote engine ready: {}", engine.model_name());
        return Ok(Engines {
            main: Box::new(engine),
            local: None,
        });
    }
    // Provider engine (flavor b, issue #26): third-party LLM APIs behind the
    // Engine trait, available on every platform (pure Rust + HTTP).
    if let Some(provider) = cfg.provider {
        use plank::config::ProviderSelector;
        use plank::remote::provider::{ProviderEngine, ProviderKind};
        let kind = match provider {
            ProviderSelector::OpenAi => ProviderKind::OpenAi,
            ProviderSelector::Anthropic => ProviderKind::Anthropic,
        };
        let model = cfg
            .provider_model
            .clone()
            .ok_or_else(|| "--provider requires --model NAME".to_string())?;
        let api_key = cfg.provider_api_key.clone().unwrap_or_default();
        eprintln!("plank: using provider {} model {model}...", kind.label());
        plank::status::set_engine_origin(&plank::status::url_host(
            cfg.provider_base_url
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| kind.default_base_url()),
        ));
        // With no explicit `-c`, the configured window is the local-model
        // default and says nothing about this provider's model — ask the
        // provider. Best-effort: on any failure the configured value stands.
        let ctx_size = if cfg.ctx_size_explicit {
            cfg.generation.ctx_size
        } else {
            ProviderEngine::discover_ctx_size(
                kind,
                cfg.provider_base_url.as_deref(),
                &api_key,
                &model,
            )
            .unwrap_or(cfg.generation.ctx_size)
        };
        let engine = ProviderEngine::new(
            kind,
            cfg.provider_base_url.clone(),
            api_key,
            model,
            ctx_size,
            cfg.provider_cache,
        )
        .map_err(|e| format!("provider init: {e}"))?;
        eprintln!(
            "plank: provider engine ready: {} (ctx {ctx_size})",
            engine.model_name()
        );
        // A `provider: local` definition means the ds4 engine specifically, so
        // load it alongside the provider — otherwise such a sidechain would have
        // nothing to run on. Costly and deliberate: only an explicit definition
        // triggers it, and it fails here rather than mid-turn.
        let local = if wants_local_subagent() {
            eprintln!(
                "plank: a sub-agent definition asks for the local engine; loading it alongside the provider..."
            );
            // Both engines are live for the whole session, so the footer names
            // both: the provider above, this one here.
            plank::status::set_engine_origin(plank::status::LOCAL_ORIGIN);
            Some(make_local_engine(cfg)?)
        } else {
            None
        };
        return Ok(Engines {
            main: Box::new(engine),
            local,
        });
    }
    Ok(Engines {
        main: make_local_engine(cfg)?,
        local: None,
    })
}

/// Loads the local ds4 engine. Used both as the main engine and — when a
/// `provider: local` sub-agent definition needs one under a provider main agent
/// — as the spare handed to the `Agent` (see [`make_engine`]).
///
/// # Errors
/// Returns a message when RAM is insufficient, another instance holds the model,
/// the model file is absent and cannot be fetched, or the engine fails to open.
fn make_local_engine(cfg: &AgentConfig) -> Result<Box<dyn Engine>, String> {
    #[cfg(ds4_engine)]
    {
        use plank::config::Backend;
        use plank::ds4engine::Ds4Engine;
        use plank::ffi::Ds4Backend;

        // The default quant needs ~82 GB resident; refuse on machines that
        // cannot hold it, before downloading or loading anything.
        require_min_ram()?;

        // Only one instance can hold the ~82 GB model at a time — a second
        // would fail deep in the engine while mapping model views, with a
        // cryptic "insufficient memory / accelerator VM budget" abort. Fail
        // fast here with a clear message instead.
        acquire_model_lock()?;

        // With no explicit model, fall back to the default location and offer
        // to download it when it is not present.
        let model = cfg
            .model_path
            .clone()
            .unwrap_or_else(plank::download::default_model_path);
        plank::download::ensure_model(&model)?;
        // `--dspark` without `--mtp` resolves to the default support model,
        // fetched on demand. Kept local rather than written back into `cfg`:
        // only the engine open needs it.
        let mut tuning = cfg.engine.clone();
        plank::download::ensure_dspark_support(&mut tuning)?;

        let backend = match cfg.backend {
            Some(Backend::Cuda) => Ds4Backend::Cuda,
            Some(Backend::Cpu) => Ds4Backend::Cpu,
            // Metal is the platform default where the engine is built.
            Some(Backend::Metal) | None => Ds4Backend::Metal,
        };
        eprintln!("plank: loading model {}...", model.display());
        // Render the C engine's noisy startup log in place on one row.
        let replacer = plank::stderrline::StderrLineReplacer::start();
        let engine = Ds4Engine::open(
            &model,
            backend,
            cfg.generation.ctx_size,
            cfg.n_threads,
            cfg.power_percent,
            &tuning,
        )
        .map_err(|e| e.to_string())?;
        drop(replacer);
        eprintln!("plank: model ready: {}", engine.model_name());
        Ok(Box::new(engine))
    }
    #[cfg(not(ds4_engine))]
    {
        if let Some(model) = &cfg.model_path {
            return Err(format!(
                "-m {} requires the ds4 engine, which is not built on this platform",
                model.display()
            ));
        }
        Ok(Box::new(EchoEngine::new(cfg.generation.ctx_size)))
    }
}

/// Parses `plank remote <url> [--token <t>] [--resume-from <id>]` and runs the
/// interactive remote-control client. The token falls back to
/// `PLANK_REMOTE_TOKEN`; the URL is `ws://host:port/` (tunnel to loopback for a
/// remote box, matching the SSH hint the server prints).
fn run_remote_client(args: &[String]) -> ExitCode {
    let mut url: Option<String> = None;
    let mut token: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--token" if i + 1 < args.len() => {
                token = Some(args[i + 1].clone());
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: plank remote <ws-url> [--token <token>]\n\
                     \n\
                     Connects to a plank instance with remote control on (/rc), mirrors its\n\
                     output, and sends typed lines as prompts (slash lines as commands,\n\
                     \"/btw <q>\" as a side question) and Ctrl-C as an interrupt.\n\
                     The token defaults to $PLANK_REMOTE_TOKEN."
                );
                return ExitCode::SUCCESS;
            }
            other if url.is_none() && !other.starts_with('-') => {
                url = Some(other.to_string());
                i += 1;
            }
            other => {
                eprintln!("plank remote: unexpected argument: {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(url) = url else {
        eprintln!("usage: plank remote <ws-url> [--token <token>]");
        return ExitCode::from(2);
    };
    match plank::remote::client::run(&url, token) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("plank remote: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Parses `plank serve` arguments and runs the host. Model/backend flags are
/// forwarded to `make_engine` via a normal [`AgentConfig`]; `--listen`/`--token`
/// are serve-specific.
fn run_serve(args: &[String]) -> ExitCode {
    use plank::serve::ServeConfig;

    let mut listen = "127.0.0.1:8080".to_string();
    let mut token = std::env::var("PLANK_REMOTE_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let mut passthrough: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--listen" | "-l" if i + 1 < args.len() => {
                listen.clone_from(&args[i + 1]);
                i += 2;
            }
            "--token" if i + 1 < args.len() => {
                token = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                passthrough.push(other.to_string());
                i += 1;
            }
        }
    }
    let settings = plank::settings::Settings::load();
    plank::settings::install(settings.clone());
    let cfg = match plank::config::parse_options_with(&settings, &passthrough) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("plank serve: {msg}");
            return ExitCode::from(2);
        }
    };
    plank::interrupt::install();

    // Shared-engine mode (issue #28): host one model for many concurrent
    // per-session_id clients. Off by default; the local single-tenant path is
    // byte-for-byte unchanged. Not combined with --remote (that is a client).
    if cfg.shared_engine && cfg.remote_url.is_none() {
        let host = match make_host(&cfg) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("plank serve: {e}");
                return ExitCode::FAILURE;
            }
        };
        return match plank::serve::run_shared(host, &ServeConfig { listen, token }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("plank serve: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let engine = match make_engine(&cfg) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("plank serve: {e}");
            return ExitCode::FAILURE;
        }
    };
    match plank::serve::run(engine.main, &ServeConfig { listen, token }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("plank serve: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Builds the shared [`EngineHost`](plank::host::EngineHost) for
/// `plank serve --shared-engine`: the real ds4 model warmed once on macOS, or
/// the echo stub elsewhere. The host owns the single GPU thread and hands out a
/// session per network client (design §4, §8).
fn make_host(cfg: &AgentConfig) -> Result<plank::host::EngineHost, String> {
    use plank::host::{DEFAULT_SLICE_TOKENS, EngineHost, HostConfig};

    let host_cfg = HostConfig {
        max_sessions: usize::try_from(cfg.max_sessions.max(1)).unwrap_or(1),
        slice_tokens: DEFAULT_SLICE_TOKENS,
        idle_reclaim: (cfg.idle_reclaim_secs > 0)
            .then(|| std::time::Duration::from_secs(cfg.idle_reclaim_secs)),
        // v2 (design §7): default per-session ctx (0 = model max) and an
        // aggregate KV-bytes budget (0 = count-only admission). Both default to
        // today's behavior so existing runs are unchanged.
        session_ctx_size: (cfg.session_ctx_size > 0).then_some(cfg.session_ctx_size),
        kv_budget_bytes: (cfg.kv_budget_bytes > 0).then_some(cfg.kv_budget_bytes),
    };
    #[cfg(ds4_engine)]
    {
        use plank::config::Backend;
        use plank::ds4engine::Ds4Model;
        use plank::ffi::Ds4Backend;

        require_min_ram()?;
        acquire_model_lock()?;
        let model_path = cfg
            .model_path
            .clone()
            .unwrap_or_else(plank::download::default_model_path);
        plank::download::ensure_model(&model_path)?;
        // See the local-engine path: resolved into a local copy, not `cfg`.
        let mut tuning = cfg.engine.clone();
        plank::download::ensure_dspark_support(&mut tuning)?;
        let backend = match cfg.backend {
            Some(Backend::Cuda) => Ds4Backend::Cuda,
            Some(Backend::Cpu) => Ds4Backend::Cpu,
            Some(Backend::Metal) | None => Ds4Backend::Metal,
        };
        eprintln!("plank: loading shared model {}...", model_path.display());
        let replacer = plank::stderrline::StderrLineReplacer::start();
        let model = Ds4Model::open_shared(
            &model_path,
            backend,
            cfg.generation.ctx_size,
            cfg.n_threads,
            cfg.power_percent,
            &tuning,
            &cfg.system,
        )
        .map_err(|e| e.to_string())?;
        drop(replacer);
        eprintln!("plank: shared model ready: {}", model.model_name());
        Ok(EngineHost::new(model, host_cfg))
    }
    #[cfg(not(ds4_engine))]
    {
        use std::sync::Arc;
        if let Some(model) = &cfg.model_path {
            return Err(format!(
                "-m {} requires the ds4 engine, which is not built on this platform",
                model.display()
            ));
        }
        let model = Arc::new(plank::host::EchoSharedModel::new(cfg.generation.ctx_size));
        Ok(EngineHost::new(model, host_cfg))
    }
}

fn run(
    engine: Box<dyn Engine>,
    local_engine: Option<Box<dyn Engine>>,
    cfg: &AgentConfig,
) -> Result<(), String> {
    let color = std::io::stdout().is_terminal();
    if cfg.non_interactive {
        return plank::ui::run_non_interactive(engine, cfg, local_engine);
    }
    plank::title::set(plank::title::State::Loading);
    // The full-screen TUI (a real terminal on both ends) draws its own header,
    // so the banner is only printed for the plain piped fallback.
    let tui = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if !tui {
        print!("{}", plank::logo::banner());
        print!("{}", status::welcome_banner(cfg.generation.ctx_size, color));
        // Non-intrusive one-time update hint (issue #56); silent when up to date.
        if let Some(notice) = plank::upgrade::update_notice() {
            println!("{notice}\n");
        }
        // Only the echo stub has no model name; the TUI prints the same lines
        // into its own scrollback.
        if engine.model_name().is_empty() {
            for line in status::no_model_lines() {
                println!("{line}");
            }
            println!();
        }
        std::io::stdout().flush().map_err(|e| e.to_string())?;
    }
    plank::ui::run_interactive(engine, cfg, local_engine)
}
