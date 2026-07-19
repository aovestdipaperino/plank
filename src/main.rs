//! Plank agent binary: interactive REPL and one-shot headless mode.

use std::io::{IsTerminal, Write as _};
use std::process::ExitCode;

use plank::config::{AgentConfig, parse_options, usage};
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

    let cfg = match parse_options(&args) {
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
    if let Some(dir) = &cfg.chdir_path
        && let Err(e) = std::env::set_current_dir(dir)
    {
        eprintln!("plank: chdir {}: {e}", dir.display());
        return ExitCode::FAILURE;
    }

    // First launch after an upgrade: drop caches the new binary may no longer
    // understand (the version delta encodes what to remove — see upgrade.rs).
    if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        let plank_dir = std::path::PathBuf::from(home).join(".plank");
        let t = plank::upgrade::run_startup_maintenance(&plank_dir, env!("CARGO_PKG_VERSION"));
        if t >= plank::upgrade::Transition::Minor {
            eprintln!("plank: version change detected; cleared stale caches ({t:?} upgrade)");
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
    match run(engine, &cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("plank: {e}");
            ExitCode::FAILURE
        }
    }
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
    use plank::singleton::{LockProbe, probe_lock};

    let path = std::env::var_os("DS4_LOCK_FILE")
        .filter(|p| !p.is_empty())
        .map_or_else(|| std::path::PathBuf::from("/tmp/ds4.lock"), Into::into);
    if probe_lock(&path) == LockProbe::Contended {
        return Err(
            "another plank (ds4) instance is already running. Only one instance can load the \
             ~82 GB DeepSeek V4 Flash model at a time — close the other instance and try again."
                .to_string(),
        );
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

/// Builds the inference engine: the real ds4 engine on macOS (from `-m` or the
/// default `~/.plank/ds4flash.gguf`, downloading it if missing), else the stub.
fn make_engine(cfg: &AgentConfig) -> Result<Box<dyn Engine>, String> {
    // Remote engine (flavor a, issue #26) is available on every platform and
    // takes precedence over the local selectors when `--remote` is given.
    if let Some(url) = &cfg.remote_url {
        use plank::remote::ds4_client::RemoteDs4Engine;
        eprintln!("plank: connecting to remote engine {url}...");
        let engine = RemoteDs4Engine::connect(url, cfg.remote_token.clone())
            .map_err(|e| format!("remote connect: {e}"))?;
        eprintln!("plank: remote engine ready: {}", engine.model_name());
        return Ok(Box::new(engine));
    }
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
            &cfg.engine,
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
    let cfg = match parse_options(&passthrough) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("plank serve: {msg}");
            return ExitCode::from(2);
        }
    };
    plank::interrupt::install();
    let engine = match make_engine(&cfg) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("plank serve: {e}");
            return ExitCode::FAILURE;
        }
    };
    match plank::serve::run(engine, &ServeConfig { listen, token }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("plank serve: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(engine: Box<dyn Engine>, cfg: &AgentConfig) -> Result<(), String> {
    let color = std::io::stdout().is_terminal();
    if cfg.non_interactive {
        return plank::ui::run_non_interactive(engine, cfg);
    }
    // The full-screen TUI (a real terminal on both ends) draws its own header,
    // so the banner is only printed for the plain piped fallback.
    let tui = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if !tui {
        print!("{}", plank::logo::banner());
        print!("{}", status::welcome_banner(cfg.generation.ctx_size, color));
        std::io::stdout().flush().map_err(|e| e.to_string())?;
    }
    plank::ui::run_interactive(engine, cfg)
}
