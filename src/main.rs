//! Plank agent binary: interactive REPL and one-shot headless mode.

use std::io::{IsTerminal, Write as _};
use std::process::ExitCode;

use plank::config::{AgentConfig, parse_options, usage};
use plank::engine::{EchoEngine, Engine};
use plank::status;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
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

/// Builds the inference engine: the real ds4 engine when `-m` is given and a
/// model backend is compiled in, otherwise the echo stub.
fn make_engine(cfg: &AgentConfig) -> Result<Box<dyn Engine>, String> {
    if let Some(model) = &cfg.model_path {
        #[cfg(ds4_engine)]
        {
            use plank::config::Backend;
            use plank::ds4engine::Ds4Engine;
            use plank::ffi::Ds4Backend;
            let backend = match cfg.backend {
                Some(Backend::Cuda) => Ds4Backend::Cuda,
                Some(Backend::Cpu) => Ds4Backend::Cpu,
                // Metal is the platform default where the engine is built.
                Some(Backend::Metal) | None => Ds4Backend::Metal,
            };
            eprintln!("plank: loading model {}...", model.display());
            let engine = Ds4Engine::open(
                model,
                backend,
                cfg.generation.ctx_size,
                cfg.n_threads,
                cfg.power_percent,
            )
            .map_err(|e| e.to_string())?;
            eprintln!("plank: model ready: {}", engine.model_name());
            return Ok(Box::new(engine));
        }
        #[cfg(not(ds4_engine))]
        {
            return Err(format!(
                "-m {} requires the ds4 engine, which is not built on this platform",
                model.display()
            ));
        }
    }
    Ok(Box::new(EchoEngine::new(cfg.generation.ctx_size)))
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
