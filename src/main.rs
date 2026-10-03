// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Plank agent binary: entry point for all agent modes.
//!
//! This binary provides four entry points, selected by CLI arguments:
//!
//! 1. **Interactive agent** (default, TUI or plain stdout): the main agent loop
//!    that drives inference, tool dispatch, and user interaction. Uses Ratatui
//!    when both stdin and stdout are real terminals, otherwise a plain line REPL.
//! 2. **Non-interactive / headless** (`--ui console`, `--ui chart`): reads commands from
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

/// Arms the panic repro dump for the rest of the run.
///
/// It dumps only while the debug mirror is on, so `/debug on` mid-session arms
/// it and an ordinary run pays nothing for having the hook installed.
fn arm_panic_dump() {
    plank::repro::install_panic_hook(plank::repro::repro_dir(
        &std::env::current_dir().unwrap_or_default(),
    ));
}

/// The user's home directory, if `HOME` is set.
fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Prints every warning in `plugins` to stderr, one per line.
///
/// Factored out so both startup call sites can drain warnings *after*
/// resolving `--profile` (a spliced profile can push its own) without each
/// repeating the loop inline and blowing the 100-line function cap.
fn print_plugin_warnings(plugins: &plank::plugins::PluginSet) {
    for w in plugins.all_warnings() {
        eprintln!("plugin warning: {w}");
    }
}

/// Offers to update the installed profile `name` when `source` now declares a
/// higher `version`. Reads only the source's manifest, so a profile that is
/// current costs no download; a check that fails launches what is installed.
fn update_if_newer(source: &str, name: &str, dir: &std::path::Path, home: &std::path::Path) {
    let installed = plank::profiles::version_of(dir);
    let available = plank::claudeplugin::source_version(source);
    let plank::profiles::Freshness::Newer {
        installed,
        available,
    } = plank::profiles::freshness(installed.as_deref(), available.as_deref())
    else {
        return;
    };
    if !can_ask() {
        eprintln!(
            "plank: profile {name} {available} is available (installed: {installed}); launch interactively to update"
        );
        return;
    }
    if !ask_yes(&format!(
        "Update profile {name} {installed} -> {available}? This replaces the installed copy, including changes made with /edit-profile."
    )) {
        return;
    }
    match plank::claudeplugin::replace_profile(source, name, home) {
        Ok(_) => eprintln!("plank: updated profile {name} to {available}"),
        Err(e) => eprintln!("plank: cannot update profile {name}: {e}; launching {installed}"),
    }
}

/// Whether there is a person at a terminal to answer a startup question.
fn can_ask() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// Asks `question` on stderr and reads one line; an empty answer is yes.
fn ask_yes(question: &str) -> bool {
    eprint!("{question} [Y/n] ");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).is_ok()
        && matches!(
            answer.trim().to_ascii_lowercase().as_str(),
            "" | "y" | "yes"
        )
}

/// Turns a `--profile` source (a local directory, `owner/repo:folder` or
/// `owner/repo`) into the name of its installed copy.
///
/// An install that recorded this same source is used as is, with no fetch and
/// no question, which is what lets the same command line launch it every time
/// (and `/edit-profile`'s restart reuse it). Otherwise the user is asked
/// before anything is fetched; with no terminal to ask on, nothing is
/// installed and the error names the command that would do it.
fn resolve_profile_source(
    source: &str,
    home: Option<&std::path::Path>,
) -> Result<String, ExitCode> {
    let Some(home) = home else {
        eprintln!("plank: no HOME, so there is nowhere to install the profile {source:?}");
        return Err(ExitCode::from(2));
    };
    let normalized = plank::profiles::normalize_source(source);
    if let Some((name, dir)) = plank::profiles::find_by_source(home, &normalized) {
        update_if_newer(source, &name, &dir, home);
        return Ok(name);
    }
    if !can_ask() {
        eprintln!(
            "plank: the profile {source:?} is not installed; install it with /install-profile {source}, or launch interactively to be asked"
        );
        return Err(ExitCode::from(2));
    }
    if !ask_yes(&format!(
        "Install the profile {source} into {}?",
        plank::profiles::dir(home).display()
    )) {
        eprintln!("plank: not installed");
        return Err(ExitCode::SUCCESS);
    }
    match plank::claudeplugin::install_profile(source, None, home, false) {
        Ok(installed) => {
            eprintln!(
                "plank: installed profile {} at {}",
                installed.name,
                installed.dest.display()
            );
            Ok(installed.name)
        }
        Err(e) => {
            eprintln!("plank: cannot install the profile {source:?}: {e}");
            Err(ExitCode::from(2))
        }
    }
}

/// Resolves `--profile`'s argument against the loaded plugins and installs
/// it, or reports what to do instead.
///
/// Returns `Some(code)` when startup must stop here (a bare-flag listing, or
/// a fatal resolution error); `None` means resolution succeeded (including
/// "no `--profile` given") and startup should continue.
fn resolve_and_activate_profile(
    requested: Option<&str>,
    explicit_empty: bool,
    plugins: &mut plank::plugins::PluginSet,
    home: Option<&std::path::Path>,
) -> Option<ExitCode> {
    // A path or `owner/repo:folder` becomes the name of the installed copy,
    // installing it first when it is not installed yet.
    let resolved = match requested.filter(|r| plank::profiles::is_source(r)) {
        Some(source) => match resolve_profile_source(source, home) {
            Ok(name) => Some(name),
            Err(code) => return Some(code),
        },
        None => requested.map(str::to_owned),
    };
    let requested = resolved.as_deref();
    // An installed profile is loaded only now, because `--profile` named it:
    // `plugins::load_in` never scans the profiles root, so nothing there has
    // contributed anything to this session yet.
    if let (Some(name), Some(home)) = (requested.filter(|n| !n.is_empty()), home)
        && !plugins.plugins.iter().any(|p| p.name == name)
        && let Some(dir) = plank::profiles::find(home, name)
        && plank::plugins::splice_profile(plugins, &dir).is_none()
    {
        // `profiles::find` already parsed the manifest, so this means the
        // plugin's name (directory-name fallback included) failed the same
        // `valid_name` gate `load_in` applies to every scanned plugin; the
        // reason is in the warning `splice_profile` just pushed onto
        // `plugins`. Reporting it as a distinct failure instead of falling
        // through to `NoSuchPlugin` matters because `name` genuinely does
        // appear in the profile listing.
        eprintln!(
            "plank: profile {name:?} at {}: could not be loaded",
            dir.display()
        );
        return Some(ExitCode::from(2));
    }
    let installed = home.map(plank::profiles::names).unwrap_or_default();
    match plank::profile::resolve_profile(requested, explicit_empty, plugins) {
        plank::profile::Resolution::None => None,
        plank::profile::Resolution::EmptyName => {
            eprintln!("plank: --profile requires a non-empty name");
            Some(ExitCode::from(2))
        }
        plank::profile::Resolution::Activate(active) => {
            let prompt = match std::fs::read_to_string(&active.spec.system_prompt) {
                Ok(text) => text,
                Err(e) => {
                    eprintln!(
                        "plank: profile {}: cannot read {}: {e}",
                        active.name,
                        active.spec.system_prompt.display()
                    );
                    return Some(ExitCode::from(2));
                }
            };
            // An empty or whitespace-only prompt file is the same failure as
            // an unreadable one: the docs call an unreadable prompt fatal,
            // and a profile whose prompt is nothing but the generated schema
            // block is not a valid identity either (finding 7).
            if prompt.trim().is_empty() {
                eprintln!(
                    "plank: profile {}: {} is empty",
                    active.name,
                    active.spec.system_prompt.display()
                );
                return Some(ExitCode::from(2));
            }
            if let Some(w) = plank::profile::missing_protocol_warning(&active.name, &prompt) {
                eprintln!("{w}");
            }
            // The one read of the prompt file for the whole run: stored on
            // the `ActiveProfile` so composition (`sysprompt.rs`) is
            // infallible and never re-reads the file mid-session.
            plank::profile::install(plank::profile::ActiveProfile { prompt, ..*active });
            None
        }
        plank::profile::Resolution::List(names) => {
            let names = plank::plugins::merge_profile_names(names, &installed);
            if names.is_empty() {
                println!("no profiles installed");
            } else {
                for n in names {
                    println!("{n}");
                }
            }
            Some(ExitCode::SUCCESS)
        }
        plank::profile::Resolution::NoSuchPlugin(name, names) => {
            let available = plank::plugins::merge_profile_names(names, &installed);
            eprintln!("plank: no plugin named {name:?}");
            if available.is_empty() {
                eprintln!("plank: no profiles are installed");
            } else {
                eprintln!("plank: available profiles: {}", available.join(", "));
            }
            Some(ExitCode::from(2))
        }
        plank::profile::Resolution::NotAProfile(name) => {
            eprintln!("plank: plugin {name:?} declares no profile block");
            Some(ExitCode::from(2))
        }
    }
}

/// Loads settings with the plugin and (if one is active) profile layers, as
/// `defaults < plugins < profile < ~/.plank < ./.plank`.
///
/// Both interactive-startup call sites need this after loading plugins and
/// resolving the profile; factored out so line count doesn't force one of
/// them to duplicate the other's logic instead.
fn load_settings_with_profile(
    plugins: &plank::plugins::PluginSet,
    cwd: &std::path::Path,
) -> plank::settings::Settings {
    let profile_settings = plank::profile::active().and_then(|a| {
        a.spec
            .settings_json
            .as_deref()
            .map(|t| (a.name.as_str(), t))
    });
    let home = home_dir();
    plank::settings::Settings::load_with_plugins_and_profile_in(
        home.as_deref(),
        cwd,
        &plank::plugins::settings_paths(plugins),
        profile_settings,
    )
}

/// Reports the result of the KV cache migration performed by
/// `SessionStore::migrate_kvcache_if_present`. The associated function performs
/// the one-shot wipe of pre-`.kv_raw` KV blobs; this function is best-effort
/// reporting only: a store that fails to open is skipped silently, and the next
/// launch retries.
fn report_kvcache_migration() {
    let kv_dir = plank::session::SessionStore::default_dir();
    if let Some(bytes) = plank::session::SessionStore::migrate_kvcache_if_present(&kv_dir)
        && bytes > 0
    {
        #[allow(clippy::cast_precision_loss)] // GB display only; loses no meaningful precision
        let gb = bytes as f64 / 1_073_741_824.0;
        eprintln!("kvcache: migrated to the .kv_raw format, reclaimed {gb:.1} GB");
    }
}

/// The model path that will actually load, from the real `cfg` (parsed
/// against the fully-loaded settings, not the CLI-only provisional parse).
/// Shared by every consumer that needs to know which model family is loading,
/// so none of them can resolve it a different way and disagree.
///
/// `parse_config` has already resolved the catalog choice into a path. A path
/// that does not exist yet — a first run, before the download — is probed by
/// name, so it still tags the family of the model that is about to land.
fn resolve_model_path(cfg: &plank::config::AgentConfig) -> std::path::PathBuf {
    cfg.model_path
        .clone()
        .expect("parse_config resolves the model before anything reads it")
}

/// The user's model choice, as the catalog resolver takes it.
fn model_choice(cfg: &plank::config::AgentConfig) -> plank::engines::Choice<'_> {
    match (cfg.model_spec.as_deref(), cfg.model_named) {
        (None, _) => plank::engines::Choice::Default,
        (Some(s), true) => plank::engines::Choice::Named(s),
        (Some(s), false) => plank::engines::Choice::Spec(s),
    }
}

/// Whether the model choice came from `--model`, `-m` or `--model:` rather
/// than from `engine.model`: only the command line outranks a profile's
/// `recommendedModel`.
fn model_from_cli(cfg: &plank::config::AgentConfig) -> bool {
    cfg.model_spec.is_some()
        && cfg.cli_provenance.get("engine.model") == Some(&plank::provenance::Origin::Cli)
}

/// The running profile's `recommendedModel`, when `--profile` activated one
/// that declares it. Without `--profile` there is none, whatever the plugin
/// declares.
///
/// Named by `displayName` when the profile sets one, else the plugin name —
/// the same fallback `profile::display_name` uses for the banner — since
/// that is what the user sees, and the "using"/"not installed"/"not an
/// engine" notes must all agree with it.
fn active_recommendation() -> Option<plank::engines::Recommendation<'static>> {
    let active = plank::profile::active()?;
    Some(plank::engines::Recommendation {
        profile: plank::profile::display_name(),
        engine: active.spec.recommended_model.as_deref()?,
    })
}

/// Resolves the model choice against the catalog, letting the running
/// profile's `recommended` engine outrank `engine.model` (but never the
/// command line) when it is already on disk. Must precede
/// `resolve_model_delta`, which reads the resolved `model_path`.
///
/// Does not call `plank::engines::set_active`: the selection's `main` path
/// still names the pre-delta model here, and a `.ggd` spec gets rewritten to
/// the patched clone afterward. `parse_config` activates the final,
/// delta-adjusted selection once both steps have run.
///
/// The catalog and managed paths come from `root`, which is `~/.plank` in
/// every real run and a scratch directory in tests. The loaded catalog is
/// returned so [`finish_selection`] can reuse it rather than load it twice.
fn resolve_selection(
    cfg: &mut plank::config::AgentConfig,
    root: &std::path::Path,
    recommended: Option<plank::engines::Recommendation<'_>>,
) -> Result<plank::engines::Catalog, String> {
    let mut warn = Vec::new();
    let catalog = plank::engines::load_in(root, &mut warn);
    for w in warn {
        eprintln!("plank: {w}");
    }
    let choice = model_choice(cfg);
    let from_cli = model_from_cli(cfg);
    // A recommendation is for the local model a run would load, so a remote
    // or provider run, which loads none, never announces one.
    let local = cfg.remote_url.is_none() && cfg.provider.is_none();
    let (sel, notes) = plank::engines::choose_with_recommendation_in(
        root,
        &catalog,
        from_cli.then_some(choice),
        recommended.filter(|_| local),
        if from_cli {
            plank::engines::Choice::Default
        } else {
            choice
        },
    )?;
    let color = std::io::stderr().is_terminal();
    for note in notes {
        if note.warning && color {
            eprintln!("\x1b[33mplank: {}\x1b[0m", note.text);
        } else {
            eprintln!("plank: {}", note.text);
        }
    }
    // The recommendation, when it won, replaces `engine.model` as the thing
    // actually loading: keep `model_spec` in sync so anything that reports
    // the model choice (the startup note, the no-engine-build error) names
    // the engine that is really running rather than the settings value it
    // overrode.
    if let Some(rec) = recommended.filter(|_| local && !from_cli)
        && sel.id.is_some_and(|id| id.as_str() == rec.engine)
    {
        cfg.model_spec = Some(rec.engine.to_string());
    }
    cfg.model_path = Some(sel.main.clone());
    cfg.selection = Some(sel);
    Ok(catalog)
}

/// Points the selection at the final `model_path`, and gives a `.ggd`
/// delta patched onto a managed engine's `main` that engine's companions
/// (`engines::inherit_companions_in`), so an abliterated V4 delta still gets
/// the vision encoder and the `DSpark` drafter. The clone itself stays
/// unmanaged: it is never upgraded or re-downloaded.
///
/// Runs after `resolve_model_delta` and before `set_active`, keeping the
/// invariant `selection.main == model_path`.
fn finish_selection(
    cfg: &mut plank::config::AgentConfig,
    root: &std::path::Path,
    catalog: &plank::engines::Catalog,
) {
    let Some(path) = cfg.model_path.clone() else {
        return;
    };
    let Some(mut sel) = cfg.selection.take() else {
        return;
    };
    sel.main = path;
    if let Some(delta) = &cfg.model_delta {
        sel = plank::engines::inherit_companions_in(root, catalog, &delta.base, sel);
    }
    cfg.selection = Some(sel);
}

/// Whether a `resolve_selection`/`resolve_model_delta` error must abort
/// startup. `--dump-config` is a diagnostic: `Settings::from_settings`
/// promises "a settings file must never stop plank from starting", so a bad
/// `engine.model` must still let the dump print (with no selection) rather
/// than exit before printing anything. Every other run keeps failing fast.
fn resolution_is_fatal(cfg: &plank::config::AgentConfig) -> bool {
    !cfg.dump_config
}

/// The real config parse, with the model choice resolved against the engine
/// catalog and a `.ggd` model swapped for its patched clone. Errors are
/// already printed under `prog`; the caller just returns the code.
///
/// Invariant on return: `cfg.selection.as_ref().map(|s| &s.main) ==
/// cfg.model_path.as_ref()` whenever a selection is present, and
/// `plank::engines::ACTIVE` holds that same, final selection.
fn parse_config(
    settings: &plank::settings::Settings,
    args: &[String],
    prog: &str,
) -> Result<plank::config::AgentConfig, ExitCode> {
    parse_config_in(
        settings,
        args,
        prog,
        &plank::manifest::plank_dir(),
        active_recommendation(),
    )
}

/// [`parse_config`] with the engine catalog and managed paths under `root`,
/// and the profile's `recommended` engine passed in rather than read from
/// the active profile.
fn parse_config_in(
    settings: &plank::settings::Settings,
    args: &[String],
    prog: &str,
    root: &std::path::Path,
    recommended: Option<plank::engines::Recommendation<'_>>,
) -> Result<plank::config::AgentConfig, ExitCode> {
    plank::config::parse_options_with(settings, args)
        .and_then(|mut cfg| {
            cfg.drop_default_system_under_profile(plank::profile::active().is_some());
            match resolve_selection(&mut cfg, root, recommended)
                .and_then(|catalog| resolve_model_delta(&mut cfg).map(|()| catalog))
            {
                Ok(catalog) => finish_selection(&mut cfg, root, &catalog),
                Err(e) => {
                    if resolution_is_fatal(&cfg) {
                        return Err(e);
                    }
                    eprintln!("plank: {e}");
                    cfg.selection = None;
                }
            }
            if let Some(sel) = cfg.selection.clone() {
                plank::engines::set_active(sel);
            }
            Ok(cfg)
        })
        .map_err(|msg| {
            eprintln!("{prog}: {msg}");
            ExitCode::from(2)
        })
}

/// Swaps a `.ggd` weight delta given as the model for the patched clone it
/// resolves to, so everything downstream — family probe, manifest check,
/// companion lookup, the engine open — sees an ordinary GGUF.
///
/// Runs right after the config is parsed and before anything reads a model
/// header. The clone is materialized on first use and reused afterwards
/// (`ggufdelta::resolve`). Prints one line naming the delta and its base.
fn resolve_model_delta(cfg: &mut plank::config::AgentConfig) -> Result<(), String> {
    let Some(delta) = cfg
        .model_path
        .as_deref()
        .filter(|p| plank::ggufdelta::is_delta_path(p))
        .map(std::path::Path::to_path_buf)
    else {
        return Ok(());
    };
    let resolved = plank::ggufdelta::resolve(&delta)?;
    eprintln!(
        "plank: {} ({}): {}",
        delta.display(),
        resolved.describe(),
        resolved.path.display()
    );
    cfg.model_path = Some(resolved.path.clone());
    cfg.model_delta = Some(resolved);
    Ok(())
}

/// Records the live model family for the session store.
///
/// Must run before anything opens the store, which both startup paths do
/// within a few lines: the family decides the transcript extension, and a
/// store opened before it would name files for the wrong one.
fn select_session_family(cfg: &plank::config::AgentConfig) {
    plank::session::set_family(plank::gguf::family_of(&resolve_model_path(cfg)));
}

/// The detached downloader's entry point.
///
/// Its engine is the second argument. A helper spawned by a plank that
/// predates engines passes `ds4` or none, which reads as `ds4vision` — the
/// engine plank managed when there was only one.
fn run_model_downloader(args: &[String]) -> i32 {
    let Some(id) =
        plank::manifest::EngineId::from_legacy_arg(args.get(1).map_or("", String::as_str))
    else {
        eprintln!("plank: --model-downloader: invalid engine name");
        return 2;
    };
    plank::downloader::run_helper(id)
}

/// The checks that fire once the real `cfg` (settings + CLI, not the
/// CLI-only `provisional` parse) is known, before anything user-visible:
/// the `--help`/`--version`/`--dump-config` backstops that the provisional
/// parse already answers in practice but must be re-decided here against the
/// layered settings.
///
/// Factored out of `main` (and reused by nothing else — `run_serve` has no
/// help/version/dump-config surface) purely to keep `main` under the
/// 100-line cap; it owns no state of its own.
fn post_cfg_early_exit(
    cfg: &plank::config::AgentConfig,
    settings: &plank::settings::Settings,
) -> Option<ExitCode> {
    if cfg.show_help {
        print!("{}", usage());
        return Some(ExitCode::SUCCESS);
    }
    if cfg.show_version {
        println!("{}", plank::logo::version_line());
        return Some(ExitCode::SUCCESS);
    }
    // `--dump-config` prints the resolved configuration (every effective key
    // with the layer it came from) and exits, without starting a session. It
    // works under `--ui console` because it needs no UI.
    if cfg.dump_config {
        print!("{}", plank::provenance::render_resolved(settings, cfg));
        return Some(ExitCode::SUCCESS);
    }
    // The same contract for the other two configurations plank resolves at
    // launch but never otherwise shows: read-only, no session, no UI.
    if cfg.dump_profiles {
        // `profiles::dir` roots at `$HOME` and appends `.plank/profiles`,
        // where `engines::load_in` takes `~/.plank` itself. Handing either one
        // the other's root silently reads an empty directory.
        match home_dir() {
            Some(home) => print!(
                "{}",
                plank::dump::render_profiles_in(&home, std::io::stdout().is_terminal())
            ),
            None => eprintln!("plank: no home directory; cannot list profiles"),
        }
        return Some(ExitCode::SUCCESS);
    }
    if cfg.dump_engines {
        print!(
            "{}",
            plank::dump::render_engines_in(&plank::manifest::plank_dir())
        );
        return Some(ExitCode::SUCCESS);
    }
    None
}

/// Whether this launch should run the one-way engine-layout migration.
///
/// `--help` and `--version` answer and exit without touching `~/.plank`, and
/// `--dump-config` is a read-only diagnostic, so none of them may rename the
/// user's model files. Every other launch migrates before the real
/// `parse_config`, because the `engines.local.json` default a ds41-only
/// install gets must exist before the catalog choice is resolved.
fn should_migrate(provisional: &plank::config::AgentConfig) -> bool {
    !(provisional.show_help
        || provisional.show_version
        || provisional.dump_config
        || provisional.dump_profiles
        || provisional.dump_engines)
}

/// Renames any old `ModelSet` layout into the engine layout, when
/// [`should_migrate`] allows it, printing one line per skipped move.
fn migrate_engine_layout(provisional: &plank::config::AgentConfig) {
    if !should_migrate(provisional) {
        return;
    }
    migrate_engine_layout_unconditionally();
}

/// Renames any old `ModelSet` layout into the engine layout, unconditionally.
///
/// `plank serve` has no `--help`/`--version`/`--dump-config` early exits of
/// its own — it always goes on to `make_engine` — so it must always migrate
/// first, unlike `main`'s gated [`migrate_engine_layout`]. Skipping this
/// would let `plank serve --help` on an old home reach `make_engine` without
/// ever having migrated, and offer an 87 GB download.
fn migrate_engine_layout_unconditionally() {
    for w in plank::enginemigrate::migrate() {
        eprintln!("{w}");
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Before anything can move the process: `/edit-profile`'s restart re-runs
    // these arguments from this directory.
    plank::profileedit::record_launch(std::env::current_dir().unwrap_or_default(), args.clone());

    // `plank --model-downloader` is the detached background model downloader
    // (`src/downloader.rs`), re-execing this same binary so the helper can
    // never disagree with the plank that spawned it. It loads no engine, reads
    // no settings and touches no session: it takes the download lock, works
    // through its `~/.plank/downloads/job-<engine>.json`, and exits. Handled before every
    // other dispatch so nothing above can print to a stream that is /dev/null.
    if args.first().map(String::as_str) == Some("--model-downloader") {
        return ExitCode::from(u8::try_from(run_model_downloader(&args)).unwrap_or(1));
    }

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

    // The plugin set has to be built before settings are read, so that a
    // plugin's `settings.json` can be layered in below the user file — but
    // `--chdir` and `--profile`, which decide what it scans and splices, are
    // only known once parsed. A throwaway provisional parse (base settings, no
    // plugin layer) breaks that cycle: only its `profile`, `chdir_path` and `debug` are used (all pure
    // CLI flags, unaffected by settings layering, so they agree with the real
    // parse below), and the real parse re-derives everything else with the
    // enriched settings.
    let provisional =
        plank::config::parse_options_with(&plank::settings::Settings::default(), &args)
            .unwrap_or_else(|_| {
                plank::config::AgentConfig::from_settings(&plank::settings::Settings::default())
            });
    // `--help` is answered from the provisional parse, before `--chdir` and
    // before the plugin scan. Both of those can fail or print warnings, and
    // `plank --chdir /nonexistent --help` printing a chdir error instead of the
    // usage text — or prefixing the usage with plugin warnings — is a
    // regression against every other CLI. `show_help` is a pure flag, so the
    // provisional parse resolves it identically to the real one; a provisional
    // parse that failed leaves it false and falls through to the real parse,
    // which reports the argument error.
    if provisional.show_help {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }
    // Same reasoning as `--help`: answer from the provisional parse, before
    // `--chdir` and the plugin scan can print anything ahead of the one line
    // a caller scraping `plank --version` expects.
    if provisional.show_version {
        println!("{}", plank::logo::version_line());
        return ExitCode::SUCCESS;
    }
    migrate_engine_layout(&provisional);
    // `--chdir` has to happen before the plugin scan (and therefore before
    // project settings, which are also cwd-scoped) rather than after, or the
    // plugin set built here would reflect the launch directory instead of the
    // directory the session actually runs in — the eager local-engine preload
    // in `wants_local_subagent` and the worktree's `hooks.json` discovery both
    // consult this same set, so a mismatch here means a plugin declared under
    // `<chdir>/.plank/plugins` silently never loads.
    if let Some(dir) = &provisional.chdir_path
        && let Err(e) = std::env::set_current_dir(dir)
    {
        eprintln!("plank: chdir {}: {e}", dir.display());
        return ExitCode::FAILURE;
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    // This set is deliberately the PRE-worktree one: it is scanned against the
    // post-`--chdir` cwd but before `enter_startup_worktree` moves us again.
    // That ordering is forced, not accidental — `enter_startup_worktree` needs
    // `hooks_with_plugins(&cwd, plugins)` to decide whether to create the
    // worktree at all, so the set has to exist first. Rebuilding it after the
    // worktree move would mean a second full plugin scan (and a second round of
    // warnings) purely to observe a directory that is a checkout of the same
    // repo, so the pre-worktree set is reused for the rest of startup.
    let mut plugins = plank::plugins::load_default(&cwd);
    // The profile is resolved from the provisional parse because everything
    // downstream — the settings layer, the system prompt, the tool table —
    // needs it, and the real parse at `parse_options_with` happens after the
    // settings it would feed. Warnings drain *after*, not before: a spliced
    // profile can push its own, and draining first would lose them silently.
    let code = resolve_and_activate_profile(
        provisional.profile.as_deref(),
        provisional.profile_explicit_empty,
        &mut plugins,
        home_dir().as_deref(),
    );
    print_plugin_warnings(&plugins);
    if let Some(code) = code {
        return code;
    }
    let settings = load_settings_with_profile(&plugins, &cwd);
    // `--debug` is a pure CLI flag, so the provisional parse agrees with the
    // real one; it must be set before `install`, whose reconcile is the first
    // chance to dial the console.
    plank::debugmirror::set_enabled(provisional.debug);
    plank::settings::install(settings.clone());
    let cfg = match parse_config(&settings, &args, "plank") {
        Ok(cfg) => cfg,
        Err(code) => return code,
    };
    if let Some(code) = post_cfg_early_exit(&cfg, &settings) {
        return code;
    }
    // Settings can move plank off Metal or shrink the context, and both are
    // invisible once the UI is up — you just notice it got slow. Say so.
    if let Some(note) = plank::settings::startup_note(&settings, &cfg) {
        eprintln!("{note}");
    }
    // One-shot wipe of pre-`.kv_raw` KV blobs, before any terminal setup so
    // the note prints as a plain line on every front end (TUI, plain REPL,
    // and `--ui console` all funnel through here). Best-effort: a store
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
    select_session_family(&cfg);
    report_kvcache_migration();
    // `--worktree` runs before anything reads the working directory, because
    // the whole session — its hooks, agent definitions, and every tool's cwd —
    // is meant to live inside the worktree rather than the original checkout.
    if let Err(e) = enter_startup_worktree(&cfg, &settings, &plugins) {
        eprintln!("plank: {e}");
        return ExitCode::FAILURE;
    }

    // First launch after an upgrade: the KV caches self-validate and survive,
    // but a major version change drops the image cache (see upgrade.rs).
    if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        let plank_dir = plank::home::plank_home_in(home);
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
    plank::gpuyield::sweep_stray_signals();
    arm_panic_dump();
    let engine = match make_engine(&cfg, &plugins) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("plank: {e}");
            return ExitCode::FAILURE;
        }
    };
    match run(engine.main, engine.local, engine.reopen, &cfg, plugins) {
        Ok(code) => ExitCode::from(code),
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
    plugins: &plank::plugins::PluginSet,
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
    let hooks = plank::plugins::hooks_with_plugins(&cwd, plugins);
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
    if let Some(bytes) = plank::download::total_ram_bytes()
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

/// Builds the inference engine: the real ds4 engine on macOS (the engine or
/// path chosen by `-m`, else `engine.model` in settings.json, else the catalog
/// `default` engine, downloading a managed engine's files if missing), else
/// the stub.
/// The engines a session runs on: the main one, and — only when the main engine
/// is a provider *and* a `provider: local` sub-agent definition exists — the
/// local ds4 engine held for those sidechains.
struct Engines {
    main: Box<dyn Engine>,
    /// `None` unless a definition explicitly asked for the local engine while
    /// the main agent is remote. Loading it costs the full ~82 GB residency, so
    /// it is never speculative.
    local: Option<Box<dyn Engine>>,
    /// Reopens whichever of the two is the local ds4 engine, with the exact
    /// parameters it was opened with, for the GPU-yield cycle (`gpuyield`).
    /// `None` when no local model is loaded.
    reopen: Option<plank::gpuyield::ReopenFn>,
}

/// Whether any sub-agent definition visible from `cwd` asks for the local
/// engine explicitly (`provider: local`). Omitting `provider:` does *not* count:
/// that means "the parent's engine", whatever it happens to be.
fn wants_local_subagent(plugins: &plank::plugins::PluginSet) -> bool {
    let Ok(cwd) = std::env::current_dir() else {
        return false;
    };
    plank::plugins::agents_with_plugins(&cwd, plugins)
        .0
        .iter()
        .any(|d| matches!(d.engine, Some(plank::agents::AgentEngine::Local)))
}

fn make_engine(cfg: &AgentConfig, plugins: &plank::plugins::PluginSet) -> Result<Engines, String> {
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
            reopen: None,
        });
    }
    // Provider engine (flavor b, issue #26): third-party LLM APIs behind the
    // Engine trait, available on every platform (pure Rust + HTTP).
    if let Some(provider) = cfg.provider {
        use plank::config::ProviderSelector;
        use plank::remote::provider::{ProviderEngine, ProviderKind};
        let kind = match provider {
            ProviderSelector::OpenAi => ProviderKind::OpenAi,
            ProviderSelector::OpenAiResponses => ProviderKind::OpenAiResponses,
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
        let local = if wants_local_subagent(plugins) {
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
        let (local, reopen) = local.map_or((None, None), |(e, r)| (Some(e), r));
        return Ok(Engines {
            main: Box::new(engine),
            local,
            reopen,
        });
    }
    let (main, reopen) = make_local_engine(cfg)?;
    Ok(Engines {
        main,
        local: None,
        reopen,
    })
}

/// Loads the local ds4 engine. Used both as the main engine and — when a
/// `provider: local` sub-agent definition needs one under a provider main agent
/// — as the spare handed to the `Agent` (see [`make_engine`]).
///
/// Also returns the factory that reopens the same model with the same resolved
/// parameters (the tuning that actually opened, after any companion retry),
/// which the GPU-yield cycle calls after it has dropped the engine. The echo
/// stub has nothing to reopen.
///
/// # Errors
/// Returns a message when RAM is insufficient, another instance holds the model,
/// the model file is absent and cannot be fetched, or the engine fails to open.
/// The `--fake-gpu` stand-in engine.
///
/// Built before any of the real engine's gates, and deliberately so: the mode
/// exists to run *beside* an instance that already holds the model, so it skips
/// the RAM floor, the single-instance lock and the model download alike.
///
/// It reports the engine name the catalog resolved, not a placeholder, because
/// `kvtier::system_fingerprint` hashes the model name: keying anywhere else
/// would put the fake run's checkpoints in a corner of the cache no real run
/// ever looks at, and the reproduction would be of nothing.
///
/// There is no [`plank::gpuyield::ReopenFn`]: nothing was loaded, so nothing
/// can be reopened, and arming the GPU-yield cycle around an absent model
/// would be theatre.
fn make_fake_engine(cfg: &AgentConfig) -> Box<dyn Engine> {
    plank::fakegpu::set_active(true);
    let model = cfg
        .selection
        .as_ref()
        .and_then(|s| s.id)
        .map_or_else(|| "fake-gpu".to_string(), |id| id.as_str().to_string());
    eprintln!("plank: --fake-gpu, no model will be loaded; KV checkpoints key on {model:?}");
    Box::new(plank::fakegpu::FakeGpuEngine::new(
        &model,
        cfg.generation.ctx_size,
    ))
}

fn make_local_engine(
    cfg: &AgentConfig,
) -> Result<(Box<dyn Engine>, Option<plank::gpuyield::ReopenFn>), String> {
    if cfg.fake_gpu {
        return Ok((make_fake_engine(cfg), None));
    }
    #[cfg(ds4_engine)]
    {
        use plank::{config::Backend, ds4engine::Ds4Engine, ffi::Ds4Backend};

        // The default quant needs ~82 GB resident; refuse on machines that
        // cannot hold it, before downloading or loading anything.
        require_min_ram()?;

        // Only one instance can hold the ~82 GB model at a time — a second
        // would fail deep in the engine while mapping model views, with a
        // cryptic "insufficient memory / accelerator VM budget" abort. Fail
        // fast here with a clear message instead.
        acquire_model_lock()?;

        // `parse_config` resolved the catalog choice; offer to download the
        // model when it is not present.
        let model = cfg.model_path.clone().expect("resolved by parse_config");
        let sel = cfg.selection.as_ref().expect("resolved by parse_config");
        // Install anything a previous run downloaded and verified, then decide
        // whether to start a new background download. Must precede
        // `ensure_model`, so a staged upgrade is in place before the engine
        // maps the file. Never fatal.
        plank::download::check_manifest_at_startup(sel);
        // Mirrors the background downloader's state into the status bar. Cheap
        // and idempotent: it does nothing at all when no download is running.
        plank::downloader::spawn_watcher();
        plank::download::ensure_model(sel)?;
        // The vision encoder sits beside the main model and is fetched on
        // demand when the model can use it (the pinned Vision-Exp checkpoint);
        // any other DeepSeek checkpoint runs text-only.
        // Speculation is on by default; without `--mtp-model` a DeepSeek run
        // takes its companion from the selected engine's `mtp` role and
        // fetches it on demand (`--mtp-off` skips that). A run with no such
        // companion — a bare `--model PATH` or a `.ggd` on no managed base,
        // or an engine that declares none — has speculation turned off
        // instead of failing to open. Kept local rather than written back
        // into `cfg`: only the engine open needs it. A Qwen model skips both side artifacts, since it opens
        // neither.
        let mut tuning = cfg.engine.clone();
        plank::download::ensure_side_artifacts(sel, cfg.generation.ctx_size, &mut tuning)?;

        let backend = match cfg.backend {
            Some(Backend::Cuda) => Ds4Backend::Cuda,
            Some(Backend::Cpu) => Ds4Backend::Cpu,
            // Metal is the platform default where the engine is built.
            Some(Backend::Metal) | None => Ds4Backend::Metal,
        };
        eprintln!("plank: loading model {}...", model.display());
        // Render the C engine's noisy startup log in place on one row.
        let replacer = plank::stderrline::StderrLineReplacer::start();
        let opened = (|| {
            let first = match Ds4Engine::open(
                &model,
                backend,
                cfg.generation.ctx_size,
                cfg.n_threads,
                cfg.power_percent,
                &tuning,
            ) {
                Ok(engine) => return Ok((engine, tuning.clone())),
                Err(e) => e.to_string(),
            };
            // The C refuses to open a checkpoint at all when the DSpark draft
            // model does not match it. When plank picked that companion itself,
            // retry once (never in a loop) rather than making the user discover
            // `--mtp-off`; a companion the user named is never dropped (see
            // `EngineTuning::without_auto_companion`).
            let Some(solo) = tuning.without_auto_companion() else {
                return Err(first);
            };
            eprintln!(
                "note: speculative decoding disabled (the DSpark draft model is not compatible with this checkpoint)"
            );
            Ds4Engine::open(
                &model,
                backend,
                cfg.generation.ctx_size,
                cfg.n_threads,
                cfg.power_percent,
                &solo,
            )
            .map(|engine| (engine, solo))
            // Report the original failure, with the retry's as context: the
            // retry only rules the companion out, it does not diagnose a
            // corrupt model.
            .map_err(|second| {
                format!(
                    "{first}\n(retried without the DSpark draft model, which also failed: {second})"
                )
            })
        })();
        drop(replacer);
        let (engine, opened_with) = opened?;
        eprintln!(
            "plank: model ready: {}{}",
            engine.model_name(),
            cfg.model_delta
                .as_ref()
                .map_or_else(String::new, |d| format!(" ({})", d.describe()))
        );
        let reopen_path = model;
        let ctx_size = cfg.generation.ctx_size;
        let (n_threads, power) = (cfg.n_threads, cfg.power_percent);
        let reopen: plank::gpuyield::ReopenFn = Box::new(move || {
            // Everything the C would `exit` on, checked here first so it is an
            // error instead: a file gone or unreadable since startup is an
            // `exit(1)` inside `model_open`, and a contended lock inside
            // `ds4_engine_open` is an `exit(2)` (another plank may have
            // started while this one had the model unloaded).
            let mut files = vec![("model", reopen_path.as_path())];
            if let Some(p) = &opened_with.mtp_path {
                files.push(("DSpark draft model", p.as_path()));
            }
            if let Some(p) = &opened_with.vision_path {
                files.push(("vision encoder", p.as_path()));
            }
            plank::gpuyield::check_model_files(files)?;
            acquire_model_lock()?;
            Ds4Engine::open(
                &reopen_path,
                backend,
                ctx_size,
                n_threads,
                power,
                &opened_with,
            )
            .map(|engine| Box::new(engine) as Box<dyn Engine>)
            .map_err(|e| e.to_string())
        });
        Ok((Box::new(engine), Some(reopen)))
    }
    #[cfg(not(ds4_engine))]
    {
        if let Some(model) = &cfg.model_spec {
            return Err(format!(
                "-m {model} requires the ds4 engine, which is not built on this platform"
            ));
        }
        Ok((Box::new(EchoEngine::new(cfg.generation.ctx_size)), None))
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
/// /`--insecure` are serve-specific (`--insecure` waives the refusal to serve a
/// non-loopback address without a token).
fn run_serve(args: &[String]) -> ExitCode {
    use plank::serve::ServeConfig;

    let mut listen = "127.0.0.1:8080".to_string();
    let mut token = std::env::var("PLANK_REMOTE_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let mut insecure = false;
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
            "--insecure" => {
                insecure = true;
                i += 1;
            }
            other => {
                passthrough.push(other.to_string());
                i += 1;
            }
        }
    }
    // A token-less bind off loopback exposes the model to the network; refuse
    // unless the operator opted in explicitly.
    if let Err(msg) = plank::serve::check_exposure(&listen, token.is_some(), insecure) {
        eprintln!("plank serve: {msg}");
        eprintln!(
            "usage: plank serve [--listen ADDR] [--token TOKEN] [--insecure] [engine options]"
        );
        return ExitCode::from(2);
    }
    // See the `main` provisional-parse comment: the plugin set and the
    // profile must be settled before settings are read, but `--profile` is
    // only known once parsed, so a throwaway base-settings parse breaks the
    // cycle.
    let provisional =
        plank::config::parse_options_with(&plank::settings::Settings::default(), &passthrough)
            .unwrap_or_else(|_| {
                plank::config::AgentConfig::from_settings(&plank::settings::Settings::default())
            });
    migrate_engine_layout_unconditionally();
    let launch_cwd = std::env::current_dir().unwrap_or_default();
    let mut plugins = plank::plugins::load_default(&launch_cwd);
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    // The profile is resolved from the provisional parse because everything
    // downstream — the settings layer, the system prompt, the tool table —
    // needs it, and the real parse at `parse_options_with` happens after the
    // settings it would feed. Warnings are drained *after* this call; see the
    // matching comment in `main`.
    let code = resolve_and_activate_profile(
        provisional.profile.as_deref(),
        provisional.profile_explicit_empty,
        &mut plugins,
        home.as_deref(),
    );
    print_plugin_warnings(&plugins);
    if let Some(code) = code {
        return code;
    }
    let settings = load_settings_with_profile(&plugins, &launch_cwd);
    // `--debug` is a pure CLI flag, so the provisional parse agrees with the
    // real one; it must be set before `install`, whose reconcile is the first
    // chance to dial the console.
    plank::debugmirror::set_enabled(provisional.debug);
    plank::settings::install(settings.clone());
    let cfg = match parse_config(&settings, &passthrough, "plank serve") {
        Ok(cfg) => cfg,
        Err(code) => return code,
    };
    plank::interrupt::install();
    plank::gpuyield::sweep_stray_signals();

    select_session_family(&cfg);

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

    let engine = match make_engine(&cfg, &plugins) {
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
        let model_path = cfg.model_path.clone().expect("resolved by parse_config");
        let sel = cfg.selection.as_ref().expect("resolved by parse_config");
        // Install anything a previous run downloaded and verified, then decide
        // whether to start a new background download. Must precede
        // `ensure_model`, so a staged upgrade is in place before the engine
        // maps the file. Never fatal.
        plank::download::check_manifest_at_startup(sel);
        // Mirrors the background downloader's state into the status bar. Cheap
        // and idempotent: it does nothing at all when no download is running.
        plank::downloader::spawn_watcher();
        plank::download::ensure_model(sel)?;
        // The vision encoder sits beside the main model and is fetched on
        // demand when the model can use it (the pinned Vision-Exp checkpoint);
        // any other DeepSeek checkpoint runs text-only.
        // See the local-engine path: resolved into a local copy, not `cfg`.
        let mut tuning = cfg.engine.clone();
        plank::download::ensure_side_artifacts(sel, cfg.generation.ctx_size, &mut tuning)?;
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
        eprintln!(
            "plank: shared model ready: {}{}",
            model.model_name(),
            cfg.model_delta
                .as_ref()
                .map_or_else(String::new, |d| format!(" ({})", d.describe()))
        );
        Ok(EngineHost::new(model, host_cfg))
    }
    #[cfg(not(ds4_engine))]
    {
        use std::sync::Arc;
        if let Some(model) = &cfg.model_spec {
            return Err(format!(
                "-m {model} requires the ds4 engine, which is not built on this platform"
            ));
        }
        let model = Arc::new(plank::host::EchoSharedModel::new(cfg.generation.ctx_size));
        Ok(EngineHost::new(model, host_cfg))
    }
}

fn run(
    engine: Box<dyn Engine>,
    local_engine: Option<Box<dyn Engine>>,
    reopen: Option<plank::gpuyield::ReopenFn>,
    cfg: &AgentConfig,
    plugins: plank::plugins::PluginSet,
) -> Result<u8, String> {
    // The family check the C makes right after opening the engine: a numeric
    // effort is meaningless to anything but V4.1, and falling back to `high`
    // silently would be worse than refusing.
    if plank::engine::think_level_unsupported(cfg.generation.think_mode, &engine.model_name()) {
        return Err(plank::engine::THINK_LEVEL_REQUIRES_V41.to_string());
    }
    let color = std::io::stdout().is_terminal();
    if cfg.ui.is_headless() {
        return plank::ui::run_headless(engine, cfg, local_engine, reopen, plugins);
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
    match plank::ui::run_interactive(engine, cfg, local_engine, reopen, plugins)? {
        None => Ok(0),
        // `/edit-profile` asked to reopen the session under the edited
        // profile. `exec_restart` returns only when the exec failed.
        Some(restart) => Err(format!(
            "restart failed: {}",
            plank::profileedit::exec_restart(&restart)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_is_named_default_or_spec() {
        let mut cfg =
            plank::config::AgentConfig::from_settings(&plank::settings::Settings::default());
        cfg.model_spec = None;
        assert_eq!(model_choice(&cfg), plank::engines::Choice::Default);
        cfg.model_spec = Some("qwen".into());
        assert_eq!(model_choice(&cfg), plank::engines::Choice::Spec("qwen"));
        cfg.model_named = true;
        assert_eq!(model_choice(&cfg), plank::engines::Choice::Named("qwen"));
    }

    #[test]
    fn resolution_is_fatal_unless_dumping_config() {
        let mut cfg =
            plank::config::AgentConfig::from_settings(&plank::settings::Settings::default());
        assert!(resolution_is_fatal(&cfg));
        cfg.dump_config = true;
        assert!(!resolution_is_fatal(&cfg));
    }

    #[test]
    fn help_version_and_dump_config_never_migrate() {
        let base = plank::config::AgentConfig::from_settings(&plank::settings::Settings::default());
        assert!(should_migrate(&base));
        for flag in [
            "--help",
            "--version",
            "--dump-config",
            "--dump-profiles",
            "--dump-engines",
        ] {
            let cfg = plank::config::parse_options_with(
                &plank::settings::Settings::default(),
                &[flag.to_string()],
            )
            .expect("parses");
            assert!(!should_migrate(&cfg), "{flag} must not migrate");
        }
    }

    #[test]
    fn serve_migrates_regardless_of_should_migrate() {
        // `run_serve` has no early exit for `--help`/`--version`/`--dump-config`
        // — it always reaches `make_engine` — so its migration call must not be
        // gated by `should_migrate` the way `main`'s is.
        // `migrate_engine_layout_unconditionally` (what `run_serve` calls) takes
        // no `AgentConfig` at all, so there is nothing for a flag to gate — the
        // type signature itself is the guarantee. This test pins that: `main`'s
        // own launch still skips migration for these flags, via the separate,
        // gated `migrate_engine_layout`/`should_migrate` path.
        for flag in [
            "--help",
            "--version",
            "--dump-config",
            "--dump-profiles",
            "--dump-engines",
        ] {
            let cfg = plank::config::parse_options_with(
                &plank::settings::Settings::default(),
                &[flag.to_string()],
            )
            .expect("parses");
            assert!(
                !should_migrate(&cfg),
                "{flag} still must not migrate main's own launch"
            );
        }
        // Not calling `migrate_engine_layout_unconditionally` here: it reads and
        // writes the real `~/.plank`, which tests must never touch.
    }

    fn scratch_root(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("plank-main-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("scratch root");
        p
    }

    /// A resolved delta, as `resolve_model_delta` leaves it, without a real
    /// `.ggd`: the selection must follow the clone and inherit the base
    /// engine's companions without becoming managed.
    #[test]
    fn a_delta_on_the_managed_v4_base_inherits_its_companions() {
        let root = scratch_root("delta-inherit");
        let base = root.join("ds4vision.gguf");
        std::fs::write(&base, "m").expect("base");
        let clone = root.join("models/patched/abl.gguf");
        let mut cfg =
            plank::config::AgentConfig::from_settings(&plank::settings::Settings::default());
        cfg.model_spec = Some(root.join("models/abl.ggd").display().to_string());
        let catalog = resolve_selection(&mut cfg, &root, None).expect("a .ggd spec is a bare path");
        assert!(cfg.selection.as_ref().unwrap().mtp.is_none());
        cfg.model_path = Some(clone.clone());
        cfg.model_delta = Some(plank::ggufdelta::Resolved {
            path: clone.clone(),
            base,
            label: "abl".into(),
            id: "0123456789ab".into(),
        });
        finish_selection(&mut cfg, &root, &catalog);
        let sel = cfg.selection.as_ref().expect("selection");
        assert_eq!(sel.main, clone);
        assert_eq!(cfg.model_path.as_ref(), Some(&sel.main));
        assert_eq!(sel.id, Some(plank::manifest::EngineId::DS4VISION));
        assert_eq!(sel.mtp, Some(root.join("ds4vision.mtp.gguf")));
        assert_eq!(sel.vision, Some(root.join("ds4vision.vision.gguf")));
        assert!(!sel.managed_main);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Finding 4: when a recommendation wins over `engine.model`, the
    /// resolved `model_spec` must follow it, so anything that reports the
    /// model choice (the startup note, the no-engine-build error) names the
    /// engine that actually loaded rather than replaying the settings value
    /// it overrode.
    #[test]
    fn a_winning_recommendation_updates_model_spec() {
        let root = scratch_root("rec-model-spec");
        std::fs::write(root.join("qwen.gguf"), "q").expect("main");
        std::fs::write(root.join("qwen.vision.gguf"), "v").expect("vision");
        let mut cfg =
            plank::config::AgentConfig::from_settings(&plank::settings::Settings::default());
        cfg.model_spec = Some("ds41".to_string());
        let rec = plank::engines::Recommendation {
            profile: "HAL",
            engine: "qwen",
        };
        resolve_selection(&mut cfg, &root, Some(rec)).expect("resolves");
        assert_eq!(cfg.model_spec.as_deref(), Some("qwen"));
        assert_eq!(
            cfg.selection.as_ref().and_then(|s| s.id),
            Some(plank::manifest::EngineId::QWEN)
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A recommendation that loses (its file, or a companion, is missing)
    /// must leave `model_spec` alone: the settings value is still what is
    /// actually loading.
    #[test]
    fn a_losing_recommendation_leaves_model_spec_alone() {
        let root = scratch_root("rec-model-spec-lose");
        let mut cfg =
            plank::config::AgentConfig::from_settings(&plank::settings::Settings::default());
        cfg.model_spec = Some("ds41".to_string());
        let rec = plank::engines::Recommendation {
            profile: "HAL",
            engine: "qwen",
        };
        resolve_selection(&mut cfg, &root, Some(rec)).expect("resolves");
        assert_eq!(cfg.model_spec.as_deref(), Some("ds41"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// No delta: `finish_selection` only aligns `main` with `model_path`.
    #[test]
    fn without_a_delta_the_selection_is_untouched_but_for_main() {
        let root = scratch_root("delta-none");
        let other = root.join("mine.gguf");
        std::fs::write(&other, "o").expect("file");
        let mut cfg =
            plank::config::AgentConfig::from_settings(&plank::settings::Settings::default());
        cfg.model_spec = Some(other.display().to_string());
        let catalog = resolve_selection(&mut cfg, &root, None).expect("resolves");
        finish_selection(&mut cfg, &root, &catalog);
        let sel = cfg.selection.as_ref().expect("selection");
        assert_eq!(sel.main, other);
        assert!(sel.id.is_none() && sel.mtp.is_none() && sel.vision.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn dump_config_survives_a_bad_engine_model() {
        let settings = plank::settings::Settings::default();
        let args: Vec<String> = vec![
            "--model".into(),
            "definitely-not-a-real-engine-name".into(),
            "--dump-config".into(),
        ];
        // An empty scratch root: the catalog is the compiled-in one and the
        // real `~/.plank` is never read.
        let root = std::env::temp_dir().join(format!("plank-dump-config-{}", std::process::id()));
        let cfg = parse_config_in(&settings, &args, "plank", &root, None)
            .expect("dump-config must not abort");
        assert!(cfg.dump_config);
        assert!(cfg.selection.is_none());
    }

    const HAL: Option<plank::engines::Recommendation<'static>> =
        Some(plank::engines::Recommendation {
            profile: "HAL",
            engine: "qwen",
        });

    fn settings_model(spec: &str) -> plank::settings::Settings {
        let mut s = plank::settings::Settings::default();
        s.engine.model = Some(spec.into());
        s
    }

    fn picked(
        settings: &plank::settings::Settings,
        args: &[&str],
        root: &std::path::Path,
    ) -> String {
        let args: Vec<String> = args.iter().map(ToString::to_string).collect();
        let cfg = parse_config_in(settings, &args, "plank", root, HAL).expect("parses");
        cfg.selection
            .and_then(|s| s.id)
            .map(|id| id.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn only_a_command_line_model_counts_as_cli() {
        let cfg = plank::config::AgentConfig::from_settings(&settings_model("ds41"));
        assert!(!model_from_cli(&cfg), "engine.model is a settings choice");
        for args in [["--model", "ds41"], ["-m", "ds41"]] {
            let args: Vec<String> = args.iter().map(ToString::to_string).collect();
            let cfg = plank::config::parse_options_with(&settings_model("qwen"), &args).unwrap();
            assert!(model_from_cli(&cfg), "{args:?}");
        }
        let cfg = plank::config::parse_options_with(
            &plank::settings::Settings::default(),
            &["--model:ds41".to_string()],
        )
        .unwrap();
        assert!(model_from_cli(&cfg));
    }

    #[test]
    fn an_installed_recommendation_outranks_engine_model_but_not_the_flag() {
        let root = scratch_root("rec-precedence");
        std::fs::write(root.join("qwen.gguf"), "q").expect("qwen main");
        std::fs::write(root.join("qwen.vision.gguf"), "v").expect("qwen vision companion");
        let settings = settings_model("ds41");
        assert_eq!(picked(&settings, &[], &root), "qwen");
        assert_eq!(picked(&settings, &["--model", "ds41"], &root), "ds41");
        assert_eq!(
            picked(&settings, &["--model:ds4vision"], &root),
            "ds4vision"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_recommendation_not_on_disk_leaves_engine_model_in_charge() {
        let root = scratch_root("rec-absent");
        assert_eq!(picked(&settings_model("ds41"), &[], &root), "ds41");
        assert_eq!(
            picked(&plank::settings::Settings::default(), &[], &root),
            "ds4vision"
        );
        assert!(!root.join("qwen.gguf").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn dump_config_survives_a_bad_engine_model_under_a_recommendation() {
        let root = scratch_root("rec-dump");
        let args: Vec<String> = vec!["--dump-config".into()];
        let cfg = parse_config_in(
            &settings_model("not-an-engine-at-all"),
            &args,
            "plank",
            &root,
            HAL,
        )
        .expect("dump-config must not abort");
        assert!(cfg.dump_config);
        assert!(cfg.selection.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
}
