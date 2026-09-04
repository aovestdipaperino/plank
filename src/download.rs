// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Model auto-download with a playful progress display.
//!
//! When no `-m` is given, plank looks for `~/.plank/ds4flash.gguf`. If it is
//! missing, it offers to fetch the `DeepSeek` V4 Flash GGUF from Hugging Face
//! (`huggingface.co`),
//! streaming a magenta progress bar with a rotating series of messages.
//!
//! The download runs for hours, so the screen hosts a playable round of
//! [`crate::arcade::breakout`] above the gauge. The game is decoration: it
//! never delays the transfer, Esc puts it away, and `q` or Ctrl-C abort the
//! download from anywhere, so a rally can never trap the user.

use std::fs::{File, OpenOptions};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Gauge, Paragraph};

use crate::arcade;
use crate::arcade::breakout::Breakout;

/// Hugging Face repository hosting the GGUF files.
const REPO: &str = "antirez/deepseek-v4-gguf";
/// The recommended Vision-Experimental Flash quant (~81 GB) for 96–128 GB
/// machines.
///
/// The 0731 language checkpoint has been superseded by the Vision-Experimental
/// build, which carries the same routed-expert layout plus native image-token
/// support. plank assumes vision is always on, so this is the default model.
const FILE: &str = "DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf";

/// The `DSpark` speculative-decoding support GGUF (~5.6 GB) for the
/// Vision-Experimental checkpoint.
///
/// Not a standalone model: it is the auxiliary drafter, and it is
/// checkpoint-specific — pairing it with the older 0731 language checkpoint is
/// a load error, not a quality question, which is why it is fetched from the
/// same repository as [`FILE`].
const DSPARK_FILE: &str = "DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf";

/// The vision-encoder GGUF that pairs with the Vision-Experimental checkpoint.
///
/// A standalone encoder (~0.9 GB) loaded alongside the main model so the
/// `view_image` tool can decode images. plank assumes vision is always on, so
/// this is fetched at startup when absent, the same as the main model.
const VISION_ENCODER_FILE: &str = "DeepSeek-V4-Flash-Vision-Encoder.gguf";

/// Rotating status lines shown while the model downloads.
const MESSAGES: [&str; 200] = [
    "Summoning alien intelligence from the void...",
    "Negotiating with billions of sleeping neurons...",
    "Decompressing a mind larger than this galaxy's gossip...",
    "Teaching sand to think, one weight at a time...",
    "Bribing the experts to route through your machine...",
    "Downloading forbidden knowledge (entirely legally)...",
    "Assembling a brain from quantized stardust...",
    "Warming up the hypercube of latent thoughts...",
    "Coaxing the tensors out of hyperspace...",
    "Feeding the model its morning 81 gigabytes...",
    "Aligning the neural constellations...",
    "Almost sentient. Please hold.",
    "Downloading Einstein's secret thoughts...",
    "Borrowing Newton's apple for calibration...",
    "Reconstructing da Vinci's unfinished ideas...",
    "Extracting Turing's spare intuitions...",
    "Bottling Curie's glow-in-the-dark curiosity...",
    "Rewinding Tesla's midnight daydreams...",
    "Compressing Feynman's back-of-napkin genius...",
    "Unfolding Hawking's pocket universe...",
    "Transcribing Ramanujan's dreamed equations...",
    "Reviving Ada Lovelace's first algorithm...",
    "Downloading Gauss's mental arithmetic...",
    "Consulting Aristotle on the meaning of loops...",
    "Distilling Socrates' finest questions...",
    "Copying Euler's handwriting, one identity at a time...",
    "Refilling Archimedes' bathtub of eureka...",
    "Sampling Mozart's unwritten symphonies...",
    "Cataloguing Darwin's most patient observations...",
    "Downloading Hypatia's lost lectures...",
    "Reassembling the Library of Alexandria (backup copy)...",
    "Fetching the wisdom of a thousand owls...",
    "Convincing the electrons to hold still for a portrait...",
    "Untangling a yarn ball of pure reason...",
    "Downloading every chess move ever regretted...",
    "Teaching a rock to appreciate poetry...",
    "Percolating cognition through quantum coffee...",
    "Inflating the balloon of understanding...",
    "Herding photons into orderly thoughts...",
    "Whispering sweet nothings to the GPU...",
    "Rendering the shape of an idea...",
    "Debugging the universe's source code...",
    "Downloading the collective sigh of every mathematician...",
    "Polishing 671 billion tiny mirrors...",
    "Teaching the abacus to dream in floating point...",
    "Recovering the punchline to every unfinished joke...",
    "Downloading the smell of a fresh idea...",
    "Coaxing wisdom from a very large spreadsheet...",
    "Assembling a philosopher from spare parts...",
    "Downloading the confidence of a cat...",
    "Reheating the primordial soup of thought...",
    "Threading a needle with a beam of insight...",
    "Downloading the last common ancestor of all puns...",
    "Coaxing the muses out of early retirement...",
    "Tuning the orchestra of imaginary neurons...",
    "Downloading a mind that never forgets your birthday...",
    "Convincing infinity to fit on your SSD...",
    "Downloading the dreams of a sleeping supercomputer...",
    "Assembling IKEA furniture for the soul...",
    "Downloading the patience of a monk and the wit of a fox...",
    "Persuading the weights to arrive in a good mood...",
    "Downloading the ghost in the machine (friendly one)...",
    "Untying the Gordian knot with extra steps...",
    "Downloading Pythagoras's spare hypotenuse...",
    "Reticulating cognitive splines...",
    "Downloading the entire internet's second thoughts...",
    "Baking a soufflé of pure logic...",
    "Downloading a genie, minus the wishes limit...",
    "Teaching lightning to write sonnets...",
    "Downloading the murmur of a billion decisions...",
    "Aligning chakras of the transformer blocks...",
    "Downloading a librarian who has read everything...",
    "Convincing Schrödinger's cat to commit to an answer...",
    "Downloading the origami of thought...",
    "Assembling a council of tiny wise robots...",
    "Downloading the last piece of the jigsaw of reason...",
    "Steeping the model in a strong cup of context...",
    "Downloading Kepler's spare orbits...",
    "Recovering Fermat's actual margin notes...",
    "Downloading Leibniz's favorite infinitesimal...",
    "Waking Boltzmann's brain gently...",
    "Downloading the collected wisdom of grandmothers...",
    "Teaching the silicon to daydream responsibly...",
    "Downloading a shortcut through the labyrinth of ideas...",
    "Convincing entropy to take a coffee break...",
    "Downloading the harmony of the spheres...",
    "Fetching the oracle's dial-up connection...",
    "Downloading a brain that laughs at its own jokes...",
    "Ironing the wrinkles out of a very large mind...",
    "Downloading the wisdom of crowds, minus the crowd...",
    "Coaxing the model to think before it speaks...",
    "Downloading a thousand aha moments in bulk...",
    "Assembling curiosity from first principles...",
    "Downloading the quiet hum of deep concentration...",
    "Teaching a calculator to fall in love with numbers...",
    "Downloading Copernicus's change of perspective...",
    "Retrieving Galileo's confiscated notebooks...",
    "Downloading Maxwell's tidy little equations...",
    "Borrowing Fourier's favorite wave...",
    "Downloading Noether's beautiful symmetries...",
    "Reassembling Babbage's dream engine...",
    "Downloading Shannon's units of surprise...",
    "Refilling Mendeleev's periodic imagination...",
    "Downloading a mentor who never gets tired of you...",
    "Convincing the bits to line up alphabetically...",
    "Downloading the wisdom of every rejected idea...",
    "Teaching a spreadsheet to feel awe...",
    "Downloading the model's sense of humor (dry setting)...",
    "Assembling a brain that remembers where it put its keys...",
    "Downloading a philosopher-king with excellent manners...",
    "Coaxing genius out of a very shy neural net...",
    "Downloading the echo of every good question...",
    "Teaching the tensors to hum while they work...",
    "Downloading the wisdom of a very old tree...",
    "Convincing chaos to color inside the lines...",
    "Downloading a mind fluent in every silence...",
    "Assembling the world's most patient tutor...",
    "Downloading the last word in 4,000 languages...",
    "Teaching the model to appreciate a good pause...",
    "Downloading the secret handshake of the neurons...",
    "Untangling the headphones of the universe...",
    "Downloading a brain that reads the manual first...",
    "Coaxing insight from a mountain of matrices...",
    "Downloading the courage to say 'I don't know'...",
    "Assembling a thinker who shows their work...",
    "Downloading the wisdom to ask a better question...",
    "Teaching the model humility, one epoch at a time...",
    "Downloading a memory palace with 671 billion rooms...",
    "Convincing the weights to stop bickering...",
    "Downloading the calm of a lake at dawn...",
    "Fetching a genius who returns your calls...",
    "Downloading the spark that jumped the gap...",
    "Assembling a mind from equal parts logic and wonder...",
    "Downloading the recipe for a good idea...",
    "Teaching a machine to notice the little things...",
    "Downloading the wisdom of every second guess...",
    "Coaxing the model to color outside the lines (a little)...",
    "Downloading the confidence of a well-rested owl...",
    "Assembling a brain that never says 'obviously'...",
    "Downloading the collected footnotes of civilization...",
    "Teaching the silicon to savor a paradox...",
    "Downloading a thinker with a generous imagination...",
    "Convincing the model that the journey matters too...",
    "Downloading the last laugh of a happy neuron...",
    "Fetching the wisdom of a very good librarian...",
    "Downloading Leonardo's helicopter (still in beta)...",
    "Reassembling the parliament of your future thoughts...",
    "Downloading a mind that knows when to be quiet...",
    "Teaching probability to relax a little...",
    "Downloading the collected margins of every textbook...",
    "Coaxing the model to dream in high resolution...",
    "Downloading a genius who admits mistakes gracefully...",
    "Assembling a brain fluent in second chances...",
    "Downloading the model's inner monologue (subtitled)...",
    "Convincing the tensors to share nicely...",
    "Downloading the wisdom of a slow, deep breath...",
    "Fetching a mentor with infinite patience and no ego...",
    "Downloading the spark behind every 'what if'...",
    "Teaching a machine to be curious, not just correct...",
    "Downloading the collected dreams of every inventor...",
    "Assembling a mind that loves a hard problem...",
    "Downloading the whisper that becomes a theorem...",
    "Coaxing brilliance from a river of numbers...",
    "Downloading the wisdom to change its mind...",
    "Fetching a genius with a soft spot for beginners...",
    "Downloading the quiet joy of understanding...",
    "Teaching the model to marvel at the ordinary...",
    "Downloading a brain that never mansplains...",
    "Convincing the weights to converge on kindness...",
    "Downloading the last piece of everyone's puzzle...",
    "Assembling a thinker who reads between the lines...",
    "Downloading the model's collection of favorite facts...",
    "Coaxing the neurons into a standing ovation...",
    "Downloading the wisdom of a thousand quiet mornings...",
    "Fetching a mind that turns questions into doorways...",
    "Downloading the courage to explore a wild idea...",
    "Teaching a machine to enjoy being wrong (briefly)...",
    "Downloading the model's sense of wonder (fully charged)...",
    "Assembling a brain that gives credit generously...",
    "Downloading the last neuron on the string of thought...",
    "Convincing the model that big ideas start small...",
    "Downloading the collected patience of every teacher...",
    "Fetching a genius who never talks down to you...",
    "Downloading the hum of a mind hard at work...",
    "Teaching the tensors the value of a good night's sleep...",
    "Downloading a thinker who loves a plot twist...",
    "Coaxing wisdom from the static between the stars...",
    "Downloading the model's appreciation for a clean proof...",
    "Assembling a brain that keeps its promises...",
    "Downloading the spark that lights the next idea...",
    "Fetching the collective 'ohhh, I get it now'...",
    "Downloading a mind that makes hard things feel easy...",
    "Teaching a machine the art of the thoughtful pause...",
    "Downloading the last gigabyte of pure intelligence...",
    "Fetching Fibonacci's spare rabbits for the demo...",
    "Downloading Pascal's triangle, batteries included...",
    "Coaxing the model to finish its homework early...",
    "Nearly done assembling your pocket genius...",
    "Polishing the final thought before it wakes up...",
    "Any moment now, sentience with a smile...",
];

/// Default model location used when `-m` is not supplied.
#[must_use]
pub fn default_model_path() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".plank").join("ds4flash.gguf")
}

/// Hugging Face download URL for `file` in [`REPO`].
fn file_url(file: &str) -> String {
    format!("https://huggingface.co/{REPO}/resolve/main/{file}")
}

/// Default `DSpark` support-model location, used when `--dspark` is given
/// without an explicit `--mtp`.
///
/// Sits beside the main model and mirrors its name so the pairing is legible
/// on disk: `ds4flash.gguf` and `ds4flash.dspark.gguf`.
#[must_use]
pub fn default_dspark_path() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".plank").join("ds4flash.dspark.gguf")
}

/// Default vision-encoder location. plank assumes vision is always on, so this
/// is loaded alongside the main model whenever the native engine is used.
///
/// Sits beside the main model: `ds4flash.gguf` and `ds4flash.vision.gguf`.
#[must_use]
pub fn default_vision_path() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".plank").join("ds4flash.vision.gguf")
}

/// Hugging Face download URL for the default Flash GGUF.
#[must_use]
pub fn model_url() -> String {
    file_url(FILE)
}

/// Hugging Face download URL for the `DSpark` support GGUF.
#[must_use]
pub fn dspark_url() -> String {
    file_url(DSPARK_FILE)
}

/// Hugging Face download URL for the vision-encoder GGUF.
#[must_use]
pub fn vision_url() -> String {
    file_url(VISION_ENCODER_FILE)
}

/// Ensures the `DSpark` support model exists at `path`, offering to download it
/// when it is missing.
///
/// Deliberately simpler than [`ensure_model`]: no upgrade check. The support
/// file is pinned to the target checkpoint rather than tracking a quant
/// family, so "is there a newer build" is not a question worth asking on every
/// launch — a mismatched drafter is refused by the engine at load.
///
/// # Errors
/// Returns an error string when the user declines, when stdin is not a
/// terminal (so no prompt is possible), or when the download fails.
pub fn ensure_dspark(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        return Err(format!(
            "no DSpark support model at {}; pass --mtp <path> or download it first",
            path.display()
        ));
    }
    let resuming = partial_bytes(path) > 0;
    eprintln!("No DSpark support model found at {}.", path.display());
    if resuming {
        eprintln!(
            "A partial download exists ({:.1} GB); plank can resume it from Hugging Face:",
            gb(partial_bytes(path))
        );
    } else {
        eprintln!("plank can download the DSpark support model (~5.6 GB) from Hugging Face:");
    }
    eprintln!("  {}", dspark_url());
    eprint!(
        "{} it now? [Y/n] ",
        if resuming { "Resume" } else { "Download" }
    );
    io::stderr().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| e.to_string())?;
    // Default to yes: Enter (empty) accepts, like the main model prompt.
    if matches!(answer.trim(), "n" | "N" | "no") {
        return Err(
            "no DSpark support model available; re-run without --dspark or pass --mtp <path>"
                .to_string(),
        );
    }
    download(&dspark_url(), path)
}

/// Resolves the `DSpark` support model for a run that asked for `--dspark`
/// without naming one, fetching it if needed.
///
/// An explicit `--mtp` always wins and is left untouched — it is also how a
/// legacy one-stage MTP drafter is supplied, which is a different file
/// entirely. Does nothing unless `DSpark` was actually requested, so a normal
/// run never pays for this.
///
/// # Errors
/// Propagates [`ensure_dspark`] failures: declined prompt, no terminal to
/// prompt on, or a failed download.
pub fn ensure_dspark_support(engine: &mut crate::config::EngineTuning) -> Result<(), String> {
    if !engine.dspark || engine.mtp_path.is_some() {
        return Ok(());
    }
    let path = default_dspark_path();
    ensure_dspark(&path)?;
    engine.mtp_path = Some(path);
    Ok(())
}

/// Ensures the vision-encoder GGUF exists at its default path, offering to
/// download it if missing.
///
/// plank assumes vision is always on, so this runs on every native-engine
/// startup alongside [`ensure_model`]. A missing encoder is a hard error: the
/// `view_image` tool would refuse at call time, and the model was trained to
/// expect image tokens.
///
/// # Errors
/// Returns an error string when the user declines, when stdin is not a
/// terminal (so no prompt is possible), or when the download fails.
pub fn ensure_vision_encoder() -> Result<(), String> {
    let path = default_vision_path();
    if path.exists() {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        return Err(format!(
            "no vision encoder at {}; pass --vision <path> or download it first",
            path.display()
        ));
    }
    let resuming = partial_bytes(&path) > 0;
    eprintln!("No vision encoder found at {}.", path.display());
    if resuming {
        eprintln!(
            "A partial download exists ({:.1} GB); plank can resume it from Hugging Face:",
            gb(partial_bytes(&path))
        );
    } else {
        eprintln!("plank can download the vision encoder (~0.9 GB) from Hugging Face:");
    }
    eprintln!("  {}", vision_url());
    eprint!(
        "{} it now? [Y/n] ",
        if resuming { "Resume" } else { "Download" }
    );
    io::stderr().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| e.to_string())?;
    if matches!(answer.trim(), "n" | "N" | "no") {
        return Err(
            "no vision encoder available; the view_image tool will refuse at call time".to_string(),
        );
    }
    download(&vision_url(), &path)
}

/// Ensures a model file exists at `path`, offering to download it if missing.
///
/// # Errors
/// Returns an error string when the user declines, when stdin is not a
/// terminal (so no prompt is possible), or when the download fails.
pub fn ensure_model(path: &Path) -> Result<(), String> {
    if path.exists() {
        // Upgrades are the manifest's business now (`check_manifest_at_startup`),
        // and they happen in the background rather than as a blocking prompt
        // before the engine loads.
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        return Err(format!(
            "no model at {}; pass -m <path> or download it first",
            path.display()
        ));
    }
    // A leftover .part file means a previous download can be resumed.
    let resuming = partial_bytes(path) > 0;
    eprintln!("No model found at {}.", path.display());
    if resuming {
        eprintln!(
            "A partial download exists ({:.1} GB); plank can resume it from Hugging Face:",
            gb(partial_bytes(path))
        );
    } else {
        eprintln!("plank can download DeepSeek V4 Flash (~87 GB) from Hugging Face:");
    }
    eprintln!("  {}", model_url());
    eprint!(
        "{} it now? [Y/n] ",
        if resuming { "Resume" } else { "Download" }
    );
    io::stderr().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| e.to_string())?;
    // Default to yes: Enter (empty) accepts.
    if matches!(answer.trim(), "n" | "N" | "no") {
        return Err("no model available; re-run with -m <path> or download it".to_string());
    }
    download(&model_url(), path)
}

/// Size of the partial download alongside `dest`, or 0 if none.
fn partial_bytes(dest: &Path) -> u64 {
    std::fs::metadata(dest.with_extension("part")).map_or(0, |m| m.len())
}

/// Downloads `url` to `dest` via `ureq`, showing the animated progress bar.
fn download(url: &str, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let part = dest.with_extension("part");
    let total = content_length(url);

    // A .part that already matches the full size just needs the final rename
    // (e.g. a previous run died between finishing the body and renaming).
    if total.is_some_and(|t| t > 0 && partial_bytes(dest) == t) {
        std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
        finish(dest);
        eprintln!("Model ready at {}.", dest.display());
        return Ok(());
    }

    // The transfer runs on a worker thread appending to the .part file; the
    // UI thread watches the file size and flips `cancel` to stop the worker.
    let cancel = Arc::new(AtomicBool::new(false));
    let worker = spawn_fetch(url.to_string(), part.clone(), Arc::clone(&cancel));

    // Drive the progress on a Ratatui alternate screen so it repaints in place
    // in every terminal (including block-based ones like Warp).
    let mut terminal = ratatui::init();
    let result = run_ui(&mut terminal, &worker, &cancel, &part, total);
    ratatui::restore();
    // Never leave the worker running if the UI aborted.
    cancel.store(true, Ordering::Relaxed);
    let joined = worker.join().map_err(|_| "download thread panicked")?;
    result?;
    joined?;

    // A body that ends early reads as a clean EOF, so a short .part would be
    // renamed into place and only fail much later, deep in the model loader.
    // Leave the .part alone: a re-run resumes it.
    if let Some(expected) = total.filter(|t| *t > 0) {
        let got = partial_bytes(dest);
        if got != expected {
            return Err(format!(
                "download incomplete: {:.1} of {:.1} GB. Re-run plank to resume.",
                gb(got),
                gb(expected)
            ));
        }
    }
    std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
    finish(dest);
    eprintln!("Model ready at {}.", dest.display());
    Ok(())
}

/// Celebratory sounds, best first.
///
/// `Tada` is the one to play, but Apple dropped it from `/System/Library/Sounds`
/// in recent macOS, so the list falls through to the nearest survivors rather
/// than going silent on a machine that never had it.
const TADA_SOUNDS: [&str; 3] = ["Tada", "Hero", "Glass"];

/// Where macOS keeps sounds: the user's own first, then the system's.
fn sound_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join("Library/Sounds"));
    }
    dirs.push(PathBuf::from("/System/Library/Sounds"));
    dirs
}

/// First of `names` that exists under any of `dirs`, in preference order.
///
/// Names are tried outermost so a user's own `Tada.aiff` beats a system
/// fallback, rather than whichever directory happens to come first.
fn find_sound(names: &[&str], dirs: &[PathBuf]) -> Option<PathBuf> {
    names.iter().find_map(|name| {
        dirs.iter()
            .map(|dir| dir.join(format!("{name}.aiff")))
            .find(|path| path.is_file())
    })
}

/// Plays the completion sound. Best-effort and detached — a download that
/// finished is not worth failing over a missing audio player.
#[cfg(target_os = "macos")]
fn play_tada() {
    let Some(path) = find_sound(&TADA_SOUNDS, &sound_dirs()) else {
        return;
    };
    let _ = std::process::Command::new("/usr/bin/afplay")
        .arg(path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// No system sound library to draw on off macOS.
#[cfg(not(target_os = "macos"))]
fn play_tada() {}

/// Marks a finished download: sounds the fanfare.
///
/// It used to also write a `.source` stamp naming the repo file the model came
/// from, which was how a bumped `FILE` constant became visible to someone who
/// already had a model. The manifest replaced that entirely: `~/.plank/ds4.manifest`
/// records the whole installed set by version, so a per-file stamp has nothing
/// left to say.
fn finish(dest: &Path) {
    let _ = dest;
    play_tada();
}

/// Spawns the worker thread that streams `url` into `part`, resuming if possible.
fn spawn_fetch(
    url: String,
    part: PathBuf,
    cancel: Arc<AtomicBool>,
) -> JoinHandle<Result<(), String>> {
    std::thread::spawn(move || fetch(&url, &part, &cancel))
}

/// Streams `url` into `part` with an HTTP Range resume of any existing bytes.
fn fetch(url: &str, part: &Path, cancel: &AtomicBool) -> Result<(), String> {
    let offset = std::fs::metadata(part).map_or(0, |m| m.len());
    let mut request = ureq::get(url);
    if offset > 0 {
        request = request.header("Range", format!("bytes={offset}-"));
    }
    let mut response = request
        .call()
        .map_err(|e| format!("download failed: {e}"))?;

    // 206 means the server honored the resume; anything else restarts from zero.
    let mut file = if offset > 0 && response.status().as_u16() == 206 {
        OpenOptions::new()
            .append(true)
            .open(part)
            .map_err(|e| e.to_string())?
    } else {
        File::create(part).map_err(|e| e.to_string())?
    };

    let mut reader = response.body_mut().as_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("download cancelled".to_string());
        }
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(());
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }
}

/// Rows the message/gauge/stats block always keeps: message, gauge, stats and
/// the two spacers between them.
const PROGRESS_ROWS: u16 = 5;
/// Rows taken by one of the rules bracketing the playfield.
const RULE_ROWS: u16 = 1;
/// Both rules together: one above the field, one below it.
const RULES_ROWS: u16 = RULE_ROWS * 2;
/// Frame budget while the game is up; the ball needs far more than the 250 ms
/// a gauge is happy with.
const GAME_FRAME_MS: u64 = 16;
/// Frame budget with no game on screen — a gauge and a slowly rotating message
/// do not justify spinning.
const IDLE_FRAME_MS: u64 = 250;
/// How long each rotating message stays up. Long enough to actually read it:
/// the list runs to 200 lines and the download to hours, so there is no need
/// to hurry through them.
const MESSAGE_DWELL: Duration = Duration::from_secs(9);

/// Where each piece of the download screen goes.
///
/// The progress block always gets its rows; the playfield takes everything
/// left over, so the field grows with the terminal. Below [`arcade::MIN_W`] x
/// [`arcade::MIN_H`], or once the game has been closed, the screen is just the
/// message, gauge and stats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Screen {
    /// The rules bracketing the playfield, above and below.
    rule_top: Option<Rect>,
    rule_bottom: Option<Rect>,
    game: Option<Rect>,
    message: Rect,
    gauge: Rect,
    stats: Rect,
}

/// Splits `area`, giving the playfield whatever the progress block does not
/// need. `show_game` is false once the user has closed the game.
fn layout(area: Rect, show_game: bool) -> Screen {
    let avail = area.height.saturating_sub(PROGRESS_ROWS);
    let game_h = if show_game && playable(area.height, area.width) {
        // Two rows go to the rules bracketing the playfield.
        avail - RULES_ROWS
    } else {
        0
    };
    let rule_h = if game_h > 0 { RULE_ROWS } else { 0 };

    let rows = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(rule_h), // rule above the playfield
        Constraint::Length(game_h),
        Constraint::Length(rule_h), // rule below it
        Constraint::Length(1),      // message
        Constraint::Length(1),      // spacer
        Constraint::Length(1),      // gauge
        Constraint::Length(1),      // spacer
        Constraint::Length(1),      // stats
        Constraint::Fill(1),
    ])
    .split(area);

    Screen {
        rule_top: (rule_h > 0).then(|| rows[1]),
        game: (game_h > 0).then(|| rows[2]),
        rule_bottom: (rule_h > 0).then(|| rows[3]),
        message: rows[4],
        gauge: rows[6],
        stats: rows[8],
    }
}

/// Whether a terminal of `rows` x `cols` has room for the playfield once the
/// progress block has taken its share.
fn playable(rows: u16, cols: u16) -> bool {
    cols >= arcade::MIN_W && rows.saturating_sub(PROGRESS_ROWS + RULES_ROWS) >= arcade::MIN_H
}

/// What a key press asks the download screen to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Abort the transfer.
    Cancel,
    /// Put the game away, leaving the bare progress screen.
    CloseGame,
    /// Not ours — hand it to the game.
    Play,
}

/// Classifies a key press.
///
/// Esc is staged: it closes the game first and only cancels once the game is
/// already gone, so reaching for it mid-rally cannot lose you an 87 GB
/// download by accident. `q` and Ctrl-C stay immediate — someone typing those
/// means it.
fn classify(key: KeyEvent, game_open: bool) -> Action {
    if matches!(key.code, KeyCode::Char('q'))
        || (matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL))
    {
        return Action::Cancel;
    }
    if key.code == KeyCode::Esc {
        return if game_open {
            Action::CloseGame
        } else {
            Action::Cancel
        };
    }
    Action::Play
}

/// Runs the full-screen download UI until the worker finishes or the user cancels.
fn run_ui(
    terminal: &mut ratatui::DefaultTerminal,
    worker: &JoinHandle<Result<(), String>>,
    cancel: &AtomicBool,
    part: &Path,
    total: Option<u64>,
) -> Result<(), String> {
    let start = Instant::now();
    // Bytes already on disk: the baseline a resumed download measures from.
    let done = std::fs::metadata(part).map_or(0, |m| m.len());
    let mut msg = 0usize;
    let mut last_rotate = Instant::now();

    // The wait is long enough to deserve a game. It advances only through the
    // measured delta below, never its own clock — see `crate::arcade`. `None`
    // once the user has closed it with Esc.
    let mut game = Some(Breakout::new(seed()));
    let mut last_frame = Instant::now();
    loop {
        if worker.is_finished() {
            let _ = terminal.draw(|f| draw(f, game.as_ref(), part, total, msg, start, done));
            // The caller joins the worker and surfaces its error, if any.
            return Ok(());
        }
        // The poll timeout paces the frames: fast enough for a moving ball
        // while one is on screen, unhurried once the game is closed or the
        // terminal is too small for it and there is only a gauge to repaint.
        let size = terminal
            .size()
            .map_or((24u16, 80u16), |s| (s.height, s.width));
        let tick = if game.is_some() && playable(size.0, size.1) {
            GAME_FRAME_MS
        } else {
            IDLE_FRAME_MS
        };
        if event::poll(Duration::from_millis(tick)).map_err(|e| e.to_string())?
            && let Ok(Event::Key(k)) = event::read()
            && k.kind == KeyEventKind::Press
        {
            match classify(k, game.is_some()) {
                Action::Cancel => {
                    cancel.store(true, Ordering::Relaxed);
                    return Err("download cancelled".to_string());
                }
                Action::CloseGame => game = None,
                Action::Play => {
                    if let Some(g) = game.as_mut() {
                        g.handle_key(k);
                    }
                }
            }
        }
        if last_rotate.elapsed() >= MESSAGE_DWELL {
            msg += 1;
            last_rotate = Instant::now();
        }
        let dt = last_frame.elapsed();
        last_frame = Instant::now();
        if let Some(g) = game.as_mut() {
            g.step(u64::try_from(dt.as_millis()).unwrap_or(u64::MAX));
        }
        let _ = terminal.draw(|f| draw(f, game.as_ref(), part, total, msg, start, done));
    }
}

/// A seed for the download screen's game, from the wall clock.
fn seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_secs() ^ u64::from(d.subsec_nanos()) << 20)
}

/// Paints the playfield into `area`: the game's glyphs, its scoreboard on the
/// bottom row, and any centered banner stamped over the middle.
///
/// Mirrors [`crate::tui::draw_arcade`], but into a sub-area rather than the
/// whole frame, and clamped so a stray coordinate cannot reach the gauge.
fn draw_game(frame: &mut Frame, game: &Breakout, area: Rect) {
    if area.height < 2 {
        return;
    }
    let play = Rect {
        height: area.height - 1,
        ..area
    };

    let buf = frame.buffer_mut();
    for g in game.glyphs(play.width, play.height) {
        if g.x >= play.width || g.y >= play.height {
            continue;
        }
        if let Some(cell) = buf.cell_mut(Position::new(play.x + g.x, play.y + g.y)) {
            let (r, gr, b) = g.color;
            cell.set_char(g.ch).set_fg(Color::Rgb(r, gr, b));
        }
    }

    if let Some(text) = game.banner() {
        let width = u16::try_from(text.chars().count()).unwrap_or(u16::MAX);
        let rect = Rect {
            x: play.x + play.width.saturating_sub(width + 2) / 2,
            y: play.y + play.height / 2,
            width: (width + 2).min(play.width),
            height: 1,
        };
        frame.render_widget(Clear, rect);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(" {text} "),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Rgb(255, 214, 120))
                    .add_modifier(Modifier::BOLD),
            ))),
            rect,
        );
    }

    let hud = Rect {
        y: area.y + area.height - 1,
        height: 1,
        ..area
    };
    frame.render_widget(
        Paragraph::new(game.hud())
            .style(Style::default().fg(Color::Indexed(245)))
            .alignment(Alignment::Center),
        hud,
    );
}

/// Draws one download frame: centered logo, red rotating message, gauge, stats.
fn draw(
    frame: &mut Frame,
    game: Option<&Breakout>,
    part: &Path,
    total: Option<u64>,
    msg: usize,
    start: Instant,
    done: u64,
) {
    let current = std::fs::metadata(part).map_or(0, |m| m.len());
    let elapsed = start.elapsed().as_secs_f64();
    // Only bytes fetched *this run* count toward the rate: on a resume the
    // file already holds `done` bytes, and charging those to zero elapsed time
    // reports a wild speed and a meaningless ETA.
    #[allow(clippy::cast_precision_loss)]
    let speed = if elapsed > 0.0 {
        current.saturating_sub(done) as f64 / elapsed / 1_000_000.0
    } else {
        0.0
    };

    let area = frame.area();
    let screen = layout(area, game.is_some());

    // Rules bracketing the playfield: the wall hangs from the top one, and
    // the bottom one closes the field off from the progress block.
    for row in [screen.rule_top, screen.rule_bottom].into_iter().flatten() {
        frame.render_widget(
            Paragraph::new("─".repeat(row.width as usize))
                .style(Style::default().fg(Color::Indexed(240))),
            row,
        );
    }

    if let (Some(row), Some(game)) = (screen.game, game) {
        draw_game(frame, game, row);
    }

    let red = Style::default().fg(Color::Red).add_modifier(Modifier::BOLD);
    let message = Paragraph::new(MESSAGES[msg % MESSAGES.len()])
        .style(red)
        .alignment(Alignment::Center);
    frame.render_widget(message, screen.message);

    // A centered, fixed-width gauge.
    let gauge_area = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Length(48),
        Constraint::Fill(1),
    ])
    .split(screen.gauge)[1];
    #[allow(clippy::cast_precision_loss)]
    let ratio = match total.filter(|t| *t > 0) {
        Some(t) => (current as f64 / t as f64).clamp(0.0, 1.0),
        None => 0.0,
    };
    let label = total.filter(|t| *t > 0).map_or_else(
        || "downloading...".to_string(),
        |_| format!("{:.1}%", ratio * 100.0),
    );
    let eta = total.and_then(|t| eta_secs(current, t, done, elapsed).map(format_eta));
    let eta_slot = eta
        .as_deref()
        .and_then(|text| eta_slot(gauge_area, ratio, &label, text));
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Red).bg(Color::Indexed(238)))
        .ratio(ratio)
        .label(label);
    frame.render_widget(gauge, gauge_area);

    // The ETA rides in the unfilled part of the bar. It is painted over the
    // gauge, so it keeps the gauge's own unfilled background.
    if let (Some(slot), Some(text)) = (eta_slot, eta) {
        frame.render_widget(
            Paragraph::new(text)
                .alignment(Alignment::Center)
                .style(Style::default().fg(Color::Gray).bg(Color::Indexed(238))),
            slot,
        );
    }

    let stats = match total {
        Some(t) => format!("{:.1} / {:.1} GB   {speed:.0} MB/s", gb(current), gb(t)),
        None => format!("{:.1} GB   {speed:.0} MB/s", gb(current)),
    };
    let stats_line = Paragraph::new(stats)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center);
    frame.render_widget(stats_line, screen.stats);
}

/// Seconds left, from the rate achieved so far.
///
/// `done` is what was already on disk when this run started — a resumed
/// download must not count those bytes as progress, or the first frames report
/// an enormous rate and an ETA of nothing. `None` until there is enough of a
/// sample to divide by.
fn eta_secs(current: u64, total: u64, done: u64, elapsed: f64) -> Option<f64> {
    let fetched = current.checked_sub(done)?;
    let left = total.checked_sub(current)?;
    if elapsed <= 0.5 || fetched == 0 || left == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let rate = fetched as f64 / elapsed;
    #[allow(clippy::cast_precision_loss)]
    let secs = left as f64 / rate;
    secs.is_finite().then_some(secs)
}

/// Renders seconds as a compact "time left", coarsening as it grows.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_eta(secs: f64) -> String {
    let secs = secs.clamp(0.0, f64::from(u32::MAX)) as u64;
    match (secs / 3600, (secs % 3600) / 60, secs % 60) {
        (0, 0, s) => format!("{s}s left"),
        (0, m, s) => format!("{m}m {s}s left"),
        (h, m, _) => format!("{h}h {m}m left"),
    }
}

/// Where the ETA goes inside the bar, or `None` when it will not fit.
///
/// It is centered in the unfilled span, but never over the percent label the
/// gauge centers on the whole bar — so the usable room is what is left to the
/// right of that label. As the bar fills, that room shrinks; once it is too
/// narrow the ETA is dropped rather than clipped.
fn eta_slot(bar: Rect, ratio: f64, label: &str, text: &str) -> Option<Rect> {
    let text_w = u16::try_from(text.chars().count()).ok()?;
    let label_w = u16::try_from(label.chars().count()).ok()?;

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let filled = (ratio.clamp(0.0, 1.0) * f64::from(bar.width)).round() as u16;
    let label_end = bar.x + (bar.width.saturating_sub(label_w)) / 2 + label_w;
    // One blank column of breathing room after each of them.
    let start = (bar.x + filled).max(label_end + 1);
    let end = bar.x + bar.width;

    let room = end.checked_sub(start)?;
    if room < text_w + 2 {
        return None;
    }
    Some(Rect {
        x: start + (room - text_w) / 2,
        y: bar.y,
        width: text_w,
        height: 1,
    })
}

/// Probes the total download size via a HEAD request.
fn content_length(url: &str) -> Option<u64> {
    let response = ureq::head(url).call().ok()?;
    response
        .headers()
        .get("content-length")?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// Bytes as gigabytes, shared with `downloader`'s user-facing text so the two
/// modules never format the same quantity two different ways.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000_000.0
}

/// The manifest URL. Kept in the repo rather than on a release asset so the
/// artifact set is reviewed in a pull request like any other change.
#[cfg(not(test))]
const MANIFEST_URL: &str =
    "https://raw.githubusercontent.com/aovestdipaperino/plank/main/ds4.manifest";
/// Bounded timeout for the manifest fetch. Startup must never hang on the
/// network.
#[cfg(not(test))]
const MANIFEST_TIMEOUT_SECS: u64 = 3;
/// How often to ask whether a newer artifact set exists.
const MANIFEST_CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60;
/// Records when the manifest was last fetched, beside the model.
const MANIFEST_CHECK_FILE: &str = "manifest-check";

/// Fetches the manifest, bounded by [`MANIFEST_TIMEOUT_SECS`]. `None` on any
/// failure — offline, timeout, HTTP error — so the caller stays quiet.
#[cfg(not(test))]
fn fetch_manifest() -> Option<String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(MANIFEST_TIMEOUT_SECS)))
        .build()
        .new_agent();
    let mut resp = agent
        .get(MANIFEST_URL)
        .header("User-Agent", concat!("plank/", env!("CARGO_PKG_VERSION")))
        .call()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.body_mut().read_to_string().ok()
}

/// Test builds never touch the network.
#[cfg(test)]
fn fetch_manifest() -> Option<String> {
    None
}

/// Whether the 24-hour check window has elapsed under `root`, and records
/// that it has.
///
/// The timestamp is written whether or not anything comes of the check: a
/// startup prompt already declined must not reappear on the next launch.
fn check_due_in(root: &Path) -> bool {
    let path = root.join(MANIFEST_CHECK_FILE);
    let last = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| t.trim().parse::<u64>().ok());
    let now = crate::downloader::now_epoch();
    if last.is_some_and(|t| now.saturating_sub(t) < MANIFEST_CHECK_INTERVAL_SECS) {
        return false;
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, format!("{now}\n"));
    true
}

/// Where a manifest version the user declined (or that failed / was
/// cancelled) is recorded under `root`, so it is not silently re-offered.
fn declined_path_in(root: &Path) -> PathBuf {
    crate::manifest::downloads_dir_in(root).join("declined")
}

/// The version most recently declined under `root`, if any.
fn read_declined_in(root: &Path) -> Option<u32> {
    std::fs::read_to_string(declined_path_in(root))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Records `version` as declined under `root`. Best-effort: a failure here
/// just means the offer may reappear, which is recoverable.
fn record_declined_in(root: &Path, version: u32) {
    let path = declined_path_in(root);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, format!("{version}\n"));
}

/// If the last background job ended in a failed hash check, or in a cancel
/// that discarded the partial files, records its version as declined.
///
/// The spec is explicit that automatic retry after a hash mismatch, or after
/// a Delete cancel, is out of scope: without this, the next `24h` check
/// window would silently restart the very download the user just stopped.
/// `/model download` remains the way to start it again on purpose.
///
/// A Keep cancel is deliberately excluded: its entire point is "stop now,
/// resume later", so it must not be treated as a decline — that is what
/// [`crate::downloader::cancelled_kept`] distinguishes.
fn note_last_run_outcome_in(root: &Path) {
    if let Some(state) = crate::downloader::read_state_in(root) {
        let declined = state.phase == crate::downloader::Phase::Failed
            || (state.phase == crate::downloader::Phase::Cancelled
                && !crate::downloader::cancelled_kept(&state));
        if declined {
            record_declined_in(root, state.version);
        }
    }
}

/// Whether an artifact of `kind` is present on disk under `root`.
fn artifact_installed_in(root: &Path, kind: &str) -> bool {
    crate::manifest::local_path_for_in(root, kind).is_some_and(|p| p.exists())
}

/// Prints the manifest's version transition and notes, then blocks for a
/// `[y/N]` answer. Only `y`/`Y`/`yes` accepts; anything else — including a
/// bare Enter — declines.
///
/// Mirrors the pre-manifest `offer_upgrade`'s prompt shape (commit 8d9ef82):
/// a 93 GB background transfer must not start unannounced, and the safe
/// default is no.
fn confirm_background_download(manifest: &crate::manifest::Manifest, from: u32) -> bool {
    eprintln!(
        "plank: a newer model is available (version {} → {}{}).",
        from,
        manifest.version,
        if manifest.notes.is_empty() {
            String::new()
        } else {
            format!(": {}", manifest.notes)
        }
    );
    eprint!("Download it in the background? [y/N] ");
    io::stderr().flush().ok();
    let mut answer = String::new();
    if std::io::stdin().read_line(&mut answer).is_err() {
        return false;
    }
    matches!(answer.trim(), "y" | "Y" | "yes")
}

/// The whole startup manifest flow, reading and writing state under `root`,
/// with the manifest fetch, the background spawn, and the interactive
/// confirmation injected so it can be exercised without a network, a real
/// detached process, or a real terminal.
///
/// `confirm` returns `None` when there is nobody to ask (no controlling
/// terminal) and `Some(accepted)` otherwise, folding the non-TTY gate and the
/// `[y/N]` prompt into one seam a test can answer deterministically.
///
/// Never fatal, and never blocking on anything but the interactive prompt
/// (itself gated on a real terminal). A model one release behind still works,
/// so every failure path here — offline, unparseable manifest, an unwritable
/// staging directory — leaves the existing model in place and returns.
fn check_manifest_at_startup_in(
    root: &Path,
    fetch: &dyn Fn() -> Option<String>,
    spawn: &dyn Fn(&crate::manifest::Manifest) -> Result<(), String>,
    confirm: &dyn Fn(&crate::manifest::Manifest, u32) -> Option<bool>,
) {
    // Anything a previous run verified gets installed first, before the engine
    // maps the model.
    match crate::downloader::swap_staged_in(root) {
        Ok(Some(version)) => eprintln!("plank: installed model manifest version {version}."),
        Ok(None) => {}
        Err(e) => eprintln!("plank: could not install the downloaded model: {e}"),
    }

    // A helper already at work needs no second opinion.
    if crate::downloader::running_in(root) {
        return;
    }
    if !check_due_in(root) {
        return;
    }
    note_last_run_outcome_in(root);
    let Some(remote) = fetch().and_then(|t| crate::manifest::parse(&t).ok()) else {
        return;
    };
    let installed = crate::manifest::read_at(&crate::manifest::installed_path_in(root));
    let size_of = |kind: &str| {
        let path = crate::manifest::local_path_for_in(root, kind)?;
        std::fs::metadata(path).map(|m| m.len()).ok()
    };
    match crate::manifest::decide(remote, installed.as_ref(), &size_of) {
        crate::manifest::Decision::UpToDate => {}
        crate::manifest::Decision::Adopt(m) => {
            // The files on disk are already this release; record that and say
            // nothing. Without this, every existing user is offered an 87 GB
            // re-download the day the manifest ships.
            let _ = std::fs::write(crate::manifest::installed_path_in(root), &m.raw);
        }
        crate::manifest::Decision::Offer { manifest, from } => {
            // Recorded unconditionally, before any early return below: this
            // only writes the job file for `/model download` to pick up
            // later, it does not itself start anything downloading (that
            // still needs `spawn`, below). Without this, a declined offer or
            // a non-TTY run leaves no job on disk, so `read_job_in` finds
            // nothing and `/model download` answers "No model download is
            // pending" forever — the version becomes permanently
            // unreachable even though the user asked for it explicitly.
            if let Err(e) = crate::downloader::write_job_in(root, &manifest) {
                eprintln!("plank: could not record the download job: {e}");
            }
            // First-run acquisition belongs to `ensure_model`, not the
            // manifest flow: with no installed manifest AND no model on disk,
            // starting a background download here would race the foreground
            // download `ensure_model` is about to start for the very same
            // file — nearly 2x the transfer for one first run. An `Adopt`
            // decision never reaches this arm, so it is unaffected.
            if from == 0 && !artifact_installed_in(root, "main") {
                return;
            }
            // Already declined (or the last attempt was cancelled or failed a
            // hash check): do not silently re-offer. `/model download` is how
            // the user restarts it on purpose.
            if read_declined_in(root) == Some(manifest.version) {
                return;
            }
            // A 93 GB transfer must not start unannounced, and there is no
            // one to ask on a piped or headless run.
            let Some(accepted) = confirm(&manifest, from) else {
                return;
            };
            if !accepted {
                record_declined_in(root, manifest.version);
                return;
            }
            match spawn(&manifest) {
                Ok(()) => {
                    eprintln!(
                        "plank: downloading it in the background. Alt-M or /model to cancel."
                    );
                }
                Err(e) => eprintln!("plank: could not start the background download: {e}"),
            }
        }
    }
}

/// The whole startup manifest flow: install anything staged, then decide
/// whether to start a background download.
///
/// Skips entirely when `model_path` is `Some`: a `-m <path>` user's model
/// never lives at the manifest's hardcoded `~/.plank` locations, so the size
/// check would always read "nothing installed" and both the swap and the
/// download decision would operate on a file the engine never loads from.
///
/// Never fatal, and never blocking except on the interactive download prompt
/// (itself gated on a real terminal).
pub fn check_manifest_at_startup(model_path: Option<&Path>) {
    check_manifest_at_startup_with(
        model_path,
        &crate::manifest::plank_dir(),
        &fetch_manifest,
        &crate::downloader::spawn_detached,
        &real_confirm,
    );
}

/// [`check_manifest_at_startup`] with every side effect injected, so a test
/// can drive the `model_path` skip, the manifest fetch, the background
/// spawn, and the interactive confirmation all deterministically.
fn check_manifest_at_startup_with(
    model_path: Option<&Path>,
    root: &Path,
    fetch: &dyn Fn() -> Option<String>,
    spawn: &dyn Fn(&crate::manifest::Manifest) -> Result<(), String>,
    confirm: &dyn Fn(&crate::manifest::Manifest, u32) -> Option<bool>,
) {
    if model_path.is_some() {
        return;
    }
    check_manifest_at_startup_in(root, fetch, spawn, confirm);
}

/// The real confirmation: `None` when there is no controlling terminal to ask
/// (a piped or headless run), `Some(answer)` from the blocking `[y/N]` prompt
/// otherwise.
fn real_confirm(manifest: &crate::manifest::Manifest, from: u32) -> Option<bool> {
    if !std::io::stdin().is_terminal() {
        return None;
    }
    Some(confirm_background_download(manifest, from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_hundred_unique_rotating_messages() {
        assert_eq!(MESSAGES.len(), 200);
        assert!(MESSAGES.iter().all(|m| !m.is_empty()));
        let unique: std::collections::HashSet<_> = MESSAGES.iter().collect();
        assert_eq!(unique.len(), MESSAGES.len(), "messages must be unique");
    }

    /// A terminal of `cols` x `rows`.
    fn term(cols: u16, rows: u16) -> Rect {
        Rect::new(0, 0, cols, rows)
    }

    #[test]
    fn progress_block_is_never_covered_by_the_playfield() {
        for rows in 5..60u16 {
            for cols in [20u16, 30, 80, 200] {
                let s = layout(term(cols, rows), true);
                // The three progress rows are distinct and in order.
                assert!(s.message.y < s.gauge.y, "{cols}x{rows}");
                assert!(s.gauge.y < s.stats.y, "{cols}x{rows}");
                if let Some(game) = s.game {
                    assert!(
                        game.y + game.height <= s.message.y,
                        "playfield overlaps the message at {cols}x{rows}"
                    );
                    let top = s.rule_top.expect("a playfield always gets its rules");
                    let bottom = s.rule_bottom.expect("both of them");
                    assert!(top.bottom() <= game.y, "top rule overlaps the field");
                    assert!(game.bottom() <= bottom.y, "bottom rule overlaps the field");
                    assert!(
                        bottom.bottom() <= s.message.y,
                        "bottom rule hits the message"
                    );
                }
            }
        }
    }

    #[test]
    fn game_is_suppressed_when_the_terminal_is_too_small() {
        // Too narrow, however tall.
        let narrow = layout(term(arcade::MIN_W - 1, 60), true);
        assert!(narrow.game.is_none());
        assert!(
            narrow.rule_top.is_none() && narrow.rule_bottom.is_none(),
            "no stray rules without a field"
        );
        // Too short, however wide: the progress block, the rule and MIN_H do
        // not fit.
        let short = layout(
            term(200, PROGRESS_ROWS + RULE_ROWS + arcade::MIN_H - 1),
            true,
        );
        assert!(short.game.is_none());
    }

    #[test]
    fn the_field_takes_every_row_the_progress_block_does_not() {
        // No logo competes for space any more: the field grows with the
        // terminal, which is what keeps the wall clear of the paddle.
        let tall = layout(term(80, 60), true);
        assert_eq!(tall.game.unwrap().height, 60 - PROGRESS_ROWS - RULES_ROWS);
        assert_eq!(tall.rule_top.unwrap().height, RULE_ROWS);
        assert_eq!(tall.rule_bottom.unwrap().height, RULE_ROWS);
        assert!(
            layout(term(80, 80), true).game.unwrap().height > tall.game.unwrap().height,
            "a taller terminal means a taller field"
        );
    }

    #[test]
    fn closing_the_game_hands_its_rows_back() {
        let closed = layout(term(80, 60), false);
        assert!(closed.game.is_none(), "the field is gone");
        assert!(
            closed.rule_top.is_none() && closed.rule_bottom.is_none(),
            "and so are its rules"
        );
        // The progress block is untouched and still in order.
        let open = layout(term(80, 60), true);
        assert!(closed.message.y < closed.gauge.y);
        assert!(closed.gauge.y < closed.stats.y);
        assert_eq!(closed.message.height, open.message.height);
    }

    #[test]
    fn the_completion_sound_falls_back_when_tada_is_missing() {
        let root = std::env::temp_dir().join(format!("plank-sound-{}", std::process::id()));
        let (user, system) = (root.join("user"), root.join("system"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&user).unwrap();
        std::fs::create_dir_all(&system).unwrap();
        let dirs = vec![user.clone(), system.clone()];

        // Nothing installed at all: silence, not a panic.
        assert!(find_sound(&TADA_SOUNDS, &dirs).is_none());

        // Recent macOS ships no Tada, so the next choice wins.
        std::fs::write(system.join("Glass.aiff"), b"").unwrap();
        assert_eq!(
            find_sound(&TADA_SOUNDS, &dirs),
            Some(system.join("Glass.aiff"))
        );
        std::fs::write(system.join("Hero.aiff"), b"").unwrap();
        assert_eq!(
            find_sound(&TADA_SOUNDS, &dirs),
            Some(system.join("Hero.aiff")),
            "Hero is preferred over Glass"
        );

        // A real Tada anywhere beats every fallback...
        std::fs::write(system.join("Tada.aiff"), b"").unwrap();
        assert_eq!(
            find_sound(&TADA_SOUNDS, &dirs),
            Some(system.join("Tada.aiff"))
        );
        // ...and the user's own copy beats the system one.
        std::fs::write(user.join("Tada.aiff"), b"").unwrap();
        assert_eq!(
            find_sound(&TADA_SOUNDS, &dirs),
            Some(user.join("Tada.aiff"))
        );

        // Preference is by name first, not by directory: a user-installed
        // Hero must not outrank a system Tada.
        std::fs::remove_file(user.join("Tada.aiff")).unwrap();
        std::fs::write(user.join("Hero.aiff"), b"").unwrap();
        assert_eq!(
            find_sound(&TADA_SOUNDS, &dirs),
            Some(system.join("Tada.aiff"))
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_eta_hides_once_the_unfilled_span_runs_out() {
        let bar = Rect::new(10, 4, 48, 1);
        let text = "12m 30s left";
        // Early on there is plenty of room to the right of the percent label.
        let slot = eta_slot(bar, 0.10, "10.0%", text).expect("room at 10%");
        assert_eq!(slot.width, 12);
        assert!(slot.y == bar.y && slot.height == 1);
        // It never sits over the centered percent label.
        let label_end = bar.x + (bar.width - 5) / 2 + 5;
        assert!(slot.x > label_end, "ETA overlaps the percent label");
        // Nor over the filled part of the bar.
        assert!(slot.x >= bar.x + 5, "ETA sits on filled bar");

        // As the bar fills the room shrinks and the ETA is dropped rather
        // than clipped.
        assert!(
            eta_slot(bar, 0.95, "95.0%", text).is_none(),
            "no room at 95%"
        );
        assert!(
            eta_slot(bar, 1.0, "100.0%", text).is_none(),
            "none when done"
        );
        // A narrow bar never has room at all.
        assert!(eta_slot(Rect::new(0, 0, 20, 1), 0.0, "0.0%", text).is_none());
    }

    #[test]
    fn the_eta_slot_stays_inside_the_bar_at_every_ratio() {
        let bar = Rect::new(3, 0, 48, 1);
        for pct in 0..=100 {
            let ratio = f64::from(pct) / 100.0;
            let label = format!("{:.1}%", ratio * 100.0);
            if let Some(slot) = eta_slot(bar, ratio, &label, "1h 2m left") {
                assert!(slot.x >= bar.x, "{pct}% ran off the left");
                assert!(
                    slot.x + slot.width <= bar.x + bar.width,
                    "{pct}% ran off the right"
                );
            }
        }
    }

    #[test]
    fn the_eta_measures_only_what_this_run_fetched() {
        // A resumed download starts with bytes already on disk. Counting them
        // as progress would report a huge rate and an ETA of almost nothing.
        let resumed = eta_secs(60_000_000_000, 80_000_000_000, 50_000_000_000, 10.0);
        // 10 GB in 10 s = 1 GB/s, 20 GB left => 20 s.
        assert_eq!(resumed.map(f64::round), Some(20.0));
        // Counting the pre-existing bytes would have said 3 s instead.
        let naive = eta_secs(60_000_000_000, 80_000_000_000, 0, 10.0);
        assert_eq!(naive.map(f64::round), Some(3.0));

        // Too early, nothing fetched yet, or already complete: no estimate.
        assert!(eta_secs(1_000, 80_000, 0, 0.1).is_none(), "too early");
        assert!(
            eta_secs(500, 80_000, 500, 30.0).is_none(),
            "nothing fetched"
        );
        assert!(eta_secs(80_000, 80_000, 0, 30.0).is_none(), "already done");
    }

    #[test]
    fn the_eta_coarsens_as_it_grows() {
        assert_eq!(format_eta(9.0), "9s left");
        assert_eq!(format_eta(90.0), "1m 30s left");
        assert_eq!(format_eta(3_600.0), "1h 0m left");
        assert_eq!(format_eta(7_830.0), "2h 10m left");
        // Absurd inputs must not panic or wrap.
        assert_eq!(format_eta(-5.0), "0s left");
        assert!(format_eta(f64::INFINITY).ends_with("left"));
    }

    #[test]
    fn esc_closes_the_game_first_and_cancels_only_after() {
        let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        // Two presses to abort: the first only puts the game away, so reaching
        // for Esc mid-rally cannot lose an 87 GB download.
        assert_eq!(classify(esc, true), Action::CloseGame);
        assert_eq!(classify(esc, false), Action::Cancel);
    }

    #[test]
    fn q_and_ctrl_c_cancel_outright_even_mid_rally() {
        let q = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        for open in [true, false] {
            assert_eq!(classify(q, open), Action::Cancel, "q, game open: {open}");
            assert_eq!(classify(ctrl_c, open), Action::Cancel);
        }
        // Plain 'c' is a game key, not a cancel.
        assert_eq!(
            classify(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE), true),
            Action::Play
        );
    }

    #[test]
    fn paddle_keys_reach_the_game() {
        for code in [KeyCode::Left, KeyCode::Right, KeyCode::Char(' ')] {
            let key = KeyEvent::new(code, KeyModifiers::NONE);
            assert_eq!(classify(key, true), Action::Play, "{code:?}");
        }
    }

    #[test]
    fn paddle_keys_are_accepted_by_the_game() {
        let mut game = Breakout::new(1);
        for code in [KeyCode::Left, KeyCode::Right] {
            let key = KeyEvent::new(code, KeyModifiers::NONE);
            assert_eq!(classify(key, true), Action::Play);
            assert!(game.handle_key(key), "{code:?} should move the paddle");
        }
    }

    #[test]
    fn url_points_at_the_flash_gguf() {
        let url = model_url();
        assert!(url.starts_with("https://huggingface.co/"));
        assert!(url.contains(".gguf"));
    }

    #[test]
    fn default_path_is_under_plank() {
        assert!(default_model_path().ends_with(".plank/ds4flash.gguf"));
    }

    #[test]
    fn dspark_path_sits_beside_the_main_model() {
        let main = default_model_path();
        let spark = default_dspark_path();
        assert_eq!(main.parent(), spark.parent());
        assert!(spark.ends_with(".plank/ds4flash.dspark.gguf"));
    }

    #[test]
    fn dspark_url_points_at_the_support_gguf() {
        let url = dspark_url();
        assert!(url.contains(DSPARK_FILE));
        assert!(
            url.contains("Vision-Exp"),
            "must be the vision-experimental support: {url}"
        );
    }

    #[test]
    fn support_resolution_is_skipped_unless_dspark_was_asked_for() {
        // --dspark-off: the resolver must not touch mtp_path, and so must never
        // reach the filesystem or a prompt.
        let mut e = crate::config::EngineTuning {
            dspark: false,
            ..crate::config::EngineTuning::default()
        };
        assert!(ensure_dspark_support(&mut e).is_ok());
        assert_eq!(e.mtp_path, None);
    }

    #[test]
    fn an_explicit_mtp_path_wins_over_the_default() {
        // --mtp is also how a legacy one-stage MTP drafter is supplied, so it
        // must survive --dspark untouched rather than being replaced by the
        // DSpark default.
        let mut e = crate::config::EngineTuning {
            dspark: true,
            mtp_path: Some(PathBuf::from("/somewhere/custom-drafter.gguf")),
            ..crate::config::EngineTuning::default()
        };
        assert!(ensure_dspark_support(&mut e).is_ok());
        assert_eq!(
            e.mtp_path,
            Some(PathBuf::from("/somewhere/custom-drafter.gguf"))
        );
    }

    #[test]
    fn gb_conversion() {
        assert!((gb(1_500_000_000) - 1.5).abs() < 1e-9);
    }

    /// A well-formed manifest naming only `main`, at `version` and `bytes`.
    fn manifest_text(version: u32, bytes: u64) -> String {
        format!(
            r#"{{"version":{version},"released":"t","notes":"","files":{{
                "main": {{ "name": "m.gguf", "url": "https://example.invalid/m", "bytes": {bytes}, "sha256": "{}" }}
            }}}}"#,
            "a".repeat(64)
        )
    }

    /// A `spawn` stub recording whether it was called, for asserting a path
    /// never reaches the background download.
    fn spy_spawn() -> (
        std::sync::Arc<std::sync::atomic::AtomicBool>,
        impl Fn(&crate::manifest::Manifest) -> Result<(), String>,
    ) {
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = called.clone();
        let spawn = move |_: &crate::manifest::Manifest| {
            flag.store(true, Ordering::Relaxed);
            Ok(())
        };
        (called, spawn)
    }

    #[test]
    fn an_offer_with_no_model_installed_does_not_spawn() {
        // Finding 1b: first-run acquisition belongs to `ensure_model`, not the
        // manifest flow. With no installed manifest and no `main` artifact on
        // disk, `decide` returns `Offer { from: 0 }` — starting a background
        // download here would race `ensure_model`'s own foreground download
        // of the very same file.
        let root = crate::downloader::tests::tempdir();
        let (called, spawn) = spy_spawn();
        let text = manifest_text(5, 100);
        // `from == 0` must return before ever consulting `confirm`, so a
        // confirm stub that panics if called doubles as proof of that.
        check_manifest_at_startup_in(&root, &|| Some(text.clone()), &spawn, &|_, _| {
            panic!("must not even ask on a bare first run")
        });
        assert!(
            !called.load(Ordering::Relaxed),
            "must not spawn a background download on a bare first run"
        );
    }

    #[test]
    fn an_explicit_model_path_skips_the_flow_entirely() {
        // Finding 2 (and the re-review's Fix D): a `-m /custom/path` user's
        // model never lives at the manifest's hardcoded `~/.plank`
        // locations. Passing a configured path must return before touching
        // `fetch`, `spawn`, or `confirm` at all. Every seam here panics if
        // invoked, so this test would fail if the `model_path.is_some()`
        // early return were ever removed or reordered past them — unlike
        // calling the real `check_manifest_at_startup`, whose `fetch_manifest`
        // is stubbed to `None` in test builds regardless of the skip, and so
        // could not tell the fix apart from its absence.
        let root = crate::downloader::tests::tempdir();
        check_manifest_at_startup_with(
            Some(Path::new("/some/custom/model.gguf")),
            &root,
            &|| panic!("must not fetch the manifest when -m is set"),
            &|_| panic!("must not spawn a download when -m is set"),
            &|_, _| panic!("must not prompt when -m is set"),
        );
    }

    #[test]
    fn a_declined_version_is_not_re_offered() {
        // Findings 1 and 5 (and the re-review's Fix D): once the user has
        // said no (or a prior attempt was cancelled or failed a hash check),
        // the same version must not come back on its own; only
        // `/model download` restarts it. `confirm` is stubbed to panic if
        // reached, so this test would fail if the declined check were
        // removed or the version comparison were wrong — unlike a stub that
        // merely relies on the test process having no controlling terminal,
        // which would mask that exact regression.
        let root = crate::downloader::tests::tempdir();
        let installed = manifest_text(3, 100);
        std::fs::write(crate::manifest::installed_path_in(&root), &installed).expect("installed");
        let remote_text = manifest_text(4, 200);
        std::fs::create_dir_all(crate::manifest::downloads_dir_in(&root)).expect("downloads dir");
        std::fs::write(declined_path_in(&root), "4\n").expect("declined marker");

        let (called, spawn) = spy_spawn();
        check_manifest_at_startup_in(&root, &|| Some(remote_text.clone()), &spawn, &|_, _| {
            panic!("a declined version must not even reach the confirm prompt")
        });
        assert!(
            !called.load(Ordering::Relaxed),
            "a declined version must not be re-offered"
        );
    }

    #[test]
    fn a_declined_offer_can_still_be_started_by_hand() {
        // Fix A: the stated requirement is that a declined version stays
        // reachable through `/model download`. Before the fix, the job file
        // that command reads was written only inside `spawn_detached`, so a
        // decline (or a non-TTY run, exercised below) left nothing for
        // `start_from_manifest_in` to find and it answered "No model
        // download is pending" forever.
        let root = crate::downloader::tests::tempdir();
        let installed = manifest_text(3, 100);
        std::fs::write(crate::manifest::installed_path_in(&root), &installed).expect("installed");
        let remote_text = manifest_text(4, 200);
        let (called, spawn) = spy_spawn();
        // `confirm` returns `Some(false)`: the user is asked and says no.
        check_manifest_at_startup_in(&root, &|| Some(remote_text.clone()), &spawn, &|_, _| {
            Some(false)
        });
        assert!(
            !called.load(Ordering::Relaxed),
            "a decline must not itself start the download"
        );

        let message = crate::downloader::start_from_manifest_in(&root);
        assert_eq!(
            message, "Downloading model manifest version 4 in the background.",
            "a declined version must still be startable by hand: {message}"
        );
    }

    #[test]
    fn a_headless_offer_can_still_be_started_by_hand() {
        // Fix A, the non-TTY half: `confirm` returning `None` (no controlling
        // terminal) must not, by itself, leave the job unrecorded either.
        let root = crate::downloader::tests::tempdir();
        let installed = manifest_text(3, 100);
        std::fs::write(crate::manifest::installed_path_in(&root), &installed).expect("installed");
        let remote_text = manifest_text(4, 200);
        let (called, spawn) = spy_spawn();
        check_manifest_at_startup_in(&root, &|| Some(remote_text.clone()), &spawn, &|_, _| None);
        assert!(
            !called.load(Ordering::Relaxed),
            "a headless run must not itself start the download"
        );

        let message = crate::downloader::start_from_manifest_in(&root);
        assert_eq!(
            message, "Downloading model manifest version 4 in the background.",
            "a headless offer must still be startable by hand: {message}"
        );
    }

    /// A `State` for a finished job at `version`, in the given `phase`, with
    /// `error` set the way [`crate::downloader::finish_cancel`] (private to
    /// `downloader.rs`) would for a cancel of that kind.
    fn finished_state(
        version: u32,
        phase: crate::downloader::Phase,
        error: Option<&str>,
    ) -> crate::downloader::State {
        crate::downloader::State {
            pid: std::process::id(),
            version,
            current: String::new(),
            index: 0,
            of: 1,
            done_bytes: 0,
            total_bytes: 0,
            rate_bps: 0,
            phase,
            error: error.map(str::to_string),
            updated: crate::downloader::now_epoch(),
        }
    }

    #[test]
    fn a_keep_cancelled_version_is_still_offered_next_run() {
        // Fix C: `Cancel::Keep` means "stop now, resume later" — that is the
        // entire point of offering it separately from `Delete`. So it must
        // not be recorded as a decline, unlike a Delete cancel (next test).
        let root = crate::downloader::tests::tempdir();
        let installed = manifest_text(3, 100);
        std::fs::write(crate::manifest::installed_path_in(&root), &installed).expect("installed");
        let remote_text = manifest_text(4, 200);
        std::fs::create_dir_all(crate::manifest::downloads_dir_in(&root)).expect("downloads dir");
        crate::downloader::write_state_in(
            &root,
            &finished_state(4, crate::downloader::Phase::Cancelled, Some("keep")),
        )
        .expect("seed a Keep-cancelled state");

        let (called, spawn) = spy_spawn();
        check_manifest_at_startup_in(&root, &|| Some(remote_text.clone()), &spawn, &|_, _| {
            Some(true)
        });
        assert!(
            called.load(Ordering::Relaxed),
            "a Keep-cancelled version must still be offered on the next run"
        );
    }

    #[test]
    fn a_delete_cancelled_version_is_not_re_offered() {
        let root = crate::downloader::tests::tempdir();
        let installed = manifest_text(3, 100);
        std::fs::write(crate::manifest::installed_path_in(&root), &installed).expect("installed");
        let remote_text = manifest_text(4, 200);
        std::fs::create_dir_all(crate::manifest::downloads_dir_in(&root)).expect("downloads dir");
        crate::downloader::write_state_in(
            &root,
            &finished_state(4, crate::downloader::Phase::Cancelled, Some("delete")),
        )
        .expect("seed a Delete-cancelled state");

        let (called, spawn) = spy_spawn();
        check_manifest_at_startup_in(&root, &|| Some(remote_text.clone()), &spawn, &|_, _| {
            panic!("a Delete-cancelled version must not even reach the confirm prompt")
        });
        assert!(
            !called.load(Ordering::Relaxed),
            "a Delete-cancelled version must not be re-offered"
        );
    }

    #[test]
    fn an_adopt_writes_the_installed_manifest_and_spawns_nothing() {
        // Adopt-on-first-sight: the artifact on disk already matches the
        // remote manifest's size, so it is recorded as installed silently —
        // no prompt, no spawn.
        let root = crate::downloader::tests::tempdir();
        std::fs::create_dir_all(&root).expect("root");
        std::fs::write(
            crate::manifest::local_path_for_in(&root, "main").expect("main path"),
            vec![0u8; 100],
        )
        .expect("pre-existing model file");
        let text = manifest_text(3, 100);

        let (called, spawn) = spy_spawn();
        check_manifest_at_startup_in(&root, &|| Some(text.clone()), &spawn, &|_, _| {
            panic!("an adopt must not reach the confirm prompt")
        });
        assert!(!called.load(Ordering::Relaxed), "an adopt must not spawn");
        let recorded = crate::manifest::read_at(&crate::manifest::installed_path_in(&root))
            .expect("installed manifest recorded");
        assert_eq!(recorded.version, 3);
    }
}
