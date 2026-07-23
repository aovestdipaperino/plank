// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Model auto-download with a playful progress display.
//!
//! When no `-m` is given, plank looks for `~/.plank/ds4flash.gguf`. If it is
//! missing, it offers to fetch the `DeepSeek` V4 Flash GGUF from Hugging Face
//! (`huggingface.co`),
//! streaming a magenta progress bar with a rotating series of messages.

use std::fs::{File, OpenOptions};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Gauge, Paragraph};

use crate::{logo, tui};

/// Hugging Face repository hosting the GGUF files.
const REPO: &str = "antirez/deepseek-v4-gguf";
/// The recommended Flash quant (~81 GB) for 96–128 GB machines.
const FILE: &str = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

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

/// Hugging Face download URL for the default Flash GGUF.
#[must_use]
pub fn model_url() -> String {
    format!("https://huggingface.co/{REPO}/resolve/main/{FILE}")
}

/// Ensures a model file exists at `path`, offering to download it if missing.
///
/// # Errors
/// Returns an error string when the user declines, when stdin is not a
/// terminal (so no prompt is possible), or when the download fails.
pub fn ensure_model(path: &Path) -> Result<(), String> {
    if path.exists() {
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
        eprintln!("plank can download DeepSeek V4 Flash (~81 GB) from Hugging Face:");
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
    eprintln!("Model ready at {}.", dest.display());
    Ok(())
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

/// Runs the full-screen download UI until the worker finishes or the user cancels.
fn run_ui(
    terminal: &mut ratatui::DefaultTerminal,
    worker: &JoinHandle<Result<(), String>>,
    cancel: &AtomicBool,
    part: &Path,
    total: Option<u64>,
) -> Result<(), String> {
    let start = Instant::now();
    let mut msg = 0usize;
    let mut last_rotate = Instant::now();
    // Render the logo once (it never changes), sized to fill most of the
    // splash. The image is portrait (~0.87:1), so one output row spans ~0.57
    // columns; a width of ~1.74x the target row count fills that many rows.
    // Pick a width that fills ~75% of the height and still fits the width.
    let (rows, cols) = terminal
        .size()
        .map_or((24u16, 80u16), |s| (s.height, s.width));
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let logo_width = {
        let by_height = (f32::from(rows) * 0.75 * 1.74) as u32;
        // Never let the logo crowd out the message/gauge/stats block (6 rows).
        let fit = (f32::from(rows.saturating_sub(7)) * 1.74) as u32;
        by_height
            .min(fit)
            .min(u32::from(cols).saturating_sub(2))
            .max(24)
    };
    let logo = tui::ansi_to_lines(&logo::art(logo_width));
    loop {
        if worker.is_finished() {
            let _ = terminal.draw(|f| draw(f, &logo, part, total, msg, start));
            // The caller joins the worker and surfaces its error, if any.
            return Ok(());
        }
        // Ctrl-C / Esc / q cancels; the poll timeout also paces the frames.
        if event::poll(Duration::from_millis(250)).map_err(|e| e.to_string())?
            && let Ok(Event::Key(k)) = event::read()
            && k.kind == KeyEventKind::Press
            && (matches!(k.code, KeyCode::Esc | KeyCode::Char('q'))
                || (matches!(k.code, KeyCode::Char('c'))
                    && k.modifiers.contains(KeyModifiers::CONTROL)))
        {
            cancel.store(true, Ordering::Relaxed);
            return Err("download cancelled".to_string());
        }
        if last_rotate.elapsed() >= Duration::from_secs(3) {
            msg += 1;
            last_rotate = Instant::now();
        }
        let _ = terminal.draw(|f| draw(f, &logo, part, total, msg, start));
    }
}

/// Draws one download frame: centered logo, red rotating message, gauge, stats.
fn draw(
    frame: &mut Frame,
    logo: &[Line<'static>],
    part: &Path,
    total: Option<u64>,
    msg: usize,
    start: Instant,
) {
    let current = std::fs::metadata(part).map_or(0, |m| m.len());
    let elapsed = start.elapsed().as_secs_f64();
    #[allow(clippy::cast_precision_loss)]
    let speed = if elapsed > 0.0 {
        current as f64 / elapsed / 1_000_000.0
    } else {
        0.0
    };

    let area = frame.area();
    let logo_h = u16::try_from(logo.len()).unwrap_or(0);
    // Center the logo + progress block vertically.
    let rows = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(logo_h), // logo
        Constraint::Length(1),      // spacer
        Constraint::Length(1),      // message
        Constraint::Length(1),      // spacer
        Constraint::Length(1),      // gauge
        Constraint::Length(1),      // spacer
        Constraint::Length(1),      // stats
        Constraint::Fill(1),
    ])
    .split(area);

    // Logo, horizontally centered at its rendered width.
    let logo_w = u16::try_from(logo.iter().map(Line::width).max().unwrap_or(0)).unwrap_or(0);
    let logo_area = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Length(logo_w),
        Constraint::Fill(1),
    ])
    .split(rows[1])[1];
    frame.render_widget(Paragraph::new(logo.to_vec()), logo_area);

    let red = Style::default().fg(Color::Red).add_modifier(Modifier::BOLD);
    let message = Paragraph::new(MESSAGES[msg % MESSAGES.len()])
        .style(red)
        .alignment(Alignment::Center);
    frame.render_widget(message, rows[3]);

    // A centered, fixed-width gauge.
    let gauge_area = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Length(48),
        Constraint::Fill(1),
    ])
    .split(rows[5])[1];
    #[allow(clippy::cast_precision_loss)]
    let ratio = match total.filter(|t| *t > 0) {
        Some(t) => (current as f64 / t as f64).clamp(0.0, 1.0),
        None => 0.0,
    };
    let label = total.filter(|t| *t > 0).map_or_else(
        || "downloading...".to_string(),
        |_| format!("{:.1}%", ratio * 100.0),
    );
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Red).bg(Color::Indexed(238)))
        .ratio(ratio)
        .label(label);
    frame.render_widget(gauge, gauge_area);

    let stats = match total {
        Some(t) => format!("{:.1} / {:.1} GB   {speed:.0} MB/s", gb(current), gb(t)),
        None => format!("{:.1} GB   {speed:.0} MB/s", gb(current)),
    };
    let stats_line = Paragraph::new(stats)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center);
    frame.render_widget(stats_line, rows[7]);
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

#[allow(clippy::cast_precision_loss)]
fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000_000.0
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
    fn gb_conversion() {
        assert!((gb(1_500_000_000) - 1.5).abs() < 1e-9);
    }
}
