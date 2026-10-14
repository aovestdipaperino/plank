// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `/buddy on|off|install`: an animated buddy standing on the rule above the
//! prompt while plank is idle.
//!
//! It does nothing, which is the point. Only terminals that speak the kitty
//! graphics protocol can show it: kitty and Ghostty, which also implement its
//! Unicode placeholders, and Warp, which does not ([`Mode`]). Everywhere else
//! the command says so and leaves the screen alone.
//!
//! A buddy is a folder of `.mp4` clips in the public
//! [`REPO`](https://github.com/aovestdipaperino/plank-buddies) repository.
//! A user has one: installing another replaces it ([`install`]). It lives in
//! `~/.plank/buddy/`, with its folder name in `.name`. `/buddy on` with none
//! installed, and `/buddy install`, offer the repository's folders as a menu
//! ([`Outcome::Pick`]).
//!
//! Turning it on or off is remembered across runs by the presence of
//! `~/.plank/buddy.on` ([`STATE_FILE`]), and the TUI puts a remembered buddy
//! back at startup ([`restore`]). The file sits outside [`CLIP_DIR`] because
//! an install replaces that directory whole.
//!
//! The clips play one after the other ([`next_clip`]): a clip whose name ends
//! in a letter is picked three times out of four, at random among the letter
//! clips; otherwise the next clip whose name ends in a digit plays, in
//! rotation. Clips are decoded once by `ffmpeg` on a background thread, their
//! white background keyed out ([`key_background`]) and then halved in size,
//! and cached as PNG frames under `~/.plank/buddy/.frames/`, so only the first
//! showing after an install pays for it.
//!
//! The image is placed with the protocol's Unicode placeholders rather than at
//! a cursor position. Transmitting it creates a *virtual* placement
//! ([`transmit_sequence`]); it then shows wherever the screen holds U+10EEEE
//! cells whose foreground color is the image id and whose two combining marks
//! name the cell's row and column of the picture ([`render`]). Those are
//! ordinary cells in ratatui's buffer, so the frame diff moves, covers and
//! erases the picture like any text: a pane drawn over it hides it, a frame
//! without it drops it, and nothing outside the buffer can go stale. A clip
//! plays by double buffering: the next frame is transmitted under the image id
//! the screen is *not* showing, and the same draw then repaints the cells with
//! that id, so a frame is never seen half-replaced.
//!
//! Warp draws kitty images only at the cursor ([`Mode::Direct`]), outside the
//! cell grid, so there the frame diff cannot hide the picture. The draw paints
//! marker cells where it would stand instead ([`render`]), and once the frame
//! is on screen [`after_draw`] places the image there only if every marker
//! survived: a menu, pane or report drawn over any of them means the picture
//! would cover it, and it is deleted, as it is on any frame [`render`] was not
//! part of. Every change is a fresh image at the position followed by deleting
//! the previous one, the two operations Warp is known to honour.
//!
//! Only the idle repaint calls [`render`], which is what keeps the buddy off
//! the screen while a turn runs.
//!
//! `/buddy sound on` plays each clip's soundtrack with it, remembered like
//! the buddy itself (`~/.plank/buddy.sound`, [`SOUND_FILE`]). The decoder
//! extracts it once next to the frames as a WAV, and the system's own player
//! plays it ([`voice`]): `afplay` on macOS, else `paplay`, `aplay` or
//! `ffplay`, whichever is installed, so nothing is linked for it. A clip's
//! sound starts with its first frame and stops once the buddy has not been
//! drawn for [`voice::GRACE`]: a turn starting, a pane over it, `/buddy off`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

/// The repository whose top-level folders are the buddies on offer.
pub const REPO: &str = "aovestdipaperino/plank-buddies";

/// The installed buddy's directory, under the plank home.
pub const CLIP_DIR: &str = "buddy";

/// The file, under the plank home, whose presence means the buddy is on.
/// Not a setting: `/config` saves its merged settings into the project's
/// file, which would pin a stale value there and outrank `/buddy` in that
/// folder from then on.
pub const STATE_FILE: &str = "buddy.on";

/// The file, under the plank home, whose presence means the clips play with
/// their sound, for the same reason as [`STATE_FILE`].
pub const SOUND_FILE: &str = "buddy.sound";

/// Where an install downloads before it replaces [`CLIP_DIR`], under the plank
/// home, so a failed download leaves the old buddy in place.
const STAGING_DIR: &str = ".buddy-staging";

/// The file in [`CLIP_DIR`] naming the folder the buddy came from.
const NAME_FILE: &str = ".name";

/// Where decoded frames are cached, inside [`CLIP_DIR`].
const FRAME_CACHE: &str = ".frames";

/// The marker a fully decoded clip's cache directory holds; a directory
/// without it is a decode that was interrupted, and is redone.
const DONE_MARKER: &str = "done";

/// A clip's soundtrack, extracted into its cache directory.
const AUDIO_FILE: &str = "audio.wav";

/// The marker a clip's cache directory holds when the clip has no sound, so
/// extraction is not retried at every start.
const SILENT_MARKER: &str = "silent";

/// Bumped whenever decoding changes what a cached frame holds, so frames
/// cached by an older plank are decoded afresh. 2: keyed at full size.
const DECODE_VERSION: u32 = 2;

/// Playback rate. Clips are sampled down to it when decoded, which keeps the
/// bytes sent to the terminal per second modest.
pub const FPS: u32 = 12;

/// Size of the box the picture is fit into, in cells. With cells about twice
/// as tall as wide this box is about 6:5, the shape of the clips.
pub const COLS: u16 = 24;
/// See [`COLS`].
pub const ROWS: u16 = 10;

/// Columns of bare rule kept to the buddy's right.
const RIGHT_GAP: u16 = 2;

/// The kitty image ids a clip alternates between. Placeholder cells carry the
/// id as their 24-bit foreground color, so each must fit in 24 bits; the only
/// other kitty image plank sends is the banner logo (`logo::KITTY_ID`).
const CLIP_IDS: [u32; 2] = [0x00B0_DD1F, 0x00B0_DD20];

/// The placeholder character every picture cell holds.
pub(crate) const PLACEHOLDER: char = '\u{10EEEE}';

/// Base64 bytes per transmission chunk, the protocol's maximum.
const CHUNK: usize = 4096;

/// The first entries of kitty's `rowcolumn-diacritics.txt`: the combining mark
/// at index `n` names row (or column) `n` of the picture. Only as many as the
/// larger side of the box are needed.
pub(crate) const DIACRITICS: [char; 24] = [
    '\u{0305}', '\u{030D}', '\u{030E}', '\u{0310}', '\u{0312}', '\u{033D}', '\u{033E}', '\u{033F}',
    '\u{0346}', '\u{034A}', '\u{034B}', '\u{034C}', '\u{0350}', '\u{0351}', '\u{0352}', '\u{0357}',
    '\u{035B}', '\u{0363}', '\u{0364}', '\u{0365}', '\u{0366}', '\u{0367}', '\u{0368}', '\u{0369}',
];

/// A pixel whose every channel is at or above this, and which the border can
/// reach through such pixels, is background ([`key_background`]).
const BACKGROUND_THRESHOLD: u8 = 228;

/// The symbol [`Mode::Direct`] marks the picture's cells with: blank, and
/// drawn by nothing else in plank but the banner logo's own Warp cells
/// (`logo::after_draw`, told apart by their colour), so an overlay that lands
/// on a cell is seen.
pub(crate) const MARKER: &str = "\u{2800}";

/// Whether the buddy is on in this run; [`STATE_FILE`] carries it to the next.
static SHOWN: AtomicBool = AtomicBool::new(false);

/// Whether clips play with their sound; [`SOUND_FILE`] carries it to the next
/// run.
static SOUND: AtomicBool = AtomicBool::new(false);

/// Whether the running buddy is in [`Mode::Direct`] (meaningful while on).
static DIRECT: AtomicBool = AtomicBool::new(false);

/// Set by a finished install; the next [`render`], on the UI thread, clears
/// the old buddy off the screen and starts the new one.
static RELOAD: AtomicBool = AtomicBool::new(false);

/// What an install in progress, or the last one that failed, has to say.
static INSTALL_NOTE: Mutex<Option<String>> = Mutex::new(None);

/// The clip player, present while the installed buddy has clips to play.
static PLAYER: Mutex<Option<Player>> = Mutex::new(None);

/// Whether the buddy is on.
#[must_use]
pub fn shown() -> bool {
    SHOWN.load(Ordering::Relaxed)
}

/// Whether a clip is playing, so the idle loop should redraw at the animation
/// tick rather than its idle rate.
#[must_use]
pub fn animating() -> bool {
    shown()
        && PLAYER
            .lock()
            .ok()
            .and_then(|p| p.as_ref().map(Player::has_clips))
            .unwrap_or(false)
}

/// How the terminal is told where the picture goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Unicode placeholder cells (kitty, Ghostty): the picture is part of the
    /// cell grid.
    Placeholders,
    /// An image placed at the cursor (Warp), kept in step with the frame by
    /// [`after_draw`].
    Direct,
}

fn mode() -> Mode {
    if DIRECT.load(Ordering::Relaxed) {
        Mode::Direct
    } else {
        Mode::Placeholders
    }
}

/// How this process's terminal can show the buddy, if at all; see
/// [`supported_by`].
#[must_use]
pub fn supported() -> Option<Mode> {
    let var = |k| std::env::var(k).ok();
    supported_by(
        var("TERM").as_deref(),
        var("TERM_PROGRAM").as_deref(),
        std::env::var_os("KITTY_WINDOW_ID").is_some(),
        std::env::var_os("TMUX").is_some(),
    )
}

/// Kitty or Ghostty, which implement Unicode placeholders, or Warp, which
/// draws kitty images only at the cursor; never inside tmux, which would
/// swallow the graphics escapes (and leaks `KITTY_WINDOW_ID` into its panes,
/// so that variable alone proves nothing).
#[must_use]
pub fn supported_by(
    term: Option<&str>,
    term_program: Option<&str>,
    kitty_window: bool,
    tmux: bool,
) -> Option<Mode> {
    if tmux {
        return None;
    }
    let term = term.unwrap_or_default();
    let program = term_program.unwrap_or_default().to_ascii_lowercase();
    if kitty_window
        || term == "xterm-kitty"
        || term == "xterm-ghostty"
        || program == "kitty"
        || program == "ghostty"
    {
        Some(Mode::Placeholders)
    } else if program == "warpterminal" {
        Some(Mode::Direct)
    } else {
        None
    }
}

const UNSUPPORTED: &str =
    "/buddy needs a terminal with kitty graphics (kitty, Ghostty or Warp, outside tmux)";

/// The escapes that hand `png` to the terminal as image `id` with a virtual
/// placement of [`COLS`] x [`ROWS`] cells (`U=1`), split into chunks as the
/// protocol requires. Sending to an id already in use replaces that image.
/// `q=2` keeps the terminal from answering: a reply would arrive on stdin as
/// keystrokes.
#[must_use]
pub fn transmit_sequence(id: u32, png: &[u8]) -> String {
    transmit_sequence_sized(id, png, COLS, ROWS)
}

/// [`transmit_sequence`] for a virtual placement of `cols` x `rows` cells.
#[must_use]
pub fn transmit_sequence_sized(id: u32, png: &[u8], cols: u16, rows: u16) -> String {
    let data = crate::tui::base64(png);
    let chunks: Vec<&str> = data
        .as_bytes()
        .chunks(CHUNK)
        .map(|c| std::str::from_utf8(c).unwrap_or_default())
        .collect();
    let mut out = String::with_capacity(data.len() + chunks.len() * 16 + 64);
    for (i, chunk) in chunks.iter().enumerate() {
        let more = u8::from(i + 1 < chunks.len());
        // Writing to a `String` cannot fail.
        let _ = if i == 0 {
            write!(
                out,
                "\x1b_Ga=T,U=1,f=100,i={id},c={cols},r={rows},q=2,m={more};{chunk}\x1b\\"
            )
        } else {
            write!(out, "\x1b_Gm={more};{chunk}\x1b\\")
        };
    }
    out
}

/// The escape that frees image `id` and its placements.
#[must_use]
pub fn delete_sequence(id: u32) -> String {
    format!("\x1b_Ga=d,d=I,i={id},q=2\x1b\\")
}

pub(crate) fn write_to_terminal(seq: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    out.write_all(seq.as_bytes())?;
    out.flush()
}

/// What `/buddy` asks of the front end.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Show this line.
    Say(String),
    /// Offer `choices` (buddy names) with `question`, then pass the pick to
    /// [`chosen`].
    Pick {
        question: String,
        choices: Vec<String>,
    },
}

/// `/buddy [on|off|install [name]]`. Bare, it reports the state.
pub fn command(arg: &str) -> Outcome {
    let mut words = arg.split_whitespace();
    let verb = words.next().unwrap_or_default();
    let name = words.next();
    let installed = installed();
    match (verb, name) {
        ("", None) => Outcome::Say(status(installed.as_deref())),
        ("off", None) => {
            if !shown() {
                return Outcome::Say("buddy already off".to_owned());
            }
            SHOWN.store(false, Ordering::Relaxed);
            clear_screen();
            Outcome::Say(format!("buddy dismissed{}", remember(false)))
        }
        ("on", None) => {
            if shown() {
                return Outcome::Say("buddy already on".to_owned());
            }
            let Some(mode) = supported() else {
                return Outcome::Say(UNSUPPORTED.to_owned());
            };
            match installed {
                Some(name) => Outcome::Say(format!(
                    "{}{}",
                    turn_on(mode, &format!("{name} reporting for duty")),
                    remember(true)
                )),
                None => pick("No buddy installed yet. Which one?"),
            }
        }
        ("install", None) => pick(&match installed {
            Some(name) => format!("Which buddy should replace {name}?"),
            None => "Which buddy?".to_owned(),
        }),
        ("install", Some(name)) => Outcome::Say(chosen(name)),
        ("sound", None) => Outcome::Say(format!(
            "buddy sound {} (/buddy sound on|off)",
            if SOUND.load(Ordering::Relaxed) {
                "on"
            } else {
                "off"
            }
        )),
        ("sound", Some(word @ ("on" | "off"))) if words.next().is_none() => {
            let on = word == "on";
            SOUND.store(on, Ordering::Relaxed);
            if !on {
                voice::hush();
            }
            let note = match remember_file(&crate::home::plank_home(), SOUND_FILE, on) {
                Ok(()) => String::new(),
                Err(e) => format!(" (not remembered for the next run: {e})"),
            };
            Outcome::Say(format!("buddy sound {word}{note}"))
        }
        _ => Outcome::Say(format!(
            "/buddy: expected on|off|install [name]|sound on|off, got `{arg}`"
        )),
    }
}

/// The bare `/buddy` line.
fn status(installed: Option<&str>) -> String {
    let state = if shown() { "on" } else { "off" };
    let which = installed.map_or_else(|| "none installed".to_owned(), |n| format!("{n} installed"));
    let note = INSTALL_NOTE
        .lock()
        .ok()
        .and_then(|n| n.clone())
        .map(|n| format!("; {n}"))
        .unwrap_or_default();
    let sound = if SOUND.load(Ordering::Relaxed) {
        "on"
    } else {
        "off"
    };
    format!("buddy {state}, {which}{note}, sound {sound} (/buddy on|off|install|sound)")
}

/// The menu of the repository's buddies, or why there is none.
fn pick(question: &str) -> Outcome {
    match list_available() {
        Ok(choices) if choices.is_empty() => Outcome::Say(format!("no buddies in {REPO} yet")),
        Ok(choices) => Outcome::Pick {
            question: question.to_owned(),
            choices,
        },
        Err(e) => Outcome::Say(format!("/buddy: cannot list {REPO}: {e}")),
    }
}

/// Installs the buddy `name` from the repository in the background, replacing
/// the installed one, and turns the buddy on where the terminal can show it.
pub fn chosen(name: &str) -> String {
    if !valid_name(name) {
        return format!("/buddy: `{name}` is not a buddy name");
    }
    if let Ok(mut note) = INSTALL_NOTE.lock() {
        if note.as_deref().is_some_and(|n| n.starts_with("installing")) {
            return format!(
                "/buddy: {}; wait for it to finish",
                note.clone().unwrap_or_default()
            );
        }
        *note = Some(format!("installing {name}"));
    }
    let home = crate::home::plank_home();
    let owned = name.to_owned();
    std::thread::spawn(move || {
        let result = install(&owned, &home);
        if let Ok(mut note) = INSTALL_NOTE.lock() {
            *note = result
                .err()
                .map(|e| format!("installing {owned} failed: {e}"));
        }
    });
    let started = format!("downloading {name} from {REPO}; it appears once ready");
    let reply = match supported() {
        Some(mode) if !shown() => turn_on(mode, &started),
        Some(_) => started,
        None => format!("{started}, in a terminal that can show it ({UNSUPPORTED})"),
    };
    format!("{reply}{}", remember(true))
}

/// Puts the buddy back on at TUI startup when the last run left it on, the
/// terminal can show it, and one is installed. Silent: the user already
/// asked for it.
pub fn restore() {
    let home = crate::home::plank_home();
    SOUND.store(home.join(SOUND_FILE).is_file(), Ordering::Relaxed);
    if shown() || !remembered_in(&home) || installed().is_none() {
        return;
    }
    if let Some(mode) = supported() {
        let _ = turn_on(mode, "");
    }
}

/// Records `on` for the next run, returning a note for the command's reply
/// when it could not be (empty otherwise).
fn remember(on: bool) -> String {
    match remember_in(&crate::home::plank_home(), on) {
        Ok(()) => String::new(),
        Err(e) => format!(" (not remembered for the next run: {e})"),
    }
}

/// [`remember`] under the plank home `home`.
fn remember_in(home: &Path, on: bool) -> std::io::Result<()> {
    remember_file(home, STATE_FILE, on)
}

/// Creates (`on`) or removes the marker `file` under the plank home `home`.
fn remember_file(home: &Path, file: &str, on: bool) -> std::io::Result<()> {
    let path = home.join(file);
    if on {
        std::fs::create_dir_all(home)?;
        std::fs::write(path, "")
    } else {
        match std::fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }
}

/// Whether the last run under the plank home `home` left the buddy on.
fn remembered_in(home: &Path) -> bool {
    home.join(STATE_FILE).is_file()
}

/// Turns the buddy on in `mode`, returning `reply` with any note about the
/// clips appended.
fn turn_on(mode: Mode, reply: &str) -> String {
    DIRECT.store(mode == Mode::Direct, Ordering::Relaxed);
    SHOWN.store(true, Ordering::Relaxed);
    format!("{reply}{}", start_player())
}

/// A folder name the repository may use and an install may write: ASCII
/// letters, digits, `-` and `_`.
#[must_use]
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// The installed buddy's name, when there are clips in [`CLIP_DIR`].
#[must_use]
pub fn installed() -> Option<String> {
    installed_in(&crate::home::plank_home().join(CLIP_DIR))
}

fn installed_in(dir: &Path) -> Option<String> {
    if list_videos(dir).is_empty() {
        return None;
    }
    let name = std::fs::read_to_string(dir.join(NAME_FILE)).unwrap_or_default();
    let name = name.trim();
    Some(if name.is_empty() { "a buddy" } else { name }.to_owned())
}

#[cfg(not(test))]
fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .build()
        .new_agent()
}

/// The repository listing at `path` (the root when empty), as `(name, type,
/// download_url)` triples.
#[cfg(not(test))]
fn list_repo(path: &str) -> Result<Vec<(String, String, Option<String>)>, String> {
    let url = format!("https://api.github.com/repos/{REPO}/contents/{path}");
    let mut resp = agent(Duration::from_secs(10))
        .get(&url)
        .header("User-Agent", concat!("plank/", env!("CARGO_PKG_VERSION")))
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| e.to_string())?;
    let body = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| e.to_string())?;
    parse_listing(&body)
}

#[cfg(test)]
fn list_repo(_path: &str) -> Result<Vec<(String, String, Option<String>)>, String> {
    Err("no network in tests".to_owned())
}

/// Parses a GitHub contents listing into `(name, type, download_url)`.
fn parse_listing(body: &str) -> Result<Vec<(String, String, Option<String>)>, String> {
    let value: serde_json::Value = serde_json::from_str(body).map_err(|e| e.to_string())?;
    let entries = value
        .as_array()
        .ok_or_else(|| "the listing is not a folder".to_owned())?;
    Ok(entries
        .iter()
        .filter_map(|e| {
            let name = e.get("name")?.as_str()?.to_owned();
            let kind = e.get("type")?.as_str()?.to_owned();
            let url = e
                .get("download_url")
                .and_then(|u| u.as_str())
                .map(str::to_owned);
            Some((name, kind, url))
        })
        .collect())
}

/// The buddies the repository offers: its top-level folders, by name.
fn list_available() -> Result<Vec<String>, String> {
    let mut names: Vec<String> = list_repo("")?
        .into_iter()
        .filter(|(name, kind, _)| kind == "dir" && valid_name(name))
        .map(|(name, _, _)| name)
        .collect();
    names.sort();
    Ok(names)
}

#[cfg(not(test))]
fn download(url: &str, to: &Path) -> Result<(), String> {
    let mut resp = agent(Duration::from_secs(300))
        .get(url)
        .header("User-Agent", concat!("plank/", env!("CARGO_PKG_VERSION")))
        .call()
        .map_err(|e| e.to_string())?;
    let mut file = std::fs::File::create(to).map_err(|e| e.to_string())?;
    std::io::copy(&mut resp.body_mut().as_reader(), &mut file).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
fn download(_url: &str, _to: &Path) -> Result<(), String> {
    Err("no network in tests".to_owned())
}

/// Downloads the buddy `name` into a staging directory under `home` and, once
/// every clip is there, puts it in place of the installed one. The UI thread
/// picks the change up at its next frame ([`RELOAD`]).
fn install(name: &str, home: &Path) -> Result<(), String> {
    let clips: Vec<(String, String)> = list_repo(name)?
        .into_iter()
        .filter(|(file, kind, _)| {
            kind == "file"
                && Path::new(file)
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("mp4"))
                && kind_of(Path::new(file)).is_some()
                && file
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
        .filter_map(|(file, _, url)| Some((file, url?)))
        .collect();
    if clips.is_empty() {
        return Err(format!("{name} has no clips"));
    }
    let staging = home.join(STAGING_DIR);
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
    let fetched = clips
        .iter()
        .try_for_each(|(file, url)| download(url, &staging.join(file)))
        .and_then(|()| std::fs::write(staging.join(NAME_FILE), name).map_err(|e| e.to_string()));
    if let Err(e) = fetched {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    replace_installed(&staging, &home.join(CLIP_DIR))?;
    RELOAD.store(true, Ordering::Relaxed);
    Ok(())
}

/// Moves `staging` to `dir`, removing whatever buddy was there (clips and
/// cached frames alike): a user has one buddy.
fn replace_installed(staging: &Path, dir: &Path) -> Result<(), String> {
    if dir.exists() {
        std::fs::remove_dir_all(dir).map_err(|e| format!("cannot remove the old buddy: {e}"))?;
    }
    std::fs::rename(staging, dir).map_err(|e| format!("cannot install: {e}"))
}

/// Starts playing the installed buddy unless it already plays, returning a
/// note for the command's reply (empty when there is nothing to say).
fn start_player() -> String {
    let dir = crate::home::plank_home().join(CLIP_DIR);
    let Ok(mut player) = PLAYER.lock() else {
        return String::new();
    };
    if player.is_some() {
        return String::new();
    }
    let videos = list_videos(&dir);
    if videos.is_empty() {
        return String::new();
    }
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_err()
    {
        return "; its clips need ffmpeg, which is not installed".to_owned();
    }
    let p = Player::new(seed());
    let (ready, done) = (Arc::clone(&p.clips), Arc::clone(&p.decoded));
    let cache = dir.join(FRAME_CACHE);
    std::thread::spawn(move || decode_all(&videos, &cache, &ready, &done));
    *player = Some(p);
    String::new()
}

/// Takes the buddy off the screen and forgets what was there, so the next
/// showing retransmits rather than trusting deleted images.
fn clear_screen() {
    voice::hush();
    if let Ok(mut player) = PLAYER.lock()
        && let Some(p) = player.as_mut()
    {
        p.stop();
    }
    if let Ok(mut direct) = DIRECT_STATE.lock() {
        direct.want = None;
        direct.placed = None;
    }
    let all: String = CLIP_IDS.into_iter().map(delete_sequence).collect();
    let _ = write_to_terminal(&all);
}

/// The `.mp4` files directly in `dir`, by name.
fn list_videos(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e.eq_ignore_ascii_case("mp4")) && kind_of(p).is_some()
        })
        .collect();
    out.sort();
    out
}

/// Which pool a clip belongs to, by the last character of its file stem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Ends in a digit: played in rotation, one time in four.
    Digit,
    /// Ends in a letter: picked at random, three times in four.
    Letter,
}

/// The [`Kind`] of the clip at `path`, or `None` when its stem ends in
/// neither a digit nor a letter.
#[must_use]
pub fn kind_of(path: &Path) -> Option<Kind> {
    let last = path.file_stem()?.to_str()?.chars().last()?;
    if last.is_ascii_digit() {
        Some(Kind::Digit)
    } else if last.is_alphabetic() {
        Some(Kind::Letter)
    } else {
        None
    }
}

/// A decoded clip: its kind, its frames, in order, and its sound, if any.
#[derive(Clone, Debug)]
struct Clip {
    kind: Kind,
    frames: Vec<PathBuf>,
    audio: Option<PathBuf>,
}

/// Decodes each video into the cache (or finds it there) into `ready`, then
/// sets `done`. A clip that fails to decode is skipped.
fn decode_all(videos: &[PathBuf], cache: &Path, ready: &Mutex<Vec<Clip>>, done: &AtomicBool) {
    for video in videos {
        let Some(kind) = kind_of(video) else {
            continue;
        };
        if let Some(frames) = decode(video, cache)
            && !frames.is_empty()
            && let Ok(mut clips) = ready.lock()
        {
            let audio = cache_dir_for(video, cache).and_then(|dir| extract_audio(video, &dir));
            clips.push(Clip {
                kind,
                frames,
                audio,
            });
        }
    }
    done.store(true, Ordering::Relaxed);
}

/// The cache directory for `video`: its stem, size and modification time, so
/// a replaced clip is decoded afresh.
fn cache_dir_for(video: &Path, cache: &Path) -> Option<PathBuf> {
    let meta = std::fs::metadata(video).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let stem = video.file_stem()?.to_str()?;
    Some(cache.join(format!("{stem}-{}-{mtime}-v{DECODE_VERSION}", meta.len())))
}

/// The frames of `video`, decoding them into the cache when they are not
/// already there. The background is keyed at the clip's full size and only
/// then halved ([`halve`]): halving first blurs a narrow gap of background
/// (between an arm and a body, into an open loop) shut, and the white behind
/// it would stay opaque, cut off from the border.
fn decode(video: &Path, cache: &Path) -> Option<Vec<PathBuf>> {
    let dir = cache_dir_for(video, cache)?;
    if !dir.join(DONE_MARKER).exists() {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok()?;
        let status = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-y", "-i"])
            .arg(video)
            .args(["-vf", &format!("fps={FPS}")])
            .arg(dir.join("%05d.png"))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        for frame in frames_in(&dir) {
            let keyed = image::open(&frame)
                .ok()
                .map(|img| halve(&key_background(img.to_rgba8())));
            if let Some(img) = keyed {
                img.save(&frame).ok()?;
            }
        }
        std::fs::write(dir.join(DONE_MARKER), b"").ok()?;
    }
    Some(frames_in(&dir))
}

/// The soundtrack of `video` as a WAV in its cache directory `dir`,
/// extracting it the first time; `None` when the clip has none. Kept apart
/// from the frames' [`DONE_MARKER`] so frames cached before sound existed
/// are not decoded again for it.
fn extract_audio(video: &Path, dir: &Path) -> Option<PathBuf> {
    let wav = dir.join(AUDIO_FILE);
    if wav.is_file() {
        return Some(wav);
    }
    if dir.join(SILENT_MARKER).exists() {
        return None;
    }
    let part = dir.join(format!("{AUDIO_FILE}.part"));
    let ok = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-y", "-i"])
        .arg(video)
        .args(["-vn", "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "wav"])
        .arg(&part)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if ok && std::fs::rename(&part, &wav).is_ok() {
        return Some(wav);
    }
    let _ = std::fs::remove_file(&part);
    let _ = std::fs::write(dir.join(SILENT_MARKER), b"");
    None
}

/// `img` at half its size, each pixel the alpha-weighted mean of a 2×2 block,
/// so the keyed background's white does not bleed into the edge pixels.
fn halve(img: &image::RgbaImage) -> image::RgbaImage {
    let (width, height) = ((img.width() / 2).max(1), (img.height() / 2).max(1));
    image::RgbaImage::from_fn(width, height, |x, y| {
        let mut sum = [0u32; 3];
        let (mut alpha, mut count) = (0u32, 0u32);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (sx, sy) = (2 * x + dx, 2 * y + dy);
            if sx >= img.width() || sy >= img.height() {
                continue;
            }
            let p = img.get_pixel(sx, sy).0;
            let a = u32::from(p[3]);
            for (s, c) in sum.iter_mut().zip(p) {
                *s += u32::from(c) * a;
            }
            alpha += a;
            count += 1;
        }
        let channel = |s: u32| u8::try_from(s / alpha.max(1)).unwrap_or(u8::MAX);
        image::Rgba([
            channel(sum[0]),
            channel(sum[1]),
            channel(sum[2]),
            u8::try_from(alpha / count.max(1)).unwrap_or(u8::MAX),
        ])
    })
}

/// The PNG frames in `dir`, in order.
fn frames_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    out.sort();
    out
}

/// Makes the near-white background transparent: every near-white pixel the
/// image border reaches through near-white pixels. A white region enclosed by
/// the subject (an apron, a collar, the whites of the eyes) is not reached and
/// stays opaque, which a plain color key would punch through.
#[must_use]
pub fn key_background(mut img: image::RgbaImage) -> image::RgbaImage {
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return img;
    }
    let light = |img: &image::RgbaImage, x: u32, y: u32| {
        let [red, green, blue, _] = img.get_pixel(x, y).0;
        red.min(green).min(blue) >= BACKGROUND_THRESHOLD
    };
    let mut seen = vec![false; (width * height) as usize];
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for x in 0..width {
        stack.push((x, 0));
        stack.push((x, height - 1));
    }
    for y in 0..height {
        stack.push((0, y));
        stack.push((width - 1, y));
    }
    while let Some((x, y)) = stack.pop() {
        let i = (y * width + x) as usize;
        if seen[i] || !light(&img, x, y) {
            continue;
        }
        seen[i] = true;
        img.get_pixel_mut(x, y).0[3] = 0;
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < width {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < height {
            stack.push((x, y + 1));
        }
    }
    img
}

/// A seed for the clip picker from the clock.
fn seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0x9E37_79B9_7F4A_7C15, |d| {
            u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
        })
        | 1
}

/// One xorshift64 step: plenty for choosing a clip.
fn next_random(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Picks the next clip from the `kinds` ready so far: a letter clip, at
/// random, three times in four; otherwise the digit clip after the last one
/// played (`digit_turn` counts them). Falls back to whichever pool has clips.
/// `None` when there are none.
fn next_clip(kinds: &[Kind], rng: &mut u64, digit_turn: &mut usize) -> Option<usize> {
    let pool = |k: Kind| -> Vec<usize> {
        kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == k)
            .map(|(i, _)| i)
            .collect()
    };
    let (letters, digits) = (pool(Kind::Letter), pool(Kind::Digit));
    let roll = next_random(rng);
    let want_letter = !roll.is_multiple_of(4);
    if !letters.is_empty() && (want_letter || digits.is_empty()) {
        let pick = usize::try_from((roll >> 2) % letters.len() as u64).unwrap_or(0);
        return Some(letters[pick]);
    }
    if digits.is_empty() {
        return None;
    }
    let pick = digits[*digit_turn % digits.len()];
    *digit_turn += 1;
    Some(pick)
}

/// Which clip is playing and since when, and which frame the screen holds.
struct Player {
    /// Clips decoded so far, filled by the decoder thread.
    clips: Arc<Mutex<Vec<Clip>>>,
    /// Set by the decoder thread once every clip is in [`Self::clips`].
    /// Nothing plays before: picking from the clips decoded so far would
    /// skew the mix toward whichever kind decodes first (the digit clips sort
    /// first, and the letter clips are the long ones), so a fresh install
    /// would open with a run of digit clips where three in four should be
    /// letters.
    decoded: Arc<AtomicBool>,
    rng: u64,
    digit_turn: usize,
    /// The clip playing and when it started.
    playing: Option<(usize, Instant)>,
    /// The clip start whose sound was last handed out ([`Self::new_sound`]).
    sounded: Option<(usize, Instant)>,
    /// The `(clip, frame)` last transmitted, and into which [`CLIP_IDS`] slot.
    on_screen: Option<((usize, usize), usize)>,
}

impl Player {
    fn new(seed: u64) -> Self {
        Self {
            clips: Arc::new(Mutex::new(Vec::new())),
            decoded: Arc::new(AtomicBool::new(false)),
            rng: seed,
            digit_turn: 0,
            playing: None,
            sounded: None,
            on_screen: None,
        }
    }

    /// Whether decoding is over and left something to play.
    fn has_clips(&self) -> bool {
        self.decoded.load(Ordering::Relaxed) && self.clips.lock().is_ok_and(|c| !c.is_empty())
    }

    /// Forgets what is on screen, so the next showing starts a fresh clip and
    /// retransmits rather than trusting images that were deleted.
    fn stop(&mut self) {
        self.playing = None;
        self.sounded = None;
        self.on_screen = None;
    }

    /// The sound of the clip playing, the first time it is asked after the
    /// clip started; `None` after that, and for a clip without sound.
    fn new_sound(&mut self) -> Option<PathBuf> {
        let playing = self.playing?;
        if self.sounded == Some(playing) {
            return None;
        }
        self.sounded = Some(playing);
        self.clips.lock().ok()?.get(playing.0)?.audio.clone()
    }

    /// Advances playback to `now` and returns the `(clip, frame)` due;
    /// `None` until every clip is decoded ([`Self::decoded`]).
    fn due(&mut self, now: Instant) -> Option<(usize, usize)> {
        if !self.decoded.load(Ordering::Relaxed) {
            return None;
        }
        let lengths: Vec<(Kind, usize)> = self
            .clips
            .lock()
            .ok()?
            .iter()
            .map(|c| (c.kind, c.frames.len()))
            .collect();
        if lengths.is_empty() {
            return None;
        }
        let frame_ms = 1000 / u128::from(FPS);
        // Two tries: the clip playing may have ended, and the next one starts.
        for _ in 0..2 {
            let (clip, started) = if let Some(p) = self.playing {
                p
            } else {
                let kinds: Vec<Kind> = lengths.iter().map(|(k, _)| *k).collect();
                let next = next_clip(&kinds, &mut self.rng, &mut self.digit_turn)?;
                self.playing = Some((next, now));
                (next, now)
            };
            let frame = usize::try_from(now.duration_since(started).as_millis() / frame_ms)
                .unwrap_or(usize::MAX);
            if frame < lengths[clip].1 {
                return Some((clip, frame));
            }
            self.playing = None;
        }
        None
    }

    /// The PNG of frame `frame` of clip `clip`.
    fn frame_png(&self, (clip, frame): (usize, usize)) -> Option<Vec<u8>> {
        let path = self
            .clips
            .lock()
            .ok()?
            .get(clip)?
            .frames
            .get(frame)?
            .clone();
        std::fs::read(path).ok()
    }

    /// For [`Mode::Placeholders`]: advances playback to `now`, transmitting
    /// the frame due if it is not the one on screen, and returns the image id
    /// the cells should name; `None` while no clip is ready.
    fn placeholder_id(&mut self, now: Instant) -> Option<u32> {
        let due = self.due(now)?;
        if let Some((shown, slot)) = self.on_screen
            && shown == due
        {
            return Some(CLIP_IDS[slot]);
        }
        let slot = self.on_screen.map_or(0, |(_, s)| 1 - s);
        let png = self.frame_png(due)?;
        write_to_terminal(&transmit_sequence(CLIP_IDS[slot], &png)).ok()?;
        self.on_screen = Some((due, slot));
        Some(CLIP_IDS[slot])
    }
}

/// [`Mode::Direct`] bookkeeping between a draw and its [`after_draw`].
struct DirectState {
    /// Where and which `(clip, frame)` this frame's [`render`] asked for,
    /// taken by [`after_draw`].
    want: Option<(Rect, (usize, usize))>,
    /// What is on screen: where, which frame, and under which [`CLIP_IDS`]
    /// slot.
    placed: Option<(Rect, (usize, usize), usize)>,
}

static DIRECT_STATE: Mutex<DirectState> = Mutex::new(DirectState {
    want: None,
    placed: None,
});

/// The escapes that draw `png` as image `id` with its top-left at `at`,
/// stretched to the box, leaving the cursor where it was.
pub(crate) fn place_sequence(id: u32, at: Rect, png: &[u8]) -> String {
    let data = crate::tui::base64(png);
    let chunks: Vec<&str> = data
        .as_bytes()
        .chunks(CHUNK)
        .map(|c| std::str::from_utf8(c).unwrap_or_default())
        .collect();
    let mut out = String::with_capacity(data.len() + chunks.len() * 16 + 96);
    // Save the cursor and move to the box; it is put back at the end.
    // Writing to a `String` cannot fail.
    let _ = write!(out, "\x1b7\x1b[{};{}H", at.y + 1, at.x + 1);
    for (i, chunk) in chunks.iter().enumerate() {
        let more = u8::from(i + 1 < chunks.len());
        let _ = if i == 0 {
            write!(
                out,
                "\x1b_Ga=T,f=100,i={id},c={},r={},C=1,q=2,m={more};{chunk}\x1b\\",
                at.width, at.height
            )
        } else {
            write!(out, "\x1b_Gm={more};{chunk}\x1b\\")
        };
    }
    out.push_str("\x1b8");
    out
}

/// Whether every cell of `at` in `buf` still holds the [`MARKER`]: nothing
/// was drawn over the picture's place this frame.
fn markers_intact(buf: &Buffer, at: Rect) -> bool {
    (at.top()..at.bottom()).all(|y| {
        (at.left()..at.right()).all(|x| buf.cell((x, y)).is_some_and(|c| c.symbol() == MARKER))
    })
}

/// For [`Mode::Direct`]: called with the buffer a draw just put on screen,
/// places, moves, advances or removes the picture to match it. A frame
/// [`render`] was not part of removes it. A no-op in [`Mode::Placeholders`]
/// and while the buddy is off.
pub fn after_draw(buf: &Buffer) {
    if !shown() || mode() != Mode::Direct {
        return;
    }
    let Ok(mut direct) = DIRECT_STATE.lock() else {
        return;
    };
    let want = direct
        .want
        .take()
        .filter(|(at, _)| markers_intact(buf, *at));
    if want == direct.placed.map(|(at, due, _)| (at, due)) {
        return;
    }
    let mut seq = String::new();
    let mut placed = None;
    if let Some((at, due)) = want {
        let slot = direct.placed.map_or(0, |(_, _, s)| 1 - s);
        let png = PLAYER
            .lock()
            .ok()
            .and_then(|p| p.as_ref().and_then(|p| p.frame_png(due)));
        if let Some(png) = png {
            seq.push_str(&place_sequence(CLIP_IDS[slot], at, &png));
            placed = Some((at, due, slot));
        }
    }
    // The old picture goes only once the new one is up, so nothing blinks.
    if let Some((_, _, old)) = direct.placed {
        seq.push_str(&delete_sequence(CLIP_IDS[old]));
    }
    direct.placed = placed;
    let _ = write_to_terminal(&seq);
}

/// For [`Mode::Direct`]: removes the picture now, for frames drawn outside
/// the idle repaint (a compaction, a shell escape, a WASM frame), whose own
/// draw is not followed by [`after_draw`]. A no-op otherwise.
pub fn hide() {
    voice::hush();
    if !shown() || mode() != Mode::Direct {
        return;
    }
    if let Ok(mut direct) = DIRECT_STATE.lock() {
        direct.want = None;
        if let Some((_, _, old)) = direct.placed.take() {
            let _ = write_to_terminal(&delete_sequence(CLIP_IDS[old]));
        }
    }
}

/// Takes the buddy off the screen as plank exits, in either mode: a picture
/// left behind would outlive the process (in Warp it sits outside the cell
/// grid, so leaving the alternate screen does not take it away). Leaves
/// [`STATE_FILE`] alone, so the next launch still brings it back.
///
/// Locks with `try_lock`: on a force quit the UI thread may be holding one,
/// and a picture left on screen beats an exit that hangs.
pub fn hide_for_exit() {
    voice::hush();
    if !SHOWN.swap(false, Ordering::Relaxed) {
        return;
    }
    if let Ok(mut player) = PLAYER.try_lock()
        && let Some(p) = player.as_mut()
    {
        p.stop();
    }
    if let Ok(mut direct) = DIRECT_STATE.try_lock() {
        direct.want = None;
        direct.placed = None;
    }
    let all: String = CLIP_IDS.into_iter().map(delete_sequence).collect();
    let _ = write_to_terminal(&all);
}

thread_local! {
    /// Set by [`offer`] for the idle repaint, the one frame the buddy may
    /// stand in, and cleared by [`render`]. Per thread, since frames are drawn
    /// on the UI thread and tests must not see one another's frames.
    static OFFERED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The box the output pane, drawn earlier in this frame, wrapped its text
    /// around ([`claim`]); [`render`] draws only there.
    static CLAIMED: std::cell::Cell<Option<Rect>> = const { std::cell::Cell::new(None) };
}

/// Marks the frame about to be drawn as the idle repaint, which ends with
/// [`render`]. Until then [`room_wanted`] asks the output pane to keep the
/// buddy's box clear.
pub fn offer() {
    OFFERED.set(true);
    CLAIMED.set(None);
}

/// The box the output pane should wrap its text around in this frame: the
/// buddy's [`placement`] in `output`, when it is on and the frame was
/// [`offer`]ed. The pane [`claim`]s it once it has.
#[must_use]
pub fn room_wanted(output: Rect) -> Option<Rect> {
    if !OFFERED.get() || !shown() {
        return None;
    }
    placement(output)
}

/// Records that the output pane left `at` clear of text, so [`render`] may
/// draw the buddy there. A pane that cannot (scrolled back, a selection being
/// made) does not claim it, and the buddy sits that frame out rather than
/// cover text.
pub fn claim(at: Rect) {
    CLAIMED.set(Some(at));
}

/// Where the buddy goes in `output`: right-aligned with [`RIGHT_GAP`] columns
/// to spare and standing on its bottom edge, which is the rule above the
/// prompt. `None` when the pane is too small to hold it with room left over.
#[must_use]
pub fn placement(output: Rect) -> Option<Rect> {
    if output.width < COLS + RIGHT_GAP + 2 || output.height < ROWS + 2 {
        return None;
    }
    Some(Rect {
        x: output.right() - RIGHT_GAP - COLS,
        y: output.bottom() - ROWS,
        width: COLS,
        height: ROWS,
    })
}

/// The symbol of the placeholder cell at `row`, `col` of the picture.
fn cell_symbol(row: u16, col: u16) -> String {
    [
        PLACEHOLDER,
        DIACRITICS[usize::from(row)],
        DIACRITICS[usize::from(col)],
    ]
    .iter()
    .collect()
}

/// Draws the buddy into `buf` at [`placement`] in `output`, when it is on, a
/// clip is ready and the output pane [`claim`]ed the box by wrapping its text
/// around it, after moving playback on to the frame now due. The idle
/// repaint calls it, after the output pane is drawn (which would paint over
/// it) and before any overlay (which should paint over it); nothing else
/// does, so the buddy is never up while a turn runs. In [`Mode::Placeholders`]
/// the frame is transmitted from inside the draw, before ratatui writes the
/// frame, on the same thread, so the two never interleave.
pub fn render(buf: &mut Buffer, output: Rect) {
    if RELOAD.swap(false, Ordering::Relaxed) {
        // A new buddy was installed: the old one's images go, and so does its
        // player, whose clip indices mean nothing now.
        clear_screen();
        if let Ok(mut player) = PLAYER.lock() {
            *player = None;
        }
        if shown() {
            start_player();
        }
    }
    OFFERED.set(false);
    let Some(at) = CLAIMED
        .take()
        .filter(|at| shown() && placement(output) == Some(*at))
    else {
        return;
    };
    let now = Instant::now();
    if mode() == Mode::Direct {
        let Some(due) = PLAYER
            .lock()
            .ok()
            .and_then(|mut p| p.as_mut().and_then(|p| p.due(now)))
        else {
            return;
        };
        if let Ok(mut direct) = DIRECT_STATE.lock() {
            direct.want = Some((at, due));
        }
        sound_with_clip();
        for y in at.top()..at.bottom() {
            for x in at.left()..at.right() {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_symbol(MARKER);
                }
            }
        }
        return;
    }
    if let Some(id) = PLAYER
        .lock()
        .ok()
        .and_then(|mut p| p.as_mut().and_then(|p| p.placeholder_id(now)))
    {
        paint(buf, at, id);
        sound_with_clip();
    }
}

/// Called from [`render`] once the frame due is on its way to the screen:
/// keeps the clip's sound going ([`voice::drawn`]) and starts it with the
/// clip's first frame when sound is on.
fn sound_with_clip() {
    voice::drawn();
    let sound = PLAYER
        .lock()
        .ok()
        .and_then(|mut p| p.as_mut().and_then(Player::new_sound));
    if let Some(path) = sound
        && SOUND.load(Ordering::Relaxed)
    {
        voice::speak(&path);
    }
}

/// Plays clip soundtracks through the system's player, one at a time, and
/// stops one once the buddy is no longer being drawn.
mod voice {
    use std::path::Path;
    use std::process::{Child, Command, Stdio};
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    /// How long the buddy may go undrawn before its sound stops. The idle
    /// loop redraws at the clip's frame rate while it plays, and the busy
    /// loop of a quiet pass at least every 100 ms, so only a buddy that is
    /// really gone goes this long.
    pub(super) const GRACE: Duration = Duration::from_millis(500);

    /// The player process of the sound playing.
    static PLAYING: Mutex<Option<Child>> = Mutex::new(None);

    /// When [`super::render`] last drew the buddy.
    static DRAWN: Mutex<Option<Instant>> = Mutex::new(None);

    /// Players tried in order, each with its arguments before the file.
    const PLAYERS: &[(&str, &[&str])] = &[
        ("afplay", &[]),
        ("paplay", &[]),
        ("aplay", &["-q"]),
        ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ];

    /// Notes that the buddy was drawn now.
    pub(super) fn drawn() {
        if let Ok(mut at) = DRAWN.lock() {
            *at = Some(Instant::now());
        }
    }

    /// Plays `wav`, stopping whatever played before. Silent when no player
    /// is installed.
    pub(super) fn speak(wav: &Path) {
        hush();
        let child = PLAYERS.iter().find_map(|(cmd, args)| {
            Command::new(cmd)
                .args(*args)
                .arg(wav)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .ok()
        });
        if let (Some(child), Ok(mut playing)) = (child, PLAYING.lock()) {
            *playing = Some(child);
            watch();
        }
    }

    /// Stops the sound playing, if any.
    pub(super) fn hush() {
        // `try_lock`: this runs on exit too, where hanging is worse than a
        // sound that plays out.
        if let Ok(mut playing) = PLAYING.try_lock()
            && let Some(mut child) = playing.take()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Starts, once, the thread that stops the sound when the buddy has not
    /// been drawn for [`GRACE`], or was turned off or silenced, and reaps a
    /// player that finished.
    fn watch() {
        static STARTED: std::sync::Once = std::sync::Once::new();
        STARTED.call_once(|| {
            std::thread::spawn(|| {
                loop {
                    std::thread::sleep(Duration::from_millis(100));
                    let gone = DRAWN
                        .lock()
                        .ok()
                        .and_then(|at| *at)
                        .is_none_or(|at| at.elapsed() > GRACE);
                    let wanted = super::shown() && super::SOUND.load(Ordering::Relaxed);
                    let Ok(mut playing) = PLAYING.lock() else {
                        continue;
                    };
                    let Some(child) = playing.as_mut() else {
                        continue;
                    };
                    if gone || !wanted {
                        let _ = child.kill();
                        let _ = child.wait();
                        *playing = None;
                    } else if child.try_wait().is_ok_and(|done| done.is_some()) {
                        *playing = None;
                    }
                }
            });
        });
    }
}

/// Paints placeholder cells naming image `id` over `at`.
fn paint(buf: &mut Buffer, at: Rect, id: u32) {
    let [_, r, g, b] = id.to_be_bytes();
    for row in 0..at.height {
        for col in 0..at.width {
            if let Some(cell) = buf.cell_mut((at.x + col, at.y + row)) {
                cell.reset();
                cell.set_symbol(&cell_symbol(row, col))
                    .set_fg(Color::Rgb(r, g, b));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_on_off_state_is_remembered_by_a_file_in_the_home() {
        let home = std::env::temp_dir().join(format!("plank-buddy-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        assert!(!super::remembered_in(&home), "off until turned on");
        super::remember_in(&home, true).unwrap();
        assert!(super::remembered_in(&home));
        // Turning on twice, or off twice, is not an error.
        super::remember_in(&home, true).unwrap();
        super::remember_in(&home, false).unwrap();
        assert!(!super::remembered_in(&home));
        super::remember_in(&home, false).unwrap();
        let _ = std::fs::remove_dir_all(&home);
    }

    use super::*;

    #[test]
    fn kitty_ghostty_and_warp_outside_tmux_are_supported() {
        let p = Some(Mode::Placeholders);
        assert_eq!(supported_by(Some("xterm-kitty"), None, false, false), p);
        assert_eq!(supported_by(Some("xterm-ghostty"), None, false, false), p);
        assert_eq!(supported_by(None, Some("ghostty"), false, false), p);
        assert_eq!(supported_by(Some("xterm-256color"), None, true, false), p);
        assert_eq!(
            supported_by(Some("xterm-256color"), Some("WarpTerminal"), false, false),
            Some(Mode::Direct)
        );
        assert_eq!(
            supported_by(Some("xterm-256color"), Some("iTerm.app"), false, false),
            None
        );
        assert_eq!(supported_by(Some("tmux-256color"), None, true, true), None);
        assert_eq!(supported_by(Some("xterm-kitty"), None, false, true), None);
        assert_eq!(supported_by(None, Some("WarpTerminal"), false, true), None);
    }

    #[test]
    fn transmission_is_chunked_and_creates_a_virtual_placement() {
        let id = CLIP_IDS[0];
        let png = vec![7u8; CHUNK * 2];
        let seq = transmit_sequence(id, &png);
        let parts: Vec<&str> = seq.split("\x1b\\").filter(|p| !p.is_empty()).collect();
        assert!(parts.len() > 2, "{} chunks", parts.len());
        assert!(parts[0].starts_with(&format!(
            "\x1b_Ga=T,U=1,f=100,i={id},c={COLS},r={ROWS},q=2,m=1;"
        )));
        for mid in &parts[1..parts.len() - 1] {
            assert!(mid.starts_with("\x1b_Gm=1;"));
        }
        assert!(parts.last().unwrap().starts_with("\x1b_Gm=0;"));
        let payload: String = parts.iter().map(|p| p.split_once(';').unwrap().1).collect();
        assert_eq!(payload, crate::tui::base64(&png));
        for part in &parts {
            assert!(part.split_once(';').unwrap().1.len() <= CHUNK);
        }
    }

    #[test]
    fn a_small_picture_goes_in_one_final_chunk() {
        let seq = transmit_sequence(CLIP_IDS[1], b"\x89PNG\r\n\x1a\n");
        assert_eq!(seq.matches("\x1b_G").count(), 1);
        assert!(seq.contains(&format!("i={},", CLIP_IDS[1])));
        assert!(seq.contains(",m=0;"));
    }

    #[test]
    fn the_ids_fit_the_24_bit_foreground_and_differ() {
        for id in CLIP_IDS {
            assert!(id < 1 << 24);
        }
        assert_ne!(CLIP_IDS[0], CLIP_IDS[1]);
        assert!(DIACRITICS.len() >= usize::from(COLS.max(ROWS)));
    }

    #[test]
    fn placement_stands_on_the_bottom_edge_right_aligned() {
        let output = Rect::new(0, 0, 80, 30);
        let at = placement(output).unwrap();
        assert_eq!(at.bottom(), output.bottom());
        assert_eq!(at.right(), output.right() - RIGHT_GAP);
        assert_eq!((at.width, at.height), (COLS, ROWS));
        assert!(placement(Rect::new(0, 0, 80, ROWS + 1)).is_none());
        assert!(placement(Rect::new(0, 0, COLS + 3, 30)).is_none());
    }

    #[test]
    fn painted_cells_are_one_column_placeholders_naming_row_and_column() {
        let output = Rect::new(0, 0, 60, 20);
        let mut buf = Buffer::empty(output);
        let at = placement(output).unwrap();
        paint(&mut buf, at, CLIP_IDS[0]);
        let [_, r, g, b] = CLIP_IDS[0].to_be_bytes();
        for (row, col) in [(0, 0), (3, 7), (ROWS - 1, COLS - 1)] {
            let cell = &buf[(at.x + col, at.y + row)];
            let chars: Vec<char> = cell.symbol().chars().collect();
            assert_eq!(
                chars,
                [
                    PLACEHOLDER,
                    DIACRITICS[usize::from(row)],
                    DIACRITICS[usize::from(col)]
                ]
            );
            assert_eq!(cell.fg, Color::Rgb(r, g, b));
            assert_eq!(unicode_width::UnicodeWidthStr::width(cell.symbol()), 1);
        }
        // Nothing outside the box is touched.
        assert_eq!(buf[(at.x - 1, at.y)].symbol(), " ");
        assert_eq!(buf[(at.x, at.y - 1)].symbol(), " ");
    }

    #[test]
    fn clips_are_sorted_by_the_last_character_of_their_name() {
        assert_eq!(kind_of(Path::new("a/minion1.mp4")), Some(Kind::Digit));
        assert_eq!(kind_of(Path::new("minion12.mp4")), Some(Kind::Digit));
        assert_eq!(kind_of(Path::new("minionA.mp4")), Some(Kind::Letter));
        assert_eq!(kind_of(Path::new("minion_.mp4")), None);
    }

    #[test]
    fn letters_play_three_times_in_four_and_digits_rotate() {
        use Kind::{Digit, Letter};
        let kinds = [Digit, Digit, Digit, Letter, Letter];
        let (mut rng, mut turn) = (0x1234_5678_9ABC_DEF1, 0);
        let picks: Vec<usize> = (0..4000)
            .map(|_| next_clip(&kinds, &mut rng, &mut turn).unwrap())
            .collect();
        let letters = picks.iter().filter(|&&i| kinds[i] == Letter).count();
        assert!(
            (2850..=3150).contains(&letters),
            "{letters} letters in 4000"
        );
        // Both letters get picked.
        assert!(picks.contains(&3) && picks.contains(&4));
        // Digits come in order, wrapping.
        let digits: Vec<usize> = picks.into_iter().filter(|&i| kinds[i] == Digit).collect();
        for (n, d) in digits.iter().enumerate() {
            assert_eq!(*d, n % 3);
        }
    }

    #[test]
    fn a_single_pool_is_used_whatever_the_roll() {
        use Kind::{Digit, Letter};
        let (mut rng, mut turn) = (7, 0);
        for _ in 0..50 {
            assert_eq!(next_clip(&[Letter], &mut rng, &mut turn), Some(0));
        }
        for n in 0..50 {
            assert_eq!(next_clip(&[Digit, Digit], &mut rng, &mut turn), Some(n % 2));
        }
        assert_eq!(next_clip(&[], &mut rng, &mut turn), None);
    }

    #[test]
    fn keying_clears_the_border_white_but_not_white_the_subject_encloses() {
        let white = image::Rgba([250, 250, 250, 255]);
        let ink = image::Rgba([20, 20, 20, 255]);
        // A dark ring with a white hole, on white.
        let mut img = image::RgbaImage::from_pixel(9, 9, white);
        for i in 2..7 {
            for j in [2, 6] {
                img.put_pixel(i, j, ink);
                img.put_pixel(j, i, ink);
            }
        }
        let keyed = key_background(img);
        assert_eq!(keyed.get_pixel(0, 0).0[3], 0);
        assert_eq!(keyed.get_pixel(8, 4).0[3], 0);
        assert_eq!(keyed.get_pixel(2, 2).0[3], 255);
        assert_eq!(keyed.get_pixel(4, 4).0[3], 255, "the enclosed white stays");
    }

    #[test]
    fn a_one_pixel_gap_still_lets_the_background_through_once_halved() {
        let white = image::Rgba([250, 250, 250, 255]);
        let ink = image::Rgba([20, 20, 20, 255]);
        // A two-pixel-thick dark ring on white with a one-pixel opening in its
        // top side: halving first would blur the opening shut.
        let mut img = image::RgbaImage::from_pixel(12, 12, white);
        for i in 2..10 {
            for j in [2, 3, 8, 9] {
                img.put_pixel(i, j, ink);
                img.put_pixel(j, i, ink);
            }
        }
        img.put_pixel(5, 2, white);
        img.put_pixel(5, 3, white);
        let halved = halve(&key_background(img));
        assert_eq!(halved.dimensions(), (6, 6));
        assert_eq!(halved.get_pixel(3, 3).0[3], 0, "the inside is background");
        assert_eq!(halved.get_pixel(1, 3).0[3], 255, "the ring stays");
    }

    #[test]
    fn halving_does_not_bleed_the_keyed_white_into_the_edge() {
        let mut img = image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 255, 255, 0]));
        img.put_pixel(0, 0, image::Rgba([20, 20, 20, 255]));
        let halved = halve(&img);
        assert_eq!(halved.get_pixel(0, 0).0, [20, 20, 20, 63]);
    }

    #[test]
    fn a_player_without_clips_shows_nothing() {
        let mut p = Player::new(1);
        assert!(!p.has_clips());
        assert_eq!(p.due(Instant::now()), None);
        assert_eq!(p.placeholder_id(Instant::now()), None);
    }

    #[test]
    fn a_player_plays_a_clip_through_then_picks_the_next() {
        let mut p = Player::new(1);
        let frames = (0..3).map(|i| PathBuf::from(format!("{i}.png"))).collect();
        p.clips.lock().unwrap().push(Clip {
            kind: Kind::Letter,
            frames,
            audio: None,
        });
        let t0 = Instant::now();
        // Nothing plays while decoding is still going.
        assert_eq!(p.due(t0), None);
        assert!(!p.has_clips());
        p.decoded.store(true, Ordering::Relaxed);
        assert!(p.has_clips());
        let at = |n: u64| t0 + Duration::from_millis(n);
        let step = 1000 / u64::from(FPS);
        assert_eq!(p.due(t0), Some((0, 0)));
        assert_eq!(p.due(at(step)), Some((0, 1)));
        assert_eq!(p.due(at(2 * step + 1)), Some((0, 2)));
        // Past the end: the only clip starts over from its first frame.
        assert_eq!(p.due(at(3 * step + 1)), Some((0, 0)));
    }

    #[test]
    fn a_clip_hands_out_its_sound_once_per_start() {
        let mut p = Player::new(1);
        p.clips.lock().unwrap().push(Clip {
            kind: Kind::Letter,
            frames: vec![PathBuf::from("0.png"), PathBuf::from("1.png")],
            audio: Some(PathBuf::from("audio.wav")),
        });
        p.decoded.store(true, Ordering::Relaxed);
        assert_eq!(p.new_sound(), None, "nothing plays yet");
        let t0 = Instant::now();
        p.due(t0);
        assert_eq!(p.new_sound(), Some(PathBuf::from("audio.wav")));
        // Later frames of the same clip do not start it again.
        p.due(t0 + Duration::from_millis(1000 / u64::from(FPS)));
        assert_eq!(p.new_sound(), None);
        // The clip starting over is a new start, with its sound again.
        p.due(t0 + Duration::from_millis(2 * 1000 / u64::from(FPS) + 1));
        assert_eq!(p.new_sound(), Some(PathBuf::from("audio.wav")));
        // A stopped player starts afresh, sound included.
        p.stop();
        p.due(t0 + Duration::from_secs(10));
        assert_eq!(p.new_sound(), Some(PathBuf::from("audio.wav")));
    }

    #[test]
    fn sound_is_extracted_once_and_a_silent_clip_is_not_retried() {
        let dir = std::env::temp_dir().join(format!("plank-buddy-audio-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Already extracted: found without running anything.
        std::fs::write(dir.join(AUDIO_FILE), b"RIFF").unwrap();
        assert_eq!(
            extract_audio(Path::new("/nonexistent.mp4"), &dir),
            Some(dir.join(AUDIO_FILE))
        );
        // No sound in the clip (here: no clip at all): marked silent, and
        // asked again it stays silent without a second extraction.
        std::fs::remove_file(dir.join(AUDIO_FILE)).unwrap();
        assert_eq!(extract_audio(Path::new("/nonexistent.mp4"), &dir), None);
        assert!(dir.join(SILENT_MARKER).exists());
        assert!(!dir.join(format!("{AUDIO_FILE}.part")).exists());
        assert_eq!(extract_audio(Path::new("/nonexistent.mp4"), &dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn through_the_player_three_clips_in_four_are_letter_clips() {
        // Clip lengths like the minions': six short digit clips, two long
        // letter ones. Counting clip starts over simulated time checks the
        // whole player, not just the picker.
        let mut p = Player::new(0x2545_F491_4F6C_DD1D);
        for (kind, n) in [(Kind::Digit, 37), (Kind::Digit, 54), (Kind::Digit, 71)]
            .into_iter()
            .chain([(Kind::Letter, 181), (Kind::Letter, 181)])
        {
            let frames = (0..n).map(|i| PathBuf::from(format!("{i}.png"))).collect();
            p.clips.lock().unwrap().push(Clip {
                kind,
                frames,
                audio: None,
            });
        }
        p.decoded.store(true, Ordering::Relaxed);
        let kinds: Vec<Kind> = p.clips.lock().unwrap().iter().map(|c| c.kind).collect();
        let t0 = Instant::now();
        let (mut letters, mut total) = (0u32, 0u32);
        for ms in (0..4_000_000u64).step_by(usize::try_from(1000 / FPS).unwrap()) {
            let now = t0 + Duration::from_millis(ms);
            p.due(now);
            if let Some((clip, started)) = p.playing
                && started == now
            {
                total += 1;
                letters += u32::from(kinds[clip] == Kind::Letter);
            }
        }
        assert!(
            (71 * total..=79 * total).contains(&(100 * letters)),
            "{letters} of {total} clips were letters"
        );
    }

    #[test]
    fn a_direct_placement_saves_moves_and_restores_the_cursor() {
        let seq = place_sequence(CLIP_IDS[0], Rect::new(10, 4, COLS, ROWS), b"png");
        assert!(seq.starts_with("\x1b7\x1b[5;11H\x1b_Ga=T,f=100,"));
        assert!(seq.contains(&format!("i={},c={COLS},r={ROWS},C=1,q=2,m=0;", CLIP_IDS[0])));
        assert!(seq.ends_with("\x1b\\\x1b8"));
    }

    #[test]
    fn markers_are_seen_and_an_overlay_on_one_breaks_them() {
        let area = Rect::new(0, 0, 40, 20);
        let at = Rect::new(5, 5, 4, 3);
        let mut buf = Buffer::empty(area);
        assert!(!markers_intact(&buf, at));
        for y in at.top()..at.bottom() {
            for x in at.left()..at.right() {
                buf[(x, y)].set_symbol(MARKER);
            }
        }
        assert!(markers_intact(&buf, at));
        buf[(at.right() - 1, at.bottom() - 1)].set_symbol("x");
        assert!(!markers_intact(&buf, at));
        assert_eq!(unicode_width::UnicodeWidthStr::width(MARKER), 1);
    }

    #[test]
    fn the_listing_keeps_folders_and_files_with_their_urls() {
        let body = r#"[
            {"name": "minions", "type": "dir", "download_url": null},
            {"name": "README.md", "type": "file", "download_url": "https://x/README.md"}
        ]"#;
        let listing = parse_listing(body).unwrap();
        assert_eq!(
            listing,
            [
                ("minions".to_owned(), "dir".to_owned(), None),
                (
                    "README.md".to_owned(),
                    "file".to_owned(),
                    Some("https://x/README.md".to_owned())
                ),
            ]
        );
        assert!(parse_listing(r#"{"message": "Not Found"}"#).is_err());
    }

    #[test]
    fn buddy_names_cannot_climb_out_of_the_install() {
        assert!(valid_name("minions"));
        assert!(valid_name("cat_2-b"));
        for bad in ["", "..", "a/b", "a b", ".hidden"] {
            assert!(!valid_name(bad), "{bad}");
        }
    }

    #[test]
    fn the_installed_buddy_is_named_by_its_record_and_needs_clips() {
        let dir =
            std::env::temp_dir().join(format!("plank-buddy-installed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(installed_in(&dir), None);
        std::fs::write(dir.join("clipA.mp4"), b"").unwrap();
        assert_eq!(installed_in(&dir).as_deref(), Some("a buddy"));
        std::fs::write(dir.join(NAME_FILE), "minions\n").unwrap();
        assert_eq!(installed_in(&dir).as_deref(), Some("minions"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replacing_swaps_the_whole_directory() {
        let root = std::env::temp_dir().join(format!("plank-buddy-replace-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (staging, dir) = (root.join("staging"), root.join("buddy"));
        std::fs::create_dir_all(dir.join(FRAME_CACHE)).unwrap();
        std::fs::write(dir.join("oldA.mp4"), b"").unwrap();
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("newA.mp4"), b"").unwrap();
        replace_installed(&staging, &dir).unwrap();
        assert!(dir.join("newA.mp4").exists());
        assert!(!dir.join("oldA.mp4").exists());
        assert!(!dir.join(FRAME_CACHE).exists(), "the old frames go too");
        assert!(!staging.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unknown_words_are_refused_with_the_usage() {
        let Outcome::Say(line) = command("sideways") else {
            panic!("expected a line");
        };
        assert!(line.contains("on|off|install"), "{line}");
    }
}
