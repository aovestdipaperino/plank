// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Desktop integrations behind one API (issue #89, step 2): the clipboard,
//! handing a file to its default app, playing a sound, and notifications.
//!
//! Each one shells out to the tool the OS ships or commonly installs, as the
//! macOS path always did: `pbcopy`/`osascript`/`open`/`afplay` there,
//! `wl-copy`/`xclip`/`xsel`/`xdg-open`/`paplay`/`notify-send` on Linux, and
//! Windows PowerShell on Windows. No new crates, and a missing tool degrades
//! to "nothing happened" exactly as a missing `pngpaste` always has.
//!
//! The candidate lists are pure functions so the choice of tool is tested on
//! every OS without launching anything.

use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

/// The key that pastes into the terminal, for hints: the terminal, not plank,
/// owns the paste, and each OS's terminals bind it differently.
pub const PASTE_KEY: &str = if cfg!(target_os = "macos") {
    "Cmd-V"
} else if cfg!(windows) {
    "Ctrl-V"
} else {
    "Ctrl-Shift-V"
};

/// One external command: a program and its arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    /// The program, looked up on `PATH`.
    pub program: &'static str,
    /// Its arguments.
    pub args: Vec<OsString>,
}

impl Tool {
    fn new(program: &'static str, args: &[&str]) -> Self {
        Self {
            program,
            args: args.iter().map(OsString::from).collect(),
        }
    }

    /// `powershell -NoProfile -NonInteractive -Command <script>`. Windows
    /// PowerShell 5.1 ships with every supported Windows, unlike `pwsh`.
    #[cfg_attr(not(windows), allow(dead_code))]
    fn powershell(script: &str) -> Self {
        Self::new(
            "powershell",
            &["-NoProfile", "-NonInteractive", "-Command", script],
        )
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(self.program);
        cmd.args(&self.args);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            // CREATE_NO_WINDOW: a console-less PowerShell must not flash a
            // window over the TUI.
            cmd.creation_flags(0x0800_0000);
        }
        cmd
    }

    /// Runs the tool with `input` on stdin; true when it exited 0.
    fn feed(&self, input: &[u8]) -> bool {
        let Ok(mut child) = self
            .command()
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        else {
            return false;
        };
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = stdin.write_all(input);
        }
        drop(child.stdin.take());
        child.wait().is_ok_and(|s| s.success())
    }

    /// Runs the tool; its stdout when it exited 0.
    fn read(&self) -> Option<Vec<u8>> {
        let out = self
            .command()
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        out.status.success().then_some(out.stdout)
    }

    /// Starts the tool detached from plank's stdio.
    fn spawn(&self) -> std::io::Result<std::process::Child> {
        self.command()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    }
}

/// A PowerShell single-quoted string literal holding `s`.
#[cfg_attr(not(windows), allow(dead_code))]
fn ps_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// True when a Wayland compositor is reachable, so the `wl-*` tools apply.
#[cfg(all(unix, not(target_os = "macos")))]
fn wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some_and(|d| !d.is_empty())
}

/// Tools that take text on stdin into the clipboard, tried in order.
#[must_use]
pub fn copy_tools(wayland: bool) -> Vec<Tool> {
    if cfg!(target_os = "macos") {
        return vec![Tool::new("pbcopy", &[])];
    }
    if cfg!(windows) {
        // `clip.exe` reads the console code page, not UTF-8.
        return vec![Tool::powershell(
            "[Console]::InputEncoding = [Text.UTF8Encoding]::new($false); \
             Set-Clipboard -Value ([Console]::In.ReadToEnd())",
        )];
    }
    let mut tools = Vec::new();
    if wayland {
        tools.push(Tool::new("wl-copy", &[]));
    }
    tools.push(Tool::new("xclip", &["-selection", "clipboard"]));
    tools.push(Tool::new("xsel", &["--clipboard", "--input"]));
    tools
}

/// Tools that print the clipboard's text, tried in order.
#[must_use]
pub fn paste_tools(wayland: bool) -> Vec<Tool> {
    if cfg!(target_os = "macos") {
        return vec![Tool::new("pbpaste", &[])];
    }
    if cfg!(windows) {
        return vec![Tool::powershell(
            "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); \
             Get-Clipboard -Raw",
        )];
    }
    let mut tools = Vec::new();
    if wayland {
        tools.push(Tool::new("wl-paste", &["--no-newline"]));
    }
    tools.push(Tool::new("xclip", &["-selection", "clipboard", "-o"]));
    tools.push(Tool::new("xsel", &["--clipboard", "--output"]));
    tools
}

#[cfg(target_os = "macos")]
fn on_wayland() -> bool {
    false
}
#[cfg(all(unix, not(target_os = "macos")))]
fn on_wayland() -> bool {
    wayland()
}
#[cfg(windows)]
fn on_wayland() -> bool {
    false
}

/// Copies `text` to the local clipboard with the first tool that works.
/// Best-effort: nothing happens when none does (none installed, or no
/// display), which is why the TUI also sends OSC 52.
pub fn copy_text(text: &str) {
    let _ = copy_tools(on_wayland())
        .iter()
        .any(|t| t.feed(text.as_bytes()));
}

/// The clipboard's text, or `None` when it is empty, holds no text, or no tool
/// could read it.
#[must_use]
pub fn paste_text() -> Option<String> {
    let bytes = paste_tools(on_wayland()).iter().find_map(Tool::read)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    (!text.is_empty()).then_some(text)
}

/// True when the clipboard holds image data.
///
/// Cheap enough for the status-bar hint, re-probed every few seconds: on macOS
/// `clipboard info for «class PNGf»` prints the class only when a PNG
/// representation exists, without reading the bytes; on Linux the clipboard's
/// offered types are listed; on Windows it is one Win32 call, no process.
#[must_use]
pub fn clipboard_has_image() -> bool {
    #[cfg(target_os = "macos")]
    {
        Tool::new("osascript", &["-e", "clipboard info for «class PNGf»"])
            .read()
            .is_some_and(|out| !out.trim_ascii().is_empty())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        image_target_tools(wayland())
            .iter()
            .find_map(Tool::read)
            .is_some_and(|out| lists_png(&out))
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::DataExchange::IsClipboardFormatAvailable;
        /// `CF_DIB`, which Windows synthesizes from any bitmap on the clipboard.
        const CF_DIB: u32 = 8;
        // SAFETY: a read-only query that needs no open clipboard.
        unsafe { IsClipboardFormatAvailable(CF_DIB) != 0 }
    }
}

/// Linux tools that list the clipboard's offered types, one per line.
#[cfg_attr(
    any(target_os = "macos", windows),
    allow(dead_code, reason = "the Linux probe only")
)]
fn image_target_tools(wayland: bool) -> Vec<Tool> {
    let mut tools = Vec::new();
    if wayland {
        tools.push(Tool::new("wl-paste", &["--list-types"]));
    }
    tools.push(Tool::new(
        "xclip",
        &["-selection", "clipboard", "-t", "TARGETS", "-o"],
    ));
    tools
}

/// True when a type listing offers `image/png`.
#[cfg_attr(
    any(target_os = "macos", windows),
    allow(dead_code, reason = "the Linux probe only")
)]
fn lists_png(listing: &[u8]) -> bool {
    listing
        .split(|&b| b == b'\n')
        .any(|line| line.trim_ascii() == b"image/png")
}

/// The clipboard image as PNG bytes, or `None` when it holds none.
#[must_use]
pub fn clipboard_png() -> Option<Vec<u8>> {
    #[cfg(target_os = "macos")]
    {
        // `pngpaste` first (fast when installed), then an `osascript` that
        // writes the clipboard's PNG data to a temp file.
        Tool::new("pngpaste", &["-"])
            .read()
            .filter(|b| !b.is_empty())
            .or_else(|| {
                via_temp_file(|tmp| {
                    let script = format!(
                        "set png to the clipboard as «class PNGf»\n\
                         set f to open for access POSIX file \"{tmp}\" with write permission\n\
                         set eof of f to 0\n\
                         write png to f\n\
                         close access f"
                    );
                    Tool::new("osascript", &["-e", &script])
                })
            })
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let mut tools = Vec::new();
        if wayland() {
            tools.push(Tool::new(
                "wl-paste",
                &["--no-newline", "--type", "image/png"],
            ));
        }
        tools.push(Tool::new(
            "xclip",
            &["-selection", "clipboard", "-t", "image/png", "-o"],
        ));
        tools
            .iter()
            .find_map(Tool::read)
            .filter(|b| b.starts_with(b"\x89PNG"))
    }
    #[cfg(windows)]
    {
        via_temp_file(|tmp| {
            Tool::powershell(&format!(
                "Add-Type -AssemblyName System.Windows.Forms, System.Drawing; \
                 $i = [System.Windows.Forms.Clipboard]::GetImage(); \
                 if ($i) {{ $i.Save({}, [System.Drawing.Imaging.ImageFormat]::Png) }}",
                ps_quote(tmp)
            ))
        })
    }
}

/// Runs the tool `make` builds for a temp path, then reads and removes the
/// file it wrote.
#[cfg(any(target_os = "macos", windows))]
fn via_temp_file(make: impl FnOnce(&str) -> Tool) -> Option<Vec<u8>> {
    let tmp = std::env::temp_dir().join(format!("plank-clipboard-{}.png", std::process::id()));
    let ok = make(tmp.to_str()?)
        .command()
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let bytes = ok.then(|| std::fs::read(&tmp).ok()).flatten();
    let _ = std::fs::remove_file(&tmp);
    bytes.filter(|b| !b.is_empty())
}

/// The command that hands `path` (or a URL) to its default application.
///
/// Windows uses `explorer.exe`, which opens a path with its associated app and
/// takes it as one argument; `cmd /C start` would re-parse it, so a file name
/// holding `&` or `^` would run as a command.
#[must_use]
pub fn open_tool(path: &Path) -> Tool {
    let program = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    Tool {
        program,
        args: vec![path.as_os_str().to_os_string()],
    }
}

/// Opens `path` in its default application, without waiting: `xdg-open` can
/// outlive the app it starts, and plank must not block on it.
///
/// # Errors
/// When the opener cannot be spawned, e.g. a headless Linux box with no
/// `xdg-open`.
pub fn open_detached(path: &Path) -> std::io::Result<()> {
    open_tool(path).spawn().map(drop)
}

/// Players for a WAV file, tried in order.
#[must_use]
pub fn sound_tools(wav: &Path) -> Vec<Tool> {
    if cfg!(windows) {
        return vec![Tool::powershell(&format!(
            "(New-Object Media.SoundPlayer {}).PlaySync()",
            ps_quote(&wav.to_string_lossy())
        ))];
    }
    let players: [(&'static str, &[&str]); 4] = [
        ("afplay", &[]),
        ("paplay", &[]),
        ("aplay", &["-q"]),
        ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ];
    players
        .into_iter()
        .map(|(program, args)| {
            let mut tool = Tool::new(program, args);
            tool.args.push(wav.as_os_str().to_os_string());
            tool
        })
        .collect()
}

/// Starts the first player for `wav` that spawns. `None` when none is
/// installed. The caller owns the child, to stop the sound by killing it.
#[must_use]
pub fn play_sound(wav: &Path) -> Option<std::process::Child> {
    sound_tools(wav).iter().find_map(|t| t.spawn().ok())
}

/// The desktop notification command for Linux and Windows. macOS delivers
/// through `notify-rust` in `crate::notify` instead.
///
/// `sticky` asks for a notification that stays until dismissed: `critical`
/// urgency for `notify-send`, a long-duration toast on Windows.
#[must_use]
pub fn notify_tool(summary: &str, subtitle: Option<&str>, body: &str, sticky: bool) -> Tool {
    let text = match subtitle {
        Some(sub) if !sub.is_empty() => format!("{sub}\n{body}"),
        _ => body.to_owned(),
    };
    if cfg!(windows) {
        let duration = if sticky { "long" } else { "short" };
        let xml = format!(
            "<toast duration=\"{duration}\"><visual><binding template=\"ToastGeneric\">\
             <text>{}</text><text>{}</text></binding></visual></toast>",
            xml_escape(summary),
            xml_escape(&text)
        );
        // A toast needs an AppUserModelID; an unpackaged CLI has none, so it
        // borrows Windows PowerShell's, which every install registers.
        return Tool::powershell(&format!(
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; \
             [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null; \
             $x = New-Object Windows.Data.Xml.Dom.XmlDocument; $x.LoadXml({}); \
             [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}}\\WindowsPowerShell\\v1.0\\powershell.exe').Show([Windows.UI.Notifications.ToastNotification]::new($x))",
            ps_quote(&xml)
        ));
    }
    let urgency = if sticky { "critical" } else { "normal" };
    let mut tool = Tool::new("notify-send", &["--app-name=plank", "--urgency", urgency]);
    // `--` so a summary starting with `-` is not read as an option.
    tool.args.push("--".into());
    tool.args.push(summary.into());
    tool.args.push(text.into());
    tool
}

/// Shows a desktop notification on Linux or Windows and reaps the helper.
/// Blocks until the helper exits, so call it off the UI thread.
pub fn notify(summary: &str, subtitle: Option<&str>, body: &str, sticky: bool) {
    if let Ok(mut child) = notify_tool(summary, subtitle, body, sticky).spawn() {
        let _ = child.wait();
    }
}

/// `s` escaped for XML text content.
#[cfg_attr(not(windows), allow(dead_code))]
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn programs(tools: &[Tool]) -> Vec<&'static str> {
        tools.iter().map(|t| t.program).collect()
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn linux_clipboard_prefers_wayland_then_x11() {
        assert_eq!(programs(&copy_tools(true)), ["wl-copy", "xclip", "xsel"]);
        assert_eq!(programs(&copy_tools(false)), ["xclip", "xsel"]);
        assert_eq!(programs(&paste_tools(true)), ["wl-paste", "xclip", "xsel"]);
        assert_eq!(programs(&paste_tools(false)), ["xclip", "xsel"]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_clipboard_is_pbcopy_and_pbpaste() {
        assert_eq!(programs(&copy_tools(true)), ["pbcopy"]);
        assert_eq!(programs(&paste_tools(true)), ["pbpaste"]);
    }

    #[cfg(windows)]
    #[test]
    fn windows_clipboard_is_powershell() {
        assert_eq!(programs(&copy_tools(false)), ["powershell"]);
        assert_eq!(programs(&paste_tools(false)), ["powershell"]);
    }

    #[test]
    fn a_type_listing_offers_png_only_on_its_own_line() {
        assert!(lists_png(b"TARGETS\ntext/plain\nimage/png\n"));
        assert!(lists_png(b"image/png\r\n"));
        assert!(!lists_png(b"text/plain\nimage/png-ish\n"));
        assert!(!lists_png(b""));
    }

    #[test]
    fn image_types_are_listed_with_wl_paste_first_on_wayland() {
        assert_eq!(programs(&image_target_tools(true)), ["wl-paste", "xclip"]);
        assert_eq!(programs(&image_target_tools(false)), ["xclip"]);
    }

    #[test]
    fn the_opener_takes_the_path_as_its_one_argument() {
        let path = Path::new("/work/a & b.html");
        let tool = open_tool(path);
        assert_eq!(tool.args, [path.as_os_str()]);
        let want = if cfg!(target_os = "macos") {
            "open"
        } else if cfg!(windows) {
            "explorer"
        } else {
            "xdg-open"
        };
        assert_eq!(tool.program, want);
    }

    #[test]
    fn every_unix_player_ends_with_the_file() {
        if cfg!(windows) {
            return;
        }
        let wav = Path::new("/tmp/clip.wav");
        let tools = sound_tools(wav);
        assert_eq!(programs(&tools), ["afplay", "paplay", "aplay", "ffplay"]);
        assert!(
            tools
                .iter()
                .all(|t| t.args.last().map(OsString::as_os_str) == Some(wav.as_os_str()))
        );
    }

    #[test]
    fn powershell_quoting_doubles_single_quotes() {
        assert_eq!(ps_quote("C:\\it's here.wav"), "'C:\\it''s here.wav'");
    }

    #[test]
    fn xml_escaping_covers_markup_characters() {
        assert_eq!(
            xml_escape(r#"<a href="x">&'"#),
            "&lt;a href=&quot;x&quot;&gt;&amp;&apos;"
        );
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn notify_send_gets_urgency_and_a_separator_before_the_text() {
        let tool = notify_tool("-done", Some("sub"), "body", true);
        assert_eq!(tool.program, "notify-send");
        let args: Vec<_> = tool.args.iter().map(|a| a.to_string_lossy()).collect();
        assert_eq!(
            args,
            [
                "--app-name=plank",
                "--urgency",
                "critical",
                "--",
                "-done",
                "sub\nbody"
            ]
        );
        assert!(
            notify_tool("t", None, "b", false)
                .args
                .contains(&OsString::from("normal"))
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_toast_escapes_its_text() {
        let tool = notify_tool("a<b", None, "it's", false);
        let script = tool
            .args
            .last()
            .expect("script")
            .to_string_lossy()
            .into_owned();
        assert!(script.contains("a&lt;b"), "{script}");
        assert!(script.contains("it&apos;s"), "{script}");
    }
}
