//! Image pasting for the TUI prompt (macOS).
//!
//! Port of Claude Code's image-paste pipeline (see the vault note
//! `image-paste-prompt.md`) adapted to plank: an empty bracketed paste means
//! an image is on the clipboard; pasted text that is a path to an image file
//! attaches that file. Images are downscaled to API-style limits, stored in a
//! content-addressed LRU cache, and attached to the outgoing message as file
//! references the model can open with its tools (the ds4 engine is text-only,
//! so there is no base64 content-block path).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Maximum width/height after downsampling, mirroring `IMAGE_MAX_WIDTH/HEIGHT`.
const MAX_DIMENSION: u32 = 2000;
/// Maximum images kept in the cache (LRU by modification time).
const MAX_CACHED_IMAGES: usize = 200;

/// A pasted image stored in the cache, ready to attach to a message.
#[derive(Debug, Clone)]
pub struct PastedImage {
    /// Content-addressed path in the image cache.
    pub path: PathBuf,
    /// Pixel dimensions when known (PNG only; other formats pass through).
    pub dimensions: Option<(u32, u32)>,
    /// MIME type detected from magic bytes.
    pub media_type: &'static str,
    /// Original file path for path/drag pastes; `None` for clipboard images.
    pub source_path: Option<PathBuf>,
}

impl PastedImage {
    /// One-line description used for the attachment echo and the message text.
    #[must_use]
    pub fn describe(&self) -> String {
        let dims = self
            .dimensions
            .map_or(String::new(), |(w, h)| format!(", {w}x{h}"));
        format!("{} ({}{dims})", self.path.display(), self.media_type)
    }
}

/// Detects the media type from magic bytes; `None` when not a supported image.
#[must_use]
pub fn detect_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG") {
        Some("image/png")
    } else if bytes.starts_with(b"\xFF\xD8\xFF") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// True when the clipboard currently holds image data.
///
/// Cheap probe for the status-bar hint: `clipboard info for «class PNGf»`
/// prints the class and byte count when a PNG representation exists and
/// nothing otherwise (it exits 0 either way), without reading the bytes.
#[must_use]
pub fn clipboard_has_image() -> bool {
    Command::new("osascript")
        .args(["-e", "clipboard info for «class PNGf»"])
        .stderr(Stdio::null())
        .output()
        .is_ok_and(|out| out.status.success() && !out.stdout.trim_ascii().is_empty())
}

/// Reads an image off the system clipboard, processes it, and caches it.
///
/// Tries `pngpaste` first (fast when installed), then falls back to an
/// `osascript` that writes the clipboard's PNG data to a temp file.
#[must_use]
pub fn from_clipboard() -> Option<PastedImage> {
    let bytes = pngpaste_clipboard().or_else(osascript_clipboard)?;
    process_and_store(bytes, None)
}

/// Attaches an image file when the pasted text is a path to one.
///
/// Strips the quotes/escapes terminals add when a file is dragged onto them.
#[must_use]
pub fn from_path_text(pasted: &str) -> Option<PastedImage> {
    let cleaned = pasted
        .trim()
        .trim_matches('\'')
        .trim_matches('"')
        .replace("\\ ", " ");
    let has_image_ext = Path::new(&cleaned)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| {
            matches!(
                e.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp"
            )
        });
    if !has_image_ext {
        return None;
    }
    let bytes = std::fs::read(&cleaned).ok()?;
    process_and_store(bytes, Some(PathBuf::from(cleaned)))
}

/// `pngpaste -` writes the clipboard image as PNG to stdout.
fn pngpaste_clipboard() -> Option<Vec<u8>> {
    let out = Command::new("pngpaste").arg("-").output().ok()?;
    (out.status.success() && !out.stdout.is_empty()).then_some(out.stdout)
}

/// osascript fallback: writes `the clipboard as «class PNGf»` to a temp file.
fn osascript_clipboard() -> Option<Vec<u8>> {
    let tmp = std::env::temp_dir().join(format!("plank-clipboard-{}.png", std::process::id()));
    let tmp_str = tmp.to_str()?.to_owned();
    let script = format!(
        "set png to the clipboard as «class PNGf»\n\
         set f to open for access POSIX file \"{tmp_str}\" with write permission\n\
         set eof of f to 0\n\
         write png to f\n\
         close access f"
    );
    let status = Command::new("osascript")
        .args(["-e", &script])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?;
    let bytes = status.success().then(|| std::fs::read(&tmp).ok()).flatten();
    let _ = std::fs::remove_file(&tmp);
    bytes.filter(|b| !b.is_empty())
}

/// Downsamples (PNG only — the sole decoder plank links), dedups by SHA-256,
/// stores in the cache, and prunes it to the LRU limit.
fn process_and_store(mut bytes: Vec<u8>, source_path: Option<PathBuf>) -> Option<PastedImage> {
    let media_type = detect_media_type(&bytes)?;
    let mut dimensions = None;
    if media_type == "image/png"
        && let Ok(img) = image::load_from_memory(&bytes)
    {
        let img = if img.width() > MAX_DIMENSION || img.height() > MAX_DIMENSION {
            img.thumbnail(MAX_DIMENSION, MAX_DIMENSION)
        } else {
            img
        };
        dimensions = Some((img.width(), img.height()));
        let mut png = std::io::Cursor::new(Vec::new());
        if img.write_to(&mut png, image::ImageFormat::Png).is_ok() {
            bytes = png.into_inner();
        }
    }

    let dir = cache_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    let ext = match media_type {
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "png",
    };
    let path = dir.join(format!("{}.{ext}", sha256_hex(&bytes)?));
    if !path.exists() {
        std::fs::write(&path, &bytes).ok()?;
        prune_cache(&dir);
    }
    Some(PastedImage {
        path,
        dimensions,
        media_type,
        source_path,
    })
}

/// `~/.plank/image-cache`, mirroring Claude Code's `~/.claude/image-cache`.
fn cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".plank").join("image-cache"))
}

/// Content hash via the system `shasum` (macOS ships it; avoids a crypto dep).
fn sha256_hex(bytes: &[u8]) -> Option<String> {
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(bytes).ok()?;
    drop(child.stdin.take());
    let out = child.wait_with_output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let hash = text.split_whitespace().next()?.to_owned();
    (hash.len() == 64).then_some(hash)
}

/// Keeps the newest [`MAX_CACHED_IMAGES`] files, deleting the oldest extras.
fn prune_cache(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            meta.is_file().then_some((meta.modified().ok()?, e.path()))
        })
        .collect();
    if files.len() <= MAX_CACHED_IMAGES {
        return;
    }
    files.sort_by_key(|(t, _)| *t);
    let excess = files.len() - MAX_CACHED_IMAGES;
    for (_, path) in files.into_iter().take(excess) {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_type_detection() {
        assert_eq!(
            detect_media_type(b"\x89PNG\r\n\x1a\n rest"),
            Some("image/png")
        );
        assert_eq!(
            detect_media_type(b"\xFF\xD8\xFF\xE0 rest"),
            Some("image/jpeg")
        );
        assert_eq!(detect_media_type(b"GIF89a rest"), Some("image/gif"));
        assert_eq!(
            detect_media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            Some("image/webp")
        );
        assert_eq!(detect_media_type(b"plain text"), None);
        assert_eq!(detect_media_type(b""), None);
    }

    #[test]
    fn path_text_rejects_non_images() {
        assert!(from_path_text("/tmp/notes.txt").is_none());
        assert!(from_path_text("hello world").is_none());
        assert!(from_path_text("/tmp/definitely-missing-98127.png").is_none());
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc").as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
    }
}
