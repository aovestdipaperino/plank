// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Document ingestion: PDFs read as Markdown.
//!
//! Implements `docs/LIGHT-PARSE.md`. `read` on a `.pdf` converts the file to
//! Markdown through [`liteparse`] (spatial text extraction over PDFium, with
//! bundled Tesseract OCR filling in pages that have no text layer), caches the
//! result under `~/.plank/doc-cache`, and then hands the *cache path* to the
//! ordinary [`crate::tools::files::read_range`] machinery.
//!
//! That last step is the whole design. Paging, the line-number decoration, the
//! `continue_offset=` footer, and the `more` continuation are not reimplemented
//! here; a converted document is just a text file as far as the rest of the
//! tool layer is concerned. The only accommodation `read_range` makes is
//! carrying a display path, so the model never sees `~/.plank/doc-cache` in a
//! tool observation.
//!
//! Deliberately *not* here: a new tool. The system prompt's tool table is
//! frozen by `tests/c_parity.rs`, and appending to it would churn the Tier 1 KV
//! fingerprint and force a full re-prefill on every session. Extending `read`
//! by extension costs nothing on the prompt side.

#[cfg(feature = "docparse")]
pub(crate) mod cache;

use std::path::{Path, PathBuf};

/// Extensions `read` converts to Markdown before serving.
///
/// PDF only, on purpose. liteparse accepts DOCX/XLSX/PPTX, but it reaches them
/// by shelling out to `LibreOffice` or `ImageMagick` to make a PDF first — an
/// undeclared external dependency that would turn `read` on a `.docx` into
/// either a multi-second surprise or an obscure failure depending on what the
/// user happens to have installed. Office formats stay a non-goal until that is
/// handled explicitly.
const DOC_EXTENSIONS: &[&str] = &["pdf"];

/// True when `read` should route this path through conversion.
#[must_use]
pub fn is_document(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| DOC_EXTENSIONS.iter().any(|d| ext.eq_ignore_ascii_case(d)))
}

/// Converts `path` to Markdown, returning a text file the caller can read.
///
/// The returned path is inside the document cache, not the user's tree. On a
/// cache hit no parsing happens at all.
///
/// # Errors
///
/// Returns a bare message (no `Tool error:` prefix — the caller adds it) when
/// the file cannot be read or liteparse rejects it.
#[cfg(feature = "docparse")]
pub fn to_markdown(path: &Path, display: &str) -> Result<PathBuf, String> {
    // Deliberately not `read_file_bytes`: that enforces `FILE_MAX_BYTES`, the
    // cap on how much text a read may put in context. A PDF's bytes never
    // enter the context — they are input to a converter whose Markdown is then
    // paged like any other file — so the cap would reject exactly the large
    // documents this feature exists for. Only the hash is needed here, and
    // `shasum` streams the file itself.
    if !path.is_file() {
        return Err(format!("open {display}: not a readable file"));
    }
    let hash = crate::imagepaste::sha256_file_hex(path)
        .ok_or_else(|| format!("hash {display}: shasum unavailable"))?;
    if let Some(hit) = cache::lookup(&hash) {
        return Ok(hit);
    }
    let markdown = convert(path, display)?;
    cache::store(&hash, &markdown).ok_or_else(|| {
        format!("convert {display}: cannot write the document cache (~/.plank/doc-cache)")
    })
}

/// Runs liteparse over `path` and returns its Markdown rendering.
#[cfg(feature = "docparse")]
fn convert(path: &Path, display: &str) -> Result<String, String> {
    use liteparse::{LiteParse, LiteParseConfig, OutputFormat};

    let config = LiteParseConfig {
        output_format: OutputFormat::Markdown,
        // A page whose OCR fails should cost that page, not the document. The
        // crate defaults this to fatal; plank prefers 90% of a document over
        // none of it, matching how the file tools treat partial reads.
        ocr_failure_fatal: false,
        // liteparse logs timings to stderr unless silenced, which would land
        // in the middle of the TUI. This covers only the crate's own logging;
        // the bundled Tesseract's C++ prints bypass it and are held back by
        // StderrSilencer around the parse below.
        quiet: true,
        ..Default::default()
    };
    let parser = LiteParse::new(config);
    // liteparse is async and plank's tool dispatch is not; the same block-on
    // shim the obscura web tools use applies here. A current-thread runtime is
    // enough — the parse is CPU-bound and does its own internal threading.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("convert {display}: {e}"))?;
    let result = {
        let _silencer = StderrSilencer::engage();
        runtime.block_on(parser.parse(&path.to_string_lossy()))
    }
    .map_err(|e| format!("convert {display}: {e}"))?;

    if result.text.trim().is_empty() {
        return Err(format!(
            "convert {display}: no readable text ({} page(s); the file is likely a scan \
             that OCR could not resolve)",
            result.pages.len()
        ));
    }
    Ok(wrap_markdown(&result.text))
}

/// Routes fd 2 to `/dev/null` for the lifetime of the guard.
///
/// liteparse's `quiet` config only silences the crate's own logging. The
/// bundled Tesseract writes its C++ diagnostics (`Detected N diacritics` and
/// friends) straight to stderr through C stdio, which no Rust-side flag
/// reaches — and because plank parses in-process, those bytes land wherever
/// the terminal cursor happens to be: the TUI's prompt line. `dup2` is the
/// one lever that covers both Rust and C writers, since both bottom out at
/// `write(2, …)`.
///
/// Process-wide by nature: while engaged, *any* thread's stderr is
/// discarded. plank's own diagnostics go to a log file (see `errlog`) and
/// the TUI draws on stdout, so the only bytes lost are the parser's. A
/// static mutex serializes engagements: two overlapping guards could restore
/// the fd in the wrong order and strand fd 2 on `/dev/null` for good.
#[cfg(all(feature = "docparse", unix))]
struct StderrSilencer {
    /// The dup of the real fd 2, or -1 when engagement failed (the guard is
    /// then a no-op).
    saved: std::os::fd::RawFd,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(all(feature = "docparse", unix))]
static STDERR_SILENCE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(feature = "docparse", unix))]
impl StderrSilencer {
    fn engage() -> Self {
        // A poisoned lock still gets a guard: the previous holder panicking
        // does not make silencing this parse any less correct.
        let lock = STDERR_SILENCE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: every call uses valid arguments; failures are reported
        // through the return value / a negative fd, never UB.
        let saved = unsafe {
            let saved = libc::dup(libc::STDERR_FILENO);
            if saved < 0 {
                return Self { saved, _lock: lock };
            }
            let null_fd = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
            if null_fd < 0 || libc::dup2(null_fd, libc::STDERR_FILENO) < 0 {
                if null_fd >= 0 {
                    libc::close(null_fd);
                }
                libc::close(saved);
                return Self {
                    saved: -1,
                    _lock: lock,
                };
            }
            libc::close(null_fd);
            saved
        };
        Self { saved, _lock: lock }
    }
}

#[cfg(all(feature = "docparse", unix))]
impl Drop for StderrSilencer {
    fn drop(&mut self) {
        if self.saved < 0 {
            return;
        }
        // SAFETY: `saved` is a live dup of the original fd 2 held by this
        // guard, and the lock guarantees no other guard is mid-redirect.
        unsafe {
            // Flush C stdio while fd 2 still points at /dev/null, so a
            // buffered diagnostic cannot slip out after the fd comes back.
            libc::fflush(std::ptr::null_mut());
            libc::dup2(self.saved, libc::STDERR_FILENO);
            libc::close(self.saved);
        }
    }
}

/// No fd redirection without a unix fd table: the guard is a no-op there.
#[cfg(all(feature = "docparse", not(unix)))]
struct StderrSilencer;

#[cfg(all(feature = "docparse", not(unix)))]
impl StderrSilencer {
    fn engage() -> Self {
        Self
    }
}

/// Column the converted Markdown is wrapped to before caching.
#[cfg(any(feature = "docparse", test))]
const WRAP_COLUMNS: usize = 100;

/// Hard-wraps prose paragraphs so line-based paging stays meaningful.
///
/// liteparse reflows each paragraph onto one line, so a page of prose arrives
/// as a single 5000-character line. `read` windows by *line*, so without this
/// an entire document is one or two lines: `max_lines` stops bounding the
/// output, the `continue_offset=` footer becomes meaningless, and `more` has
/// nothing to continue. Wrapping restores the correspondence between lines and
/// screenfuls that every other file the tool serves already has.
///
/// Table rows, fenced code, and headings pass through untouched — wrapping a
/// `|`-delimited row would destroy the table, and a wrapped heading stops being
/// one.
#[cfg(any(feature = "docparse", test))]
fn wrap_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + text.len() / 8);
    let mut in_fence = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
        }
        if in_fence
            || line.chars().count() <= WRAP_COLUMNS
            || trimmed.starts_with('|')
            || trimmed.starts_with('#')
        {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        // Preserve the leading indent on every continuation line so list items
        // and block quotes keep their shape.
        let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
        let mut width = 0;
        let mut first = true;
        out.push_str(&indent);
        for word in trimmed.split_whitespace() {
            let len = word.chars().count();
            if !first && width + 1 + len > WRAP_COLUMNS.saturating_sub(indent.len()) {
                out.push('\n');
                out.push_str(&indent);
                width = 0;
                first = true;
            }
            if !first {
                out.push(' ');
                width += 1;
            }
            out.push_str(word);
            width += len;
            first = false;
        }
        out.push('\n');
    }
    out
}

/// Conversion is unavailable in builds without the `docparse` feature.
///
/// # Errors
///
/// Always: this build has no document parser linked in.
#[cfg(not(feature = "docparse"))]
pub fn to_markdown(_path: &Path, display: &str) -> Result<PathBuf, String> {
    Err(format!(
        "read {display}: document conversion is not available in this build \
         (rebuild with the `docparse` feature)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extension routing is case-insensitive and does not catch neighbours.
    #[test]
    fn recognises_pdf_only() {
        assert!(is_document(Path::new("a/b/report.pdf")));
        assert!(is_document(Path::new("REPORT.PDF")));
        assert!(!is_document(Path::new("notes.md")));
        assert!(!is_document(Path::new("archive.pdf.gz")));
        assert!(!is_document(Path::new("pdf")));
        // Office formats are excluded until the LibreOffice dependency they
        // imply is handled; see DOC_EXTENSIONS.
        assert!(!is_document(Path::new("deck.pptx")));
    }

    /// A reflowed paragraph becomes several bounded lines, and no word is lost
    /// or split in the process.
    #[test]
    fn wraps_long_prose() {
        let para = "alpha beta gamma delta ".repeat(40);
        let wrapped = wrap_markdown(&para);
        assert!(wrapped.lines().count() > 5, "{wrapped}");
        assert!(wrapped.lines().all(|l| l.chars().count() <= WRAP_COLUMNS));
        assert_eq!(
            wrapped.split_whitespace().collect::<Vec<_>>(),
            para.split_whitespace().collect::<Vec<_>>()
        );
    }

    /// Structural lines survive verbatim even past the wrap column: wrapping a
    /// table row or a heading would change what it means.
    #[test]
    fn leaves_structure_unwrapped() {
        let row = format!("| {} |", "cell | ".repeat(30));
        let heading = format!("# {}", "a very long heading ".repeat(10));
        let fenced = format!("```\n{}\n```", "x".repeat(200));
        for input in [&row, &heading, &fenced] {
            let out = wrap_markdown(input);
            assert_eq!(out.trim_end(), input.trim_end(), "reflowed: {input}");
        }
    }

    /// Short lines and blank lines pass through unchanged.
    #[test]
    fn preserves_short_lines() {
        let text = "# Title\n\nshort line\n\nanother\n";
        assert_eq!(wrap_markdown(text), text);
    }

    /// Regression for the Tesseract stderr leak. The bundled OCR engine
    /// writes C++ diagnostics ("Detected N diacritics") straight to fd 2
    /// through C stdio, which liteparse's `quiet` cannot gate — and because
    /// plank parses in-process, those bytes land wherever the terminal cursor
    /// is: the TUI's prompt line. The child process converts a noisy scan
    /// whose OCR reliably triggers that print, then writes a sentinel to
    /// fd 2. The child's entire stderr must be exactly that sentinel, proving
    /// both that the parser was silenced and that fd 2 was handed back
    /// intact afterwards.
    ///
    /// A subprocess, not an in-process capture: fd redirection is
    /// process-wide, and parallel tests redirecting the same fd 2 would race.
    #[cfg(all(unix, feature = "docparse"))]
    #[test]
    fn parser_diagnostics_never_reach_stderr() {
        const CHILD_ENV: &str = "PLANK_DOC_STDERR_TEST_CHILD";
        const SENTINEL: &str = "stderr-alive-after-conversion";
        if std::env::var_os(CHILD_ENV).is_some() {
            let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("fixtures")
                .join("doc_noisy.pdf");
            let markdown = convert(&fixture, "doc_noisy.pdf").expect("noisy scan converts");
            assert!(
                markdown.contains("Caf"),
                "OCR lost the page text: {markdown:?}"
            );
            // Written via libc so no Rust-side buffering or test-output
            // capture interferes: this must reach fd 2 verbatim.
            unsafe {
                libc::write(
                    libc::STDERR_FILENO,
                    SENTINEL.as_ptr().cast(),
                    SENTINEL.len(),
                );
            }
            // A distinctive success code: a filter typo running zero tests
            // exits 0, a panic exits 101 — neither passes for success here.
            std::process::exit(42);
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "doc::tests::parser_diagnostics_never_reach_stderr",
                "--exact",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(42), "child failed: {out:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            SENTINEL,
            "parser wrote to stderr during conversion"
        );
    }
}
