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
        // in the middle of the TUI.
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
    let result = runtime
        .block_on(parser.parse(&path.to_string_lossy()))
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
}
