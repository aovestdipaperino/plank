// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The packed glyph buffer a `frame` component returns
//! (`docs/WASM-PLUGINS.md`, "Glyph wire format").
//!
//! Every other payload in this system is JSON. This one is not, and the reason
//! is arithmetic: a 120×40 screen is 4800 glyphs, and at 30 fps that is 144k
//! JSON objects a second to encode on one side and parse on the other. Packed,
//! the same frame is a fixed-size read per glyph and nothing else.
//!
//! ```text
//! header: u32 magic 'PGLY' | u16 version | u16 count | u16 w | u16 h
//! glyph:  u16 x | u16 y | u32 ch (UTF-32) | u8 r | u8 g | u8 b | u8 flags
//! ```
//!
//! All integers are little-endian, which is what every wasm32 guest writes
//! natively. `flags` bit 0 is bold; bit 1 says a background color follows as
//! three more bytes, so the common case stays ten bytes per glyph.
//!
//! Decoding is total: any malformed buffer yields [`GlyphError`] rather than a
//! panic or a partial frame. A component that miscounts its own glyphs must
//! lose its frame, never the session.

use crate::arcade::Glyph;

/// `PGLY`, little-endian.
const MAGIC: u32 = u32::from_le_bytes(*b"PGLY");

/// Wire format version, bumped only if the layout changes incompatibly.
pub const GLYPH_VERSION: u16 = 1;

/// Bytes of fixed header.
const HEADER_LEN: usize = 4 + 2 + 2 + 2 + 2;

/// Bytes of a glyph without a background color.
const GLYPH_LEN: usize = 2 + 2 + 4 + 3 + 1;

/// Bit 0: draw bold.
pub const FLAG_BOLD: u8 = 1;
/// Bit 1: three background-color bytes follow this glyph.
pub const FLAG_BG: u8 = 2;

/// Why a buffer could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlyphError {
    /// Shorter than a header, or a glyph runs off the end.
    Truncated,
    /// The magic is not `PGLY` — the component returned something else
    /// entirely, most likely JSON.
    NotAGlyphBuffer,
    /// A version this build does not decode.
    Version(u16),
    /// More glyphs than the declared area could hold. A frame is a screen,
    /// not a stream: a component claiming ten thousand glyphs for an 80×24
    /// area has miscounted, and decoding it would allocate on its say-so.
    TooManyGlyphs { count: usize, capacity: usize },
}

impl std::fmt::Display for GlyphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "glyph buffer is truncated"),
            Self::NotAGlyphBuffer => write!(f, "not a glyph buffer (bad magic)"),
            Self::Version(v) => write!(f, "glyph buffer version {v} is not supported"),
            Self::TooManyGlyphs { count, capacity } => {
                write!(
                    f,
                    "glyph buffer declares {count} glyphs for {capacity} cells"
                )
            }
        }
    }
}

/// One decoded frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GlyphFrame {
    /// The area the component drew for, which may differ from the area it was
    /// given if it is a frame behind.
    pub w: u16,
    /// Rows drawn for.
    pub h: u16,
    /// The glyphs, in the order the component emitted them.
    pub glyphs: Vec<Glyph>,
    /// Bold flags, parallel to `glyphs`. Kept alongside rather than inside
    /// [`Glyph`] so the arcade's own type — which the built-in faces share and
    /// which has no notion of bold — does not have to grow a field for this.
    pub bold: Vec<bool>,
    /// Background colors, parallel to `glyphs`; `None` where the glyph carried
    /// no background.
    pub bg: Vec<Option<(u8, u8, u8)>>,
}

/// Decodes a packed buffer.
///
/// # Errors
/// [`GlyphError`] for any malformed input; never panics and never allocates on
/// a declared count it has not validated against the declared area.
pub fn decode(bytes: &[u8]) -> Result<GlyphFrame, GlyphError> {
    if bytes.len() < HEADER_LEN {
        return Err(GlyphError::Truncated);
    }
    let u32_at =
        |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);

    if u32_at(0) != MAGIC {
        return Err(GlyphError::NotAGlyphBuffer);
    }
    let version = u16_at(4);
    if version != GLYPH_VERSION {
        return Err(GlyphError::Version(version));
    }
    let count = u16_at(6) as usize;
    let (width, height) = (u16_at(8), u16_at(10));

    // A frame is a screen: at most one glyph per cell, plus a small allowance
    // for a component that overdraws a cell it already wrote. Validated before
    // the allocation below, so a bad count costs nothing.
    let capacity = (usize::from(width) * usize::from(height))
        .saturating_mul(2)
        .max(64);
    if count > capacity {
        return Err(GlyphError::TooManyGlyphs { count, capacity });
    }

    let mut out = GlyphFrame {
        w: width,
        h: height,
        glyphs: Vec::with_capacity(count),
        bold: Vec::with_capacity(count),
        bg: Vec::with_capacity(count),
    };
    let mut at = HEADER_LEN;
    for _ in 0..count {
        if at + GLYPH_LEN > bytes.len() {
            return Err(GlyphError::Truncated);
        }
        let col = u16_at(at);
        let row = u16_at(at + 2);
        let scalar = u32_at(at + 4);
        let color = (bytes[at + 8], bytes[at + 9], bytes[at + 10]);
        let flags = bytes[at + 11];
        at += GLYPH_LEN;

        let bg = if flags & FLAG_BG != 0 {
            if at + 3 > bytes.len() {
                return Err(GlyphError::Truncated);
            }
            let triple = (bytes[at], bytes[at + 1], bytes[at + 2]);
            at += 3;
            Some(triple)
        } else {
            None
        };

        // An invalid scalar value is replaced, not rejected: one bad character
        // in a frame is a rendering artifact, and losing the whole frame over
        // it would be a far worse failure than a replacement glyph.
        let ch = char::from_u32(scalar).unwrap_or(char::REPLACEMENT_CHARACTER);
        out.glyphs.push(Glyph {
            x: col,
            y: row,
            ch,
            color,
        });
        out.bold.push(flags & FLAG_BOLD != 0);
        out.bg.push(bg);
    }
    Ok(out)
}

/// Encodes a frame. Used by tests and by any Rust guest that wants the host's
/// own encoder rather than hand-rolling the layout.
///
/// Glyphs beyond `u16::MAX` are dropped: the count field cannot describe them,
/// and silently writing a count that disagrees with the body would produce a
/// buffer that fails to decode for reasons the author could not see.
#[must_use]
pub fn encode(frame: &GlyphFrame) -> Vec<u8> {
    let count = u16::try_from(frame.glyphs.len()).unwrap_or(u16::MAX);
    let mut out = Vec::with_capacity(HEADER_LEN + usize::from(count) * GLYPH_LEN);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&GLYPH_VERSION.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&frame.w.to_le_bytes());
    out.extend_from_slice(&frame.h.to_le_bytes());
    for (idx, glyph) in frame.glyphs.iter().take(usize::from(count)).enumerate() {
        let bold = frame.bold.get(idx).copied().unwrap_or(false);
        let bg = frame.bg.get(idx).copied().flatten();
        let mut flags = 0;
        if bold {
            flags |= FLAG_BOLD;
        }
        if bg.is_some() {
            flags |= FLAG_BG;
        }
        out.extend_from_slice(&glyph.x.to_le_bytes());
        out.extend_from_slice(&glyph.y.to_le_bytes());
        out.extend_from_slice(&(glyph.ch as u32).to_le_bytes());
        out.extend_from_slice(&[glyph.color.0, glyph.color.1, glyph.color.2, flags]);
        if let Some(triple) = bg {
            out.extend_from_slice(&[triple.0, triple.1, triple.2]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_of(glyphs: Vec<Glyph>) -> GlyphFrame {
        let n = glyphs.len();
        GlyphFrame {
            w: 80,
            h: 24,
            glyphs,
            bold: vec![false; n],
            bg: vec![None; n],
        }
    }

    #[test]
    fn a_frame_round_trips() {
        let mut f = frame_of(vec![
            Glyph {
                x: 1,
                y: 2,
                ch: 'A',
                color: (10, 20, 30),
            },
            Glyph {
                x: 79,
                y: 23,
                ch: '★',
                color: (255, 0, 128),
            },
        ]);
        f.bold[1] = true;
        f.bg[0] = Some((1, 2, 3));

        let decoded = decode(&encode(&f)).expect("round trip");
        assert_eq!(decoded, f);
        assert_eq!(decoded.glyphs[1].ch, '★', "non-ASCII survives as UTF-32");
    }

    #[test]
    fn an_empty_frame_is_valid() {
        let decoded = decode(&encode(&frame_of(vec![]))).expect("valid");
        assert!(decoded.glyphs.is_empty());
        assert_eq!((decoded.w, decoded.h), (80, 24));
    }

    /// A component returning JSON — the commonest mistake, since every other
    /// export in the ABI returns JSON — must be told what is actually wrong.
    #[test]
    fn json_is_rejected_as_not_a_glyph_buffer() {
        assert_eq!(
            decode(b"{\"glyphs\": []}"),
            Err(GlyphError::NotAGlyphBuffer)
        );
        assert_eq!(decode(b""), Err(GlyphError::Truncated));
        assert_eq!(decode(b"PGL"), Err(GlyphError::Truncated));
    }

    #[test]
    fn a_truncated_glyph_is_rejected_rather_than_partially_drawn() {
        let good = encode(&frame_of(vec![Glyph {
            x: 0,
            y: 0,
            ch: 'x',
            color: (1, 1, 1),
        }]));
        for cut in HEADER_LEN..good.len() {
            assert_eq!(
                decode(&good[..cut]),
                Err(GlyphError::Truncated),
                "a buffer cut at {cut} decoded"
            );
        }
    }

    /// The count is the component's claim about its own buffer, and the
    /// allocation follows it — so it is checked against the declared area
    /// first. Otherwise a two-byte edit asks the host for a huge allocation.
    #[test]
    fn an_impossible_glyph_count_is_refused_before_allocating() {
        let mut bytes = encode(&frame_of(vec![]));
        bytes[6..8].copy_from_slice(&u16::MAX.to_le_bytes());
        // 80×24 = 1920 cells, doubled = 3840 allowed.
        assert_eq!(
            decode(&bytes),
            Err(GlyphError::TooManyGlyphs {
                count: 65535,
                capacity: 3840
            })
        );
    }

    /// A tiny area still gets a floor, so a component that draws a few glyphs
    /// before it knows its size is not refused for it.
    #[test]
    fn a_zero_area_frame_still_allows_a_few_glyphs() {
        let mut f = frame_of(vec![Glyph {
            x: 0,
            y: 0,
            ch: 'x',
            color: (0, 0, 0),
        }]);
        f.w = 0;
        f.h = 0;
        assert!(decode(&encode(&f)).is_ok());
    }

    /// One bad scalar must not cost the frame: it is replaced and the rest of
    /// the buffer decodes.
    #[test]
    fn an_invalid_char_becomes_the_replacement_glyph() {
        let mut bytes = encode(&frame_of(vec![Glyph {
            x: 0,
            y: 0,
            ch: 'a',
            color: (0, 0, 0),
        }]));
        // A surrogate: valid u32, not a valid char.
        bytes[HEADER_LEN + 4..HEADER_LEN + 8].copy_from_slice(&0xD800_u32.to_le_bytes());
        let decoded = decode(&bytes).expect("still decodes");
        assert_eq!(decoded.glyphs[0].ch, char::REPLACEMENT_CHARACTER);
    }

    #[test]
    fn a_future_version_is_named_rather_than_guessed_at() {
        let mut bytes = encode(&frame_of(vec![]));
        bytes[4..6].copy_from_slice(&99_u16.to_le_bytes());
        assert_eq!(decode(&bytes), Err(GlyphError::Version(99)));
    }
}
