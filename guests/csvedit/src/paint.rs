//! Turbo Vision's cell buffer as plank glyphs.

use plank_guest_support::CellGlyph;
use turbo_vision::core::draw::Cell;
use turbo_vision::core::palette::Style;

/// Every cell, background included: a Turbo Vision screen is defined by its
/// backgrounds, so none is left for the host to fill.
#[must_use]
pub fn cells(buffer: &[Vec<Cell>]) -> Vec<CellGlyph> {
    let mut out = Vec::with_capacity(buffer.len() * buffer.first().map_or(0, Vec::len));
    for (y, row) in buffer.iter().enumerate() {
        let Ok(y) = u16::try_from(y) else { break };
        for (x, cell) in row.iter().enumerate() {
            let Ok(x) = u16::try_from(x) else { break };
            out.push(CellGlyph {
                x,
                y,
                ch: cell.ch,
                fg: cell.attr.fg.to_rgb(),
                bg: Some(cell.attr.bg.to_rgb()),
                bold: cell.attr.style.contains(Style::BOLD),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_vision::core::palette::{Attr, TvColor};

    #[test]
    fn every_cell_becomes_a_glyph_with_its_background() {
        let plain = Attr::new(TvColor::White, TvColor::Blue);
        let buffer = vec![vec![Cell::new('a', plain), Cell::new('b', plain.bold())]];
        let glyphs = cells(&buffer);
        assert_eq!(glyphs.len(), 2);
        assert!(glyphs.iter().all(|g| g.bg == Some(TvColor::Blue.to_rgb())));
        assert_eq!((glyphs[1].x, glyphs[1].y, glyphs[1].ch), (1, 0, 'b'));
        assert!(!glyphs[0].bold);
        assert!(glyphs[1].bold);
    }
}
