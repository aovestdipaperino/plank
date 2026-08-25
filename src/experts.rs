//! Expert-routing animation: a braille stand-in for the `MoE` router.
//!
//! `DeepSeek` V4 Flash routes each token through a handful of experts out of
//! hundreds, and the C engine computes that set per layer per token
//! (`layer_hash_selected_experts` / `layer_topk_selected_experts` in
//! `refs/ds4/ds4.c`). None of it is exported: `ds4.h` stops at
//! `ds4_engine_generate_argmax`, and on the Metal path the selection never
//! leaves the GPU. So this module does *not* report the real routing — it
//! derives a plausible one from the token being decoded.
//!
//! That is deliberate, and the shape is not arbitrary. The early DS4 layers
//! route by a pure table lookup on the token id, so "the token id determines
//! the expert set" is the same *kind* of function the model actually applies;
//! what differs is the table. The display therefore tells the truth about
//! sparsity (a few of many), about the routing changing every token, and about
//! the same token lighting the same experts — and nothing about which experts.
//!
//! Rendering is two braille cells, matching [`crate::status::THINK_MARK`]'s two
//! columns so swapping one for the other never reflows the status bar. Sixteen
//! dots cover [`N_EXPERT`] experts, so each dot stands for a *block* of them
//! and lights when enough of its block is routed.

/// Experts the router chooses among. The real shape comes from the GGUF
/// (`g_ds4_shape.n_expert`), which we cannot read without an engine, so this is
/// the V4 Flash figure held as a constant.
pub const N_EXPERT: usize = 256;

/// Experts routed per token (`DS4_N_EXPERT_USED`, capped at 8 by the C).
pub const N_EXPERT_USED: usize = 6;

/// Braille cells in the glyph — two, to match `THINK_MARK`'s display width.
/// Must keep `CELLS * 8` a divisor of [`N_EXPERT`] so every dot covers the same
/// number of experts.
pub const CELLS: usize = 2;

/// Experts covered by one dot.
pub const BLOCK: usize = N_EXPERT / (CELLS * 8);

/// Share of a block that must be routed for its dot to light.
///
/// Six of 256 experts is 2.3% sparsity, so one routed expert in a 16-expert
/// block is 6.25% — a threshold above that would leave the glyph blank almost
/// always, which is why this sits just under it rather than at some rounder
/// number. Kept as a percentage anyway: it is the knob to turn if the expert
/// shape (and with it `BLOCK`) ever changes.
pub const DOT_THRESHOLD_PCT: usize = 6;

/// Dot bits of `U+2800`, in reading order: left column then right, top row to
/// bottom. Braille numbers its dots down the columns instead, hence the table.
const DOT_BITS: [u8; 8] = [0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80];

/// One round of splitmix64 — enough mixing to spread neighbouring token ids
/// across the whole expert range, which is all the display asks of it.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The [`N_EXPERT_USED`] distinct experts `seed` routes to.
///
/// Distinctness matters for the look as much as for the fiction: a repeated id
/// would quietly cost the frame a lit dot.
#[must_use]
pub fn selected(seed: u64) -> [u16; N_EXPERT_USED] {
    let mut out = [0u16; N_EXPERT_USED];
    let mut n = 0;
    let mut h = mix(seed);
    // Rehashing until the slot is fresh terminates in practice — 6 draws from
    // 256 collide rarely — but the counter bounds it regardless, since a
    // status-bar redraw must never be the thing that hangs.
    for _ in 0..(N_EXPERT_USED * 16) {
        if n == N_EXPERT_USED {
            break;
        }
        let id = u16::try_from(h % N_EXPERT as u64).unwrap_or(0);
        if !out[..n].contains(&id) {
            out[n] = id;
            n += 1;
        }
        h = mix(h);
    }
    // A pathological run leaves the tail at expert 0, which is a duller frame,
    // not a wrong one.
    out
}

/// Renders `seed`'s routing as [`CELLS`] braille characters.
#[must_use]
pub fn glyphs(seed: u64) -> String {
    let mut counts = [0usize; CELLS * 8];
    for id in selected(seed) {
        let dot = usize::from(id) / BLOCK;
        if let Some(c) = counts.get_mut(dot) {
            *c += 1;
        }
    }
    (0..CELLS)
        .map(|cell| {
            let mut bits = 0u8;
            for dot in 0..8 {
                if counts[cell * 8 + dot] * 100 >= DOT_THRESHOLD_PCT * BLOCK {
                    bits |= DOT_BITS[dot];
                }
            }
            char::from_u32(0x2800 + u32::from(bits)).unwrap_or('⠀')
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_width::UnicodeWidthStr;

    /// The glyph has to be a drop-in for the brain emoji: same columns, always,
    /// whatever the routing is. A one-column frame would shove the whole status
    /// bar left for a single token. The divisor check keeps every dot worth the
    /// same number of experts.
    #[test]
    fn a_frame_is_always_the_brain_s_width() {
        let brain = UnicodeWidthStr::width(crate::status::THINK_MARK);
        assert_eq!(N_EXPERT % (CELLS * 8), 0, "dots must divide the experts");
        for seed in 0..500u64 {
            let g = glyphs(seed);
            assert_eq!(g.chars().count(), CELLS, "{seed}: {g:?}");
            assert_eq!(UnicodeWidthStr::width(g.as_str()), brain, "{seed}: {g:?}");
        }
    }

    /// Six experts of 256 must read as sparse *and* as alive: never the blank
    /// braille cell, never a solid block. Both failures look like a bug in the
    /// status bar rather than a routing display.
    #[test]
    fn every_frame_is_sparse_but_never_empty() {
        for seed in 0..500u64 {
            let lit: u32 = glyphs(seed)
                .chars()
                .map(|c| (u32::from(c) - 0x2800).count_ones())
                .sum();
            assert!(
                (1..=u32::try_from(N_EXPERT_USED).unwrap()).contains(&lit),
                "{seed}: {lit} dots lit"
            );
        }
    }

    /// The two properties the fiction is allowed to claim: the same token always
    /// lights the same experts, and neighbouring tokens do not light the same
    /// ones. Without the second the glyph would look frozen mid-generation.
    #[test]
    fn routing_is_stable_per_token_and_moves_between_tokens() {
        for seed in 0..200u64 {
            assert_eq!(glyphs(seed), glyphs(seed), "{seed} is not stable");
        }
        let frames: std::collections::HashSet<String> = (0..60u64).map(glyphs).collect();
        assert!(frames.len() > 20, "only {} distinct frames", frames.len());
    }

    /// Exactly `N_EXPERT_USED` distinct experts, all in range — the invariant
    /// `glyphs` counts on when it maps ids onto dots.
    #[test]
    fn selection_is_distinct_and_in_range() {
        for seed in 0..500u64 {
            let sel = selected(seed);
            let uniq: std::collections::HashSet<u16> = sel.iter().copied().collect();
            assert_eq!(uniq.len(), N_EXPERT_USED, "{seed}: {sel:?}");
            assert!(sel.iter().all(|&e| usize::from(e) < N_EXPERT), "{sel:?}");
        }
    }
}
