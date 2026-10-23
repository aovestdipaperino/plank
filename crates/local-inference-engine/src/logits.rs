// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Logit helpers the candle backends (Gemma, Flint) share: the `top_k` mask,
//! the sampler-seed step and the log-probability read-out.

use std::borrow::Cow;

use crate::TokenScore;

/// `logits` with everything below the `k`-th largest set to `-inf`; the
/// input itself when `k` is 0 or covers every logit.
pub(crate) fn top_k_mask(logits: &[f32], k: i32) -> Cow<'_, [f32]> {
    let k = usize::try_from(k).unwrap_or(0);
    if k == 0 || k >= logits.len() {
        return Cow::Borrowed(logits);
    }
    let mut sorted = logits.to_vec();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let floor = sorted[k - 1];
    Cow::Owned(
        logits
            .iter()
            .map(|&l| if l >= floor { l } else { f32::NEG_INFINITY })
            .collect(),
    )
}

/// One `SplitMix64` step, used to advance a caller's sampler seed.
pub(crate) fn splitmix(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `k` highest logits as scores, log-softmax normalised over all of them.
pub(crate) fn top_logprobs(logits: &[f32], k: usize) -> Vec<TokenScore> {
    let max = logits
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Vec::new();
    }
    let sum: f64 = logits
        .iter()
        .filter(|l| l.is_finite())
        .map(|&l| f64::from(l - max).exp())
        .sum();
    #[allow(clippy::cast_possible_truncation, reason = "a log-sum-exp fits an f32")]
    let lse = max + sum.ln() as f32;
    let mut order: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    order
        .into_iter()
        .take(k)
        .map(|i| TokenScore {
            id: i32::try_from(i).unwrap_or(-1),
            logit: logits[i],
            logprob: logits[i] - lse,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_logprobs_are_ordered_and_normalised() {
        let scores = top_logprobs(&[0.0, 2.0, 1.0, f32::NEG_INFINITY], 2);
        assert_eq!(scores.iter().map(|s| s.id).collect::<Vec<_>>(), [1, 2]);
        let all: f32 = top_logprobs(&[0.0, 2.0, 1.0], 3)
            .iter()
            .map(|s| s.logprob.exp())
            .sum();
        assert!((all - 1.0).abs() < 1e-5, "{all}");
    }

    #[test]
    fn non_finite_logits_give_no_scores() {
        assert!(top_logprobs(&[f32::NAN, f32::NEG_INFINITY], 2).is_empty());
    }
}
