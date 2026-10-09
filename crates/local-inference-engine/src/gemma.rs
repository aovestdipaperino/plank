// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The Gemma 4 backend over `crates/gemma-engine` (candle).
//!
//! [`Session`] gives Gemma the same shape as the ds4 session: `sync` keeps
//! the longest common prefix and prefills the rest, and the logits of the
//! last evaluated position are kept for [`Session::sample`] and
//! [`Session::top_logprobs`], which the ds4 engine keeps on its side.

use std::path::Path;
use std::sync::Arc;

use gemma_engine::Device;
use gemma_engine::model::Model as GemmaModel;
use gemma_engine::sample::Sampler;
use gemma_engine::session::Session as GemmaSession;
use gemma_engine::template::{self, Kind};

use crate::{Error, Think, TokenScore};

/// Positions evaluated per prefill step, reported as one progress event.
const CHUNK: usize = 512;

fn error(e: impl std::fmt::Display) -> Error {
    Error::new(e.to_string())
}

/// Metal when the machine has it and `cpu` is not forced, else the CPU.
fn pick_device(cpu: bool) -> Device {
    #[cfg(target_os = "macos")]
    if !cpu && let Ok(d) = Device::new_metal(0) {
        return d;
    }
    let _ = cpu;
    Device::Cpu
}

/// Opens the Gemma 4 GGUF at `path` with rotary tables for `ctx` positions
/// (clamped to the model's own context).
pub(crate) fn open(path: &Path, ctx: usize, cpu: bool) -> Result<Arc<GemmaModel>, Error> {
    GemmaModel::open_with_ctx(path, &pick_device(cpu), ctx.max(1)).map_err(error)
}

/// The one-turn chat the ds4 CLI would build, in Gemma's template: BOS, the
/// system turn (carrying `<|think|>` when thinking, so it is rendered even
/// with no system text), the user turn, and the model-turn opener.
///
/// The system text is tokenized as plain text: a caller's string never turns
/// into control tokens.
pub(crate) fn encode_chat(
    model: &GemmaModel,
    system: &str,
    prompt: &str,
    think: Think,
) -> Vec<i32> {
    let think = think != Think::Off;
    let tok = &model.tokenizer;
    let mut ids: Vec<u32> = tok.bos().into_iter().collect();
    let mut prev = None;
    if !system.is_empty() || think {
        ids.extend(tok.encode_pieces(&template::render(Kind::System, system, None, think, 0)));
        prev = Some(Kind::System);
    }
    ids.extend(tok.encode_pieces(&template::render(Kind::User, prompt, prev, think, 0)));
    ids.extend(tok.encode_pieces(&template::generation_prefix(Some(Kind::User))));
    to_i32(&ids)
}

/// Tokenizes text in which control spellings become their control ids.
pub(crate) fn tokenize_rendered(model: &GemmaModel, text: &str) -> Vec<i32> {
    to_i32(&model.tokenizer.encode_trusted(text))
}

pub(crate) fn token_bytes(model: &GemmaModel, token: i32) -> Vec<u8> {
    u32::try_from(token).map_or_else(|_| Vec::new(), |t| model.tokenizer.piece_bytes(t))
}

pub(crate) fn eos(model: &GemmaModel) -> i32 {
    i32::try_from(model.tokenizer.eos()).unwrap_or(-1)
}

fn to_i32(ids: &[u32]) -> Vec<i32> {
    ids.iter()
        .map(|&t| i32::try_from(t).unwrap_or(-1))
        .collect()
}

fn to_u32(ids: &[i32]) -> Result<Vec<u32>, Error> {
    ids.iter()
        .map(|&t| u32::try_from(t).map_err(|_| Error::new(format!("invalid token id {t}"))))
        .collect()
}

/// A Gemma session plus the logits of its last evaluated position.
#[derive(Debug)]
pub(crate) struct Session {
    inner: GemmaSession,
    logits: Option<Vec<f32>>,
}

impl Session {
    pub(crate) fn new(model: &Arc<GemmaModel>, ctx: usize) -> Self {
        Self {
            inner: GemmaSession::new(Arc::clone(model), ctx.max(1)),
            logits: None,
        }
    }

    pub(crate) fn inner(&self) -> &GemmaSession {
        &self.inner
    }

    pub(crate) fn ctx(&self) -> usize {
        self.inner.ctx()
    }

    pub(crate) fn pos(&self) -> usize {
        self.inner.tokens().len()
    }

    pub(crate) fn invalidate(&mut self) {
        self.inner.truncate(0);
        self.logits = None;
    }

    /// Makes the cache hold exactly `tokens`, reusing the common prefix.
    ///
    /// The last token is always evaluated again when the whole prompt is
    /// already cached and no logits are held for it, so sampling right after
    /// a sync always has something to sample from.
    pub(crate) fn sync(
        &mut self,
        tokens: &[i32],
        on_progress: &mut dyn FnMut(&str, i32, i32),
    ) -> Result<(), Error> {
        let tokens = to_u32(tokens)?;
        if tokens.len() > self.inner.ctx() {
            return Err(Error::new(format!(
                "context full: {} tokens > {}",
                tokens.len(),
                self.inner.ctx()
            )));
        }
        let common = self
            .inner
            .tokens()
            .iter()
            .zip(&tokens)
            .take_while(|(a, b)| a == b)
            .count();
        let unchanged = common == tokens.len() && common == self.inner.tokens().len();
        if unchanged && self.logits.is_some() {
            return Ok(());
        }
        // Keep at most all but the last token, so the final position's
        // logits are recomputed.
        let keep = self
            .inner
            .truncate(common.min(tokens.len().saturating_sub(1)));
        self.logits = None;
        let total = i32::try_from(tokens.len()).unwrap_or(i32::MAX);
        for chunk in tokens[keep..].chunks(CHUNK) {
            match self.inner.prefill(chunk, &|| false) {
                Ok(logits) => self.logits = logits,
                Err(e) => {
                    self.logits = None;
                    return Err(error(e));
                }
            }
            let pos = i32::try_from(self.inner.tokens().len()).unwrap_or(i32::MAX);
            on_progress("prefill_chunk", pos, total);
        }
        Ok(())
    }

    pub(crate) fn eval(&mut self, token: i32) -> Result<(), Error> {
        let token =
            u32::try_from(token).map_err(|_| Error::new(format!("invalid token id {token}")))?;
        self.logits = Some(self.inner.step(token).map_err(error)?);
        Ok(())
    }

    /// Samples from the held logits; `-1` when none are held. `top_k` keeps
    /// only the `k` likeliest tokens first (0 keeps all). `rng` seeds a
    /// `SplitMix64` draw and is advanced so the next call draws afresh.
    pub(crate) fn sample(
        &self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        let Some(logits) = &self.logits else {
            return -1;
        };
        let k = usize::try_from(top_k).unwrap_or(0);
        let masked;
        let logits = if k > 0 && k < logits.len() {
            let mut sorted = logits.clone();
            sorted.sort_unstable_by(|a, b| b.total_cmp(a));
            let floor = sorted[k - 1];
            masked = logits
                .iter()
                .map(|&l| if l >= floor { l } else { f32::NEG_INFINITY })
                .collect::<Vec<_>>();
            &masked
        } else {
            logits
        };
        let token = Sampler::new(*rng).sample(logits, temperature, top_p, min_p, false);
        *rng = splitmix(*rng);
        i32::try_from(token).unwrap_or(-1)
    }

    /// The `k` likeliest next tokens with log-probabilities over the whole
    /// vocabulary, best first; empty when no logits are held.
    pub(crate) fn top_logprobs(&self, k: usize) -> Vec<TokenScore> {
        let Some(logits) = &self.logits else {
            return Vec::new();
        };
        top_logprobs(logits, k)
    }
}

/// One `SplitMix64` step, used to advance a caller's sampler seed.
fn splitmix(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `k` highest logits as scores, log-softmax normalised over all of them.
fn top_logprobs(logits: &[f32], k: usize) -> Vec<TokenScore> {
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
