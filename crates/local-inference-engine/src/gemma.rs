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

use crate::logits::{splitmix, top_k_mask, top_logprobs};
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
        let logits = top_k_mask(logits, top_k);
        let token = Sampler::new(*rng).sample(&logits, temperature, top_p, min_p, false);
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
