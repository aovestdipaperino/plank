//! A live Gemma 4 session: the token history and the KV cache that matches it.
//!
//! The cache is prefix-truncatable, so a new turn that shares a prefix with
//! the last one only prefills the suffix. A snapshot is
//! `b"PLGK"`, format `2u32`, the model signature (32 bytes), dtype `0u8`
//! (f32), `n_tokens: u32`, the tokens as `u32` LE, then the
//! [`KvCache`] body.
//!
//! Format 2 keeps only the last `sliding_window + SNAPSHOT_SLACK` positions
//! of each sliding layer (a later query reaches only its window; the slack
//! lets a restore truncate a little behind its end), recording per layer the
//! first position held; global layers are written whole. The reader takes any
//! recorded base, so the slack is the writer's choice alone. Format 1 (every
//! position of every layer) is refused, and the caller rebuilds by prefill.
//! A restored session can truncate exactly only while each trimmed layer
//! still covers the window the next query needs; below that it empties.

use std::sync::Arc;

use crate::kv::KvCache;
use crate::model::Model;
use crate::{Error, Result};

const MAGIC: &[u8; 4] = b"PLGK";
const FORMAT: u32 = 2;
const DTYPE_F32: u8 = 0;
/// magic + format + signature + dtype + `n_tokens`.
const HEADER: usize = 4 + 4 + 32 + 1 + 4;

/// Tokens a prefill evaluates per forward.
const CHUNK: usize = 512;

/// Positions a snapshot keeps on each sliding layer beyond its window.
///
/// A query at position `n` reads keys `n + 1 - window..=n`, so a layer holding
/// exactly its window lets a restored session truncate exactly only to its
/// last token. The agent's next turn often diverges a little further back
/// (a sidechain tail the engine has not dropped, a re-rendered last span), and
/// below that floor the session empties and re-prefills from zero. Keeping
/// `SNAPSHOT_SLACK` more positions moves the floor `SNAPSHOT_SLACK` tokens
/// further back: a restored `L`-token session truncates exactly to any
/// `n >= L - SNAPSHOT_SLACK - 1`. Cost: 64 positions per sliding layer, about
/// 42 MB on the 12B, against ~0.67 GB for its 1024-position window.
pub const SNAPSHOT_SLACK: usize = 64;

/// One conversation's tokens and KV cache over a shared [`Model`].
#[derive(Debug)]
pub struct Session {
    model: Arc<Model>,
    cache: KvCache,
    tokens: Vec<u32>,
    ctx: usize,
    /// Per KV layer, its sliding window (`None` for a global layer).
    windows: Vec<Option<usize>>,
    /// Positions [`Session::snapshot`] keeps past each sliding window:
    /// [`SNAPSHOT_SLACK`], or 0 in tests that pin the bare window.
    slack: usize,
}

impl Session {
    /// An empty session holding at most `ctx` tokens (never more than the
    /// model's [`Model::context_length`]).
    #[must_use]
    pub fn new(model: Arc<Model>, ctx: usize) -> Self {
        let n = model.kv_shape().len();
        let ctx = ctx.min(model.context_length());
        let windows = model.kv_windows();
        Self {
            model,
            cache: KvCache::new(n),
            tokens: Vec::new(),
            ctx,
            windows,
            slack: SNAPSHOT_SLACK,
        }
    }

    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// The tokens [`Session::truncate`] would keep for `n`: `min(n, len)`
    /// when the cache can still evaluate the next position after them, else
    /// 0.
    ///
    /// A fresh session keeps every position, so it always can. A restored
    /// one whose sliding layer starts at `base > 0` can only while the query
    /// at `n` still finds its whole window, `n + 1 - window >= base`.
    #[must_use]
    pub fn reusable(&self, n: usize) -> usize {
        let n = n.min(self.tokens.len());
        let covered = self
            .cache
            .layers
            .iter()
            .zip(&self.windows)
            .all(|(layer, window)| {
                layer.base() == 0 || window.is_some_and(|w| layer.base() + w <= n + 1)
            });
        if covered { n } else { 0 }
    }

    /// Keeps the first `n` tokens (a no-op when there are fewer) and returns
    /// how many are left. That is `min(n, len)` unless the cache can no
    /// longer represent that prefix ([`Session::reusable`] is 0): the session
    /// then empties, and the next prefill starts from scratch.
    pub fn truncate(&mut self, n: usize) -> usize {
        let keep = self.reusable(n);
        self.cache.truncate(keep);
        self.tokens.truncate(keep);
        keep
    }

    /// Evaluates `toks` after the current tokens, in chunks of 512, and
    /// returns the last position's logits.
    ///
    /// `interrupt` is polled before each chunk; when it answers `true` the
    /// prefill stops and returns `None`, keeping the chunks already
    /// evaluated.
    ///
    /// # Errors
    /// `context full: {need} tokens > {ctx}` before anything is evaluated,
    /// or a model failure, after which the session holds only the chunks
    /// that completed.
    pub fn prefill(
        &mut self,
        toks: &[u32],
        interrupt: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>> {
        let need = self.tokens.len() + toks.len();
        if need > self.ctx {
            return Err(Error(format!("context full: {need} tokens > {}", self.ctx)));
        }
        let mut last = None;
        for chunk in toks.chunks(CHUNK) {
            if interrupt() {
                return Ok(None);
            }
            let offset = self.tokens.len();
            match self.model.forward(chunk, offset, &mut self.cache) {
                Ok(logits) => last = Some(logits),
                Err(e) => {
                    // Owners that ran before the failure extended their cache.
                    self.cache.truncate(offset);
                    return Err(e);
                }
            }
            self.tokens.extend_from_slice(chunk);
        }
        Ok(last)
    }

    /// Evaluates one token and returns its logits.
    ///
    /// # Errors
    /// As [`Session::prefill`].
    pub fn step(&mut self, tok: u32) -> Result<Vec<f32>> {
        match self.prefill(&[tok], &|| false)? {
            Some(l) => Ok(l),
            None => Err(Error("step produced no logits".into())),
        }
    }

    /// The tokens and KV cache as one blob for [`Session::restore`].
    ///
    /// # Errors
    /// When the cache cannot be read back from the device.
    pub fn snapshot(&self) -> Result<Vec<u8>> {
        let keep: Vec<Option<usize>> = self
            .windows
            .iter()
            .map(|w| w.map(|w| w + self.slack))
            .collect();
        let body = self.cache.to_bytes_windowed(&keep)?;
        let n = u32::try_from(self.tokens.len()).map_err(|_| {
            Error(format!(
                "{} tokens do not fit a snapshot",
                self.tokens.len()
            ))
        })?;
        let mut out = Vec::with_capacity(HEADER + 4 * self.tokens.len() + body.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&self.model.signature());
        out.push(DTYPE_F32);
        out.extend_from_slice(&n.to_le_bytes());
        for t in &self.tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Replaces the session with a [`Session::snapshot`] blob.
    ///
    /// # Errors
    /// When any header field (magic, format, signature, dtype, token count)
    /// or the cache body does not match this model and context; the
    /// session is then unchanged.
    pub fn restore(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < HEADER {
            return Err(Error(format!(
                "snapshot: {} bytes is shorter than the header",
                bytes.len()
            )));
        }
        if &bytes[0..4] != MAGIC {
            return Err(Error("snapshot: bad magic".into()));
        }
        let format = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if format != FORMAT {
            return Err(Error(format!(
                "snapshot: format {format}, expected {FORMAT}"
            )));
        }
        if bytes[8..40] != self.model.signature() {
            return Err(Error("snapshot: signature is for another model".into()));
        }
        if bytes[40] != DTYPE_F32 {
            return Err(Error(format!(
                "snapshot: dtype {}, expected f32",
                bytes[40]
            )));
        }
        let n = u32::from_le_bytes([bytes[41], bytes[42], bytes[43], bytes[44]]) as usize;
        if n > self.ctx {
            return Err(Error(format!(
                "snapshot: n_tokens {n} exceeds the context of {}",
                self.ctx
            )));
        }
        let Some(body) = bytes.get(HEADER + 4 * n..) else {
            return Err(Error(format!(
                "snapshot: n_tokens {n} needs more bytes than the {} given",
                bytes.len()
            )));
        };
        let tokens: Vec<u32> = bytes[HEADER..HEADER + 4 * n]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let shape = self.model.kv_shape();
        let mut cache = KvCache::new(shape.len());
        cache.restore(body, self.model.device(), &shape)?;
        if let Some(i) = cache.layers.iter().position(|l| l.len() != n) {
            return Err(Error(format!(
                "snapshot: n_tokens {n} but KV layer {i} holds {}",
                cache.layers[i].len()
            )));
        }
        // A trimmed layer must still hold the window the next query reads.
        if let Some(i) = cache
            .layers
            .iter()
            .zip(&self.windows)
            .position(|(l, w)| l.base() > 0 && !w.is_some_and(|w| l.base() + w <= n + 1))
        {
            return Err(Error(format!(
                "snapshot: KV layer {i} starts at {} and cannot serve position {n}",
                cache.layers[i].base()
            )));
        }
        self.cache = cache;
        self.tokens = tokens;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgguf::{TinyConfig, write_tiny};
    use candle_core::Device;

    fn model() -> Arc<Model> {
        let path = std::env::temp_dir().join(format!(
            "gemma-sess-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        write_tiny(&path, &TinyConfig::default()).unwrap();
        Model::open(&path, &Device::Cpu).unwrap()
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    const PROMPT: [u32; 9] = [12, 300, 301, 302, 303, 304, 305, 306, 307];

    #[test]
    fn truncate_and_reprefill_equals_a_fresh_prefill() {
        let m = model();
        let mut fresh = Session::new(m.clone(), 128);
        let want = fresh.prefill(&PROMPT, &|| false).unwrap().unwrap();
        let mut s = Session::new(m, 128);
        s.prefill(&[12, 300, 301, 302, 999 % 280 + 16, 17], &|| false)
            .unwrap();
        s.truncate(4);
        let got = s.prefill(&PROMPT[4..], &|| false).unwrap().unwrap();
        assert!(close(&got, &want));
        assert_eq!(s.tokens(), PROMPT);
    }

    #[test]
    fn decode_through_the_window_matches_one_shot_prefill() {
        // window is 4 in the tiny model; 9 tokens cross it twice.
        let m = model();
        let mut one = Session::new(m.clone(), 128);
        let want = one.prefill(&PROMPT, &|| false).unwrap().unwrap();
        let mut step = Session::new(m, 128);
        step.prefill(&PROMPT[..1], &|| false).unwrap();
        let mut got = Vec::new();
        for &t in &PROMPT[1..] {
            got = step.step(t).unwrap();
        }
        assert!(close(&got, &want));
    }

    #[test]
    fn the_sliding_window_changes_the_logits() {
        // Guards the test above against a model in which the window is never
        // applied: with a window wider than the prompt, the logits differ.
        let path = std::env::temp_dir().join(format!(
            "gemma-sess-wide-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        let cfg = TinyConfig {
            sliding_window: 64,
            ..TinyConfig::default()
        };
        write_tiny(&path, &cfg).unwrap();
        let wide = Model::open(&path, &Device::Cpu).unwrap();
        let narrow = model();
        let a = Session::new(wide, 128)
            .prefill(&PROMPT, &|| false)
            .unwrap()
            .unwrap();
        let b = Session::new(narrow, 128)
            .prefill(&PROMPT, &|| false)
            .unwrap()
            .unwrap();
        assert!(!close(&a, &b));
    }

    #[test]
    fn snapshot_restore_gives_the_same_next_logits() {
        let m = model();
        let mut a = Session::new(m.clone(), 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let blob = a.snapshot().unwrap();
        let next_a = a.step(20).unwrap();
        let mut b = Session::new(m, 128);
        b.restore(&blob).unwrap();
        assert_eq!(b.tokens(), PROMPT);
        assert!(close(&b.step(20).unwrap(), &next_a));
    }

    #[test]
    fn restore_refuses_each_corrupted_header_field() {
        let m = model();
        let mut a = Session::new(m.clone(), 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let blob = a.snapshot().unwrap();
        for at in [0usize, 4, 8, 40, 41] {
            // magic, format, signature, dtype, n_tokens
            let mut bad = blob.clone();
            bad[at] ^= 0xFF;
            let mut b = Session::new(m.clone(), 128);
            assert!(b.restore(&bad).is_err(), "byte {at}");
            assert!(b.tokens().is_empty(), "byte {at} left state behind");
        }
    }

    #[test]
    fn a_failed_restore_keeps_the_previous_state() {
        let m = model();
        let mut a = Session::new(m.clone(), 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let blob = a.snapshot().unwrap();
        let mut b = Session::new(m, 128);
        b.prefill(&PROMPT[..3], &|| false).unwrap();
        let before = b.snapshot().unwrap();
        assert!(b.restore(&blob[..blob.len() - 1]).is_err());
        assert_eq!(b.tokens(), &PROMPT[..3]);
        assert_eq!(b.snapshot().unwrap(), before);
    }

    /// A session whose snapshots keep the bare window, no
    /// [`SNAPSHOT_SLACK`], so a 9-token prompt already trims.
    fn bare(m: Arc<Model>, ctx: usize) -> Session {
        let mut s = Session::new(m, ctx);
        s.slack = 0;
        s
    }

    /// A prompt `n` tokens long, over ordinary byte tokens.
    fn long_prompt(n: usize) -> Vec<u32> {
        (0..n)
            .map(|i| 16 + u32::try_from(i % 200).unwrap())
            .collect()
    }

    /// A snapshot keeps `window + SNAPSHOT_SLACK` positions per sliding
    /// layer, so after a restore truncation is exact down to
    /// `SNAPSHOT_SLACK + 1` tokens behind the end and empties one further.
    #[test]
    fn a_snapshot_keeps_slack_past_the_window() {
        let model = model();
        let prompt = long_prompt(80);
        let len = prompt.len();
        let mut orig = Session::new(model.clone(), 128);
        orig.prefill(&prompt, &|| false).unwrap();
        let blob = orig.snapshot().unwrap();
        assert_eq!(
            blob.len(),
            HEADER + 4 * len + 4 + layer_bytes(4 + SNAPSHOT_SLACK, 8) + layer_bytes(len, 16)
        );
        let mut restored = Session::new(model.clone(), 128);
        restored.restore(&blob).unwrap();
        assert_eq!(restored.cache.layers[0].base(), len - 4 - SNAPSHOT_SLACK);
        let floor = len - SNAPSHOT_SLACK - 1;
        assert_eq!(
            restored.reusable(len - SNAPSHOT_SLACK),
            len - SNAPSHOT_SLACK
        );
        assert_eq!(restored.reusable(floor - 1), 0);
        assert_eq!(restored.truncate(floor), floor);
        orig.truncate(floor);
        assert_eq!(restored.step(20).unwrap(), orig.step(20).unwrap());
        let mut other = Session::new(model, 128);
        other.restore(&blob).unwrap();
        assert_eq!(other.truncate(floor - 1), 0);
        assert!(other.tokens().is_empty());
    }

    /// `PROMPT` plus three more tokens, for sessions that grow after a
    /// restore.
    const MORE: [u32; 3] = [40, 41, 42];

    /// The tiny model's KV owners: a sliding layer (1 head x 8) and a global
    /// one (1 head x 16), window 4.
    fn layer_bytes(positions: usize, head_dim: usize) -> usize {
        16 + 2 * positions * head_dim * 4
    }

    #[test]
    fn a_snapshot_keeps_only_the_window_of_a_sliding_layer() {
        let m = model();
        let mut a = bare(m, 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let blob = a.snapshot().unwrap();
        let n = PROMPT.len();
        // Sliding layer: the last 4 positions; global layer: all 9.
        assert_eq!(
            blob.len(),
            HEADER + 4 * n + 4 + layer_bytes(4, 8) + layer_bytes(n, 16)
        );
    }

    #[test]
    fn a_trimmed_restore_steps_exactly_like_the_untrimmed_session() {
        let m = model();
        let mut a = bare(m.clone(), 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let blob = a.snapshot().unwrap();
        let mut b = bare(m, 128);
        b.restore(&blob).unwrap();
        assert_eq!(b.cache.layers[0].base(), PROMPT.len() - 4);
        assert_eq!(b.cache.layers[1].base(), 0);
        assert_eq!(b.step(20).unwrap(), a.step(20).unwrap());
    }

    #[test]
    fn a_restored_session_grows_and_snapshots_again() {
        let m = model();
        let mut a = bare(m.clone(), 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let mut b = bare(m.clone(), 128);
        b.restore(&a.snapshot().unwrap()).unwrap();
        a.prefill(&MORE, &|| false).unwrap();
        b.prefill(&MORE, &|| false).unwrap();
        let mut c = bare(m, 128);
        c.restore(&b.snapshot().unwrap()).unwrap();
        assert_eq!(c.tokens(), a.tokens());
        assert_eq!(c.cache.layers[0].base(), PROMPT.len() + MORE.len() - 4);
        assert_eq!(c.step(20).unwrap(), a.step(20).unwrap());
    }

    #[test]
    fn truncating_a_restore_within_its_window_is_exact() {
        let m = model();
        let mut a = bare(m.clone(), 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let mut b = bare(m, 128);
        b.restore(&a.snapshot().unwrap()).unwrap();
        // base 5, window 4: a query at 8 needs keys 5..=8, all still held.
        assert_eq!(b.truncate(8), 8);
        a.truncate(8);
        assert_eq!(b.tokens(), &PROMPT[..8]);
        assert_eq!(b.step(20).unwrap(), a.step(20).unwrap());
    }

    #[test]
    fn truncating_a_restore_below_its_window_empties_it() {
        let m = model();
        let mut fresh = bare(m.clone(), 128);
        let want = fresh.prefill(&PROMPT, &|| false).unwrap().unwrap();
        let mut a = bare(m.clone(), 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let mut b = bare(m, 128);
        b.restore(&a.snapshot().unwrap()).unwrap();
        // A query at 7 needs key 4, which the snapshot dropped.
        assert_eq!(b.reusable(7), 0);
        assert_eq!(b.reusable(8), 8);
        assert_eq!(b.truncate(7), 0);
        assert!(b.tokens().is_empty());
        assert!(b.cache.layers.iter().all(|l| l.is_empty() && l.base() == 0));
        assert_eq!(b.prefill(&PROMPT, &|| false).unwrap().unwrap(), want);
    }

    #[test]
    fn a_format_one_snapshot_is_refused() {
        let m = model();
        let mut a = Session::new(m.clone(), 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        let mut blob = a.snapshot().unwrap();
        blob[4..8].copy_from_slice(&1u32.to_le_bytes());
        let mut b = Session::new(m, 128);
        let e = b.restore(&blob).unwrap_err();
        assert!(e.0.contains("format 1"), "{e}");
        assert!(b.tokens().is_empty());
    }

    #[test]
    fn a_fresh_session_keeps_its_whole_history() {
        let m = model();
        let mut a = Session::new(m, 128);
        a.prefill(&PROMPT, &|| false).unwrap();
        assert!(a.cache.layers.iter().all(|l| l.base() == 0));
        assert_eq!(a.reusable(1), 1);
        assert_eq!(a.truncate(1), 1);
    }

    #[test]
    fn interrupted_prefill_keeps_only_evaluated_tokens() {
        let m = model();
        let mut s = Session::new(m, 4096);
        let long: Vec<u32> = (0..1300).map(|i| 16 + (i % 200)).collect();
        let calls = std::cell::Cell::new(0);
        let r = s
            .prefill(&long, &|| {
                calls.set(calls.get() + 1);
                calls.get() > 1
            })
            .unwrap();
        assert!(r.is_none());
        assert_eq!(s.tokens(), &long[..512]);
    }

    #[test]
    fn beyond_context_is_an_error() {
        let m = model();
        let mut s = Session::new(m, 8);
        let e = s.prefill(&PROMPT, &|| false).unwrap_err();
        assert!(e.0.contains("context full"), "{e}");
        assert!(s.tokens().is_empty());
    }
}
