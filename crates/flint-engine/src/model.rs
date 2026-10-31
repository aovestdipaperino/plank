//! The Flint model: loading every tensor from a mapped GGUF, and one forward
//! pass over a prefix-truncatable KV cache (`llama_model_flint::graph`).
//!
//! The hidden state is `[t, hidden]` f32 throughout. Each layer adds
//! attention over `rmsnorm(h, attn_norm)`, then the FFN over
//! `rmsnorm(h, ffn_norm)`: a dense `SwiGLU` on the leading layers, the routed
//! experts plus the shared expert on the rest. The head runs on the last
//! position only. The embedding is gathered from the mapped BF16 rows on
//! the CPU, so it never occupies device memory.

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use candle_core::{Device, Tensor};
use gemma_engine::kv::KvCache;
use gemma_engine::session::SessionModel;
use sha2::{Digest, Sha256};

use crate::attention::{Attention, LazyMask, causal_mask};
use crate::config::FlintConfig;
use crate::ffn::{Dense, Moe};
use crate::file::GgufFile;
use crate::gguf::{GGML_BF16, Header};
use crate::nvfp4::ops;
use crate::nvfp4::tensor::{MoeExperts, NvWeight};
use crate::rope::Rope;
use crate::{Result, err};

#[derive(Debug)]
pub(crate) enum Ffn {
    Dense(Dense),
    Moe(Moe),
}

#[derive(Debug)]
pub(crate) struct Layer {
    pub(crate) attn_norm: Tensor,
    pub(crate) attn: Attention,
    pub(crate) ffn_norm: Tensor,
    pub(crate) ffn: Ffn,
}

/// A loaded Flint model.
#[derive(Debug)]
pub struct Model {
    config: FlintConfig,
    file: GgufFile,
    /// Byte range of `token_embd.weight` in the mapping.
    embd: (usize, usize),
    vocab: usize,
    layers: Vec<Layer>,
    output_norm: Tensor,
    output: Arc<NvWeight>,
    rope: Rope,
    device: Device,
    context_length: usize,
    signature: [u8; 32],
    /// Held on Metal for a whole forward, and around a session's snapshot
    /// and restore, which also run kernels on the cache. candle 0.9.2's
    /// `MetalDevice` is only correct for one thread at a time: its encoders
    /// come from a pool of command buffers that commit independently, and a
    /// thread whose encoder finds another thread's still open takes a
    /// different buffer, so one forward's kernels can land in several
    /// buffers and run out of order on the GPU (FINDINGS, "Two threads on
    /// one Metal device"). Serialising that work keeps it in order.
    gpu: Mutex<()>,
    /// The single-token decode graph, built on the first Metal step: `None`
    /// inside when this model's shapes need the candle path.
    #[cfg(target_os = "macos")]
    decode: std::sync::OnceLock<Option<crate::decode::DecodeGraph>>,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// SHA-256 over the file's shape: every metadata key with its value's
/// debug form, and every tensor's name, shape and type. Never the weights.
///
/// `tokenizer.*` is skipped. Snapshots store token ids, so a file whose
/// vocab changed but whose shapes did not would accept a snapshot whose ids
/// now mean other text. That is acceptable: such a file has another sha,
/// and plank keys its caches by the file, not by this signature alone.
fn signature(h: &Header) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(b"flint-engine model v1\n");
    for (k, v) in &h.metadata {
        if k.starts_with("tokenizer.") {
            continue;
        }
        s.update(format!("{k}={v:?}\n").as_bytes());
    }
    for (name, t) in &h.tensors {
        s.update(format!("{name} {:?} {}\n", t.dims, t.ggml_type).as_bytes());
    }
    s.finalize().into()
}

struct Loader<'a> {
    file: &'a GgufFile,
    device: &'a Device,
}

impl Loader<'_> {
    fn f32s(&self, name: &str, len: usize) -> Result<Tensor> {
        let v = self.file.f32_tensor(name)?;
        if v.len() != len {
            return Err(err(format!("{name} holds {} values, want {len}", v.len())));
        }
        Ok(Tensor::from_vec(v, len, self.device)?)
    }

    fn bf16_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Tensor> {
        let t = self.file.header.tensor(name)?;
        if t.ggml_type != GGML_BF16 || t.dims != [cols as u64, rows as u64] {
            return Err(err(format!(
                "{name} is {:?} type {}, want BF16 [{cols}, {rows}]",
                t.dims, t.ggml_type
            )));
        }
        let v: Vec<f32> = self
            .file
            .tensor_bytes(name)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
            .collect();
        Ok(Tensor::from_vec(v, (rows, cols), self.device)?)
    }

    fn nv(&self, prefix: &str, cols: usize, rows: usize, experts: usize) -> Result<Arc<NvWeight>> {
        let w = self.file.nv_weight(prefix)?;
        if (w.cols(), w.rows(), w.experts()) != (cols, rows, experts) {
            return Err(err(format!(
                "{prefix} is [{}, {}, {}], want [{cols}, {rows}, {experts}]",
                w.cols(),
                w.rows(),
                w.experts()
            )));
        }
        Ok(Arc::new(w))
    }

    fn dense(&self, p: &str, suffix: &str, hidden: usize, ffn: usize) -> Result<Dense> {
        Ok(Dense {
            gate: self.nv(&format!("{p}.ffn_gate{suffix}"), hidden, ffn, 1)?,
            up: self.nv(&format!("{p}.ffn_up{suffix}"), hidden, ffn, 1)?,
            down: self.nv(&format!("{p}.ffn_down{suffix}"), ffn, hidden, 1)?,
        })
    }

    fn layer(&self, c: &FlintConfig, l: usize) -> Result<Layer> {
        let p = format!("blk.{l}");
        let (qd, kd) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let attn = Attention {
            q: self.nv(&format!("{p}.attn_q"), c.hidden, qd, 1)?,
            k: self.nv(&format!("{p}.attn_k"), c.hidden, kd, 1)?,
            v: self.nv(&format!("{p}.attn_v"), c.hidden, kd, 1)?,
            o: self.nv(&format!("{p}.attn_output"), qd, c.hidden, 1)?,
            q_norm: self.f32s(&format!("{p}.attn_q_norm.weight"), c.head_dim)?,
            k_norm: self.f32s(&format!("{p}.attn_k_norm.weight"), c.head_dim)?,
            sinks: self.f32s(&format!("{p}.attn_sinks.weight"), c.heads)?,
            heads: c.heads,
            kv_heads: c.kv_heads,
            head_dim: c.head_dim,
            window: c.window(l),
            rope: c.rope[l],
            eps: c.rms_eps,
        };
        let ffn = if c.is_moe(l) {
            let experts = MoeExperts::new(
                self.nv(
                    &format!("{p}.ffn_gate_exps"),
                    c.hidden,
                    c.expert_ffn,
                    c.experts,
                )?,
                self.nv(
                    &format!("{p}.ffn_up_exps"),
                    c.hidden,
                    c.expert_ffn,
                    c.experts,
                )?,
                self.nv(
                    &format!("{p}.ffn_down_exps"),
                    c.expert_ffn,
                    c.hidden,
                    c.experts,
                )?,
            )?;
            Ffn::Moe(Moe {
                router: self.bf16_matrix(
                    &format!("{p}.ffn_gate_inp.weight"),
                    c.experts,
                    c.hidden,
                )?,
                experts,
                shared: self.dense(&p, "_shexp", c.hidden, c.shared_ffn)?,
                k: c.experts_used,
                norm: c.weights_norm,
                scale: c.weights_scale,
            })
        } else {
            Ffn::Dense(self.dense(&p, "", c.hidden, c.ffn)?)
        };
        Ok(Layer {
            attn_norm: self.f32s(&format!("{p}.attn_norm.weight"), c.hidden)?,
            attn,
            ffn_norm: self.f32s(&format!("{p}.ffn_norm.weight"), c.hidden)?,
            ffn,
        })
    }
}

impl Model {
    /// Opens the GGUF at `path` with its full context.
    ///
    /// # Errors
    /// As [`Self::from_file`], or the file cannot be mapped.
    pub fn open(path: &Path, device: &Device) -> Result<Arc<Self>> {
        Self::open_with_ctx(path, device, usize::MAX)
    }

    /// Opens the GGUF at `path`, attending over at most `ctx_cap` positions.
    ///
    /// # Errors
    /// As [`Self::from_file`], or the file cannot be mapped.
    pub fn open_with_ctx(path: &Path, device: &Device, ctx_cap: usize) -> Result<Arc<Self>> {
        Self::from_file(GgufFile::open(path)?, device, ctx_cap)
    }

    /// Loads every tensor of an already mapped file.
    ///
    /// # Errors
    /// A config [`FlintConfig::from_header`] refuses, an expert bias tensor
    /// (`blk.N.ffn_exp_probs_b.bias`, which the engine does not apply), a
    /// missing tensor, or a tensor whose type or shape is not the one Flint
    /// has.
    pub fn from_file(file: GgufFile, device: &Device, ctx_cap: usize) -> Result<Arc<Self>> {
        let config = FlintConfig::from_header(&file.header)?;
        if let Some(name) = file
            .header
            .tensors
            .keys()
            .find(|n| n.contains(".ffn_exp_probs_b."))
        {
            return Err(err(format!(
                "{name} is present; the engine does not apply an expert bias"
            )));
        }
        let embd = file.header.tensor("token_embd.weight")?;
        let [cols, rows] = embd.dims[..] else {
            return Err(err(format!("token_embd.weight has shape {:?}", embd.dims)));
        };
        if embd.ggml_type != GGML_BF16 || cols != config.hidden as u64 {
            return Err(err(format!(
                "token_embd.weight is type {} [{cols}, {rows}]",
                embd.ggml_type
            )));
        }
        let vocab = usize::try_from(rows).map_err(|_| err("vocab too large"))?;
        let start = usize::try_from(embd.offset).map_err(|_| err("embedding offset too large"))?;
        let embd = (start, start + file.tensor_bytes("token_embd.weight")?.len());
        let loader = Loader {
            file: &file,
            device,
        };
        let layers = (0..config.layers)
            .map(|l| loader.layer(&config, l))
            .collect::<Result<Vec<_>>>()?;
        let output_norm = loader.f32s("output_norm.weight", config.hidden)?;
        let output = loader.nv("output", config.hidden, vocab, 1)?;
        let context_length = config.context_length.min(ctx_cap);
        let rope = Rope::new(config.head_dim, config.rope_base, context_length, device)?;
        let signature = signature(&file.header);
        Ok(Arc::new(Self {
            config,
            file,
            embd,
            vocab,
            layers,
            output_norm,
            output,
            rope,
            device: device.clone(),
            context_length,
            signature,
            gpu: Mutex::new(()),
            #[cfg(target_os = "macos")]
            decode: std::sync::OnceLock::new(),
        }))
    }

    /// The hyperparameters read from the GGUF metadata.
    #[must_use]
    pub fn config(&self) -> &FlintConfig {
        &self.config
    }

    /// The header, for building the tokenizer from the same file.
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.file.header
    }

    /// Rows in the token embedding: the vocab size.
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Positions the model can attend over: the file's context length,
    /// capped at open.
    #[must_use]
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    /// SHA-256 over the file's metadata and tensor shapes, never its
    /// weights: what a snapshot must match to be restored.
    #[must_use]
    pub fn signature(&self) -> [u8; 32] {
        self.signature
    }

    /// `(kv_heads, head_dim)` per layer: every Flint layer owns its KV.
    #[must_use]
    pub fn kv_shape(&self) -> Vec<(usize, usize)> {
        vec![(self.config.kv_heads, self.config.head_dim); self.config.layers]
    }

    /// Each layer's sliding window, `None` for a global layer.
    #[must_use]
    pub fn kv_windows(&self) -> Vec<Option<usize>> {
        (0..self.config.layers)
            .map(|l| self.config.window(l))
            .collect()
    }

    /// Bytes one position adds to the cache: K and V, f16, every layer
    /// ([`FlintConfig::kv_bytes_per_token`]).
    #[must_use]
    pub fn kv_bytes_per_token(&self) -> usize {
        self.config.kv_bytes_per_token()
    }

    /// The device every tensor and the KV cache live on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Token `token`'s embedding row, widened from BF16 into `out`
    /// (`hidden` values).
    #[cfg_attr(
        not(target_os = "macos"),
        allow(dead_code, reason = "read by the Metal decode path only")
    )]
    pub(crate) fn embed_row(&self, token: u32, out: &mut [f32]) -> Result<()> {
        let d = self.config.hidden;
        let i = token as usize;
        if i >= self.vocab || out.len() != d {
            return Err(err(format!(
                "token {token} is outside the {}-token vocab",
                self.vocab
            )));
        }
        let bytes = &self.file.mapped_file().bytes()[self.embd.0..self.embd.1];
        for (o, c) in out
            .iter_mut()
            .zip(bytes[i * d * 2..(i + 1) * d * 2].as_chunks::<2>().0)
        {
            *o = bf16_to_f32(u16::from_le_bytes(*c));
        }
        Ok(())
    }

    /// The layers, for the decode graph.
    #[cfg(target_os = "macos")]
    pub(crate) fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// The rotary tables, for the decode graph.
    #[cfg(target_os = "macos")]
    pub(crate) fn rope(&self) -> &Rope {
        &self.rope
    }

    /// The final norm's weight, for the decode graph.
    #[cfg(target_os = "macos")]
    pub(crate) fn output_norm(&self) -> &Tensor {
        &self.output_norm
    }

    /// The output head, for the decode graph.
    #[cfg(target_os = "macos")]
    pub(crate) fn output(&self) -> &NvWeight {
        &self.output
    }

    /// The decode graph, built on first use; `None` off Metal or for shapes
    /// it does not take.
    #[cfg(target_os = "macos")]
    fn decode_graph(&self) -> Result<Option<&crate::decode::DecodeGraph>> {
        if let Some(g) = self.decode.get() {
            return Ok(g.as_ref());
        }
        let g = crate::decode::DecodeGraph::new(self)?;
        Ok(self.decode.get_or_init(|| g).as_ref())
    }

    fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let d = self.config.hidden;
        let bytes = &self.file.mapped_file().bytes()[self.embd.0..self.embd.1];
        let mut out = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let i = t as usize;
            if i >= self.vocab {
                return Err(err(format!(
                    "token {t} is outside the {}-token vocab",
                    self.vocab
                )));
            }
            out.extend(
                bytes[i * d * 2..(i + 1) * d * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| bf16_to_f32(u16::from_le_bytes(*c))),
            );
        }
        Ok(Tensor::from_vec(out, (tokens.len(), d), &self.device)?)
    }

    /// The [`Self::gpu`] lock on Metal, `None` on the CPU. A panic while it
    /// was held leaves nothing the next holder depends on, so a poisoned
    /// lock is still usable.
    fn gpu_guard(&self) -> Option<MutexGuard<'_, ()>> {
        self.device
            .is_metal()
            .then(|| self.gpu.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Runs `tokens` at positions `offset..`, extending `cache`, and returns
    /// the last position's logits.
    ///
    /// # Errors
    /// An empty `tokens`, a token outside the vocab, a cache that does not
    /// match the model, or a device failure. On an error the cache may hold
    /// part of this forward; the caller truncates it back to `offset`.
    pub fn forward(&self, tokens: &[u32], offset: usize, cache: &mut KvCache) -> Result<Vec<f32>> {
        self.forward_with(tokens, offset, cache, true)
    }

    /// [`Self::forward`], through the decode graph for a single token on
    /// Metal when `graph` is set and the graph takes this model, else
    /// through candle ops.
    pub(crate) fn forward_with(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
        graph: bool,
    ) -> Result<Vec<f32>> {
        // Released when the forward returns, after `to_vec1` has waited for
        // the GPU.
        let _gpu = self.gpu_guard();
        let t = tokens.len();
        if t == 0 {
            return Err(err("forward over no tokens"));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(err(format!(
                "cache has {} layers, model {}",
                cache.layers.len(),
                self.layers.len()
            )));
        }
        let Some(end) = offset.checked_add(t).filter(|&e| e <= self.context_length) else {
            return Err(err(format!(
                "{t} tokens at offset {offset} exceed the context of {}",
                self.context_length
            )));
        };
        #[cfg(target_os = "macos")]
        if graph
            && t == 1
            && let Some(g) = self.decode_graph()?
        {
            return g.step(self, tokens[0], offset, cache);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = graph;
        let eps = self.config.rms_eps;
        // Built only if a layer's attention takes the op chain: the fused
        // prefill kernel masks from positions.
        let w = self.config.sliding_window;
        let start = (offset + 1).saturating_sub(w);
        let (global, sliding) = if t > 1 {
            (
                Some(LazyMask::new(|| {
                    causal_mask(t, offset, 0, end, None, &self.device)
                })),
                Some(LazyMask::new(move || {
                    causal_mask(t, offset, start, end - start, Some(w), &self.device)
                })),
            )
        } else {
            (None, None)
        };
        let mut hidden = self.embed(tokens)?;
        for (layer, kv) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let mask = if layer.attn.window.is_some() {
                sliding.as_ref()
            } else {
                global.as_ref()
            };
            let attn_out = layer.attn.forward(
                &candle_nn::ops::rms_norm(&hidden, &layer.attn_norm, eps)?,
                offset,
                kv,
                &self.rope,
                mask,
            )?;
            let h1 = (hidden + attn_out)?;
            let normed = candle_nn::ops::rms_norm(&h1, &layer.ffn_norm, eps)?;
            let ffn_out = match &layer.ffn {
                Ffn::Dense(d) => d.forward(&normed)?,
                Ffn::Moe(m) => m.forward(&normed)?,
            };
            hidden = (h1 + ffn_out)?;
        }
        let last = candle_nn::ops::rms_norm(
            &hidden.narrow(0, t - 1, 1)?.contiguous()?,
            &self.output_norm,
            eps,
        )?;
        Ok(ops::matmul(&self.output, &last)?
            .squeeze(0)?
            .to_vec1::<f32>()?)
    }
}

impl SessionModel for Model {
    const SNAPSHOT_MAGIC: [u8; 4] = *b"PLFK";

    fn forward(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
    ) -> gemma_engine::Result<Vec<f32>> {
        Model::forward(self, tokens, offset, cache).map_err(|e| gemma_engine::Error(e.0))
    }

    fn context_length(&self) -> usize {
        Model::context_length(self)
    }

    fn signature(&self) -> [u8; 32] {
        Model::signature(self)
    }

    fn kv_shape(&self) -> Vec<(usize, usize)> {
        Model::kv_shape(self)
    }

    fn kv_windows(&self) -> Vec<Option<usize>> {
        Model::kv_windows(self)
    }

    /// f16: half the bytes attention streams per key, as llama.cpp keeps it.
    fn kv_dtype(&self) -> candle_core::DType {
        candle_core::DType::F16
    }

    fn device(&self) -> &Device {
        Model::device(self)
    }

    fn device_exclusive<R>(&self, f: impl FnOnce() -> R) -> R {
        let _gpu = self.gpu_guard();
        f()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_config;
    use crate::testgguf::tiny_flint;
    #[cfg(target_os = "macos")]
    use candle_core::DType;
    use gemma_engine::session::Session;

    const VOCAB: usize = 256;

    fn model_on(device: &Device, sink: f32, window: usize) -> Arc<Model> {
        let mut c = test_config();
        c.sliding_window = window;
        let file = GgufFile::from_bytes(&tiny_flint(&c, VOCAB, sink, 7)).unwrap();
        Model::from_file(file, device, 64).unwrap()
    }

    fn model() -> Arc<Model> {
        model_on(&Device::Cpu, 0.5, 4)
    }

    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
    }

    const PROMPT: [u32; 9] = [12, 200, 201, 202, 203, 17, 99, 5, 255];

    #[test]
    fn forward_gives_finite_logits_over_the_vocab() {
        let m = model();
        let mut s = Session::new(m, 64);
        let logits = s.prefill(&PROMPT, &|| false).unwrap().unwrap();
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn decoding_step_by_step_equals_one_prefill() {
        let m = model();
        let mut one = Session::new(m.clone(), 64);
        let want = one.prefill(&PROMPT, &|| false).unwrap().unwrap();
        let mut step = Session::new(m, 64);
        step.prefill(&PROMPT[..1], &|| false).unwrap();
        let mut got = Vec::new();
        for &t in &PROMPT[1..] {
            got = step.step(t).unwrap();
        }
        assert!(close(&got, &want, 1e-4));
    }

    #[test]
    fn truncating_and_reprefilling_equals_a_fresh_prefill() {
        let m = model();
        let mut fresh = Session::new(m.clone(), 64);
        let want = fresh.prefill(&PROMPT, &|| false).unwrap().unwrap();
        let mut s = Session::new(m, 64);
        s.prefill(&[12, 200, 201, 202, 7, 7, 7], &|| false).unwrap();
        assert_eq!(s.truncate(4), 4);
        let got = s.prefill(&PROMPT[4..], &|| false).unwrap().unwrap();
        assert!(close(&got, &want, 1e-4));
    }

    #[test]
    fn a_snapshot_restores_the_same_next_logits() {
        let m = model();
        let mut a = Session::new(m.clone(), 64);
        a.prefill(&PROMPT, &|| false).unwrap();
        let blob = a.snapshot().unwrap();
        assert_eq!(&blob[..4], b"PLFK");
        let mut b = Session::new(m, 64);
        b.restore(&blob).unwrap();
        assert!(close(&b.step(3).unwrap(), &a.step(3).unwrap(), 1e-5));
    }

    #[test]
    fn the_window_and_the_sinks_change_the_logits() {
        let base = Session::new(model(), 64)
            .prefill(&PROMPT, &|| false)
            .unwrap()
            .unwrap();
        let wide = Session::new(model_on(&Device::Cpu, 0.5, 64), 64)
            .prefill(&PROMPT, &|| false)
            .unwrap()
            .unwrap();
        let sunk = Session::new(model_on(&Device::Cpu, 4.0, 4), 64)
            .prefill(&PROMPT, &|| false)
            .unwrap()
            .unwrap();
        assert!(
            !close(&base, &wide, 1e-4),
            "9 tokens cross a 4-token window"
        );
        assert!(!close(&base, &sunk, 1e-4), "the sink logit takes mass");
    }

    #[test]
    fn an_expert_bias_tensor_is_refused() {
        let c = test_config();
        let bytes = crate::testgguf::tiny_flint_builder(&c, VOCAB, 0.5, 7)
            .tensor(
                "blk.1.ffn_exp_probs_b.bias",
                &[c.experts as u64],
                crate::gguf::GGML_F32,
                &vec![0; 4 * c.experts],
            )
            .bytes();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        let Err(e) = Model::from_file(file, &Device::Cpu, 64) else {
            panic!("a file with an expert bias loaded");
        };
        assert!(e.0.contains("blk.1.ffn_exp_probs_b.bias"), "{e}");
    }

    #[test]
    fn a_token_outside_the_vocab_is_an_error() {
        let mut s = Session::new(model(), 64);
        assert!(s.prefill(&[300], &|| false).is_err());
    }

    #[test]
    fn forward_refuses_an_offset_that_would_overflow() {
        let m = model();
        let mut cache = KvCache::new(m.kv_shape().len());
        let e = m.forward(&[1, 2], usize::MAX, &mut cache).unwrap_err();
        assert!(e.0.contains("context"), "{e}");
    }

    #[test]
    fn forward_refuses_no_tokens() {
        let m = model();
        let mut cache = KvCache::new(m.kv_shape().len());
        assert!(m.forward(&[], 0, &mut cache).is_err());
    }

    #[test]
    fn forward_refuses_a_context_overrun() {
        let m = model();
        let ctx = m.context_length();
        let mut cache = KvCache::new(m.kv_shape().len());
        let e = m.forward(&vec![1; ctx + 1], 0, &mut cache).unwrap_err();
        assert!(e.0.contains("exceed the context"), "{e}");
        assert!(
            cache.layers.iter().all(gemma_engine::kv::LayerKv::is_empty),
            "nothing ran"
        );
    }

    /// A 256-position model with a 4-token window, so a snapshot after 100
    /// tokens trims its sliding layers ([`SNAPSHOT_SLACK`] is 64).
    ///
    /// [`SNAPSHOT_SLACK`]: gemma_engine::session::SNAPSHOT_SLACK
    fn long_model_on(device: &Device) -> Arc<Model> {
        let mut c = test_config();
        c.context_length = 256;
        let file = GgufFile::from_bytes(&tiny_flint(&c, VOCAB, 0.5, 7)).unwrap();
        Model::from_file(file, device, 256).unwrap()
    }

    fn long_prompt(n: u32) -> Vec<u32> {
        (0..n).map(|i| (i * 37 + 11) % 256).collect()
    }

    fn a_trimmed_restore_matches_a_session_never_snapshotted(dev: &Device) {
        let m = long_model_on(dev);
        let prompt = long_prompt(100);
        let tail = long_prompt(121)[101..].to_vec();
        let fresh = || {
            let mut s = Session::new(m.clone(), 256);
            s.prefill(&prompt, &|| false).unwrap();
            s
        };
        let blob = fresh().snapshot().unwrap();
        let restored = || {
            let mut s = Session::new(m.clone(), 256);
            s.restore(&blob).unwrap();
            s
        };
        assert_eq!(
            restored().reusable(10),
            0,
            "the sliding layers were trimmed"
        );

        let (mut a, mut b) = (fresh(), restored());
        assert!(close(&b.step(3).unwrap(), &a.step(3).unwrap(), 1e-5));
        let want = a.prefill(&tail, &|| false).unwrap().unwrap();
        let got = b.prefill(&tail, &|| false).unwrap().unwrap();
        assert_eq!(tail.len(), 20);
        assert!(close(&got, &want, 1e-5), "a 20-token prefill after restore");

        let mut c = restored();
        assert_eq!(c.truncate(50), 50);
        let got = c.prefill(&prompt[50..], &|| false).unwrap().unwrap();
        let want = Session::new(m.clone(), 256)
            .prefill(&prompt, &|| false)
            .unwrap()
            .unwrap();
        assert!(close(&got, &want, 1e-4), "truncate(50) then re-prefill");
    }

    #[test]
    fn a_trimmed_restore_matches_a_session_never_snapshotted_on_the_cpu() {
        a_trimmed_restore_matches_a_session_never_snapshotted(&Device::Cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_trimmed_restore_matches_a_session_never_snapshotted_on_metal() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        a_trimmed_restore_matches_a_session_never_snapshotted(&dev);
    }

    /// One prefill of `long_prompt(40)` and eight steps, every step's logits.
    #[cfg(target_os = "macos")]
    fn prefill_and_steps(m: &Arc<Model>) -> Vec<Vec<f32>> {
        let mut s = Session::new(m.clone(), 256);
        let mut out = vec![s.prefill(&long_prompt(40), &|| false).unwrap().unwrap()];
        for t in 0..8 {
            out.push(s.step(t * 29 % 256).unwrap());
        }
        out
    }

    /// Sessions on two threads over one `Arc<Model>` on Metal each get what
    /// a lone session gets: forwards from different threads must not
    /// interleave their GPU work (see `Model::gpu`).
    #[cfg(target_os = "macos")]
    #[test]
    fn two_threads_on_one_metal_model_match_one_thread() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let m = long_model_on(&dev);
        let want = prefill_and_steps(&m);
        for round in 0..4 {
            let runs: Vec<_> = std::thread::scope(|s| {
                let hs: Vec<_> = (0..2).map(|_| s.spawn(|| prefill_and_steps(&m))).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (th, got) in runs.iter().enumerate() {
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(close(g, w, 1e-5), "round {round}, thread {th}, forward {i}");
                }
            }
        }
    }

    /// A snapshot and a restore also run kernels (the cache's copies), so
    /// on Metal they must not interleave with another thread's forward.
    #[cfg(target_os = "macos")]
    #[test]
    fn snapshots_and_restores_beside_a_forward_on_metal_are_exact() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let m = long_model_on(&dev);
        let mut a = Session::new(m.clone(), 256);
        a.prefill(&long_prompt(100), &|| false).unwrap();
        let blob = a.snapshot().unwrap();
        let next = {
            let mut b = Session::new(m.clone(), 256);
            b.restore(&blob).unwrap();
            b.step(3).unwrap()
        };
        let want = prefill_and_steps(&m);
        for round in 0..4 {
            std::thread::scope(|s| {
                let side = s.spawn(|| {
                    for _ in 0..10 {
                        assert!(a.snapshot().unwrap() == blob, "round {round}: snapshot");
                        let mut b = Session::new(m.clone(), 256);
                        b.restore(&blob).unwrap();
                        assert!(
                            close(&b.step(3).unwrap(), &next, 1e-5),
                            "round {round}: restore"
                        );
                    }
                });
                for (i, (g, w)) in prefill_and_steps(&m).iter().zip(&want).enumerate() {
                    assert!(close(g, w, 1e-5), "round {round}, forward {i}");
                }
                side.join().unwrap();
            });
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_the_cpu() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let cpu = Session::new(model(), 64)
            .prefill(&PROMPT, &|| false)
            .unwrap()
            .unwrap();
        let mut s = Session::new(model_on(&dev, 0.5, 4), 64);
        let gpu = s.prefill(&PROMPT, &|| false).unwrap().unwrap();
        // Metal's prefill multiplies in f16 (accumulating in f32), as
        // llama.cpp does, so it agrees with the f32 CPU to about 1e-2.
        assert!(close(&gpu, &cpu, 2e-2));
        let mut c = Session::new(model(), 64);
        c.prefill(&PROMPT, &|| false).unwrap();
        assert!(close(&s.step(42).unwrap(), &c.step(42).unwrap(), 2e-2));
    }

    /// The decode graph against the candle path on Metal, step by step past
    /// a 200-token prefill: attention crosses the 128-key chunk boundary,
    /// and a 4-token window narrows every sliding layer's keys.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_decode_graph_matches_the_candle_path_on_metal() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipped");
            return;
        };
        for (window, dtype) in [(4usize, DType::F16), (300, DType::F16), (4, DType::F32)] {
            let mut c = test_config();
            c.head_dim = 128;
            c.context_length = 512;
            c.sliding_window = window;
            let file = GgufFile::from_bytes(&tiny_flint(&c, VOCAB, 0.5, 7)).unwrap();
            let m = Model::from_file(file, &dev, 512).unwrap();
            assert!(
                m.decode_graph().unwrap().is_some(),
                "the graph takes this model"
            );
            let prompt: Vec<u32> = (0..200u32).map(|i| (i * 37) % 256).collect();
            let mut ops = KvCache::with_dtype(c.layers, dtype);
            let mut graph = KvCache::with_dtype(c.layers, dtype);
            m.forward_with(&prompt, 0, &mut ops, false).unwrap();
            m.forward_with(&prompt, 0, &mut graph, false).unwrap();
            let mut tok = 5u32;
            for pos in 200..240 {
                let want = m.forward_with(&[tok], pos, &mut ops, false).unwrap();
                let got = m.forward_with(&[tok], pos, &mut graph, true).unwrap();
                assert!(
                    close(&got, &want, 1e-3),
                    "window {window}, {dtype:?}, position {pos}"
                );
                tok = want
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| u32::try_from(i).unwrap());
            }
            assert_eq!(graph.len(), 240);
        }
    }
}
