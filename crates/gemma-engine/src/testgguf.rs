//! A tiny, randomly weighted Gemma 4 GGUF for tests.
//!
//! The metadata keys and tensor names are the ones the model loader reads, so
//! a file written here loads like a real one, only small. Shapes are given in
//! candle's order: `gguf_file::write` stores dims reversed, as GGUF does, and
//! `Content::read` reverses them back, so a `[a, b]` tensor reads as `[a, b]`.

use std::path::Path;

use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{Device, Tensor};

use crate::template::CONTROL_SPELLINGS;
use crate::{Error, Result};

/// The shape of the model [`write_tiny`] produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TinyConfig {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    /// Head dim of the global layers (`key_length`).
    pub head_dim: usize,
    /// Head dim of the sliding layers (`key_length_swa`), distinct by default
    /// because the real model's are (512 vs 256).
    pub swa_head_dim: usize,
    pub sliding_window: usize,
    pub shared_kv_layers: usize,
    pub context: usize,
    /// `embedding_length_per_layer_input`; `0` writes no per-layer tensors.
    pub per_layer_dim: usize,
    pub seed: u64,
}

impl Default for TinyConfig {
    fn default() -> Self {
        Self {
            layers: 4,
            hidden: 32,
            heads: 2,
            kv_heads: 1,
            head_dim: 16,
            swa_head_dim: 8,
            sliding_window: 4,
            shared_kv_layers: 2,
            context: 4096,
            per_layer_dim: 8,
            seed: 7,
        }
    }
}

const TYPE_NORMAL: i32 = 1;
const TYPE_CONTROL: i32 = 3;
const TYPE_BYTE: i32 = 6;

/// The tiny vocabulary: controls, `<bos>`, `<eos>`, `<unk>`, the 256 byte
/// tokens, `▁`, printable ASCII, then the merged pieces. Returns the tokens,
/// their GGUF token types, and the merges.
#[must_use]
pub fn tiny_vocab() -> (Vec<String>, Vec<i32>, Vec<String>) {
    let mut tokens: Vec<String> = Vec::new();
    let mut types = Vec::new();
    for s in CONTROL_SPELLINGS {
        tokens.push(s.to_string());
        types.push(TYPE_CONTROL);
    }
    for s in ["<bos>", "<eos>"] {
        tokens.push(s.to_string());
        types.push(TYPE_CONTROL);
    }
    tokens.push("<unk>".to_string());
    types.push(TYPE_NORMAL);
    for b in 0..=255u8 {
        tokens.push(format!("<0x{b:02X}>"));
        types.push(TYPE_BYTE);
    }
    tokens.push("▁".to_string());
    types.push(TYPE_NORMAL);
    for c in '!'..='~' {
        tokens.push(c.to_string());
        types.push(TYPE_NORMAL);
    }
    for s in ["▁t", "he", "▁the"] {
        tokens.push(s.to_string());
        types.push(TYPE_NORMAL);
    }
    let merges = ["▁ t", "h e", "▁t he"].map(String::from).to_vec();
    (tokens, types, merges)
}

fn index_of(tokens: &[String], s: &str) -> Result<u32> {
    let i = tokens
        .iter()
        .position(|t| t == s)
        .ok_or_else(|| Error(format!("tiny vocab lacks {s}")))?;
    u32_of(i)
}

fn u32_of(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error(format!("{n} does not fit a GGUF u32")))
}

/// The sliding/global pattern: alternating, aligned so the last two
/// KV-owning layers are sliding then global, as the loader requires.
fn sliding_pattern(cfg: &TinyConfig) -> Vec<bool> {
    let kv = cfg.layers - cfg.shared_kv_layers;
    (0..cfg.layers).map(|i| i % 2 != (kv - 1) % 2).collect()
}

/// Every metadata entry [`write_tiny`] writes, in order.
///
/// # Errors
/// When the config cannot describe a loadable model.
pub fn tiny_metadata(cfg: &TinyConfig) -> Result<Vec<(String, gguf_file::Value)>> {
    use gguf_file::Value as V;
    if cfg.shared_kv_layers + 2 > cfg.layers {
        return Err(Error(format!(
            "tiny gguf needs two KV-owning layers: {} layers, {} shared",
            cfg.layers, cfg.shared_kv_layers
        )));
    }
    let (tokens, types, merges) = tiny_vocab();
    let bos = index_of(&tokens, "<bos>")?;
    let eos = index_of(&tokens, "<eos>")?;
    let u = |n: usize| u32_of(n).map(V::U32);
    let mut m: Vec<(&str, V)> = vec![
        ("general.architecture", V::String("gemma4".into())),
        ("gemma4.block_count", u(cfg.layers)?),
        ("gemma4.context_length", u(cfg.context)?),
        ("gemma4.embedding_length", u(cfg.hidden)?),
        ("gemma4.attention.head_count", u(cfg.heads)?),
        ("gemma4.attention.head_count_kv", u(cfg.kv_heads)?),
        ("gemma4.attention.key_length", u(cfg.head_dim)?),
        ("gemma4.attention.key_length_swa", u(cfg.swa_head_dim)?),
        ("gemma4.attention.layer_norm_rms_epsilon", V::F32(1e-6)),
        ("gemma4.rope.freq_base", V::F32(1e6)),
        ("gemma4.rope.freq_base_swa", V::F32(1e4)),
        ("gemma4.final_logit_softcapping", V::F32(30.0)),
        ("gemma4.attention.sliding_window", u(cfg.sliding_window)?),
    ];
    m.push((
        "gemma4.attention.sliding_window_pattern",
        V::Array(sliding_pattern(cfg).into_iter().map(V::Bool).collect()),
    ));
    m.extend([
        (
            "gemma4.attention.shared_kv_layers",
            u(cfg.shared_kv_layers)?,
        ),
        (
            "gemma4.embedding_length_per_layer_input",
            u(cfg.per_layer_dim)?,
        ),
        ("tokenizer.ggml.model", V::String("gemma4".into())),
        (
            "tokenizer.ggml.tokens",
            V::Array(tokens.into_iter().map(V::String).collect()),
        ),
        (
            "tokenizer.ggml.merges",
            V::Array(merges.into_iter().map(V::String).collect()),
        ),
        (
            "tokenizer.ggml.token_type",
            V::Array(types.into_iter().map(V::I32).collect()),
        ),
        ("tokenizer.ggml.bos_token_id", V::U32(bos)),
        ("tokenizer.ggml.eos_token_id", V::U32(eos)),
        ("tokenizer.ggml.add_bos_token", V::Bool(true)),
    ]);
    Ok(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// splitmix64: a fixed-seed generator, because candle cannot seed its CPU rng.
#[derive(Debug)]
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1], from the high 32 bits.
    fn unit(&mut self) -> f64 {
        let hi = u32::try_from(self.next_u64() >> 32).unwrap_or(u32::MAX);
        (f64::from(hi) + 1.0) / 4_294_967_296.0
    }

    /// Standard normal, by Box-Muller.
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// Normal weights scaled by 0.02, stored as `dtype`.
fn random_as(rng: &mut Rng, shape: &[usize], dtype: GgmlDType) -> Result<QTensor> {
    let n = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|_| rng.normal() * 0.02).collect();
    let t = Tensor::from_vec(data, shape, &Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
    Ok(QTensor::quantize(&t, dtype)?)
}

/// Normal weights scaled by 0.02, stored as f32.
fn random(rng: &mut Rng, shape: &[usize]) -> Result<QTensor> {
    random_as(rng, shape, GgmlDType::F32)
}

fn filled(shape: &[usize], value: f64) -> Result<QTensor> {
    let t = (Tensor::ones(shape, candle_core::DType::F32, &Device::Cpu)? * value)?;
    Ok(QTensor::quantize(&t, GgmlDType::F32)?)
}

fn ones(shape: &[usize]) -> Result<QTensor> {
    filled(shape, 1.0)
}

/// Every tensor [`write_tiny`] writes, in order, in candle shape order.
///
/// Sliding layers use `swa_head_dim`, global layers `head_dim`, as in the
/// real model. With `per_layer_dim > 0` the per-layer input tensors are
/// written in the real file's types where candle can quantize them: the
/// per-layer token table as `Q8_0` (when its rows are whole `Q8_0` blocks, else
/// f32), the model projection as BF16, gates and projections as f32.
///
/// # Errors
/// When candle fails to build a tensor.
pub fn tiny_tensors(cfg: &TinyConfig) -> Result<Vec<(String, QTensor)>> {
    let mut rng = Rng(cfg.seed);
    let vocab = tiny_vocab().0.len();
    let (h, ff, pl) = (cfg.hidden, 4 * cfg.hidden, cfg.per_layer_dim);
    let pl_all = cfg.layers * pl;
    let mut out = vec![
        (
            "token_embd.weight".to_string(),
            random(&mut rng, &[vocab, h])?,
        ),
        ("rope_freqs.weight".to_string(), ones(&[cfg.head_dim / 2])?),
        ("output_norm.weight".to_string(), ones(&[h])?),
    ];
    if pl > 0 {
        let table = if pl_all.is_multiple_of(32) {
            GgmlDType::Q8_0
        } else {
            GgmlDType::F32
        };
        out.push((
            "per_layer_token_embd.weight".to_string(),
            random_as(&mut rng, &[vocab, pl_all], table)?,
        ));
        out.push((
            "per_layer_model_proj.weight".to_string(),
            random_as(&mut rng, &[pl_all, h], GgmlDType::BF16)?,
        ));
        out.push(("per_layer_proj_norm.weight".to_string(), ones(&[pl])?));
    }
    let sliding = sliding_pattern(cfg);
    let kv_layers = cfg.layers - cfg.shared_kv_layers;
    for (i, &is_sliding) in sliding.iter().enumerate() {
        let hd = if is_sliding {
            cfg.swa_head_dim
        } else {
            cfg.head_dim
        };
        let (q, kv) = (cfg.heads * hd, cfg.kv_heads * hd);
        let p = |name: &str| format!("blk.{i}.{name}.weight");
        out.push((p("attn_q"), random(&mut rng, &[q, h])?));
        if i < kv_layers {
            out.push((p("attn_k"), random(&mut rng, &[kv, h])?));
            out.push((p("attn_v"), random(&mut rng, &[kv, h])?));
        }
        out.push((p("attn_output"), random(&mut rng, &[h, q])?));
        out.push((p("attn_q_norm"), ones(&[hd])?));
        out.push((p("attn_k_norm"), ones(&[hd])?));
        for norm in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            out.push((p(norm), ones(&[h])?));
        }
        out.push((p("ffn_gate"), random(&mut rng, &[ff, h])?));
        out.push((p("ffn_up"), random(&mut rng, &[ff, h])?));
        out.push((p("ffn_down"), random(&mut rng, &[h, ff])?));
        if pl > 0 {
            out.push((p("inp_gate"), random(&mut rng, &[pl, h])?));
            out.push((p("proj"), random(&mut rng, &[h, pl])?));
            out.push((p("post_norm"), ones(&[h])?));
        }
        out.push((p("layer_output_scale"), filled(&[1], 0.75)?));
    }
    Ok(out)
}

/// Writes `metadata` and `tensors` as a GGUF at `path`, for tests that
/// need a file [`write_tiny`] would refuse to describe.
///
/// # Errors
/// On an I/O or candle failure.
pub fn write_gguf(
    path: &Path,
    metadata: &[(String, gguf_file::Value)],
    tensors: &[(String, QTensor)],
) -> Result<()> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| Error(format!("create {}: {e}", path.display())))?;
    let md: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let ts: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    gguf_file::write(&mut f, &md, &ts)?;
    Ok(())
}

/// Writes a tiny Gemma 4 GGUF to `path`.
///
/// # Errors
/// When the config is not loadable, or on an I/O or candle failure.
pub fn write_tiny(path: &Path, cfg: &TinyConfig) -> Result<()> {
    write_gguf(path, &tiny_metadata(cfg)?, &tiny_tensors(cfg)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gemma-testgguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn read(path: &Path) -> (std::fs::File, gguf_file::Content) {
        let mut f = std::fs::File::open(path).unwrap();
        let ct = gguf_file::Content::read(&mut f).unwrap();
        (f, ct)
    }

    #[test]
    fn a_candle_shape_reads_back_unchanged_with_its_values() {
        let path = temp("shape.gguf");
        let data: Vec<f32> = (0..6u8).map(f32::from).collect();
        let t = Tensor::from_vec(data.clone(), &[2, 3], &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::F32).unwrap();
        write_gguf(&path, &[], &[("w".to_string(), q)]).unwrap();
        let (mut f, ct) = read(&path);
        assert_eq!(ct.tensor_infos["w"].shape.dims(), &[2, 3]);
        let back = ct.tensor(&mut f, "w", &Device::Cpu).unwrap();
        let back = back.dequantize(&Device::Cpu).unwrap();
        assert_eq!(back.dims(), &[2, 3]);
        assert_eq!(back.flatten_all().unwrap().to_vec1::<f32>().unwrap(), data);
    }

    #[test]
    fn tiny_file_carries_the_loader_shapes_and_pattern() {
        let cfg = TinyConfig::default();
        let path = temp("tiny.gguf");
        write_tiny(&path, &cfg).unwrap();
        let (_, ct) = read(&path);
        let vocab = tiny_vocab().0.len();
        let dims = |n: &str| ct.tensor_infos[n].shape.dims().to_vec();
        assert_eq!(dims("token_embd.weight"), [vocab, 32]);
        assert_eq!(dims("blk.0.ffn_gate.weight"), [128, 32]);
        assert_eq!(dims("blk.0.ffn_down.weight"), [32, 128]);
        assert_eq!(dims("blk.1.attn_k.weight"), [16, 32]);
        assert!(!ct.tensor_infos.contains_key("blk.2.attn_k.weight"));
        // Sliding layers take `swa_head_dim` (8), global ones `head_dim` (16).
        assert_eq!(dims("blk.0.attn_q.weight"), [16, 32]);
        assert_eq!(dims("blk.1.attn_q.weight"), [32, 32]);
        assert_eq!(dims("blk.0.attn_q_norm.weight"), [8]);
        assert_eq!(dims("per_layer_token_embd.weight"), [vocab, 32]);
        assert_eq!(
            ct.tensor_infos["per_layer_token_embd.weight"].ggml_dtype,
            GgmlDType::Q8_0
        );
        assert_eq!(
            ct.tensor_infos["per_layer_model_proj.weight"].ggml_dtype,
            GgmlDType::BF16
        );
        let pattern: Vec<bool> = ct.metadata["gemma4.attention.sliding_window_pattern"]
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bool().unwrap())
            .collect();
        assert_eq!(pattern, [true, false, true, false]);
    }

    #[test]
    fn same_seed_writes_the_same_weights() {
        let cfg = TinyConfig::default();
        let a = tiny_tensors(&cfg).unwrap();
        let b = tiny_tensors(&cfg).unwrap();
        assert_eq!(a[0].1.data().unwrap(), b[0].1.data().unwrap());
        let other = tiny_tensors(&TinyConfig { seed: 8, ..cfg }).unwrap();
        assert_ne!(a[0].1.data().unwrap(), other[0].1.data().unwrap());
    }

    #[test]
    fn too_few_kv_layers_is_an_error() {
        let cfg = TinyConfig {
            shared_kv_layers: 3,
            ..TinyConfig::default()
        };
        assert!(tiny_metadata(&cfg).is_err());
    }
}
