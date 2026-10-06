//! Gemma 4 forward pass over a quantized GGUF.
//!
//! Adapted from candle's `candle-transformers/src/models/quantized_gemma4.rs`
//! as proposed in candle PR #3932 (gabgiani/candle @ 02d5c49), Copyright the
//! candle authors, MIT OR Apache-2.0. The config, embeddings, rotary
//! embedding, attention, MLP, layer, norms, mask and metadata helpers are the
//! PR's; the changes are:
//!
//! - candle 0.9.2 has no `QMatMul::embedding`, so [`QEmbedding`] looks rows
//!   up in the raw quantized bytes, dequantizing only the rows it returns.
//! - The PR's ring/concat KV caches are gone: attention appends to and reads
//!   from [`crate::kv::LayerKv`], which keeps every position, so sliding
//!   layers mask on every forward, decode steps included.
//! - Weights are split from state: [`Model`] is immutable and `forward` takes
//!   the caller's [`KvCache`].
//! - Rotary tables cover `min(context_length, ctx_cap)` positions.
//! - Rotary tables are f32 on purpose, unlike the PR, which rounds them
//!   through `general.dtype` (f16 by default): llama.cpp computes rope in f32
//!   and every activation here is f32.
//! - `Config::from_gguf` validates head counts, head dims and the window, so
//!   a crafted file fails with an error instead of a panic or a silently
//!   wrong head grouping.
//! - Sliding layers attend only to the keys their window can reach (the rest
//!   would be masked to exactly zero weight anyway).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, ggml_file, gguf_file};
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::Activation;
use sha2::{Digest, Sha256};

use crate::kv::{KvCache, LayerKv};
use crate::tokenizer::GemmaTokenizer;

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (batch, heads, sequence, dim) = x.dims4()?;
    Tensor::cat(&vec![&x; n_rep], 2)?.reshape((batch, heads * n_rep, sequence, dim))
}

/// A crate error as a candle one, inside the candle-typed internals.
#[allow(clippy::needless_pass_by_value)] // shaped for `map_err`
fn candle_err(e: crate::Error) -> candle_core::Error {
    candle_core::Error::Msg(e.0)
}

#[derive(Clone, Debug)]
struct Config {
    block_count: usize,
    context_length: usize,
    hidden_size: usize,
    attention_heads: Vec<usize>,
    kv_heads: Vec<usize>,
    global_head_dim: usize,
    local_head_dim: usize,
    rms_norm_eps: f64,
    global_rope_base: f64,
    local_rope_base: f64,
    final_logit_softcap: f64,
    sliding_window: usize,
    shared_kv_layers: usize,
    per_layer_input_dim: usize,
    has_per_layer_inputs: bool,
    sliding_pattern: Vec<bool>,
}

impl Config {
    fn from_gguf(content: &gguf_file::Content) -> Result<Self> {
        let architecture = metadata(content, "general.architecture")?.to_string()?;
        if architecture != "gemma4" {
            candle_core::bail!("model architecture is {architecture}, expected gemma4")
        }
        // The 26B-A4B mixture-of-experts files share the architecture; running only their dense
        // feed-forward weights would load and silently produce wrong logits.
        if let Some(experts) = content.metadata.get("gemma4.expert_count")
            && experts.to_u32()? > 0
        {
            candle_core::bail!("gemma4 MoE models are not supported yet")
        }
        let sliding_pattern = metadata(content, "gemma4.attention.sliding_window_pattern")?
            .to_vec()?
            .iter()
            .map(gguf_file::Value::to_bool)
            .collect::<Result<Vec<_>>>()?;
        let block_count = usize_metadata(content, "gemma4.block_count")?;
        if sliding_pattern.len() != block_count {
            candle_core::bail!(
                "Gemma 4 sliding pattern has {} entries for {block_count} blocks",
                sliding_pattern.len()
            )
        }
        let shared_kv_layers = usize_metadata(content, "gemma4.attention.shared_kv_layers")?;
        if shared_kv_layers + 2 > block_count {
            candle_core::bail!(
                "Gemma 4 requires at least two KV-owning layers, got {block_count} blocks and {shared_kv_layers} shared layers"
            )
        }
        let kv_layer_count = block_count - shared_kv_layers;
        if !sliding_pattern[kv_layer_count - 2] || sliding_pattern[kv_layer_count - 1] {
            candle_core::bail!(
                "Gemma 4 shared KV source layers must end with one sliding and one global layer"
            )
        }
        let per_layer_input_dim =
            usize_metadata(content, "gemma4.embedding_length_per_layer_input")?;
        let config = Self {
            block_count,
            context_length: usize_metadata(content, "gemma4.context_length")?,
            hidden_size: usize_metadata(content, "gemma4.embedding_length")?,
            attention_heads: per_layer_usize_metadata(
                content,
                "gemma4.attention.head_count",
                block_count,
            )?,
            kv_heads: per_layer_usize_metadata(
                content,
                "gemma4.attention.head_count_kv",
                block_count,
            )?,
            global_head_dim: usize_metadata(content, "gemma4.attention.key_length")?,
            local_head_dim: usize_metadata(content, "gemma4.attention.key_length_swa")?,
            rms_norm_eps: f64::from(
                metadata(content, "gemma4.attention.layer_norm_rms_epsilon")?.to_f32()?,
            ),
            global_rope_base: f64::from(metadata(content, "gemma4.rope.freq_base")?.to_f32()?),
            local_rope_base: f64::from(metadata(content, "gemma4.rope.freq_base_swa")?.to_f32()?),
            final_logit_softcap: f64::from(
                metadata(content, "gemma4.final_logit_softcapping")?.to_f32()?,
            ),
            sliding_window: usize_metadata(content, "gemma4.attention.sliding_window")?,
            shared_kv_layers,
            per_layer_input_dim: per_layer_input_dim.max(1),
            has_per_layer_inputs: per_layer_input_dim > 0,
            sliding_pattern,
        };
        config.validate()?;
        Ok(config)
    }

    /// Refuses values the forward pass would divide by, halve, or group
    /// heads by, so a crafted file errors at open instead of panicking.
    fn validate(&self) -> Result<()> {
        for (layer, (&heads, &kv_heads)) in
            self.attention_heads.iter().zip(&self.kv_heads).enumerate()
        {
            if heads == 0 || kv_heads == 0 {
                candle_core::bail!(
                    "Gemma 4 layer {layer} has {heads} attention heads and {kv_heads} KV heads; both must be positive"
                )
            }
            if !heads.is_multiple_of(kv_heads) {
                candle_core::bail!(
                    "Gemma 4 layer {layer}: {heads} attention heads are not a multiple of {kv_heads} KV heads"
                )
            }
        }
        if self.sliding_window == 0 {
            candle_core::bail!("Gemma 4 sliding window must be positive")
        }
        for (name, dim) in [
            ("key_length", self.global_head_dim),
            ("key_length_swa", self.local_head_dim),
        ] {
            if dim == 0 || !dim.is_multiple_of(2) {
                candle_core::bail!("Gemma 4 {name} is {dim}; it must be positive and even")
            }
        }
        Ok(())
    }

    fn is_sliding(&self, layer: usize) -> bool {
        self.sliding_pattern[layer]
    }

    fn head_dim(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.local_head_dim
        } else {
            self.global_head_dim
        }
    }
}

/// A row-lookup embedding over a quantized table.
///
/// It keeps the tensor's raw GGUF bytes and dequantizes only the rows a
/// lookup asks for, so a table too large to dequantize whole (the E4B
/// per-layer table is 262144 x 10752 Q5K, ~2.8B values) costs only its
/// quantized size. Rows are whole blocks: `cols` must be a multiple of the
/// type's block size, and a row is `cols / block_size * type_size` bytes.
#[derive(Debug)]
pub(crate) struct QEmbedding {
    data: Vec<u8>,
    dtype: GgmlDType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

impl QEmbedding {
    fn from_raw(data: Vec<u8>, dtype: GgmlDType, rows: usize, cols: usize) -> Result<Self> {
        match dtype {
            GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8_0
            | GgmlDType::Q4_0
            | GgmlDType::F16
            | GgmlDType::BF16
            | GgmlDType::F32 => {}
            other => candle_core::bail!("embedding: unsupported tensor type {other:?}"),
        }
        let block = dtype.block_size();
        if !cols.is_multiple_of(block) {
            candle_core::bail!(
                "embedding: {cols} columns are not whole {dtype:?} blocks of {block}"
            )
        }
        let row_bytes = cols / block * dtype.type_size();
        if data.len() != rows * row_bytes {
            candle_core::bail!(
                "embedding: {} bytes for {rows} rows of {row_bytes}",
                data.len()
            )
        }
        Ok(Self {
            data,
            dtype,
            rows,
            cols,
            row_bytes,
        })
    }

    /// Reads tensor `name`'s bytes straight from the file, never building
    /// the whole tensor on a device.
    fn read<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        name: &str,
    ) -> Result<Self> {
        let Some(info) = content.tensor_infos.get(name) else {
            candle_core::bail!("cannot find tensor info for {name}")
        };
        let (rows, cols) = info.shape.dims2()?;
        let block = info.ggml_dtype.block_size();
        let Some(size) = rows
            .checked_mul(cols)
            .and_then(|n| (n / block).checked_mul(info.ggml_dtype.type_size()))
        else {
            candle_core::bail!("{name}: shape [{rows}, {cols}] overflows its byte size")
        };
        // Check the file holds the bytes before allocating what the header
        // claims.
        let end = reader.seek(SeekFrom::End(0))?;
        let start = content
            .tensor_data_offset
            .checked_add(info.offset)
            .filter(|&start| start <= end);
        let Some(start) = start.filter(|&start| (end - start) >= size as u64) else {
            candle_core::bail!("{name}: {size} bytes declared, past the end of the file")
        };
        let mut data = vec![0u8; size];
        reader.seek(SeekFrom::Start(start))?;
        reader.read_exact(&mut data)?;
        Self::from_raw(data, info.ggml_dtype, rows, cols)
    }

    #[cfg(test)]
    fn from_qtensor(q: &candle_core::quantized::QTensor) -> Result<Self> {
        let (rows, cols) = q.shape().dims2()?;
        Self::from_raw(q.data()?.into_owned(), q.dtype(), rows, cols)
    }

    /// The rows for `ids`, as an f32 `[ids.len(), cols]` tensor on `device`.
    ///
    /// The selected rows are gathered into one buffer and dequantized on
    /// the CPU through candle's own block routines (`GgmlType::to_float`,
    /// via `qtensor_from_ggml` + `dequantize`), so the values are exactly
    /// those of a full dequantization.
    fn forward(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        let mut rows = Vec::with_capacity(ids.len() * self.row_bytes);
        for &id in ids {
            let row = id as usize;
            if row >= self.rows {
                candle_core::bail!("token id {id} is outside the {}-row embedding", self.rows)
            }
            rows.extend_from_slice(&self.data[row * self.row_bytes..(row + 1) * self.row_bytes]);
        }
        let q = ggml_file::qtensor_from_ggml(
            self.dtype,
            &rows,
            vec![ids.len(), self.cols],
            &Device::Cpu,
        )?;
        q.dequantize(&Device::Cpu)?.to_device(device)
    }
}

struct Embeddings {
    token_embedding: QEmbedding,
    per_layer_token_embedding: Option<QEmbedding>,
    per_layer_model_projection: Option<QMatMul>,
    per_layer_projection_norm: Option<GemmaRmsNorm>,
    hidden_size: usize,
    layer_count: usize,
    per_layer_input_dim: usize,
}

impl Embeddings {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        config: &Config,
    ) -> Result<Self> {
        Ok(Self {
            token_embedding: QEmbedding::read(content, reader, "token_embd.weight")?,
            per_layer_token_embedding: config
                .has_per_layer_inputs
                .then(|| QEmbedding::read(content, reader, "per_layer_token_embd.weight"))
                .transpose()?,
            per_layer_model_projection: config
                .has_per_layer_inputs
                .then(|| qmatmul(content, reader, "per_layer_model_proj.weight", device))
                .transpose()?,
            per_layer_projection_norm: config
                .has_per_layer_inputs
                .then(|| {
                    GemmaRmsNorm::load(
                        content,
                        reader,
                        "per_layer_proj_norm.weight",
                        device,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?,
            hidden_size: config.hidden_size,
            layer_count: config.block_count,
            per_layer_input_dim: config.per_layer_input_dim,
        })
    }

    // Model dimensions are far below 2^52, exact in an f64.
    #[allow(clippy::cast_precision_loss)]
    fn forward(&self, tokens: &[u32], device: &Device) -> Result<(Tensor, Tensor)> {
        let (batch, sequence) = (1, tokens.len());
        let hidden = (self.token_embedding.forward(tokens, device)?.unsqueeze(0)?
            * (self.hidden_size as f64).sqrt())?;
        let per_layer_inputs = match (
            &self.per_layer_token_embedding,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            (Some(token_embedding), Some(model_projection), Some(projection_norm)) => {
                let token_inputs = (token_embedding.forward(tokens, device)?
                    * (self.per_layer_input_dim as f64).sqrt())?
                .reshape((batch, sequence, self.layer_count, self.per_layer_input_dim))?;
                let projected_inputs = (model_projection.forward(&hidden)?
                    * (1.0 / (self.hidden_size as f64).sqrt()))?
                .reshape((batch, sequence, self.layer_count, self.per_layer_input_dim))?;
                let projected_inputs = projection_norm.forward(&projected_inputs)?;
                ((token_inputs + projected_inputs)? * (1.0 / 2f64.sqrt()))?
            }
            (None, None, None) => Tensor::zeros(
                (batch, sequence, self.layer_count, self.per_layer_input_dim),
                hidden.dtype(),
                hidden.device(),
            )?,
            _ => candle_core::bail!("Gemma 4 per-layer embedding tensors are incomplete"),
        };
        Ok((hidden, per_layer_inputs))
    }
}

#[derive(Debug)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    // Head dims and positions are small integers, exact in f64/f32; the
    // inverse frequencies are computed in f64 and stored as f32 on purpose.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_base: f64,
        freq_factors: Option<&Tensor>,
        context_length: usize,
        device: &Device,
    ) -> Result<Self> {
        let inverse_frequencies = (0..head_dim / 2)
            .map(|index| 1f32 / rope_base.powf((index * 2) as f64 / head_dim as f64) as f32)
            .collect::<Vec<_>>();
        let mut inverse_frequencies =
            Tensor::from_vec(inverse_frequencies, (1, head_dim / 2), device)?;
        if let Some(freq_factors) = freq_factors {
            inverse_frequencies =
                inverse_frequencies.broadcast_div(&freq_factors.reshape((1, head_dim / 2))?)?;
        }
        let Ok(positions) = u32::try_from(context_length) else {
            candle_core::bail!("context length {context_length} does not fit a u32")
        };
        let positions = Tensor::arange(0u32, positions, device)?
            .to_dtype(DType::F32)?
            .reshape((context_length, 1))?;
        let frequencies = positions.matmul(&inverse_frequencies)?;
        Ok(Self {
            sin: frequencies.sin()?.to_dtype(dtype)?,
            cos: frequencies.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(
        &self,
        query: &Tensor,
        key: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let sequence = query.dim(2)?;
        let cosine = self
            .cos
            .narrow(0, offset, sequence)?
            .to_dtype(query.dtype())?;
        let sine = self
            .sin
            .narrow(0, offset, sequence)?
            .to_dtype(query.dtype())?;
        let query = candle_nn::rotary_emb::rope(&query.contiguous()?, &cosine, &sine)?;
        let key = key
            .map(|key| candle_nn::rotary_emb::rope(&key.contiguous()?, &cosine, &sine))
            .transpose()?;
        Ok((query, key))
    }
}

struct Attention {
    query: QMatMul,
    key: Option<QMatMul>,
    value: Option<QMatMul>,
    output: QMatMul,
    query_norm: GemmaRmsNorm,
    key_norm: Option<GemmaRmsNorm>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    is_sliding: bool,
    cache_index: usize,
    rotary: Arc<RotaryEmbedding>,
    rms_norm_eps: f64,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        config: &Config,
        layer_index: usize,
        kv_layer_count: usize,
        global_rotary: Arc<RotaryEmbedding>,
        local_rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_index}");
        let is_sliding = config.is_sliding(layer_index);
        let owns_kv = layer_index < kv_layer_count;
        let cache_index = if owns_kv {
            layer_index
        } else if is_sliding {
            kv_layer_count - 2
        } else {
            kv_layer_count - 1
        };
        Ok(Self {
            query: qmatmul(content, reader, &format!("{prefix}.attn_q.weight"), device)?,
            key: owns_kv
                .then(|| qmatmul(content, reader, &format!("{prefix}.attn_k.weight"), device))
                .transpose()?,
            value: (owns_kv
                && content
                    .tensor_infos
                    .contains_key(&format!("{prefix}.attn_v.weight")))
            .then(|| qmatmul(content, reader, &format!("{prefix}.attn_v.weight"), device))
            .transpose()?,
            output: qmatmul(
                content,
                reader,
                &format!("{prefix}.attn_output.weight"),
                device,
            )?,
            query_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.attn_q_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            key_norm: owns_kv
                .then(|| {
                    GemmaRmsNorm::load(
                        content,
                        reader,
                        &format!("{prefix}.attn_k_norm.weight"),
                        device,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?,
            heads: config.attention_heads[layer_index],
            kv_heads: config.kv_heads[layer_index],
            head_dim: config.head_dim(layer_index),
            is_sliding,
            cache_index,
            rotary: if is_sliding {
                local_rotary
            } else {
                global_rotary
            },
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        offset: usize,
        cache: &mut LayerKv,
        device: &Device,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        let query = self
            .query
            .forward(hidden)?
            .reshape((batch, sequence, self.heads, self.head_dim))?
            .transpose(1, 2)?;
        let query = self.query_norm.forward(&query)?;
        let key = self
            .key
            .as_ref()
            .map(|key| {
                key.forward(hidden)?
                    .reshape((batch, sequence, self.kv_heads, self.head_dim))?
                    .transpose(1, 2)
            })
            .transpose()?;
        let value = self
            .value
            .as_ref()
            .map(|value| {
                value
                    .forward(hidden)?
                    .reshape((batch, sequence, self.kv_heads, self.head_dim))?
                    .transpose(1, 2)
            })
            .transpose()?;
        let value = match (value, key.as_ref()) {
            (Some(value), _) => Some(value),
            (None, Some(key)) => Some(key.clone()),
            (None, None) => None,
        };
        let key = match (key, self.key_norm.as_ref()) {
            (Some(key), Some(norm)) => Some(norm.forward(&key)?),
            (None, None) => None,
            _ => candle_core::bail!("Gemma 4 key projection and key norm ownership differ"),
        };
        let value = value
            .map(|value| value_norm(&value, self.rms_norm_eps))
            .transpose()?;
        let (query, key) = self.rotary.apply(&query, key.as_ref(), offset)?;
        // Owners extend their cache; sharers read their owner's, which the
        // owner (an earlier layer) already extended in this forward.
        match (key, value) {
            (Some(key), Some(value)) => cache.append(&key, &value).map_err(candle_err)?,
            (None, None) => {}
            _ => candle_core::bail!("Gemma 4 key and value ownership differ"),
        }
        let Some((key, value)) = cache.view().map_err(candle_err)? else {
            candle_core::bail!("Gemma 4 shared KV cache is empty")
        };
        let stored = key.dim(2)?;
        if stored != offset + sequence {
            candle_core::bail!(
                "Gemma 4 KV cache holds {stored} positions, expected {}",
                offset + sequence
            )
        }
        // The cache keeps every position. A sliding layer's earliest query
        // (at `offset`) sees no key before `offset + 1 - window`, so the
        // keys before that are dropped here; the mask still runs below.
        let start = if self.is_sliding {
            (offset + 1).saturating_sub(sliding_window)
        } else {
            0
        };
        let (key, value) = if start > 0 {
            (
                key.narrow(2, start, stored - start)?,
                value.narrow(2, start, stored - start)?,
            )
        } else {
            (key, value)
        };
        let key_positions: Vec<usize> = (start..stored).collect();
        let key = repeat_kv(key, self.heads / self.kv_heads)?.contiguous()?;
        let value = repeat_kv(value, self.heads / self.kv_heads)?.contiguous()?;
        let mut scores = query.matmul(&key.transpose(2, 3)?)?;
        if self.is_sliding || sequence > 1 {
            let mask = attention_mask(
                sequence,
                offset,
                &key_positions,
                self.is_sliding.then_some(sliding_window),
                device,
                scores.dtype(),
            )?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probabilities.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            sequence,
            self.heads * self.head_dim,
        ))?;
        self.output.forward(&context)
    }
}

struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            gate: qmatmul(
                content,
                reader,
                &format!("{prefix}.ffn_gate.weight"),
                device,
            )?,
            up: qmatmul(content, reader, &format!("{prefix}.ffn_up.weight"), device)?,
            down: qmatmul(
                content,
                reader,
                &format!("{prefix}.ffn_down.weight"),
                device,
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate
            .forward(hidden)?
            .apply(&Activation::GeluPytorchTanh)?;
        self.down.forward(&(gate * self.up.forward(hidden)?)?)
    }
}

struct Layer {
    attention: Attention,
    mlp: Mlp,
    attention_norm: GemmaRmsNorm,
    post_attention_norm: GemmaRmsNorm,
    ffn_norm: GemmaRmsNorm,
    post_ffn_norm: GemmaRmsNorm,
    per_layer_input_gate: Option<QMatMul>,
    per_layer_projection: Option<QMatMul>,
    per_layer_post_norm: Option<GemmaRmsNorm>,
    output_scale: Option<Tensor>,
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        config: &Config,
        layer_index: usize,
        kv_layer_count: usize,
        global_rotary: Arc<RotaryEmbedding>,
        local_rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_index}");
        Ok(Self {
            attention: Attention::load(
                content,
                reader,
                device,
                config,
                layer_index,
                kv_layer_count,
                global_rotary,
                local_rotary,
            )?,
            mlp: Mlp::load(content, reader, device, &prefix)?,
            attention_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.attn_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            post_attention_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.post_attention_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            ffn_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.ffn_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            post_ffn_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.post_ffw_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            per_layer_input_gate: config
                .has_per_layer_inputs
                .then(|| {
                    qmatmul(
                        content,
                        reader,
                        &format!("{prefix}.inp_gate.weight"),
                        device,
                    )
                })
                .transpose()?,
            per_layer_projection: config
                .has_per_layer_inputs
                .then(|| qmatmul(content, reader, &format!("{prefix}.proj.weight"), device))
                .transpose()?,
            per_layer_post_norm: config
                .has_per_layer_inputs
                .then(|| {
                    GemmaRmsNorm::load(
                        content,
                        reader,
                        &format!("{prefix}.post_norm.weight"),
                        device,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?,
            output_scale: content
                .tensor_infos
                .contains_key(&format!("{prefix}.layer_output_scale.weight"))
                .then(|| {
                    content
                        .tensor(
                            reader,
                            &format!("{prefix}.layer_output_scale.weight"),
                            device,
                        )?
                        .dequantize(device)
                })
                .transpose()?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        per_layer_input: &Tensor,
        offset: usize,
        cache: &mut LayerKv,
        device: &Device,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let attention = self.attention.forward(
            &self.attention_norm.forward(hidden)?,
            offset,
            cache,
            device,
            sliding_window,
        )?;
        let attention = self.post_attention_norm.forward(&attention)?;
        let hidden = (hidden + attention)?;
        let mlp = self.mlp.forward(&self.ffn_norm.forward(&hidden)?)?;
        let mlp = self.post_ffn_norm.forward(&mlp)?;
        let hidden = (hidden + mlp)?;
        let hidden = match (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.per_layer_post_norm,
        ) {
            (Some(input_gate), Some(projection), Some(post_norm)) => {
                let gated_input = input_gate
                    .forward(&hidden)?
                    .apply(&Activation::GeluPytorchTanh)?;
                let per_layer_output = projection.forward(&(gated_input * per_layer_input)?)?;
                hidden + post_norm.forward(&per_layer_output)?
            }
            (None, None, None) => Ok(hidden),
            _ => candle_core::bail!("Gemma 4 per-layer residual tensors are incomplete"),
        }?;
        match &self.output_scale {
            Some(output_scale) => hidden.broadcast_mul(output_scale),
            None => Ok(hidden),
        }
    }
}

/// An immutable Gemma 4 text model loaded from a GGUF file.
///
/// All state lives in the caller's [`KvCache`] (see [`crate::session`]), so
/// one model can serve any number of sessions.
pub struct Model {
    pub tokenizer: GemmaTokenizer,
    /// `"Gemma 4"`, then the variant when the file names one (`"Gemma 4 E4B"`).
    pub name: String,
    embeddings: Embeddings,
    layers: Vec<Layer>,
    norm: GemmaRmsNorm,
    output: QMatMul,
    final_logit_softcap: f64,
    device: Device,
    sliding_window: usize,
    context_length: usize,
    signature: [u8; 32],
    kv_shape: Vec<(usize, usize)>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name)
            .field("layers", &self.layers.len())
            .field("context_length", &self.context_length)
            .field("kv_shape", &self.kv_shape)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Model {
    /// Opens `path` with rotary tables for the model's whole context.
    ///
    /// # Errors
    /// On an I/O failure, a file that is not a dense Gemma 4 GGUF, or a
    /// candle failure while loading the weights.
    pub fn open(path: &Path, device: &Device) -> crate::Result<Arc<Self>> {
        Self::open_with_ctx(path, device, usize::MAX)
    }

    /// Opens `path`, building rotary tables for only
    /// `min(context_length, ctx_cap)` positions, which then bounds
    /// [`Model::context_length`].
    ///
    /// # Errors
    /// As [`Model::open`].
    pub fn open_with_ctx(path: &Path, device: &Device, ctx_cap: usize) -> crate::Result<Arc<Self>> {
        let file =
            File::open(path).map_err(|e| crate::Error(format!("open {}: {e}", path.display())))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)?;
        let tokenizer = GemmaTokenizer::from_gguf(&content)?;
        let name = model_name(&content.metadata);
        let signature = signature(&content);
        let model = Self::load(
            &content,
            &mut reader,
            device,
            ctx_cap,
            tokenizer,
            name,
            signature,
        )?;
        Ok(Arc::new(model))
    }

    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        ctx_cap: usize,
        tokenizer: GemmaTokenizer,
        name: String,
        signature: [u8; 32],
    ) -> Result<Self> {
        let config = Config::from_gguf(content)?;
        let kv_layer_count = config.block_count - config.shared_kv_layers;
        let context_length = config.context_length.min(ctx_cap);
        let global_rope_factors = content
            .tensor(reader, "rope_freqs.weight", device)?
            .dequantize(device)?;
        let global_rotary = Arc::new(RotaryEmbedding::new(
            DType::F32,
            config.global_head_dim,
            config.global_rope_base,
            Some(&global_rope_factors),
            context_length,
            device,
        )?);
        let local_rotary = Arc::new(RotaryEmbedding::new(
            DType::F32,
            config.local_head_dim,
            config.local_rope_base,
            None,
            context_length,
            device,
        )?);
        let embeddings = Embeddings::load(content, reader, device, &config)?;
        let mut layers = Vec::with_capacity(config.block_count);
        for index in 0..config.block_count {
            layers.push(Layer::load(
                content,
                reader,
                device,
                &config,
                index,
                kv_layer_count,
                global_rotary.clone(),
                local_rotary.clone(),
            )?);
        }
        let kv_shape = (0..kv_layer_count)
            .map(|index| (config.kv_heads[index], config.head_dim(index)))
            .collect();
        let output_name = if content.tensor_infos.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        Ok(Self {
            tokenizer,
            name,
            embeddings,
            layers,
            norm: GemmaRmsNorm::load(
                content,
                reader,
                "output_norm.weight",
                device,
                config.rms_norm_eps,
            )?,
            output: qmatmul(content, reader, output_name, device)?,
            final_logit_softcap: config.final_logit_softcap,
            device: device.clone(),
            sliding_window: config.sliding_window,
            context_length,
            signature,
            kv_shape,
        })
    }

    /// Positions the model can attend over: the file's context length,
    /// capped by the `ctx_cap` given at open.
    #[must_use]
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    /// SHA-256 over the file's shape: magic/version, every metadata entry
    /// and every tensor's name, shape and type. Never the weights.
    #[must_use]
    pub fn signature(&self) -> [u8; 32] {
        self.signature
    }

    /// `(kv_heads, head_dim)` for each KV-owning layer, in cache order.
    #[must_use]
    pub fn kv_shape(&self) -> Vec<(usize, usize)> {
        self.kv_shape.clone()
    }

    /// Bytes one position adds to the cache: K and V, f32, every owner.
    #[must_use]
    pub fn kv_bytes_per_token(&self) -> usize {
        self.kv_shape.iter().map(|(h, d)| 2 * 4 * h * d).sum()
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Runs `tokens` at positions `offset..`, extending `cache`, and
    /// returns the softcapped logits of the last position as f32.
    ///
    /// On an error the cache may hold part of this forward; the caller
    /// truncates it back to `offset`.
    pub(crate) fn forward(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
    ) -> crate::Result<Vec<f32>> {
        Ok(self.forward_inner(tokens, offset, cache)?)
    }

    fn forward_inner(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            candle_core::bail!("Gemma 4 forward needs at least one token")
        }
        if offset + tokens.len() > self.context_length {
            candle_core::bail!(
                "context full: {} tokens > {}",
                offset + tokens.len(),
                self.context_length
            )
        }
        if cache.layers.len() != self.kv_shape.len() {
            candle_core::bail!(
                "KV cache has {} layers, the model owns {}",
                cache.layers.len(),
                self.kv_shape.len()
            )
        }
        let (mut hidden, per_layer_inputs) = self.embeddings.forward(tokens, &self.device)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let per_layer_input = per_layer_inputs.narrow(2, index, 1)?.squeeze(2)?;
            hidden = layer.forward(
                &hidden,
                &per_layer_input,
                offset,
                &mut cache.layers[layer.attention.cache_index],
                &self.device,
                self.sliding_window,
            )?;
        }
        let sequence = hidden.dim(1)?;
        let hidden = self.norm.forward(&hidden.narrow(1, sequence - 1, 1)?)?;
        let logits = self.output.forward(&hidden)?.squeeze(1)?;
        let logits = ((logits / self.final_logit_softcap)?.tanh()? * self.final_logit_softcap)?;
        logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()
    }
}

/// `"Gemma 4"` plus the variant: the third `-`-separated segment of a
/// `Gemma-4-…` basename (`"Gemma-4-E4B-It"` gives `"Gemma 4 E4B"`), else the
/// size label, else nothing. Always starts with `"Gemma 4"`.
fn model_name(metadata: &HashMap<String, gguf_file::Value>) -> String {
    let text = |key: &str| {
        metadata
            .get(key)
            .and_then(|v| v.to_string().ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let variant = text("general.basename")
        .and_then(|b| {
            b.strip_prefix("Gemma-4-")
                .and_then(|rest| rest.split('-').next())
                .filter(|v| !v.is_empty())
                .map(str::to_string)
        })
        .or_else(|| text("general.size_label"));
    match variant {
        Some(v) => format!("Gemma 4 {v}"),
        None => "Gemma 4".to_string(),
    }
}

/// Feeds `Debug` output straight into the hasher: the token list alone is
/// megabytes of text, never worth materializing.
struct HashWriter<'a>(&'a mut Sha256);

impl std::fmt::Write for HashWriter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.update(s.as_bytes());
        Ok(())
    }
}

fn signature(content: &gguf_file::Content) -> [u8; 32] {
    use std::fmt::Write;
    let mut hasher = Sha256::new();
    let mut w = HashWriter(&mut hasher);
    // Writing into a hasher cannot fail.
    let _ = write!(w, "{:?}\0", content.magic);
    let mut keys: Vec<&String> = content.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let _ = write!(w, "{key}\0{:?}\0", content.metadata[key]);
    }
    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort();
    for name in names {
        let info = &content.tensor_infos[name];
        let _ = write!(
            w,
            "{name}\0{:?}\0{:?}\0",
            info.shape.dims(),
            info.ggml_dtype
        );
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    out
}

fn value_norm(value: &Tensor, epsilon: f64) -> Result<Tensor> {
    let original_dtype = value.dtype();
    let value_f32 = value.to_dtype(DType::F32)?;
    let variance = value_f32.sqr()?.mean_keepdim(D::Minus1)?;
    value_f32
        .broadcast_div(&(variance + epsilon)?.sqrt()?)?
        .to_dtype(original_dtype)
}

fn attention_mask(
    sequence: usize,
    offset: usize,
    key_positions: &[usize],
    sliding_window: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mask = (0..sequence)
        .flat_map(|query_index| {
            let query_position = offset + query_index;
            key_positions.iter().map(move |&key_position| {
                // A sliding layer sees its own position and the `window - 1` before it, as in
                // the reference (`kv_idx > q_idx - sliding_window`).
                let outside_window =
                    sliding_window.is_some_and(|window| key_position + window <= query_position);
                if key_position > query_position || outside_window {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&mask, (1, 1, sequence, key_positions.len()), device)?.to_dtype(dtype)
}

struct GemmaRmsNorm {
    weight: Tensor,
    epsilon: f64,
}

impl GemmaRmsNorm {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
        epsilon: f64,
    ) -> Result<Self> {
        Ok(Self {
            weight: content.tensor(reader, name, device)?.dequantize(device)?,
            epsilon,
        })
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let original_dtype = hidden.dtype();
        let hidden_f32 = hidden.to_dtype(DType::F32)?;
        let variance = hidden_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized = hidden_f32.broadcast_div(&(variance + self.epsilon)?.sqrt()?)?;
        normalized
            .to_dtype(original_dtype)?
            .broadcast_mul(&self.weight)
    }
}

fn metadata<'a>(content: &'a gguf_file::Content, name: &str) -> Result<&'a gguf_file::Value> {
    content
        .metadata
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("cannot find {name} in metadata")))
}

fn usize_metadata(content: &gguf_file::Content, name: &str) -> Result<usize> {
    Ok(metadata(content, name)?.to_u32()? as usize)
}

fn per_layer_usize_metadata(
    content: &gguf_file::Content,
    name: &str,
    block_count: usize,
) -> Result<Vec<usize>> {
    let value = metadata(content, name)?;
    let values = match value {
        gguf_file::Value::U32(value) => vec![*value as usize; block_count],
        gguf_file::Value::I32(value) if *value >= 0 => {
            vec![value.unsigned_abs() as usize; block_count]
        }
        gguf_file::Value::Array(values) => values
            .iter()
            .map(|value| match value {
                gguf_file::Value::U32(value) => Ok(*value as usize),
                gguf_file::Value::I32(value) if *value >= 0 => Ok(value.unsigned_abs() as usize),
                value => {
                    candle_core::bail!("{name} contains a non-negative 32-bit integer: {value:?}")
                }
            })
            .collect::<Result<Vec<_>>>()?,
        value => {
            candle_core::bail!("{name} is not a scalar or array of 32-bit integers: {value:?}")
        }
    };
    if values.len() != block_count {
        candle_core::bail!(
            "{name} has {} entries for {block_count} blocks",
            values.len()
        )
    }
    Ok(values)
}

fn qmatmul<R: Read + Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
) -> Result<QMatMul> {
    QMatMul::from_qtensor(content.tensor(reader, name, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgguf::{TinyConfig, write_tiny};
    use candle_core::quantized::QTensor;

    fn values(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (f32::from(u8::try_from(i * 37 % 101).unwrap()) - 50.0) / 25.0)
            .collect()
    }

    fn rows_match(dtype: GgmlDType) {
        let (rows, cols) = (5, 64);
        let t = Tensor::from_vec(values(rows * cols), (rows, cols), &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, dtype).unwrap();
        let full = q.dequantize(&Device::Cpu).unwrap();
        let emb = QEmbedding::from_qtensor(&q).unwrap();
        let ids = [3u32, 0, 3, 4];
        let got = emb.forward(&ids, &Device::Cpu).unwrap();
        assert_eq!(got.dims(), &[4, cols]);
        for (row, &id) in ids.iter().enumerate() {
            let want = full.get(id as usize).unwrap().to_vec1::<f32>().unwrap();
            let have = got.get(row).unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(have, want, "{dtype:?} row {id}");
        }
    }

    #[test]
    fn qembedding_rows_equal_the_dequantized_rows() {
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
        ] {
            rows_match(dtype);
        }
    }

    #[test]
    fn qembedding_rows_equal_the_dequantized_rows_for_k_quants() {
        let (rows, cols) = (3, 256);
        let t = Tensor::from_vec(values(rows * cols), (rows, cols), &Device::Cpu).unwrap();
        for dtype in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
            let q = QTensor::quantize(&t, dtype).unwrap();
            let full = q.dequantize(&Device::Cpu).unwrap();
            let got = QEmbedding::from_qtensor(&q)
                .unwrap()
                .forward(&[2, 1], &Device::Cpu)
                .unwrap();
            assert_eq!(
                got.get(0).unwrap().to_vec1::<f32>().unwrap(),
                full.get(2).unwrap().to_vec1::<f32>().unwrap(),
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn qembedding_refuses_an_unsupported_type_and_a_bad_id() {
        let t = Tensor::from_vec(values(2 * 32), (2, 32), &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::Q4_1).unwrap();
        assert!(QEmbedding::from_qtensor(&q).is_err());
        let q = QTensor::quantize(&t, GgmlDType::F32).unwrap();
        let emb = QEmbedding::from_qtensor(&q).unwrap();
        assert!(emb.forward(&[2], &Device::Cpu).is_err());
    }

    fn md(pairs: &[(&str, &str)]) -> HashMap<String, gguf_file::Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), gguf_file::Value::String((*v).to_string())))
            .collect()
    }

    #[test]
    fn the_name_comes_from_the_basename_then_the_size_label() {
        let real = md(&[
            ("general.basename", "Gemma-4-E4B-It"),
            ("general.size_label", "7.5B"),
        ]);
        assert_eq!(model_name(&real), "Gemma 4 E4B");
        assert_eq!(
            model_name(&md(&[
                ("general.basename", "Other-Model"),
                ("general.size_label", "7.5B")
            ])),
            "Gemma 4 7.5B"
        );
        assert_eq!(
            model_name(&md(&[("general.size_label", "31B")])),
            "Gemma 4 31B"
        );
        assert_eq!(
            model_name(&md(&[("general.basename", "Gemma-4")])),
            "Gemma 4"
        );
        assert_eq!(model_name(&md(&[])), "Gemma 4");
    }

    fn tiny(name: &str, cfg: &TinyConfig) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "gemma-model-{name}-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        write_tiny(&path, cfg).unwrap();
        path
    }

    #[test]
    fn tiny_model_opens_with_split_head_dims_and_kv_owners() {
        let cfg = TinyConfig::default();
        let m = Model::open(&tiny("open", &cfg), &Device::Cpu).unwrap();
        assert!(m.name.starts_with("Gemma 4"), "{}", m.name);
        assert_eq!(m.context_length(), cfg.context);
        // Two KV-owning layers: sliding (swa_head_dim), then global.
        assert_eq!(m.kv_shape(), [(1, cfg.swa_head_dim), (1, cfg.head_dim)]);
        assert_eq!(
            m.kv_bytes_per_token(),
            2 * 4 * (cfg.swa_head_dim + cfg.head_dim)
        );
        assert_eq!(
            m.tokenizer.vocab_size(),
            crate::testgguf::tiny_vocab().0.len()
        );
    }

    #[test]
    fn ctx_cap_bounds_the_context() {
        let m =
            Model::open_with_ctx(&tiny("cap", &TinyConfig::default()), &Device::Cpu, 64).unwrap();
        assert_eq!(m.context_length(), 64);
    }

    #[test]
    fn forward_returns_softcapped_finite_logits() {
        let cfg = TinyConfig::default();
        let m = Model::open(&tiny("fwd", &cfg), &Device::Cpu).unwrap();
        let mut cache = KvCache::new(m.kv_shape().len());
        let logits = m.forward(&[12, 300, 301], 0, &mut cache).unwrap();
        assert_eq!(logits.len(), m.tokenizer.vocab_size());
        assert!(logits.iter().all(|l| l.is_finite() && l.abs() <= 30.0));
        assert_eq!(cache.len(), 3);
        assert!(m.forward(&[], 3, &mut cache).is_err());
    }

    #[test]
    fn signature_tracks_the_shape_not_the_weights() {
        let a = Model::open(&tiny("sig-a", &TinyConfig::default()), &Device::Cpu).unwrap();
        let b = Model::open(
            &tiny(
                "sig-b",
                &TinyConfig {
                    seed: 99,
                    ..TinyConfig::default()
                },
            ),
            &Device::Cpu,
        )
        .unwrap();
        let c = Model::open(
            &tiny(
                "sig-c",
                &TinyConfig {
                    sliding_window: 8,
                    ..TinyConfig::default()
                },
            ),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(a.signature(), b.signature());
        assert_ne!(a.signature(), c.signature());
    }

    fn open_with(
        name: &str,
        edit: impl Fn(&mut Vec<(String, gguf_file::Value)>),
    ) -> crate::Result<Arc<Model>> {
        let cfg = TinyConfig::default();
        let mut md = crate::testgguf::tiny_metadata(&cfg).unwrap();
        edit(&mut md);
        let path = std::env::temp_dir().join(format!(
            "gemma-model-bad-{name}-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        crate::testgguf::write_gguf(&path, &md, &crate::testgguf::tiny_tensors(&cfg).unwrap())
            .unwrap();
        Model::open(&path, &Device::Cpu)
    }

    fn set(md: &mut [(String, gguf_file::Value)], key: &str, value: gguf_file::Value) {
        md.iter_mut().find(|(k, _)| k == key).unwrap().1 = value;
    }

    #[test]
    fn zero_kv_heads_is_an_error_not_a_panic() {
        let e = open_with("kv0", |md| {
            set(
                md,
                "gemma4.attention.head_count_kv",
                gguf_file::Value::U32(0),
            );
        })
        .unwrap_err();
        assert!(e.0.contains("KV heads"), "{e}");
    }

    #[test]
    fn heads_not_a_multiple_of_kv_heads_is_an_error() {
        let e = open_with("grouping", |md| {
            set(md, "gemma4.attention.head_count", gguf_file::Value::U32(3));
            set(
                md,
                "gemma4.attention.head_count_kv",
                gguf_file::Value::U32(2),
            );
        })
        .unwrap_err();
        assert!(e.0.contains("not a multiple"), "{e}");
    }

    #[test]
    fn a_zero_window_or_odd_head_dim_is_an_error() {
        let e = open_with("window0", |md| {
            set(
                md,
                "gemma4.attention.sliding_window",
                gguf_file::Value::U32(0),
            );
        })
        .unwrap_err();
        assert!(e.0.contains("sliding window"), "{e}");
        let e = open_with("odd", |md| {
            set(
                md,
                "gemma4.attention.key_length_swa",
                gguf_file::Value::U32(7),
            );
        })
        .unwrap_err();
        assert!(e.0.contains("key_length_swa"), "{e}");
    }

    #[test]
    fn qembedding_larger_than_the_file_is_an_error() {
        let path = tiny("short", &TinyConfig::default());
        let mut f = std::fs::File::open(&path).unwrap();
        let mut ct = gguf_file::Content::read(&mut f).unwrap();
        let info = ct.tensor_infos.get_mut("token_embd.weight").unwrap();
        info.shape = candle_core::Shape::from((1_000_000, 32));
        let e = QEmbedding::read(&ct, &mut f, "token_embd.weight").unwrap_err();
        assert!(e.to_string().contains("past the end"), "{e}");
        let info = ct.tensor_infos.get_mut("token_embd.weight").unwrap();
        info.shape = candle_core::Shape::from((usize::MAX / 2, 32));
        let e = QEmbedding::read(&ct, &mut f, "token_embd.weight").unwrap_err();
        assert!(e.to_string().contains("overflows"), "{e}");
    }

    #[test]
    fn mixture_of_experts_files_are_rejected() {
        let content = gguf_file::Content {
            magic: gguf_file::VersionedMagic::GgufV3,
            metadata: [
                (
                    "general.architecture",
                    gguf_file::Value::String("gemma4".into()),
                ),
                ("gemma4.expert_count", gguf_file::Value::U32(128)),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
            tensor_infos: HashMap::default(),
            tensor_data_offset: 0,
        };
        let error = Config::from_gguf(&content).unwrap_err().to_string();
        assert!(
            error.contains("gemma4 MoE models are not supported yet"),
            "{error}"
        );
    }

    fn visible(mask: &Tensor, query: usize) -> Vec<usize> {
        let row = mask
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .get(query)
            .unwrap();
        row.to_vec1::<f32>()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == 0.0)
            .map(|(key, _)| key)
            .collect()
    }

    /// A window of 3 lets each query see itself and the two positions before it, no more.
    #[test]
    fn sliding_mask_keeps_the_current_and_window_minus_one_positions() {
        let mask =
            attention_mask(5, 0, &[0, 1, 2, 3, 4], Some(3), &Device::Cpu, DType::F32).unwrap();
        assert_eq!(visible(&mask, 0), [0]);
        assert_eq!(visible(&mask, 2), [0, 1, 2]);
        assert_eq!(visible(&mask, 4), [2, 3, 4]);
    }

    /// A single decode query on a sliding layer still loses keys outside the window.
    #[test]
    fn sliding_mask_applies_to_a_decode_step() {
        let mask = attention_mask(
            1,
            6,
            &(0..7).collect::<Vec<_>>(),
            Some(3),
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        assert_eq!(visible(&mask, 0), [4, 5, 6]);
    }
}
