//! An in-memory GGUF v3 writer, so tests never depend on a real model file.

use crate::config::FlintConfig;
use crate::gguf::{GGML_BF16, GGML_F32, GGML_NVFP4, Value};
use crate::nvfp4::block::quantize_row;

/// Builds a GGUF file from metadata values and tensors.
#[derive(Debug, Default, Clone)]
pub struct Builder {
    kv: Vec<u8>,
    n_kv: u64,
    infos: Vec<u8>,
    n_tensors: u64,
    data: Vec<u8>,
}

// Fixture sizes are tiny; a length always fits a u64.
#[allow(clippy::cast_possible_truncation)]
impl Builder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one metadata value.
    #[must_use]
    pub fn kv(mut self, key: &str, v: &Value) -> Self {
        put_str(&mut self.kv, key);
        self.kv.extend_from_slice(&v.type_id().to_le_bytes());
        put_value(&mut self.kv, v);
        self.n_kv += 1;
        self
    }

    /// Appends a tensor, its data 32-byte aligned after the previous one's.
    #[must_use]
    pub fn tensor(mut self, name: &str, dims: &[u64], ggml_type: u32, data: &[u8]) -> Self {
        let rel = (self.data.len() as u64).div_ceil(32) * 32;
        self.data.resize(rel as usize, 0);
        self.data.extend_from_slice(data);
        put_str(&mut self.infos, name);
        self.infos
            .extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            self.infos.extend_from_slice(&d.to_le_bytes());
        }
        self.infos.extend_from_slice(&ggml_type.to_le_bytes());
        self.infos.extend_from_slice(&rel.to_le_bytes());
        self.n_tensors += 1;
        self
    }

    /// The complete file.
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&self.n_tensors.to_le_bytes());
        out.extend_from_slice(&self.n_kv.to_le_bytes());
        out.extend_from_slice(&self.kv);
        out.extend_from_slice(&self.infos);
        out.resize((out.len() as u64).div_ceil(32) as usize * 32, 0);
        out.extend_from_slice(&self.data);
        out
    }
}

#[allow(clippy::cast_possible_truncation)]
fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

#[allow(clippy::cast_possible_truncation)]
fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(u8::from(*x)),
        Value::Str(s) => put_str(out, s),
        Value::Array(items) => {
            let et = items.first().map_or(0, Value::type_id);
            out.extend_from_slice(&et.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                put_value(out, item);
            }
        }
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
    }
}

/// A seeded value source for fixtures, `SplitMix64` like the sampler's.
struct Values(u64);

// 24 random bits convert to f32 exactly.
#[allow(clippy::cast_precision_loss)]
impl Values {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// BF16 by truncation, which is how the loader widens it back.
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// A small but complete Flint GGUF with `c`'s shape and a `vocab`-token
/// embedding and head (no tokenizer metadata): the real file's tensor
/// names, types and layout, with seeded random weights, unit norms and
/// every sink logit set to `sink`.
///
/// # Panics
/// When a width is not a multiple of 64, which NVFP4 needs.
#[must_use]
pub fn tiny_flint(c: &FlintConfig, vocab: usize, sink: f32, seed: u64) -> Vec<u8> {
    tiny_flint_builder(c, vocab, sink, seed).bytes()
}

/// The real file's chat markers with their GGUF token types: the turn
/// markers are control (3), the tags the model writes are user-defined (4).
/// [`tiny_vocab`] gives the marker at index `i` the id `256 + i`.
pub const FLINT_MARKERS: [(&str, i32); 18] = [
    ("<|end_of_text|>", 3),
    ("<|system|>", 3),
    ("<|developer|>", 3),
    ("<|user|>", 3),
    ("<|assistant|>", 3),
    ("<|observation|>", 3),
    ("<think>", 4),
    ("</think>", 4),
    ("<tool_call>", 4),
    ("</tool_call>", 4),
    ("<arg_key>", 4),
    ("</arg_key>", 4),
    ("<arg_value>", 4),
    ("</arg_value>", 4),
    ("<tool_response>", 4),
    ("</tool_response>", 4),
    ("<tools>", 4),
    ("</tools>", 4),
];

/// The markers a reply ends at, as the real file's `eog_token_ids` lists
/// them.
#[cfg(feature = "candle")]
const EOG_MARKERS: [&str; 4] = [
    "<|end_of_text|>",
    "<|user|>",
    "<|observation|>",
    "<|developer|>",
];

/// Adds a byte-level `tokenizer.ggml.*` vocab to `b`: the 256 GPT-2 byte
/// tokens (ids `0..256`, in byte order), then `extra` (spelling, token
/// type) at `256 + i`, and no merges. The eos is `<|end_of_text|>` and the
/// eog ids are whichever of [`EOG_MARKERS`] `extra` holds; without
/// `<|end_of_text|>` the eos is id 0.
#[cfg(feature = "candle")]
#[must_use]
pub fn tiny_vocab(b: Builder, extra: &[(&str, i32)]) -> Builder {
    let mut tokens: Vec<Value> = crate::tokenizer::bytes_to_unicode()
        .iter()
        .map(|c| Value::Str(c.to_string()))
        .collect();
    let mut types = vec![Value::I32(1); 256];
    for (s, ty) in extra {
        tokens.push(Value::Str((*s).to_string()));
        types.push(Value::I32(*ty));
    }
    let id = |s: &str| {
        extra
            .iter()
            .position(|(t, _)| *t == s)
            .and_then(|i| u32::try_from(256 + i).ok())
    };
    let eos = id("<|end_of_text|>").unwrap_or(0);
    let mut eog: Vec<Value> = EOG_MARKERS
        .iter()
        .filter_map(|s| id(s))
        .map(Value::U32)
        .collect();
    if eog.is_empty() {
        eog.push(Value::U32(eos));
    }
    b.kv("tokenizer.ggml.model", &Value::Str("gpt2".into()))
        .kv("tokenizer.ggml.pre", &Value::Str("flint".into()))
        .kv("tokenizer.ggml.tokens", &Value::Array(tokens))
        .kv("tokenizer.ggml.token_type", &Value::Array(types))
        .kv("tokenizer.ggml.merges", &Value::Array(Vec::new()))
        .kv("tokenizer.ggml.eos_token_id", &Value::U32(eos))
        .kv("tokenizer.ggml.eog_token_ids", &Value::Array(eog))
}

/// [`tiny_flint`] with [`tiny_vocab`]'s tokenizer: a `256 + extra.len()`
/// token model that [`crate::tokenizer::FlintTokenizer::from_header`] reads.
///
/// # Panics
/// As [`tiny_flint`].
#[cfg(feature = "candle")]
#[must_use]
pub fn tiny_flint_with_vocab(
    c: &FlintConfig,
    extra: &[(&str, i32)],
    sink: f32,
    seed: u64,
) -> Vec<u8> {
    tiny_vocab(tiny_flint_builder(c, 256 + extra.len(), sink, seed), extra).bytes()
}

/// [`tiny_flint`] before it is serialised, so a test can add a tensor the
/// real file does not have.
///
/// # Panics
/// As [`tiny_flint`].
#[must_use]
// Fixture shapes are tiny; every u64/u32 conversion is exact. One tensor
// per statement mirrors the real file's layout, so length is not complexity.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub fn tiny_flint_builder(c: &FlintConfig, vocab: usize, sink: f32, seed: u64) -> Builder {
    let mut vals = Values(seed);
    let mut b = Builder::new();
    for (k, v) in c.metadata() {
        b = b.kv(&k, &v);
    }
    let u = |n: usize| n as u64;
    let nv =
        |b: Builder, name: &str, cols: usize, rows: usize, experts: usize, vals: &mut Values| {
            assert!(cols.is_multiple_of(64), "{name}: {cols} columns");
            let data: Vec<u8> = (0..rows * experts)
                .flat_map(|_| quantize_row(&vals.vec(cols, 0.3)))
                .collect();
            let dims: Vec<u64> = if experts == 1 {
                vec![u(cols), u(rows)]
            } else {
                vec![u(cols), u(rows), u(experts)]
            };
            let scales: Vec<f32> = (0..experts).map(|_| 0.75 + 0.25 * vals.next()).collect();
            b.tensor(&format!("{name}.weight"), &dims, GGML_NVFP4, &data)
                .tensor(
                    &format!("{name}.scale"),
                    &[u(experts)],
                    GGML_F32,
                    &f32_bytes(&scales),
                )
        };
    b = b.tensor(
        "token_embd.weight",
        &[u(c.hidden), u(vocab)],
        GGML_BF16,
        &bf16_bytes(&vals.vec(c.hidden * vocab, 0.5)),
    );
    b = b.tensor(
        "output_norm.weight",
        &[u(c.hidden)],
        GGML_F32,
        &f32_bytes(&vec![1.0; c.hidden]),
    );
    b = nv(b, "output", c.hidden, vocab, 1, &mut vals);
    let qd = c.heads * c.head_dim;
    let kd = c.kv_heads * c.head_dim;
    for l in 0..c.layers {
        let p = format!("blk.{l}");
        b = b.tensor(
            &format!("{p}.attn_norm.weight"),
            &[u(c.hidden)],
            GGML_F32,
            &f32_bytes(&vec![1.0; c.hidden]),
        );
        b = b.tensor(
            &format!("{p}.ffn_norm.weight"),
            &[u(c.hidden)],
            GGML_F32,
            &f32_bytes(&vec![1.0; c.hidden]),
        );
        b = b.tensor(
            &format!("{p}.attn_q_norm.weight"),
            &[u(c.head_dim)],
            GGML_F32,
            &f32_bytes(&vec![1.0; c.head_dim]),
        );
        b = b.tensor(
            &format!("{p}.attn_k_norm.weight"),
            &[u(c.head_dim)],
            GGML_F32,
            &f32_bytes(&vec![1.0; c.head_dim]),
        );
        b = b.tensor(
            &format!("{p}.attn_sinks.weight"),
            &[u(c.heads)],
            GGML_F32,
            &f32_bytes(&vec![sink; c.heads]),
        );
        b = nv(b, &format!("{p}.attn_q"), c.hidden, qd, 1, &mut vals);
        b = nv(b, &format!("{p}.attn_k"), c.hidden, kd, 1, &mut vals);
        b = nv(b, &format!("{p}.attn_v"), c.hidden, kd, 1, &mut vals);
        b = nv(b, &format!("{p}.attn_output"), qd, c.hidden, 1, &mut vals);
        if c.is_moe(l) {
            b = b.tensor(
                &format!("{p}.ffn_gate_inp.weight"),
                &[u(c.hidden), u(c.experts)],
                GGML_BF16,
                &bf16_bytes(&vals.vec(c.hidden * c.experts, 0.5)),
            );
            b = nv(
                b,
                &format!("{p}.ffn_gate_exps"),
                c.hidden,
                c.expert_ffn,
                c.experts,
                &mut vals,
            );
            b = nv(
                b,
                &format!("{p}.ffn_up_exps"),
                c.hidden,
                c.expert_ffn,
                c.experts,
                &mut vals,
            );
            b = nv(
                b,
                &format!("{p}.ffn_down_exps"),
                c.expert_ffn,
                c.hidden,
                c.experts,
                &mut vals,
            );
            b = nv(
                b,
                &format!("{p}.ffn_gate_shexp"),
                c.hidden,
                c.shared_ffn,
                1,
                &mut vals,
            );
            b = nv(
                b,
                &format!("{p}.ffn_up_shexp"),
                c.hidden,
                c.shared_ffn,
                1,
                &mut vals,
            );
            b = nv(
                b,
                &format!("{p}.ffn_down_shexp"),
                c.shared_ffn,
                c.hidden,
                1,
                &mut vals,
            );
        } else {
            b = nv(b, &format!("{p}.ffn_gate"), c.hidden, c.ffn, 1, &mut vals);
            b = nv(b, &format!("{p}.ffn_up"), c.hidden, c.ffn, 1, &mut vals);
            b = nv(b, &format!("{p}.ffn_down"), c.ffn, c.hidden, 1, &mut vals);
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_config;
    use crate::gguf::Header;

    /// The tiny vocab builds a tokenizer: the 256 bytes round-trip, each
    /// marker is a control id at `256 + its index`, and the turn markers
    /// that end a reply are the EOG ids.
    #[cfg(feature = "candle")]
    #[test]
    fn the_tiny_vocab_builds_a_flint_tokenizer() {
        use crate::tokenizer::FlintTokenizer;
        let c = test_config();
        let bytes = tiny_flint_with_vocab(&c, &FLINT_MARKERS, 0.5, 7);
        let h = Header::parse(&bytes).unwrap();
        assert_eq!(FlintConfig::from_header(&h).unwrap(), c);
        assert_eq!(
            h.tensor("token_embd.weight").unwrap().dims,
            vec![128, 256 + FLINT_MARKERS.len() as u64]
        );
        let t = FlintTokenizer::from_header(&h).unwrap();
        assert_eq!(t.vocab_size(), 256 + FLINT_MARKERS.len());
        for (i, (spelling, _)) in FLINT_MARKERS.iter().enumerate() {
            let id = 256 + u32::try_from(i).unwrap();
            assert_eq!(t.control_id(spelling), Some(id), "{spelling}");
            assert!(t.is_control(id));
        }
        let id = |s: &str| t.control_id(s).unwrap();
        assert_eq!(t.eos(), id("<|end_of_text|>"));
        for s in [
            "<|end_of_text|>",
            "<|user|>",
            "<|observation|>",
            "<|developer|>",
        ] {
            assert!(t.is_eog(id(s)), "{s}");
        }
        assert!(!t.is_eog(id("<|assistant|>")) && !t.is_eog(id("<think>")));
        let text = "héllo <tool_call> 🙂";
        let ids = t.encode_plain(text);
        assert!(ids.iter().all(|&i| i < 256), "{ids:?}");
        let back: Vec<u8> = ids.iter().flat_map(|&i| t.piece_bytes(i)).collect();
        assert_eq!(back, text.as_bytes());
    }

    #[test]
    fn tiny_flint_parses_back_to_its_config_and_layout() {
        let c = test_config();
        let bytes = tiny_flint(&c, 256, 0.5, 7);
        let h = Header::parse(&bytes).unwrap();
        assert_eq!(FlintConfig::from_header(&h).unwrap(), c);
        // 4 global tensors, 19 per dense layer, 26 per MoE layer.
        assert_eq!(h.tensors.len(), 4 + 19 + 3 * 26);
        let end = bytes.len() as u64;
        for t in h.tensors.values() {
            assert!(t.offset >= h.data_pos, "{}", t.name);
            assert!(t.offset + t.byte_len().unwrap() <= end, "{}", t.name);
        }
        let q = h.tensor("blk.1.attn_q.weight").unwrap();
        assert_eq!((q.dims.clone(), q.ggml_type), (vec![128, 128], GGML_NVFP4));
        let up = h.tensor("blk.1.ffn_up_exps.weight").unwrap();
        assert_eq!(up.dims, vec![128, 64, 8]);
        assert_eq!(h.tensor("blk.1.ffn_up_exps.scale").unwrap().dims, vec![8]);
        assert_eq!(h.tensor("token_embd.weight").unwrap().ggml_type, GGML_BF16);
    }
}
