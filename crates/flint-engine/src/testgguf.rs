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
