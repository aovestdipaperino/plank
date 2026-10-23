//! Flint's hyperparameters, read from (and writable back to) GGUF metadata.
//!
//! Every key the engine depends on is required and checked; anything this
//! crate cannot run (another architecture, softmax gating, K and V of
//! different widths, a per-layer array of the wrong length) is an error,
//! never a guess.

use crate::gguf::{Header, Value};
use crate::{Result, err};

const ARCH: &str = "flint";
/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`.
const GATING_SIGMOID: u32 = 2;

/// Sampling defaults the GGUF suggests (`general.sampling.*`). The engine
/// does not apply them itself; the caller decides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// `general.sampling.temp`: a finite temperature, 0 or more.
    pub temperature: Option<f32>,
    /// `general.sampling.top_p`: nucleus mass, in `0..=1`.
    pub top_p: Option<f32>,
    /// `general.sampling.top_k`: candidates kept before sampling. The shared
    /// sampler does not apply it yet.
    pub top_k: Option<u32>,
}

/// Flint's shape. Field names follow llama.cpp's hparams where they exist.
#[derive(Debug, Clone, PartialEq)]
pub struct FlintConfig {
    /// `block_count`.
    pub layers: usize,
    /// `embedding_length`.
    pub hidden: usize,
    /// `feed_forward_length`: the dense layers' FFN width.
    pub ffn: usize,
    /// `attention.head_count`.
    pub heads: usize,
    /// `attention.head_count_kv`.
    pub kv_heads: usize,
    /// `attention.key_length`, equal to `value_length`.
    pub head_dim: usize,
    /// `rope.freq_base`.
    pub rope_base: f32,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `context_length`.
    pub context_length: usize,
    /// `attention.sliding_window`: a key `p0` is masked for query `p1` when
    /// `p1 - p0 >= sliding_window`.
    pub sliding_window: usize,
    /// `attention.sliding_window_pattern`, one flag per layer.
    pub swa: Vec<bool>,
    /// `attention.rope_pattern`, one flag per layer.
    pub rope: Vec<bool>,
    /// `leading_dense_block_count`: layers `0..dense_layers` are dense.
    pub dense_layers: usize,
    /// `expert_count`.
    pub experts: usize,
    /// `expert_used_count`.
    pub experts_used: usize,
    /// `expert_feed_forward_length`.
    pub expert_ffn: usize,
    /// `expert_shared_feed_forward_length`: the shared expert's width.
    pub shared_ffn: usize,
    /// `expert_weights_norm`.
    pub weights_norm: bool,
    /// `expert_weights_scale` (1.0 when absent).
    pub weights_scale: f32,
    /// `general.sampling.*`.
    pub sampling: Sampling,
}

fn key(name: &str) -> String {
    format!("{ARCH}.{name}")
}

fn get<'a>(h: &'a Header, k: &str) -> Result<&'a Value> {
    h.get(k)
        .ok_or_else(|| err(format!("missing GGUF metadata key `{k}`")))
}

fn uint_of(v: &Value) -> Option<u64> {
    match v {
        Value::U8(n) => Some(u64::from(*n)),
        Value::U16(n) => Some(u64::from(*n)),
        Value::U32(n) => Some(u64::from(*n)),
        Value::U64(n) => Some(*n),
        Value::I8(n) => u64::try_from(*n).ok(),
        Value::I16(n) => u64::try_from(*n).ok(),
        Value::I32(n) => u64::try_from(*n).ok(),
        Value::I64(n) => u64::try_from(*n).ok(),
        _ => None,
    }
}

fn uint(h: &Header, k: &str) -> Result<usize> {
    uint_of(get(h, k)?)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| err(format!("`{k}` is not a non-negative integer")))
}

/// A required count or width that the engine divides or indexes by.
fn positive(h: &Header, k: &str) -> Result<usize> {
    match uint(h, k)? {
        0 => Err(err(format!("`{k}` is 0; it must be positive"))),
        n => Ok(n),
    }
}

fn float_of(v: &Value) -> Option<f32> {
    match v {
        Value::F32(x) => Some(*x),
        // GGUF writers store these as f32; an f64 is narrowed as llama.cpp does.
        #[allow(clippy::cast_possible_truncation)]
        Value::F64(x) => Some(*x as f32),
        _ => None,
    }
}

fn float(h: &Header, k: &str) -> Result<f32> {
    float_of(get(h, k)?).ok_or_else(|| err(format!("`{k}` is not a float")))
}

/// An optional sampling float: absent is `None`; present, it must be a
/// float within `range`.
fn sampling_float(
    h: &Header,
    k: &str,
    range: std::ops::RangeInclusive<f32>,
) -> Result<Option<f32>> {
    let Some(v) = h.get(k) else {
        return Ok(None);
    };
    match float_of(v) {
        Some(x) if range.contains(&x) => Ok(Some(x)),
        _ => Err(err(format!(
            "`{k}` is {v:?}, not a float in {}..={}",
            range.start(),
            range.end()
        ))),
    }
}

fn flags(h: &Header, k: &str, layers: usize) -> Result<Vec<bool>> {
    let a = get(h, k)?
        .as_array()
        .ok_or_else(|| err(format!("`{k}` is not an array")))?;
    if a.len() != layers {
        return Err(err(format!(
            "`{k}` has {} entries for {layers} layers",
            a.len()
        )));
    }
    a.iter()
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| err(format!("`{k}` holds a non-bool")))
        })
        .collect()
}

impl FlintConfig {
    /// Reads the config from a GGUF header.
    ///
    /// # Errors
    /// Another architecture, a missing or mistyped key, a zero window or
    /// dimension, rope scaling, an unsupported gating function, inconsistent head counts or widths, a
    /// per-layer array whose length is not `block_count`, or a sampling key
    /// that is present but mistyped or out of range.
    pub fn from_header(h: &Header) -> Result<Self> {
        let arch = get(h, "general.architecture")?
            .as_str()
            .ok_or_else(|| err("`general.architecture` is not a string"))?;
        if arch != ARCH {
            return Err(err(format!("architecture `{arch}` is not `{ARCH}`")));
        }
        let gating = uint(h, &key("expert_gating_func"))?;
        if gating != GATING_SIGMOID as usize {
            return Err(err(format!(
                "expert_gating_func {gating} is not sigmoid ({GATING_SIGMOID})"
            )));
        }
        if let Some(k) = h
            .metadata
            .keys()
            .find(|k| k.starts_with(&key("rope.scaling.")))
        {
            return Err(err(format!(
                "`{k}` is set; Flint runs without rope scaling"
            )));
        }
        let layers = positive(h, &key("block_count"))?;
        let heads = positive(h, &key("attention.head_count"))?;
        let kv_heads = uint(h, &key("attention.head_count_kv"))?;
        if kv_heads == 0 || heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(err(format!(
                "{heads} query heads cannot share {kv_heads} KV heads"
            )));
        }
        let head_dim = positive(h, &key("attention.key_length"))?;
        let value_dim = uint(h, &key("attention.value_length"))?;
        if value_dim != head_dim || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(err(format!(
                "attention.value_length {value_dim} must equal an even key_length {head_dim}"
            )));
        }
        let dense_layers = uint(h, &key("leading_dense_block_count"))?;
        let experts = positive(h, &key("expert_count"))?;
        let experts_used = uint(h, &key("expert_used_count"))?;
        if dense_layers > layers || experts_used == 0 || experts_used > experts {
            return Err(err(format!(
                "{dense_layers} dense of {layers} layers, {experts_used} of {experts} experts"
            )));
        }
        let weights_scale = match h.get(&key("expert_weights_scale")) {
            None => 1.0,
            Some(v) => float_of(v).ok_or_else(|| err("`expert_weights_scale` is not a float"))?,
        };
        Ok(Self {
            layers,
            hidden: positive(h, &key("embedding_length"))?,
            ffn: uint(h, &key("feed_forward_length"))?,
            heads,
            kv_heads,
            head_dim,
            rope_base: float(h, &key("rope.freq_base"))?,
            rms_eps: float(h, &key("attention.layer_norm_rms_epsilon"))?,
            context_length: positive(h, &key("context_length"))?,
            sliding_window: positive(h, &key("attention.sliding_window"))?,
            swa: flags(h, &key("attention.sliding_window_pattern"), layers)?,
            rope: flags(h, &key("attention.rope_pattern"), layers)?,
            dense_layers,
            experts,
            experts_used,
            expert_ffn: uint(h, &key("expert_feed_forward_length"))?,
            shared_ffn: uint(h, &key("expert_shared_feed_forward_length"))?,
            weights_norm: get(h, &key("expert_weights_norm"))?
                .as_bool()
                .ok_or_else(|| err("`expert_weights_norm` is not a bool"))?,
            weights_scale,
            sampling: Sampling {
                temperature: sampling_float(h, "general.sampling.temp", 0.0..=f32::MAX)?,
                top_p: sampling_float(h, "general.sampling.top_p", 0.0..=1.0)?,
                top_k: match h.get("general.sampling.top_k") {
                    None => None,
                    Some(v) => Some(uint_of(v).and_then(|n| u32::try_from(n).ok()).ok_or_else(
                        || {
                            err(format!(
                                "`general.sampling.top_k` is {v:?}, not a u32 integer"
                            ))
                        },
                    )?),
                },
            },
        })
    }

    /// The sliding window of `layer`, or `None` for a global layer.
    #[must_use]
    pub fn window(&self, layer: usize) -> Option<usize> {
        self.swa
            .get(layer)
            .copied()
            .unwrap_or(false)
            .then_some(self.sliding_window)
    }

    /// Bytes one position adds to the KV cache: K and V in f32 on every
    /// layer, so a caller can size the cache from the header alone.
    #[must_use]
    pub fn kv_bytes_per_token(&self) -> usize {
        self.layers * 2 * self.kv_heads * self.head_dim * 4
    }

    /// Whether `layer` routes through experts.
    #[must_use]
    pub fn is_moe(&self, layer: usize) -> bool {
        layer >= self.dense_layers
    }

    /// The config as GGUF metadata, the inverse of [`Self::from_header`].
    #[must_use]
    // Config widths are far below u32::MAX in any real or test model.
    #[allow(clippy::cast_possible_truncation)]
    pub fn metadata(&self) -> Vec<(String, Value)> {
        let u = |n: usize| Value::U32(n as u32);
        let bools = |v: &[bool]| Value::Array(v.iter().map(|b| Value::Bool(*b)).collect());
        let mut m = vec![
            ("general.architecture".to_string(), Value::Str(ARCH.into())),
            (key("block_count"), u(self.layers)),
            (key("context_length"), u(self.context_length)),
            (key("embedding_length"), u(self.hidden)),
            (key("feed_forward_length"), u(self.ffn)),
            (key("attention.head_count"), u(self.heads)),
            (key("attention.head_count_kv"), u(self.kv_heads)),
            (key("attention.key_length"), u(self.head_dim)),
            (key("attention.value_length"), u(self.head_dim)),
            (key("rope.freq_base"), Value::F32(self.rope_base)),
            (
                key("attention.layer_norm_rms_epsilon"),
                Value::F32(self.rms_eps),
            ),
            (key("expert_count"), u(self.experts)),
            (key("expert_used_count"), u(self.experts_used)),
            (key("expert_gating_func"), Value::U32(GATING_SIGMOID)),
            (key("expert_feed_forward_length"), u(self.expert_ffn)),
            (key("expert_shared_feed_forward_length"), u(self.shared_ffn)),
            (key("leading_dense_block_count"), u(self.dense_layers)),
            (key("expert_shared_count"), Value::U32(1)),
            (key("expert_weights_norm"), Value::Bool(self.weights_norm)),
            (key("expert_weights_scale"), Value::F32(self.weights_scale)),
            (key("attention.sliding_window"), u(self.sliding_window)),
            (key("attention.sliding_window_pattern"), bools(&self.swa)),
            (key("attention.rope_pattern"), bools(&self.rope)),
        ];
        if let Some(t) = self.sampling.temperature {
            m.push(("general.sampling.temp".into(), Value::F32(t)));
        }
        if let Some(p) = self.sampling.top_p {
            m.push(("general.sampling.top_p".into(), Value::F32(p)));
        }
        if let Some(k) = self.sampling.top_k {
            m.push(("general.sampling.top_k".into(), Value::U32(k)));
        }
        m
    }
}
/// A small, valid config shared by the crate's tests (the tiny GGUF of
/// Task 6 is built from it).
#[cfg(test)]
pub(crate) fn test_config() -> FlintConfig {
    FlintConfig {
        layers: 4,
        hidden: 128,
        ffn: 128,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        rope_base: 10000.0,
        rms_eps: 1e-6,
        context_length: 64,
        sliding_window: 4,
        swa: vec![false, true, true, false],
        rope: vec![false, true, true, false],
        dense_layers: 1,
        experts: 8,
        experts_used: 2,
        expert_ffn: 64,
        shared_ffn: 64,
        weights_norm: true,
        weights_scale: 1.0,
        sampling: Sampling {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgguf::Builder;
    use std::io::Read;

    fn header(meta: &[(String, Value)]) -> Header {
        let mut b = Builder::new();
        for (k, v) in meta {
            b = b.kv(k, v);
        }
        Header::parse(&b.bytes()).unwrap()
    }

    #[test]
    fn round_trips_through_gguf_metadata() {
        let c = test_config();
        assert_eq!(FlintConfig::from_header(&header(&c.metadata())).unwrap(), c);
    }

    #[test]
    fn windows_and_moe_layers_follow_the_arrays() {
        let c = test_config();
        assert_eq!(c.window(0), None);
        assert_eq!(c.window(1), Some(4));
        assert!(!c.is_moe(0));
        assert!(c.is_moe(1));
    }

    #[test]
    fn sampling_defaults_are_optional() {
        let mut c = test_config();
        c.sampling = Sampling {
            temperature: None,
            top_p: None,
            top_k: None,
        };
        assert_eq!(FlintConfig::from_header(&header(&c.metadata())).unwrap(), c);
    }

    fn without(meta: Vec<(String, Value)>, key: &str) -> Vec<(String, Value)> {
        meta.into_iter().filter(|(k, _)| k != key).collect()
    }

    fn with(meta: Vec<(String, Value)>, key: &str, v: Value) -> Vec<(String, Value)> {
        let mut m = without(meta, key);
        m.push((key.to_string(), v));
        m
    }

    #[test]
    fn refuses_what_the_engine_cannot_run() {
        let m = test_config().metadata();
        let err =
            |meta: Vec<(String, Value)>| FlintConfig::from_header(&header(&meta)).unwrap_err().0;
        assert!(
            err(with(
                m.clone(),
                "general.architecture",
                Value::Str("llama".into())
            ))
            .contains("llama")
        );
        assert!(err(without(m.clone(), "flint.block_count")).contains("flint.block_count"));
        assert!(
            err(with(m.clone(), "flint.expert_gating_func", Value::U32(1))).contains("sigmoid")
        );
        assert!(
            err(with(
                m.clone(),
                "flint.attention.head_count_kv",
                Value::U32(3)
            ))
            .contains("heads")
        );
        assert!(
            err(with(
                m.clone(),
                "flint.attention.sliding_window_pattern",
                Value::Array(vec![Value::Bool(true)])
            ))
            .contains("sliding_window_pattern")
        );
        assert!(
            err(with(m, "flint.attention.value_length", Value::U32(64))).contains("value_length")
        );
    }

    #[test]
    fn refuses_a_zero_window_or_dimension() {
        let m = test_config().metadata();
        let err =
            |meta: Vec<(String, Value)>| FlintConfig::from_header(&header(&meta)).unwrap_err().0;
        for k in [
            "flint.attention.sliding_window",
            "flint.embedding_length",
            "flint.block_count",
            "flint.attention.head_count",
            "flint.expert_count",
            "flint.attention.key_length",
            "flint.context_length",
        ] {
            let e = err(with(m.clone(), k, Value::U32(0)));
            assert!(e.contains(k) && e.contains("positive"), "{k}: {e}");
        }
    }

    #[test]
    fn refuses_rope_scaling() {
        let m = test_config().metadata();
        let err =
            |meta: Vec<(String, Value)>| FlintConfig::from_header(&header(&meta)).unwrap_err().0;
        let e = err(with(
            m.clone(),
            "flint.rope.scaling.type",
            Value::Str("yarn".into()),
        ));
        assert!(e.contains("flint.rope.scaling.type"), "{e}");
        let e = err(with(m, "flint.rope.scaling.factor", Value::F32(4.0)));
        assert!(e.contains("flint.rope.scaling.factor"), "{e}");
    }

    #[test]
    fn refuses_a_mistyped_or_out_of_range_sampling_key() {
        let m = test_config().metadata();
        let err =
            |meta: Vec<(String, Value)>| FlintConfig::from_header(&header(&meta)).unwrap_err().0;
        for (k, v) in [
            ("general.sampling.temp", Value::Str("hot".into())),
            ("general.sampling.temp", Value::F32(-1.0)),
            ("general.sampling.temp", Value::F32(f32::NAN)),
            ("general.sampling.top_p", Value::U32(1)),
            ("general.sampling.top_p", Value::F32(1.5)),
            ("general.sampling.top_k", Value::F32(20.0)),
            ("general.sampling.top_k", Value::I32(-1)),
            ("general.sampling.top_k", Value::U64(1 << 40)),
        ] {
            let e = err(with(m.clone(), k, v.clone()));
            assert!(e.contains(k), "{k} = {v:?}: {e}");
        }
    }

    #[test]
    fn kv_bytes_per_token_follows_the_shape() {
        let c = FlintConfig {
            layers: 49,
            kv_heads: 4,
            head_dim: 128,
            ..test_config()
        };
        assert_eq!(c.kv_bytes_per_token(), 200_704);
    }

    #[test]
    fn reads_the_real_files_values_when_present() {
        let Some(path) = std::env::var_os("PLANK_FLINT_GGUF") else {
            return;
        };
        // Only the header is needed: never load the whole multi-GB file.
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .unwrap()
            .take(32 << 20)
            .read_to_end(&mut bytes)
            .unwrap();
        let c = FlintConfig::from_header(&Header::parse(&bytes).unwrap()).unwrap();
        assert_eq!(
            (c.layers, c.hidden, c.heads, c.kv_heads, c.head_dim),
            (49, 2048, 32, 4, 128)
        );
        assert_eq!(
            (c.sliding_window, c.experts, c.experts_used, c.dense_layers),
            (513, 256, 6, 2)
        );
        assert_eq!((c.expert_ffn, c.shared_ffn, c.ffn), (512, 1024, 6144));
        assert_eq!(c.sampling.top_k, Some(20));
    }
}
