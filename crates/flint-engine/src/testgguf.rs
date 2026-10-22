//! An in-memory GGUF v3 writer, so tests never depend on a real model file.

use crate::gguf::Value;

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
