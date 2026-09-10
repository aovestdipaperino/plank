//! An in-memory GGUF v3 builder, so tests never depend on a real model file.

/// Builds a GGUF file with metadata keys and tensors.
#[derive(Debug, Default, Clone)]
pub struct Gguf {
    kv: Vec<u8>,
    count: u64,
    infos: Vec<u8>,
    n_tensors: u64,
    data: Vec<u8>,
}

#[allow(clippy::cast_possible_truncation)]
impl Gguf {
    /// Adds a string metadata value.
    #[must_use]
    pub fn str_val(mut self, key: &str, val: &str) -> Self {
        self.push_str(key);
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        self.push_str(val);
        self.count += 1;
        self
    }

    /// Adds a u32 metadata value (`general.alignment`, say).
    #[must_use]
    pub fn u32_val(mut self, key: &str, val: u32) -> Self {
        self.push_str(key);
        self.kv.extend_from_slice(&4u32.to_le_bytes());
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.count += 1;
        self
    }

    /// Adds a string-array metadata value.
    #[must_use]
    pub fn str_array(mut self, key: &str, vals: &[&str]) -> Self {
        self.push_str(key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        self.kv
            .extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.push_str(v);
        }
        self.count += 1;
        self
    }

    fn push_str(&mut self, s: &str) {
        self.kv.extend_from_slice(&(s.len() as u64).to_le_bytes());
        self.kv.extend_from_slice(s.as_bytes());
    }

    /// Appends a tensor whose data is `data`, placed right after the previous
    /// tensor's data, 32-byte aligned as real writers do.
    #[must_use]
    pub fn tensor(mut self, name: &str, dims: &[u64], ty: u32, data: &[u8]) -> Self {
        let rel = (self.data.len() as u64).div_ceil(32) * 32;
        self.data.resize(rel as usize, 0);
        self.data.extend_from_slice(data);
        self.infos
            .extend_from_slice(&(name.len() as u64).to_le_bytes());
        self.infos.extend_from_slice(name.as_bytes());
        self.infos
            .extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            self.infos.extend_from_slice(&d.to_le_bytes());
        }
        self.infos.extend_from_slice(&ty.to_le_bytes());
        self.infos.extend_from_slice(&rel.to_le_bytes());
        self.n_tensors += 1;
        self
    }

    /// The complete file bytes.
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&self.n_tensors.to_le_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        out.extend_from_slice(&self.kv);
        out.extend_from_slice(&self.infos);
        let data_pos = (out.len() as u64).div_ceil(32) * 32;
        out.resize(data_pos as usize, 0);
        out.extend_from_slice(&self.data);
        out
    }
}
