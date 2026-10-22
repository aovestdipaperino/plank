//! Just enough GGUF v3 to load Flint: every metadata value, the tensor
//! directory, and where the tensor data starts. Parsing works on a byte
//! slice, so it is pure; the mapping lives in `file`.

use std::collections::BTreeMap;

use crate::{Result, err};

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_BF16: u32 = 30;
pub const GGML_NVFP4: u32 = 40;

/// GGUF v3's default `general.alignment`.
const DEFAULT_ALIGNMENT: u64 = 32;
/// Arrays of arrays are legal; nothing Flint ships nests deeper than this.
const MAX_ARRAY_DEPTH: u32 = 2;
/// The most array elements reserved before any of them has parsed.
const MAX_ARRAY_RESERVE: usize = 65_536;

/// One metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// The GGUF type id this value is written with.
    #[must_use]
    pub fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::Str(_) => 8,
            Self::Array(_) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }

    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        if let Self::U32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        if let Self::F32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(v) = self {
            Some(v)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_array(&self) -> Option<&[Value]> {
        if let Self::Array(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

/// One entry of the tensor directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// `ne0` first: `[cols, rows]` or `[cols, rows, experts]`.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Absolute file offset of the first byte.
    pub offset: u64,
}

impl TensorInfo {
    /// Number of values, `None` on overflow.
    #[must_use]
    pub fn elements(&self) -> Option<u64> {
        self.dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d))
    }

    /// Bytes the tensor's data takes.
    ///
    /// # Errors
    /// A type this crate does not read, an NVFP4 tensor that is not whole
    /// blocks, or a size that overflows.
    pub fn byte_len(&self) -> Result<u64> {
        let n = self
            .elements()
            .ok_or_else(|| err(format!("tensor {} is too large", self.name)))?;
        let bytes = match self.ggml_type {
            GGML_F32 => n.checked_mul(4),
            GGML_F16 | GGML_BF16 => n.checked_mul(2),
            GGML_NVFP4 => {
                if !n.is_multiple_of(64) {
                    return Err(err(format!(
                        "NVFP4 tensor {} has {n} values, not whole 64-value blocks",
                        self.name
                    )));
                }
                (n / 64).checked_mul(36)
            }
            t => {
                return Err(err(format!(
                    "tensor {} has ggml type {t}, which flint-engine does not read",
                    self.name
                )));
            }
        };
        bytes.ok_or_else(|| err(format!("tensor {} is too large", self.name)))
    }
}

/// A parsed GGUF header.
#[derive(Debug, Clone, Default)]
pub struct Header {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    /// First byte of the tensor data: the header end rounded up to
    /// `general.alignment`.
    pub data_pos: u64,
}

impl Header {
    /// Parses the header at the start of `bytes`.
    ///
    /// # Errors
    /// Not GGUF, not v3, truncated, an unknown value type, a duplicate
    /// tensor, or an invalid `general.alignment`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut c = Cursor { b: bytes, pos: 0 };
        if c.take(4)? != b"GGUF" {
            return Err(err("not a GGUF file"));
        }
        let version = c.u32()?;
        if version != 3 {
            return Err(err(format!(
                "GGUF version {version} is not supported (only v3)"
            )));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let v = c.value(ty, 0)?;
            metadata.insert(key, v);
        }
        let mut infos = Vec::new();
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims == 0 || n_dims > 4 {
                return Err(err(format!("tensor {name} has {n_dims} dimensions")));
            }
            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ggml_type = c.u32()?;
            let rel = c.u64()?;
            infos.push((name, dims, ggml_type, rel));
        }
        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(Value::U32(a)) if *a > 0 => u64::from(*a),
            Some(_) => return Err(err("general.alignment must be a positive u32")),
        };
        let header_end = u64::try_from(c.pos).map_err(|_| err("GGUF header too large"))?;
        let data_pos = header_end.div_ceil(alignment) * alignment;
        let mut tensors = BTreeMap::new();
        for (name, dims, ggml_type, rel) in infos {
            let offset = data_pos
                .checked_add(rel)
                .ok_or_else(|| err(format!("tensor {name} offset overflows")))?;
            let info = TensorInfo {
                name: name.clone(),
                dims,
                ggml_type,
                offset,
            };
            if tensors.insert(name.clone(), info).is_some() {
                return Err(err(format!("duplicate tensor {name}")));
            }
        }
        Ok(Self {
            metadata,
            tensors,
            data_pos,
        })
    }

    /// A metadata value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// A tensor by name.
    ///
    /// # Errors
    /// No tensor has that name.
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| err(format!("the GGUF has no tensor {name}")))
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| err("truncated GGUF header"))?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    fn len(&mut self) -> Result<usize> {
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| err("GGUF length does not fit in memory"))?;
        if n > self.b.len() - self.pos {
            return Err(err("truncated GGUF header"));
        }
        Ok(n)
    }

    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| err("GGUF string is not UTF-8"))
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(i8::from_le_bytes(self.arr()?)),
            2 => Value::U16(u16::from_le_bytes(self.arr()?)),
            3 => Value::I16(i16::from_le_bytes(self.arr()?)),
            4 => Value::U32(u32::from_le_bytes(self.arr()?)),
            5 => Value::I32(i32::from_le_bytes(self.arr()?)),
            6 => Value::F32(f32::from_le_bytes(self.arr()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(err("GGUF arrays nest too deeply"));
                }
                let et = self.u32()?;
                // `len` only bounds the count by the bytes left in the file,
                // and each `Value` is far larger than its encoded byte, so a
                // corrupt count could still ask for hundreds of GB. Reserve
                // at most a modest prefix and let the vector grow as
                // elements actually parse.
                let n = self.len()?;
                let mut items = Vec::with_capacity(n.min(MAX_ARRAY_RESERVE));
                for _ in 0..n {
                    items.push(self.value(et, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(u64::from_le_bytes(self.arr()?)),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            t => return Err(err(format!("unknown GGUF value type {t}"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgguf::Builder;

    fn sample() -> Vec<u8> {
        Builder::new()
            .kv("general.architecture", &Value::Str("flint".into()))
            .kv("flint.block_count", &Value::U32(49))
            .kv("flint.rope.freq_base", &Value::F32(10000.0))
            .kv("flint.expert_weights_norm", &Value::Bool(true))
            .kv(
                "flint.attention.sliding_window_pattern",
                &Value::Array(vec![Value::Bool(false), Value::Bool(true)]),
            )
            .tensor("a.weight", &[64, 2], GGML_NVFP4, &[7u8; 72])
            .tensor("a.scale", &[1], GGML_F32, &2.0f32.to_le_bytes())
            .bytes()
    }

    #[test]
    fn parses_metadata_of_every_kind_used() {
        let h = Header::parse(&sample()).unwrap();
        assert_eq!(
            h.get("general.architecture").and_then(Value::as_str),
            Some("flint")
        );
        assert_eq!(h.get("flint.block_count").and_then(Value::as_u32), Some(49));
        assert_eq!(
            h.get("flint.rope.freq_base").and_then(Value::as_f32),
            Some(10000.0)
        );
        assert_eq!(
            h.get("flint.expert_weights_norm").and_then(Value::as_bool),
            Some(true)
        );
        let pat = h
            .get("flint.attention.sliding_window_pattern")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(pat, &[Value::Bool(false), Value::Bool(true)]);
    }

    #[test]
    fn tensor_offsets_are_absolute_and_sizes_follow_the_type() {
        let bytes = sample();
        let h = Header::parse(&bytes).unwrap();
        let a = h.tensor("a.weight").unwrap();
        assert_eq!(a.dims, vec![64, 2]);
        assert_eq!(a.byte_len().unwrap(), 72, "two NVFP4 blocks");
        assert_eq!(a.offset, h.data_pos);
        let s = h.tensor("a.scale").unwrap();
        assert_eq!(s.byte_len().unwrap(), 4);
        assert_eq!(s.offset, h.data_pos + 96, "72 bytes rounded up to 32");
        let at = usize::try_from(s.offset).unwrap();
        assert_eq!(&bytes[at..at + 4], &2.0f32.to_le_bytes());
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        assert!(Header::parse(b"GGUX").is_err());
        let mut v2 = sample();
        v2[4] = 2;
        assert!(Header::parse(&v2).unwrap_err().0.contains("version 2"));
        let full = sample();
        assert!(
            Header::parse(&full[..40])
                .unwrap_err()
                .0
                .contains("truncated")
        );
        let h = Header::parse(&full).unwrap();
        assert!(
            h.tensor("missing.weight")
                .unwrap_err()
                .0
                .contains("missing.weight")
        );
        let odd = TensorInfo {
            name: "x".into(),
            dims: vec![65],
            ggml_type: GGML_NVFP4,
            offset: 0,
        };
        assert!(odd.byte_len().is_err(), "NVFP4 needs whole 64-value blocks");
        let unknown = TensorInfo {
            name: "x".into(),
            dims: vec![4],
            ggml_type: 99,
            offset: 0,
        };
        assert!(unknown.byte_len().unwrap_err().0.contains("99"));
    }

    #[test]
    fn an_array_longer_than_the_file_is_refused_before_allocating() {
        let mut b = Builder::new()
            .kv("k", &Value::Array(vec![Value::U8(1)]))
            .bytes();
        // The count sits after magic(4) version(4) counts(16) key(8+1) type(4) elem type(4).
        let at = 4 + 4 + 16 + 8 + 1 + 4 + 4;
        b[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Header::parse(&b).is_err());
    }

    #[test]
    fn a_huge_array_count_inside_a_large_buffer_is_truncated_not_preallocated() {
        let b = Builder::new()
            .kv("k", &Value::Array(vec![Value::Str(String::new())]))
            .bytes();
        // Same layout as above; the count is followed by the first element.
        let at = 4 + 4 + 16 + 8 + 1 + 4 + 4;
        let mut b = b[..at + 8].to_vec();
        // 16 MiB of 0xFF: the count fits in what is left, but the first
        // string claims u64::MAX bytes, so parsing stops at once. Reserving
        // the full count up front would ask for 16 Mi values (512 MiB).
        let tail = 16 << 20;
        b[at..at + 8].copy_from_slice(&u64::try_from(tail).unwrap().to_le_bytes());
        b.resize(b.len() + tail, 0xFF);
        let (res, largest) = crate::testalloc::largest_allocation(|| Header::parse(&b));
        assert!(res.unwrap_err().0.contains("truncated"));
        assert!(
            largest <= MAX_ARRAY_RESERVE * std::mem::size_of::<Value>(),
            "parsing asked for {largest} bytes at once"
        );
    }
}
