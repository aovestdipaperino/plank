//! Just enough GGUF v3 reading to learn where the tensors are and to pull a
//! few string metadata values. Only the header is read.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::Error;

/// One tensor's byte span in a GGUF file, in absolute file offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpan {
    /// The tensor's name, e.g. `blk.10.attn_output_b.weight`.
    pub name: String,
    /// Absolute offset of the first byte of the tensor's data.
    pub offset: u64,
    /// Bytes up to the next tensor's data, or to the end of the file for the
    /// last one. Derived from the offsets rather than from the quantization
    /// type, so no type table has to be kept in step with any engine.
    pub len: u64,
}

/// Where the tensor data lives in a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Layout {
    /// First byte of the tensor data region: the header end rounded up to
    /// `general.alignment` (32 when the key is absent).
    pub data_pos: u64,
    /// Every tensor, sorted by offset.
    pub tensors: Vec<TensorSpan>,
}

impl Layout {
    /// The tensor whose span contains file offset `pos`.
    #[must_use]
    pub fn tensor_at(&self, pos: u64) -> Option<&TensorSpan> {
        let i = self.tensors.partition_point(|t| t.offset <= pos);
        self.tensors
            .get(i.checked_sub(1)?)
            .filter(|t| pos < t.offset + t.len)
    }
}

/// GGUF v3's default `general.alignment`.
const DEFAULT_ALIGNMENT: u64 = 32;
/// Refuses a tensor count beyond this; the largest real model has a few
/// thousand.
const MAX_TENSORS: u64 = 1 << 20;
/// Metadata keys are walked in order; a file claiming more than this is not
/// one any engine would load either.
const MAX_KV_PAIRS: u64 = 1 << 20;
/// Refuses to allocate for a declared string length beyond this.
const MAX_STRING_BYTES: u64 = 64 * 1024;

fn bad(what: &str) -> Error {
    Error::Format(what.to_owned())
}

fn trunc() -> Error {
    bad("truncated GGUF header")
}

/// Reads the tensor layout of the GGUF file at `path`.
///
/// # Errors
/// Any read failure, or a malformed header (not GGUF, not v3, absurd counts,
/// a tensor pointing outside the file).
pub fn layout(path: &Path) -> Result<Layout, Error> {
    let file_len = std::fs::metadata(path)?.len();
    let mut r = BufReader::new(File::open(path)?);
    if &read_exact::<4>(&mut r).ok_or_else(|| bad("not a GGUF file"))? != b"GGUF" {
        return Err(bad("not a GGUF file"));
    }
    let version = u32::from_le_bytes(read_exact::<4>(&mut r).ok_or_else(trunc)?);
    if version != 3 {
        return Err(bad("only GGUF v3 is supported"));
    }
    let tensor_count = read_u64(&mut r).ok_or_else(trunc)?;
    let kv_count = read_u64(&mut r).ok_or_else(trunc)?;
    if kv_count > MAX_KV_PAIRS || tensor_count > MAX_TENSORS {
        return Err(bad("absurd GGUF header counts"));
    }
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..kv_count {
        let key = read_string(&mut r).ok_or_else(|| bad("bad GGUF metadata key"))?;
        let ty = u32::from_le_bytes(read_exact::<4>(&mut r).ok_or_else(trunc)?);
        if key == "general.alignment" && ty == 4 {
            let v = u32::from_le_bytes(read_exact::<4>(&mut r).ok_or_else(trunc)?);
            if v == 0 {
                return Err(bad("general.alignment is zero"));
            }
            alignment = u64::from(v);
            continue;
        }
        skip_value(&mut r, ty).ok_or_else(|| bad("bad GGUF metadata value"))?;
    }
    let mut infos = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));
    for _ in 0..tensor_count {
        let name = read_string(&mut r).ok_or_else(|| bad("bad GGUF tensor name"))?;
        let n_dims = u32::from_le_bytes(read_exact::<4>(&mut r).ok_or_else(trunc)?);
        if n_dims > 8 {
            return Err(bad("GGUF tensor has too many dimensions"));
        }
        for _ in 0..n_dims {
            read_u64(&mut r).ok_or_else(trunc)?;
        }
        let _ty = read_exact::<4>(&mut r).ok_or_else(trunc)?;
        let rel = read_u64(&mut r).ok_or_else(trunc)?;
        infos.push((name, rel));
    }
    let header_end = r.stream_position()?;
    let data_pos = header_end.div_ceil(alignment) * alignment;
    infos.sort_by_key(|(_, rel)| *rel);
    let mut tensors = Vec::with_capacity(infos.len());
    for (i, (name, rel)) in infos.iter().enumerate() {
        let offset = data_pos
            .checked_add(*rel)
            .ok_or_else(|| bad("GGUF tensor offset overflows"))?;
        let end = match infos.get(i + 1) {
            Some((_, next)) => data_pos + next,
            None => file_len,
        };
        if end < offset || offset > file_len {
            return Err(bad("GGUF tensor points outside the file"));
        }
        tensors.push(TensorSpan {
            name: name.clone(),
            offset,
            len: end - offset,
        });
    }
    Ok(Layout { data_pos, tensors })
}

/// Reads one string-typed metadata value out of a GGUF file.
///
/// `None` when the file is not GGUF, is truncated, has no such key, or the
/// key holds a non-string value.
#[must_use]
pub fn string_value(path: &Path, wanted: &str) -> Option<String> {
    let mut r = BufReader::new(File::open(path).ok()?);
    if &read_exact::<4>(&mut r)? != b"GGUF" {
        return None;
    }
    let _version = u32::from_le_bytes(read_exact::<4>(&mut r)?);
    let _tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if kv_count > MAX_KV_PAIRS {
        return None;
    }
    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let ty = u32::from_le_bytes(read_exact::<4>(&mut r)?);
        if key == wanted {
            // Type 8 is STRING.
            return (ty == 8).then(|| read_string(&mut r)).flatten();
        }
        skip_value(&mut r, ty)?;
    }
    None
}

fn read_exact<const N: usize>(r: &mut impl Read) -> Option<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn read_u64(r: &mut impl Read) -> Option<u64> {
    Some(u64::from_le_bytes(read_exact::<8>(r)?))
}

fn read_string(r: &mut impl Read) -> Option<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING_BYTES {
        return None;
    }
    let mut buf = vec![0u8; usize::try_from(len).ok()?];
    r.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

/// Width of a fixed-size GGUF scalar, or `None` for the variable-size types.
fn scalar_width(ty: u32) -> Option<i64> {
    match ty {
        0 | 1 | 7 => Some(1), // uint8, int8, bool
        2..=3 => Some(2),     // uint16, int16
        4..=6 => Some(4),     // uint32, int32, float32
        10..=12 => Some(8),   // uint64, int64, float64
        _ => None,
    }
}

/// Advances past one metadata value of type `ty`.
fn skip_value<R: Read + Seek>(r: &mut BufReader<R>, ty: u32) -> Option<()> {
    if let Some(width) = scalar_width(ty) {
        r.seek(SeekFrom::Current(width)).ok()?;
        return Some(());
    }
    match ty {
        8 => {
            let len = read_u64(r)?;
            r.seek(SeekFrom::Current(i64::try_from(len).ok()?)).ok()?;
            Some(())
        }
        9 => {
            let elem = u32::from_le_bytes(read_exact::<4>(r)?);
            let count = read_u64(r)?;
            if let Some(width) = scalar_width(elem) {
                let bytes = i64::try_from(count).ok()?.checked_mul(width)?;
                r.seek(SeekFrom::Current(bytes)).ok()?;
                return Some(());
            }
            // A string array is the only variable-width element the format
            // uses in practice; nested arrays are refused rather than walked.
            if elem != 8 {
                return None;
            }
            for _ in 0..count {
                let len = read_u64(r)?;
                r.seek(SeekFrom::Current(i64::try_from(len).ok()?)).ok()?;
            }
            Some(())
        }
        _ => None,
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Gguf;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("gguf-delta-{}-{name}", std::process::id()))
    }

    #[test]
    fn layout_reports_absolute_spans_in_offset_order() {
        let p = tmp("layout");
        std::fs::write(
            &p,
            Gguf::default()
                .str_val("general.architecture", "deepseek4")
                .tensor("b", &[4], 0, &[2u8; 40])
                .tensor("a", &[8], 0, &[1u8; 8])
                .bytes(),
        )
        .unwrap();
        let l = layout(&p).expect("layout");
        assert_eq!(l.data_pos % 32, 0);
        assert_eq!(l.tensors.len(), 2);
        assert_eq!(l.tensors[0].name, "b");
        assert_eq!(l.tensors[0].offset, l.data_pos);
        // 40 bytes of data padded to the next 32-byte boundary.
        assert_eq!(l.tensors[0].len, 64);
        assert_eq!(l.tensors[1].name, "a");
        assert_eq!(l.tensors[1].offset, l.data_pos + 64);
        assert_eq!(l.tensors[1].len, 8);
        assert_eq!(
            l.tensors[1].offset + 8,
            std::fs::metadata(&p).unwrap().len()
        );
        assert_eq!(
            l.tensor_at(l.data_pos + 63).map(|t| t.name.as_str()),
            Some("b")
        );
        assert_eq!(
            l.tensor_at(l.data_pos + 64).map(|t| t.name.as_str()),
            Some("a")
        );
        assert_eq!(l.tensor_at(l.data_pos + 72), None);
        assert_eq!(l.tensor_at(0), None);
        assert_eq!(
            string_value(&p, "general.architecture").as_deref(),
            Some("deepseek4")
        );
        assert_eq!(string_value(&p, "missing"), None);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn layout_of_a_header_only_file_is_empty() {
        let p = tmp("layout-empty");
        std::fs::write(&p, Gguf::default().str_val("k", "v").bytes()).unwrap();
        let l = layout(&p).expect("layout");
        assert!(l.tensors.is_empty());
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn layout_refuses_a_non_gguf_file() {
        let p = tmp("notgguf");
        std::fs::write(&p, b"nope").unwrap();
        assert!(matches!(layout(&p), Err(Error::Format(_))));
        let _ = std::fs::remove_file(p);
    }
}
