// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Reading a GGUF file's architecture before any engine opens it, which is
//! what picks the backend.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// The `general.architecture` value of a Gemma 4 GGUF.
pub const GEMMA4_ARCH: &str = "gemma4";

/// Refuses to allocate for a declared length beyond this. The file may be
/// truncated or not a GGUF at all, and a bogus 64-bit length would otherwise
/// become an allocation.
const MAX_STRING_BYTES: u64 = 64 * 1024;

/// Metadata keys are walked in order; a file claiming more than this is not
/// one an engine would load either.
const MAX_KV_PAIRS: u64 = 1 << 20;

/// Reads `general.architecture` out of a GGUF file's metadata.
///
/// `None` when the file is not GGUF, is truncated, or has no such key.
#[must_use]
pub fn architecture(path: &Path) -> Option<String> {
    string_value(path, "general.architecture")
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
            // Type 8 is STRING; anything else is not a value to coerce.
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
