//! A GGUF file mapped into memory, and the NVFP4 weights it holds.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::sync::Mutex;

use memmap2::Mmap;

use crate::gguf::{GGML_F32, GGML_NVFP4, Header};
use crate::nvfp4::tensor::NvWeight;
use crate::{Result, err};

/// The mapping every weight borrows from.
#[derive(Debug)]
pub struct MappedFile {
    /// On macOS, one no-copy Metal buffer over the whole mapping, built on
    /// first use: outer `None` until tried, `Some(None)` when Metal refused.
    /// Declared before `map` so it is dropped first; the buffer itself also
    /// owns a clone of `map`, so the mapping outlives any GPU use of it.
    #[cfg(target_os = "macos")]
    #[allow(clippy::option_option)] // outer None = untried, Some(None) = refused
    pub(crate) metal: Mutex<Option<Option<Arc<candle_metal_kernels::metal::Buffer>>>>,
    map: Arc<Mmap>,
}

impl MappedFile {
    fn new(map: Mmap) -> Self {
        Self {
            #[cfg(target_os = "macos")]
            metal: Mutex::new(None),
            map: Arc::new(map),
        }
    }

    /// The whole file.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }

    /// The shared mapping, for a no-copy GPU buffer to keep alive.
    #[cfg(target_os = "macos")]
    pub(crate) fn map(&self) -> &Arc<Mmap> {
        &self.map
    }
}

/// A parsed, mapped GGUF file.
#[derive(Debug)]
pub struct GgufFile {
    pub header: Header,
    file: Arc<MappedFile>,
}

impl GgufFile {
    /// Maps the file at `path` and parses its header.
    ///
    /// # Errors
    /// The file cannot be opened or mapped, or its header does not parse.
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
        // SAFETY: plank never writes a model file in place; installs replace
        // it by rename, which leaves this mapping on the old inode.
        let map = unsafe { Mmap::map(&f) }?;
        Self::from_map(map)
    }

    /// Copies `bytes` into an anonymous mapping and parses it. For tests,
    /// and page-aligned like a real file mapping.
    ///
    /// # Errors
    /// The mapping fails or the header does not parse.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut m = memmap2::MmapMut::map_anon(bytes.len().max(1))?;
        m[..bytes.len()].copy_from_slice(bytes);
        Self::from_map(m.make_read_only()?)
    }

    fn from_map(map: Mmap) -> Result<Self> {
        let header = Header::parse(&map)?;
        Ok(Self {
            header,
            file: Arc::new(MappedFile::new(map)),
        })
    }

    /// The mapping, for code that binds it to a device.
    #[must_use]
    pub fn mapped_file(&self) -> &Arc<MappedFile> {
        &self.file
    }

    /// The data bytes of tensor `name`.
    ///
    /// # Errors
    /// No such tensor, an unreadable type, or data past the end of the file.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.header.tensor(name)?;
        let len = t.byte_len()?;
        let bytes = self.file.bytes();
        let (start, end) = span(t.offset, len, bytes.len())
            .ok_or_else(|| err(format!("tensor {name} points outside the file")))?;
        Ok(&bytes[start..end])
    }

    /// An F32 tensor's values.
    ///
    /// # Errors
    /// As [`Self::tensor_bytes`], or the tensor is not F32.
    pub fn f32_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.header.tensor(name)?;
        if t.ggml_type != GGML_F32 {
            return Err(err(format!(
                "tensor {name} is not F32 (ggml type {})",
                t.ggml_type
            )));
        }
        let (chunks, _) = self.tensor_bytes(name)?.as_chunks::<4>();
        Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    /// The NVFP4 weight `<prefix>.weight` with its `<prefix>.scale`: one
    /// scale for a 2-D weight, one per expert for a 3-D one.
    ///
    /// # Errors
    /// Either tensor missing, the weight not NVFP4, a shape that is neither
    /// 2-D nor 3-D, or a scale count that does not match.
    pub fn nv_weight(&self, prefix: &str) -> Result<NvWeight> {
        let name = format!("{prefix}.weight");
        let t = self.header.tensor(&name)?;
        if t.ggml_type != GGML_NVFP4 {
            return Err(err(format!(
                "tensor {name} is not NVFP4 (ggml type {})",
                t.ggml_type
            )));
        }
        let dim =
            |d: u64| usize::try_from(d).map_err(|_| err(format!("tensor {name} is too large")));
        let (cols, rows, experts) = match t.dims.as_slice() {
            [c, r] => (dim(*c)?, dim(*r)?, 1),
            [c, r, e] => (dim(*c)?, dim(*r)?, dim(*e)?),
            d => {
                return Err(err(format!(
                    "tensor {name} has shape {d:?}, want 2-D or 3-D"
                )));
            }
        };
        let scales = self.f32_tensor(&format!("{prefix}.scale"))?;
        let (start, end) = span(t.offset, t.byte_len()?, self.file.bytes().len())
            .ok_or_else(|| err(format!("tensor {name} points outside the file")))?;
        NvWeight::mapped(
            Arc::clone(&self.file),
            start,
            end - start,
            cols,
            rows,
            experts,
            scales,
        )
    }
}

/// `offset..offset+len` as indices into a file of `file_len` bytes.
fn span(offset: u64, len: u64, file_len: usize) -> Option<(usize, usize)> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(len).ok()?)?;
    (end <= file_len).then_some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GGML_BF16, Value};
    use crate::nvfp4::block::quantize_row;
    use crate::testgguf::Builder;

    /// Two experts of 2 rows x 64 cols, plus a dense 3 x 128 weight.
    fn sample() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let exps: Vec<u8> = (0..4u16)
            .flat_map(|r| {
                quantize_row(
                    &(0..64u16)
                        .map(|i| f32::from((i + r) % 9) - 4.0)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let dense: Vec<u8> = (0..3u16)
            .flat_map(|r| {
                quantize_row(
                    &(0..128u16)
                        .map(|i| f32::from((i * r) % 5))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let scales: Vec<u8> = [0.5f32, 2.0].iter().flat_map(|s| s.to_le_bytes()).collect();
        let file = Builder::new()
            .kv("general.architecture", &Value::Str("flint".into()))
            .tensor("blk.2.ffn_up_exps.weight", &[64, 2, 2], GGML_NVFP4, &exps)
            .tensor("blk.2.ffn_up_exps.scale", &[2], GGML_F32, &scales)
            .tensor("blk.0.attn_q.weight", &[128, 3], GGML_NVFP4, &dense)
            .tensor("blk.0.attn_q.scale", &[1], GGML_F32, &3.0f32.to_le_bytes())
            .tensor("token_embd.weight", &[64, 1], GGML_BF16, &[0u8; 128])
            .bytes();
        (file, exps, dense)
    }

    #[test]
    fn nv_weight_reads_shape_bytes_and_per_expert_scales() {
        let (bytes, exps, dense) = sample();
        let f = GgufFile::from_bytes(&bytes).unwrap();
        let up = f.nv_weight("blk.2.ffn_up_exps").unwrap();
        assert_eq!((up.cols(), up.rows(), up.experts()), (64, 2, 2));
        assert_eq!(up.scales(), &[0.5, 2.0]);
        assert_eq!(up.bytes(), exps.as_slice());
        assert_eq!(
            up.row(1, 0),
            &exps[2 * 36..3 * 36],
            "expert 1 starts after 2 rows"
        );
        let q = f.nv_weight("blk.0.attn_q").unwrap();
        assert_eq!((q.cols(), q.rows(), q.experts()), (128, 3, 1));
        assert_eq!(q.scale(0), 3.0);
        assert_eq!(q.bytes(), dense.as_slice());
        assert!(q.mapping().is_some(), "a file weight borrows the mapping");
    }

    #[test]
    fn dequant_applies_the_tensor_scale() {
        let (bytes, exps, _) = sample();
        let up = GgufFile::from_bytes(&bytes)
            .unwrap()
            .nv_weight("blk.2.ffn_up_exps")
            .unwrap();
        let mut raw = vec![0f32; 64];
        crate::nvfp4::block::dequant_row(&exps[2 * 36..3 * 36], &mut raw);
        let d = up.dequant(1);
        assert_eq!(d.len(), 2 * 64);
        for (a, b) in d[..64].iter().zip(&raw) {
            assert_eq!(*a, b * 2.0);
        }
    }

    #[test]
    fn mismatches_are_errors_not_panics() {
        let (bytes, _, _) = sample();
        let f = GgufFile::from_bytes(&bytes).unwrap();
        assert!(f.nv_weight("token_embd").unwrap_err().0.contains("NVFP4"));
        assert!(f.nv_weight("blk.9.nothing").is_err());
        assert!(
            NvWeight::owned(64, 2, 2, vec![0; 4 * 36], vec![1.0])
                .unwrap_err()
                .0
                .contains("scales")
        );
        assert!(NvWeight::owned(65, 1, 1, vec![0; 36], vec![1.0]).is_err());
        assert!(
            NvWeight::owned(64, 2, 1, vec![0; 36], vec![1.0])
                .unwrap_err()
                .0
                .contains("bytes")
        );
        let bad_scale = Builder::new()
            .tensor("w.weight", &[64, 1, 2], GGML_NVFP4, &[0u8; 72])
            .tensor("w.scale", &[1], GGML_F32, &1.0f32.to_le_bytes())
            .bytes();
        assert!(
            GgufFile::from_bytes(&bad_scale)
                .unwrap()
                .nv_weight("w")
                .is_err()
        );
        let mut cut = sample().0;
        cut.truncate(cut.len() - 200);
        let f = GgufFile::from_bytes(&cut).unwrap();
        assert!(
            f.tensor_bytes("token_embd.weight")
                .unwrap_err()
                .0
                .contains("outside")
        );
    }
}
