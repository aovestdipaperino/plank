//! Prefix-truncatable KV cache.
//!
//! Each layer stores every position's K and V in a preallocated
//! `[1, kv_heads, cap, head_dim]` tensor, written in place with
//! `Tensor::slice_set`. `truncate` only moves the logical length back — the
//! data beyond it is left in the buffer and is overwritten by the next
//! `append` — so a shared prefix can be reused across turns without
//! re-prefilling it, and the forward pass attends over `view()`.

use candle_core::{DType, Device, Tensor};

use crate::{Error, Result};

/// One transformer layer's K/V history.
///
/// `k` and `v` are `None` until the first `append`. Once allocated, they are
/// `[1, kv_heads, cap, head_dim]` tensors; only the first `len` positions
/// along dim 2 are live. `cap` is read back from the tensor's own shape, not
/// stored separately.
#[derive(Debug, Default)]
pub struct LayerKv {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
}

impl LayerKv {
    /// Positions currently live in this layer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no position is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Move the logical length back to `n` (a no-op if already `<= n`).
    ///
    /// The data beyond the new length is left untouched in the buffer; it is
    /// dead until the next `append` overwrites it in place.
    pub fn truncate(&mut self, n: usize) {
        self.len = self.len.min(n);
    }

    /// The live K/V: the growable buffers narrowed to `len` along the
    /// position axis, without copying.
    ///
    /// The tensors share the buffers' storage, so they are valid only until
    /// the next [`LayerKv::truncate`] followed by an [`LayerKv::append`]: that
    /// append writes in place over positions the view still covers. An
    /// append alone never changes them (it writes past `len`, or grows into
    /// a fresh buffer and leaves the old one alone). The forward pass
    /// consumes each view within the layer that took it, before anything
    /// truncates; [`KvCache::to_bytes`] reads it back at once.
    ///
    /// # Errors
    /// Propagates any candle error from narrowing.
    pub fn view(&self) -> Result<Option<(Tensor, Tensor)>> {
        let (Some(k), Some(v)) = (&self.k, &self.v) else {
            return Ok(None);
        };
        if self.len == 0 {
            return Ok(None);
        }
        Ok(Some((k.narrow(2, 0, self.len)?, v.narrow(2, 0, self.len)?)))
    }

    /// Append `n` new positions (read off dim 2 of `k`/`v`), growing the
    /// backing buffer (doubling capacity, copying the live prefix once) when
    /// it doesn't fit.
    ///
    /// # Errors
    /// Propagates any candle error from allocation, dtype conversion, or
    /// `slice_set`.
    ///
    /// # Panics
    /// Never in practice: every `unwrap()` below reads back `self.k`/`self.v`
    /// in a branch that just finished assigning them to `Some`.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let k = k.to_dtype(DType::F32)?.contiguous()?;
        let v = v.to_dtype(DType::F32)?.contiguous()?;
        let (_, kv_heads, n, head_dim) = k.dims4()?;
        if n == 0 {
            return Ok(());
        }
        let device = k.device().clone();

        match &self.k {
            None => {
                let cap = (self.len + n).max(256).next_power_of_two();
                self.k = Some(Tensor::zeros(
                    (1, kv_heads, cap, head_dim),
                    DType::F32,
                    &device,
                )?);
                self.v = Some(Tensor::zeros(
                    (1, kv_heads, cap, head_dim),
                    DType::F32,
                    &device,
                )?);
            }
            Some(existing) => {
                let (_, _, cap, _) = existing.dims4()?;
                if self.len + n > cap {
                    let mut new_cap = cap;
                    while self.len + n > new_cap {
                        new_cap *= 2;
                    }
                    let new_k =
                        Tensor::zeros((1, kv_heads, new_cap, head_dim), DType::F32, &device)?;
                    let new_v =
                        Tensor::zeros((1, kv_heads, new_cap, head_dim), DType::F32, &device)?;
                    if self.len > 0 {
                        let old_k = self
                            .k
                            .as_ref()
                            .unwrap()
                            .narrow(2, 0, self.len)?
                            .contiguous()?;
                        let old_v = self
                            .v
                            .as_ref()
                            .unwrap()
                            .narrow(2, 0, self.len)?
                            .contiguous()?;
                        new_k.slice_set(&old_k, 2, 0)?;
                        new_v.slice_set(&old_v, 2, 0)?;
                    }
                    self.k = Some(new_k);
                    self.v = Some(new_v);
                }
            }
        }

        self.k.as_ref().unwrap().slice_set(&k, 2, self.len)?;
        self.v.as_ref().unwrap().slice_set(&v, 2, self.len)?;
        self.len += n;
        Ok(())
    }
}

/// Per-layer KV history for a whole model.
#[derive(Debug)]
pub struct KvCache {
    pub layers: Vec<LayerKv>,
}

impl KvCache {
    /// `n_layers` empty layers.
    #[must_use]
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerKv::default()).collect(),
        }
    }

    /// The shared live length across all layers (0 when empty). All layers
    /// are always advanced together, so the first layer's length speaks for
    /// all of them.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, LayerKv::len)
    }

    /// True when no position is live in any layer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Truncate every layer to `n`.
    pub fn truncate(&mut self, n: usize) {
        for layer in &mut self.layers {
            layer.truncate(n);
        }
    }

    /// Serialize every layer's live K/V as `n_layers: u32`, then per layer
    /// `len: u32`, `kv_heads: u32`, `head_dim: u32`, K f32 LE bytes, V f32 LE
    /// bytes. An empty layer writes zeros for `len`/`kv_heads`/`head_dim` and
    /// no tensor bytes.
    ///
    /// # Errors
    /// Propagates any candle error from reading the live view back to host
    /// memory.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        // Views first, so the buffer is sized once: at about 114 KB per token
        // on E4B, growing it by doubling would copy gigabytes at 30k tokens.
        let views = self
            .layers
            .iter()
            .map(LayerKv::view)
            .collect::<Result<Vec<_>>>()?;
        let mut capacity = 4usize;
        for view in &views {
            capacity += 12;
            if let Some((k, v)) = view {
                capacity += (k.elem_count() + v.elem_count()) * 4;
            }
        }
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(
            &u32::try_from(self.layers.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for view in views {
            match view {
                None => {
                    out.extend_from_slice(&0u32.to_le_bytes());
                    out.extend_from_slice(&0u32.to_le_bytes());
                    out.extend_from_slice(&0u32.to_le_bytes());
                }
                Some((k, v)) => {
                    let (_, kv_heads, len, head_dim) = k.dims4()?;
                    out.extend_from_slice(&u32::try_from(len).unwrap_or(u32::MAX).to_le_bytes());
                    out.extend_from_slice(
                        &u32::try_from(kv_heads).unwrap_or(u32::MAX).to_le_bytes(),
                    );
                    out.extend_from_slice(
                        &u32::try_from(head_dim).unwrap_or(u32::MAX).to_le_bytes(),
                    );
                    extend_f32_le(&mut out, &k.flatten_all()?.to_vec1::<f32>()?);
                    extend_f32_le(&mut out, &v.flatten_all()?.to_vec1::<f32>()?);
                }
            }
        }
        Ok(out)
    }

    /// Parse `bytes` (the layout `to_bytes` writes) into a staging buffer,
    /// validating every count against `shape[i] = (kv_heads, head_dim)` for
    /// layer `i` and against the byte length before touching `self` at all;
    /// on any error `self` is unchanged.
    ///
    /// # Errors
    /// Returns an error (leaving `self` unchanged) when the layer count,
    /// a layer's recorded shape, or the byte length doesn't match `shape`,
    /// or the blob is truncated or has trailing bytes.
    pub fn restore(
        &mut self,
        bytes: &[u8],
        device: &Device,
        shape: &[(usize, usize)],
    ) -> Result<()> {
        let mut pos = 0usize;
        let n_layers = read_u32(bytes, &mut pos)? as usize;
        if n_layers != shape.len() {
            return Err(Error(format!(
                "kv snapshot: {n_layers} layers but {} shapes given",
                shape.len()
            )));
        }

        let mut staged = Vec::with_capacity(n_layers);
        for (i, &(exp_heads, exp_dim)) in shape.iter().enumerate() {
            let len = read_u32(bytes, &mut pos)? as usize;
            let kv_heads = read_u32(bytes, &mut pos)? as usize;
            let head_dim = read_u32(bytes, &mut pos)? as usize;
            if len > 0 && (kv_heads != exp_heads || head_dim != exp_dim) {
                return Err(Error(format!(
                    "kv snapshot: layer {i} shape ({kv_heads}, {head_dim}) does not match expected ({exp_heads}, {exp_dim})"
                )));
            }

            let mut layer = LayerKv::default();
            if len > 0 {
                let n_floats = len * kv_heads * head_dim;
                let byte_len = n_floats * 4;
                let k_bytes = read_bytes(bytes, &mut pos, byte_len)?;
                let kf: Vec<f32> = k_bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let v_bytes = read_bytes(bytes, &mut pos, byte_len)?;
                let vf: Vec<f32> = v_bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let kt = Tensor::from_vec(kf, (1, kv_heads, len, head_dim), device)?;
                let vt = Tensor::from_vec(vf, (1, kv_heads, len, head_dim), device)?;
                layer.append(&kt, &vt)?;
            }
            staged.push(layer);
        }

        if pos != bytes.len() {
            return Err(Error(format!(
                "kv snapshot: {} trailing byte(s)",
                bytes.len() - pos
            )));
        }

        self.layers = staged;
        Ok(())
    }
}

/// Appends `floats` as little-endian bytes in one resize, filling the new
/// tail in fixed four-byte chunks the compiler turns into a plain copy on a
/// little-endian host.
fn extend_f32_le(out: &mut Vec<u8>, floats: &[f32]) {
    let start = out.len();
    out.resize(start + floats.len() * 4, 0);
    for (dst, f) in out[start..].chunks_exact_mut(4).zip(floats) {
        dst.copy_from_slice(&f.to_le_bytes());
    }
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    let slice = read_bytes(bytes, pos, 4)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_bytes<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
    if *pos + n > bytes.len() {
        return Err(Error(format!(
            "kv snapshot: truncated (need {n} more byte(s) at offset {pos}, have {})",
            bytes.len()
        )));
    }
    let slice = &bytes[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

#[cfg(test)]
// Test data uses small loop indices as tensor values; f32 has 23 mantissa
// bits, far more than these tests ever need, so the precision-loss lint is a
// false positive here.
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn kv(start: f32, n: usize) -> Tensor {
        Tensor::arange(start, start + n as f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, n, 1))
            .unwrap()
    }

    #[test]
    fn append_grows_and_view_is_exact() {
        let mut l = LayerKv::default();
        for i in 0..40 {
            l.append(&kv(i as f32, 1), &kv(i as f32, 1)).unwrap();
        }
        let (k, _) = l.view().unwrap().unwrap();
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            (0..40).map(|i| i as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn truncate_then_append_overwrites_the_tail() {
        let mut l = LayerKv::default();
        l.append(&kv(0.0, 5), &kv(0.0, 5)).unwrap();
        l.truncate(2);
        l.append(&kv(100.0, 2), &kv(100.0, 2)).unwrap();
        let (k, _) = l.view().unwrap().unwrap();
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [0.0, 1.0, 100.0, 101.0]
        );
    }

    #[test]
    fn snapshot_round_trips_and_a_bad_blob_changes_nothing() {
        let mut c = KvCache::new(2);
        for l in &mut c.layers {
            l.append(&kv(0.0, 3), &kv(10.0, 3)).unwrap();
        }
        let bytes = c.to_bytes().unwrap();
        // 4 (layer count) + 2 * (12 header + 2 * 3 floats * 4 bytes).
        assert_eq!(bytes.len(), 4 + 2 * (12 + 2 * 3 * 4));
        assert_eq!(
            bytes.capacity(),
            bytes.len(),
            "the buffer is sized once, not grown"
        );
        // Little-endian floats, K first: layer 0's K starts at 0.0.
        assert_eq!(&bytes[16..20], &0.0f32.to_le_bytes());
        let mut d = KvCache::new(2);
        d.restore(&bytes, &Device::Cpu, &[(1, 1), (1, 1)]).unwrap();
        assert_eq!(d.to_bytes().unwrap(), bytes);
        let before = d.to_bytes().unwrap();
        assert!(
            d.restore(&bytes[..bytes.len() - 1], &Device::Cpu, &[(1, 1), (1, 1)])
                .is_err()
        );
        assert!(d.restore(&bytes, &Device::Cpu, &[(2, 1), (1, 1)]).is_err());
        assert_eq!(d.to_bytes().unwrap(), before);
    }

    #[test]
    fn view_taken_before_append_does_not_change_afterwards() {
        let mut l = LayerKv::default();
        l.append(&kv(0.0, 3), &kv(0.0, 3)).unwrap();
        let (k, _) = l.view().unwrap().unwrap();
        let before = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Append more — if `view()` had returned an aliased narrow of the
        // growable buffer, this in-place `slice_set` would corrupt `k`.
        l.append(&kv(100.0, 3), &kv(100.0, 3)).unwrap();
        let after = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(before, after);
        assert_eq!(before, [0.0, 1.0, 2.0]);
    }

    #[test]
    fn view_taken_at_full_capacity_survives_growth() {
        // At len == cap (256) the view spans the whole buffer; growing moves
        // the live prefix into a fresh buffer and never writes the old one,
        // so the view keeps its values.
        let mut l = LayerKv::default();
        for i in 0..256 {
            l.append(&kv(i as f32, 1), &kv(i as f32, 1)).unwrap();
        }
        let (k, _) = l.view().unwrap().unwrap();
        let before = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        l.append(&kv(9000.0, 1), &kv(9000.0, 1)).unwrap();
        let after = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(before, after);
        assert_eq!(before.len(), 256);
    }

    #[test]
    fn view_is_zero_copy_so_truncate_then_append_ends_its_validity() {
        // `view()` hands out the buffer itself, not a copy: the decode hot
        // path must never copy the whole cache per layer per step. The cost
        // is the documented contract — after a truncate, the next append
        // overwrites positions an older view still covers.
        let mut l = LayerKv::default();
        l.append(&kv(0.0, 4), &kv(0.0, 4)).unwrap();
        let (k, _) = l.view().unwrap().unwrap();
        l.truncate(2);
        l.append(&kv(100.0, 2), &kv(100.0, 2)).unwrap();
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [0.0, 1.0, 100.0, 101.0]
        );
    }

    #[test]
    fn kv_cache_len_is_zero_until_appended() {
        let c = KvCache::new(3);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
    }
}
