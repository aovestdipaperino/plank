//! An NVFP4 weight: `experts` stacked matrices of `rows` by `cols`, each row
//! a run of 36-byte blocks, with one output scale per expert. A dense weight
//! is the one-expert case.

use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::sync::Mutex;

use crate::file::MappedFile;
use crate::nvfp4::block::{BLOCK_BYTES, QK, dequant_row};
use crate::{Result, err};

#[derive(Debug)]
enum Bytes {
    Owned(Vec<u8>),
    Mapped {
        file: Arc<MappedFile>,
        offset: usize,
        len: usize,
    },
}

/// One NVFP4 weight tensor.
#[derive(Debug)]
pub struct NvWeight {
    cols: usize,
    rows: usize,
    experts: usize,
    scales: Vec<f32>,
    /// The Metal buffers this weight is bound to, made on first use.
    /// Declared before `bytes` so it is dropped first: a binding may be a
    /// no-copy buffer over the mapping `bytes` keeps alive.
    #[cfg(target_os = "macos")]
    pub(crate) metal: Mutex<Option<crate::nvfp4::metal::Binding>>,
    bytes: Bytes,
}

impl NvWeight {
    /// A weight over bytes it owns.
    ///
    /// # Errors
    /// `cols` not a multiple of 64, an empty shape, a scale count other than
    /// `experts`, or a byte count that does not match the shape.
    pub fn owned(
        cols: usize,
        rows: usize,
        experts: usize,
        data: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        check(cols, rows, experts, data.len(), &scales)?;
        Ok(Self {
            cols,
            rows,
            experts,
            scales,
            bytes: Bytes::Owned(data),
            #[cfg(target_os = "macos")]
            metal: Mutex::new(None),
        })
    }

    pub(crate) fn mapped(
        file: Arc<MappedFile>,
        offset: usize,
        len: usize,
        cols: usize,
        rows: usize,
        experts: usize,
        scales: Vec<f32>,
    ) -> Result<Self> {
        check(cols, rows, experts, len, &scales)?;
        Ok(Self {
            cols,
            rows,
            experts,
            scales,
            bytes: Bytes::Mapped { file, offset, len },
            #[cfg(target_os = "macos")]
            metal: Mutex::new(None),
        })
    }

    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Bytes per row.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.cols / QK * BLOCK_BYTES
    }

    /// Bytes per expert matrix.
    #[must_use]
    pub fn expert_bytes(&self) -> usize {
        self.row_bytes() * self.rows
    }

    /// Every block, expert by expert, row by row.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        match &self.bytes {
            Bytes::Owned(v) => v,
            Bytes::Mapped { file, offset, len } => &file.bytes()[*offset..*offset + *len],
        }
    }

    /// Row `row` of expert `expert`.
    ///
    /// # Panics
    /// When either index is out of range.
    #[must_use]
    pub fn row(&self, expert: usize, row: usize) -> &[u8] {
        let rb = self.row_bytes();
        let start = (expert * self.rows + row) * rb;
        &self.bytes()[start..start + rb]
    }

    /// The output scale of expert `expert` (0 for a dense weight).
    ///
    /// # Panics
    /// When `expert` is out of range.
    #[must_use]
    pub fn scale(&self, expert: usize) -> f32 {
        self.scales[expert]
    }

    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Expert `expert` decoded to `rows * cols` values with its scale
    /// applied. The test oracle; never on a hot path.
    #[must_use]
    pub fn dequant(&self, expert: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.rows * self.cols];
        for (r, o) in out.chunks_exact_mut(self.cols).enumerate() {
            dequant_row(self.row(expert, r), o);
        }
        let s = self.scale(expert);
        for v in &mut out {
            *v *= s;
        }
        out
    }

    /// The mapping and offset this weight borrows, when it comes from a file.
    pub(crate) fn mapping(&self) -> Option<(&Arc<MappedFile>, usize)> {
        match &self.bytes {
            Bytes::Owned(_) => None,
            Bytes::Mapped { file, offset, .. } => Some((file, *offset)),
        }
    }
}

fn check(cols: usize, rows: usize, experts: usize, len: usize, scales: &[f32]) -> Result<()> {
    if cols == 0 || !cols.is_multiple_of(QK) {
        return Err(err(format!(
            "NVFP4 rows must be whole 64-value blocks, got {cols} columns"
        )));
    }
    if rows == 0 || experts == 0 {
        return Err(err("empty NVFP4 weight"));
    }
    if scales.len() != experts {
        return Err(err(format!(
            "{} scales for {experts} experts",
            scales.len()
        )));
    }
    let want = (cols / QK * BLOCK_BYTES)
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(experts))
        .ok_or_else(|| err("NVFP4 weight too large"))?;
    if len != want {
        return Err(err(format!(
            "NVFP4 weight holds {len} bytes, its shape needs {want}"
        )));
    }
    Ok(())
}

/// The three expert tensors of one `MoE` layer: gate and up are
/// `[experts, ffn, hidden]`, down is `[experts, hidden, ffn]`.
#[derive(Debug, Clone)]
pub struct MoeExperts {
    pub gate: Arc<NvWeight>,
    pub up: Arc<NvWeight>,
    pub down: Arc<NvWeight>,
}

impl MoeExperts {
    /// Groups the three tensors after checking their shapes agree.
    ///
    /// # Errors
    /// Different expert counts, gate and up shaped differently, or down not
    /// the transpose shape of gate.
    pub fn new(gate: Arc<NvWeight>, up: Arc<NvWeight>, down: Arc<NvWeight>) -> Result<Self> {
        let shape = |w: &NvWeight| (w.experts(), w.rows(), w.cols());
        if shape(&gate) != shape(&up) {
            return Err(err(format!(
                "expert gate {:?} and up {:?} differ",
                shape(&gate),
                shape(&up)
            )));
        }
        if shape(&down) != (gate.experts(), gate.cols(), gate.rows()) {
            return Err(err(format!(
                "expert down {:?} does not match gate {:?}",
                shape(&down),
                shape(&gate)
            )));
        }
        Ok(Self { gate, up, down })
    }

    /// Model width (`d`).
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.gate.cols()
    }

    /// Expert FFN width (`f`).
    #[must_use]
    pub fn ffn(&self) -> usize {
        self.gate.rows()
    }

    /// Number of experts.
    #[must_use]
    pub fn count(&self) -> usize {
        self.gate.experts()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(cols: usize, rows: usize, experts: usize) -> Arc<NvWeight> {
        let n = cols / 64 * 36 * rows * experts;
        Arc::new(NvWeight::owned(cols, rows, experts, vec![0; n], vec![1.0; experts]).unwrap())
    }

    #[test]
    fn moe_experts_checks_the_three_shapes_agree() {
        let ok = MoeExperts::new(w(128, 64, 4), w(128, 64, 4), w(64, 128, 4)).unwrap();
        assert_eq!((ok.hidden(), ok.ffn(), ok.count()), (128, 64, 4));
        assert!(MoeExperts::new(w(128, 64, 4), w(128, 64, 3), w(64, 128, 4)).is_err());
        assert!(MoeExperts::new(w(128, 64, 4), w(128, 64, 4), w(128, 64, 4)).is_err());
        assert!(MoeExperts::new(w(128, 64, 4), w(128, 128, 4), w(64, 128, 4)).is_err());
    }
}
