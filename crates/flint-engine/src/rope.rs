//! Rotary position tables, `NeoX` layout (llama.cpp `LLAMA_ROPE_TYPE_NEOX`):
//! dimension `i` pairs with `i + head_dim / 2`, over the whole head.

use candle_core::{DType, Device, Result, Tensor};

/// `cos` and `sin` of `position * base^(-2i / head_dim)` for every position.
#[derive(Debug)]
pub(crate) struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    // Head dims and positions are small integers, exact in f64; the inverse
    // frequencies are computed in f64 and stored as f32 on purpose.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub(crate) fn new(
        head_dim: usize,
        base: f32,
        positions: usize,
        device: &Device,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| (1.0 / f64::from(base).powf((2 * i) as f64 / head_dim as f64)) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), device)?;
        let Ok(n) = u32::try_from(positions) else {
            candle_core::bail!("{positions} positions do not fit a u32")
        };
        let pos = Tensor::arange(0u32, n, device)?
            .to_dtype(DType::F32)?
            .reshape((positions, 1))?;
        let freqs = pos.matmul(&inv)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    /// The `[positions, head_dim / 2]` cos and sin tables.
    #[cfg(target_os = "macos")]
    pub(crate) fn tables(&self) -> (&Tensor, &Tensor) {
        (&self.cos, &self.sin)
    }

    /// Rotates `x` (`[1, heads, t, head_dim]`) as positions `offset..offset + t`.
    pub(crate) fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let t = x.dim(2)?;
        let cos = self.cos.narrow(0, offset, t)?;
        let sin = self.sin.narrow(0, offset, t)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }
}
