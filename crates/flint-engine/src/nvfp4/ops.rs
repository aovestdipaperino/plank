//! candle entry points: the NVFP4 matmul and the gathered `MoE` as custom ops,
//! so model code is ordinary tensor code on the CPU and on Metal alike.

use std::sync::Arc;

use candle_core::{CpuStorage, CustomOp1, CustomOp3, Layout, Shape, Tensor, bail};

use crate::nvfp4::cpu;
use crate::nvfp4::tensor::{MoeExperts, NvWeight};

/// `x [tokens, cols]` (f32, contiguous) times a dense NVFP4 weight, scaled:
/// `[tokens, rows]`.
///
/// # Errors
/// `x` not `[tokens, cols]` f32 contiguous, a weight with several experts,
/// or a device failure.
pub fn matmul(w: &Arc<NvWeight>, x: &Tensor) -> candle_core::Result<Tensor> {
    x.apply_op1_no_bwd(&MatmulOp { w: Arc::clone(w) })
}

/// The gathered `MoE` over `x [tokens, hidden]` with `ids [tokens, k]` (u32)
/// and `route [tokens, k]` (f32): `[tokens, hidden]`.
///
/// # Errors
/// Shapes or dtypes that do not match, an id past the last expert (checked
/// on the CPU; on Metal an out-of-range slot contributes zero), or a device
/// failure.
pub fn moe(
    m: &MoeExperts,
    x: &Tensor,
    ids: &Tensor,
    route: &Tensor,
) -> candle_core::Result<Tensor> {
    x.apply_op3_no_bwd(ids, route, &MoeOp { m: m.clone() })
}

pub(crate) fn to_candle(e: crate::Error) -> candle_core::Error {
    candle_core::Error::msg(e.0)
}

/// Where a matmul input starts and how many tokens it holds.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatDims {
    pub tokens: usize,
    pub start: usize,
}

pub(crate) fn mat_dims(w: &NvWeight, l: &Layout) -> candle_core::Result<MatDims> {
    if w.experts() != 1 {
        bail!(
            "nvfp4 matmul takes a dense weight, this one has {} experts",
            w.experts()
        );
    }
    let (tokens, cols) = l.shape().dims2()?;
    if cols != w.cols() {
        bail!(
            "nvfp4 matmul: input has {cols} columns, the weight {}",
            w.cols()
        );
    }
    let (start, _) = l
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::msg("nvfp4 matmul wants a contiguous input"))?;
    Ok(MatDims { tokens, start })
}

/// Shapes and starts of the three `MoE` inputs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MoeDims {
    pub tokens: usize,
    pub k_used: usize,
    pub x_start: usize,
    pub ids_start: usize,
    pub route_start: usize,
}

pub(crate) fn moe_dims(
    m: &MoeExperts,
    l1: &Layout,
    l2: &Layout,
    l3: &Layout,
) -> candle_core::Result<MoeDims> {
    let (tokens, d) = l1.shape().dims2()?;
    if d != m.hidden() {
        bail!("moe: input has {d} columns, the experts {}", m.hidden());
    }
    let (t2, k_used) = l2.shape().dims2()?;
    let (t3, k3) = l3.shape().dims2()?;
    if t2 != tokens || t3 != tokens || k3 != k_used {
        bail!(
            "moe: ids {:?} and route {:?} must both be [{tokens}, k]",
            l2.dims(),
            l3.dims()
        );
    }
    let start = |l: &Layout| {
        l.contiguous_offsets()
            .map(|(s, _)| s)
            .ok_or_else(|| candle_core::Error::msg("moe wants contiguous inputs"))
    };
    Ok(MoeDims {
        tokens,
        k_used,
        x_start: start(l1)?,
        ids_start: start(l2)?,
        route_start: start(l3)?,
    })
}

#[derive(Debug)]
struct MatmulOp {
    w: Arc<NvWeight>,
}

impl CustomOp1 for MatmulOp {
    fn name(&self) -> &'static str {
        "nvfp4-matmul"
    }

    // s, l, x and y are the storage, layout and data names candle uses for this hook.
    #[allow(clippy::many_single_char_names)]
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let dims = mat_dims(&self.w, l)?;
        let CpuStorage::F32(v) = s else {
            bail!("nvfp4 matmul wants f32 input");
        };
        let cols = self.w.cols();
        let x = &v[dims.start..dims.start + dims.tokens * cols];
        let mut y = vec![0f32; dims.tokens * self.w.rows()];
        cpu::matmul(&self.w, 0, x, dims.tokens, &mut y).map_err(to_candle)?;
        Ok((
            CpuStorage::F32(y),
            Shape::from((dims.tokens, self.w.rows())),
        ))
    }
}

#[derive(Debug)]
struct MoeOp {
    m: MoeExperts,
}

impl CustomOp3 for MoeOp {
    fn name(&self) -> &'static str {
        "nvfp4-moe"
    }

    // s1..s3 and l1..l3 are the storage and layout pairs candle passes in.
    #[allow(clippy::many_single_char_names)]
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let dims = moe_dims(&self.m, l1, l2, l3)?;
        let (CpuStorage::F32(x), CpuStorage::U32(ids), CpuStorage::F32(route)) = (s1, s2, s3)
        else {
            bail!("moe wants f32 x, u32 ids and f32 route");
        };
        let (t, k, d) = (dims.tokens, dims.k_used, self.m.hidden());
        let mut y = vec![0f32; t * d];
        cpu::moe(
            &self.m,
            &x[dims.x_start..dims.x_start + t * d],
            t,
            &ids[dims.ids_start..dims.ids_start + t * k],
            &route[dims.route_start..dims.route_start + t * k],
            k,
            &mut y,
        )
        .map_err(to_candle)?;
        Ok((CpuStorage::F32(y), Shape::from((t, d))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Rng, assert_close, experts, naive_matmul, naive_moe, weight};
    use candle_core::Device;

    #[test]
    fn matmul_op_on_cpu_matches_the_oracle() {
        let mut rng = Rng::new(10);
        let w = Arc::new(weight(&mut rng, 256, 40, 1));
        let xv = rng.vec(5 * 256);
        let x = Tensor::from_vec(xv.clone(), (5, 256), &Device::Cpu).unwrap();
        let y = matmul(&w, &x).unwrap();
        assert_eq!(y.dims(), &[5, 40]);
        let got: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert_close(&got, &naive_matmul(&w, 0, &xv, 5), 1e-5);
    }

    #[test]
    fn matmul_op_reads_a_narrowed_input_at_its_offset() {
        let mut rng = Rng::new(11);
        let w = Arc::new(weight(&mut rng, 64, 3, 1));
        let xv = rng.vec(4 * 64);
        let x = Tensor::from_vec(xv.clone(), (4, 64), &Device::Cpu)
            .unwrap()
            .narrow(0, 2, 2)
            .unwrap();
        let got: Vec<f32> = matmul(&w, &x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_close(&got, &naive_matmul(&w, 0, &xv[2 * 64..], 2), 1e-5);
    }

    #[test]
    fn moe_op_on_cpu_matches_the_oracle() {
        let mut rng = Rng::new(12);
        let m = experts(&mut rng, 128, 64, 8);
        let (tokens, k_used) = (2, 6);
        let xv = rng.vec(tokens * 128);
        let idv: Vec<u32> = (0..tokens * k_used).map(|_| rng.below(8)).collect();
        let rv: Vec<f32> = (0..tokens * k_used).map(|_| rng.uniform().abs()).collect();
        let dev = Device::Cpu;
        let y = moe(
            &m,
            &Tensor::from_vec(xv.clone(), (tokens, 128), &dev).unwrap(),
            &Tensor::from_vec(idv.clone(), (tokens, k_used), &dev).unwrap(),
            &Tensor::from_vec(rv.clone(), (tokens, k_used), &dev).unwrap(),
        )
        .unwrap();
        let got: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert_close(&got, &naive_moe(&m, &xv, tokens, &idv, &rv, k_used), 1e-4);
    }

    #[test]
    fn ops_refuse_bad_inputs() {
        let mut rng = Rng::new(13);
        let dev = Device::Cpu;
        let w = Arc::new(weight(&mut rng, 64, 2, 1));
        let x = Tensor::zeros((1, 128), candle_core::DType::F32, &dev).unwrap();
        assert!(matmul(&w, &x).is_err(), "wrong width");
        let xt = Tensor::zeros((64, 2), candle_core::DType::F32, &dev)
            .unwrap()
            .t()
            .unwrap();
        assert!(matmul(&w, &xt).is_err(), "not contiguous");
        let multi = Arc::new(weight(&mut rng, 64, 2, 2));
        assert!(
            matmul(
                &multi,
                &Tensor::zeros((1, 64), candle_core::DType::F32, &dev).unwrap()
            )
            .is_err()
        );
        let m = experts(&mut rng, 64, 64, 2);
        let x = Tensor::zeros((2, 64), candle_core::DType::F32, &dev).unwrap();
        let ids = Tensor::zeros((1, 2), candle_core::DType::U32, &dev).unwrap();
        let r = Tensor::zeros((2, 2), candle_core::DType::F32, &dev).unwrap();
        assert!(moe(&m, &x, &ids, &r).is_err(), "ids rows != tokens");
    }
}
