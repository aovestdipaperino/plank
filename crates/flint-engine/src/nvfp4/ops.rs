//! candle entry points: the NVFP4 matmul and the gathered `MoE` as custom ops,
//! so model code is ordinary tensor code on the CPU and on Metal alike.

use std::sync::Arc;

use candle_core::{CpuStorage, CustomOp1, CustomOp3, DType, Layout, Shape, Tensor, bail};

use crate::nvfp4::cpu;
use crate::nvfp4::tensor::{MoeExperts, NvWeight};

/// `x [tokens, cols]` (f32, contiguous) times a dense NVFP4 weight, scaled:
/// `[tokens, rows]`. Zero tokens give an empty `[0, rows]` without touching
/// the device.
///
/// # Errors
/// `x` not `[tokens, cols]` f32 contiguous, a weight with several experts,
/// or a device failure.
pub fn matmul(w: &Arc<NvWeight>, x: &Tensor) -> candle_core::Result<Tensor> {
    matmul_with(w, x, false)
}

/// [`matmul`] kept in f32 on Metal for any token count, for an input that
/// can exceed f16's range (a down projection's `SiLU(gate) * up`); the
/// plain one multiplies many tokens in f16.
///
/// # Errors
/// As [`matmul`].
pub fn matmul_exact(w: &Arc<NvWeight>, x: &Tensor) -> candle_core::Result<Tensor> {
    matmul_with(w, x, true)
}

fn matmul_with(w: &Arc<NvWeight>, x: &Tensor, exact: bool) -> candle_core::Result<Tensor> {
    let dims = mat_dims(w, x.layout())?;
    if dims.tokens == 0 {
        // Nothing to compute: an empty result, without binding or dispatching.
        if x.dtype() != DType::F32 {
            bail!("nvfp4 matmul wants f32 input");
        }
        return Tensor::zeros((0, w.rows()), DType::F32, x.device());
    }
    x.apply_op1_no_bwd(&MatmulOp {
        w: Arc::clone(w),
        exact,
    })
}

/// The gathered `MoE` over `x [tokens, hidden]` with `ids [tokens, k]` (u32)
/// and `route [tokens, k]` (f32): `[tokens, hidden]`. With no tokens or no
/// experts per token (`k == 0`) the result is zeros, computed on no device.
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
    let dims = moe_dims(m, x.layout(), ids.layout(), route.layout())?;
    if dims.tokens == 0 || dims.k_used == 0 {
        // No slots: every output row is an empty sum, so zero, without
        // binding or dispatching.
        if x.dtype() != DType::F32 || ids.dtype() != DType::U32 || route.dtype() != DType::F32 {
            bail!("moe wants f32 x, u32 ids and f32 route");
        }
        return Tensor::zeros((dims.tokens, m.hidden()), DType::F32, x.device());
    }
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
    /// Multiply in f32 on Metal even from `MM_MIN_TOKENS` up.
    #[cfg_attr(
        not(target_os = "macos"),
        allow(dead_code, reason = "read by the Metal decode path only")
    )]
    exact: bool,
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

    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        s: &candle_core::MetalStorage,
        l: &Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        let dims = mat_dims(&self.w, l)?;
        crate::nvfp4::metal::matmul(&self.w, s, dims, self.exact)
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

    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        let dims = moe_dims(&self.m, l1, l2, l3)?;
        crate::nvfp4::metal::moe(&self.m, s1, s2, s3, dims)
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

    /// Zero tokens, and zero experts per token, on `dev`: empty or zero
    /// outputs of the right shape, on the input's device.
    fn zero_size_case(dev: &Device) {
        let mut rng = Rng::new(14);
        let w = Arc::new(weight(&mut rng, 64, 5, 1));
        let x = Tensor::zeros((0, 64), candle_core::DType::F32, dev).unwrap();
        let y = matmul(&w, &x).unwrap();
        assert_eq!(y.dims(), &[0, 5]);
        assert!(y.device().same_device(dev));
        let wide = Tensor::zeros((0, 128), candle_core::DType::F32, dev).unwrap();
        assert!(matmul(&w, &wide).is_err(), "still checks the width");

        let m = experts(&mut rng, 64, 64, 2);
        let f32z = |s: (usize, usize)| Tensor::zeros(s, candle_core::DType::F32, dev).unwrap();
        let u32z = |s: (usize, usize)| Tensor::zeros(s, candle_core::DType::U32, dev).unwrap();
        let y = moe(&m, &f32z((0, 64)), &u32z((0, 6)), &f32z((0, 6))).unwrap();
        assert_eq!(y.dims(), &[0, 64]);
        let x = Tensor::from_vec(rng.vec(3 * 64), (3, 64), dev).unwrap();
        let y = moe(&m, &x, &u32z((3, 0)), &f32z((3, 0))).unwrap();
        assert_eq!(y.dims(), &[3, 64]);
        assert!(y.device().same_device(dev));
        let got: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert!(got.iter().all(|v| *v == 0.0), "no experts, no output");
    }

    #[test]
    fn zero_size_inputs_on_cpu() {
        zero_size_case(&Device::Cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn zero_size_inputs_on_metal() {
        let Some(dev) = metal() else { return };
        zero_size_case(&dev);
        // Nothing was dispatched, so nothing was bound either.
        let mut rng = Rng::new(15);
        let w = Arc::new(weight(&mut rng, 64, 5, 1));
        let x = Tensor::zeros((0, 64), candle_core::DType::F32, &dev).unwrap();
        matmul(&w, &x).unwrap();
        assert!(w.metal.lock().unwrap().is_none(), "matmul bound the weight");
        let m = experts(&mut rng, 64, 64, 2);
        let x = Tensor::zeros((2, 64), candle_core::DType::F32, &dev).unwrap();
        let none = Tensor::zeros((2, 0), candle_core::DType::U32, &dev).unwrap();
        let route = Tensor::zeros((2, 0), candle_core::DType::F32, &dev).unwrap();
        moe(&m, &x, &none, &route).unwrap();
        for p in [m.gate(), m.up(), m.down()] {
            assert!(p.metal.lock().unwrap().is_none(), "moe bound an expert");
        }
    }

    #[cfg(target_os = "macos")]
    fn metal() -> Option<Device> {
        match Device::new_metal(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("no Metal device ({e}); skipped");
                None
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn metal_matmul_case(seed: u64, cols: usize, rows: usize, tokens: usize) {
        let Some(dev) = metal() else { return };
        let mut rng = Rng::new(seed);
        let w = Arc::new(weight(&mut rng, cols, rows, 1));
        let xv = rng.vec(tokens * cols);
        let cpu = matmul(
            &w,
            &Tensor::from_vec(xv.clone(), (tokens, cols), &Device::Cpu).unwrap(),
        )
        .unwrap();
        let gpu = matmul(&w, &Tensor::from_vec(xv, (tokens, cols), &dev).unwrap()).unwrap();
        let want: Vec<f32> = cpu.flatten_all().unwrap().to_vec1().unwrap();
        let got: Vec<f32> = gpu.flatten_all().unwrap().to_vec1().unwrap();
        assert_close(&got, &want, 1e-3);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matvec_matches_cpu() {
        for (i, (cols, rows, tokens)) in [(64, 1, 1), (192, 33, 3), (2048, 100, 1), (512, 7, 7)]
            .into_iter()
            .enumerate()
        {
            metal_matmul_case(20 + u64::try_from(i).unwrap(), cols, rows, tokens);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tiled_matmul_matches_cpu_with_ragged_edges() {
        // 8 is the first tiled size; 37 tokens and 45 rows leave partial tiles.
        for (i, (cols, rows, tokens)) in [(64, 32, 8), (192, 45, 37), (2048, 64, 64), (512, 1, 9)]
            .into_iter()
            .enumerate()
        {
            metal_matmul_case(40 + u64::try_from(i).unwrap(), cols, rows, tokens);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_mapped_weight_binds_the_whole_file_without_copying() {
        let Some(dev) = metal() else { return };
        let mut rng = Rng::new(30);
        let src = weight(&mut rng, 128, 9, 1);
        let bytes = crate::testgguf::Builder::new()
            .tensor("w.weight", &[128, 9], crate::gguf::GGML_NVFP4, src.bytes())
            .tensor(
                "w.scale",
                &[1],
                crate::gguf::GGML_F32,
                &src.scale(0).to_le_bytes(),
            )
            .bytes();
        let file = crate::file::GgufFile::from_bytes(&bytes).unwrap();
        let w = Arc::new(file.nv_weight("w").unwrap());
        assert_eq!(file.metal_no_copy(), None, "not bound before first use");
        let xv = rng.vec(128);
        let got: Vec<f32> = matmul(&w, &Tensor::from_vec(xv.clone(), (1, 128), &dev).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_close(&got, &naive_matmul(&w, 0, &xv, 1), 1e-3);
        assert_eq!(
            file.metal_no_copy(),
            Some(true),
            "the mapping became one no-copy buffer"
        );
    }

    /// A file with one weight, `w`, of 64 columns by 3 rows.
    #[cfg(target_os = "macos")]
    fn one_weight_file(rng: &mut Rng) -> crate::file::GgufFile {
        let src = weight(rng, 64, 3, 1);
        let bytes = crate::testgguf::Builder::new()
            .tensor("w.weight", &[64, 3], crate::gguf::GGML_NVFP4, src.bytes())
            .tensor(
                "w.scale",
                &[1],
                crate::gguf::GGML_F32,
                &src.scale(0).to_le_bytes(),
            )
            .bytes();
        crate::file::GgufFile::from_bytes(&bytes).unwrap()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_binding_made_for_another_device_is_refused() {
        let Some(dev) = metal() else { return };
        let mut rng = Rng::new(34);
        let file = one_weight_file(&mut rng);
        let x = Tensor::from_vec(rng.vec(64), (1, 64), &dev).unwrap();
        let w = Arc::new(file.nv_weight("w").unwrap());
        matmul(&w, &x).unwrap();
        // Pretend the weight was bound on some other GPU.
        w.metal.lock().unwrap().as_mut().unwrap().device ^= 1;
        let e = matmul(&w, &x).unwrap_err().to_string();
        assert!(e.contains("another Metal device"), "{e}");

        // A fresh weight from the same file binds through the file's shared
        // no-copy buffer, which is refused the same way.
        let fresh = Arc::new(file.nv_weight("w").unwrap());
        file.mapped_file().metal.lock().unwrap().as_mut().unwrap().0 ^= 1;
        let e = matmul(&fresh, &x).unwrap_err().to_string();
        assert!(e.contains("another Metal device"), "{e}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_decodes_edge_case_scales_like_the_cpu() {
        let Some(dev) = metal() else { return };
        let mut rng = Rng::new(31);
        // Zero, the 0x7F NaN code (decodes to 0), a subnormal, and bit 7 set
        // (ignored), in every sub-block position across the rows.
        let edge = [0x00u8, 0x7F, 0x03, 0xB8];
        let rows = 4;
        let mut data = Vec::new();
        for r in 0..rows {
            for s in 0..4 {
                data.push(edge[(r + s) % 4]);
            }
            data.extend((0..32).map(|_| u8::try_from(rng.below(256)).unwrap()));
        }
        let w = Arc::new(NvWeight::owned(64, rows, 1, data, vec![1.25]).unwrap());
        let xv = rng.vec(2 * 64);
        let cpu = matmul(
            &w,
            &Tensor::from_vec(xv.clone(), (2, 64), &Device::Cpu).unwrap(),
        )
        .unwrap();
        let gpu = matmul(&w, &Tensor::from_vec(xv.clone(), (2, 64), &dev).unwrap()).unwrap();
        let want: Vec<f32> = cpu.flatten_all().unwrap().to_vec1().unwrap();
        let got: Vec<f32> = gpu.flatten_all().unwrap().to_vec1().unwrap();
        assert_close(&got, &want, 1e-5);
        assert_close(&got, &naive_matmul(&w, 0, &xv, 2), 1e-5);
        assert!(
            want.iter().any(|v| *v != 0.0),
            "the subnormal and 0xB8 scales contribute"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matmul_reads_a_narrowed_input_at_its_offset() {
        let Some(dev) = metal() else { return };
        let mut rng = Rng::new(32);
        let w = Arc::new(weight(&mut rng, 192, 11, 1));
        let xv = rng.vec(5 * 192);
        let x = Tensor::from_vec(xv.clone(), (5, 192), &dev)
            .unwrap()
            .narrow(0, 2, 3)
            .unwrap();
        let got: Vec<f32> = matmul(&w, &x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_close(&got, &naive_matmul(&w, 0, &xv[2 * 192..], 3), 1e-3);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_no_copy_buffer_keeps_the_mapping_alive_until_metal_frees_it() {
        let Some(dev) = metal() else { return };
        let mut rng = Rng::new(33);
        let src = weight(&mut rng, 64, 2, 1);
        let bytes = crate::testgguf::Builder::new()
            .tensor("w.weight", &[64, 2], crate::gguf::GGML_NVFP4, src.bytes())
            .tensor(
                "w.scale",
                &[1],
                crate::gguf::GGML_F32,
                &src.scale(0).to_le_bytes(),
            )
            .bytes();
        let file = crate::file::GgufFile::from_bytes(&bytes).unwrap();
        let map = Arc::clone(file.mapped_file().map());
        let w = Arc::new(file.nv_weight("w").unwrap());
        let x = Tensor::from_vec(rng.vec(64), (1, 64), &dev).unwrap();
        let y = matmul(&w, &x).unwrap();
        assert_eq!(Arc::strong_count(&map), 3, "file, test and the deallocator");
        drop((w, file));
        let _ = y.to_vec2::<f32>().unwrap();
        drop(y);
        dev.synchronize().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while Arc::strong_count(&map) > 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "Metal never released the mapping"
            );
            std::thread::yield_now();
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_moe_matches_cpu() {
        let Some(dev) = metal() else { return };
        let mut rng = Rng::new(50);
        let m = experts(&mut rng, 256, 128, 16);
        // From 8 tokens up the grouped kernels run; 100 tokens give each
        // expert more than one 32-slot tile.
        for (tokens, k_used) in [(1, 6), (5, 6), (9, 2), (100, 6)] {
            let xv = rng.vec(tokens * 256);
            let idv: Vec<u32> = (0..tokens * k_used).map(|_| rng.below(16)).collect();
            let rv: Vec<f32> = (0..tokens * k_used).map(|_| rng.uniform().abs()).collect();
            let run = |d: &Device| -> Vec<f32> {
                moe(
                    &m,
                    &Tensor::from_vec(xv.clone(), (tokens, 256), d).unwrap(),
                    &Tensor::from_vec(idv.clone(), (tokens, k_used), d).unwrap(),
                    &Tensor::from_vec(rv.clone(), (tokens, k_used), d).unwrap(),
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
            };
            assert_close(&run(&dev), &run(&Device::Cpu), 1e-3);
        }
    }
}
