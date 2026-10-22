//! Opt-in checks on the real Flint GGUF:
//!
//! ```sh
//! PLANK_FLINT_GGUF=~/.plank/models/flint-nvfp4.gguf \
//!   cargo test -p flint-engine --features candle --release --test reference_kernels -- --ignored --nocapture
//! ```
#![cfg(feature = "candle")]

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use flint_engine::file::GgufFile;
use flint_engine::nvfp4::ops;
use flint_engine::nvfp4::tensor::{MoeExperts, NvWeight};

fn gguf() -> GgufFile {
    let p = std::env::var_os("PLANK_FLINT_GGUF").expect("set PLANK_FLINT_GGUF to the Flint GGUF");
    GgufFile::open(&PathBuf::from(p)).expect("open the GGUF")
}

#[allow(clippy::cast_precision_loss)]
fn input(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2_654_435_761 + seed) % 1000) as f32 / 500.0) - 1.0)
        .collect()
}

fn close(got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= rel * scale, "index {i}: got {g}, want {w}");
    }
}

#[allow(clippy::cast_possible_truncation)]
fn naive(w: &NvWeight, x: &[f32]) -> Vec<f32> {
    let m = w.dequant(0);
    m.chunks_exact(w.cols())
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>() as f32
        })
        .collect()
}

fn run(w: &Arc<NvWeight>, x: &[f32], tokens: usize, d: &Device) -> Vec<f32> {
    let t = Tensor::from_vec(x.to_vec(), (tokens, w.cols()), d).unwrap();
    ops::matmul(w, &t)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn the_file_has_the_shapes_the_engine_expects() {
    let f = gguf();
    let q = f.nv_weight("blk.0.attn_q").unwrap();
    assert_eq!((q.cols(), q.rows(), q.experts()), (2048, 4096, 1));
    let g = f.nv_weight("blk.2.ffn_gate_exps").unwrap();
    assert_eq!((g.cols(), g.rows(), g.experts()), (2048, 512, 256));
    assert_eq!(g.scales().len(), 256, "one scale per expert");
    let out = f.nv_weight("output").unwrap();
    assert_eq!((out.cols(), out.rows()), (2048, 150_144));
}

#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn cpu_matmul_matches_dequant_on_attn_q() {
    let w = Arc::new(gguf().nv_weight("blk.0.attn_q").unwrap());
    let x = input(2048, 1);
    close(&run(&w, &x, 1, &Device::Cpu), &naive(&w, &x), 1e-5);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn metal_matches_cpu_on_real_tensors() {
    let dev = Device::new_metal(0).expect("Metal device");
    let f = gguf();
    for name in [
        "blk.0.attn_q",
        "blk.2.attn_output",
        "blk.0.ffn_down",
        "output",
    ] {
        let w = Arc::new(f.nv_weight(name).unwrap());
        for tokens in [1, 37] {
            let x = input(tokens * w.cols(), tokens);
            close(
                &run(&w, &x, tokens, &dev),
                &run(&w, &x, tokens, &Device::Cpu),
                1e-3,
            );
        }
    }
    let m = MoeExperts::new(
        Arc::new(f.nv_weight("blk.2.ffn_gate_exps").unwrap()),
        Arc::new(f.nv_weight("blk.2.ffn_up_exps").unwrap()),
        Arc::new(f.nv_weight("blk.2.ffn_down_exps").unwrap()),
    )
    .unwrap();
    let x = input(3 * 2048, 7);
    let ids: Vec<u32> = vec![
        0, 17, 42, 99, 128, 255, 1, 2, 3, 4, 5, 6, 200, 201, 202, 203, 204, 205,
    ];
    let route: Vec<f32> = (0..18)
        .map(|i| 0.05 + 0.01 * f32::from(u8::try_from(i).unwrap()))
        .collect();
    let go = |d: &Device| -> Vec<f32> {
        ops::moe(
            &m,
            &Tensor::from_vec(x.clone(), (3, 2048), d).unwrap(),
            &Tensor::from_vec(ids.clone(), (3, 6), d).unwrap(),
            &Tensor::from_vec(route.clone(), (3, 6), d).unwrap(),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
    };
    close(&go(&dev), &go(&Device::Cpu), 1e-3);
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Row-major matrix times vector in f64, on already-dequantized weights.
#[allow(clippy::cast_possible_truncation)]
fn matvec(m: &[f32], cols: usize, x: &[f32]) -> Vec<f32> {
    m.chunks_exact(cols)
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
#[ignore = "needs the Flint GGUF: PLANK_FLINT_GGUF=path"]
fn cpu_moe_matches_a_hand_composed_layer() {
    let f = gguf();
    let m = MoeExperts::new(
        Arc::new(f.nv_weight("blk.2.ffn_gate_exps").unwrap()),
        Arc::new(f.nv_weight("blk.2.ffn_up_exps").unwrap()),
        Arc::new(f.nv_weight("blk.2.ffn_down_exps").unwrap()),
    )
    .unwrap();
    let x = input(2048, 11);
    let ids: Vec<u32> = vec![5, 77, 130, 3, 255, 64];
    let route: Vec<f32> = vec![0.3, 0.2, 0.15, 0.1, 0.1, 0.05];
    let mut want = vec![0f32; 2048];
    for (slot, &e) in ids.iter().enumerate() {
        let e = e as usize;
        let gate = matvec(&m.gate.dequant(e), m.gate.cols(), &x);
        let up = matvec(&m.up.dequant(e), m.up.cols(), &x);
        let hid: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        let down = matvec(&m.down.dequant(e), m.down.cols(), &hid);
        for (acc, v) in want.iter_mut().zip(&down) {
            *acc += route[slot] * v;
        }
    }
    let cpu = Device::Cpu;
    let got: Vec<f32> = ops::moe(
        &m,
        &Tensor::from_vec(x, (1, 2048), &cpu).unwrap(),
        &Tensor::from_vec(ids, (1, 6), &cpu).unwrap(),
        &Tensor::from_vec(route, (1, 6), &cpu).unwrap(),
    )
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1()
    .unwrap();
    close(&got, &want, 1e-4);
}
