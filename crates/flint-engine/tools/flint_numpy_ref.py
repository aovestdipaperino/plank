#!/usr/bin/env python3
"""An independent Flint forward pass in NumPy float64: the parity oracle.

llama.cpp at the pinned commit runs Flint's NVFP4 matmuls on the CPU with
activations rounded to Q8_0 and keeps an f16 KV cache, so its probabilities
move by up to a few tenths at single positions and it cannot serve as a
precise per-position oracle (FINDINGS). This script recomputes the model in
float64 from upstream semantics instead: llama.cpp 970b51c
`src/models/flint.cpp`, and `src/llama-graph.cpp` `build_norm`,
`build_lora_mm`, `build_lora_mm_id`, `build_moe_ffn`, `build_ffn` and
`build_attn_mha`. It is written from those sources, not from plank's Rust.

Tensors are read, and NVFP4 dequantized, by llama.cpp's own gguf-py from the
pinned clone. Each step that needs a weight dequantizes only that weight (for
the experts, only the selected experts' rows of the memmapped data), so peak
memory stays near one output-head chunk plus one layer.

For each case it runs `tokens + greedy[:4]` from flint_reference.json in one
causal pass and records the float64 softmax top 20 `[[id, p], ...]` at the
prompt's last position and after each of the four teacher-forced tokens,
which by causality equals a prefill followed by four single steps.

Run from the repository root (the venv is the fixtures one, plus numpy and
the pinned gguf-py):

    $HOME/.cache/plank/flint-fixtures-venv/bin/pip install numpy \\
        $HOME/.cache/plank/llama.cpp-970b51c/gguf-py
    GGUF=$HOME/.plank/models/flint-nvfp4.gguf \\
    $HOME/.cache/plank/flint-fixtures-venv/bin/python -I \\
        crates/flint-engine/tools/flint_numpy_ref.py crates/flint-engine/tests/fixtures
"""
import json
import os
import pathlib
import resource
import sys
import time

import gguf
import numpy as np

COMMIT = "970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc"
CASES = ["user_only", "system_user"]
FORCED = 4  # teacher-forced steps after the prompt
TOP = 20


class Flint:
    def __init__(self, path):
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        f = self.r.fields

        def kv(name):
            return f[name].contents()

        self.n_layer = kv("flint.block_count")
        self.n_embd = kv("flint.embedding_length")
        self.n_head = kv("flint.attention.head_count")
        self.n_head_kv = kv("flint.attention.head_count_kv")
        self.head_dim = kv("flint.attention.key_length")
        # The f32 value llama.cpp reads, widened.
        self.eps = float(np.float32(kv("flint.attention.layer_norm_rms_epsilon")))
        self.rope_base = float(kv("flint.rope.freq_base"))
        self.n_swa = kv("flint.attention.sliding_window")
        self.swa = kv("flint.attention.sliding_window_pattern")
        self.rope = kv("flint.attention.rope_pattern")
        self.n_dense = kv("flint.leading_dense_block_count")
        self.n_used = kv("flint.expert_used_count")
        assert kv("flint.expert_gating_func") == 2, "sigmoid gating"  # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
        assert bool(kv("flint.expert_weights_norm"))
        self.w_scale = float(kv("flint.expert_weights_scale"))
        assert not any(n.startswith("blk.") and n.endswith("exp_probs_b.bias") for n in self.t)

    # --- weights -----------------------------------------------------------

    def f32(self, name):
        return np.asarray(self.t[name].data, dtype=np.float64)

    def scale(self, name):
        # `<name>.scale`: build_lora_mm multiplies the matmul *output* by it.
        return self.f32(name.removesuffix(".weight") + ".scale")

    def dense(self, name, rows=None):
        """Dequantized weight [out, in] as float64 (gguf-py does the NVFP4 decode)."""
        t = self.t[name]
        raw = np.asarray(t.data if rows is None else t.data[rows])
        return gguf.quants.dequantize(raw, t.tensor_type).astype(np.float64)

    def expert(self, name, e):
        """One expert's [out, in] matrix, sliced from the memmap before decoding."""
        t = self.t[name]
        return gguf.quants.dequantize(np.asarray(t.data[e]), t.tensor_type).astype(np.float64)

    def bf16_rows(self, name, ids):
        # token_embd / ffn_gate_inp are BF16: widen the stored bits to f32, then f64.
        raw = np.asarray(self.t[name].data[ids]).view(np.uint16).astype(np.uint32) << 16
        return raw.view(np.float32).astype(np.float64)

    def mm(self, name, x):
        """build_lora_mm: x @ W^T, then times the weight's scale."""
        return (x @ self.dense(name).T) * self.scale(name)

    # --- blocks ------------------------------------------------------------

    def rms_norm(self, x, w):
        # build_norm(LLM_NORM_RMS): x / sqrt(mean(x^2) + eps) * w
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps) * w

    def rope_neox(self, x, pos):
        # ggml_rope_ext, LLAMA_ROPE_TYPE_NEOX over the whole head (n_rot = head_dim):
        # pairs (i, i + d/2) rotate by pos * base^(-2i/d). No rope scaling.
        d = x.shape[-1]
        half = d // 2
        inv = self.rope_base ** (-2.0 * np.arange(half, dtype=np.float64) / d)
        ang = pos[:, None].astype(np.float64) * inv[None, :]  # [T, half]
        cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        a, b = x[..., :half], x[..., half:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)

    def attention(self, il, x):
        p = f"blk.{il}."
        T = x.shape[0]
        hd = self.head_dim
        # build_qkv: three scaled projections, viewed as heads.
        q = self.mm(p + "attn_q.weight", x).reshape(T, self.n_head, hd)
        k = self.mm(p + "attn_k.weight", x).reshape(T, self.n_head_kv, hd)
        v = self.mm(p + "attn_v.weight", x).reshape(T, self.n_head_kv, hd)
        # flint.cpp: per-head RMSNorm on q and k, then RoPE on flagged layers.
        q = self.rms_norm(q, self.f32(p + "attn_q_norm.weight"))
        k = self.rms_norm(k, self.f32(p + "attn_k_norm.weight"))
        pos = np.arange(T)
        if self.rope[il]:
            q, k = self.rope_neox(q, pos), self.rope_neox(k, pos)
        # build_attn_mha: kq = k.q * 1/sqrt(hd), masked; GQA maps q head h to
        # kv head h // (n_head / n_head_kv) (ggml_mul_mat broadcast).
        rep = self.n_head // self.n_head_kv
        kh = np.repeat(k, rep, axis=1)  # [T, H, hd]
        vh = np.repeat(v, rep, axis=1)
        kq = np.einsum("qhd,khd->hqk", q, kh) / np.sqrt(hd)
        p0, p1 = pos[None, :], pos[:, None]
        masked = p0 > p1  # causal
        if self.swa[il]:
            # llama_hparams::is_masked_swa, LLAMA_SWA_TYPE_STANDARD
            masked |= (p1 - p0) >= self.n_swa
        kq = np.where(masked[None], -np.inf, kq)
        # ggml_soft_max_add_sinks: the sink is one more logit per head in the
        # denominator (and the max), contributing no value.
        sink = self.f32(p + "attn_sinks.weight")[:, None, None]  # [H, 1, 1]
        m = np.maximum(kq.max(axis=-1, keepdims=True), sink)
        e = np.exp(kq - m)
        w = e / (e.sum(axis=-1, keepdims=True) + np.exp(sink - m))
        out = np.einsum("hqk,khd->qhd", w, vh).reshape(T, self.n_head * hd)
        return self.mm(p + "attn_output.weight", out)

    def swiglu(self, x, gate, up, down):
        # build_ffn(LLM_FFN_SILU, LLM_FFN_PAR): down(silu(gate(x)) * up(x)), each output scaled.
        g = self.mm(gate, x)
        return self.mm(down, g / (1.0 + np.exp(-g)) * self.mm(up, x))

    def moe(self, il, x):
        p = f"blk.{il}."
        T = x.shape[0]
        # build_moe_ffn: logits from the BF16 router (no scale), sigmoid gating.
        router = self.bf16_rows(p + "ffn_gate_inp.weight", slice(None))  # [n_expert, n_embd]
        probs = 1.0 / (1.0 + np.exp(-(x @ router.T)))
        sel = np.argsort(-probs, axis=-1, kind="stable")[:, : self.n_used]  # ggml_argsort_top_k
        wts = np.take_along_axis(probs, sel, axis=-1)
        # norm_w: divide by the sum clamped to the smallest f16 normal.
        wts = wts / np.maximum(wts.sum(axis=-1, keepdims=True), 6.103515625e-5)
        if self.w_scale not in (0.0, 1.0):
            wts = wts * self.w_scale
        gs = self.scale(p + "ffn_gate_exps.weight")
        us = self.scale(p + "ffn_up_exps.weight")
        ds = self.scale(p + "ffn_down_exps.weight")
        out = np.zeros_like(x)
        for e in np.unique(sel):
            tok, slot = np.nonzero(sel == e)
            xe = x[tok]
            # build_lora_mm_id: each expert's outputs times that expert's scale.
            g = (xe @ self.expert(p + "ffn_gate_exps.weight", e).T) * gs[e]
            u = (xe @ self.expert(p + "ffn_up_exps.weight", e).T) * us[e]
            h = g / (1.0 + np.exp(-g)) * u
            # The amax rescale around the down projection is a power-of-two
            # divide and multiply: the identity in exact arithmetic.
            d = (h @ self.expert(p + "ffn_down_exps.weight", e).T) * ds[e]
            out[tok] += d * wts[tok, slot][:, None]
        # flint.cpp: the shared expert is added to the routed sum.
        shexp = self.swiglu(x, p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight",
                            p + "ffn_down_shexp.weight")
        return out + shexp

    def logits(self, tokens, last):
        """Logits [last, n_vocab] at the final `last` positions of one causal pass."""
        x = self.bf16_rows("token_embd.weight", np.asarray(tokens))
        for il in range(self.n_layer):
            p = f"blk.{il}."
            h = x + self.attention(il, self.rms_norm(x, self.f32(p + "attn_norm.weight")))
            y = self.rms_norm(h, self.f32(p + "ffn_norm.weight"))
            if il < self.n_dense:
                f = self.swiglu(y, p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight")
            else:
                f = self.moe(il, y)
            x = h + f
        y = self.rms_norm(x[-last:], self.f32("output_norm.weight"))
        # lm_head in row chunks so the f64 head is never whole in memory.
        n_vocab = self.t["output.weight"].data.shape[0]
        out = np.empty((last, n_vocab))
        step = 16384
        for r0 in range(0, n_vocab, step):
            rows = slice(r0, min(r0 + step, n_vocab))
            out[:, rows] = y @ self.dense("output.weight", rows).T
        return out * self.scale("output.weight")


def top(logits, k):
    p = np.exp(logits - logits.max())
    p /= p.sum()
    idx = np.argsort(-p, kind="stable")[:k]
    return [[int(i), float(p[i])] for i in idx]


def main():
    out = pathlib.Path(sys.argv[1])
    model = Flint(os.path.expanduser(os.environ["GGUF"]))
    ref = json.loads((out / "flint_reference.json").read_text())
    by_name = {c["name"]: c for c in ref["cases"]}
    cases = []
    t0 = time.time()
    for name in CASES:
        c = by_name[name]
        forced = c["greedy"][:FORCED]
        lg = model.logits(c["tokens"] + forced, FORCED + 1)
        cases.append({"name": name, "tokens": c["tokens"], "forced": forced,
                      "top": [top(row, TOP) for row in lg]})
        print(f"{name}: {len(c['tokens'])} prompt tokens + {FORCED} forced, "
              f"{time.time() - t0:.0f} s", file=sys.stderr)
    # ru_maxrss is in bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"total {time.time() - t0:.0f} s, peak RSS {peak / 2**30:.2f} GiB", file=sys.stderr)
    doc = {"source": f"crates/flint-engine/tools/flint_numpy_ref.py: an independent NumPy "
                     f"float64 Flint forward pass from llama.cpp {COMMIT} semantics, weights "
                     "decoded by that commit's gguf-py; `top` holds the float64 softmax top "
                     f"{TOP} after the prompt and after each of `forced` (llama.cpp's first "
                     f"{FORCED} greedy tokens)",
           "file_sha256": ref["file_sha256"], "cases": cases}
    (out / "flint_numpy_reference.json").write_text(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
