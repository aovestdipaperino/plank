// The single-token decode graph's own kernels (decode.rs encodes a whole
// token's forward into one encoder with these, the NVFP4 matvec and MoE
// kernels, and the fused attention and routing ones). Compiled after
// nvfp4.metal and fused.metal, whose includes and helpers it shares; the
// launch constants (SIMD_WIDTH, MV_ROWS_PER_SG, MV_SG_PER_TG, ATTN_HEAD_DIM,
// NORM_THREADS) come from metal.rs's prelude.

struct NormParams {
    uint n;
    float eps;
    uint add;   // nonzero: normalise a + b, and store that sum in sum
};

// One threadgroup: y = a (+ b) / sqrt(mean((a (+ b))^2) + eps) * w.
kernel void dec_rmsnorm(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *sum [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant NormParams &p [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float part[NORM_THREADS / SIMD_WIDTH];
    float ss = 0.0f;
    for (uint i = tid; i < p.n; i += NORM_THREADS) {
        float v = a[i];
        if (p.add != 0) {
            v += b[i];
            sum[i] = v;
        }
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        part[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = lane < NORM_THREADS / SIMD_WIDTH ? part[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) {
            part[0] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(part[0] / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += NORM_THREADS) {
        const float v = p.add != 0 ? sum[i] : a[i];
        y[i] = v * scale * w[i];
    }
}

struct Mv3Params {
    uint k;
    uint row_bytes;
    uint n0;
    uint n1;
    uint n2;
    float s0;
    float s1;
    float s2;
};

// Three NVFP4 matvecs over the same x in one dispatch, their outputs
// concatenated in y: rows [0, n0) of w0, then n1 of w1, then n2 of w2. Each
// segment's length is a multiple of MV_ROWS_PER_SG, so a simdgroup's rows
// never straddle two weights.
kernel void dec_mv3_nvfp4(
    device const uchar *w0 [[buffer(0)]],
    device const uchar *w1 [[buffer(1)]],
    device const uchar *w2 [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant Mv3Params &p [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = (tg.x * MV_SG_PER_TG + sg) * MV_ROWS_PER_SG;
    const uint total = p.n0 + p.n1 + p.n2;
    if (row0 >= total) {
        return;
    }
    device const uchar *w = w0;
    uint local = row0;
    uint end = p.n0;
    float scale = p.s0;
    if (row0 >= p.n0 + p.n1) {
        w = w2;
        local = row0 - p.n0 - p.n1;
        end = total;
        scale = p.s2;
    } else if (row0 >= p.n0) {
        w = w1;
        local = row0 - p.n0;
        end = p.n0 + p.n1;
        scale = p.s1;
    }
    const uint rows = min(uint(MV_ROWS_PER_SG), end - row0);
    float acc[MV_ROWS_PER_SG] = {};
    rows_dot(w + (ulong)local * p.row_bytes, p.row_bytes, rows, x, p.k, lane, acc);
    for (uint r = 0; r < rows; ++r) {
        const float a = simd_sum(acc[r]);
        if (lane == 0) {
            y[row0 + r] = a * scale;
        }
    }
}

struct QkParams {
    uint heads;
    uint kv_heads;
    uint rope;
    float eps;
    uint pos;           // absolute position: the RoPE table row
    uint slot;          // its index in the cache buffers
    uint k_head_stride; // values between KV heads in kc
    uint v_head_stride; // values between KV heads in vc
};

// From qkv = [q heads | k heads | v heads] of one token: each q and k head
// RMS-normalised by its weight and, when rope is set, rotated NeoX-style
// (pairs i and i + half) at pos; q to q_out, k and v into the cache at slot,
// f32 or f16 (a kernel each). One threadgroup per q, k and v head (in that
// order), one thread per value.
template <typename T>
inline void qk_norm_rope(
    device const float *qkv,
    device const float *qnw,
    device const float *knw,
    device const float *cos_t,
    device const float *sin_t,
    device float *q_out,
    device T *kc,
    device T *vc,
    constant QkParams &p,
    uint g,
    uint i,
    uint sg,
    uint lane,
    threadgroup float *part,
    threadgroup float *xs)
{
    const uint half_d = ATTN_HEAD_DIM / 2;
    if (g >= p.heads + p.kv_heads) {
        const uint h = g - p.heads - p.kv_heads;
        vc[h * p.v_head_stride + p.slot * ATTN_HEAD_DIM + i] =
            T(qkv[(p.heads + p.kv_heads + h) * ATTN_HEAD_DIM + i]);
        return;
    }
    const bool is_q = g < p.heads;
    const uint h = is_q ? g : g - p.heads;
    const float x = qkv[(is_q ? h : p.heads + h) * ATTN_HEAD_DIM + i];
    const float ss = simd_sum(x * x);
    if (lane == 0) {
        part[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < ATTN_HEAD_DIM / SIMD_WIDTH; ++s) {
        total += part[s];
    }
    float y = x * rsqrt(total / float(ATTN_HEAD_DIM) + p.eps) * (is_q ? qnw[i] : knw[i]);
    if (p.rope != 0) {
        xs[i] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint j = i % half_d;
        const float c = cos_t[p.pos * half_d + j];
        const float s = sin_t[p.pos * half_d + j];
        y = i < half_d ? xs[i] * c - xs[i + half_d] * s : xs[i] * c + xs[i - half_d] * s;
    }
    if (is_q) {
        q_out[h * ATTN_HEAD_DIM + i] = y;
    } else {
        kc[h * p.k_head_stride + p.slot * ATTN_HEAD_DIM + i] = T(y);
    }
}

#define QK_NORM_ROPE_KERNEL(NAME, T)                                          \
    kernel void NAME(                                                         \
        device const float *qkv [[buffer(0)]],                                \
        device const float *qnw [[buffer(1)]],                                \
        device const float *knw [[buffer(2)]],                                \
        device const float *cos_t [[buffer(3)]],                              \
        device const float *sin_t [[buffer(4)]],                              \
        device float *q_out [[buffer(5)]],                                    \
        device T *kc [[buffer(6)]],                                           \
        device T *vc [[buffer(7)]],                                           \
        constant QkParams &p [[buffer(8)]],                                   \
        uint g [[threadgroup_position_in_grid]],                              \
        uint i [[thread_position_in_threadgroup]],                            \
        uint sg [[simdgroup_index_in_threadgroup]],                           \
        uint lane [[thread_index_in_simdgroup]])                              \
    {                                                                         \
        threadgroup float part[ATTN_HEAD_DIM / SIMD_WIDTH];                   \
        threadgroup float xs[ATTN_HEAD_DIM];                                  \
        qk_norm_rope<T>(qkv, qnw, knw, cos_t, sin_t, q_out, kc, vc, p, g, i, \
                        sg, lane, part, xs);                                  \
    }

QK_NORM_ROPE_KERNEL(dec_qk_norm_rope_f32, float)
QK_NORM_ROPE_KERNEL(dec_qk_norm_rope_f16, half)

// y = W x for a dense f32 W [n, k] (k a multiple of 4), one simdgroup per row.
kernel void dec_mv_f32(
    device const float *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint2 &kn [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tg.x * MV_SG_PER_TG + sg;
    if (row >= kn.y) {
        return;
    }
    device const float4 *wr = (device const float4 *)(w + (ulong)row * kn.x);
    device const float4 *xv = (device const float4 *)x;
    float acc = 0.0f;
    for (uint c = lane; c < kn.x / 4; c += SIMD_WIDTH) {
        acc += dot(wr[c], xv[c]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = acc;
    }
}

struct CombineParams {
    uint d;
    uint slots;
    uint extra;
    float eps;
};

// The end of a layer, in one threadgroup: x = h + (the slots' rows of ys +
// e when extra is set), the candle path's order, then xn = rmsnorm(x) * w,
// the next layer's (or the output head's) normalised input. Each thread
// keeps its values of x in registers between the two passes.
kernel void dec_combine_norm(
    device const float *h [[buffer(0)]],
    device const float *ys [[buffer(1)]],
    device const float *e [[buffer(2)]],
    device float *x [[buffer(3)]],
    device const float *w [[buffer(4)]],
    device float *xn [[buffer(5)]],
    constant CombineParams &p [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float part[NORM_THREADS / SIMD_WIDTH];
    float vals[COMBINE_PER_THREAD];
    float ss = 0.0f;
    for (uint r = 0; r < COMBINE_PER_THREAD; ++r) {
        const uint i = tid + r * NORM_THREADS;
        float v = 0.0f;
        if (i < p.d) {
            float routed = 0.0f;
            for (uint s = 0; s < p.slots; ++s) {
                routed += ys[(ulong)s * p.d + i];
            }
            if (p.extra != 0) {
                routed += e[i];
            }
            v = h[i] + routed;
            x[i] = v;
        }
        vals[r] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        part[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = lane < NORM_THREADS / SIMD_WIDTH ? part[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) {
            part[0] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(part[0] / float(p.d) + p.eps);
    for (uint r = 0; r < COMBINE_PER_THREAD; ++r) {
        const uint i = tid + r * NORM_THREADS;
        if (i < p.d) {
            xn[i] = vals[r] * scale * w[i];
        }
    }
}
