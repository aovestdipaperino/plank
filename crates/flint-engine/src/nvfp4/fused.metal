// Fused decode kernels that replace chains of candle ops: single-token
// attention, and the MoE router.
//
// Single-token attention ("flash decoding") is split over runs of keys.
// attn_dec_partial: one threadgroup per (run, KV head) scores every query
// head that shares the KV head with an online softmax over the run and
// writes the run's per-head max, exp-sum and unnormalised output.
// attn_dec_combine: one threadgroup per query head merges the runs and the
// head's sink logit, which takes softmax mass but adds no value.
//
// K and V are read in place from the KV cache, f32 or f16 (a kernel each): head h's key j is at
// k[h * k_head_stride + j * ATTN_HEAD_DIM]. ATTN_HEAD_DIM is four values per
// lane of one simdgroup. The launch constants (SIMD_WIDTH, ATTN_HEAD_DIM,
// ATTN_BLOCK, ATTN_SG, ROUTE_MAX_K, ROUTE_THREADS, ROUTE_PER_THREAD) come
// from metal.rs's prelude, and the file is compiled after nvfp4.metal, whose
// includes it shares.

struct AttnParams {
    uint keys;          // keys attended
    uint chunks;        // threadgroups per KV head: ceil(keys / per_tg)
    uint rep;           // query heads per KV head, at most ATTN_SG
    uint k_head_stride; // values between KV heads in k
    uint v_head_stride; // values between KV heads in v
    float scale;
    uint per_tg;        // keys per threadgroup, a multiple of ATTN_BLOCK
};

// One threadgroup per (run of per_tg keys, KV head); simdgroup h owns query
// head h of the KV head. The run is walked in blocks of ATTN_BLOCK keys, one
// key per lane: each lane scores its key against the head's q, the block's
// max and exp-sum fold into a running (online) softmax, and the values are
// accumulated with each lane holding four dimensions, every key's
// probability broadcast from its lane. Writes the run's per-head max,
// exp-sum and unnormalised output.
template <typename T>
inline void attn_partial(
    device const float *q,
    device const T *k,
    device const T *v,
    device float *part_o,
    device float *part_ml,
    constant AttnParams &p,
    uint2 tg,
    uint sg,
    uint lane,
    threadgroup float *qs)
{
    const uint c = tg.x;
    const uint kvh = tg.y;
    const uint tid = sg * SIMD_WIDTH + lane;
    for (uint i = tid; i < p.rep * ATTN_HEAD_DIM; i += ATTN_SG * SIMD_WIDTH) {
        qs[i] = q[kvh * p.rep * ATTN_HEAD_DIM + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg >= p.rep) {
        return;
    }
    const uint begin = c * p.per_tg;
    const uint end = min(p.keys, begin + p.per_tg);
    device const T *kh = k + (ulong)kvh * p.k_head_stride;
    device const T *vh = v + (ulong)kvh * p.v_head_stride;
    threadgroup const float4 *qh = (threadgroup const float4 *)(qs + sg * ATTN_HEAD_DIM);
    float m = -INFINITY;
    float l = 0.0f;
    float4 acc = 0.0f;
    for (uint b = begin; b < end; b += ATTN_BLOCK) {
        const uint j = b + lane;
        float s = -INFINITY;
        if (j < end) {
            device const vec<T, 4> *kj = (device const vec<T, 4> *)(kh + (ulong)j * ATTN_HEAD_DIM);
            float d = 0.0f;
            for (uint i = 0; i < ATTN_HEAD_DIM / 4; ++i) {
                d += dot(qh[i], float4(kj[i]));
            }
            s = d * p.scale;
        }
        const float m_new = max(m, simd_max(s));
        const float corr = exp(m - m_new);
        const float pj = exp(s - m_new);
        l = l * corr + simd_sum(pj);
        acc *= corr;
        const uint n = min(uint(ATTN_BLOCK), end - b);
        device const vec<T, 4> *vb = (device const vec<T, 4> *)(vh + (ulong)b * ATTN_HEAD_DIM) + lane;
        if (n == ATTN_BLOCK) {
            // A full block: four independent accumulators, so the loads and
            // multiply-adds of different keys overlap.
            float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (uint t = 0; t < ATTN_BLOCK; t += 4) {
                a0 += simd_shuffle(pj, t) * float4(vb[(t + 0) * (ATTN_HEAD_DIM / 4)]);
                a1 += simd_shuffle(pj, t + 1) * float4(vb[(t + 1) * (ATTN_HEAD_DIM / 4)]);
                a2 += simd_shuffle(pj, t + 2) * float4(vb[(t + 2) * (ATTN_HEAD_DIM / 4)]);
                a3 += simd_shuffle(pj, t + 3) * float4(vb[(t + 3) * (ATTN_HEAD_DIM / 4)]);
            }
            acc += (a0 + a1) + (a2 + a3);
        } else {
            for (uint t = 0; t < n; ++t) {
                acc += simd_shuffle(pj, t) * float4(vb[t * (ATTN_HEAD_DIM / 4)]);
            }
        }
        m = m_new;
    }
    const uint slot = (kvh * p.chunks + c) * p.rep + sg;
    ((device float4 *)(part_o + slot * ATTN_HEAD_DIM))[lane] = acc;
    if (lane == 0) {
        part_ml[slot * 2] = m;
        part_ml[slot * 2 + 1] = l;
    }
}

kernel void attn_dec_partial_f32(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *part_o [[buffer(3)]],
    device float *part_ml [[buffer(4)]],
    constant AttnParams &p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float qs[ATTN_SG * ATTN_HEAD_DIM];
    attn_partial<float>(q, k, v, part_o, part_ml, p, tg, sg, lane, qs);
}

kernel void attn_dec_partial_f16(
    device const float *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device float *part_o [[buffer(3)]],
    device float *part_ml [[buffer(4)]],
    constant AttnParams &p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float qs[ATTN_SG * ATTN_HEAD_DIM];
    attn_partial<half>(q, k, v, part_o, part_ml, p, tg, sg, lane, qs);
}

// One threadgroup per query head: COMBINE_GROUPS groups of ATTN_HEAD_DIM
// threads, one per output dimension. All threads first find the max over
// every run (and the sink) together; then group g sums every
// COMBINE_GROUPS-th run's rescaled output and exp-sum, and the groups' sums
// are added through threadgroup memory.
kernel void attn_dec_combine(
    device const float *part_o [[buffer(0)]],
    device const float *part_ml [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant AttnParams &p [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[COMBINE_GROUPS * ATTN_HEAD_DIM / SIMD_WIDTH];
    threadgroup float accs[COMBINE_GROUPS * ATTN_HEAD_DIM];
    threadgroup float ls[COMBINE_GROUPS];
    const uint threads = COMBINE_GROUPS * ATTN_HEAD_DIM;
    const uint kvh = head / p.rep;
    const uint h = head % p.rep;
    const float sink = sinks[head];
    float m = sink;
    for (uint c = tid; c < p.chunks; c += threads) {
        m = max(m, part_ml[((kvh * p.chunks + c) * p.rep + h) * 2]);
    }
    m = simd_max(m);
    if (lane == 0) {
        red[sg] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = sink;
    for (uint s = 0; s < threads / SIMD_WIDTH; ++s) {
        m = max(m, red[s]);
    }
    const uint g = tid / ATTN_HEAD_DIM;
    const uint d = tid % ATTN_HEAD_DIM;
    float l = 0.0f;
    float acc = 0.0f;
    for (uint c = g; c < p.chunks; c += COMBINE_GROUPS) {
        const uint slot = (kvh * p.chunks + c) * p.rep + h;
        const float w = exp(part_ml[slot * 2] - m);
        l += part_ml[slot * 2 + 1] * w;
        acc += part_o[slot * ATTN_HEAD_DIM + d] * w;
    }
    accs[tid] = acc;
    if (d == 0) {
        ls[g] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g == 0) {
        float total_l = exp(sink - m);
        float total = 0.0f;
        for (uint k = 0; k < COMBINE_GROUPS; ++k) {
            total_l += ls[k];
            total += accs[k * ATTN_HEAD_DIM + d];
        }
        out[head * ATTN_HEAD_DIM + d] = total / total_l;
    }
}

// Sigmoid routing, one threadgroup per token, each thread holding up to
// ROUTE_PER_THREAD experts' probabilities: k rounds of a max reduction pick
// the most probable expert not yet taken (ties to the lower index). The
// weights are the picked probabilities, divided by their sum floored at
// min_sum when norm is set, then times scale (the host passes 1 when no
// scale applies). k is at most ROUTE_MAX_K.
struct RouteParams {
    uint experts;
    uint k;
    uint tokens;
    uint norm;
    float scale;
    float min_sum;
};

kernel void moe_route(
    device const float *logits [[buffer(0)]],
    device uint *ids [[buffer(1)]],
    device float *weights [[buffer(2)]],
    constant RouteParams &p [[buffer(3)]],
    uint t [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sg_best[ROUTE_THREADS / SIMD_WIDTH];
    threadgroup uint sg_at[ROUTE_THREADS / SIMD_WIDTH];
    threadgroup float picked[ROUTE_MAX_K];
    threadgroup uint picked_at[ROUTE_MAX_K];
    device const float *row = logits + t * p.experts;
    float pr[ROUTE_PER_THREAD];
    for (uint r = 0; r < ROUTE_PER_THREAD; ++r) {
        const uint e = tid + r * ROUTE_THREADS;
        pr[r] = e < p.experts ? 1.0f / (1.0f + exp(-row[e])) : -INFINITY;
    }
    for (uint i = 0; i < p.k; ++i) {
        float v = -INFINITY;
        uint at = UINT_MAX;
        for (uint r = 0; r < ROUTE_PER_THREAD; ++r) {
            if (pr[r] > v) {
                v = pr[r];
                at = tid + r * ROUTE_THREADS;
            }
        }
        float m = simd_max(v);
        uint a = simd_min(v == m ? at : UINT_MAX);
        if (lane == 0) {
            sg_best[sg] = m;
            sg_at[sg] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const bool live = lane < ROUTE_THREADS / SIMD_WIDTH;
            const float bv = live ? sg_best[lane] : -INFINITY;
            const uint ba = live ? sg_at[lane] : UINT_MAX;
            m = simd_max(bv);
            a = simd_min(bv == m ? ba : UINT_MAX);
            if (lane == 0) {
                picked[i] = m;
                picked_at[i] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint chosen = picked_at[i];
        if (chosen % ROUTE_THREADS == tid) {
            pr[chosen / ROUTE_THREADS] = -INFINITY;
        }
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < p.k; ++i) {
            sum += picked[i];
        }
        const float div = p.norm != 0 ? max(sum, p.min_sum) : 1.0f;
        for (uint i = 0; i < p.k; ++i) {
            ids[t * p.k + i] = picked_at[i];
            weights[t * p.k + i] = picked[i] / div * p.scale;
        }
    }
}

// Attention for many queries (prefill), FlashAttention-style on simdgroup
// matrices. One threadgroup per (PF_Q queries, query head), four simdgroups
// of 8 query rows each. Q and the output stay in registers as 8x8 tiles; the
// keys the block can see are walked in PF_K-key blocks staged as f32 in
// threadgroup memory (first K, then V in the same buffer). Masking is from
// positions: a key is visible when it is at or before the query and, with a
// window, fewer than `window` positions behind it. The sink starts each
// row's online softmax as one more logit (m = sink, l = 1) that adds no
// value. q and out are [heads][t_pad][ATTN_HEAD_DIM]; rows past t are
// masked out entirely and come out zero.
struct PrefillParams {
    uint t;              // queries
    uint t_pad;          // rows per head in q and out
    uint q_pos0;         // absolute position of query 0
    uint key_base;       // absolute position of cache index 0
    uint window;         // 0: none
    uint rep;            // query heads per KV head
    uint kv_head_stride; // values between KV heads in k and v
    float scale;
};

template <typename T>
inline void attn_prefill_impl(
    device const float *q,
    device const T *k,
    device const T *v,
    device const float *sinks,
    device float *out,
    constant PrefillParams &p,
    uint2 tg,
    uint tid,
    uint sg,
    uint lane,
    threadgroup float *kv,
    threadgroup float *scr)
{
    const uint D = ATTN_HEAD_DIM;
    const uint h = tg.y;
    const uint kvh = h / p.rep;
    const uint row0 = tg.x * PF_Q + sg * 8;
    simdgroup_float8x8 qt[ATTN_HEAD_DIM / 8];
    simdgroup_float8x8 o[ATTN_HEAD_DIM / 8];
    device const float *qh = q + ((ulong)h * p.t_pad + row0) * D;
    for (uint i = 0; i < D / 8; ++i) {
        simdgroup_load(qt[i], qh + i * 8, D);
        o[i] = simdgroup_float8x8(0.0f);
    }
    const float sink = sinks[h];
    float m[8];
    float l[8];
    for (uint r = 0; r < 8; ++r) {
        m[r] = sink;
        l[r] = 1.0f;
    }
    // The keys any query of this threadgroup can see.
    const uint qmin = p.q_pos0 + tg.x * PF_Q;
    const uint hi = min(p.q_pos0 + p.t, qmin + PF_Q);
    uint lo = p.key_base;
    if (p.window != 0 && qmin + 1 > p.window) {
        lo = max(lo, qmin + 1 - p.window);
    }
    device const T *kh = k + (ulong)kvh * p.kv_head_stride;
    device const T *vh = v + (ulong)kvh * p.kv_head_stride;
    threadgroup float *sc = scr + sg * 8 * PF_K;
    threadgroup float *dg = scr + 4 * 8 * PF_K + sg * 64;
    for (uint kb = lo; kb < hi; kb += PF_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < PF_K * D; i += PF_THREADS) {
            const uint kp = kb + i / D;
            kv[i] = kp < hi ? float(kh[(ulong)(kp - p.key_base) * D + i % D]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for 8 rows and PF_K keys.
        for (uint j = 0; j < PF_K / 8; ++j) {
            simdgroup_float8x8 s = simdgroup_float8x8(0.0f);
            for (uint i = 0; i < D / 8; ++i) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, kv + (j * 8) * D + i * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qt[i], kt, s);
            }
            simdgroup_store(s, sc + j * 8, PF_K);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // The online softmax, one key per lane.
        const uint kp = kb + lane;
        float corr[8];
        for (uint r = 0; r < 8; ++r) {
            const uint qi = row0 + r;
            const uint qp = p.q_pos0 + qi;
            const bool seen = qi < p.t && kp < hi && kp <= qp && (p.window == 0 || qp - kp < p.window);
            const float sv = seen ? sc[r * PF_K + lane] * p.scale : -INFINITY;
            const float mn = max(m[r], simd_max(sv));
            corr[r] = exp(m[r] - mn);
            const float pv = exp(sv - mn);
            l[r] = l[r] * corr[r] + simd_sum(pv);
            m[r] = mn;
            sc[r * PF_K + lane] = pv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < PF_K * D; i += PF_THREADS) {
            const uint kp2 = kb + i / D;
            kv[i] = kp2 < hi ? float(vh[(ulong)(kp2 - p.key_base) * D + i % D]) : 0.0f;
        }
        // The rows' rescale as a diagonal matrix.
        for (uint e = lane; e < 64; e += SIMD_WIDTH) {
            dg[e] = e % 9 == 0 ? corr[e / 9] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm;
        simdgroup_load(dm, dg, 8);
        simdgroup_float8x8 pm[PF_K / 8];
        for (uint j = 0; j < PF_K / 8; ++j) {
            simdgroup_load(pm[j], sc + j * 8, PF_K);
        }
        for (uint i = 0; i < D / 8; ++i) {
            simdgroup_multiply(o[i], dm, o[i]);
            for (uint j = 0; j < PF_K / 8; ++j) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, kv + (j * 8) * D + i * 8, D);
                simdgroup_multiply_accumulate(o[i], pm[j], vt, o[i]);
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64; e += SIMD_WIDTH) {
        dg[e] = e % 9 == 0 ? 1.0f / l[e / 9] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, dg, 8);
    device float *oh = out + ((ulong)h * p.t_pad + row0) * D;
    for (uint i = 0; i < D / 8; ++i) {
        simdgroup_multiply(o[i], dm, o[i]);
        simdgroup_store(o[i], oh + i * 8, D);
    }
}

#define ATTN_PREFILL_KERNEL(NAME, T)                                           \
    kernel void NAME(                                                          \
        device const float *q [[buffer(0)]],                                   \
        device const T *k [[buffer(1)]],                                       \
        device const T *v [[buffer(2)]],                                       \
        device const float *sinks [[buffer(3)]],                               \
        device float *out [[buffer(4)]],                                       \
        constant PrefillParams &p [[buffer(5)]],                               \
        uint2 tg [[threadgroup_position_in_grid]],                             \
        uint tid [[thread_index_in_threadgroup]],                              \
        uint sg [[simdgroup_index_in_threadgroup]],                            \
        uint lane [[thread_index_in_simdgroup]])                               \
    {                                                                          \
        threadgroup float kv[PF_K * ATTN_HEAD_DIM];                            \
        threadgroup float scr[4 * 8 * PF_K + 4 * 64];                          \
        attn_prefill_impl<T>(q, k, v, sinks, out, p, tg, tid, sg, lane, kv, scr); \
    }

ATTN_PREFILL_KERNEL(attn_prefill_f32, float)
ATTN_PREFILL_KERNEL(attn_prefill_f16, half)
