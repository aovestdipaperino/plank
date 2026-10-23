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
