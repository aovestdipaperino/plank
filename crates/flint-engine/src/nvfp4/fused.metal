// Fused decode kernels that replace chains of candle ops: single-token
// attention, and the MoE router.
//
// Single-token attention ("flash decoding") is split over key chunks.
// attn_dec_partial: one threadgroup per (key chunk, KV head) scores every
// query head that shares the KV head, reading each key and value once, and
// writes the chunk's per-head max, exp-sum and unnormalised output.
// attn_dec_combine: one threadgroup per query head merges the chunks and the
// head's sink logit, which takes softmax mass but adds no value.
//
// K and V are read in place from the KV cache: head h's key j is at
// k[h * k_head_stride + j * ATTN_HEAD_DIM]. ATTN_HEAD_DIM is four values per
// lane of one simdgroup. The launch constants (SIMD_WIDTH, ATTN_HEAD_DIM,
// ATTN_CHUNK, ATTN_SG, ROUTE_MAX_K, ROUTE_THREADS, ROUTE_PER_THREAD) come
// from metal.rs's prelude, and the file is compiled after nvfp4.metal, whose
// includes it shares.

struct AttnParams {
    uint keys;          // keys attended
    uint chunks;        // ceil(keys / ATTN_CHUNK)
    uint rep;           // query heads per KV head, at most ATTN_SG
    uint k_head_stride; // values between KV heads in k
    uint v_head_stride; // values between KV heads in v
    float scale;
};

kernel void attn_dec_partial(
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
    threadgroup float sc[ATTN_SG * ATTN_CHUNK];
    const uint c = tg.x;
    const uint kvh = tg.y;
    const uint tid = sg * SIMD_WIDTH + lane;
    for (uint i = tid; i < p.rep * ATTN_HEAD_DIM; i += ATTN_SG * SIMD_WIDTH) {
        qs[i] = q[kvh * p.rep * ATTN_HEAD_DIM + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = c * ATTN_CHUNK;
    const uint n = min(uint(ATTN_CHUNK), p.keys - k0);
    device const float *kh = k + kvh * p.k_head_stride + k0 * ATTN_HEAD_DIM;
    device const float *vh = v + kvh * p.v_head_stride + k0 * ATTN_HEAD_DIM;

    // Scores: each simdgroup takes every ATTN_SG-th key of the chunk, its
    // lanes holding four dimensions each.
    for (uint j = sg; j < n; j += ATTN_SG) {
        const float4 kj = ((device const float4 *)(kh + j * ATTN_HEAD_DIM))[lane];
        for (uint h = 0; h < p.rep; ++h) {
            const float4 qh = ((threadgroup const float4 *)(qs + h * ATTN_HEAD_DIM))[lane];
            const float s = simd_sum(dot(qh, kj));
            if (lane == 0) {
                sc[h * ATTN_CHUNK + j] = s * p.scale;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup h owns query head h: the chunk's max, the exps in place,
    // their sum, then the exp-weighted sum of the values.
    if (sg < p.rep) {
        threadgroup float *s = sc + sg * ATTN_CHUNK;
        float m = -INFINITY;
        for (uint j = lane; j < n; j += SIMD_WIDTH) {
            m = max(m, s[j]);
        }
        m = simd_max(m);
        float l = 0.0f;
        for (uint j = lane; j < n; j += SIMD_WIDTH) {
            const float e = exp(s[j] - m);
            s[j] = e;
            l += e;
        }
        l = simd_sum(l);
        // Every lane reads every exp below, written by other lanes above.
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float4 acc = 0.0f;
        for (uint j = 0; j < n; ++j) {
            acc += s[j] * ((device const float4 *)(vh + j * ATTN_HEAD_DIM))[lane];
        }
        const uint slot = (kvh * p.chunks + c) * p.rep + sg;
        ((device float4 *)(part_o + slot * ATTN_HEAD_DIM))[lane] = acc;
        if (lane == 0) {
            part_ml[slot * 2] = m;
            part_ml[slot * 2 + 1] = l;
        }
    }
}

// One threadgroup per query head, one thread per output dimension.
kernel void attn_dec_combine(
    device const float *part_o [[buffer(0)]],
    device const float *part_ml [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant AttnParams &p [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint d [[thread_position_in_threadgroup]])
{
    const uint kvh = head / p.rep;
    const uint h = head % p.rep;
    const float sink = sinks[head];
    float m = sink;
    for (uint c = 0; c < p.chunks; ++c) {
        m = max(m, part_ml[((kvh * p.chunks + c) * p.rep + h) * 2]);
    }
    float l = exp(sink - m);
    float acc = 0.0f;
    for (uint c = 0; c < p.chunks; ++c) {
        const uint slot = (kvh * p.chunks + c) * p.rep + h;
        const float w = exp(part_ml[slot * 2] - m);
        l += part_ml[slot * 2 + 1] * w;
        acc += part_o[slot * ATTN_HEAD_DIM + d] * w;
    }
    out[head * ATTN_HEAD_DIM + d] = acc / l;
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
