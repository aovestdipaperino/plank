// The MoE for many tokens, grouped by expert. A slot is one (token, pick)
// pair, slot = token * k_used + pick, as moe_gate_up and moe_down number
// them; these kernels read and write per slot in the same layout, so
// moe_sum adds them up unchanged.
//
// moe_group sorts the slots by expert in one threadgroup and lists tiles of
// up to MM_TT slots of one expert; moe_mm_gate_up and moe_mm_down then run
// one tiled matmul per (MM_TN weight rows, tile), reading each expert's
// weights once per tile instead of once per slot. Compiled after
// nvfp4.metal, whose helpers and launch constants it uses (plus
// GROUP_THREADS from metal.rs's prelude).

// moe_group's outputs: perm lists the slots grouped by expert; tile t covers
// perm[tiles[t].y .. + tiles[t].z), all routed to expert tiles[t].x; n_tiles
// holds the number of tiles. A slot routed to no valid expert gets a zero
// row in ys and no tile.
kernel void moe_group(
    device const uint *ids [[buffer(0)]],
    device uint *perm [[buffer(1)]],
    device uint3 *tiles [[buffer(2)]],
    device uint *n_tiles [[buffer(3)]],
    device float *ys [[buffer(4)]],
    constant MoeParams &p [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint counts[GROUP_MAX_EXPERTS];
    threadgroup uint offsets[GROUP_MAX_EXPERTS];
    threadgroup uint tile0[GROUP_MAX_EXPERTS];
    threadgroup atomic_uint cursor[GROUP_MAX_EXPERTS];
    for (uint e = tid; e < p.experts; e += GROUP_THREADS) {
        atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
        atomic_store_explicit(&cursor[e], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tid; s < p.slots; s += GROUP_THREADS) {
        const uint e = ids[s];
        if (e < p.experts) {
            atomic_fetch_add_explicit(&counts[e], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint off = 0;
        uint t = 0;
        for (uint e = 0; e < p.experts; ++e) {
            const uint c = atomic_load_explicit(&counts[e], memory_order_relaxed);
            offsets[e] = off;
            tile0[e] = t;
            off += c;
            t += (c + MM_TT - 1) / MM_TT;
        }
        n_tiles[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.experts; e += GROUP_THREADS) {
        const uint c = atomic_load_explicit(&counts[e], memory_order_relaxed);
        for (uint i = 0; i * MM_TT < c; ++i) {
            tiles[tile0[e] + i] = uint3(e, offsets[e] + i * MM_TT, min(uint(MM_TT), c - i * MM_TT));
        }
    }
    for (uint s = tid; s < p.slots; s += GROUP_THREADS) {
        const uint e = ids[s];
        if (e < p.experts) {
            const uint at = atomic_fetch_add_explicit(&cursor[e], 1, memory_order_relaxed);
            perm[offsets[e] + at] = s;
        } else {
            // An out-of-range slot contributes nothing, as in moe_down.
            for (uint i = 0; i < p.d; ++i) {
                ys[(ulong)s * p.d + i] = 0.0f;
            }
        }
    }
}

// Decodes one 64-value block column of MM_TN weight rows (rows n0.. of the
// expert at `w`) into ws, zero past `rows`; each thread one 16-value run.
inline void load_weight_block(
    device const uchar *w,
    uint row_bytes,
    uint n0,
    uint rows,
    uint b,
    uint r,
    uint s,
    threadgroup float *ws)
{
    threadgroup float *wd = ws + r * 64 + s * 16;
    const uint row = n0 + r;
    if (row < rows) {
        device const uchar *blk = w + (ulong)row * row_bytes + b * 36;
        const float d = ue4m3_to_f32(blk[s]);
        device const uchar *qs = blk + 4 + s * 8;
        for (uint j = 0; j < 8; ++j) {
            const uchar q = qs[j];
            wd[j] = KVALUES[q & 0xF] * d;
            wd[j + 8] = KVALUES[q >> 4] * d;
        }
    } else {
        for (uint j = 0; j < 16; ++j) {
            wd[j] = 0.0f;
        }
    }
}

// acc[i][j] += W[n][k] X[t][k]^T over one 64-wide block in ws and xs.
inline void mma_block(
    threadgroup const float *ws,
    threadgroup const float *xs,
    uint sn,
    uint st,
    thread simdgroup_float8x8 (&acc)[2][2])
{
    for (uint kk = 0; kk < 64; kk += 8) {
        simdgroup_float8x8 a[2];
        simdgroup_float8x8 bm[2];
        for (uint i = 0; i < 2; ++i) {
            simdgroup_load(a[i], ws + (sn + i * 8) * 64 + kk, 64);
        }
        for (uint j = 0; j < 2; ++j) {
            simdgroup_load(bm[j], xs + (st + j * 8) * 64 + kk, 64, ulong2(0, 0), true);
        }
        for (uint i = 0; i < 2; ++i) {
            for (uint j = 0; j < 2; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], a[i], bm[j], acc[i][j]);
            }
        }
    }
}

// h[slot][row] = silu(gs[e] gate_e[row] . x[token]) * (us[e] up_e[row] . x[token])
// for MM_TN rows of one tile's expert and up to MM_TT of its slots.
kernel void moe_mm_gate_up(
    device const uchar *wg [[buffer(0)]],
    device const uchar *wu [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uint *perm [[buffer(3)]],
    device const uint3 *tiles [[buffer(4)]],
    device const uint *n_tiles [[buffer(5)]],
    device const float *gs [[buffer(6)]],
    device const float *us [[buffer(7)]],
    device float *h [[buffer(8)]],
    constant MoeParams &p [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    if (tg.y >= n_tiles[0]) {
        return;
    }
    threadgroup float wsg[MM_TN * 64];
    threadgroup float wsu[MM_TN * 64];
    threadgroup float xs[MM_TT * 64];
    const uint3 tile = tiles[tg.y];
    const uint e = tile.x;
    const uint n0 = tg.x * MM_TN;
    const uint sn = (sg & 1) * 16;
    const uint st = (sg >> 1) * 16;
    simdgroup_float8x8 ag[2][2];
    simdgroup_float8x8 au[2][2];
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            ag[i][j] = simdgroup_float8x8(0.0f);
            au[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    const uint r = tid / 4;
    const uint s = tid % 4;
    const ulong base = (ulong)e * p.gu_expert_bytes;
    // The token this thread gathers x for: tile slot r.
    const bool live = r < tile.z;
    const uint token = live ? perm[tile.y + r] / p.k_used : 0;
    for (uint b = 0; b < p.d / 64; ++b) {
        load_weight_block(wg + base, p.gu_row_bytes, n0, p.f, b, r, s, wsg);
        load_weight_block(wu + base, p.gu_row_bytes, n0, p.f, b, r, s, wsu);
        threadgroup float *xd = xs + r * 64 + s * 16;
        if (live) {
            device const float *src = x + (ulong)token * p.d + b * 64 + s * 16;
            for (uint j = 0; j < 16; ++j) {
                xd[j] = src[j];
            }
        } else {
            for (uint j = 0; j < 16; ++j) {
                xd[j] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_block(wsg, xs, sn, st, ag);
        mma_block(wsu, xs, sn, st, au);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The weight buffers are free now: stage C[t][n] for gate and up there.
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            simdgroup_store(ag[i][j], wsg + (st + j * 8) * MM_TN + (sn + i * 8), MM_TN, ulong2(0, 0), true);
            simdgroup_store(au[i][j], wsu + (st + j * 8) * MM_TN + (sn + i * 8), MM_TN, ulong2(0, 0), true);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < MM_TT * MM_TN; c += MM_THREADS) {
        const uint tt = c / MM_TN;
        const uint nn = c % MM_TN;
        const uint n = n0 + nn;
        if (tt < tile.z && n < p.f) {
            const float g = wsg[c] * gs[e];
            const float u = wsu[c] * us[e];
            h[(ulong)perm[tile.y + tt] * p.f + n] = g / (1.0f + exp(-g)) * u;
        }
    }
}

// ys[slot][row] = route[slot] * ds[e] * down_e[row] . h[slot] for MM_TN rows of
// one tile's expert and up to MM_TT of its slots.
kernel void moe_mm_down(
    device const uchar *wd [[buffer(0)]],
    device const float *h [[buffer(1)]],
    device const uint *perm [[buffer(2)]],
    device const uint3 *tiles [[buffer(3)]],
    device const uint *n_tiles [[buffer(4)]],
    device const float *ds [[buffer(5)]],
    device const float *route [[buffer(6)]],
    device float *ys [[buffer(7)]],
    constant MoeParams &p [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    if (tg.y >= n_tiles[0]) {
        return;
    }
    threadgroup float ws[MM_TN * 64];
    threadgroup float xs[MM_TT * 64];
    const uint3 tile = tiles[tg.y];
    const uint e = tile.x;
    const uint n0 = tg.x * MM_TN;
    const uint sn = (sg & 1) * 16;
    const uint st = (sg >> 1) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    const uint r = tid / 4;
    const uint s = tid % 4;
    const ulong base = (ulong)e * p.down_expert_bytes;
    const bool live = r < tile.z;
    const uint slot = live ? perm[tile.y + r] : 0;
    for (uint b = 0; b < p.f / 64; ++b) {
        load_weight_block(wd + base, p.down_row_bytes, n0, p.d, b, r, s, ws);
        threadgroup float *xd = xs + r * 64 + s * 16;
        if (live) {
            device const float *src = h + (ulong)slot * p.f + b * 64 + s * 16;
            for (uint j = 0; j < 16; ++j) {
                xd[j] = src[j];
            }
        } else {
            for (uint j = 0; j < 16; ++j) {
                xd[j] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_block(ws, xs, sn, st, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            simdgroup_store(acc[i][j], ws + (st + j * 8) * MM_TN + (sn + i * 8), MM_TN, ulong2(0, 0), true);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < MM_TT * MM_TN; c += MM_THREADS) {
        const uint tt = c / MM_TN;
        const uint nn = c % MM_TN;
        const uint n = n0 + nn;
        if (tt < tile.z && n < p.d) {
            const uint sl = perm[tile.y + tt];
            ys[(ulong)sl * p.d + n] = ws[c] * ds[e] * route[sl];
        }
    }
}
