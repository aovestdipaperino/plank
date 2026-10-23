// The MoE for many tokens, grouped by expert. A slot is one (token, pick)
// pair, slot = token * k_used + pick, as moe_gate_up and moe_down number
// them; these kernels read and write per slot in the same layout, so
// moe_sum adds them up unchanged.
//
// moe_group sorts the slots by expert in one threadgroup and lists tiles of
// up to MOE_TT slots of one expert; moe_mm_gate_up and moe_mm_down then run
// one tiled matmul per (MOE_GU_TN or MOE_DN_TN weight rows, tile), reading
// each expert's weights once per tile instead of once per slot. A prefill
// chunk routes about a dozen slots to each expert, so tiles are 16 slots,
// not the dense matmul's 32. Compiled after
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
            t += (c + MOE_TT - 1) / MOE_TT;
        }
        n_tiles[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.experts; e += GROUP_THREADS) {
        const uint c = atomic_load_explicit(&counts[e], memory_order_relaxed);
        for (uint i = 0; i * MOE_TT < c; ++i) {
            tiles[tile0[e] + i] = uint3(e, offsets[e] + i * MOE_TT, min(uint(MOE_TT), c - i * MOE_TT));
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

// Decodes one 64-value block column of weight row n0 + r (of the expert at
// `w`) into ws row r, zero past `rows`: this thread's 16-value run s.
inline void load_weight_run(
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

// Gathers this thread's 16-value run s of block b of slot r's input row
// (row index `src`, of width k) into xs, zero for a dead slot.
inline void load_input_run(
    device const float *x,
    uint src,
    bool live,
    uint k,
    uint b,
    uint r,
    uint s,
    threadgroup float *xs)
{
    threadgroup float *xd = xs + r * 64 + s * 16;
    if (live) {
        device const float *row = x + (ulong)src * k + b * 64 + s * 16;
        for (uint j = 0; j < 16; ++j) {
            xd[j] = row[j];
        }
    } else {
        for (uint j = 0; j < 16; ++j) {
            xd[j] = 0.0f;
        }
    }
}

// h[slot][row] = silu(gs[e] gate_e[row] . x[token]) * (us[e] up_e[row] . x[token])
// for MOE_GU_TN rows of one tile's expert and up to MOE_TT of its slots. Each
// simdgroup computes an 8-row strip across the MOE_TT slots, for gate and up.
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
    threadgroup float wsg[MOE_GU_TN * 64];
    threadgroup float wsu[MOE_GU_TN * 64];
    threadgroup float xs[MOE_TT * 64];
    const uint3 tile = tiles[tg.y];
    const uint e = tile.x;
    const uint n0 = tg.x * MOE_GU_TN;
    const uint sn = sg * 8;
    simdgroup_float8x8 ag[MOE_TT / 8];
    simdgroup_float8x8 au[MOE_TT / 8];
    for (uint j = 0; j < MOE_TT / 8; ++j) {
        ag[j] = simdgroup_float8x8(0.0f);
        au[j] = simdgroup_float8x8(0.0f);
    }
    const uint r = tid / 4;
    const uint s = tid % 4;
    const ulong base = (ulong)e * p.gu_expert_bytes;
    const bool loads_x = r < MOE_TT;
    const bool live = loads_x && r < tile.z;
    const uint token = live ? perm[tile.y + r] / p.k_used : 0;
    for (uint b = 0; b < p.d / 64; ++b) {
        load_weight_run(wg + base, p.gu_row_bytes, n0, p.f, b, r, s, wsg);
        load_weight_run(wu + base, p.gu_row_bytes, n0, p.f, b, r, s, wsu);
        if (loads_x) {
            load_input_run(x, token, live, p.d, b, r, s, xs);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_float8x8 a_g;
            simdgroup_float8x8 a_u;
            simdgroup_load(a_g, wsg + sn * 64 + kk, 64);
            simdgroup_load(a_u, wsu + sn * 64 + kk, 64);
            for (uint j = 0; j < MOE_TT / 8; ++j) {
                simdgroup_float8x8 bm;
                simdgroup_load(bm, xs + (j * 8) * 64 + kk, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(ag[j], a_g, bm, ag[j]);
                simdgroup_multiply_accumulate(au[j], a_u, bm, au[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The weight buffers are free now: stage C[t][n] for gate and up there.
    for (uint j = 0; j < MOE_TT / 8; ++j) {
        simdgroup_store(ag[j], wsg + (j * 8) * MOE_GU_TN + sn, MOE_GU_TN, ulong2(0, 0), true);
        simdgroup_store(au[j], wsu + (j * 8) * MOE_GU_TN + sn, MOE_GU_TN, ulong2(0, 0), true);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < MOE_TT * MOE_GU_TN; c += MM_THREADS) {
        const uint tt = c / MOE_GU_TN;
        const uint n = n0 + c % MOE_GU_TN;
        if (tt < tile.z && n < p.f) {
            const float g = wsg[c] * gs[e];
            const float u = wsu[c] * us[e];
            h[(ulong)perm[tile.y + tt] * p.f + n] = g / (1.0f + exp(-g)) * u;
        }
    }
}

// ys[slot][row] = route[slot] * ds[e] * down_e[row] . h[slot] for MOE_DN_TN
// rows of one tile's expert and up to MOE_TT of its slots. Each simdgroup
// computes a 16-row strip across the MOE_TT slots.
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
    threadgroup float ws[MOE_DN_TN * 64];
    threadgroup float xs[MOE_TT * 64];
    const uint3 tile = tiles[tg.y];
    const uint e = tile.x;
    const uint n0 = tg.x * MOE_DN_TN;
    const uint sn = sg * 16;
    simdgroup_float8x8 acc[2][MOE_TT / 8];
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < MOE_TT / 8; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    const uint r = tid / 4;
    const uint s = tid % 4;
    const ulong base = (ulong)e * p.down_expert_bytes;
    const bool loads_x = r < MOE_TT;
    const bool live = loads_x && r < tile.z;
    const uint slot = live ? perm[tile.y + r] : 0;
    for (uint b = 0; b < p.f / 64; ++b) {
        for (uint rr = r; rr < MOE_DN_TN; rr += MM_THREADS / 4) {
            load_weight_run(wd + base, p.down_row_bytes, n0, p.d, b, rr, s, ws);
        }
        if (loads_x) {
            load_input_run(h, slot, live, p.f, b, r, s, xs);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_float8x8 a[2];
            for (uint i = 0; i < 2; ++i) {
                simdgroup_load(a[i], ws + (sn + i * 8) * 64 + kk, 64);
            }
            for (uint j = 0; j < MOE_TT / 8; ++j) {
                simdgroup_float8x8 bm;
                simdgroup_load(bm, xs + (j * 8) * 64 + kk, 64, ulong2(0, 0), true);
                for (uint i = 0; i < 2; ++i) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], bm, acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < MOE_TT / 8; ++j) {
            simdgroup_store(acc[i][j], ws + (j * 8) * MOE_DN_TN + (sn + i * 8), MOE_DN_TN, ulong2(0, 0), true);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < MOE_TT * MOE_DN_TN; c += MM_THREADS) {
        const uint tt = c / MOE_DN_TN;
        const uint n = n0 + c % MOE_DN_TN;
        if (tt < tile.z && n < p.d) {
            const uint sl = perm[tile.y + tt];
            ys[(ulong)sl * p.d + n] = ws[c] * ds[e] * route[sl];
        }
    }
}
