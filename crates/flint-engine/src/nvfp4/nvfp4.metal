// NVFP4 kernels for flint-engine. The block decode mirrors upstream ggml
// (llama.cpp 970b51c): 36-byte blocks of 64 values, four UE4M3 scales
// (halved on decode) then 32 bytes of codes into the doubled E2M1 table.
// Activations and accumulation are f32. Every kernel multiplies its output
// by the weight's tensor scale (per expert for the MoE kernels).
//
// The launch constants (SIMD_WIDTH, MV_ROWS_PER_SG, MV_SG_PER_TG, MM_TN,
// MM_TT, MM_THREADS, SUM_THREADS) are not defined here: metal.rs prepends
// them from the Rust constants its dispatches use.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant float KVALUES[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float ue4m3_to_f32(uchar x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int e = (x >> 3) & 0xF;
    float m = float(x & 0x7);
    float raw = e == 0 ? ldexp(m, -9) : ldexp(1.0f + m / 8.0f, e - 7);
    return raw * 0.5f;
}

// Dot of sub-block `sb` (16 values) of one row with the matching x values.
inline float sub_dot(device const uchar *row, uint sb, device const float *x) {
    uint b = sb >> 2;
    uint s = sb & 3;
    device const uchar *blk = row + b * 36;
    device const uchar *qs = blk + 4 + s * 8;
    device const float *xs = x + b * 64 + s * 16;
    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        uchar q = qs[j];
        acc += KVALUES[q & 0xF] * xs[j] + KVALUES[q >> 4] * xs[j + 8];
    }
    return acc * ue4m3_to_f32(blk[s]);
}

struct MatParams {
    uint k;          // columns (input width)
    uint n;          // rows (output width)
    uint tokens;
    uint row_bytes;
    float scale;
};

// One token per threadgroup row of the grid; each simdgroup computes
// MV_ROWS_PER_SG rows, its SIMD_WIDTH lanes splitting the row's 16-value runs.
kernel void mv_nvfp4(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant MatParams &p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint t = tg.y;
    uint row0 = (tg.x * MV_SG_PER_TG + sg) * MV_ROWS_PER_SG;
    device const float *xt = x + (ulong)t * p.k;
    uint nsub = p.k / 16;
    for (uint r = 0; r < MV_ROWS_PER_SG; ++r) {
        uint row = row0 + r;
        if (row >= p.n) {
            break; // uniform across the simdgroup
        }
        device const uchar *wr = w + (ulong)row * p.row_bytes;
        float acc = 0.0f;
        for (uint sb = lane; sb < nsub; sb += SIMD_WIDTH) {
            acc += sub_dot(wr, sb, xt);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            y[(ulong)t * p.n + row] = acc * p.scale;
        }
    }
}

// Each thread loads one 16-value run (of four per block) of one row and one
// token, so the threads cover the tile's MM_TN rows and MM_TT tokens.
static_assert(MM_THREADS == 4 * MM_TN && MM_TN == MM_TT, "mm_nvfp4 thread mapping");
static_assert(MM_TN == 32, "mm_nvfp4 uses a 2x2 grid of 16x16 simdgroup patches");

// MM_THREADS threads = 4 simdgroups in a 2x2 arrangement, each owning a
// 16x16 patch of the MM_TN x MM_TT (rows x tokens) output tile as 2x2 8x8
// f32 matrices. Each step decodes one 64-value block column of MM_TN weight
// rows into threadgroup memory next to the matching MM_TT tokens of x.
kernel void mm_nvfp4(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant MatParams &p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float ws[MM_TN * 64];
    threadgroup float xs[MM_TT * 64];
    threadgroup float cs[MM_TT * MM_TN];
    uint n0 = tg.x * MM_TN;
    uint t0 = tg.y * MM_TT;
    uint sn = (sg & 1) * 16;
    uint st = (sg >> 1) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    uint r = tid / 4; // 0..MM_TN-1: the weight row / token this thread loads
    uint s = tid % 4; // 0..3: which 16-value run
    uint nb = p.k / 64;
    for (uint b = 0; b < nb; ++b) {
        threadgroup float *wd = ws + r * 64 + s * 16;
        uint row = n0 + r;
        if (row < p.n) {
            device const uchar *blk = w + (ulong)row * p.row_bytes + b * 36;
            float d = ue4m3_to_f32(blk[s]);
            device const uchar *qs = blk + 4 + s * 8;
            for (uint j = 0; j < 8; ++j) {
                uchar q = qs[j];
                wd[j] = KVALUES[q & 0xF] * d;
                wd[j + 8] = KVALUES[q >> 4] * d;
            }
        } else {
            for (uint j = 0; j < 16; ++j) {
                wd[j] = 0.0f;
            }
        }
        threadgroup float *xd = xs + r * 64 + s * 16;
        uint t = t0 + r;
        if (t < p.tokens) {
            device const float *src = x + (ulong)t * p.k + b * 64 + s * 16;
            for (uint j = 0; j < 16; ++j) {
                xd[j] = src[j];
            }
        } else {
            for (uint j = 0; j < 16; ++j) {
                xd[j] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_float8x8 a[2];
            simdgroup_float8x8 bm[2];
            for (uint i = 0; i < 2; ++i) {
                // A[n][k] straight from ws.
                simdgroup_load(a[i], ws + (sn + i * 8) * 64 + kk, 64);
            }
            for (uint j = 0; j < 2; ++j) {
                // xs holds [t][k]; transposed it is B[k][t].
                simdgroup_load(bm[j], xs + (st + j * 8) * 64 + kk, 64, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 2; ++i) {
                for (uint j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], bm[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // acc[i][j] is C[n][t]; store it transposed so cs is [t][n].
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            simdgroup_store(acc[i][j], cs + (st + j * 8) * MM_TN + (sn + i * 8), MM_TN, ulong2(0, 0), true);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < MM_TT * MM_TN; e += MM_THREADS) {
        uint tt = e / MM_TN;
        uint nn = e % MM_TN;
        uint t = t0 + tt;
        uint n = n0 + nn;
        if (t < p.tokens && n < p.n) {
            y[(ulong)t * p.n + n] = cs[e] * p.scale;
        }
    }
}

struct MoeParams {
    uint d;                 // model width (gate/up columns, down rows)
    uint f;                 // expert FFN width (gate/up rows, down columns)
    uint slots;             // tokens * k_used
    uint k_used;
    uint experts;
    uint gu_row_bytes;
    uint gu_expert_bytes;
    uint down_row_bytes;
    uint down_expert_bytes;
};

// h[slot][r] = silu(gs[e] * gate_e[r].x) * (us[e] * up_e[r].x), e = ids[slot].
kernel void moe_gate_up(
    device const uchar *wg [[buffer(0)]],
    device const uchar *wu [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uint *ids [[buffer(3)]],
    device const float *gs [[buffer(4)]],
    device const float *us [[buffer(5)]],
    device float *h [[buffer(6)]],
    constant MoeParams &p [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint slot = tg.y;
    uint e = ids[slot];
    uint row0 = (tg.x * MV_SG_PER_TG + sg) * MV_ROWS_PER_SG;
    device float *hs = h + (ulong)slot * p.f;
    if (e >= p.experts) {
        // Out-of-range slot: contributes nothing (the CPU path errors instead).
        if (lane == 0) {
            for (uint r = 0; r < MV_ROWS_PER_SG && row0 + r < p.f; ++r) {
                hs[row0 + r] = 0.0f;
            }
        }
        return;
    }
    device const float *xt = x + (ulong)(slot / p.k_used) * p.d;
    ulong base = (ulong)e * p.gu_expert_bytes;
    uint nsub = p.d / 16;
    for (uint r = 0; r < MV_ROWS_PER_SG; ++r) {
        uint row = row0 + r;
        if (row >= p.f) {
            break;
        }
        device const uchar *rg = wg + base + (ulong)row * p.gu_row_bytes;
        device const uchar *ru = wu + base + (ulong)row * p.gu_row_bytes;
        float g = 0.0f;
        float u = 0.0f;
        for (uint sb = lane; sb < nsub; sb += SIMD_WIDTH) {
            g += sub_dot(rg, sb, xt);
            u += sub_dot(ru, sb, xt);
        }
        g = simd_sum(g) * gs[e];
        u = simd_sum(u) * us[e];
        if (lane == 0) {
            hs[row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

// ys[slot][r] = (ds[e] * down_e[r].h[slot]) * route[slot].
kernel void moe_down(
    device const uchar *wd [[buffer(0)]],
    device const float *h [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device const float *ds [[buffer(3)]],
    device const float *route [[buffer(4)]],
    device float *ys [[buffer(5)]],
    constant MoeParams &p [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint slot = tg.y;
    uint e = ids[slot];
    uint row0 = (tg.x * MV_SG_PER_TG + sg) * MV_ROWS_PER_SG;
    device float *yo = ys + (ulong)slot * p.d;
    if (e >= p.experts) {
        if (lane == 0) {
            for (uint r = 0; r < MV_ROWS_PER_SG && row0 + r < p.d; ++r) {
                yo[row0 + r] = 0.0f;
            }
        }
        return;
    }
    device const float *hs = h + (ulong)slot * p.f;
    ulong base = (ulong)e * p.down_expert_bytes;
    uint nsub = p.f / 16;
    for (uint r = 0; r < MV_ROWS_PER_SG; ++r) {
        uint row = row0 + r;
        if (row >= p.d) {
            break;
        }
        device const uchar *rw = wd + base + (ulong)row * p.down_row_bytes;
        float acc = 0.0f;
        for (uint sb = lane; sb < nsub; sb += SIMD_WIDTH) {
            acc += sub_dot(rw, sb, hs);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            yo[row] = acc * ds[e] * route[slot];
        }
    }
}

// out[t][i] = sum over j in slot order of ys[t * k_used + j][i].
kernel void moe_sum(
    device const float *ys [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant MoeParams &p [[buffer(2)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint t = tg.y;
    uint i = tg.x * SUM_THREADS + tid;
    if (i >= p.d) {
        return;
    }
    float acc = 0.0f;
    for (uint j = 0; j < p.k_used; ++j) {
        acc += ys[((ulong)t * p.k_used + j) * p.d + i];
    }
    out[(ulong)t * p.d + i] = acc;
}
