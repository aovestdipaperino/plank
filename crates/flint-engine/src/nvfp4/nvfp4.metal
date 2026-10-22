// NVFP4 kernels for flint-engine. The block decode mirrors upstream ggml
// (llama.cpp 970b51c): 36-byte blocks of 64 values, four UE4M3 scales
// (halved on decode) then 32 bytes of codes into the doubled E2M1 table.
// Activations and accumulation are f32. Every kernel multiplies its output
// by the weight's tensor scale (per expert for the MoE kernels).

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

#define MV_ROWS_PER_SG 4
#define MV_SG_PER_TG 2

// One token per threadgroup row of the grid; each simdgroup computes
// MV_ROWS_PER_SG rows, its 32 lanes splitting the row's 16-value runs.
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
        for (uint sb = lane; sb < nsub; sb += 32) {
            acc += sub_dot(wr, sb, xt);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            y[(ulong)t * p.n + row] = acc * p.scale;
        }
    }
}
