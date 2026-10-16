#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-dpas.hpp"

// spir64_gen builds for sub-group 8 targets (Xe-HPG) or mixed target lists cannot compile the DPAS below
#if defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 16
#define GGML_SYCL_FA_DPAS_BUILD 0
#else
#define GGML_SYCL_FA_DPAS_BUILD 1
#endif

static constexpr int   FA_DPAS_WG    = 256;      // work-group size
static constexpr int   FA_DPAS_NSG   = 16;       // sub-groups of 16 lanes
static constexpr int   FA_DPAS_MAX_Q = 31;       // larger batches go to the prefill kernels
static constexpr int   FA_DPAS_MAX_G = 32;       // query heads per KV head
static constexpr float FA_DPAS_MINF  = -1.0e30f; // finite start value of the running max

static bool fattn_dpas_arch_ok(gpu_arch arch) {
    return arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31 ||
           arch == gpu_arch::intel_gpu_lnl_m   || arch == gpu_arch::intel_gpu_ptl_h   ||
           arch == gpu_arch::intel_gpu_ptl_u   || arch == gpu_arch::intel_gpu_wcl;
}

bool ggml_sycl_flash_attn_ext_dpas_supported(int device, const ggml_tensor * dst) {
#if !GGML_SYCL_FA_DPAS_BUILD
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!g_ggml_sycl_fa_dpas || !fattn_dpas_arch_ok(ggml_sycl_info().devices[device].hw_info.arch)) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0 || sinks) {
        return false;
    }
    if ((K->ne[0] != 128 && K->ne[0] != 256) || V->ne[0] != K->ne[0] || Q->ne[0] != K->ne[0]) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > FA_DPAS_MAX_Q || K->ne[1] < 1) {
        return false;
    }
    if (K->ne[2] < 1 || V->ne[2] != K->ne[2] || Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] > FA_DPAS_MAX_G) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[0] != sizeof(sycl::half))) {
        return false;
    }
    if (Q->nb[0] != sizeof(float)) {
        return false;
    }
    // q4_0 rows are read as dwords
    for (const ggml_tensor * t : { K, V }) {
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 4 != 0 || t->nb[2] % 4 != 0 || t->nb[3] % 4 != 0 ||
            (uintptr_t) t->data % 4 != 0) {
            return false;
        }
    }
    return true;
#endif
}

#if GGML_SYCL_FA_DPAS_BUILD

typedef short fa_dpas_short8 __attribute__((ext_vector_type(8)));
typedef int   fa_dpas_int4   __attribute__((ext_vector_type(4)));
typedef int   fa_dpas_int8   __attribute__((ext_vector_type(8)));
typedef float fa_dpas_float8 __attribute__((ext_vector_type(8)));

// Xe2 DPAS (SIMD16, systolic depth 8, repeat count 8) as inline vISA: joint_matrix has no s4 type.
// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s4, 16 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s4(const fa_dpas_short8 & a, const fa_dpas_int4 & b, fa_dpas_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s4.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// acc += a x b. a: 8 rows x 16 f16, one per lane per row. b: 16 x 16 f16 as pairs, 8 dwords per lane, one column per lane.
static inline void fa_dpas_f16(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(acc);
#endif
}

// q4_0 block b = 18 bytes at byte 18 b: f16 scale, then 16 bytes of nibbles.
// w = the 5 dwords from dword 18 b / 4. An odd block starts 2 bytes into w[0].
static inline void fa_dpas_split_block(const uint32_t * w, bool odd, uint32_t & sc, uint32_t * q) {
    if (!odd) {
        sc   = w[0] & 0xFFFFu;
        q[0] = (w[0] >> 16) | (w[1] << 16);
        q[1] = (w[1] >> 16) | (w[2] << 16);
        q[2] = (w[2] >> 16) | (w[3] << 16);
        q[3] = (w[3] >> 16) | (w[4] << 16);
    } else {
        sc   = w[0] >> 16;
        q[0] = w[1];
        q[1] = w[2];
        q[2] = w[3];
        q[3] = w[4];
    }
}

struct fa_dpas_params {
    const char *   Q;
    const char *   K;
    const char *   V;
    const char *   mask;
    float *        dst;
    float *        parts;  // per-split partial results, nsplit > 1 only
    sycl::float2 * meta;   // per-split (max, sum), nsplit > 1 only
    float          scale;
    int            G;      // query heads per KV head
    int            nq;     // query tokens per work-group
    int            nqc;    // token chunks
    int            ne01;
    int            ne02;
    int            ne11;
    int            ne33;
    int            nsplit;
    int            chunk;  // keys per split
    int64_t        nb01, nb02, nb03;
    int64_t        nb11, nb12, nb13;
    int64_t        nb21, nb22, nb23;
    int64_t        nb31, nb33;
};

// One work-group = one KV head x nq query tokens x one slice of the keys.
// Rows r = c * G + g: token c, query head g of the KV head. At most 16 * NR16 rows.
// Per tile of 16 * KG keys:
// - S = Q K^T: s8 x s4 DPAS on the raw q4_0 blocks, one key per lane. Q is quantized to s8 once per work-group.
// - online softmax per row, P stored as f16.
// - O^T += V^T P^T: f16 DPAS, V dequantized in registers. Sub-group w owns head dims D / 16 * w .. D / 16 * (w + 1) - 1.
// ST: copy each K and V tile to SLM with coalesced 16-byte loads first (single token decode, D 256).
// LG: 256-register file. The 32-row launch runs 2 work-groups x 16 sub-groups per Xe core, which is what that mode
// allows, so it costs no occupancy and removes spills. grf_size_automatic does not pick it for this kernel. Every arch in
// fattn_dpas_arch_ok() has the large register file.
template <int D, int NR16, int KG, int TC, bool ST, bool LG>
static void fattn_dpas_q4_0(const fa_dpas_params p, const int nkvh, const int ne03, dpct::queue_ptr stream) {
    constexpr int RMAX  = 16 * NR16;
    constexpr int RTMAX = RMAX / 8;                         // 8-row tiles in Q.K
    constexpr int TK    = 16 * KG;                          // keys per tile
    constexpr int NB    = D / QK4_0;                        // q4_0 blocks per row
    constexpr int DPS   = D / FA_DPAS_NSG;                  // head dims per sub-group in P.V
    constexpr int XT    = DPS / 8;                          // 8-dim tiles per sub-group in P.V
    constexpr int PS    = TK + 16;                          // f16 row stride of P
    constexpr int RW    = NB * (int) sizeof(block_q4_0) / 4; // dwords per q4_0 row
    constexpr int SW    = RW + 1;                           // odd SLM row stride: no bank conflicts for lane = key
    static_assert((D == 128 || D == 256) && DPS % 8 == 0 && QK4_0 % DPS == 0, "unsupported head size");
    static_assert(!ST || RW % 4 == 0, "staging needs rows of 16-byte multiples");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<fa_dpas_short8, 1> sQ8(sycl::range<1>(RTMAX * NB * 16), cgh); // Q s8 [tile][block][lane]
        sycl::local_accessor<fa_dpas_float8, 1> sDQ(sycl::range<1>(RTMAX * NB), cgh);      // Q scales [tile][block]
        sycl::local_accessor<float, 1>          sS(sycl::range<1>(RMAX * TK), cgh);        // S [row][key]
        sycl::local_accessor<fa_dpas_int8, 1>   sP(sycl::range<1>(RMAX * PS / 16), cgh);   // P f16 [row][key]
        sycl::local_accessor<float, 1>          sM(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sL(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sA(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<uint32_t, 1>       sKV(sycl::range<1>(ST ? TK * SW : 1), cgh);
        const sycl::nd_range<3> ndr(sycl::range<3>((size_t) ne03 * p.nqc, nkvh, (size_t) p.nsplit * FA_DPAS_WG),
                                    sycl::range<3>(1, 1, FA_DPAS_WG));
        auto kern = [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0) / p.nqc;
                const int c0    = (it.get_group(0) % p.nqc) * p.nq;
                const int G     = p.G;
                const int R     = p.nq * G;
                const int RT    = (R + 7) / 8;
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                short *   sQs   = (short *) &sQ8[0];
                short *   sPs   = (short *) &sP[0];

                // Q -> s8, one scale per row and block; lane l holds dims l and l + 16, the order of the q4_0 nibbles
                for (int e = tid; e < RT * 8 * NB; e += FA_DPAS_WG) {
                    const int  r    = e / NB;
                    const int  b    = e % NB;
                    const int  c    = c0 + r / G;
                    const bool real = r < R && c < p.ne01;
                    const float * qr = (const float *) (p.Q + seq * p.nb03 + (int64_t) (real ? c : 0) * p.nb01 +
                                                        (int64_t) (kvh * G + (real ? r % G : 0)) * p.nb02) + b * QK4_0;
                    float x[QK4_0];
                    float amax = 0.0f;
#pragma unroll
                    for (int i = 0; i < QK4_0; ++i) {
                        x[i] = real ? qr[i] * p.scale : 0.0f;
                        amax = sycl::fmax(amax, sycl::fabs(x[i]));
                    }
                    const float dq = amax / 127.0f;
                    const float iq = amax > 0.0f ? 127.0f / amax : 0.0f;
                    const int   t  = r / 8;
                    const int   m  = r % 8;
                    ((float *) &sDQ[0])[(t * NB + b) * 8 + m] = dq;
#pragma unroll
                    for (int l = 0; l < 16; ++l) {
                        const int lo = (int) sycl::rint(x[l] * iq);
                        const int hi = (int) sycl::rint(x[l + 16] * iq);
                        sQs[((t * NB + b) * 16 + l) * 8 + m] = (short) ((lo & 0xFF) | ((hi & 0xFF) << 8));
                    }
                }
                // rows without a query keep P = 0
                for (int e = tid; e < RMAX * PS; e += FA_DPAS_WG) {
                    sPs[e] = 0;
                }
                if (tid < RMAX) {
                    sM[tid] = FA_DPAS_MINF;
                    sL[tid] = 0.0f;
                    sA[tid] = 1.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int    k_begin = split * p.chunk;
                const int    k_end   = sycl::min(k_begin + p.chunk, p.ne11);
                const char * Kh      = p.K + seq * p.nb13 + kvh * p.nb12;
                const char * Vh      = p.V + seq * p.nb23 + kvh * p.nb22;
                const sycl::half * mrow = p.mask ? (const sycl::half *) (p.mask + (seq % p.ne33) * p.nb33) : nullptr;

                // O^T: lane = query row, element m = head dim
                fa_dpas_float8 o[NR16][XT];
#pragma unroll
                for (int t = 0; t < NR16; ++t) {
#pragma unroll
                    for (int x = 0; x < XT; ++x) {
                        o[t][x] = 0.0f;
                    }
                }
                // the dims of this sub-group lie in one V block
                const int vb  = (DPS * w) / QK4_0;
                const int vd0 = (DPS * w) % QK4_0;

                // keys past k_end repeat key k_begin, their scores are -inf
                auto stage = [&](const char * base, int64_t nb, int t0) {
                    constexpr int RV  = RW / 4;
                    constexpr int NV  = TK * RV;
                    constexpr int PER = (NV + FA_DPAS_WG - 1) / FA_DPAS_WG;
                    sycl::uint4 v[PER];
#pragma unroll
                    for (int i = 0; i < PER; ++i) {
                        const int e = tid + i * FA_DPAS_WG;
                        if (e < NV) {
                            const int key = e / RV;
                            const int q   = e - key * RV;
                            const int j   = t0 + key < k_end ? t0 + key : k_begin;
                            v[i] = ((const sycl::uint4 *) (base + (int64_t) j * nb))[q];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < PER; ++i) {
                        const int e = tid + i * FA_DPAS_WG;
                        if (e < NV) {
                            const int key = e / RV;
                            const int q   = e - key * RV;
                            const int o   = key * SW + 4 * q;
                            sKV[o + 0] = v[i][0];
                            sKV[o + 1] = v[i][1];
                            sKV[o + 2] = v[i][2];
                            sKV[o + 3] = v[i][3];
                        }
                    }
                };

                // the 5 dwords of q4_0 block b: of tile key kk in SLM, or of row in memory
                auto load_block = [&](const uint32_t * row, int kk, int b, uint32_t * wv) {
                    const int wd = 18 * b / 4;
                    if constexpr (ST) {
                        GGML_UNUSED(row);
#pragma unroll
                        for (int i = 0; i < 5; ++i) {
                            wv[i] = sKV[kk * SW + wd + i];
                        }
                    } else {
                        GGML_UNUSED(kk);
#pragma unroll
                        for (int i = 0; i < 5; ++i) {
                            wv[i] = row[wd + i];
                        }
                    }
                };

                // start loading the rows of keys t0 .. t0 + TK - 1 into L1 while the current phase computes
                auto prefetch_rows = [&](const char * base, int64_t nb, int t0) {
                    constexpr int NL = RW * 4 / 64 + 1; // cache lines per row
                    for (int e = tid; e < TK * NL; e += FA_DPAS_WG) {
                        const int j = t0 + e / NL;
                        if (j < k_end) {
                            char * pa = const_cast<char *>(base) + (int64_t) j * nb + (e % NL) * 64;
                            sycl::ext::oneapi::experimental::prefetch(pa, sycl::ext::oneapi::experimental::properties{ sycl::ext::oneapi::experimental::prefetch_hint_L1 });
                        }
                    }
                };

                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    if constexpr (ST) {
                        stage(Kh, p.nb11, t0);
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    prefetch_rows(Vh, p.nb21, t0);

                    // S = Q K^T, job = (key group, TC row tiles)
                    const int nch = (RT + TC - 1) / TC;
                    for (int job = w; job < KG * nch; job += FA_DPAS_NSG) {
                        const int  kg = job % KG;
                        const int  ch = job / KG;
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const uint32_t * kr = (const uint32_t *) (Kh + (int64_t) (jv ? j : k_begin) * p.nb11);
                        fa_dpas_float8 s[TC];
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            s[tt] = 0.0f;
                        }
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            uint32_t wv[5];
                            uint32_t sc;
                            uint32_t q[4];
                            load_block(kr, kg * 16 + lane, b, wv);
                            fa_dpas_split_block(wv, b % 2 != 0, sc, q);
                            const float dk = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc));
                            // u4 (n + 8) -> s4 n
                            const fa_dpas_int4 bv = { (int) (q[0] ^ 0x88888888u), (int) (q[1] ^ 0x88888888u),
                                                      (int) (q[2] ^ 0x88888888u), (int) (q[3] ^ 0x88888888u) };
#pragma unroll
                            for (int tt = 0; tt < TC; ++tt) {
                                const int t = ch * TC + tt;
                                if (t < RT) {
                                    fa_dpas_int8 ia;
                                    fa_dpas_s8_s4(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    const fa_dpas_float8 ds = sDQ[t * NB + b] * dk;
                                    s[tt] += __builtin_convertvector(ia, fa_dpas_float8) * ds;
                                }
                            }
                        }
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            const int t = ch * TC + tt;
                            if (t < RT) {
#pragma unroll
                                for (int m = 0; m < 8; ++m) {
                                    const int r = t * 8 + m;
                                    const int c = c0 + r / G;
                                    if (r < R) {
                                        const bool  ok = jv && c < p.ne01;
                                        const float mv = (ok && mrow) ? static_cast<float>(mrow[((int64_t) c * p.nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
                                        sS[r * TK + kg * 16 + lane] = ok ? s[tt][m] + mv : -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (ST) {
                        stage(Vh, p.nb21, t0); // the softmax does not use sKV
                    }

                    // online softmax per row -> P
                    for (int r = w; r < R; r += FA_DPAS_NSG) {
                        float sv[KG];
                        float mt = -INFINITY;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            sv[i] = sS[r * TK + i * 16 + lane];
                            mt    = sycl::fmax(mt, sv[i]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);
                        float       lt    = 0.0f;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            const float pv = sycl::native::exp(sv[i] - m_new);
                            sPs[r * PS + i * 16 + lane] = (short) sycl::bit_cast<uint16_t>(sycl::half(pv));
                            lt += pv;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    prefetch_rows(Kh, p.nb11, t0 + TK);

                    // O^T = diag(alpha) O^T + V^T P^T. A = V^T (8 dims x 16 keys, lane = key), B = P^T (16 keys x 16 rows)
#pragma unroll
                    for (int t = 0; t < NR16; ++t) {
                        const float a = sA[t * 16 + lane];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
                            o[t][x] *= a;
                        }
                    }
#pragma unroll 1
                    for (int kg = 0; kg < KG; ++kg) {
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const uint32_t * vr = (const uint32_t *) (Vh + (int64_t) (jv ? j : k_begin) * p.nb21);
                        uint32_t wv[5];
                        uint32_t sc;
                        uint32_t q[4];
                        load_block(vr, kg * 16 + lane, vb, wv);
                        fa_dpas_split_block(wv, vb % 2 != 0, sc, q);
                        const float dv = jv ? static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc)) : 0.0f;
                        fa_dpas_short8 va[XT];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d    = vd0 + x * 8 + m;
                                const int byte = d % 16;
                                const int nib  = (q[byte / 4] >> (8 * (byte % 4) + 4 * (d / 16))) & 0xF;
                                va[x][m] = (short) sycl::bit_cast<uint16_t>(sycl::half((float) (nib - 8) * dv));
                            }
                        }
#pragma unroll
                        for (int t = 0; t < NR16; ++t) {
                            const fa_dpas_int8 pb = sP[((t * 16 + lane) * PS + kg * 16) / 16];
#pragma unroll
                            for (int x = 0; x < XT; ++x) {
                                fa_dpas_f16(va[x], pb, o[t][x]);
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int t = 0; t < NR16; ++t) {
                    const int r = t * 16 + lane;
                    const int c = c0 + r / G;
                    if (r < R && c < p.ne01) {
                        const int64_t jdu = ((int64_t) seq * p.ne01 + c) * p.ne02 + kvh * G + r % G;
                        const float   l   = sL[r];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = w * DPS + x * 8 + m;
                                if (p.nsplit == 1) {
                                    p.dst[jdu * D + d] = l > 0.0f ? o[t][x][m] / l : 0.0f;
                                } else {
                                    p.parts[(jdu * p.nsplit + split) * D + d] = o[t][x][m];
                                }
                            }
                        }
                    }
                }
                if (p.nsplit > 1 && tid < R && c0 + tid / G < p.ne01) {
                    const int64_t jdu = ((int64_t) seq * p.ne01 + c0 + tid / G) * p.ne02 + kvh * G + tid % G;
                    p.meta[jdu * p.nsplit + split] = sycl::float2(sM[tid], sL[tid]);
                }
            };
        if constexpr (LG) {
            cgh.parallel_for(ndr, sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> }, kern);
        } else {
            cgh.parallel_for(ndr, kern);
        }
    });
}

template <int D, int NR16, int KG, int TC, bool ST>
static void fattn_dpas_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    constexpr int TK = 16 * KG;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int ne01 = Q->ne[1];
    const int ne11 = K->ne[1];
    const int nkvh = K->ne[2];
    const int ne03 = Q->ne[3];

    fa_dpas_params p;
    p.Q    = (const char *) Q->data;
    p.K    = (const char *) K->data;
    p.V    = (const char *) V->data;
    p.mask = mask ? (const char *) mask->data : nullptr;
    p.dst  = (float *) dst->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.G    = Q->ne[2] / K->ne[2];
    p.nq   = std::max(1, std::min(ne01, 16 * NR16 / p.G));
    p.nqc  = (ne01 + p.nq - 1) / p.nq;
    p.ne01 = ne01;
    p.ne02 = Q->ne[2];
    p.ne11 = ne11;
    p.ne33 = mask ? mask->ne[3] : 1;
    p.nb01 = Q->nb[1];
    p.nb02 = Q->nb[2];
    p.nb03 = Q->nb[3];
    p.nb11 = K->nb[1];
    p.nb12 = K->nb[2];
    p.nb13 = K->nb[3];
    p.nb21 = V->nb[1];
    p.nb22 = V->nb[2];
    p.nb23 = V->nb[3];
    p.nb31 = mask ? mask->nb[1] : 0;
    p.nb33 = mask ? mask->nb[3] : 0;

    // split the keys so that all work-groups run in one wave
    // SLM fits about 4 work-groups per Xe2 core with 16 rows, 2 with 32. Xe2 core = 8 EUs, nsm = EUs / 16.
    const int xe_cores = std::max(1, ggml_sycl_info().devices[ctx.device].nsm * 2);
    const int wave     = xe_cores * (NR16 == 1 ? 4 : 2);
    const int ntiles   = (ne11 + TK - 1) / TK;
    int nsplit = std::max(1, std::min(wave / (nkvh * ne03 * p.nqc), ntiles));
    p.chunk    = ((ntiles + nsplit - 1) / nsplit) * TK;
    nsplit     = (ne11 + p.chunk - 1) / p.chunk;
    p.nsplit   = nsplit;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit * ggml_nelements(dst));
        meta.alloc((size_t) nsplit * ggml_nrows(dst));
    }
    p.parts = parts.get();
    p.meta  = meta.get();

    dpct::queue_ptr stream = ctx.stream();
    // B580, us, 4 / 8 / 16 tokens: D 256 G 6 at 128K keys 1281 / 2460 / 6077 -> 1209 / 2300 / 5757,
    // D 128 G 4 (8 / 16 tokens) 2429 / 5483 -> 2098 / 4754. Below ~4K keys the launch overhead wins (512 keys: 16 -> 19).
    if (NR16 == 2 && ne11 >= 4096) {
        fattn_dpas_q4_0<D, NR16, KG, TC, ST, true>(p, nkvh, ne03, stream);
    } else {
        fattn_dpas_q4_0<D, NR16, KG, TC, ST, false>(p, nkvh, ne03, stream);
    }

    if (nsplit > 1) {
        float *        parts_p = parts.get();
        sycl::float2 * meta_p  = meta.get();
        float *        dst_p   = (float *) dst->data;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(nsplit * sizeof(sycl::float2)), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(Q->ne[3], Q->ne[2], (size_t) ne01 * D), sycl::range<3>(1, 1, D)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_combine_results<D>(parts_p, meta_p, dst_p, nsplit,
                                                  lm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int D    = K->ne[0];
    const int G    = Q->ne[2] / K->ne[2];
    const int ne01 = Q->ne[1];

    bool aligned16 = true;
    for (const ggml_tensor * t : { K, V }) {
        aligned16 = aligned16 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0 &&
                    (uintptr_t) t->data % 16 == 0;
    }

    // single token decode: staged K/V tiles are faster at long context (B580, 48K keys: 302 -> 247 us per layer)
    if (D == 256 && ne01 == 1 && G <= 16 && aligned16) {
        fattn_dpas_case<256, 1, 8, 1, true>(ctx, dst);
        return;
    }
    const bool rows16 = ne01 * G <= 16;
    if (D == 128) {
        if (rows16) {
            fattn_dpas_case<128, 1, 16, 1, false>(ctx, dst);
        } else {
            fattn_dpas_case<128, 2, 8, 2, false>(ctx, dst);
        }
    } else {
        if (rows16) {
            fattn_dpas_case<256, 1, 16, 1, false>(ctx, dst);
        } else {
            fattn_dpas_case<256, 2, 8, 2, false>(ctx, dst);
        }
    }
}

#else

void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("fattn-dpas: not built for this target");
}

#endif // GGML_SYCL_FA_DPAS_BUILD
