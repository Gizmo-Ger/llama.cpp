//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_GEMM_HPP
#define GGML_SYCL_GEMM_HPP

#include "ggml-sycl.h"

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

class DnnlGemmWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
#ifdef GGML_SYCL_HAS_BF16
        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return dt::bf16;
#endif
        else static_assert(0);
    }

    static void gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        const void * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        void * c, dt ct, const queue_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        dnnl::memory::dims a_dims = {batches_a, m, k };
        dnnl::memory::dims a_strides = {stra2, stra1, stra0};
        const auto a_in_md = dnnl::memory::desc(a_dims, at, a_strides);

        dnnl::memory::dims b_dims = {batches_b, k, n };
        dnnl::memory::dims b_strides = {strb2, strb0, strb1};
        const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_strides);

        dnnl::memory::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::memory::dims c_strides = {m*n, 1,  m };
        const auto c_md    = dnnl::memory::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef GGML_SYCL_F16
        primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#endif

        auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
        auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        const auto scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        void * scratchpad_ptr = scratchpad_md.get_size() > 0 ? scratchpad.alloc(scratchpad_md.get_size()) : nullptr;
        auto scratchpad_mem = dnnl::memory(scratchpad_md, eng, scratchpad_ptr);

        auto matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_args.insert({ DNNL_ARG_DST, c_mem });
        matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_args);
    }

    // C[n][m] (f32, row stride m) = A[n][k] (f16) x W[k][m]: W = u4 values, m rows of k/2 contiguous bytes (tag ba),
    // zero point 8, one f16 scale per 32 k values in [k/32][m] order (q4_0 reorder layout, see ggml_sycl_q4_0_w4_prep)
    // q8 (q4 == false): W = s8 values, m rows of k contiguous bytes, no zero point
    static dnnl::matmul::primitive_desc q4_0_w4_pd(const dnnl::engine & eng, int64_t m, int64_t n, int64_t k, bool q4) {
        const auto a_md = dnnl::memory::desc({ n, k }, dt::f16, tag::ab);
        const auto w_md = dnnl::memory::desc({ k, m }, q4 ? dt::u4 : dt::s8, tag::ba);
        const auto c_md = dnnl::memory::desc({ n, m }, dt::f32, tag::ab);
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        // up to ~1024 weight rows oneDNN picks split-K kernels whose partial sums land in run-dependent order (last-bit
        // differences, e.g. Gemma 4's 512-row K projection); deterministic mode there keeps results reproducible.
        // Larger shapes are already bit-reproducible and some get slower kernels in deterministic mode.
        if (m <= 1024) {
            attr.set_deterministic(true);
        }
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { 32, 1 }, dt::f16);
        if (q4) {
            attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dt::u8);
        }
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
        return dnnl::matmul::primitive_desc(eng, a_md, w_md, c_md, attr);
    }

    // whether this oneDNN builds the weight-decompression matmul at all (older versions or other GPUs may not)
    static bool q4_0_w4_supported(ggml_backend_sycl_context & ctx, int64_t m, int64_t n, int64_t k, bool q4, const queue_ptr & q) {
        try {
            q4_0_w4_pd(ctx.engine_dnnl(q), m, n, k, q4);
            return true;
        } catch (const dnnl::error & e) {
            GGML_LOG_WARN("%s: oneDNN cannot build the %s weight-decompression GEMM (%s), using the regular path\n", __func__,
                          q4 ? "q4_0" : "q8_0", e.what());
            return false;
        }
    }

    static void q4_0_w4_gemm(ggml_backend_sycl_context & ctx, int64_t m, int64_t n, int64_t k, const void * w,
                             const sycl::half * scales, const sycl::half * a, float * c, bool q4, const queue_ptr & q) {
        auto stream = ctx.stream_dnnl(q);
        auto eng    = ctx.engine_dnnl(q);
        const auto a_md = dnnl::memory::desc({ n, k }, dt::f16, tag::ab);
        const auto w_md = dnnl::memory::desc({ k, m }, q4 ? dt::u4 : dt::s8, tag::ba);
        const auto c_md = dnnl::memory::desc({ n, m }, dt::f32, tag::ab);
        auto pd = q4_0_w4_pd(eng, m, n, k, q4);

        ggml_sycl_pool_alloc<uint8_t> zp(ctx.pool(), 1);
        q->memset(zp.get(), 8, 1);
        const auto scratchpad_md = pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        void * scratchpad_ptr = scratchpad_md.get_size() > 0 ? scratchpad.alloc(scratchpad_md.get_size()) : nullptr;

        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, dnnl::memory(a_md, eng, const_cast<sycl::half *>(a)) });
        args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(w_md, eng, const_cast<void *>(w)) });
        args.insert({ DNNL_ARG_DST, dnnl::memory(c_md, eng, c) });
        args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                      dnnl::memory(dnnl::memory::desc({ k / 32, m }, dt::f16, tag::ab), eng, const_cast<sycl::half *>(scales)) });
        if (q4) {
            args.insert({ DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(dnnl::memory::desc({ 1 }, dt::u8, tag::a), eng, zp.get()) });
        }
        args.insert({ DNNL_ARG_SCRATCHPAD, dnnl::memory(scratchpad_md, eng, scratchpad_ptr) });
        dnnl::matmul(pd).execute(stream, args);
    }

    static void row_gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, const void * b, dt bt, void * c, dt ct, const queue_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#endif

#endif // GGML_SYCL_GEMM_HPP
