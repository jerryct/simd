// SPDX-License-Identifier: MIT

#ifndef SIMD_H
#define SIMD_H

#if defined(__SSE4_2__) && defined(__linux__)
#include "detail/simd_sse_backend.h"
#else
#include "detail/simd_default_backend.h"
#endif

namespace parallelism_v2 {
template <typename T, typename Abi = simd_abi::compatible<T>> class simd;
template <typename T, int N> using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;
template <typename T, typename Abi = simd_abi::compatible<T>> class simd_mask;
template <typename T, int N> using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;
} // namespace parallelism_v2

#endif // SIMD_H
