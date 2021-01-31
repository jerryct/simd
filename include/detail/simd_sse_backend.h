// SPDX-License-Identifier: MIT

#ifndef DETAIL_SIMD_SSE_BACKEND_H
#define DETAIL_SIMD_SSE_BACKEND_H

#include "detail/simd_data_types.h"
#include <cstddef>
#include <cstdint>
#include <nmmintrin.h> // only include SSE4.2

namespace parallelism_v2 {
namespace detail {

template <typename T> struct sse_mask_intrinsics;

template <> struct sse_mask_intrinsics<float> {
  static __m128 broadcast(const bool v) noexcept {
    return _mm_castsi128_ps(_mm_set1_epi32(-static_cast<std::uint32_t>(v)));
  }

  static __m128 init(const bool w, const bool x, const bool y, const bool z) noexcept {
    return _mm_castsi128_ps(_mm_set_epi32(-static_cast<std::uint32_t>(z), -static_cast<std::uint32_t>(y),
                                          -static_cast<std::uint32_t>(x), -static_cast<std::uint32_t>(w)));
  };

  static bool extract(const __m128 v, const std::size_t i) noexcept { return _mm_movemask_ps(v) & (1 << i); }

  static __m128 logical_not(const __m128 v) noexcept { return _mm_cmpeq_ps(v, _mm_setzero_ps()); }
  static __m128 logical_and(const __m128 a, __m128 b) noexcept { return _mm_and_ps(a, b); }
  static __m128 logical_or(const __m128 a, const __m128 b) noexcept { return _mm_or_ps(a, b); }

  static bool all_of(const __m128 v) noexcept { return _mm_movemask_ps(v) == 0b1111; }
  static bool any_of(const __m128 v) noexcept { return _mm_movemask_ps(v) > 0; }
  static bool none_of(const __m128 v) noexcept { return _mm_movemask_ps(v) == 0; }
};

template <typename T> struct sse_intrinsics;

template <> struct sse_intrinsics<float> {
  static __m128 broadcast(const float v) noexcept { return _mm_set1_ps(v); }

  static __m128 init(const float w, const float x, const float y, const float z) noexcept {
    return _mm_set_ps(z, y, x, w);
  };

  static __m128 load(const float *const v) noexcept { return _mm_loadu_ps(v); }
  static __m128 load_aligned(const float *const v) noexcept { return _mm_load_ps(v); }
  static void store(float *const v, __m128 a) noexcept { _mm_storeu_ps(v, a); }
  static void store_aligned(float *const v, __m128 a) noexcept { _mm_store_ps(v, a); }

  static float extract(const __m128 v, const std::size_t i) noexcept {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    return tmp[i];
  }

  static __m128 add(const __m128 a, const __m128 b) noexcept { return _mm_add_ps(a, b); }
  static __m128 subtract(const __m128 a, const __m128 b) noexcept { return _mm_sub_ps(a, b); }
  static __m128 multiply(const __m128 a, const __m128 b) noexcept { return _mm_mul_ps(a, b); }
  static __m128 divide(const __m128 a, const __m128 b) noexcept { return _mm_div_ps(a, b); }
  static __m128 negate(const __m128 v) noexcept { return _mm_xor_ps(v, _mm_set1_ps(-0.0F)); }

  static __m128 equal(const __m128 a, const __m128 b) noexcept { return _mm_cmpeq_ps(a, b); }
  static __m128 not_equal(const __m128 a, const __m128 b) noexcept { return _mm_cmpneq_ps(a, b); }
  static __m128 less_than(const __m128 a, const __m128 b) noexcept { return _mm_cmplt_ps(a, b); }
  static __m128 less_equal(const __m128 a, const __m128 b) noexcept { return _mm_cmple_ps(a, b); }
  static __m128 greater_than(const __m128 a, const __m128 b) noexcept { return _mm_cmpgt_ps(a, b); }
  static __m128 greater_equal(const __m128 a, const __m128 b) noexcept { return _mm_cmpge_ps(a, b); }

  static __m128 min(const __m128 a, const __m128 b) noexcept { return _mm_min_ps(b, a); }
  static __m128 max(const __m128 a, const __m128 b) noexcept { return _mm_max_ps(b, a); }

  static __m128 is_nan(const __m128 v) noexcept { return _mm_cmpunord_ps(v, v); }

  static __m128 blend(const __m128 a, const __m128 b, const __m128 c) noexcept { return _mm_blendv_ps(a, b, c); }
};

template <typename T> struct sse_type;
template <> struct sse_type<float> {
  using storage_type = __m128;
  using mask_type = __m128;
  static constexpr std::size_t width{4U};
};

struct sse {
  template <typename T> using storage_type = typename sse_type<T>::storage_type;
  template <typename T> using mask_storage_type = typename sse_type<T>::mask_type;
  template <typename T> static constexpr std::size_t simd_size{sse_type<T>::width};
  template <typename T> using impl = sse_intrinsics<T>;
  template <typename T> using mask_impl = sse_mask_intrinsics<T>;
};

} // namespace detail

namespace simd_abi {
template <int N> using fixed_size = detail::sse;
template <typename T> using compatible = detail::sse;
} // namespace simd_abi

template <> struct is_abi_tag<detail::sse> : std::integral_constant<bool, true> {};
template <> struct is_simd<simd<float, detail::sse>> : std::integral_constant<bool, true> {};
template <> struct is_simd_mask<simd_mask<float, detail::sse>> : std::integral_constant<bool, true> {};

} // namespace parallelism_v2

#endif // DETAIL_SIMD_SSE_BACKEND_H
