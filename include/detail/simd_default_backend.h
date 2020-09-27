// SPDX-License-Identifier: MIT

#ifndef DETAIL_SIMD_DEFAULT_BACKEND_H
#define DETAIL_SIMD_DEFAULT_BACKEND_H

#include "detail/simd_data_types.h"
#include <algorithm>
#include <cstddef>

namespace parallelism_v2 {
namespace detail {

template <typename T, int N> struct simd_vector { alignas(N * sizeof(T)) T v[N]; };

template <int N> struct simd_default_mask_impl {
  static simd_vector<bool, N> broadcast(const bool v) noexcept { return {v, v, v, v}; }

  static simd_vector<bool, N> init(const bool w, const bool x, const bool y, const bool z) noexcept {
    static_assert(N == 4U, "size mismatch");
    return {w, x, y, z};
  };

  static bool extract(const simd_vector<bool, N> &v, const size_t i) noexcept { return v.v[i]; }

  static simd_vector<bool, N> logical_not(const simd_vector<bool, N> &v) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = !v.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> logical_and(const simd_vector<bool, N> &a, const simd_vector<bool, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] && b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> logical_or(const simd_vector<bool, N> &a, const simd_vector<bool, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] || b.v[i];
    }
    return r;
  }

  static bool all_of(const simd_vector<bool, N> &v) noexcept {
    for (int i{}; i < N; ++i) {
      if (!v.v[i]) {
        return false;
      }
    }
    return true;
  }

  static bool any_of(const simd_vector<bool, N> &v) noexcept {
    for (int i{}; i < N; ++i) {
      if (v.v[i]) {
        return true;
      }
    }
    return false;
  }

  static bool none_of(const simd_vector<bool, N> &v) noexcept {
    for (int i{}; i < N; ++i) {
      if (v.v[i]) {
        return false;
      }
    }
    return true;
  }
};

template <typename T, int N> struct simd_default_impl {
  static simd_vector<T, N> broadcast(const T v) noexcept { return {v, v, v, v}; }

  static simd_vector<T, N> init(const T w, const T x, const T y, const T z) noexcept {
    static_assert(N == 4U, "size mismatch");
    return {w, x, y, z};
  };

  static simd_vector<T, N> load(const T *const v) {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = v[i];
    }
    return r;
  }

  static simd_vector<T, N> load_aligned(const T *const v) { return load(v); }

  static void store(T *const v, const simd_vector<T, N> &a) {
    for (int i = 0; i < N; ++i) {
      v[i] = a.v[i];
    }
  }

  static void store_aligned(T *const v, const simd_vector<T, N> &a) { store(v, a); }

  static T extract(const simd_vector<T, N> &v, const size_t i) noexcept { return v.v[i]; }

  static simd_vector<T, N> add(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] + b.v[i];
    }
    return r;
  }

  static simd_vector<T, N> subtract(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] - b.v[i];
    }
    return r;
  }

  static simd_vector<T, N> multiply(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] * b.v[i];
    }
    return r;
  }

  static simd_vector<T, N> divide(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] / b.v[i];
    }
    return r;
  }

  static simd_vector<T, N> negate(const simd_vector<T, N> &v) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = -v.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> equal(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] == b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> not_equal(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] != b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> less_than(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] < b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> less_equal(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] <= b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> greater_than(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] > b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> greater_equal(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] >= b.v[i];
    }
    return r;
  }

  static simd_vector<T, N> min(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = std::min(a.v[i], b.v[i]);
    }
    return r;
  }

  static simd_vector<T, N> max(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = std::max(a.v[i], b.v[i]);
    }
    return r;
  }
};

template <int N> struct simd_default_backend {
  template <typename T> using storage_type = simd_vector<T, N>;
  template <typename T> using mask_storage_type = simd_vector<bool, N>;
  template <typename T> static constexpr std::size_t simd_size{N};
  template <typename T> using impl = simd_default_impl<T, N>;
  template <typename T> using mask_impl = simd_default_mask_impl<N>;
};

} // namespace detail

namespace simd_abi {
template <int N> using fixed_size = detail::simd_default_backend<N>;
template <typename T> using compatible = fixed_size<4U>;
} // namespace simd_abi

template <> struct is_abi_tag<detail::simd_default_backend<4U>> : std::integral_constant<bool, true> {};
template <> struct is_simd<simd<float, detail::simd_default_backend<4U>>> : std::integral_constant<bool, true> {};
template <>
struct is_simd_mask<simd_mask<float, detail::simd_default_backend<4U>>> : std::integral_constant<bool, true> {};

} // namespace parallelism_v2

#endif // DETAIL_SIMD_DEFAULT_BACKEND_H
