
#include <cstddef>
#include <nmmintrin.h> // only include SSE4.2
#include <type_traits>

namespace parallelism_v2 {

template <typename T, typename Abi> class simd;

struct element_aligned_tag {};
struct vector_aligned_tag {};
constexpr element_aligned_tag element_aligned{};
constexpr vector_aligned_tag vector_aligned{};

template <typename T> struct is_abi_tag : std::integral_constant<bool, false> {};
template <typename T> constexpr bool is_abi_tag_v{is_abi_tag<T>::value};

template <typename T> struct is_simd : std::integral_constant<bool, false> {};
template <typename T> constexpr bool is_simd_v{is_simd<T>::value};

template <typename T> struct is_simd_mask : std::integral_constant<bool, false> {};
template <typename T> constexpr bool is_simd_mask_v{is_simd_mask<T>::value};

template <typename T> struct is_simd_flag_type : std::integral_constant<bool, false> {};
template <> struct is_simd_flag_type<element_aligned_tag> : std::integral_constant<bool, true> {};
template <> struct is_simd_flag_type<vector_aligned_tag> : std::integral_constant<bool, true> {};
template <typename T> constexpr bool is_simd_flag_type_v{is_simd_flag_type<T>::value};

template <typename T, typename Abi> struct simd_size {
  static constexpr std::size_t value{Abi::template simd_size<T>};
};

template <typename T, typename Abi> constexpr std::size_t simd_size_v{simd_size<T, Abi>::value};

template <typename T, typename U = typename T::value_type> struct memory_alignment {
  static constexpr std::size_t value{alignof(typename T::_storage_type)};
};
template <typename T, typename U = typename T::value_type>
constexpr std::size_t memory_alignment_v{memory_alignment<T, U>::value};
template <typename T, typename Abi> class simd_mask {
  static_assert(is_abi_tag_v<Abi>, "not an abi tag");
  static_assert(is_simd_mask_v<simd_mask>, "not a data-parallel type");

public:
  using _storage_type = typename Abi::template mask_storage_type<T>;
  using value_type = bool;
  using simd_type = simd<T, Abi>;
  using abi_type = Abi;

  static constexpr std::size_t size() noexcept { return simd_size_v<T, Abi>; }

  simd_mask() noexcept = default;

  explicit simd_mask(const value_type v) noexcept : v_{Abi::template mask_impl<T>::broadcast(v)} {}

  template <typename = std::enable_if_t<size() == 4U>>
  explicit simd_mask(const value_type w, const value_type x, const value_type y, const value_type z)
      : v_{Abi::template mask_impl<T>::init(w, x, y, z)} {}

  explicit simd_mask(const _storage_type v) : v_{v} {}

  explicit operator _storage_type() const { return v_; }

  value_type operator[](const std::size_t i) const { return Abi::template mask_impl<T>::extract(v_, i); }

  simd_mask operator!() const noexcept { return simd_mask{Abi::template mask_impl<T>::logical_not(v_)}; }

  friend simd_mask operator&&(const simd_mask &lhs, const simd_mask &rhs) noexcept {
    return simd_mask{Abi::template mask_impl<T>::logical_and(lhs.v_, rhs.v_)};
  }

  friend simd_mask operator||(const simd_mask &lhs, const simd_mask &rhs) noexcept {
    return simd_mask{Abi::template mask_impl<T>::logical_or(lhs.v_, rhs.v_)};
  }

private:
  _storage_type v_;
};

template <typename T, typename Abi> bool all_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::all_of(static_cast<typename Abi::template mask_storage_type<T>>(v));
}

template <typename T, typename Abi> bool any_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::any_of(static_cast<typename Abi::template mask_storage_type<T>>(v));
}

template <typename T, typename Abi> bool none_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::none_of(static_cast<typename Abi::template mask_storage_type<T>>(v));
}
template <typename T, typename Abi> class simd {
  static_assert(is_abi_tag_v<Abi>, "not an abi tag");
  static_assert(is_simd_v<simd>, "not a data-parallel type");

public:
  using _storage_type = typename Abi::template storage_type<T>;
  using value_type = T;
  using mask_type = simd_mask<T, Abi>;
  using abi_type = Abi;

  static constexpr std::size_t size() noexcept { return simd_size_v<T, Abi>; }

  simd() noexcept = default;

  explicit simd(const value_type v) noexcept : v_{Abi::template impl<T>::broadcast(v)} {}

  template <typename = std::enable_if_t<size() == 4U>>
  explicit simd(const value_type w, const value_type x, const value_type y, const value_type z) noexcept
      : v_{Abi::template impl<T>::init(w, x, y, z)} {}

  explicit simd(const _storage_type &v) noexcept : v_{v} {}

  void copy_from(const value_type *const v, vector_aligned_tag) {
    static_assert(is_simd_flag_type_v<vector_aligned_tag>, "not a simd flag type tag");
    v_ = Abi::template impl<T>::load_aligned(v);
  }

  void copy_from(const value_type *const v, element_aligned_tag) {
    static_assert(is_simd_flag_type_v<element_aligned_tag>, "not a simd flag type tag");
    v_ = Abi::template impl<T>::load(v);
  }

  void copy_to(value_type *const v, vector_aligned_tag) const {
    static_assert(is_simd_flag_type_v<vector_aligned_tag>, "not a simd flag type tag");
    Abi::template impl<T>::store_aligned(v, v_);
  }

  void copy_to(value_type *const v, element_aligned_tag) const {
    static_assert(is_simd_flag_type_v<element_aligned_tag>, "not a simd flag type tag");
    Abi::template impl<T>::store(v, v_);
  }

  value_type operator[](const std::size_t i) const { return Abi::template impl<T>::extract(v_, i); }

  _storage_type value() const { return v_; }

  simd operator-() const noexcept { return simd{Abi::template impl<T>::negate(v_)}; }

  simd &operator+=(const simd &other) noexcept {
    v_ = Abi::template impl<T>::add(v_, other.v_);
    return *this;
  }

  simd &operator-=(const simd &other) noexcept {
    v_ = Abi::template impl<T>::subtract(v_, other.v_);
    return *this;
  }

  simd &operator*=(const simd &other) noexcept {
    v_ = Abi::template impl<T>::multiply(v_, other.v_);
    return *this;
  }

  simd &operator/=(const simd &other) noexcept {
    v_ = Abi::template impl<T>::divide(v_, other.v_);
    return *this;
  }

private:
  _storage_type v_;
};

template <typename T, typename Abi> simd<T, Abi> operator+(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  simd<T, Abi> tmp{lhs};
  return tmp += rhs;
}

template <typename T, typename Abi> simd<T, Abi> operator-(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  simd<T, Abi> tmp{lhs};
  return tmp -= rhs;
}

template <typename T, typename Abi> simd<T, Abi> operator*(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  simd<T, Abi> tmp{lhs};
  return tmp *= rhs;
}

template <typename T, typename Abi> simd<T, Abi> operator/(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  simd<T, Abi> tmp{lhs};
  return tmp /= rhs;
}

template <typename T, typename Abi>
simd_mask<T, Abi> operator==(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_equal(lhs.value(), rhs.value())};
}

template <typename T, typename Abi>
simd_mask<T, Abi> operator!=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_not_equal(lhs.value(), rhs.value())};
}

template <typename T, typename Abi>
simd_mask<T, Abi> operator<(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_less_than(lhs.value(), rhs.value())};
}

template <typename T, typename Abi>
simd_mask<T, Abi> operator<=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_less_equal(lhs.value(), rhs.value())};
}

template <typename T, typename Abi>
simd_mask<T, Abi> operator>(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_greater_than(lhs.value(), rhs.value())};
}

template <typename T, typename Abi>
simd_mask<T, Abi> operator>=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_greater_equal(lhs.value(), rhs.value())};
}

template <typename T, typename Abi> simd<T, Abi> min(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd<T, Abi>{Abi::template impl<T>::min(lhs.value(), rhs.value())};
}

template <typename T, typename Abi> simd<T, Abi> max(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd<T, Abi>{Abi::template impl<T>::max(lhs.value(), rhs.value())};
}

template <typename T, typename Abi>
simd<T, Abi> clamp(const simd<T, Abi> &v, const simd<T, Abi> &low, const simd<T, Abi> &high) noexcept {
  return ::parallelism_v2::min(::parallelism_v2::max(v, low), high);
}

} // namespace parallelism_v2

namespace parallelism_v2 {
namespace detail {

template <typename T> struct sse_mask_intrinsics;

template <> struct sse_mask_intrinsics<float> {
  static __m128 broadcast(const bool v) noexcept { return _mm_cmpeq_ps(_mm_set1_ps(v), _mm_set1_ps(true)); }

  static __m128 init(const bool w, const bool x, const bool y, const bool z) noexcept {
    return _mm_cmpeq_ps(_mm_set_ps(z, y, x, w), _mm_set1_ps(true));
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

  static __m128 compare_equal(const __m128 a, const __m128 b) noexcept { return _mm_cmpeq_ps(a, b); }
  static __m128 compare_not_equal(const __m128 a, const __m128 b) noexcept { return _mm_cmpneq_ps(a, b); }
  static __m128 compare_less_than(const __m128 a, const __m128 b) noexcept { return _mm_cmplt_ps(a, b); }
  static __m128 compare_less_equal(const __m128 a, const __m128 b) noexcept { return _mm_cmple_ps(a, b); }
  static __m128 compare_greater_than(const __m128 a, const __m128 b) noexcept { return _mm_cmpgt_ps(a, b); }
  static __m128 compare_greater_equal(const __m128 a, const __m128 b) noexcept { return _mm_cmpge_ps(a, b); }

  static __m128 min(const __m128 a, const __m128 b) noexcept { return _mm_min_ps(a, b); }
  static __m128 max(const __m128 a, const __m128 b) noexcept { return _mm_max_ps(a, b); }
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

namespace parallelism_v2 {
template <typename T, typename Abi = simd_abi::compatible<T>> class simd;
template <typename T, int N> using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;
template <typename T, typename Abi = simd_abi::compatible<T>> class simd_mask;
template <typename T, int N> using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;
} // namespace parallelism_v2
