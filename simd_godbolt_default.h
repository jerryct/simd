

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

  static simd_vector<bool, N> compare_equal(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] == b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> compare_not_equal(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] != b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> compare_less_than(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] < b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> compare_less_equal(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] <= b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> compare_greater_than(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] > b.v[i];
    }
    return r;
  }

  static simd_vector<bool, N> compare_greater_equal(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<bool, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = a.v[i] >= b.v[i];
    }
    return r;
  }

  static simd_vector<T, N> min(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = (a.v[i] < b.v[i]) ? a.v[i] : b.v[i];
    }
    return r;
  }

  static simd_vector<T, N> max(const simd_vector<T, N> &a, const simd_vector<T, N> &b) noexcept {
    simd_vector<T, N> r;
    for (int i = 0; i < N; ++i) {
      r.v[i] = (b.v[i] < a.v[i]) ? a.v[i] : b.v[i];
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

namespace parallelism_v2 {
template <typename T, typename Abi = simd_abi::compatible<T>> class simd;
template <typename T, int N> using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;
template <typename T, typename Abi = simd_abi::compatible<T>> class simd_mask;
template <typename T, int N> using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;
} // namespace parallelism_v2
