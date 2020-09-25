// SPDX-License-Identifier: MIT

#ifndef SIMD_DATA_TYPES_H
#define SIMD_DATA_TYPES_H

#include <cstddef>
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
/// @brief The number of elements in a parallelism_v2::simd<T, Abi> object.
template <typename T, typename Abi> constexpr std::size_t simd_size_v{simd_size<T, Abi>::value};

template <typename T, typename U = typename T::value_type> struct memory_alignment {
  static constexpr std::size_t value{alignof(typename T::_storage_type)};
};
template <typename T, typename U = typename T::value_type>
constexpr std::size_t memory_alignment_v{memory_alignment<T, U>::value};

/// @brief The class template simd_mask is a data-parallel type with the element type bool.
///
/// A data-parallel type consists of elements of an underlying arithmetic type, called the element type. The number of
/// elements is a constant for each data-parallel type and called the width of that type.
///
/// An element-wise operation applies a specified operation to the elements of one or more data-parallel objects. Each
/// such application is unsequenced with respect to the others. A unary element-wise operation is an element-wise
/// operation that applies a unary operation to each element of a data-parallel object. A binary element-wise operation
/// is an element-wise operation that applies a binary operation to corresponding elements of two data-parallel objects.
template <typename T, typename Abi> class simd_mask {
  static_assert(is_abi_tag_v<Abi>, "not an abi tag");
  static_assert(is_simd_mask_v<simd_mask>, "not a data-parallel type");

public:
  using _storage_type = typename Abi::template mask_storage_type<T>;
  using value_type = bool;
  using simd_type = simd<T, Abi>;
  using abi_type = Abi;

  /// @brief The number of elements, i.e., the width, of parallelism_v2::simd<T, Abi>.
  static constexpr std::size_t size() noexcept { return simd_size_v<T, Abi>; }

  /// @brief Default initialize.
  ///
  /// Performs no initialization of the elements. Thus, leaves the elements in an indeterminate state.
  simd_mask() noexcept = default;

  /// @brief Broadcast argument to all elements.
  explicit simd_mask(const value_type v) noexcept : v_{Abi::template mask_impl<T>::broadcast(v)} {}

  /// @brief Construct from all given arguments.
  template <typename = std::enable_if_t<size() == 4U>>
  explicit simd_mask(const value_type w, const value_type x, const value_type y, const value_type z)
      : v_{Abi::template mask_impl<T>::init(w, x, y, z)} {}

  /// @brief Convert from argument.
  explicit simd_mask(const _storage_type v) : v_{v} {}

  /// @brief Convert from argument.
  explicit operator _storage_type() const { return v_; }

  /// @brief The value of the ith element.
  ///
  /// @pre i < N
  value_type operator[](const std::size_t i) const { return Abi::template mask_impl<T>::extract(v_, i); }

  /// @brief Applies logical not to each element.
  simd_mask operator!() const noexcept { return simd_mask{Abi::template mask_impl<T>::logical_not(v_)}; }

  /// @brief Applies logical and to each element.
  friend simd_mask operator&&(const simd_mask &lhs, const simd_mask &rhs) noexcept {
    return simd_mask{Abi::template mask_impl<T>::logical_and(lhs.v_, rhs.v_)};
  }

  /// @brief Applies logical or to each element.
  friend simd_mask operator||(const simd_mask &lhs, const simd_mask &rhs) noexcept {
    return simd_mask{Abi::template mask_impl<T>::logical_or(lhs.v_, rhs.v_)};
  }

private:
  _storage_type v_;
};

/// @brief Returns true if all boolean elements in v are true, false otherwise.
template <typename T, typename Abi> bool all_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::all_of(static_cast<typename Abi::template mask_storage_type<T>>(v));
}

/// @brief Returns true if at least one boolean element in v is true, false otherwise.
template <typename T, typename Abi> bool any_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::any_of(static_cast<typename Abi::template mask_storage_type<T>>(v));
}

/// @brief Returns true if none of the boolean elements in v is true, false otherwise.
template <typename T, typename Abi> bool none_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::none_of(static_cast<typename Abi::template mask_storage_type<T>>(v));
}

/// @brief The class template simd is a data-parallel type T.
///
/// A data-parallel type consists of elements of an underlying arithmetic type, called the element type. The number of
/// elements is a constant for each data-parallel type and called the width of that type.
///
/// An element-wise operation applies a specified operation to the elements of one or more data-parallel objects. Each
/// such application is unsequenced with respect to the others. A unary element-wise operation is an element-wise
/// operation that applies a unary operation to each element of a data-parallel object. A binary element-wise operation
/// is an element-wise operation that applies a binary operation to corresponding elements of two data-parallel objects.
template <typename T, typename Abi> class simd {
  static_assert(is_abi_tag_v<Abi>, "not an abi tag");
  static_assert(is_simd_v<simd>, "not a data-parallel type");

public:
  using _storage_type = typename Abi::template storage_type<T>;
  using value_type = T;
  using mask_type = simd_mask<T, Abi>;
  using abi_type = Abi;

  /// @brief The number of elements, i.e., the width, of parallelism_v2::simd<T, Abi>.
  static constexpr std::size_t size() noexcept { return simd_size_v<T, Abi>; }

  /// @brief Default initialize.
  ///
  /// Performs no initialization of the elements. Thus, leaves the elements in an indeterminate state.
  simd() noexcept = default;

  /// @brief Broadcast argument to all elements.
  explicit simd(const value_type v) noexcept : v_{Abi::template impl<T>::broadcast(v)} {}

  /// @brief Construct from all given arguments.
  template <typename = std::enable_if_t<size() == 4U>>
  explicit simd(const value_type w, const value_type x, const value_type y, const value_type z) noexcept
      : v_{Abi::template impl<T>::init(w, x, y, z)} {}

  /// @brief Convert from argument.
  explicit simd(const _storage_type &v) noexcept : v_{v} {}

  /// @brief Replaces the elements of the simd object from memory pointing to an aligned address.
  ///
  /// @pre [v, v + size()) is a valid range.
  /// @pre v shall point to storage aligned by parallelism_v2::memory_alignment_v<simd>.
  void copy_from(const value_type *const v, vector_aligned_tag) {
    static_assert(is_simd_flag_type_v<vector_aligned_tag>, "not a simd flag type tag");
    v_ = Abi::template impl<T>::load_aligned(v);
  }

  /// @brief Replaces the elements of the simd object from memory pointing to an unaligned address.
  ///
  /// @pre [v, v + size()) is a valid range.
  /// @pre v shall point to storage aligned by alignof(value_type).
  void copy_from(const value_type *const v, element_aligned_tag) {
    static_assert(is_simd_flag_type_v<element_aligned_tag>, "not a simd flag type tag");
    v_ = Abi::template impl<T>::load(v);
  }

  /// @brief Replaces the elements of the simd object from memory pointing to an aligned address.
  ///
  /// @pre [v, v + size()) is a valid range.
  /// @pre v shall point to storage aligned by parallelism_v2::memory_alignment_v<simd>.
  void copy_to(value_type *const v, vector_aligned_tag) const {
    static_assert(is_simd_flag_type_v<vector_aligned_tag>, "not a simd flag type tag");
    Abi::template impl<T>::store_aligned(v, v_);
  }

  /// @brief Replaces the elements of the simd object from memory pointing to an unaligned address.
  ///
  /// @pre [v, v + size()) is a valid range.
  /// @pre v shall point to storage aligned by alignof(value_type).
  void copy_to(value_type *const v, element_aligned_tag) const {
    static_assert(is_simd_flag_type_v<element_aligned_tag>, "not a simd flag type tag");
    Abi::template impl<T>::store(v, v_);
  }

  /// @brief The value of the ith element.
  ///
  /// @pre i < N
  value_type operator[](const std::size_t i) const { return Abi::template impl<T>::extract(v_, i); }

  /// @brief Returns 128bit vector.
  _storage_type value() const { return v_; }

  /// @brief Same as -1 * *this.
  simd operator-() const noexcept { return simd{Abi::template impl<T>::negate(v_)}; }

  /// @brief Addition assignment operator.
  simd &operator+=(const simd &other) noexcept {
    v_ = Abi::template impl<T>::add(v_, other.v_);
    return *this;
  }

  /// @brief Subtraction assignment operator.
  simd &operator-=(const simd &other) noexcept {
    v_ = Abi::template impl<T>::subtract(v_, other.v_);
    return *this;
  }

  /// @brief Multiplication assignment operator.
  simd &operator*=(const simd &other) noexcept {
    v_ = Abi::template impl<T>::multiply(v_, other.v_);
    return *this;
  }

  /// @brief Division assignment operator.
  simd &operator/=(const simd &other) noexcept {
    v_ = Abi::template impl<T>::divide(v_, other.v_);
    return *this;
  }

private:
  _storage_type v_;
};

/// @brief Addition operator.
template <typename T, typename Abi> simd<T, Abi> operator+(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  simd<T, Abi> tmp{lhs};
  return tmp += rhs;
}

/// @brief Subtraction operator.
template <typename T, typename Abi> simd<T, Abi> operator-(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  simd<T, Abi> tmp{lhs};
  return tmp -= rhs;
}

/// @brief Multiplication operator.
template <typename T, typename Abi> simd<T, Abi> operator*(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  simd<T, Abi> tmp{lhs};
  return tmp *= rhs;
}

/// @brief Division operator.
template <typename T, typename Abi> simd<T, Abi> operator/(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  simd<T, Abi> tmp{lhs};
  return tmp /= rhs;
}

/// @brief Returns true if lhs is equal to rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator==(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_equal(lhs.value(), rhs.value())};
}

/// @brief Returns true if lhs is not equal to rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator!=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_not_equal(lhs.value(), rhs.value())};
}

/// @brief Returns true if lhs is less than rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator<(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_less_than(lhs.value(), rhs.value())};
}

/// @brief Returns true if lhs is less than or equal to rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator<=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_less_equal(lhs.value(), rhs.value())};
}

/// @brief Returns true if lhs is greater than rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator>(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_greater_than(lhs.value(), rhs.value())};
}

/// @brief Returns true if lhs is greater than or equal to rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator>=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd_mask<T, Abi>{Abi::template impl<T>::compare_greater_equal(lhs.value(), rhs.value())};
}

/// @brief Returns the smaller of lhs and rhs. Returns rhs if one operand is NaN.
template <typename T, typename Abi> simd<T, Abi> min(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd<T, Abi>{Abi::template impl<T>::min(lhs.value(), rhs.value())};
}

/// @brief Returns the greater of lhs and rhs. Returns rhs if one operand is NaN.
template <typename T, typename Abi> simd<T, Abi> max(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  return simd<T, Abi>{Abi::template impl<T>::max(lhs.value(), rhs.value())};
}

/// @brief Returns v restricted to the interval [low, high].
///
/// @pre low <= high
template <typename T, typename Abi>
simd<T, Abi> clamp(const simd<T, Abi> &v, const simd<T, Abi> &low, const simd<T, Abi> &high) noexcept {
  return ::parallelism_v2::min(::parallelism_v2::max(v, low), high);
}

} // namespace parallelism_v2

#endif // SIMD_DATA_TYPES_H
