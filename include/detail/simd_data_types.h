// SPDX-License-Identifier: MIT

#ifndef DETAIL_SIMD_DATA_TYPES_H
#define DETAIL_SIMD_DATA_TYPES_H

#include "detail/utilities.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

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

  /// @brief Convert to underlying storage type.
  explicit operator _storage_type() const { return v_; }

  /// @brief The value of the ith element.
  ///
  /// @pre i < size()
  value_type operator[](const std::size_t i) const {
    ENSURES(i < size());
    return Abi::template mask_impl<T>::extract(v_, i);
  }

  /// @brief Applies logical not to each element.
  simd_mask operator!() const noexcept { return simd_mask{Abi::template mask_impl<T>::logical_not(v_)}; }

private:
  _storage_type v_;
};

/// @brief Applies logical and to each element.
template <typename T, typename Abi>
simd_mask<T, Abi> operator&&(const simd_mask<T, Abi> &lhs, const simd_mask<T, Abi> &rhs) noexcept {
  using type = typename simd_mask<T, Abi>::_storage_type;
  return simd_mask<T, Abi>{Abi::template mask_impl<T>::logical_and(static_cast<type>(lhs), static_cast<type>(rhs))};
}

/// @brief Applies logical or to each element.
template <typename T, typename Abi>
simd_mask<T, Abi> operator||(const simd_mask<T, Abi> &lhs, const simd_mask<T, Abi> &rhs) noexcept {
  using type = typename simd_mask<T, Abi>::_storage_type;
  return simd_mask<T, Abi>{Abi::template mask_impl<T>::logical_or(static_cast<type>(lhs), static_cast<type>(rhs))};
}

/// @brief Returns true if all boolean elements in v are true, false otherwise.
template <typename T, typename Abi> bool all_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::all_of(static_cast<typename simd_mask<T, Abi>::_storage_type>(v));
}

/// @brief Returns true if at least one boolean element in v is true, false otherwise.
template <typename T, typename Abi> bool any_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::any_of(static_cast<typename simd_mask<T, Abi>::_storage_type>(v));
}

/// @brief Returns true if none of the boolean elements in v is true, false otherwise.
template <typename T, typename Abi> bool none_of(const simd_mask<T, Abi> &v) noexcept {
  return Abi::template mask_impl<T>::none_of(static_cast<typename simd_mask<T, Abi>::_storage_type>(v));
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

  /// @brief Convert to underlying storage type.
  explicit operator _storage_type() const { return v_; }

  /// @brief Replaces the elements of the simd object from memory pointing to an aligned address.
  ///
  /// @pre [v, v + size()) is a valid range.
  /// @pre v shall point to storage aligned by parallelism_v2::memory_alignment_v<simd>.
  void copy_from(const value_type *const v, vector_aligned_tag) {
    static_assert(is_simd_flag_type_v<vector_aligned_tag>, "not a simd flag type tag");
    ENSURES(::parallelism_v2::detail::bit_cast<std::uintptr_t>(v) % memory_alignment_v<simd> == 0U);
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
    ENSURES(::parallelism_v2::detail::bit_cast<std::uintptr_t>(v) % memory_alignment_v<simd> == 0U);
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
  /// @pre i < size()
  value_type operator[](const std::size_t i) const {
    ENSURES(i < size());
    return Abi::template impl<T>::extract(v_, i);
  }

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
  using type = typename simd<T, Abi>::_storage_type;
  return simd_mask<T, Abi>{Abi::template impl<T>::equal(static_cast<type>(lhs), static_cast<type>(rhs))};
}

/// @brief Returns true if lhs is not equal to rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator!=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  using type = typename simd<T, Abi>::_storage_type;
  return simd_mask<T, Abi>{Abi::template impl<T>::not_equal(static_cast<type>(lhs), static_cast<type>(rhs))};
}

/// @brief Returns true if lhs is less than rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator<(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  using type = typename simd<T, Abi>::_storage_type;
  return simd_mask<T, Abi>{Abi::template impl<T>::less_than(static_cast<type>(lhs), static_cast<type>(rhs))};
}

/// @brief Returns true if lhs is less than or equal to rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator<=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  using type = typename simd<T, Abi>::_storage_type;
  return simd_mask<T, Abi>{Abi::template impl<T>::less_equal(static_cast<type>(lhs), static_cast<type>(rhs))};
}

/// @brief Returns true if lhs is greater than rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator>(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  using type = typename simd<T, Abi>::_storage_type;
  return simd_mask<T, Abi>{Abi::template impl<T>::greater_than(static_cast<type>(lhs), static_cast<type>(rhs))};
}

/// @brief Returns true if lhs is greater than or equal to rhs, false otherwise.
template <typename T, typename Abi>
simd_mask<T, Abi> operator>=(const simd<T, Abi> &lhs, const simd<T, Abi> &rhs) noexcept {
  using type = typename simd<T, Abi>::_storage_type;
  return simd_mask<T, Abi>{Abi::template impl<T>::greater_equal(static_cast<type>(lhs), static_cast<type>(rhs))};
}

/// @brief Returns the smaller of a and b. Returns a if one operand is NaN.
template <typename T, typename Abi> simd<T, Abi> min(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept {
  using type = typename simd<T, Abi>::_storage_type;
  return simd<T, Abi>{Abi::template impl<T>::min(static_cast<type>(a), static_cast<type>(b))};
}

/// @brief Returns the greater of a and b. Returns a if one operand is NaN.
template <typename T, typename Abi> simd<T, Abi> max(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept {
  using type = typename simd<T, Abi>::_storage_type;
  return simd<T, Abi>{Abi::template impl<T>::max(static_cast<type>(a), static_cast<type>(b))};
}

/// @brief Returns low if v is less than low, high if high is less than v, otherwise v.
///
/// @pre low <= high
template <typename T, typename Abi>
simd<T, Abi> clamp(const simd<T, Abi> &v, const simd<T, Abi> &low, const simd<T, Abi> &high) {
  ENSURES(all_of(low <= high));
  return ::parallelism_v2::min(::parallelism_v2::max(v, low), high);
}

/// @brief The class abstracts the notion of selecting elements of a given object of a data-parallel type.
template <typename M, typename T> class where_expression {
  static_assert(is_simd_mask_v<M>, "not a mask type");
  static_assert(is_simd_v<T>, "not a data-parallel type");
  static_assert(std::is_same<T, typename M::simd_type>::value, "incompatible mask and data-parallel type");

  using type = typename T::_storage_type;
  using mask_type = typename M::_storage_type;
  using impl = typename T::abi_type::template impl<typename T::value_type>;

public:
  /// @brief Do not call directly. Instead use `where()` function.
  where_expression(const M &mask, T &value) : m_{mask}, v_{value} {}
  where_expression(const where_expression &) = delete;
  where_expression &operator=(const where_expression &) = delete;

  /// @brief Replace the elements of value with the elements of x for elements where mask is true.
  template <typename U> void operator=(U &&x) && noexcept {
    static_assert(std::is_same<const T, const std::remove_reference_t<U>>::value, "no known conversion");
    v_ = T{impl::blend(static_cast<type>(v_), static_cast<type>(std::forward<U>(x)), static_cast<mask_type>(m_))};
  }

  /// @brief Replace the elements of value with the elements of value + x for elements where mask is true.
  template <typename U> void operator+=(U &&x) && noexcept {
    static_assert(std::is_same<const T, const std::remove_reference_t<U>>::value, "no known conversion");
    v_ = T{impl::blend(static_cast<type>(v_), static_cast<type>(v_ + std::forward<U>(x)), static_cast<mask_type>(m_))};
  }

  /// @brief Replace the elements of value with the elements of value - x for elements where mask is true.
  template <typename U> void operator-=(U &&x) && noexcept {
    static_assert(std::is_same<const T, const std::remove_reference_t<U>>::value, "no known conversion");
    v_ = T{impl::blend(static_cast<type>(v_), static_cast<type>(v_ - std::forward<U>(x)), static_cast<mask_type>(m_))};
  }

  /// @brief Replace the elements of value with the elements of value * x for elements where mask is true.
  template <typename U> void operator*=(U &&x) && noexcept {
    static_assert(std::is_same<const T, const std::remove_reference_t<U>>::value, "no known conversion");
    v_ = T{impl::blend(static_cast<type>(v_), static_cast<type>(v_ * std::forward<U>(x)), static_cast<mask_type>(m_))};
  }

  /// @brief Replace the elements of value with the elements of value / x for elements where mask is true.
  template <typename U> void operator/=(U &&x) && noexcept {
    static_assert(std::is_same<const T, const std::remove_reference_t<U>>::value, "no known conversion");
    v_ = T{impl::blend(static_cast<type>(v_), static_cast<type>(v_ / std::forward<U>(x)), static_cast<mask_type>(m_))};
  }

private:
  const M m_;
  T &v_;
};

/// @brief Select elements of v where the corresponding elements of m are true.
///
/// Usage: `where(mask, value) @ other;`.
///
/// Where `@` denotes one of the operators of `where_expression<>`.
template <typename T, typename Abi>
where_expression<simd_mask<T, Abi>, simd<T, Abi>> where(const typename simd<T, Abi>::mask_type &m,
                                                        simd<T, Abi> &v) noexcept {
  return {m, v};
}

} // namespace parallelism_v2

#endif // DETAIL_SIMD_DATA_TYPES_H
