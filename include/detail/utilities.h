// SPDX-License-Identifier: MIT

#ifndef DETAIL_UTILITIES_H
#define DETAIL_UTILITIES_H

#include <cstring>
#include <type_traits>

namespace parallelism_v2 {
namespace detail {

template <typename To, typename From> To bit_cast(const From &src) noexcept {
  static_assert(sizeof(To) == sizeof(From), "not same size");
  static_assert(std::is_trivially_copyable<From>::value, "From not trivially copyable");
  static_assert(std::is_trivially_copyable<To>::value, "To not trivially copyable");
  static_assert(std::is_trivially_constructible<To>::value, "not trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

struct condition_violated {};

} // namespace detail
} // namespace parallelism_v2

#define ENSURES(c)                                                                                                     \
  do {                                                                                                                 \
    const bool condition{c};                                                                                           \
    if (!condition) {                                                                                                  \
      throw parallelism_v2::detail::condition_violated{};                                                              \
    }                                                                                                                  \
  } while (false)

#endif // DETAIL_UTILITIES_H
