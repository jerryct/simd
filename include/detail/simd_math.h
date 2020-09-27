// SPDX-License-Identifier: MIT

#ifndef DETAIL_SIMD_MATH_H
#define DETAIL_SIMD_MATH_H

#include "simd_data_types.h"

namespace parallelism_v2 {

/// @brief Returns true if v is a NaN, false otherwise
template <typename Abi> simd_mask<float, Abi> is_nan(const simd<float, Abi> &v) noexcept {
  using type = typename simd<float, Abi>::_storage_type;
  return simd_mask<float, Abi>{Abi::template impl<float>::is_nan(static_cast<type>(v))};
}

} // namespace parallelism_v2

#endif // DETAIL_SIMD_MATH_H
