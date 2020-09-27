// SPDX-License-Identifier: MIT

#include "simd.h"
#include <gtest/gtest.h>

namespace parallelism_v2 {
namespace {

TEST(simd_math, IsNan) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(none_of(is_nan(one)));
  EXPECT_TRUE(none_of(is_nan(inf)));
  EXPECT_TRUE(all_of(is_nan(nan)));
  EXPECT_TRUE(all_of(is_nan(-nan)));
}

} // namespace
} // namespace parallelism_v2
