// SPDX-License-Identifier: MIT

#include "simd.h"
#include <gtest/gtest.h>

namespace parallelism_v2 {
namespace {

TEST(simd_mask, Size) { EXPECT_EQ(4U, (fixed_size_simd_mask<float, 4>::size())); }

TEST(simd_mask, Broadcast) {
  const simd_mask<float> a{true};
  EXPECT_TRUE(a[0U]);
  EXPECT_TRUE(a[1U]);
  EXPECT_TRUE(a[2U]);
  EXPECT_TRUE(a[3U]);
}

TEST(simd_mask, Initialize) {
  {
    const fixed_size_simd_mask<float, 4> a{true, false, false, false};
    EXPECT_TRUE(a[0U]);
    EXPECT_FALSE(a[1U]);
    EXPECT_FALSE(a[2U]);
    EXPECT_FALSE(a[3U]);
  }
  {
    const fixed_size_simd_mask<float, 4> a{false, true, false, false};
    EXPECT_FALSE(a[0U]);
    EXPECT_TRUE(a[1U]);
    EXPECT_FALSE(a[2U]);
    EXPECT_FALSE(a[3U]);
  }
  {
    const fixed_size_simd_mask<float, 4> a{false, false, true, false};
    EXPECT_FALSE(a[0U]);
    EXPECT_FALSE(a[1U]);
    EXPECT_TRUE(a[2U]);
    EXPECT_FALSE(a[3U]);
  }
  {
    const fixed_size_simd_mask<float, 4> a{false, false, false, true};
    EXPECT_FALSE(a[0U]);
    EXPECT_FALSE(a[1U]);
    EXPECT_FALSE(a[2U]);
    EXPECT_TRUE(a[3U]);
  }
}

TEST(simd_mask, Not) {
  {
    const simd_mask<float> a{true};
    EXPECT_TRUE(none_of(!a));
  }
  {
    const simd_mask<float> a{false};
    EXPECT_TRUE(all_of(!a));
  }
}

TEST(simd_mask, And) {
  {
    const simd_mask<float> a{true};
    EXPECT_TRUE(all_of(a && a));
  }
  {
    const simd_mask<float> a{true};
    EXPECT_TRUE(none_of(a && !a));
  }
  {
    const simd_mask<float> a{true};
    EXPECT_TRUE(none_of(!a && a));
  }
  {
    const simd_mask<float> a{false};
    EXPECT_TRUE(none_of(a && a));
  }
}

TEST(simd_mask, Or) {
  {
    const simd_mask<float> a{true};
    EXPECT_TRUE(all_of(a || a));
  }
  {
    const simd_mask<float> a{true};
    EXPECT_TRUE(all_of(a || !a));
  }
  {
    const simd_mask<float> a{true};
    EXPECT_TRUE(all_of(!a || a));
  }
  {
    const simd_mask<float> a{false};
    EXPECT_TRUE(none_of(a || a));
  }
}

TEST(simd_mask, AllOf) {
  {
    const fixed_size_simd_mask<float, 4> a{false, false, false, false};
    EXPECT_FALSE(all_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, false, false, false};
    EXPECT_FALSE(all_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, false, false};
    EXPECT_FALSE(all_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, true, false};
    EXPECT_FALSE(all_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, true, true};
    EXPECT_TRUE(all_of(a));
  }
}

TEST(simd_mask, AnyOf) {
  {
    const fixed_size_simd_mask<float, 4> a{false, false, false, false};
    EXPECT_FALSE(any_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, false, false, false};
    EXPECT_TRUE(any_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, false, false};
    EXPECT_TRUE(any_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, true, false};
    EXPECT_TRUE(any_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, true, true};
    EXPECT_TRUE(any_of(a));
  }
}

TEST(simd_mask, NoneOf) {
  {
    const fixed_size_simd_mask<float, 4> a{false, false, false, false};
    EXPECT_TRUE(none_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, false, false, false};
    EXPECT_FALSE(none_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, false, false};
    EXPECT_FALSE(none_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, true, false};
    EXPECT_FALSE(none_of(a));
  }
  {
    const fixed_size_simd_mask<float, 4> a{true, true, true, true};
    EXPECT_FALSE(none_of(a));
  }
}

} // namespace
} // namespace parallelism_v2
