// SPDX-License-Identifier: MIT

#include "simd.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace parallelism_v2 {
namespace {

static_assert(std::is_standard_layout<simd<float>>::value, "No standard layout.");
static_assert(std::is_trivial<simd<float>>::value, "Not a trivial type.");
static_assert(std::is_trivially_copyable<simd<float>>::value, "Not trivially copyable.");
static_assert(std::is_trivially_default_constructible<simd<float>>::value, "Not trivially default constructable.");
static_assert(std::is_trivially_copy_constructible<simd<float>>::value, "Not trivially copy constructable.");
static_assert(std::is_trivially_move_constructible<simd<float>>::value, "Not trivially move constructable.");
static_assert(std::is_trivially_copy_assignable<simd<float>>::value, "Not trivially copy assignable.");
static_assert(std::is_trivially_move_assignable<simd<float>>::value, "Not trivially move assignable.");
static_assert(std::is_trivially_destructible<simd<float>>::value, "Not trivially destructable.");

TEST(simd, Size) {
  EXPECT_EQ(16U, (memory_alignment_v<simd<float>>));
  EXPECT_EQ(4U, (simd<float>::size()));
}

TEST(simd, Broadcast) {
  const simd<float> a{23.0F};

  EXPECT_EQ(23.0F, a[0U]);
  EXPECT_EQ(23.0F, a[1U]);
  EXPECT_EQ(23.0F, a[2U]);
  EXPECT_EQ(23.0F, a[3U]);
}

TEST(simd, Initialize) {
  const fixed_size_simd<float, 4> a{1.0F, 2.0F, 3.0F, 4.0F};

  EXPECT_EQ(1.0F, a[0U]);
  EXPECT_EQ(2.0F, a[1U]);
  EXPECT_EQ(3.0F, a[2U]);
  EXPECT_EQ(4.0F, a[3U]);
}

TEST(simd, LoadUnaligned) {
  fixed_size_simd<float, 4> vector;
  const std::array<float, 4U> scalars{1.0F, 2.0F, 3.0F, 4.0F};
  vector.copy_from(scalars.data(), element_aligned);

  EXPECT_EQ(1.0F, scalars[0U]);
  EXPECT_EQ(2.0F, scalars[1U]);
  EXPECT_EQ(3.0F, scalars[2U]);
  EXPECT_EQ(4.0F, scalars[3U]);
}

TEST(simd, LoadAligned) {
  fixed_size_simd<float, 4> vector;
  alignas(16) const std::array<float, 4U> scalars{1.0F, 2.0F, 3.0F, 4.0F};
  vector.copy_from(scalars.data(), vector_aligned);

  EXPECT_EQ(1.0F, scalars[0U]);
  EXPECT_EQ(2.0F, scalars[1U]);
  EXPECT_EQ(3.0F, scalars[2U]);
  EXPECT_EQ(4.0F, scalars[3U]);
}

TEST(simd, LoadAligned_WhenCopyingFromUnalignedMemory_ThenPreconditionViolated) {
  fixed_size_simd<float, 4> vector;
  alignas(16) const std::array<float, 5U> scalars{};

  EXPECT_THROW(vector.copy_from(&scalars[1], vector_aligned), parallelism_v2::detail::condition_violated);
}

TEST(simd, StoreUnaligned) {
  const fixed_size_simd<float, 4> vector{1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4U> scalars;
  vector.copy_to(scalars.data(), element_aligned);

  EXPECT_EQ(1.0F, std::get<0U>(scalars));
  EXPECT_EQ(2.0F, std::get<1U>(scalars));
  EXPECT_EQ(3.0F, std::get<2U>(scalars));
  EXPECT_EQ(4.0F, std::get<3U>(scalars));
}

TEST(simd, StoreAligned) {
  const fixed_size_simd<float, 4> vector{1.0F, 2.0F, 3.0F, 4.0F};
  alignas(16) std::array<float, 4U> scalars;
  vector.copy_to(scalars.data(), vector_aligned);

  EXPECT_EQ(1.0F, std::get<0U>(scalars));
  EXPECT_EQ(2.0F, std::get<1U>(scalars));
  EXPECT_EQ(3.0F, std::get<2U>(scalars));
  EXPECT_EQ(4.0F, std::get<3U>(scalars));
}

TEST(simd, StoreAligned_WhenCopyingToUnalignedMemory_ThenPreconditionViolated) {
  const fixed_size_simd<float, 4> vector{23.0F};
  alignas(16) std::array<float, 5U> scalars;

  EXPECT_THROW(vector.copy_to(&scalars[1], vector_aligned), parallelism_v2::detail::condition_violated);
}

TEST(simd, Access_WhenOutOfBounds_ThenPreconditionViolated) {
  const fixed_size_simd<float, 4> a{23.0F};

  EXPECT_THROW(a[4U], parallelism_v2::detail::condition_violated);
}

TEST(simd, Add) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(all_of(simd<float>{2.0F} == one + one));
  EXPECT_TRUE(all_of(inf == one + inf));
  EXPECT_TRUE(all_of(is_nan(one + nan)));

  EXPECT_TRUE(all_of(inf == inf + inf));
  EXPECT_TRUE(all_of(-inf == -inf + -inf));

  EXPECT_TRUE(all_of(is_nan(inf + -inf)));
  EXPECT_TRUE(all_of(is_nan(-inf + inf)));
}

TEST(simd, AssignmentAdd) {
  simd<float> a{1.0F};
  a += a;
  EXPECT_TRUE(all_of(simd<float>{2.0F} == a));
}

TEST(simd, Subtract) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(all_of(simd<float>{0.0F} == one - one));
  EXPECT_TRUE(all_of(-inf == one - inf));
  EXPECT_TRUE(all_of(is_nan(one - nan)));

  EXPECT_TRUE(all_of(-inf == -inf - inf));
  EXPECT_TRUE(all_of(inf == inf - -inf));

  EXPECT_TRUE(all_of(is_nan(inf - inf)));
  EXPECT_TRUE(all_of(is_nan(-inf - -inf)));
}

TEST(simd, AssignmentSubtract) {
  simd<float> a{1.0F};
  a -= a;
  EXPECT_TRUE(all_of(simd<float>{0.0F} == a));
}

TEST(simd, Multiply) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> two{2.0F};
  const simd<float> zero{0.0F};

  EXPECT_TRUE(all_of(simd<float>{4.0F} == two * two));
  EXPECT_TRUE(all_of(inf == two * inf));
  EXPECT_TRUE(all_of(is_nan(two * nan)));

  EXPECT_TRUE(all_of(is_nan(zero * inf)));
  EXPECT_TRUE(all_of(is_nan(-zero * inf)));
  EXPECT_TRUE(all_of(is_nan(zero * -inf)));
  EXPECT_TRUE(all_of(is_nan(-zero * -inf)));
  EXPECT_TRUE(all_of(is_nan(inf * zero)));
  EXPECT_TRUE(all_of(is_nan(-inf * zero)));
  EXPECT_TRUE(all_of(is_nan(inf * -zero)));
  EXPECT_TRUE(all_of(is_nan(-inf * -zero)));
}

TEST(simd, AssignmentMultiply) {
  simd<float> a{2.0F};
  a *= a;
  EXPECT_TRUE(all_of(simd<float>{4.0F} == a));
}

TEST(simd, Divide) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> two{2.0F};
  const simd<float> zero{0.0F};

  EXPECT_TRUE(all_of(simd<float>{1.0F} == two / two));
  EXPECT_TRUE(all_of(zero == two / inf));
  EXPECT_TRUE(all_of(inf == two / zero));
  EXPECT_TRUE(all_of(is_nan(two / nan)));

  EXPECT_TRUE(all_of(is_nan(zero / zero)));
  EXPECT_TRUE(all_of(is_nan(-zero / zero)));
  EXPECT_TRUE(all_of(is_nan(zero / -zero)));
  EXPECT_TRUE(all_of(is_nan(-zero / -zero)));
  EXPECT_TRUE(all_of(is_nan(inf / inf)));
  EXPECT_TRUE(all_of(is_nan(-inf / inf)));
  EXPECT_TRUE(all_of(is_nan(inf / -inf)));
  EXPECT_TRUE(all_of(is_nan(-inf / -inf)));
}

TEST(simd, AssignmentDivide) {
  simd<float> a{2.0F};
  a /= a;
  EXPECT_TRUE(all_of(simd<float>{1.0F} == a));
}

TEST(simd, Negate) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> zero{0.0F};

  const std::uint32_t nan_bits{0x7FC00000};
  const std::uint32_t neg_nan_bits{0xFFC00000};
  const std::uint32_t inf_bits{0x7F800000};
  const std::uint32_t neg_inf_bits{0xFF800000};
  const std::uint32_t null_bits{0x00000000};
  const std::uint32_t neg_null_bits{0x80000000};
  EXPECT_EQ(nan_bits, detail::bit_cast<std::uint32_t>(nan[0]));
  EXPECT_EQ(neg_nan_bits, detail::bit_cast<std::uint32_t>((-nan)[0]));
  EXPECT_EQ(inf_bits, detail::bit_cast<std::uint32_t>(inf[0]));
  EXPECT_EQ(neg_inf_bits, detail::bit_cast<std::uint32_t>((-inf)[0]));
  EXPECT_EQ(null_bits, detail::bit_cast<std::uint32_t>(zero[0]));
  EXPECT_EQ(neg_null_bits, detail::bit_cast<std::uint32_t>((-zero)[0]));
}

TEST(simd, Equal) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(all_of(one == one));
  EXPECT_TRUE(none_of(one == -one));
  EXPECT_TRUE(none_of(one == nan));
  EXPECT_TRUE(none_of(nan == one));
  EXPECT_TRUE(none_of(nan == inf));
  EXPECT_TRUE(none_of(inf == nan));
  EXPECT_TRUE(none_of(nan == -inf));
  EXPECT_TRUE(none_of(-inf == nan));
  EXPECT_TRUE(all_of(inf == inf));
  EXPECT_TRUE(none_of(-inf == inf));
  EXPECT_TRUE(none_of(inf == -inf));
  EXPECT_TRUE(all_of(-inf == -inf));
}

TEST(simd, NotEqual) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(none_of(one != one));
  EXPECT_TRUE(all_of(one != -one));
  EXPECT_TRUE(all_of(one != nan));
  EXPECT_TRUE(all_of(nan != one));
  EXPECT_TRUE(all_of(nan != inf));
  EXPECT_TRUE(all_of(inf != nan));
  EXPECT_TRUE(all_of(nan != -inf));
  EXPECT_TRUE(all_of(-inf != nan));
  EXPECT_TRUE(none_of(inf != inf));
  EXPECT_TRUE(all_of(-inf != inf));
  EXPECT_TRUE(all_of(inf != -inf));
  EXPECT_TRUE(none_of(-inf != -inf));
}

TEST(simd, LessThan) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(all_of(-one < one));
  EXPECT_TRUE(none_of(one < one));
  EXPECT_TRUE(none_of(one < nan));
  EXPECT_TRUE(none_of(nan < one));
  EXPECT_TRUE(none_of(nan < inf));
  EXPECT_TRUE(none_of(inf < nan));
  EXPECT_TRUE(none_of(nan < -inf));
  EXPECT_TRUE(none_of(-inf < nan));
  EXPECT_TRUE(none_of(inf < inf));
  EXPECT_TRUE(all_of(-inf < inf));
  EXPECT_TRUE(none_of(inf < -inf));
  EXPECT_TRUE(none_of(-inf < -inf));
}

TEST(simd, LessEqual) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(all_of(-one <= one));
  EXPECT_TRUE(all_of(one <= one));
  EXPECT_TRUE(none_of(one <= nan));
  EXPECT_TRUE(none_of(nan <= one));
  EXPECT_TRUE(none_of(nan <= inf));
  EXPECT_TRUE(none_of(inf <= nan));
  EXPECT_TRUE(none_of(nan <= -inf));
  EXPECT_TRUE(none_of(-inf <= nan));
  EXPECT_TRUE(all_of(inf <= inf));
  EXPECT_TRUE(all_of(-inf <= inf));
  EXPECT_TRUE(none_of(inf <= -inf));
  EXPECT_TRUE(all_of(-inf <= -inf));
}

TEST(simd, GreaterThan) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(all_of(one > -one));
  EXPECT_TRUE(none_of(one > one));
  EXPECT_TRUE(none_of(one > nan));
  EXPECT_TRUE(none_of(nan > one));
  EXPECT_TRUE(none_of(nan > inf));
  EXPECT_TRUE(none_of(inf > nan));
  EXPECT_TRUE(none_of(nan > -inf));
  EXPECT_TRUE(none_of(-inf > nan));
  EXPECT_TRUE(none_of(inf > inf));
  EXPECT_TRUE(none_of(-inf > inf));
  EXPECT_TRUE(all_of(inf > -inf));
  EXPECT_TRUE(none_of(-inf > -inf));
}

TEST(simd, GreaterEqual) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(all_of(one >= -one));
  EXPECT_TRUE(all_of(one >= one));
  EXPECT_TRUE(none_of(one >= nan));
  EXPECT_TRUE(none_of(nan >= one));
  EXPECT_TRUE(none_of(inf >= nan));
  EXPECT_TRUE(none_of(-inf >= nan));
  EXPECT_TRUE(none_of(nan >= inf));
  EXPECT_TRUE(none_of(nan >= -inf));
  EXPECT_TRUE(all_of(inf >= inf));
  EXPECT_TRUE(none_of(-inf >= inf));
  EXPECT_TRUE(all_of(inf >= -inf));
  EXPECT_TRUE(all_of(-inf >= -inf));
}

TEST(simd, Min) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> one{1.0F};

  EXPECT_TRUE(all_of(one == min(one, simd<float>{2.0F})));
  EXPECT_TRUE(all_of(-inf == min(one, -inf)));
  EXPECT_TRUE(all_of(one == min(one, nan)));
  EXPECT_TRUE(all_of(is_nan(min(nan, one))));
}

TEST(simd, Max) {
  const simd<float> nan{std::numeric_limits<float>::quiet_NaN()};
  const simd<float> inf{std::numeric_limits<float>::infinity()};
  const simd<float> two{2.0F};

  EXPECT_TRUE(all_of(two == max(two, simd<float>{1.0F})));
  EXPECT_TRUE(all_of(inf == max(two, inf)));
  EXPECT_TRUE(all_of(two == max(two, nan)));
  EXPECT_TRUE(all_of(is_nan(max(nan, two))));
}

TEST(simd, Clamp) {
  const simd<float> one{1.0F};
  const simd<float> low{-1.0F};
  const simd<float> high{1.0F};

  EXPECT_TRUE(all_of(simd<float>{0.0F} == clamp(simd<float>{0.0F}, low, high)));
  EXPECT_TRUE(all_of(low == clamp(simd<float>{-2.0F}, low, high)));
  EXPECT_TRUE(all_of(high == clamp(simd<float>{2.0F}, low, high)));
}

TEST(simd, Clamp_WhenNoValidBoundaryInterval_ThenPreconditionViolated) {
  const simd<float> one{1.0F};
  const simd<float> low{-1.0F};
  const simd<float> high{1.0F};

  EXPECT_THROW(clamp(one, high, low), parallelism_v2::detail::condition_violated);
}

TEST(simd, WhereAssignment) {
  fixed_size_simd<float, 4> value{6.0F, 9.0F, 16.0F, 25.0F};
  const fixed_size_simd_mask<float, 4> mask{true, false, true, false};

  where(mask, value) = fixed_size_simd<float, 4>{2.0F, 3.0F, 4.0F, 5.0F};

  EXPECT_TRUE(all_of(fixed_size_simd<float, 4>{2.0F, 9.0F, 4.0F, 25.0F} == value));
}

TEST(simd, WhereAssignmentAdd) {
  fixed_size_simd<float, 4> value{6.0F, 9.0F, 16.0F, 25.0F};
  const fixed_size_simd_mask<float, 4> mask{true, false, true, false};

  where(mask, value) += fixed_size_simd<float, 4>{2.0F, 3.0F, 4.0F, 5.0F};

  EXPECT_TRUE(all_of(fixed_size_simd<float, 4>{8.0F, 9.0F, 20.0F, 25.0F} == value));
}

TEST(simd, WhereAssignmentSubtract) {
  fixed_size_simd<float, 4> value{6.0F, 9.0F, 16.0F, 25.0F};
  const fixed_size_simd_mask<float, 4> mask{true, false, true, false};

  where(mask, value) -= fixed_size_simd<float, 4>{2.0F, 3.0F, 4.0F, 5.0F};

  EXPECT_TRUE(all_of(fixed_size_simd<float, 4>{4.0F, 9.0F, 12.0F, 25.0F} == value));
}

TEST(simd, WhereAssignmentMultiply) {
  fixed_size_simd<float, 4> value{6.0F, 9.0F, 16.0F, 25.0F};
  const fixed_size_simd_mask<float, 4> mask{true, false, true, false};

  where(mask, value) *= fixed_size_simd<float, 4>{2.0F, 3.0F, 4.0F, 5.0F};

  EXPECT_TRUE(all_of(fixed_size_simd<float, 4>{12.0F, 9.0F, 64.0F, 25.0F} == value));
}

TEST(simd, WhereAssignmentDivide) {
  fixed_size_simd<float, 4> value{6.0F, 9.0F, 16.0F, 25.0F};
  const fixed_size_simd_mask<float, 4> mask{true, false, true, false};

  where(mask, value) /= fixed_size_simd<float, 4>{2.0F, 3.0F, 4.0F, 5.0F};

  EXPECT_TRUE(all_of(fixed_size_simd<float, 4>{3.0F, 9.0F, 4.0F, 25.0F} == value));
}

} // namespace
} // namespace parallelism_v2
