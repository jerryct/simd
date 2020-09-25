cmake_minimum_required(VERSION 3.10)

project(simd)

add_compile_options(
  -fno-omit-frame-pointer
  -msse4.2
)

add_subdirectory(/usr/src/googletest _build/googletest)

add_executable(simd
  simd.h
  simd_data_types.h
  simd_default_backend.h
  simd_mask_unit_test.cpp
  simd_sse_backend.h
  simd_unit_test.cpp
)
target_compile_options(simd PRIVATE -std=c++14)
target_link_libraries(simd PRIVATE gtest_main)
