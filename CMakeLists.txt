cmake_minimum_required(VERSION 3.10)

project(simd)

add_compile_options(
  -fno-omit-frame-pointer
  -msse4.2
  -std=c++14
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:Clang>>:-fno-vectorize>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:Clang>>:-fno-slp-vectorize>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:Clang>>:-Rpass-missed=loop-vectorize>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:Clang>>:-Rpass-analysis=loop-vectorize>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:Clang>>:-fsave-optimization-record>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:GNU>>:-ftree-vectorize>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:GNU>>:-ftree-vectorizer-verbose=5>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:GNU>>:-fopt-info-vec>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:GNU>>:-fdump-tree-vect>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:GNU>>:-fopt-info-vec-missed>
  #$<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:GNU>>:-fopt-info-vec-note>
)

add_subdirectory(/usr/src/googletest _build/googletest)

add_library(simd INTERFACE)
target_include_directories(simd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(unit_tests
  simd_mask_unit_test.cpp
  simd_unit_test.cpp
)
target_link_libraries(unit_tests INTERFACE simd PRIVATE gtest_main)
