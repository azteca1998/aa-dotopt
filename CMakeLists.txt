cmake_minimum_required(VERSION 3.18)
project(aa-dotopt C CXX)

# CMake features.
enable_testing()

# Dependencies.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(BENCHMARK_ENABLE_LIBPFM ON)
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(lib/benchmark)

# Sources.
add_subdirectory(bench)
add_subdirectory(src)
add_subdirectory(test)
