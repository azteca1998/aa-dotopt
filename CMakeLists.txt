cmake_minimum_required(VERSION 3.18)
project(aa-dotopt C CXX)

# CMake features.
enable_testing()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

# Dependencies.
find_package(OpenMP REQUIRED COMPONENTS C)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Threads REQUIRED)

set(BENCHMARK_ENABLE_LIBPFM ON)
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(lib/benchmark)

# Sources.
add_subdirectory(bench)
add_subdirectory(src)
add_subdirectory(test)
