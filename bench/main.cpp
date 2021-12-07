#include <benchmark/benchmark.h>

#include "benches/sequential.hpp"


BENCHMARK(bench_sequential)
    ->RangeMultiplier(2)
    ->Range(8, 16384);

BENCHMARK_MAIN();
