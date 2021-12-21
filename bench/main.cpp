#include <benchmark/benchmark.h>

#include "benches/imts.hpp"
#include "benches/sequential_asm.hpp"
#include "benches/sequential.hpp"


BENCHMARK(bench_imts_sequential)
    ->RangeMultiplier(2)
    ->Range(8, 16384);
BENCHMARK(bench_imts_sequential_asm)
    ->RangeMultiplier(2)
    ->Range(8, 16384);
BENCHMARK(bench_imts_sequential_asm_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384);
BENCHMARK(bench_imts_sequential_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384);
BENCHMARK(bench_sequential)
    ->RangeMultiplier(2)
    ->Range(8, 16384);
BENCHMARK(bench_sequential_asm)
    ->RangeMultiplier(2)
    ->Range(8, 16384);
BENCHMARK(bench_sequential_asm_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384);
BENCHMARK(bench_sequential_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384);

BENCHMARK_MAIN();
