#include <benchmark/benchmark.h>

#include "benches/imts.hpp"
#include "benches/scheduler.hpp"
#include "benches/sequential_asm.hpp"
#include "benches/sequential.hpp"


BENCHMARK(bench_imts_sequential)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_imts_sequential_asm)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_imts_sequential_asm_omp_tasks)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_imts_sequential_asm_omp_tasks_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_imts_sequential_asm_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_imts_sequential_omp_tasks)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_imts_sequential_omp_tasks_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_imts_sequential_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_scheduler)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_scheduler_asm)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sequential)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sequential_asm)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sequential_asm_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sequential_asm_omp_loops)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sequential_asm_omp_loops_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sequential_omp_loops)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sequential_omp_loops_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sequential_zorder)
    ->RangeMultiplier(2)
    ->Range(8, 16384)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
