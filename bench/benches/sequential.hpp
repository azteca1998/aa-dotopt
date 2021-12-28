#ifndef DOTOPT_BENCH__BENCHES__SEQUENTIAL_HPP
#define DOTOPT_BENCH__BENCHES__SEQUENTIAL_HPP

#include <benchmark/benchmark.h>


void bench_sequential(benchmark::State &state);
void bench_sequential_zorder(benchmark::State &state);

void bench_sequential_omp_loops(benchmark::State &state);
void bench_sequential_omp_loops_zorder(benchmark::State &state);


#endif // DOTOPT_BENCH__BENCHES__SEQUENTIAL_HPP
