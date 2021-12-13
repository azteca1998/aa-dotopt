#ifndef DOTOPT_BENCH__BENCHES__OPENMP_HPP
#define DOTOPT_BENCH__BENCHES__OPENMP_HPP

#include <benchmark/benchmark.h>


void bench_openmp_loops(benchmark::State &state);
void bench_openmp_tasks(benchmark::State &state);


#endif // DOTOPT_BENCH__BENCHES__OPENMP_HPP
