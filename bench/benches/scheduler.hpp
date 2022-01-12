#ifndef DOTOPT_BENCH__BENCHES__SCHEDULER_HPP
#define DOTOPT_BENCH__BENCHES__SCHEDULER_HPP

#include <benchmark/benchmark.h>


void bench_scheduler(benchmark::State &state);
void bench_scheduler_asm(benchmark::State &state);


#endif // DOTOPT_BENCH__BENCHES__SCHEDULER_HPP
