#ifndef DOTOPT_BENCH__BENCHES__IMTS_HPP
#define DOTOPT_BENCH__BENCHES__IMTS_HPP

#include <benchmark/benchmark.h>


void bench_imts_sequential(benchmark::State &state);
void bench_imts_sequential_asm(benchmark::State &state);

void bench_imts_sequential_zorder(benchmark::State &state);
void bench_imts_sequential_asm_zorder(benchmark::State &state);


#endif // DOTOPT_BENCH__BENCHES__IMTS_HPP
