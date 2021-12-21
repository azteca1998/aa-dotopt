#ifndef DOTOPT_BENCH__BENCHES__SEQUENTIAL_ASM_HPP
#define DOTOPT_BENCH__BENCHES__SEQUENTIAL_ASM_HPP

#include <benchmark/benchmark.h>


void bench_sequential_asm(benchmark::State &state);
void bench_sequential_asm_zorder(benchmark::State &state);


#endif // DOTOPT_BENCH__BENCHES__SEQUENTIAL_ASM_HPP
