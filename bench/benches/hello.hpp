#ifndef DOTOPT_BENCH__BENCHES__HELLO_HPP
#define DOTOPT_BENCH__BENCHES__HELLO_HPP

#include <benchmark/benchmark.h>


void bench_hello(benchmark::State &state)
{
    for (auto _ : state)
    {
        asm("\
                .intel_syntax noprefix  \n\
                add rax, rax            \n\
                                        \n\
                .att_syntax             \n\
            "

            :
            :
            : "rax"
        );
    }
}

#endif // DOTOPT_BENCH__BENCHES__HELLO_HPP
