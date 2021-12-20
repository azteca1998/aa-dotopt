#include "sequential_asm.hpp"

#include <cstdlib>

#include "../../src/impl/sequential_asm.h"
#include "../../src/impl/util.h"


void bench_sequential_asm(benchmark::State &state)
{
    matrix_t a, b, c;

    a.data = new float[state.range() * state.range()];
    a.num_rows = state.range();
    a.num_cols = state.range();
    a.row_stride = sizeof(float) * state.range();
    a.col_stride = sizeof(float);

    b.data = new float[state.range() * state.range()];
    b.num_rows = state.range();
    b.num_cols = state.range();
    b.row_stride = sizeof(float) * state.range();
    b.col_stride = sizeof(float);

    c.data = new float[state.range() * state.range()];
    c.num_rows = state.range();
    c.num_cols = state.range();
    c.row_stride = sizeof(float) * state.range();
    c.col_stride = sizeof(float);

    for (size_t i = 0; i < state.range() * state.range(); i++)
    {
        reinterpret_cast<float *>(a.data)[i] = (float) drand48() - 0.5f;
        reinterpret_cast<float *>(b.data)[i] = (float) drand48() - 0.5f;
    }

    sequential_version_t sv = sv_find_version(sizeof(float), &a, &b, &c);

    benchmark::ClobberMemory();

    for (auto _ : state)
        (*impl_sequential_asm[sv])(&a, &b, &c, 1);

    benchmark::ClobberMemory();

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}
