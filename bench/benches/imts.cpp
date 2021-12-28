#include "imts.hpp"

#include <cstdlib>

#include "../../src/impl/imts.h"
#include "../../src/impl/util.h"
#include "../../src/zorder.h"


void bench_imts_sequential(benchmark::State &state)
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
        (*impl_imts_sequential[sv])(&a, &b, &c, 1);

    benchmark::ClobberMemory();

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}

void bench_imts_sequential_asm(benchmark::State &state)
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
        (*impl_imts_sequential_asm[sv])(&a, &b, &c, 1);

    benchmark::ClobberMemory();

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}

void bench_imts_sequential_asm_zorder(benchmark::State &state)
{
    matrix_t a, b, c;
    float *ta, *tb, *tc, *tmp;

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

    ta = zorder_alloc(&a);
    tb = zorder_alloc(&b);
    tc = zorder_alloc(&c);

    zorder_transform(ta, &a);
    zorder_transform(tb, &b);

    tmp = reinterpret_cast<float *>(a.data);
    a.data = ta;
    ta = tmp;
    tmp = reinterpret_cast<float *>(b.data);
    b.data = tb;
    tb = tmp;
    tmp = reinterpret_cast<float *>(c.data);
    c.data = tc;
    tc = tmp;

    benchmark::ClobberMemory();

    for (auto _ : state)
        (*impl_imts_sequential_asm[sv_zz_zz_zz])(&a, &b, &c, 1);

    benchmark::ClobberMemory();

    tmp = reinterpret_cast<float *>(a.data);
    a.data = ta;
    ta = tmp;
    tmp = reinterpret_cast<float *>(b.data);
    b.data = tb;
    tb = tmp;
    tmp = reinterpret_cast<float *>(c.data);
    c.data = tc;
    tc = tmp;

    zorder_transform_inverse(tc, &c);

    zorder_free(ta);
    zorder_free(tb);
    zorder_free(tc);

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}

void bench_imts_sequential_zorder(benchmark::State &state)
{
    matrix_t a, b, c;
    float *ta, *tb, *tc, *tmp;

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

    ta = zorder_alloc(&a);
    tb = zorder_alloc(&b);
    tc = zorder_alloc(&c);

    zorder_transform(ta, &a);
    zorder_transform(tb, &b);

    tmp = reinterpret_cast<float *>(a.data);
    a.data = ta;
    ta = tmp;
    tmp = reinterpret_cast<float *>(b.data);
    b.data = tb;
    tb = tmp;
    tmp = reinterpret_cast<float *>(c.data);
    c.data = tc;
    tc = tmp;

    benchmark::ClobberMemory();

    for (auto _ : state)
        (*impl_imts_sequential[sv_zz_zz_zz])(&a, &b, &c, 1);

    benchmark::ClobberMemory();

    tmp = reinterpret_cast<float *>(a.data);
    a.data = ta;
    ta = tmp;
    tmp = reinterpret_cast<float *>(b.data);
    b.data = tb;
    tb = tmp;
    tmp = reinterpret_cast<float *>(c.data);
    c.data = tc;
    tc = tmp;

    zorder_transform_inverse(tc, &c);

    zorder_free(ta);
    zorder_free(tb);
    zorder_free(tc);

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}

void bench_imts_sequential_omp_tasks(benchmark::State &state)
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

    #pragma omp parallel
    {
        benchmark::ClobberMemory();

        for (auto _ : state)
            (*impl_omp_tasks_imts_sequential[sv])(&a, &b, &c, 1);

        benchmark::ClobberMemory();
    }

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}

void bench_imts_sequential_asm_omp_tasks(benchmark::State &state)
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

    #pragma omp parallel
    {
        benchmark::ClobberMemory();

        for (auto _ : state)
            (*impl_omp_tasks_imts_sequential_asm[sv])(&a, &b, &c, 1);

        benchmark::ClobberMemory();
    }

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}

void bench_imts_sequential_asm_omp_tasks_zorder(benchmark::State &state)
{
    matrix_t a, b, c;
    float *ta, *tb, *tc, *tmp;

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

    ta = zorder_alloc(&a);
    tb = zorder_alloc(&b);
    tc = zorder_alloc(&c);

    zorder_transform(ta, &a);
    zorder_transform(tb, &b);

    tmp = reinterpret_cast<float *>(a.data);
    a.data = ta;
    ta = tmp;
    tmp = reinterpret_cast<float *>(b.data);
    b.data = tb;
    tb = tmp;
    tmp = reinterpret_cast<float *>(c.data);
    c.data = tc;
    tc = tmp;

    #pragma omp parallel
    {
        benchmark::ClobberMemory();

        for (auto _ : state)
            (*impl_omp_tasks_imts_sequential_asm[sv_zz_zz_zz])(&a, &b, &c, 1);

        benchmark::ClobberMemory();
    }

    tmp = reinterpret_cast<float *>(a.data);
    a.data = ta;
    ta = tmp;
    tmp = reinterpret_cast<float *>(b.data);
    b.data = tb;
    tb = tmp;
    tmp = reinterpret_cast<float *>(c.data);
    c.data = tc;
    tc = tmp;

    zorder_transform_inverse(tc, &c);

    zorder_free(ta);
    zorder_free(tb);
    zorder_free(tc);

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}

void bench_imts_sequential_omp_tasks_zorder(benchmark::State &state)
{
    matrix_t a, b, c;
    float *ta, *tb, *tc, *tmp;

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

    ta = zorder_alloc(&a);
    tb = zorder_alloc(&b);
    tc = zorder_alloc(&c);

    zorder_transform(ta, &a);
    zorder_transform(tb, &b);

    tmp = reinterpret_cast<float *>(a.data);
    a.data = ta;
    ta = tmp;
    tmp = reinterpret_cast<float *>(b.data);
    b.data = tb;
    tb = tmp;
    tmp = reinterpret_cast<float *>(c.data);
    c.data = tc;
    tc = tmp;

    #pragma omp tasks
    {
        benchmark::ClobberMemory();

        for (auto _ : state)
            (*impl_omp_tasks_imts_sequential[sv_zz_zz_zz])(&a, &b, &c, 1);

        benchmark::ClobberMemory();
    }

    tmp = reinterpret_cast<float *>(a.data);
    a.data = ta;
    ta = tmp;
    tmp = reinterpret_cast<float *>(b.data);
    b.data = tb;
    tb = tmp;
    tmp = reinterpret_cast<float *>(c.data);
    c.data = tc;
    tc = tmp;

    zorder_transform_inverse(tc, &c);

    zorder_free(ta);
    zorder_free(tb);
    zorder_free(tc);

    // Clean up.
    delete[] reinterpret_cast<float *>(a.data);
    delete[] reinterpret_cast<float *>(b.data);
    delete[] reinterpret_cast<float *>(c.data);
}
