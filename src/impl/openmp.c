#include "openmp.h"

#include <omp.h>


void impl_openmp_loops(PyArrayObject *a, PyArrayObject *b, PyArrayObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    // TODO: Implement parallel matrix multiplication using OpenMP's loop
    //   parallelism.

#undef matrix_at
#undef IMPL_TYPE
}

void impl_openmp_tasks(PyArrayObject *a, PyArrayObject *b, PyArrayObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    // TODO: Implement parallel matrix multiplication using OpenMP's task
    //   parallelism.

#undef matrix_at
#undef IMPL_TYPE
}
