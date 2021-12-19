#ifndef DOTOPT__IMPL__OPENMP_H
#define DOTOPT__IMPL__OPENMP_H

#include <numpy/arrayobject.h>

#include "util.h"


/**
 * @brief Parallel matrix multiplication using nested loops.
 *
 * Parallelization is implemented using OpenMP's loop parallelism.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @param c Output array.
 */
DOTOPT_API void impl_openmp_loops(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
);

/**
 * @brief Parallel matrix multiplication using tiled recursion and nested loops.
 *
 * Each tile is divided in four until its size fits the predefined maximum tile
 * size.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @param c Output array.
 */
DOTOPT_API void impl_openmp_tasks(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
);


#endif /* DOTOPT__IMPL__OPENMP_H */
