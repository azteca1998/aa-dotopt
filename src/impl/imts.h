#ifndef DOTOPT__IMPL__IMTS_H
#define DOTOPT__IMPL__IMTS_H

#include <numpy/arrayobject.h>

#include "util.h"


/**
 * @brief Sequential matrix multiplication using the IMTS scheduler with three
 * levels of recursion.
 *
 * The levels of recursion are adjusted to fit the test machine's cache sizes.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @param c Output array.
 */
DOTOPT_API void impl_imts(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
);


#endif /* DOTOPT__IMPL__IMTS_H */
