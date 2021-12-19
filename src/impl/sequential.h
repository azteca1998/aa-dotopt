#ifndef DOTOPT__IMPL__SEQUENTIAL_H
#define DOTOPT__IMPL__SEQUENTIAL_H

#include <numpy/arrayobject.h>

#include "util.h"


typedef void (*impl_sequential_t)(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c,
    int zero_fill
);

/**
 * @brief Sequential matrix multiplication using nested loops.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @param c Output array.
 */
extern const impl_sequential_t impl_sequential[27];


#endif /* DOTOPT__IMPL__SEQUENTIAL_H */
