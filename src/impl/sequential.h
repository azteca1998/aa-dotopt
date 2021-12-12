#ifndef DOTOPT__IMPL__SEQUENTIAL_H
#define DOTOPT__IMPL__SEQUENTIAL_H

#include <numpy/arrayobject.h>

#include "util.h"


typedef void (*impl_func_t)(PyArrayObject *, PyArrayObject *, PyArrayObject *);

/**
 * @brief Matrix multiplication using NumPy arrays.
 * 
 * @param a Left operand.
 * @param b Right operand.
 * @param c Output array.
 */
extern const impl_func_t impl_sequential[27];


#endif /* DOTOPT__IMPL__SEQUENTIAL_H */
