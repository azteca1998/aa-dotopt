#ifndef DOTOPT__IMPL__SEQUENTIAL_H
#define DOTOPT__IMPL__SEQUENTIAL_H

#include "util.h"


/**
 * @brief Sequential matrix multiplication using nested loops.
 *
 * @param 0 Left operand.
 * @param 1 Right operand.
 * @param 2 Output array.
 * @param 3 Whether to zero-fill or accumulate.
 */
typedef void (*impl_sequential_t)(matrix_t *, matrix_t *, matrix_t *, int);

extern const impl_sequential_t impl_sequential[28];

extern const impl_sequential_t impl_omp_loops_sequential[28];


#endif /* DOTOPT__IMPL__SEQUENTIAL_H */
