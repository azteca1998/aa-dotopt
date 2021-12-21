#ifndef DOTOPT__IMPL__IMTS_H
#define DOTOPT__IMPL__IMTS_H

#include "util.h"


// TODO: Sequential IMTS.
// TODO: Sequential IMTS ASM.


/**
 * @brief Sequential matrix multiplication using the IMTS scheduler with three
 * levels of recursion.
 *
 * The levels of recursion are adjusted to fit the test machine's cache sizes.
 *
 * @param 0 Left operand.
 * @param 1 Right operand.
 * @param 2 Output array.
 * @param 3 Whether to zero-fill or accumulate.
 */
typedef void (*impl_imts_sequential_t)(matrix_t *, matrix_t *, matrix_t *, int);

/**
 * @brief Sequential matrix multiplication using the IMTS scheduler with four
 * levels of recursion.
 *
 * The first three levels of recursion are adjusted to fit the test machine's
 * cache sizes. The last level is fixed to the ASM tile size (8×8).
 *
 * @param a Left operand.
 * @param b Right operand.
 * @param c Output array.
 * @param zf Whether to zero-fill or accumulate.
 */
typedef void (*impl_imts_sequential_asm_t)(
    matrix_t *a,
    matrix_t *b,
    matrix_t *c,
    int zf
);


extern const impl_imts_sequential_t impl_imts_sequential[28];
extern const impl_imts_sequential_asm_t impl_imts_sequential_asm[28];


#endif /* DOTOPT__IMPL__IMTS_H */
