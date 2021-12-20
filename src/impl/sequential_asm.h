#ifndef DOTOPT__IMPL__SEQUENTIAL_ASM_H
#define DOTOPT__IMPL__SEQUENTIAL_ASM_H

#include "util.h"


/**
 * @brief Sequential matrix multiplication using nested loops of 8×8 tiles.
 *
 * Each tile multiplication is implemented in assembler. Margins, when present,
 * are calculated from within C using nested bucles.
 *
 * @param 0 Left operand.
 * @param 1 Right operand.
 * @param 2 Output array.
 * @param 3 Whether to zero-fill or accumulate.
 */
typedef void (*impl_sequential_asm_t)(matrix_t *, matrix_t *, matrix_t *, int);


extern const impl_sequential_asm_t impl_sequential_asm[27];


#endif /* DOTOPT__IMPL__SEQUENTIAL_ASM_H */
