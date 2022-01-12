#ifndef DOTOPT__IMPL__SCHEDULER_ASM_H
#define DOTOPT__IMPL__SCHEDULER_ASM_H

#include "util.h"


/**
 * @brief Custom scheduler matrix multiplication (parallel IMTS).
 *
 * @param 0 Left operand.
 * @param 1 Right operand.
 * @param 2 Output array.
 * @param 3 Whether to zero-fill or accumulate.
 */
typedef void (*impl_scheduler_asm_t)(matrix_t *, matrix_t *, matrix_t *, int);


extern const impl_scheduler_asm_t impl_scheduler_asm[28];


#endif /* DOTOPT__IMPL__SCHEDULER_ASM_H */
