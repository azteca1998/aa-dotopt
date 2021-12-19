#ifndef DOTOPT__IMPL__ASM_H
#define DOTOPT__IMPL__ASM_H

#include <numpy/arrayobject.h>


typedef void (*impl_asm_t)(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c,
    int zf
);


/**
 * @brief Sequential matrix multiplication using nested loops of 8×8 tiles.
 *
 * Each tile multiplication is implemented in assembler. Margins, when present,
 * are calculated from within C using nested bucles.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @param c Output array.
 */
extern const impl_asm_t impl_asm[27];


#endif /* DOTOPT__IMPL__ASM_H */
