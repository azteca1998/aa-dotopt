#ifndef DOTOPT__IMPL__ASM_H
#define DOTOPT__IMPL__ASM_H

#include <numpy/arrayobject.h>


typedef void (*impl_asm_t)(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c,
    int zf
);


extern const impl_asm_t impl_asm[27];


#endif /* DOTOPT__IMPL__ASM_H */
