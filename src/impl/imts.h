#ifndef DOTOPT__IMPL__IMTS_H
#define DOTOPT__IMPL__IMTS_H

#include <numpy/arrayobject.h>

#include "util.h"


DOTOPT_API void impl_imts(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
);


#endif /* DOTOPT__IMPL__IMTS_H */
