#ifndef DOTOPT__IMPL__SEQUENTIAL_H
#define DOTOPT__IMPL__SEQUENTIAL_H

#include <numpy/arrayobject.h>

#include "util.h"


DOTOPT_API void impl_sequential(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
);


#endif /* DOTOPT__IMPL__SEQUENTIAL_H */
