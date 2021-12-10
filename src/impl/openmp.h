#ifndef DOTOPT__IMPL__OPENMP_H
#define DOTOPT__IMPL__OPENMP_H

#include <numpy/arrayobject.h>

#include "util.h"


DOTOPT_API void impl_openmp_loops(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
);

DOTOPT_API void impl_openmp_tasks(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
);


#endif /* DOTOPT__IMPL__OPENMP_H */
