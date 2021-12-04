#include "sequential.h"

#include <numpy/arrayobject.h>


void impl_sequential(PyObject *a, PyObject *b, PyObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    // Shape: 
    //   PyArray_SHAPE(a)[0], PyArray_SHAPE(a)[1]
    //   PyArray_SHAPE(b)[0], PyArray_SHAPE(b)[1]
    //   PyArray_SHAPE(c)[0], PyArray_SHAPE(c)[1]

    // Accessing array cells:
    //   matrix_at(c, 0, 0) = matrix_at(a, 0, 0) + matrix_at(b, 0, 0);

    // TODO: Implement algorithm here.

#undef matrix_at
#undef IMPL_TYPE
}
