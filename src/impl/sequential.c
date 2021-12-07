#include "sequential.h"


void impl_sequential(PyArrayObject *a, PyArrayObject *b, PyArrayObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    size_t m_sz = PyArray_SHAPE(a)[0]; // Rows of A and C.
    size_t k_sz = PyArray_SHAPE(a)[1]; // Cols of A and rows of B.
    size_t n_sz = PyArray_SHAPE(b)[1]; // Cols of B and C.

    for (size_t m = 0; m < m_sz; m++)           // Iterate over M.
        for (size_t n = 0; n < n_sz; n++)       // Iterate over N.
        {
            value_t acc = (value_t) 0;
            for (size_t k = 0; k < k_sz; k++)   // Iterate over K.
                acc += matrix_at(a, m, k) * matrix_at(b, k, n);

            matrix_at(c, m, n) = acc;
        }

#undef matrix_at
#undef IMPL_TYPE
}
