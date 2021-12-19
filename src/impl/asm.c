#include "asm.h"


#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))


typedef void _impl_asm_t(
    float *ptr_a, float *ptr_b, float *ptr_c,
    ssize_t stride_a, ssize_t stride_b, ssize_t stride_c
);

extern _impl_asm_t _impl_asm_x1_x1_x1;
extern _impl_asm_t _impl_asm_x1_x1_x1_zf;


static void impl_asm_mkn(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c,
    int zf
)
{
    size_t m, m2;
    size_t k, k2;
    size_t n, n2;
    _impl_asm_t *impl_fn;

    for (m = 0; m <= PyArray_SHAPE(a)[0] - 8; m += 8)
        for (k = 0; k <= PyArray_SHAPE(a)[1] - 8; k += 8)
            for (n = 0; n <= PyArray_SHAPE(b)[1] - 8; n += 8)
            {
                impl_fn = (zf && k == 0)
                    ? &_impl_asm_x1_x1_x1_zf
                    : &_impl_asm_x1_x1_x1;

                (*impl_fn)(
                    ((float *) (PyArray_DATA(a) + PyArray_STRIDE(a, 0) * m)) + k,
                    ((float *) (PyArray_DATA(b) + PyArray_STRIDE(b, 0) * k)) + n,
                    ((float *) (PyArray_DATA(c) + PyArray_STRIDE(c, 0) * m)) + n,
                    (ssize_t) PyArray_STRIDE(a, 0),
                    (ssize_t) PyArray_STRIDE(b, 0),
                    (ssize_t) PyArray_STRIDE(c, 0)
                );
            }

    if (m < PyArray_SHAPE(a)[0])
        for (m2 = m; m2 < PyArray_SHAPE(a)[0]; m2++)
            for (k2 = 0; k2 < k; k2++)
                for (n2 = 0; n2 < n; n2++)
                {
                    if (zf && k2 == 0)
                        matrix_at(c, m2, n2) = (value_t) 0;

                    matrix_at(c, m2, n2) +=
                        matrix_at(a, m2, k2) * matrix_at(b, k2, n2);
                }

    if (n < PyArray_SHAPE(b)[1])
        for (m2 = 0; m2 < m; m2++)
            for (k2 = 0; k2 < k; k2++)
                for (n2 = n; n2 < PyArray_SHAPE(b)[1]; n2++)
                {
                    if (zf && k2 == 0)
                        matrix_at(c, m2, n2) = (value_t) 0;

                    matrix_at(c, m2, n2) +=
                        matrix_at(a, m2, k2) * matrix_at(b, k2, n2);
                }

    if (m < PyArray_SHAPE(a)[0] && n < PyArray_SHAPE(b)[1])
        for (m2 = m; m2 < PyArray_SHAPE(a)[0]; m2++)
            for (k2 = 0; k2 < k; k2++)
                for (n2 = n; n2 < PyArray_SHAPE(b)[1]; n2++)
                {
                    if (zf && k2 == 0)
                        matrix_at(c, m2, n2) = (value_t) 0;

                    matrix_at(c, m2, n2) +=
                        matrix_at(a, m2, k2) * matrix_at(b, k2, n2);
                }

    if (k < PyArray_SHAPE(a)[1])
        for (m2 = 0; m2 < PyArray_SHAPE(a)[0]; m2++)
            for (k2 = k; k2 < PyArray_SHAPE(a)[1]; k2++)
                for (n2 = 0; n2 < PyArray_SHAPE(b)[1]; n2++)
                    matrix_at(c, m2, n2) +=
                        matrix_at(a, m2, k2) * matrix_at(b, k2, n2);
}


static void impl_abort(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c,
    int zf
)
{
    abort();
}


const impl_asm_t impl_asm[27] = {
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,

    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_asm_mkn,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,

    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
    &impl_abort,
};
