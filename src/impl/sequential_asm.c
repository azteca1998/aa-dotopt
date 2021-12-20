#include "sequential_asm.h"

#include <stdlib.h>
#include <string.h>

#include "sequential.h"


typedef void _impl_asm_t(
    float *ptr_a, float *ptr_b, float *ptr_c,
    ssize_t stride_a, ssize_t stride_b, ssize_t stride_c
);

extern _impl_asm_t _impl_sequential_asm_x1_x1_x1;
extern _impl_asm_t _impl_sequential_asm_x1_x1_x1_zf;


static void impl_x1_x1_x1(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
#define matrix_at(m, row, col) \
    (((float *) ((m)->data + (m)->row_stride * (row)))[(col)])

    size_t m, k, n;
    _impl_asm_t *impl_ptr;
    matrix_t ta, tb, tc;

    for (m = 0; m < (a->num_rows & ~7); m += 8)
        for (k = 0; k < (a->num_cols & ~7); k += 8)
            for (n = 0; n < (b->num_cols & ~7); n += 8)
            {
                impl_ptr = zero_fill && k == 0
                    ? &_impl_sequential_asm_x1_x1_x1_zf
                    : &_impl_sequential_asm_x1_x1_x1;

                (*impl_ptr)(
                    &((float *) (a->data + m * a->row_stride))[k],
                    &((float *) (b->data + k * b->row_stride))[n],
                    &((float *) (c->data + m * c->row_stride))[n],
                    a->row_stride,
                    b->row_stride,
                    c->row_stride
                );
            }

    if (m == a->num_rows && k == a->num_cols && n == b->num_cols)
        return;

    ta.row_stride = a->row_stride;
    ta.col_stride = a->col_stride;
    tb.row_stride = b->row_stride;
    tb.col_stride = b->col_stride;
    tc.row_stride = c->row_stride;
    tc.col_stride = c->col_stride;

    if (a->num_rows & 7)
    {
        ta.data = a->data + m * a->row_stride;
        ta.num_rows = a->num_rows & 7;
        ta.num_cols = a->num_cols;

        tb.data = b->data;
        tb.num_rows = b->num_rows;
        tb.num_cols = b->num_cols;

        tc.data = c->data + m * c->row_stride;
        tc.num_rows = c->num_rows & 7;
        tc.num_cols = c->num_cols;

        impl_sequential[sv_x1_x1_x1](&ta, &tb, &tc, zero_fill);
    }

    if (b->num_cols & 7)
    {
        ta.data = a->data;
        ta.num_rows = a->num_rows;
        ta.num_cols = a->num_cols;

        tb.data = (void *) &((float *) b->data)[n];
        tb.num_rows = b->num_rows;
        tb.num_cols = b->num_cols & 7;

        tc.data = (void *) &((float *) c->data)[n];
        tc.num_rows = c->num_rows;
        tc.num_cols = c->num_cols & 7;

        impl_sequential[sv_x1_x1_x1](&ta, &tb, &tc, zero_fill);
    }

    if (a->num_cols & 7)
    {
        ta.data = a->data;
        ta.num_rows = a->num_rows;
        ta.num_cols = a->num_cols;

        tb.data = b->data;
        tb.num_rows = b->num_rows & 7;
        tb.num_cols = b->num_cols;

        tc.data = c->data;
        tc.num_rows = c->num_rows & 7;
        tc.num_cols = c->num_cols;

        impl_sequential[sv_x1_x1_x1](&ta, &tb, &tc, zero_fill);
    }

#undef matrix_at
}


static void impl_abort(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
    abort();
}


const impl_sequential_asm_t impl_sequential_asm[27] = {
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
    &impl_x1_x1_x1,
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



// #define value_t float
// #define matrix_at(m, row, col) \
//     (*(value_t *) (PyArray_DATA(m) \
//         + PyArray_STRIDE(m, 0) * row \
//         + PyArray_STRIDE(m, 1) * col \
//     ))


// typedef void _impl_asm_t(
//     float *ptr_a, float *ptr_b, float *ptr_c,
//     ssize_t stride_a, ssize_t stride_b, ssize_t stride_c
// );

// extern _impl_asm_t _impl_asm_x1_x1_x1;
// extern _impl_asm_t _impl_asm_x1_x1_x1_zf;


// static void impl_asm_mkn(
//     PyArrayObject *a,
//     PyArrayObject *b,
//     PyArrayObject *c,
//     int zf
// )
// {
//     size_t m, m2;
//     size_t k, k2;
//     size_t n, n2;
//     _impl_asm_t *impl_fn;

//     for (m = 0; m <= PyArray_SHAPE(a)[0] - 8; m += 8)
//         for (k = 0; k <= PyArray_SHAPE(a)[1] - 8; k += 8)
//             for (n = 0; n <= PyArray_SHAPE(b)[1] - 8; n += 8)
//             {
//                 impl_fn = (zf && k == 0)
//                     ? &_impl_asm_x1_x1_x1_zf
//                     : &_impl_asm_x1_x1_x1;

//                 (*impl_fn)(
//                     ((float *) (PyArray_DATA(a) + PyArray_STRIDE(a, 0) * m)) + k,
//                     ((float *) (PyArray_DATA(b) + PyArray_STRIDE(b, 0) * k)) + n,
//                     ((float *) (PyArray_DATA(c) + PyArray_STRIDE(c, 0) * m)) + n,
//                     (ssize_t) PyArray_STRIDE(a, 0),
//                     (ssize_t) PyArray_STRIDE(b, 0),
//                     (ssize_t) PyArray_STRIDE(c, 0)
//                 );
//             }

//     if (m < PyArray_SHAPE(a)[0])
//         for (m2 = m; m2 < PyArray_SHAPE(a)[0]; m2++)
//             for (k2 = 0; k2 < k; k2++)
//                 for (n2 = 0; n2 < n; n2++)
//                 {
//                     if (zf && k2 == 0)
//                         matrix_at(c, m2, n2) = (value_t) 0;

//                     matrix_at(c, m2, n2) +=
//                         matrix_at(a, m2, k2) * matrix_at(b, k2, n2);
//                 }

//     if (n < PyArray_SHAPE(b)[1])
//         for (m2 = 0; m2 < m; m2++)
//             for (k2 = 0; k2 < k; k2++)
//                 for (n2 = n; n2 < PyArray_SHAPE(b)[1]; n2++)
//                 {
//                     if (zf && k2 == 0)
//                         matrix_at(c, m2, n2) = (value_t) 0;

//                     matrix_at(c, m2, n2) +=
//                         matrix_at(a, m2, k2) * matrix_at(b, k2, n2);
//                 }

//     if (m < PyArray_SHAPE(a)[0] && n < PyArray_SHAPE(b)[1])
//         for (m2 = m; m2 < PyArray_SHAPE(a)[0]; m2++)
//             for (k2 = 0; k2 < k; k2++)
//                 for (n2 = n; n2 < PyArray_SHAPE(b)[1]; n2++)
//                 {
//                     if (zf && k2 == 0)
//                         matrix_at(c, m2, n2) = (value_t) 0;

//                     matrix_at(c, m2, n2) +=
//                         matrix_at(a, m2, k2) * matrix_at(b, k2, n2);
//                 }

//     if (k < PyArray_SHAPE(a)[1])
//         for (m2 = 0; m2 < PyArray_SHAPE(a)[0]; m2++)
//             for (k2 = k; k2 < PyArray_SHAPE(a)[1]; k2++)
//                 for (n2 = 0; n2 < PyArray_SHAPE(b)[1]; n2++)
//                     matrix_at(c, m2, n2) +=
//                         matrix_at(a, m2, k2) * matrix_at(b, k2, n2);
// }


// static void impl_abort(
//     PyArrayObject *a,
//     PyArrayObject *b,
//     PyArrayObject *c,
//     int zf
// )
// {
//     abort();
// }


// const impl_asm_t impl_asm[27] = {
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,

//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_asm_mkn,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,

//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
//     &impl_abort,
// };
