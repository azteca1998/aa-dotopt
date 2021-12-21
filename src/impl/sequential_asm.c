#include "sequential_asm.h"

#include <stdlib.h>
#include <string.h>

#include "sequential.h"
#include "../zorder.h"


static void impl_x1_x1_x1(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
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
}

static void impl_zz_zz_zz(
    matrix_t *a,
    matrix_t *b,
    matrix_t *c,
    int zero_fill
)
{
    size_t m, k, n;
    _impl_asm_zorder_t *impl_ptr;
    matrix_t ta, tb, tc;

    for (m = 0; m < (a->num_rows & ~7); m += 8)
        for (k = 0; k < (a->num_cols & ~7); k += 8)
            for (n = 0; n < (b->num_cols & ~7); n += 8)
            {
                impl_ptr = zero_fill && k == 0
                    ? &_impl_sequential_asm_zz_zz_zz_zf
                    : &_impl_sequential_asm_zz_zz_zz;

                (*impl_ptr)(
                    &zorder_at(a->data, m, k),
                    &zorder_at(b->data, k, n),
                    &zorder_at(c->data, m, n)
                );
            }

    // TODO: Compute margin.
}


static void impl_abort(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
    abort();
}


const impl_sequential_asm_t impl_sequential_asm[28] = {
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

    &impl_zz_zz_zz,
};
