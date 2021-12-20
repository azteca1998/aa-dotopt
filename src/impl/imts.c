#include "imts.h"

#include <stdlib.h>

#include "../imts.h"
#include "sequential.h"
#include "sequential_asm.h"


static void impl_x1_x1_x1(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
    size_t m, k, n;
    matrix_t ta, tb, tc;
    imts_t imts[3];

    imts_init_root(&imts[2], a, b, c, 512, zero_fill, 0);
    imts_init_child(&imts[1], &imts[2], 128, 0);
    imts_init_child(&imts[0], &imts[1], 32, 0);

    ta.row_stride = a->row_stride;
    ta.col_stride = a->col_stride;
    tb.row_stride = b->row_stride;
    tb.col_stride = b->col_stride;
    tc.row_stride = c->row_stride;
    tc.col_stride = c->col_stride;

    while (imts_get_work(&imts[0], &m, &k, &n, &zero_fill))
    {
        ta.data = &((float *) (a->data + m * a->row_stride))[k];
        tb.data = &((float *) (b->data + k * b->row_stride))[n];
        tc.data = &((float *) (c->data + m * c->row_stride))[n];

        ta.num_rows = imts[0].tile_size;
        ta.num_cols = imts[0].tile_size;
        tb.num_cols = imts[0].tile_size;
        if (m + ta.num_rows > imts[0].until_m)
            ta.num_rows = imts[0].until_m - m;
        if (k + ta.num_cols > imts[0].until_k)
            ta.num_cols = imts[0].until_k - k;
        if (n + tb.num_cols > imts[0].until_n)
            tb.num_cols = imts[0].until_n - n;

        tb.num_rows = ta.num_cols;
        tc.num_rows = ta.num_rows;
        tc.num_cols = tb.num_cols;

        (*impl_sequential[sv_x1_x1_x1])(&ta, &tb, &tc, zero_fill);
    }
}


static void impl_x1_x1_x1_asm(
    matrix_t *a,
    matrix_t *b,
    matrix_t *c,
    int zero_fill
)
{
    size_t m, k, n;
    matrix_t ta, tb, tc;
    imts_t imts[4];
    _impl_asm_t *impl_ptr;

    imts_init_root(&imts[3], a, b, c, 512, zero_fill, 0);
    imts_init_child(&imts[2], &imts[3], 128, 0);
    imts_init_child(&imts[1], &imts[2], 32, 0);
    imts_init_child(&imts[0], &imts[1], 8, 0);

    ta.row_stride = a->row_stride;
    ta.col_stride = a->col_stride;
    tb.row_stride = b->row_stride;
    tb.col_stride = b->col_stride;
    tc.row_stride = c->row_stride;
    tc.col_stride = c->col_stride;

    while (imts_get_work(&imts[0], &m, &k, &n, &zero_fill))
    {
        if (m + imts[0].tile_size <= imts[0].until_m
            && k + imts[0].tile_size <= imts[0].until_k
            && n + imts[0].tile_size <= imts[0].until_n)
        {
            impl_ptr = zero_fill
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
        else
        {
            ta.data = &((float *) (a->data + m * a->row_stride))[k];
            tb.data = &((float *) (b->data + k * b->row_stride))[n];
            tc.data = &((float *) (c->data + m * c->row_stride))[n];

            ta.num_rows = imts[0].tile_size;
            ta.num_cols = imts[0].tile_size;
            tb.num_cols = imts[0].tile_size;
            if (m + ta.num_rows > imts[0].until_m)
                ta.num_rows = imts[0].until_m - m;
            if (k + ta.num_cols > imts[0].until_k)
                ta.num_cols = imts[0].until_k - k;
            if (n + tb.num_cols > imts[0].until_n)
                tb.num_cols = imts[0].until_n - n;

            tb.num_rows = ta.num_cols;
            tc.num_rows = ta.num_rows;
            tc.num_cols = tb.num_cols;

            (*impl_sequential[sv_x1_x1_x1])(&ta, &tb, &tc, zero_fill);
        }
    }
}


static void impl_abort(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
    abort();
}


const impl_imts_sequential_t impl_imts_sequential[27] = {
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

const impl_imts_sequential_asm_t impl_imts_sequential_asm[27] = {
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
    &impl_x1_x1_x1_asm,
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
