#include "zorder.h"

#include <stdlib.h>


float *zorder_alloc(matrix_t *m)
{
    size_t n_rows;
    size_t n_cols;
    float *ptr;

    n_rows = next_power_of_two(m->num_rows - 1);
    n_cols = next_power_of_two(m->num_cols - 1);
    if (n_rows > n_cols)
        n_cols = n_rows;
    else if (n_cols > 2 * n_rows)
        n_rows = n_cols / 2;

    ptr = (float *) malloc(sizeof(float) * n_rows * n_cols);
    return ptr;
}

void zorder_free(float *ptr)
{
    free((void *) ptr);
}

void zorder_transform(float *target, matrix_t *m)
{
    size_t y, x;

    for (y = 0; y < m->num_rows; y++)
        for (x = 0; x < m->num_cols; x++)
            zorder_at(target, y, x) =
                *((float *) (m->data + m->row_stride * y + m->col_stride * x));
}

void zorder_transform_inverse(float *target, matrix_t *m)
{
    size_t y, x;

    for (y = 0; y < m->num_rows; y++)
        for (x = 0; x < m->num_cols; x++)
            *((float *) (m->data + m->row_stride * y + m->col_stride * x)) =
                zorder_at(target, y, x);
}
