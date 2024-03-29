#include "sequential.h"

#include <omp.h>
#include <stdlib.h>

#include "../zorder.h"


static void impl_x1_x1_x1(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
#define matrix_at(m, row, col) \
    (((float *) ((m)->data + (m)->row_stride * (row)))[(col)])

    size_t m, k, n;

    for (m = 0; m < a->num_rows; m++)
        for (k = 0; k < a->num_cols; k++)
        {
            if (zero_fill && k == 0)
                for (n = 0; n < b->num_cols; n++)
                    matrix_at(c, m, n) = (float) 0;

            for (n = 0; n < b->num_cols; n++)
                matrix_at(c, m, n) += matrix_at(a, m, k) * matrix_at(b, k, n);
        }

#undef matrix_at
}

static void impl_zz_zz_zz(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
    size_t m, k, n;

    for (m = 0; m < a->num_rows; m++)
        for (n = 0; n < b->num_cols; n++)
        {
            if (zero_fill)
                zorder_at(c->data, m, n) = (float) 0;

            for (k = 0; k < a->num_cols; k++)
                zorder_at(c->data, m, n) +=
                    zorder_at(a->data, m, k) * zorder_at(b->data, k, n);
        }
}


static void impl_omp_loops_x1_x1_x1(
    matrix_t *a,
    matrix_t *b,
    matrix_t *c,
    int zero_fill
)
{
#define matrix_at(m, row, col) \
    (((float *) ((m)->data + (m)->row_stride * (row)))[(col)])

    size_t m, k, n;

    if (zero_fill)
        #pragma omp for private(m, k, n) collapse(2)
        for (m = 0; m < a->num_rows; m++)
            for (n = 0; n < b->num_cols; n++)
                matrix_at(c, m, n) = (float) 0;

    #pragma omp for private(m, k, n)
    for (m = 0; m < a->num_rows; m++)
        for (k = 0; k < a->num_cols; k++)
            for (n = 0; n < b->num_cols; n++)
                matrix_at(c, m, n) += matrix_at(a, m, k) * matrix_at(b, k, n);

#undef matrix_at
}

static void impl_omp_loops_zz_zz_zz(
    matrix_t *a,
    matrix_t *b,
    matrix_t *c,
    int zero_fill
)
{
    size_t m, k, n;

    #pragma omp for private(m, k, n) collapse(2)
    for (m = 0; m < a->num_rows; m++)
        for (n = 0; n < b->num_cols; n++)
            zorder_at(c->data, m, n) = (float) 0;

    #pragma omp for private(m, k, n) collapse(3)
    for (m = 0; m < a->num_rows; m++)
        for (n = 0; n < b->num_cols; n++)
            for (k = 0; k < a->num_cols; k++)
                zorder_at(c->data, m, n) +=
                    zorder_at(a->data, m, k) * zorder_at(b->data, k, n);
}


static void impl_abort(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
    abort();
}


const impl_sequential_t impl_sequential[28] = {
    &impl_abort,          /* nkm, knm; nkm is 4-5% better */
    &impl_abort,          /* FIXME: WTF */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* knm, nkm; knm is 4-5% better */
    &impl_abort,          /* mkn, kmn; apparently, kmn is better at 1024, but mkn wins at 2048 */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */

    &impl_abort,          /* nkm, nmk; they perform equally */
    &impl_abort,          /* mnk, nmk; they perform equally, but nmk misses twice as much */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* FIXME: WTF */
    &impl_x1_x1_x1,       /* mkn, kmn; they perform equally, but kmn misses 1,5-2x as much */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */

    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */
    &impl_abort,          /* TODO: Complex arrays. */

    &impl_zz_zz_zz,
};

const impl_sequential_t impl_omp_loops_sequential[28] = {
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
    &impl_omp_loops_x1_x1_x1,
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

    &impl_omp_loops_zz_zz_zz,
};
