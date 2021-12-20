#include "util.h"


sequential_version_t sv_find_version(
    size_t value_size,
    matrix_t *a,
    matrix_t *b,
    matrix_t *c
)
{
    int value = 0;

    /* Find A's offset. */
    if (a->row_stride == value_size)
        value += 3 * 3 * 0;
    else if (a->col_stride == value_size)
        value += 3 * 3 * 1;
    else
        value += 3 * 3 * 2;

    /* Find B's offset. */
    if (b->row_stride == value_size)
        value += 3 * 0;
    else if (b->col_stride == value_size)
        value += 3 * 1;
    else
        value += 3 * 2;

    /* Find C's offset. */
    if (c->row_stride == value_size)
        value += 0;
    else if (c->col_stride == value_size)
        value += 1;
    else
        value += 2;

    return (sequential_version_t) value;
}
