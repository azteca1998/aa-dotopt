#include "util.h"


sequential_version_t sv_find_version(
    size_t value_size,
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
)
{
    int value = 0;

    /* Find A's offset. */
    if (PyArray_STRIDE(a, 0) == value_size)
        value += 3 * 3 * 0;
    else if (PyArray_STRIDE(a, 1) == value_size)
        value += 3 * 3 * 1;
    else
        value += 3 * 3 * 2;

    /* Find B's offset. */
    if (PyArray_STRIDE(b, 0) == value_size)
        value += 3 * 0;
    else if (PyArray_STRIDE(b, 1) == value_size)
        value += 3 * 1;
    else
        value += 3 * 2;

    /* Find C's offset. */
    if (PyArray_STRIDE(c, 0) == value_size)
        value += 0;
    else if (PyArray_STRIDE(c, 1) == value_size)
        value += 1;
    else
        value += 2;

    return (sequential_version_t) value;
}
