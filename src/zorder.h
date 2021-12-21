#ifndef DOTOPT__ZORDER_H
#define DOTOPT__ZORDER_H

#include <stddef.h>
#include <stdint.h>

#include "impl/util.h"


#define zorder_at(ptr, y, x) \
    (*_zorder_at((ptr), (y), (x)))


inline static size_t next_power_of_two(size_t value)
{
    return 1 << sizeof(value) * 8 - __builtin_clzll(value);
}

inline static float *_zorder_at(float *ptr, uint16_t y, uint16_t x)
{
    uint32_t t0 = y;
    uint32_t t1 = x;

    t0 = (t0 | (t0 << 8)) & 0x00FF00FF;
    t0 = (t0 | (t0 << 4)) & 0x0F0F0F0F;
    t0 = (t0 | (t0 << 2)) & 0x33333333;
    t0 = (t0 | (t0 << 1)) & 0x55555555;

    t1 = (t1 | (t1 << 8)) & 0x00FF00FF;
    t1 = (t1 | (t1 << 4)) & 0x0F0F0F0F;
    t1 = (t1 | (t1 << 2)) & 0x33333333;
    t1 = (t1 | (t1 << 1)) & 0x55555555;

    return ptr + (t1 | (t0 << 1));
}

DOTOPT_API float *zorder_alloc(matrix_t *m);
DOTOPT_API void zorder_free(float *ptr);

DOTOPT_API void zorder_transform(float *target, matrix_t *m);
DOTOPT_API void zorder_transform_inverse(float *target, matrix_t *m);


#endif /* DOTOPT__ZORDER_H */
