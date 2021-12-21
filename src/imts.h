#ifndef DOTOPT__IMTS_H
#define DOTOPT__IMTS_H

#include <stddef.h>
#include <stdint.h>

#include "impl/util.h"


typedef struct imts
{
    struct imts *parent;

    matrix_t *a;
    matrix_t *b;
    matrix_t *c;

    size_t since_m, since_k, since_n;
    size_t until_m, until_k, until_n;

    size_t tile_size;
    size_t pos_m, pos_k, pos_n;

    uint8_t flags;
} __attribute__((aligned(64))) imts_t;


void imts_init_root(
    imts_t *self,
    matrix_t *a,
    matrix_t *b,
    matrix_t *c,
    size_t tile_size,
    int zero_fill,
    int is_mt
);
void imts_init_child(imts_t *self, imts_t *parent, size_t tile_size, int is_mt);

int imts_get_work(imts_t *self, size_t *m, size_t *k, size_t *n, int *zf);


#endif /* DOTOPT__IMTS_H */
