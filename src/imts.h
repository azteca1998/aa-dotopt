#ifndef DOTOPT__IMTS_H
#define DOTOPT__IMTS_H

#include <numpy/arrayobject.h>
#include <stddef.h>
#include <stdint.h>


typedef struct imts
{
    struct imts *parent;

    PyArrayObject *op_a;
    PyArrayObject *op_b;
    PyArrayObject *op_c;

    size_t since_m, since_k, since_n;
    size_t until_m, until_k, until_n;

    ssize_t chunk_size;
    ssize_t pos_m;
    ssize_t pos_k;
    ssize_t pos_n;

    uint8_t state;
    uint8_t dir_k;
    uint8_t dir_n;
} __attribute__((aligned(64))) imts_t;


void imts_initialize(
    imts_t *self,
    imts_t *parent,
    PyArrayObject *op_a,
    PyArrayObject *op_b,
    PyArrayObject *op_c,
    size_t chunk_size
);

int imts_get_work(imts_t *self, size_t *m, size_t *k, size_t *n);


#endif /* DOTOPT__IMTS_H */
