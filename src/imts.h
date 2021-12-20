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



// #ifndef DOTOPT__IMTS_H
// #define DOTOPT__IMTS_H

// #include <numpy/arrayobject.h>
// #include <pthread.h>
// #include <stddef.h>
// #include <stdint.h>


// // TODO: IMTS for threaded (separate writes to the same location).
// //    M, N, K
// //   (0, 0, 0)
// //   (0, 1, 0)
// //   (1, 1, 0)
// //   (1, 0, 0)
// //   (1, 0, 1)
// //   (1, 1, 1)
// //   (0, 1, 1)
// //   (0, 0, 1)
// // TODO: IMTS for sequential (join writes to the same location).
// //    M, N, K
// //   (0, 0, 0)
// //   (0, 0, 1)
// //   (0, 1, 0)
// //   (0, 1, 1)
// //   (1, 1, 0)
// //   (1, 1, 1)
// //   (1, 0, 0)
// //   (1, 0, 1)



// typedef enum imts_state
// {
//     is_uninitialized = 0,   /* Tiler has not been initialized.      */
//     is_ready         = 1,   /* Tiler is ready to serve work.        */
//     is_needs_work    = 2,   /* Tiler should fetch work from parent. */
// } imts_state_t;

// /**
//  * @brief Intelligent Multilevel Tiling Scheduler.
//  * 
//  * @code {.c}
//  * imts_t imts[2];
//  * 
//  * imts_initialize_root(&imts[1], a, b, c, tile_size_1);
//  * imts_initialize_child(&imts[0], &imts[1], tile_size_0);
//  * 
//  * while (imts_get_work(&imts[0], &m, &k, &n))
//  * {
//  *     // Tile dimensions:
//  *     //   M <- [m, min(m + imts[0]->tile_size, imts[0]->until_m))
//  *     //   K <- [k, min(k + imts[0]->tile_size, imts[0]->until_k))
//  *     //   N <- [n, min(n + imts[0]->tile_size, imts[0]->until_n))
//  * }
//  * @endcode
//  */
// typedef struct imts
// {
//     struct imts *parent;

//     PyArrayObject *op_a;
//     PyArrayObject *op_b;
//     PyArrayObject *op_c;

//     size_t since_m, since_k, since_n;
//     size_t until_m, until_k, until_n;

//     ssize_t tile_size;
//     ssize_t pos_m;
//     ssize_t pos_k;
//     ssize_t pos_n;

//     imts_state_t state;
//     uint8_t parent_is_zero_fill;
//     uint8_t next_is_zero_fill;

//     uint8_t dir_k;
//     uint8_t dir_n;

//     /* Aligned to a different cache line to avoid false sharing. */
//     pthread_spinlock_t __attribute__((aligned(64))) lock;
// } __attribute__((aligned(64))) imts_t;


// void imts_initialize_root(
//     imts_t *self,
//     PyArrayObject *op_a,
//     PyArrayObject *op_b,
//     PyArrayObject *op_c,
//     size_t tile_size
// );
// void imts_initialize_child(
//     imts_t *self,
//     imts_t *parent,
//     size_t tile_size
// );

// int imts_get_work(imts_t *self, size_t *m, size_t *k, size_t *n, uint8_t *zf);


// #endif /* DOTOPT__IMTS_H */
