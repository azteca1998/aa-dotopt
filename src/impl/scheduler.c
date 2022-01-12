#include "scheduler.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>

#include "../imts.h"
#include "../zorder.h"
#include "./sequential.h"


typedef struct thread_data
{
    pthread_t thread;

    matrix_t *a;
    matrix_t *b;
    matrix_t *c;

    imts_t imts;
} __attribute__((aligned(64))) thread_data_t;


static void *run_zz_zz_zz(void *);


static void impl_zz_zz_zz(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
    /*
     * CPUs:
     *   0, 8  -> L30, L20, L10
     *   1, 9  -> L30, L21, L11
     *   2, 10 -> L30, L22, L12
     *   3, 11 -> L30, L23, L13
     *   4, 12 -> L31, L24, L14
     *   5, 13 -> L31, L25, L15
     *   6, 14 -> L31, L26, L16
     *   5, 15 -> L31, L27, L17
     *
     * IMTS distribution:
     *   1x IMTS-L1 per core (1 per thread)
     *   1x IMTS-L2 per core (async)
     *   2x IMTS-L3 per CPU (async, 1 per L3)
     *
     * IMTS distribution (ASM):
     *   2x IMTS-L0 per core (1 per thread)
     *   1x IMTS-L1 per core (async)
     *   1x IMTS-L2 per core (async)
     *   2x IMTS-L3 per CPU (async, 1 per L3)
     */

    cpu_set_t cpu_set;
    thread_data_t thread_data[16];
    pthread_attr_t thread_attr;
    imts_t imts_l3;
    imts_t imts_l2[2];
    size_t i;

    imts_init_root(&imts_l3, a, b, c, 512, zero_fill, 2);
    imts_init_child(&imts_l2[0], &imts_l3, 128, 8);
    imts_init_child(&imts_l2[1], &imts_l3, 128, 8);

    pthread_attr_init(&thread_attr);

    for (i = 0; i < 16; i++)
    {
        CPU_ZERO(&cpu_set);
        CPU_SET(i, &cpu_set);
        pthread_attr_setaffinity_np(&thread_attr, sizeof(cpu_set_t), &cpu_set);

        thread_data[i].a = a;
        thread_data[i].b = b;
        thread_data[i].c = c;

        imts_init_child(&thread_data[i].imts, &imts_l2[(i / 4) % 2], 32, 0);

        // TODO: Error handling.
        pthread_create(
            &thread_data[i].thread,
            &thread_attr,
            &run_zz_zz_zz,
            &thread_data[i]
        );
    }

    for (i = 0; i < 16; i++)
        pthread_join(thread_data[i].thread, NULL);
}

static void *run_zz_zz_zz(void *ptr)
{
    thread_data_t *data = (thread_data_t *) ptr;
    size_t m, k, n;
    int zero_fill;
    matrix_t ta, tb, tc;

    ta.row_stride = data->a->row_stride;
    ta.col_stride = data->a->col_stride;
    tb.row_stride = data->b->row_stride;
    tb.col_stride = data->b->col_stride;
    tc.row_stride = data->c->row_stride;
    tc.col_stride = data->c->col_stride;

    while (imts_get_work(&data->imts, &m, &k, &n, &zero_fill))
    {
        ta.data = &zorder_at((float *) data->a->data, m, k);
        tb.data = &zorder_at((float *) data->b->data, k, n);
        tc.data = &zorder_at((float *) data->c->data, m, n);

        ta.num_rows = data->imts.tile_size;
        ta.num_cols = data->imts.tile_size;
        tb.num_cols = data->imts.tile_size;
        if (m + ta.num_rows > data->imts.until_m)
            ta.num_rows = data->imts.until_m - m;
        if (k + ta.num_cols > data->imts.until_k)
            ta.num_cols = data->imts.until_k - k;
        if (n + tb.num_cols > data->imts.until_n)
            tb.num_cols = data->imts.until_n - n;

        tb.num_rows = ta.num_cols;
        tc.num_rows = ta.num_rows;
        tc.num_cols = tb.num_cols;

        (*impl_sequential[sv_zz_zz_zz])(&ta, &tb, &tc, zero_fill);
    }
}


static void impl_abort(matrix_t *a, matrix_t *b, matrix_t *c, int zero_fill)
{
    abort();
}


const impl_scheduler_t impl_scheduler[28] = {
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
    &impl_abort,

    &impl_zz_zz_zz,
};
