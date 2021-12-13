#include "imts.h"

#include <assert.h>


void imts_initialize_root(
    imts_t *self,
    PyArrayObject *op_a,
    PyArrayObject *op_b,
    PyArrayObject *op_c,
    size_t tile_size
)
{
    self->parent = NULL;                    /* Root has no parent.   */
    self->op_a = op_a;                      /* Left operand.         */
    self->op_b = op_b;                      /* Right operand.        */
    self->op_c = op_c;                      /* Output array.         */

    self->since_m = 0;                      /* Root's m starts at 0. */
    self->since_k = 0;                      /* Root's k starts at 0. */
    self->since_n = 0;                      /* Root's n starts at 0. */
    self->until_m = PyArray_SHAPE(op_a)[0]; /* Root's m ends at M.   */
    self->until_k = PyArray_SHAPE(op_a)[1]; /* Root's k ends at K.   */
    self->until_n = PyArray_SHAPE(op_b)[1]; /* Root's n ends at N.   */

    self->tile_size = tile_size;            /* Tile size.            */
    self->pos_m = self->since_m;            /* Initial m value.      */
    self->pos_k = self->since_k;            /* Initial k value.      */
    self->pos_n = self->since_n;            /* Initial n value.      */

    self->state = is_ready;                 /* Tiler is ready.       */
    self->next_is_zero_fill = 1;
    self->parent_is_zero_fill = 1;
    self->dir_k = 0;                        /* 0 -> +1.              */
    self->dir_n = 0;                        /* 0 -> +1.              */

    pthread_spin_init(&self->lock, 0);
}

void imts_initialize_child(
    imts_t *self,
    imts_t *parent,
    size_t tile_size
)
{
    self->parent = parent;                  /* Parent of this tiler. */
    self->op_a = parent->op_a;              /* Copied from parent.   */
    self->op_b = parent->op_b;              /* Copied from parent.   */
    self->op_c = parent->op_c;              /* Copied from parent.   */

    self->since_m = 0;                      /* Will be overwritten.  */
    self->since_k = 0;                      /* Will be overwritten.  */
    self->since_n = 0;                      /* Will be overwritten.  */
    self->until_m = 0;                      /* Will be overwritten.  */
    self->until_k = 0;                      /* Will be overwritten.  */
    self->until_n = 0;                      /* Will be overwritten.  */

    self->tile_size = tile_size;
    self->pos_m = 0;                        /* Will be overwritten.  */
    self->pos_k = 0;                        /* Will be overwritten.  */
    self->pos_n = 0;                        /* Will be overwritten.  */

    self->state = is_needs_work;            /* Fetch from parent.    */
    self->next_is_zero_fill = 0;            /* Will be overwritten.  */
    self->parent_is_zero_fill = 0;          /* Will be overwritten.  */
    self->dir_k = 0;                        /* Will be overwritten.  */
    self->dir_n = 0;                        /* Will be overwritten.  */

    pthread_spin_init(&self->lock, 0);
}

// TODO: Guard critical section for threaded versions.
int imts_get_work(imts_t *self, size_t *m, size_t *k, size_t *n, uint8_t *zf)
{
    assert(self != NULL);

    pthread_spin_lock(&self->lock);

    // Check if done or the tiler should fetch work from the parent.
    if (self->state == is_needs_work || self->pos_m >= (size_t) self->until_m)
    {
        if (self->parent == NULL)
            goto done;

        self->state = is_ready;

        // Get work from the parent.
        if (imts_get_work(
            self->parent,
            &self->pos_m,
            &self->pos_k,
            &self->pos_n,
            &self->parent_is_zero_fill
        ) == 0)
            goto done;

        // Reset boundaries.
        #define min(a, b) (((a) <= (b)) ? (a) : (b))
        self->since_m = self->pos_m;
        self->since_k = self->pos_k;
        self->since_n = self->pos_n;
        self->until_m = min(
            self->since_m + self->parent->tile_size,
            self->parent->until_m
        );
        self->until_k = min(
            self->since_k + self->parent->tile_size,
            self->parent->until_k
        );
        self->until_n = min(
            self->since_n + self->parent->tile_size,
            self->parent->until_n
        );
        #undef min

        self->dir_k = 0;
        self->dir_n = 0;

        self->next_is_zero_fill = 1;
    }

    // Assign work.
    *m = self->pos_m;
    *k = self->pos_k;
    *n = self->pos_n;
    if (self->parent_is_zero_fill && zf != NULL)
        *zf = self->next_is_zero_fill;

    self->next_is_zero_fill = 0;

    // Iterate over K.
    self->pos_k += !self->dir_k
        ? self->tile_size
        : -self->tile_size;

    // Iterate over N.
    if (
        self->pos_k < (ssize_t) self->since_k ||
        self->pos_k >= (ssize_t) self->until_k
    )
    {
        self->pos_k = self->pos_k < (ssize_t) self->since_k
            ? self->since_k
            : self->pos_k - self->tile_size;
        self->pos_n += !self->dir_n
            ? self->tile_size
            : -self->tile_size;

        self->dir_k = !self->dir_k;
        self->next_is_zero_fill = 1;
    }

    // Iterate over M.
    if (
        self->pos_n < (ssize_t) self->since_n ||
        self->pos_n >= (ssize_t) self->until_n
    )
    {
        self->pos_n = self->pos_n < (ssize_t) self->since_n
            ? self->since_n
            : self->pos_n - self->tile_size;
        self->pos_m += self->tile_size;

        self->dir_n = !self->dir_n;
    }

    pthread_spin_unlock(&self->lock);
    return 1;

  done:
    pthread_spin_unlock(&self->lock);
    return 0;
}
