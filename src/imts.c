#include "imts.h"

#include <assert.h>


void imts_initialize(
    imts_t *self,
    imts_t *parent,
    PyArrayObject *op_a,
    PyArrayObject *op_b,
    PyArrayObject *op_c,
    size_t chunk_size
)
{
    assert(self != NULL);
    assert(op_a != NULL);
    assert(op_b != NULL);
    assert(op_c != NULL);

    self->parent = parent;

    self->op_a = op_a;
    self->op_b = op_b;
    self->op_c = op_c;

    self->since_m = 0;
    self->since_k = 0;
    self->since_n = 0;
    self->until_m = 0;
    self->until_k = 0;
    self->until_n = 0;

    self->chunk_size = chunk_size;
    self->pos_m = 0;
    self->pos_k = 0;
    self->pos_n = 0;

    self->state = 0;
    self->dir_k = 0;
    self->dir_n = 0;
}

int imts_get_work(imts_t *self, size_t *m, size_t *k, size_t *n)
{
    assert(self != NULL);

    // Initialize.
    if (self->state == 0)
    {
        self->state = 1;

        if (self->parent == NULL)
        {
            self->until_m = PyArray_SHAPE(self->op_a)[0];
            self->until_k = PyArray_SHAPE(self->op_a)[1];
            self->until_n = PyArray_SHAPE(self->op_b)[1];
        }

        self->pos_m = self->since_m;
        self->pos_k = self->since_k;
        self->pos_n = self->since_n;
        self->dir_k = 0;
        self->dir_n = 0;
    }

    // Check if done.
    if (self->pos_m >= self->until_m)
    {
        if (self->parent == NULL)
            return 0;

        // TODO: Reset with parent work item.
        assert(false);
    }

    // Assign work.
    *m = self->pos_m;
    *k = self->pos_k;
    *n = self->pos_n;

    // Iterate over K.
    self->pos_k += !self->dir_k
        ? self->chunk_size
        : -self->chunk_size;

    // Iterate over N.
    if (self->pos_k < self->since_k || self->pos_k >= self->until_k)
    {
        self->pos_k = self->pos_k < (ssize_t) self->since_k
            ? self->since_k
            : self->until_k - 1;
        self->pos_n += !self->dir_n
            ? self->chunk_size
            : -self->chunk_size;

        self->dir_k = !self->dir_k;
    }

    // Iterate over M.
    if (self->pos_n < self->since_n || self->pos_n >= self->until_n)
    {
        self->pos_n = self->pos_n < (ssize_t) self->since_n
            ? self->since_n
            : self->until_n - 1;
        self->pos_m += self->chunk_size;

        self->dir_n = !self->dir_n;
    }

    return 1;
}
