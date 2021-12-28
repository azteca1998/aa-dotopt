#include "imts.h"

#include <string.h>


void imts_init_root(
    imts_t *self,
    matrix_t *a,
    matrix_t *b,
    matrix_t *c,
    size_t tile_size,
    int zero_fill,
    int is_mt
)
{
    memset(self, 0x00, sizeof(imts_t));
    self->a = a;
    self->b = b;
    self->c = c;

    self->until_m = a->num_rows;
    self->until_k = a->num_cols;
    self->until_n = b->num_cols;

    self->tile_size = tile_size;
    if (zero_fill)
        self->flags |= 0x08;
    if (is_mt)
        self->flags |= 0x04;
}

void imts_init_child(imts_t *self, imts_t *parent, size_t tile_size, int is_mt)
{
    memset(self, 0x00, sizeof(imts_t));
    self->parent = parent;

    self->a = parent->a;
    self->b = parent->b;
    self->c = parent->c;

    self->tile_size = tile_size;
    if (is_mt)
        self->flags |= 0x04;
}

int imts_get_work(imts_t *self, size_t *m, size_t *k, size_t *n, int *zf)
{
    int tmp;
    int ret = 1;

    if ((self->flags & 0x04) == 0)
    {
        // Single-threaded version: Equivalent to MNK.

        // Check if done and maybe fetch work from parent.
        if (self->pos_m >= self->until_m)
        {
            if (self->parent == NULL)
                return 0;

            if ((ret = imts_get_work(
                self->parent,
                &self->pos_m,
                &self->pos_k,
                &self->pos_n,
                &tmp
            )) == 0)
                return 0;

            self->since_m = self->pos_m;
            self->since_k = self->pos_k;
            self->since_n = self->pos_n;

            self->until_m = self->since_m + self->parent->tile_size;
            self->until_k = self->since_k + self->parent->tile_size;
            self->until_n = self->since_n + self->parent->tile_size;
            if (self->until_m > self->parent->until_m)
                self->until_m = self->parent->until_m;
            if (self->until_k > self->parent->until_k)
                self->until_k = self->parent->until_k;
            if (self->until_n > self->parent->until_n)
                self->until_n = self->parent->until_n;

            self->flags &= ~0x03;
            if (tmp)
                self->flags |= 0x08;
            else
                self->flags &= ~0x08;
        }

        // Assign work.
        *m = self->pos_m;
        *k = self->pos_k;
        *n = self->pos_n;
        *zf = (self->flags & 0x18) == 0x08;
        self->flags |= 0x10;

        // Iterate over K.
        if ((self->flags & 0x01) == 0)
            self->pos_k += self->tile_size;
        else
            self->pos_k -= self->tile_size;

        // Iterate over N.
        if (self->pos_k >= self->until_k || self->pos_k < self->since_k)
        {
            self->flags ^= 0x01;
            self->flags &= ~0x10;

            if ((self->flags & 0x02) == 0)
                self->pos_n += self->tile_size;
            else
                self->pos_n -= self->tile_size;

            if ((self->flags & 0x01) == 0)
                self->pos_k = self->since_k;
            else
                self->pos_k -= self->tile_size;
        }

        // Iterate over M.
        if (self->pos_n >= self->until_n || self->pos_n < self->since_n)
        {
            self->flags ^= 0x02;
            self->pos_m += self->tile_size;

            if ((self->flags & 0x02) == 0)
                self->pos_n = self->since_n;
            else
                self->pos_n -= self->tile_size;
        }
    }
    else
    {
        // Multi-threaded version: Equivalent to KMN.

        // Check if done and maybe fetch work from parent.
        if (self->pos_k >= self->until_k)
        {
            if (self->parent == NULL)
                return 0;

            if (imts_get_work(
                self->parent,
                &self->pos_m,
                &self->pos_k,
                &self->pos_n,
                &tmp
            ) == 0)
                return 0;

            self->since_m = self->pos_m;
            self->since_k = self->pos_k;
            self->since_n = self->pos_n;

            self->until_m = self->since_m + self->parent->tile_size;
            self->until_k = self->since_k + self->parent->tile_size;
            self->until_n = self->since_n + self->parent->tile_size;
            if (self->until_m > self->parent->until_m)
                self->until_m = self->parent->until_m;
            if (self->until_k > self->parent->until_k)
                self->until_k = self->parent->until_k;
            if (self->until_n > self->parent->until_n)
                self->until_n = self->parent->until_n;

            self->flags &= ~0x03;
            if (tmp)
                self->flags |= 0x08;
            else
                self->flags &= ~0x08;
        }

        // Assign work.
        *m = self->pos_m;
        *k = self->pos_k;
        *n = self->pos_n;
        *zf = (self->flags & 0x08) && self->pos_k == 0;

        // Iterate over N.
        if ((self->flags & 0x01) == 0)
            self->pos_n += self->tile_size;
        else
            self->pos_n -= self->tile_size;

        // Iterate over M.
        if (self->pos_n >= self->until_n || self->pos_n < self->since_n)
        {
            self->flags ^= 0x01;
            if ((self->flags & 0x02) == 0)
                self->pos_m += self->tile_size;
            else
                self->pos_m -= self->tile_size;

            if ((self->flags & 0x01) == 0)
                self->pos_n = self->since_n;
            else
                self->pos_n -= self->tile_size;

            ret = 2;
        }

        // Iterate over K.
        if (self->pos_m >= self->until_m || self->pos_m < self->since_m)
        {
            self->flags ^= 0x02;
            self->pos_k += self->tile_size;

            if ((self->flags & 0x02) == 0)
                self->pos_m = self->since_m;
            else
                self->pos_m -= self->tile_size;

            ret = 2;
        }
    }

    return ret;
}
