#include "imts.h"

#include <time.h>

#include "../imts.h"
#include "sequential.h"


#define value_t float


DOTOPT_API void impl_imts(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c
)
{
    imts_t imts[4];
    sequential_version_t sv;
    PyObject *slice_m, *slice_k, *slice_n;
    PyObject *index_a, *index_b, *index_c;
    PyObject *t0, *t1;
    size_t m, k, n;
    uint8_t zf;

    sv = sv_find_version(sizeof(value_t), a, b, c);

    index_a = PyTuple_New(2);
    index_b = PyTuple_New(2);
    index_c = PyTuple_New(2);

    imts_initialize_root(&imts[2], a, b, c, 512);
    imts_initialize_child(&imts[1], &imts[2], 128);
    imts_initialize_child(&imts[0], &imts[1], 32);

    while (imts_get_work(&imts[0], &m, &k, &n, &zf))
    {
        // slice_m <- m:min(m+imts.tile_len, imts.until_m)
        t0 = PyLong_FromSize_t(m);
        t1 = PyLong_FromSize_t(m + imts[0].tile_size <= imts[0].until_m
            ? m + imts[0].tile_size
            : imts[0].until_m
        );
        slice_m = PySlice_New(t0, t1, NULL);
        Py_DECREF(t0);
        Py_DECREF(t1);

        // slice_k <- k:min(k+imts.tile_len, imts.until_k)
        t0 = PyLong_FromSize_t(k);
        t1 = PyLong_FromSize_t(k + imts[0].tile_size <= imts[0].until_k
            ? k + imts[0].tile_size
            : imts[0].until_k
        );
        slice_k = PySlice_New(t0, t1, NULL);
        Py_DECREF(t0);
        Py_DECREF(t1);

        // slice_n <- n:min(n+imts.tile_len, imts.until_n)
        t0 = PyLong_FromSize_t(n);
        t1 = PyLong_FromSize_t(n + imts[0].tile_size <= imts[0].until_n
            ? n + imts[0].tile_size
            : imts[0].until_n
        );
        slice_n = PySlice_New(t0, t1, NULL);
        Py_DECREF(t0);
        Py_DECREF(t1);

        // index_a <- (slice_m, slice_k)
        PyTuple_SET_ITEM(index_a, 0, slice_m);
        PyTuple_SET_ITEM(index_a, 1, slice_k);
        
        // index_b <- (slice_k, slice_n)
        PyTuple_SET_ITEM(index_b, 0, slice_k);
        PyTuple_SET_ITEM(index_b, 1, slice_n);
        
        // index_c <- (slice_m, slice_n)
        PyTuple_SET_ITEM(index_c, 0, slice_m);
        PyTuple_SET_ITEM(index_c, 1, slice_n);

        (*impl_sequential[26])(
            (PyArrayObject *) PyObject_GetItem((PyObject *) a, index_a),
            (PyArrayObject *) PyObject_GetItem((PyObject *) b, index_b),
            (PyArrayObject *) PyObject_GetItem((PyObject *) c, index_c),
            zf
        );

        Py_DECREF(slice_m);
        Py_DECREF(slice_k);
        Py_DECREF(slice_n);
    }

    Py_DECREF(index_a);
    Py_DECREF(index_b);
    Py_DECREF(index_c);
}
