#include "openmp.h"

#include <omp.h>

#include "sequential.h"


static const size_t TASKS_MAX_SIZE = 32;


static void _impl_openmp_tasks_inner(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c,
    size_t tile_size,
    sequential_version_t sv,
    int zero_fill
);

void impl_openmp_loops(PyArrayObject *a, PyArrayObject *b, PyArrayObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    size_t m_sz = PyArray_SHAPE(a)[0];          /* Rows of A and C.         */
    size_t k_sz = PyArray_SHAPE(a)[1];          /* Cols of A and rows of B. */
    size_t n_sz = PyArray_SHAPE(b)[1];          /* Cols of B and C.         */

    #pragma omp parallel for shared(a, b, c) schedule(static) collapse(2)
    for (size_t m = 0; m < m_sz; m++)           /* Iterate over M.          */
        for (size_t n = 0; n < n_sz; n++)       /* Iterate over N.          */
        {
            value_t acc = (value_t) 0;
            
            #pragma omp reduction(+:acc)
            for (size_t k = 0; k < k_sz; k++)   /* Iterate over K.          */
                acc += matrix_at(a, m, k) * matrix_at(b, k, n);

            matrix_at(c, m, n) = acc;
        }

#undef matrix_at
#undef IMPL_TYPE
}

void impl_openmp_tasks(PyArrayObject *a, PyArrayObject *b, PyArrayObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    sequential_version_t sv;

    size_t m_sz = PyArray_SHAPE(a)[0];          /* Rows of A and C.         */
    size_t k_sz = PyArray_SHAPE(a)[1];          /* Cols of A and rows of B. */
    size_t n_sz = PyArray_SHAPE(b)[1];          /* Cols of B and C.         */

    sv = sv_find_version(sizeof(value_t), a, b, c);

    #pragma omp parallel
    #pragma omp master 
    _impl_openmp_tasks_inner(
        a, b, c,
        TASKS_MAX_SIZE,
        sv,
        1
    );

#undef matrix_at
#undef IMPL_TYPE
}

static void _impl_openmp_tasks_inner(
    PyArrayObject *a,
    PyArrayObject *b,
    PyArrayObject *c,
    size_t tile_size,
    sequential_version_t sv,
    int zero_fill
)
{
    PyObject *idx_a, *idx_b, *idx_c;
    PyObject *m0, *m1, *k0, *k1, *n0, *n1;
    PyObject *t0, *t1;

    if (tile_size <= TASKS_MAX_SIZE)
    {
        (*impl_sequential[sv])(a, b, c, zero_fill);
        return;
    }

    idx_a = PyTuple_New(2);
    idx_b = PyTuple_New(2);
    idx_c = PyTuple_New(2);
    t0 = PyLong_FromSize_t(0);

    // m0 <- 0:M/2
    // m1 <- M/2:
    t1 = PyLong_FromSize_t(PyArray_SHAPE(a)[0] / 2);
    m0 = PySlice_New(t0, t1, NULL);
    m1 = PySlice_New(t1, NULL, NULL);
    Py_DECREF(t1);

    // k0 <- 0:K/2
    // k1 <- K/2:
    t1 = PyLong_FromSize_t(PyArray_SHAPE(a)[1] / 2);
    k0 = PySlice_New(t0, t1, NULL);
    k1 = PySlice_New(t1, NULL, NULL);
    Py_DECREF(t1);

    // n0 <- 0:N/2
    // n1 <- N/2:
    t1 = PyLong_FromSize_t(PyArray_SHAPE(b)[1] / 2);
    n0 = PySlice_New(t0, t1, NULL);
    n1 = PySlice_New(t1, NULL, NULL);
    Py_DECREF(t1);

    Py_DECREF(t0);

    PyTuple_SET_ITEM(idx_a, 0, m0);
    PyTuple_SET_ITEM(idx_a, 1, k0);
    PyTuple_SET_ITEM(idx_b, 0, k0);
    PyTuple_SET_ITEM(idx_b, 1, n0);
    PyTuple_SET_ITEM(idx_c, 0, m0);
    PyTuple_SET_ITEM(idx_c, 1, n0);
    #pragma omp task
    _impl_openmp_tasks_inner(
        (PyArrayObject *) PyObject_GetItem((PyObject *) a, idx_a),
        (PyArrayObject *) PyObject_GetItem((PyObject *) b, idx_b),
        (PyArrayObject *) PyObject_GetItem((PyObject *) c, idx_c),
        tile_size, sv, zero_fill
    );

    PyTuple_SET_ITEM(idx_b, 1, n1);
    PyTuple_SET_ITEM(idx_c, 1, n1);
    #pragma omp task
    _impl_openmp_tasks_inner(
        (PyArrayObject *) PyObject_GetItem((PyObject *) a, idx_a),
        (PyArrayObject *) PyObject_GetItem((PyObject *) b, idx_b),
        (PyArrayObject *) PyObject_GetItem((PyObject *) c, idx_c),
        tile_size, sv, zero_fill
    );

    PyTuple_SET_ITEM(idx_a, 0, m1);
    PyTuple_SET_ITEM(idx_c, 0, m1);
    #pragma omp task
    _impl_openmp_tasks_inner(
        (PyArrayObject *) PyObject_GetItem((PyObject *) a, idx_a),
        (PyArrayObject *) PyObject_GetItem((PyObject *) b, idx_b),
        (PyArrayObject *) PyObject_GetItem((PyObject *) c, idx_c),
        tile_size, sv, zero_fill
    );

    PyTuple_SET_ITEM(idx_b, 1, n0);
    PyTuple_SET_ITEM(idx_c, 1, n0);
    #pragma omp task
    _impl_openmp_tasks_inner(
        (PyArrayObject *) PyObject_GetItem((PyObject *) a, idx_a),
        (PyArrayObject *) PyObject_GetItem((PyObject *) b, idx_b),
        (PyArrayObject *) PyObject_GetItem((PyObject *) c, idx_c),
        tile_size, sv, zero_fill
    );

    #pragma omp taskwait

    PyTuple_SET_ITEM(idx_a, 1, k1);
    PyTuple_SET_ITEM(idx_b, 0, k1);
    #pragma omp task
    _impl_openmp_tasks_inner(
        (PyArrayObject *) PyObject_GetItem((PyObject *) a, idx_a),
        (PyArrayObject *) PyObject_GetItem((PyObject *) b, idx_b),
        (PyArrayObject *) PyObject_GetItem((PyObject *) c, idx_c),
        tile_size, sv, zero_fill
    );

    PyTuple_SET_ITEM(idx_b, 1, n1);
    PyTuple_SET_ITEM(idx_c, 1, n1);
    #pragma omp task
    _impl_openmp_tasks_inner(
        (PyArrayObject *) PyObject_GetItem((PyObject *) a, idx_a),
        (PyArrayObject *) PyObject_GetItem((PyObject *) b, idx_b),
        (PyArrayObject *) PyObject_GetItem((PyObject *) c, idx_c),
        tile_size, sv, zero_fill
    );

    PyTuple_SET_ITEM(idx_a, 0, m0);
    PyTuple_SET_ITEM(idx_c, 0, m0);
    #pragma omp task
    _impl_openmp_tasks_inner(
        (PyArrayObject *) PyObject_GetItem((PyObject *) a, idx_a),
        (PyArrayObject *) PyObject_GetItem((PyObject *) b, idx_b),
        (PyArrayObject *) PyObject_GetItem((PyObject *) c, idx_c),
        tile_size, sv, zero_fill
    );

    PyTuple_SET_ITEM(idx_b, 1, n0);
    PyTuple_SET_ITEM(idx_c, 1, n0);
    #pragma omp task
    _impl_openmp_tasks_inner(
        (PyArrayObject *) PyObject_GetItem((PyObject *) a, idx_a),
        (PyArrayObject *) PyObject_GetItem((PyObject *) b, idx_b),
        (PyArrayObject *) PyObject_GetItem((PyObject *) c, idx_c),
        tile_size, sv, zero_fill
    );

    #pragma omp taskwait

    Py_DECREF(idx_a);
    Py_DECREF(idx_b);
    Py_DECREF(idx_c);
}
