#include "asm.hpp"

#include <cstdlib>
#include <numpy/arrayobject.h>
#include <Python.h>

#include "../../src/impl/asm.h"
#include "../../src/impl/util.h"


void bench_asm(benchmark::State &state)
{
    PyObject *a, *b, *c;
    npy_intp tmp_dims[2];

    tmp_dims[0] = state.range();
    tmp_dims[1] = state.range();

    a = PyArray_SimpleNew(2, tmp_dims, PyArray_FLOAT32);
    b = PyArray_SimpleNew(2, tmp_dims, PyArray_FLOAT32);
    c = PyArray_SimpleNew(2, tmp_dims, PyArray_FLOAT32);

    for (size_t i = 0; i < state.range() * state.range(); i++)
    {
        ((float *) PyArray_DATA(a))[i] = (float) drand48() - 0.5f;
        ((float *) PyArray_DATA(b))[i] = (float) drand48() - 0.5f;
    }

    sequential_version_t sv = sv_find_version(
        sizeof(float),
        reinterpret_cast<PyArrayObject *>(a),
        reinterpret_cast<PyArrayObject *>(b),
        reinterpret_cast<PyArrayObject *>(c)
    );

    benchmark::ClobberMemory();

    for (auto _ : state)
        (*impl_asm[sv])(
            reinterpret_cast<PyArrayObject *>(a),
            reinterpret_cast<PyArrayObject *>(b),
            reinterpret_cast<PyArrayObject *>(c),
            1
        );

    benchmark::ClobberMemory();

    // Clean up.
    Py_DECREF(a);
    Py_DECREF(b);
    Py_DECREF(c);
}
