#include "sequential.hpp"

#include <cstdlib>
#include <numpy/arrayobject.h>
#include <Python.h>

#include "../../src/impl/sequential.h"
#include "../../src/impl/util.h"


void bench_sequential(benchmark::State &state)
{
    PyObject *a, *b, *c;
    npy_intp tmp_dims[2];

    // Initialize.
    Py_Initialize();
    import_array1(state.SkipWithError("Failed to import NumPy."));

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
        (*impl_sequential[sv])(
            reinterpret_cast<PyArrayObject *>(a),
            reinterpret_cast<PyArrayObject *>(b),
            reinterpret_cast<PyArrayObject *>(c)
        );

    benchmark::ClobberMemory();

    // TODO: Clean up.
    Py_DECREF(a);
    Py_DECREF(b);
    Py_DECREF(c);

    // Py_Finalize();
}
