#include "sequential.hpp"

#include <algorithm>
#include <execution>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <random>

#include "../../src/impl/sequential.h"


void bench_sequential(benchmark::State &state)
{
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dis(-1.0f, +1.0f);

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

    std::generate_n(
        std::execution::par_unseq,
        reinterpret_cast<float *>(PyArray_DATA(a)),
        PyArray_SIZE(a),
        [&]() { return dis(gen); }
    );
    std::generate_n(
        std::execution::par_unseq,
        reinterpret_cast<float *>(PyArray_DATA(b)),
        PyArray_SIZE(b),
        [&]() { return dis(gen); }
    );

    benchmark::ClobberMemory();

    for (auto _ : state)
        impl_sequential(
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
