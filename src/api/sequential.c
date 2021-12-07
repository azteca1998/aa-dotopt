#include "sequential.h"

#include <assert.h>
#include <numpy/arrayobject.h>

#include "../impl/sequential.h"
#include "util.h"


PyObject *api_dot_sequential_impl(PyObject *self, PyObject *args)
{
    PyObject *op_a, *op_b, *op_c;
    npy_intp dims[2];
    int ret;

    /* Extract arguments. */
    ret = PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &op_a,
                           &PyArray_Type, &op_b);
    _assert(ret, PyExc_RuntimeError, "Failed to extract method arguments.");

    /* Ensure operands compatibility. */
    _assert(
        PyArray_NDIM(op_a) == 2 && PyArray_NDIM(op_b) == 2,
        PyExc_AssertionError,
        "Operands must be 2D arrays."
    );
    _assert(
        PyArray_DIM(op_a, 1) == PyArray_DIM(op_b, 0),
        PyExc_AssertionError,
        "Incompatible operands for this operation."
    );

    _assert(
        PyArray_TYPE(op_a) == PyArray_TYPE(op_b),
        PyExc_TypeError,
        "Different operand underlying types."
    );
    _assert(
        PyArray_TYPE(op_a) == NPY_FLOAT32,
        PyExc_TypeError,
        "Underlying operand type is not supported."
    );

    /* Allocate output array. */
    dims[0] = PyArray_DIM(op_a, 0);
    dims[1] = PyArray_DIM(op_b, 1);
    op_c = PyArray_SimpleNew(2, dims, PyArray_TYPE(op_a));

    /* Call the implementation. */
    impl_sequential(
        (PyArrayObject *) op_a,
        (PyArrayObject *) op_b,
        (PyArrayObject *) op_c
    );

    return op_c;
}
