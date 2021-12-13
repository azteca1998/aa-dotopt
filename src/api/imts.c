#include "openmp.h"

#include <assert.h>
#include <numpy/arrayobject.h>

#include "../impl/imts.h"
#include "util.h"


PyObject *api_dot_imts_impl(PyObject *self, PyObject *args)
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
    _assert_checks(op_a, op_b);

    /* Allocate output array. */
    dims[0] = PyArray_DIM(op_a, 0);
    dims[1] = PyArray_DIM(op_b, 1);
    op_c = PyArray_SimpleNew(2, dims, PyArray_TYPE(op_a));

    /* Call the implementation. */
    impl_imts(
        (PyArrayObject *) op_a,
        (PyArrayObject *) op_b,
        (PyArrayObject *) op_c
    );

    return op_c;
}
