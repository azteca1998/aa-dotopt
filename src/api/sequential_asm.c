#include "sequential.h"

#include <assert.h>
#include <numpy/arrayobject.h>

#include "../impl/sequential_asm.h"
#include "../impl/util.h"
#include "../zorder.h"
#include "util.h"


PyObject *api_dot_sequential_asm_impl(PyObject *self, PyObject *args)
{
    PyArrayObject *op_a, *op_b, *op_c;
    npy_intp dims[2];
    sequential_version_t sv;
    matrix_t a, b, c;
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
    op_c = (PyArrayObject *) PyArray_SimpleNew(2, dims, PyArray_TYPE(op_a));

    /* Call the implementation. */
    a.data = PyArray_DATA(op_a);
    a.num_rows = PyArray_SHAPE(op_a)[0];
    a.num_cols = PyArray_SHAPE(op_a)[1];
    a.row_stride = PyArray_STRIDE(op_a, 0);
    a.col_stride = PyArray_STRIDE(op_a, 1);
    b.data = PyArray_DATA(op_b);
    b.num_rows = PyArray_SHAPE(op_b)[0];
    b.num_cols = PyArray_SHAPE(op_b)[1];
    b.row_stride = PyArray_STRIDE(op_b, 0);
    b.col_stride = PyArray_STRIDE(op_b, 1);
    c.data = PyArray_DATA(op_c);
    c.num_rows = PyArray_SHAPE(op_c)[0];
    c.num_cols = PyArray_SHAPE(op_c)[1];
    c.row_stride = PyArray_STRIDE(op_c, 0);
    c.col_stride = PyArray_STRIDE(op_c, 1);

    sv = sv_find_version(sizeof(float), &a, &b, &c);
    (*impl_sequential_asm[sv])(&a, &b, &c, 1);

    return (PyObject *) op_c;
}

PyObject *api_dot_sequential_asm_zorder_impl(PyObject *self, PyObject *args)
{
    PyArrayObject *op_a, *op_b, *op_c;
    npy_intp dims[2];
    sequential_version_t sv;
    matrix_t a, b, c;
    float *ta, *tb, *tc, *tmp;
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
    op_c = (PyArrayObject *) PyArray_SimpleNew(2, dims, PyArray_TYPE(op_a));

    /* Call the implementation. */
    a.data = PyArray_DATA(op_a);
    a.num_rows = PyArray_SHAPE(op_a)[0];
    a.num_cols = PyArray_SHAPE(op_a)[1];
    a.row_stride = PyArray_STRIDE(op_a, 0);
    a.col_stride = PyArray_STRIDE(op_a, 1);
    b.data = PyArray_DATA(op_b);
    b.num_rows = PyArray_SHAPE(op_b)[0];
    b.num_cols = PyArray_SHAPE(op_b)[1];
    b.row_stride = PyArray_STRIDE(op_b, 0);
    b.col_stride = PyArray_STRIDE(op_b, 1);
    c.data = PyArray_DATA(op_c);
    c.num_rows = PyArray_SHAPE(op_c)[0];
    c.num_cols = PyArray_SHAPE(op_c)[1];
    c.row_stride = PyArray_STRIDE(op_c, 0);
    c.col_stride = PyArray_STRIDE(op_c, 1);

    ta = zorder_alloc(&a);
    tb = zorder_alloc(&b);
    tc = zorder_alloc(&c);

    zorder_transform(ta, &a);
    zorder_transform(tb, &b);

    tmp = a.data;
    a.data = ta;
    ta = tmp;
    tmp = b.data;
    b.data = tb;
    tb = tmp;
    tmp = c.data;
    c.data = tc;
    tc = tmp;

    (*impl_sequential_asm[sv_zz_zz_zz])(&a, &b, &c, 1);

    tmp = a.data;
    a.data = ta;
    ta = tmp;
    tmp = b.data;
    b.data = tb;
    tb = tmp;
    tmp = c.data;
    c.data = tc;
    tc = tmp;

    zorder_transform_inverse(tc, &c);

    zorder_free(ta);
    zorder_free(tb);
    zorder_free(tc);

    return (PyObject *) op_c;
}
