#ifndef DOTOPT__API__UTIL_H
#define DOTOPT__API__UTIL_H

#include <numpy/arrayobject.h>
#include <Python.h>


#define _assert(cond, exc, msg) \
    if (!(cond)) \
    { \
        PyErr_SetString((exc), (msg)); \
        return NULL; \
    } \
    do {} while (0)


#define _assert_checks(op_a, op_b) \
    _assert( \
        PyArray_NDIM((op_a)) == 2 && PyArray_NDIM((op_b)) == 2, \
        PyExc_AssertionError, \
        "Operands must be 2D arrays." \
    ); \
    _assert( \
        PyArray_DIM((op_a), 1) == PyArray_DIM((op_b), 0), \
        PyExc_AssertionError, \
        "Incompatible operands for this operation." \
    ); \
    \
    _assert( \
        PyArray_TYPE((op_a)) == PyArray_TYPE((op_b)), \
        PyExc_TypeError, \
        "Different operand underlying types." \
    ); \
    _assert( \
        PyArray_TYPE((op_a)) == NPY_FLOAT32, \
        PyExc_TypeError, \
        "Underlying operand type is not supported." \
    )


#endif /* DOTOPT__API__UTIL_H */
