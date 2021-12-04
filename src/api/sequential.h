#ifndef DOTOPT__API__SEQUENTIAL_H
#define DOTOPT__API__SEQUENTIAL_H

#include <Python.h>


PyObject *api_dot_sequential_impl(PyObject *self, PyObject *args);


const static struct PyMethodDef api_dot_sequential = {
    .ml_name = "dot_sequential",
    .ml_doc = "Sequential matrix multiplication.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_sequential_impl,
};


#endif /* DOTOPT__API__SEQUENTIAL_H */
