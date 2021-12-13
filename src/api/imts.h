#ifndef DOTOPT__API__IMTS_H
#define DOTOPT__API__IMTS_H

#include <Python.h>


PyObject *api_dot_imts_impl(PyObject *self, PyObject *args);


const static struct PyMethodDef api_dot_imts = {
    .ml_name = "dot_imts",
    .ml_doc = "Sequential matrix multiplication using IMTS.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_imts_impl,
};


#endif /* DOTOPT__API__IMTS_H */
