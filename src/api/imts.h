#ifndef DOTOPT__API__IMTS_H
#define DOTOPT__API__IMTS_H

#include <Python.h>


PyObject *api_dot_imts_sequential_impl(PyObject *self, PyObject *args);
PyObject *api_dot_imts_sequential_asm_impl(PyObject *self, PyObject *args);


const static struct PyMethodDef api_dot_imts_sequential = {
    .ml_name = "dot_imts_sequential",
    .ml_doc = "Sequential matrix multiplication using IMTS.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_imts_sequential_impl,
};

const static struct PyMethodDef api_dot_imts_sequential_asm = {
    .ml_name = "dot_imts_sequential_asm",
    .ml_doc = "Sequential ASM matrix multiplication using IMTS.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_imts_sequential_asm_impl,
};


#endif /* DOTOPT__API__IMTS_H */
