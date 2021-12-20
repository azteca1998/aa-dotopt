#ifndef DOTOPT__API__SEQUENTIAL_ASM_H
#define DOTOPT__API__SEQUENTIAL_ASM_H

#include <Python.h>


PyObject *api_dot_sequential_asm_impl(PyObject *self, PyObject *args);


const static struct PyMethodDef api_dot_sequential_asm = {
    .ml_name = "dot_asm",
    .ml_doc = "Hand-crafted assembly matrix multiplication.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_sequential_asm_impl,
};


#endif /* DOTOPT__API__SEQUENTIAL_ASM_H */
