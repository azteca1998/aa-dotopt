#ifndef DOTOPT__API__SCHEDULER_H
#define DOTOPT__API__SCHEDULER_H

#include <Python.h>


PyObject *api_dot_scheduler_impl(PyObject *self, PyObject *args);
PyObject *api_dot_scheduler_asm_impl(PyObject *self, PyObject *args);


const static struct PyMethodDef api_dot_scheduler = {
    .ml_name = "dot_scheduler",
    .ml_doc = "Parallel custom scheduler (IMTS+Z-Order) matrix multiplication.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_scheduler_impl,
};

const static struct PyMethodDef api_dot_scheduler_asm = {
    .ml_name = "dot_scheduler_asm",
    .ml_doc = "Parallel custom scheduler (IMTS+Z-Order) ASM matrix "
        "multiplication.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_scheduler_asm_impl,
};


#endif /* DOTOPT__API__SCHEDULER_H */
