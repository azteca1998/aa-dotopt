#ifndef DOTOPT__API__SEQUENTIAL_ASM_H
#define DOTOPT__API__SEQUENTIAL_ASM_H

#include <Python.h>


PyObject *api_dot_sequential_asm_impl(PyObject *self, PyObject *args);
PyObject *api_dot_sequential_asm_zorder_impl(PyObject *self, PyObject *args);

PyObject *api_dot_sequential_asm_omp_loops_impl(PyObject *self, PyObject *args);
PyObject *api_dot_sequential_asm_omp_loops_zorder_impl(
    PyObject *self,
    PyObject *args
);


const static struct PyMethodDef api_dot_sequential_asm = {
    .ml_name = "dot_sequential_asm",
    .ml_doc = "Hand-crafted assembly matrix multiplication.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_sequential_asm_impl,
};

const static struct PyMethodDef api_dot_sequential_asm_zorder = {
    .ml_name = "dot_sequential_asm_zorder",
    .ml_doc = "Hand-crafted assembly Z-order matrix multiplication.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_sequential_asm_zorder_impl,
};


const static struct PyMethodDef api_dot_sequential_asm_omp_loops = {
    .ml_name = "dot_sequential_asm_omp_loops",
    .ml_doc = "Hand-crafted parallel assembly matrix multiplication using "
        "OpenMP loop parallelism.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_sequential_asm_omp_loops_impl,
};

const static struct PyMethodDef api_dot_sequential_asm_omp_loops_zorder = {
    .ml_name = "dot_sequential_asm_omp_loops_zorder",
    .ml_doc = "Hand-crafted parallel assembly matrix multiplication using "
        "OpenMP loop parallelism and Z-order.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_sequential_asm_omp_loops_zorder_impl,
};


#endif /* DOTOPT__API__SEQUENTIAL_ASM_H */
