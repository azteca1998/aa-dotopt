#ifndef DOTOPT__API__OPENMP_H
#define DOTOPT__API__OPENMP_H

#include <Python.h>


PyObject *api_dot_openmp_loops_impl(PyObject *self, PyObject *args);
PyObject *api_dot_openmp_tasks_impl(PyObject *self, PyObject *args);


const static struct PyMethodDef api_dot_openmp_loops = {
    .ml_name = "dot_openmp_loops",
    .ml_doc = "Parallel matrix multiplication using OpenMP's loop parallelism.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_openmp_loops_impl,
};

const static struct PyMethodDef api_dot_openmp_tasks = {
    .ml_name = "dot_openmp_tasks",
    .ml_doc = "Parallel matrix multiplication using OpenMP's task parallelism.",
    .ml_flags = METH_VARARGS,
    .ml_meth = &api_dot_openmp_tasks_impl,
};


#endif /* DOTOPT__API__OPENMP_H */
