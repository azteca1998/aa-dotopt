#ifndef DOTOPT__API__HELLO_H
#define DOTOPT__API__HELLO_H

#include <Python.h>


PyObject *api_hello_impl(PyObject *self, PyObject *args);


const static struct PyMethodDef api_hello = {
    .ml_name = "hello",
    .ml_doc = "Returns 'Hello, world!'. Used for testing.",
    .ml_flags = METH_NOARGS,
    .ml_meth = &api_hello_impl,
};


#endif /* DOTOPT__API__HELLO_H */
