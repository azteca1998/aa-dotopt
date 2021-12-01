#include <Python.h>

#include "hello.h"


static struct PyMethodDef methods[] = {
#ifndef __INTELLISENSE__ // IntelliSense for C/C++ marks this as an error.
    api_hello,
#endif
    { NULL, NULL, 0, NULL },
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "dotopt",
    .m_doc = "",
    .m_size = -1,
    .m_methods = methods,
};


PyMODINIT_FUNC PyInit_dotopt()
{
    return PyModule_Create(&module);
}
