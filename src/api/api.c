#include <Python.h>

#undef NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "hello.h"
#include "sequential.h"


static struct PyMethodDef methods[] = {
#ifndef __INTELLISENSE__ // IntelliSense for C/C++ marks this as an error.
    api_hello,
    api_dot_sequential,
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
    import_array();

    return PyModule_Create(&module);
}
