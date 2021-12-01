#include "hello.h"


PyObject *api_hello_impl(PyObject *self, PyObject *args)
{
    return PyUnicode_FromString("Hello, world!");
}
