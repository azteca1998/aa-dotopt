#include <Python.h>

#undef NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "imts.h"
#include "scheduler.h"
#include "sequential.h"
#include "sequential_asm.h"


static struct PyMethodDef methods[] = {
#ifndef __INTELLISENSE__ // IntelliSense for C/C++ marks this as an error.
    api_dot_imts_sequential,
    api_dot_imts_sequential_asm,
    api_dot_imts_sequential_asm_omp_tasks,
    api_dot_imts_sequential_asm_omp_tasks_zorder,
    api_dot_imts_sequential_asm_zorder,
    api_dot_imts_sequential_omp_tasks,
    api_dot_imts_sequential_omp_tasks_zorder,
    api_dot_imts_sequential_zorder,
    api_dot_scheduler,
    api_dot_scheduler_asm,
    api_dot_sequential,
    api_dot_sequential_asm,
    api_dot_sequential_asm_omp_loops,
    api_dot_sequential_asm_omp_loops_zorder,
    api_dot_sequential_asm_zorder,
    api_dot_sequential_omp_loops,
    api_dot_sequential_omp_loops_zorder,
    api_dot_sequential_zorder,
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
