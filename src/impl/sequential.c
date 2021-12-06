#include "sequential.h"

#include <numpy/arrayobject.h>


void impl_sequential(PyObject *a, PyObject *b, PyObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    // Shape: 
    //   PyArray_SHAPE(a)[0], PyArray_SHAPE(a)[1]
    //   PyArray_SHAPE(b)[0], PyArray_SHAPE(b)[1]
    //   PyArray_SHAPE(c)[0], PyArray_SHAPE(c)[1]

    // Accessing array cells:
    //   matrix_at(c, 0, 0) = matrix_at(a, 0, 0) + matrix_at(b, 0, 0);

    // TODO: Implement algorithm here.
    int col_a = PyArray_SHAPE(a)[1];
    int row_a = PyArray_SHAPE(a)[0];

    int col_b = PyArray_SHAPE(b)[1];
    int row_b = PyArray_SHAPE(b)[0];

   
    
        
        for (int a = 0; a < col_b; a++) { // por cada columna de la matriz b
            
            for (int i = 0; i < row_a; i++) { // recorremos las filas de a
                int suma = 0;
        
                for (int j = 0; j < col_a; j++)  // Y cada columna de a
                    suma += matrix_at(a,i,j) * matrix_at(b,j,a); // Multiplicamos y sumamos resultado
                
                matrix_at(c,i,a) = suma; // guardamos en c
            }
        }          
    
   

#undef matrix_at
#undef IMPL_TYPE
}
