#include "openmp.h"

#include <omp.h>


void impl_openmp_loops(PyArrayObject *a, PyArrayObject *b, PyArrayObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    size_t m_sz = PyArray_SHAPE(a)[0];          /* Rows of A and C.         */
    size_t k_sz = PyArray_SHAPE(a)[1];          /* Cols of A and rows of B. */
    size_t n_sz = PyArray_SHAPE(b)[1];          /* Cols of B and C.         */

    #pragma omp parallel for shared(a,b,c) private(m,n,k) schedule(static) // test different schedulers, also collapse(2)
    for (size_t m = 0; m < m_sz; m++) {          /* Iterate over M.          */
        //prob the major issue will be with mem access
        for (size_t n = 0; n < n_sz; n++)       /* Iterate over N.          */
        {
            value_t acc = (value_t) 0;
            for (size_t k = 0; k < k_sz; k++)   /* Iterate over K.          */
            #pragma omp reduction(+:acc) //maybe omp critical for th sync
                acc += matrix_at(a, m, k) * matrix_at(b, k, n);

            matrix_at(c, m, n) = acc;
        }
    }
    

#undef matrix_at
#undef IMPL_TYPE
}

void impl_openmp_tasks(PyArrayObject *a, PyArrayObject *b, PyArrayObject *c)
{
#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

    // TODO: Implement parallel matrix multiplication using OpenMP's task
    //   parallelism.

    size_t m_sz = PyArray_SHAPE(a)[0];          /* Rows of A and C.         */
    size_t k_sz = PyArray_SHAPE(a)[1];          /* Cols of A and rows of B. */
    size_t n_sz = PyArray_SHAPE(b)[1];          /* Cols of B and C.         */

    //input dimension of submatrices a,b,c
    int SZ = 0;
    #pragma omp paralell
    #pragma omp master 
    MM_DQ ( a,b,c,SZ,m_sz);

#undef matrix_at
#undef IMPL_TYPE
}

void MM_DQ ( const PyArrayObject *a, const PyArrayObject *b, PyArrayObject *c, int SZ, const int N){
        // SZ: dimension of submatrices a, b and c. 
        // N:  size of original input matrices (size of a row)

        if (SZ <= DQSZ) 
        { // Classical algorithm for base case
            for (size_t m = 0; m < SZ; m++)           /* Iterate over M.          */
                for (size_t n = 0; n < SZ; n++)       /* Iterate over N.          */
                {
                    value_t acc = (value_t) 0;
                    for (size_t k = 0; k < SZ; k++)   /* Iterate over K.          */
                        acc += matrix_at(a, m, k) * matrix_at(b, k, n);

                    matrix_at(c, m, n) = acc;
                }
            return;
        }

        // Divide task into 8 subtasks

        SZ = SZ/2;  // assume SZ is a perfect power of 2
        #pragma omp task
        MM_DQ( a,          b,        c,          SZ, N);
        #pragma omp task
        MM_DQ( a,          b+SZ,     c+SZ,       SZ, N);
        #pragma omp task
        MM_DQ( a+SZ*N,     b,        c+SZ*N,     SZ, N);
        #pragma omp task
        MM_DQ( a+SZ*N,     b+SZ,     c+SZ*(N+1), SZ, N);
        #pragma omp task
        MM_DQ( a+SZ,       b+SZ*N,     c,          SZ, N);
        #pragma omp task
        MM_DQ( a+SZ,       b+SZ*(N+1), c+SZ,       SZ, N);
        #pragma omp task
        MM_DQ( a+SZ*(N+1), b+SZ*N,     c+SZ*N,     SZ, N);
        #pragma omp task
        MM_DQ( a+SZ*(N+1), b+SZ*(N+1), c+SZ*(N+1), SZ, N);
        #pragma omp taskwait
    }
