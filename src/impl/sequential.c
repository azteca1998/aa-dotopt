#include "sequential.h"


#define value_t float
#define matrix_at(m, row, col) \
    (*(value_t *) (PyArray_DATA(m) \
        + PyArray_STRIDE(m, 0) * row \
        + PyArray_STRIDE(m, 1) * col \
    ))

#define _impl_sequential_mkn(name, as0, as1, bs0, bs1, cs0, cs1) \
    static void name( \
        PyArrayObject *a, \
        PyArrayObject *b, \
        PyArrayObject *c, \
        int zero_fill \
    ) \
    { \
        const size_t m_sz = PyArray_SHAPE(a)[0]; /* Rows of A & C.         */ \
        const size_t k_sz = PyArray_SHAPE(a)[1]; /* Cols of A & rows of B. */ \
        const size_t n_sz = PyArray_SHAPE(b)[1]; /* Cols of B & C.         */ \
        \
        for (size_t m = 0; m < m_sz; m++) \
            for (size_t k = 0; k < k_sz; k++) \
            { \
                if (zero_fill && k == 0) \
                    for (size_t n = 0; n < n_sz; n++) \
                        matrix_at(c, m, n) = (value_t) 0; \
                \
                for (size_t n = 0; n < n_sz; n++) \
                    matrix_at(c, m, n) += \
                        matrix_at(a, m, k) * matrix_at(b, k, n); \
            } \
    }

#define _impl_sequential_mnk(name, as0, as1, bs0, bs1, cs0, cs1) \
    static void name( \
        PyArrayObject *a, \
        PyArrayObject *b, \
        PyArrayObject *c, \
        int zero_fill \
    ) \
    { \
        const size_t m_sz = PyArray_SHAPE(a)[0]; /* Rows of A & C.         */ \
        const size_t k_sz = PyArray_SHAPE(a)[1]; /* Cols of A & rows of B. */ \
        const size_t n_sz = PyArray_SHAPE(b)[1]; /* Cols of B & C.         */ \
        \
        for (size_t m = 0; m < m_sz; m++) \
            for (size_t n = 0; n < n_sz; n++) \
            { \
                if (zero_fill) \
                    matrix_at(c, m, n) = (value_t) 0; \
                \
                for (size_t k = 0; k < k_sz; k++) \
                    matrix_at(c, m, n) += \
                        matrix_at(a, m, k) * \
                        matrix_at(b, k, n); \
            } \
    }

#define _impl_sequential_nkm(name, as0, as1, bs0, bs1, cs0, cs1) \
    static void name( \
        PyArrayObject *a, \
        PyArrayObject *b, \
        PyArrayObject *c, \
        int zero_fill \
    ) \
    { \
        const size_t m_sz = PyArray_SHAPE(a)[0]; /* Rows of A & C.         */ \
        const size_t k_sz = PyArray_SHAPE(a)[1]; /* Cols of A & rows of B. */ \
        const size_t n_sz = PyArray_SHAPE(b)[1]; /* Cols of B & C.         */ \
        \
        for (size_t n = 0; n < n_sz; n++) \
            for (size_t k = 0; k < k_sz; k++) \
            { \
                if (zero_fill && k == 0) \
                    for (size_t m = 0; m < m_sz; m++) \
                        matrix_at(c, m, n) = (value_t) 0; \
                \
                for (size_t m = 0; m < m_sz; m++) \
                    matrix_at(c, m, n) += \
                        matrix_at(a, m, k) * \
                        matrix_at(b, k, n); \
            } \
    }

#define _impl_sequential_nmk(name, as0, as1, bs0, bs1, cs0, cs1) \
    static void name( \
        PyArrayObject *a, \
        PyArrayObject *b, \
        PyArrayObject *c, \
        int zero_fill \
    ) \
    { \
        const size_t m_sz = PyArray_SHAPE(a)[0]; /* Rows of A & C.         */ \
        const size_t k_sz = PyArray_SHAPE(a)[1]; /* Cols of A & rows of B. */ \
        const size_t n_sz = PyArray_SHAPE(b)[1]; /* Cols of B & C.         */ \
        \
        for (size_t n = 0; n < n_sz; n++) \
            for (size_t m = 0; m < m_sz; m++) \
            { \
                if (zero_fill) \
                    matrix_at(c, m, n) = (value_t) 0; \
                \
                for (size_t k = 0; k < k_sz; k++) \
                    matrix_at(c, m, n) += \
                        matrix_at(a, m, k) * \
                        matrix_at(b, k, n); \
            } \
    }

#define _impl_sequential_kmn(name, as0, as1, bs0, bs1, cs0, cs1) \
    static void name( \
        PyArrayObject *a, \
        PyArrayObject *b, \
        PyArrayObject *c, \
        int zero_fill \
    ) \
    { \
        const size_t m_sz = PyArray_SHAPE(a)[0]; /* Rows of A & C.         */ \
        const size_t k_sz = PyArray_SHAPE(a)[1]; /* Cols of A & rows of B. */ \
        const size_t n_sz = PyArray_SHAPE(b)[1]; /* Cols of B & C.         */ \
        \
        for (size_t k = 0; k < k_sz; k++) \
            for (size_t m = 0; m < m_sz; m++) \
            { \
                if (zero_fill && k == 0) \
                    for (size_t n = 0; n < n_sz; n++) \
                        matrix_at(c, m, n) = (value_t) 0; \
                \
                for (size_t n = 0; n < n_sz; n++) \
                    matrix_at(c, m, n) += \
                        matrix_at(a, m, k) * \
                        matrix_at(b, k, n); \
            } \
    }

#define _impl_sequential_knm(name, as0, as1, bs0, bs1, cs0, cs1) \
    static void name( \
        PyArrayObject *a, \
        PyArrayObject *b, \
        PyArrayObject *c, \
        int zero_fill \
    ) \
    { \
        const size_t m_sz = PyArray_SHAPE(a)[0]; /* Rows of A & C.         */ \
        const size_t k_sz = PyArray_SHAPE(a)[1]; /* Cols of A & rows of B. */ \
        const size_t n_sz = PyArray_SHAPE(b)[1]; /* Cols of B & C.         */ \
        \
        for (size_t k = 0; k < k_sz; k++) \
            for (size_t n = 0; n < n_sz; n++) \
            { \
                if (zero_fill && k == 0) \
                    for (size_t m = 0; m < m_sz; m++) \
                        matrix_at(c, m, n) = (value_t) 0; \
                \
                for (size_t m = 0; m < m_sz; m++) \
                    matrix_at(c, m, n) += \
                        matrix_at(a, m, k) * \
                        matrix_at(b, k, n); \
            } \
    }


_impl_sequential_mkn(impl_sequential_mkn,
    PyArray_STRIDE(a, 0), PyArray_STRIDE(a, 1),
    PyArray_STRIDE(b, 0), PyArray_STRIDE(b, 1),
    PyArray_STRIDE(c, 0), PyArray_STRIDE(c, 1)
)
_impl_sequential_mnk(impl_sequential_mnk,
    PyArray_STRIDE(a, 0), PyArray_STRIDE(a, 1),
    PyArray_STRIDE(b, 0), PyArray_STRIDE(b, 1),
    PyArray_STRIDE(c, 0), PyArray_STRIDE(c, 1)
)
_impl_sequential_nkm(impl_sequential_nkm,
    PyArray_STRIDE(a, 0), PyArray_STRIDE(a, 1),
    PyArray_STRIDE(b, 0), PyArray_STRIDE(b, 1),
    PyArray_STRIDE(c, 0), PyArray_STRIDE(c, 1)
)
_impl_sequential_nmk(impl_sequential_nmk,
    PyArray_STRIDE(a, 0), PyArray_STRIDE(a, 1),
    PyArray_STRIDE(b, 0), PyArray_STRIDE(b, 1),
    PyArray_STRIDE(c, 0), PyArray_STRIDE(c, 1)
)
_impl_sequential_kmn(impl_sequential_kmn,
    PyArray_STRIDE(a, 0), PyArray_STRIDE(a, 1),
    PyArray_STRIDE(b, 0), PyArray_STRIDE(b, 1),
    PyArray_STRIDE(c, 0), PyArray_STRIDE(c, 1)
)
_impl_sequential_knm(impl_sequential_knm,
    PyArray_STRIDE(a, 0), PyArray_STRIDE(a, 1),
    PyArray_STRIDE(b, 0), PyArray_STRIDE(b, 1),
    PyArray_STRIDE(c, 0), PyArray_STRIDE(c, 1)
)


const impl_func_t impl_sequential[27] = {
    &impl_sequential_nkm, /* nkm, knm; nkm is 4-5% better */
    &impl_sequential_mkn, /* FIXME: WTF */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_knm, /* knm, nkm; knm is 4-5% better */
    &impl_sequential_mkn, /* mkn, kmn; apparently, kmn is better at 1024, but mkn wins at 2048 */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */

    &impl_sequential_nkm, /* nkm, nmk; they perform equally */
    &impl_sequential_nmk, /* mnk, nmk; they perform equally, but nmk misses twice as much */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_knm, /* FIXME: WTF */
    &impl_sequential_mkn, /* mkn, kmn; they perform equally, but kmn misses 1,5-2x as much */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */

    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
    &impl_sequential_mkn, /* TODO: Complex arrays. */
};
