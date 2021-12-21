#ifndef DOTOPT__IMPL__UTIL_H
#define DOTOPT__IMPL__UTIL_H

#include <stddef.h>


#ifndef __cplusplus
# define DOTOPT_API
#else
# define DOTOPT_API extern "C"
#endif


typedef struct matrix {
    void *data;

    size_t num_rows;
    size_t num_cols;

    size_t row_stride;
    size_t col_stride;
} matrix_t;


/**
 * @brief Possible array strides' combinations.
 * 
 * For each numpy array used in the operation, there can only be three possible
 * scenarios:
 *   - Strides are [1, ?] (fortran ordering).
 *   - Strides are [?, 1] (C ordering).
 *   - Strides are [?, ?] (neither/other).
 *
 * Since there are three arrays used in the computations, there are 27 possible
 * modes. For each algorithm implemented, there will be multiple optimized
 * versions (and a generic one), that will be accessed using a lookup table with
 * this enum's values as key.
 */
typedef enum sequential_version
{
                        /*      STRIDES      | Order */
                        /*   A     B     C   |  ABC  */
                        /* ================= | ===== */
    sv_1x_1x_1x = 0x00, /* (1,x) (1,x) (1,x) |  FFF  */
    sv_1x_1x_x1 = 0x01, /* (1,x) (1,x) (x,1) |  FFC  */
    sv_1x_1x_xx = 0x02, /* (1,x) (1,x) (x,x) |  FF?  */
    sv_1x_x1_1x = 0x03, /* (1,x) (x,1) (1,x) |  FCF  */
    sv_1x_x1_x1 = 0x04, /* (1,x) (x,1) (x,1) |  FCC  */
    sv_1x_x1_xx = 0x05, /* (1,x) (x,1) (x,x) |  FC?  */
    sv_1x_xx_1x = 0x06, /* (1,x) (x,x) (1,x) |  F?F  */
    sv_1x_xx_x1 = 0x07, /* (1,x) (x,x) (x,1) |  F?C  */
    sv_1x_xx_xx = 0x08, /* (1,x) (x,x) (x,x) |  F??  */
                        /* ----------------- |       */
    sv_x1_1x_1x = 0x09, /* (x,1) (1,x) (1,x) |  CFF  */
    sv_x1_1x_x1 = 0x0A, /* (x,1) (1,x) (x,1) |  CFC  */
    sv_x1_1x_xx = 0x0B, /* (x,1) (1,x) (x,x) |  CF?  */
    sv_x1_x1_1x = 0x0C, /* (x,1) (x,1) (1,x) |  CCF  */
    sv_x1_x1_x1 = 0x0D, /* (x,1) (x,1) (x,1) |  CCC  */
    sv_x1_x1_xx = 0x0E, /* (x,1) (x,1) (x,x) |  CC?  */
    sv_x1_xx_1x = 0x0F, /* (x,1) (x,x) (1,x) |  C?F  */
    sv_x1_xx_x1 = 0x10, /* (x,1) (x,x) (x,1) |  C?C  */
    sv_x1_xx_xx = 0x11, /* (x,1) (x,x) (x,x) |  C??  */
                        /* ----------------- |       */
    sv_xx_1x_1x = 0x12, /* (x,x) (1,x) (1,x) |  ?FF  */
    sv_xx_1x_x1 = 0x13, /* (x,x) (1,x) (x,1) |  ?FC  */
    sv_xx_1x_xx = 0x14, /* (x,x) (1,x) (x,x) |  ?F?  */
    sv_xx_x1_1x = 0x15, /* (x,x) (x,1) (1,x) |  ?CF  */
    sv_xx_x1_x1 = 0x16, /* (x,x) (x,1) (x,1) |  ?CC  */
    sv_xx_x1_xx = 0x17, /* (x,x) (x,1) (x,x) |  ?C?  */
    sv_xx_xx_1x = 0x18, /* (x,x) (x,x) (1,x) |  ??F  */
    sv_xx_xx_x1 = 0x19, /* (x,x) (x,x) (x,1) |  ??C  */
    sv_xx_xx_xx = 0x1A, /* (x,x) (x,x) (x,x) |  ???  */

    sv_zz_zz_zz = 0x1B, /*   Z-Order Curve   |  ZZZ  */
} sequential_version_t;


/**
 * @brief Detect the optimal sequential version for the given array set.
 * 
 * Warning: Not using the returned version index may lead to incorrect results.
 *   If `sv_find_version` is not available, use the generic version
 *   `sv_xx_xx_xx`.
 * 
 * @param a Left operand.
 * @param b Right operand.
 * @param c Output array.
 * @return sequential_version_t Sequential version index.
 */
DOTOPT_API sequential_version_t sv_find_version(
    size_t value_size,
    matrix_t *a,
    matrix_t *b,
    matrix_t *c
);


#endif /* DOTOPT__IMPL__UTIL_H */
