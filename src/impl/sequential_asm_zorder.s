.intel_syntax noprefix


.macro  prepare row:req
    vpermilps   ymm12,  \row,   0xCC
    vpermilps   ymm13,  \row,   0x99
.endm

.macro  dot_mul r_in:req r_out:req
    vmulps      ymm14,  \r_in,  ymm12
    vmulps      ymm15,  \r_in,  ymm13
    vpermilps   ymm15,  ymm15,  0xB1
    vaddps      \r_out, ymm14,  ymm15
.endm

.macro  dot_muladd  r_in:req r_out:req
    vmulps      ymm14,  \r_in,  ymm12
    vmulps      ymm15,  \r_in,  ymm13
    vpermilps   ymm15,  ymm15,  0xB1
    vaddps      \r_out, \r_out, ymm14
    vaddps      \r_out, \r_out, ymm15
.endm


.global _impl_sequential_asm_zz_zz_zz
_impl_sequential_asm_zz_zz_zz:
    # rdi :: float *ptr_a
    # rsi :: float *ptr_b
    # rdx :: float *ptr_c

    # Load top half of A:
    vmovups     ymm0,   [rdi+0x00]
    vmovups     ymm1,   [rdi+0x20]
    vmovups     ymm2,   [rdi+0x40]
    vmovups     ymm3,   [rdi+0x60]

    # Load top half of B.
    vmovups     xmm4,   [rsi+0x00]
    vmovups     xmm5,   [rsi+0x40]
    vmovups     xmm6,   [rsi+0x80]
    vmovups     xmm7,   [rsi+0xC0]
    vinsertf128 ymm4,   ymm4,   [rsi+0x30], 1
    vinsertf128 ymm5,   ymm5,   [rsi+0x70], 1
    vinsertf128 ymm6,   ymm6,   [rsi+0xB0], 1
    vinsertf128 ymm7,   ymm7,   [rsi+0xF0], 1

    # Load top half of C:
    vmovups     ymm8,   [rdx+0x00]
    vmovups     ymm9,   [rdx+0x20]
    vmovups     ymm10,  [rdx+0x40]
    vmovups     ymm11,  [rdx+0x60]

    # Compute (1/4):
    #   c11 c12 c21 c22 c13 c14 c23 c24
    #   c31 c32 c41 c42 c33 c34 c43 c44
    prepare     ymm4
    dot_muladd  ymm0,   ymm8
    dot_muladd  ymm1,   ymm9

    # Compute (1/4):
    #   c15 c16 c25 c26 c17 c18 c27 c28
    #   c35 c36 c45 c46 c37 c38 c47 c48
    prepare     ymm5
    dot_muladd  ymm0,   ymm10
    dot_muladd  ymm1,   ymm11

    # Compute (2/4):
    #   c11 c12 c21 c22 c13 c14 c23 c24
    #   c31 c32 c41 c42 c33 c34 c43 c44
    prepare     ymm6
    dot_muladd  ymm2,   ymm8
    dot_muladd  ymm3,   ymm9

    # Compute (2/4):
    #   c15 c16 c25 c26 c17 c18 c27 c28
    #   c35 c36 c45 c46 c37 c38 c47 c48
    prepare     ymm7
    dot_muladd  ymm2,   ymm10
    dot_muladd  ymm3,   ymm11

    # Exchange f128-values of A (reloading from memory is between 2 and 4 times
    # faster, assumming it is still in L1, which it should).
    vmovups     xmm0,   [rdi+0x10]
    vmovups     xmm1,   [rdi+0x30]
    vmovups     xmm2,   [rdi+0x50]
    vmovups     xmm3,   [rdi+0x70]
    vinsertf128 ymm0,   ymm0,   [rdi+0x00], 1
    vinsertf128 ymm1,   ymm1,   [rdi+0x20], 1
    vinsertf128 ymm2,   ymm2,   [rdi+0x40], 1
    vinsertf128 ymm3,   ymm3,   [rdi+0x60], 1

    # Load bottom half of B.
    vmovups     xmm4,   [rsi+0x20]
    vmovups     xmm5,   [rsi+0x60]
    vmovups     xmm6,   [rsi+0xA0]
    vmovups     xmm7,   [rsi+0xE0]
    vinsertf128 ymm4,   ymm4,   [rsi+0x10], 1
    vinsertf128 ymm5,   ymm5,   [rsi+0x50], 1
    vinsertf128 ymm6,   ymm6,   [rsi+0x90], 1
    vinsertf128 ymm7,   ymm7,   [rsi+0xD0], 1

    # Compute (3/4):
    #   c11 c12 c21 c22 c13 c14 c23 c24
    #   c31 c32 c41 c42 c33 c34 c43 c44
    prepare     ymm4
    dot_muladd  ymm0,   ymm8
    dot_muladd  ymm1,   ymm9

    # Compute (3/4):
    #   c15 c16 c25 c26 c17 c18 c27 c28
    #   c35 c36 c45 c46 c37 c38 c47 c48
    prepare     ymm5
    dot_muladd  ymm0,   ymm10
    dot_muladd  ymm1,   ymm11

    # Compute (4/4):
    #   c11 c12 c21 c22 c13 c14 c23 c24
    #   c31 c32 c41 c42 c33 c34 c43 c44
    prepare     ymm6
    dot_muladd  ymm2,   ymm8
    dot_muladd  ymm3,   ymm9

    # Compute (4/4):
    #   c15 c16 c25 c26 c17 c18 c27 c28
    #   c35 c36 c45 c46 c37 c38 c47 c48
    prepare     ymm7
    dot_muladd  ymm2,   ymm10
    dot_muladd  ymm3,   ymm11

    # Store top half of C.
    vmovups [rdx+0x00], ymm8
    vmovups [rdx+0x20], ymm9
    vmovups [rdx+0x40], ymm10
    vmovups [rdx+0x60], ymm11


    # Load bottom half of A (exchanged):
    vmovups     xmm0,   [rdi+0x90]
    vmovups     xmm1,   [rdi+0xB0]
    vmovups     xmm2,   [rdi+0xD0]
    vmovups     xmm3,   [rdi+0xF0]
    vinsertf128 ymm0,   ymm0,   [rdi+0x80], 1
    vinsertf128 ymm1,   ymm1,   [rdi+0xA0], 1
    vinsertf128 ymm2,   ymm2,   [rdi+0xC0], 1
    vinsertf128 ymm3,   ymm3,   [rdi+0xE0], 1

    # Load bottom half of C:
    vmovups     ymm8,   [rdx+0x80]
    vmovups     ymm9,   [rdx+0xA0]
    vmovups     ymm10,  [rdx+0xC0]
    vmovups     ymm11,  [rdx+0xE0]

    # Compute (1/4):
    #   c51 c52 c61 c62 c53 c54 c63 c64
    #   c71 c72 c81 c82 c73 c74 c83 c84
    prepare     ymm4
    dot_muladd  ymm0,   ymm8
    dot_muladd  ymm1,   ymm9

    # Compute (1/4):
    #   c55 c56 c65 c66 c57 c58 c67 c68
    #   c75 c76 c85 c86 c77 c78 c87 c88
    prepare     ymm5
    dot_muladd  ymm0,   ymm10
    dot_muladd  ymm1,   ymm11

    # Compute (2/4):
    #   c51 c52 c61 c62 c53 c54 c63 c64
    #   c71 c72 c81 c82 c73 c74 c83 c84
    prepare     ymm6
    dot_muladd  ymm2,   ymm8
    dot_muladd  ymm3,   ymm9

    # Compute (2/4):
    #   c55 c56 c65 c66 c57 c58 c67 c68
    #   c75 c76 c85 c86 c77 c78 c87 c88
    prepare     ymm7
    dot_muladd  ymm2,   ymm10
    dot_muladd  ymm3,   ymm11

    # Load bottom half of A:
    vmovups     ymm0,   [rdi+0x80]
    vmovups     ymm1,   [rdi+0xA0]
    vmovups     ymm2,   [rdi+0xC0]
    vmovups     ymm3,   [rdi+0xE0]

    # Load top half of B.
    vmovups     xmm4,   [rsi+0x00]
    vmovups     xmm5,   [rsi+0x40]
    vmovups     xmm6,   [rsi+0x80]
    vmovups     xmm7,   [rsi+0xC0]
    vinsertf128 ymm4,   ymm4,   [rsi+0x30], 1
    vinsertf128 ymm5,   ymm5,   [rsi+0x70], 1
    vinsertf128 ymm6,   ymm6,   [rsi+0xB0], 1
    vinsertf128 ymm7,   ymm7,   [rsi+0xF0], 1

    # Compute (3/4):
    #   c51 c52 c61 c62 c53 c54 c63 c64
    #   c71 c72 c81 c82 c73 c74 c83 c84
    prepare     ymm4
    dot_muladd  ymm0,   ymm8
    dot_muladd  ymm1,   ymm9

    # Compute (3/4):
    #   c55 c56 c65 c66 c57 c58 c67 c68
    #   c75 c76 c85 c86 c77 c78 c87 c88
    prepare     ymm5
    dot_muladd  ymm0,   ymm10
    dot_muladd  ymm1,   ymm11

    # Compute (4/4):
    #   c51 c52 c61 c62 c53 c54 c63 c64
    #   c71 c72 c81 c82 c73 c74 c83 c84
    prepare     ymm6
    dot_muladd  ymm2,   ymm8
    dot_muladd  ymm3,   ymm9

    # Compute (4/4):
    #   c55 c56 c65 c66 c57 c58 c67 c68
    #   c75 c76 c85 c86 c77 c78 c87 c88
    prepare     ymm7
    dot_muladd  ymm2,   ymm10
    dot_muladd  ymm3,   ymm11

    # Store bottom half of C.
    vmovups [rdx+0x80], ymm8
    vmovups [rdx+0xA0], ymm9
    vmovups [rdx+0xC0], ymm10
    vmovups [rdx+0xE0], ymm11

    ret


.global _impl_sequential_asm_zz_zz_zz_zf
_impl_sequential_asm_zz_zz_zz_zf:
    # rdi :: float *ptr_a
    # rsi :: float *ptr_b
    # rdx :: float *ptr_c

    # Load top half of A:
    vmovups     ymm0,   [rdi+0x00]
    vmovups     ymm1,   [rdi+0x20]
    vmovups     ymm2,   [rdi+0x40]
    vmovups     ymm3,   [rdi+0x60]

    # Load top half of B.
    vmovups     xmm4,   [rsi+0x00]
    vmovups     xmm5,   [rsi+0x40]
    vmovups     xmm6,   [rsi+0x80]
    vmovups     xmm7,   [rsi+0xC0]
    vinsertf128 ymm4,   ymm4,   [rsi+0x30], 1
    vinsertf128 ymm5,   ymm5,   [rsi+0x70], 1
    vinsertf128 ymm6,   ymm6,   [rsi+0xB0], 1
    vinsertf128 ymm7,   ymm7,   [rsi+0xF0], 1

    # Compute (1/4):
    #   c11 c12 c21 c22 c13 c14 c23 c24
    #   c31 c32 c41 c42 c33 c34 c43 c44
    prepare     ymm4
    dot_mul     ymm0,   ymm8
    dot_mul     ymm1,   ymm9

    # Compute (1/4):
    #   c15 c16 c25 c26 c17 c18 c27 c28
    #   c35 c36 c45 c46 c37 c38 c47 c48
    prepare     ymm5
    dot_mul     ymm0,   ymm10
    dot_mul     ymm1,   ymm11

    # Compute (2/4):
    #   c11 c12 c21 c22 c13 c14 c23 c24
    #   c31 c32 c41 c42 c33 c34 c43 c44
    prepare     ymm6
    dot_muladd  ymm2,   ymm8
    dot_muladd  ymm3,   ymm9

    # Compute (2/4):
    #   c15 c16 c25 c26 c17 c18 c27 c28
    #   c35 c36 c45 c46 c37 c38 c47 c48
    prepare     ymm7
    dot_muladd  ymm2,   ymm10
    dot_muladd  ymm3,   ymm11

    # Exchange f128-values of A (reloading from memory is between 2 and 4 times
    # faster, assumming it is still in L1, which it should).
    vmovups     xmm0,   [rdi+0x10]
    vmovups     xmm1,   [rdi+0x30]
    vmovups     xmm2,   [rdi+0x50]
    vmovups     xmm3,   [rdi+0x70]
    vinsertf128 ymm0,   ymm0,   [rdi+0x00], 1
    vinsertf128 ymm1,   ymm1,   [rdi+0x20], 1
    vinsertf128 ymm2,   ymm2,   [rdi+0x40], 1
    vinsertf128 ymm3,   ymm3,   [rdi+0x60], 1

    # Load bottom half of B.
    vmovups     xmm4,   [rsi+0x20]
    vmovups     xmm5,   [rsi+0x60]
    vmovups     xmm6,   [rsi+0xA0]
    vmovups     xmm7,   [rsi+0xE0]
    vinsertf128 ymm4,   ymm4,   [rsi+0x10], 1
    vinsertf128 ymm5,   ymm5,   [rsi+0x50], 1
    vinsertf128 ymm6,   ymm6,   [rsi+0x90], 1
    vinsertf128 ymm7,   ymm7,   [rsi+0xD0], 1

    # Compute (3/4):
    #   c11 c12 c21 c22 c13 c14 c23 c24
    #   c31 c32 c41 c42 c33 c34 c43 c44
    prepare     ymm4
    dot_muladd  ymm0,   ymm8
    dot_muladd  ymm1,   ymm9

    # Compute (3/4):
    #   c15 c16 c25 c26 c17 c18 c27 c28
    #   c35 c36 c45 c46 c37 c38 c47 c48
    prepare     ymm5
    dot_muladd  ymm0,   ymm10
    dot_muladd  ymm1,   ymm11

    # Compute (4/4):
    #   c11 c12 c21 c22 c13 c14 c23 c24
    #   c31 c32 c41 c42 c33 c34 c43 c44
    prepare     ymm6
    dot_muladd  ymm2,   ymm8
    dot_muladd  ymm3,   ymm9

    # Compute (4/4):
    #   c15 c16 c25 c26 c17 c18 c27 c28
    #   c35 c36 c45 c46 c37 c38 c47 c48
    prepare     ymm7
    dot_muladd  ymm2,   ymm10
    dot_muladd  ymm3,   ymm11

    # Store top half of C.
    vmovups [rdx+0x00], ymm8
    vmovups [rdx+0x20], ymm9
    vmovups [rdx+0x40], ymm10
    vmovups [rdx+0x60], ymm11


    # Load bottom half of A (exchanged):
    vmovups     xmm0,   [rdi+0x90]
    vmovups     xmm1,   [rdi+0xB0]
    vmovups     xmm2,   [rdi+0xD0]
    vmovups     xmm3,   [rdi+0xF0]
    vinsertf128 ymm0,   ymm0,   [rdi+0x80], 1
    vinsertf128 ymm1,   ymm1,   [rdi+0xA0], 1
    vinsertf128 ymm2,   ymm2,   [rdi+0xC0], 1
    vinsertf128 ymm3,   ymm3,   [rdi+0xE0], 1

    # Compute (1/4):
    #   c51 c52 c61 c62 c53 c54 c63 c64
    #   c71 c72 c81 c82 c73 c74 c83 c84
    prepare     ymm4
    dot_mul     ymm0,   ymm8
    dot_mul     ymm1,   ymm9

    # Compute (1/4):
    #   c55 c56 c65 c66 c57 c58 c67 c68
    #   c75 c76 c85 c86 c77 c78 c87 c88
    prepare     ymm5
    dot_mul     ymm0,   ymm10
    dot_mul     ymm1,   ymm11

    # Compute (2/4):
    #   c51 c52 c61 c62 c53 c54 c63 c64
    #   c71 c72 c81 c82 c73 c74 c83 c84
    prepare     ymm6
    dot_muladd  ymm2,   ymm8
    dot_muladd  ymm3,   ymm9

    # Compute (2/4):
    #   c55 c56 c65 c66 c57 c58 c67 c68
    #   c75 c76 c85 c86 c77 c78 c87 c88
    prepare     ymm7
    dot_muladd  ymm2,   ymm10
    dot_muladd  ymm3,   ymm11

    # Load bottom half of A:
    vmovups     ymm0,   [rdi+0x80]
    vmovups     ymm1,   [rdi+0xA0]
    vmovups     ymm2,   [rdi+0xC0]
    vmovups     ymm3,   [rdi+0xE0]

    # Load top half of B.
    vmovups     xmm4,   [rsi+0x00]
    vmovups     xmm5,   [rsi+0x40]
    vmovups     xmm6,   [rsi+0x80]
    vmovups     xmm7,   [rsi+0xC0]
    vinsertf128 ymm4,   ymm4,   [rsi+0x30], 1
    vinsertf128 ymm5,   ymm5,   [rsi+0x70], 1
    vinsertf128 ymm6,   ymm6,   [rsi+0xB0], 1
    vinsertf128 ymm7,   ymm7,   [rsi+0xF0], 1

    # Compute (3/4):
    #   c51 c52 c61 c62 c53 c54 c63 c64
    #   c71 c72 c81 c82 c73 c74 c83 c84
    prepare     ymm4
    dot_muladd  ymm0,   ymm8
    dot_muladd  ymm1,   ymm9

    # Compute (3/4):
    #   c55 c56 c65 c66 c57 c58 c67 c68
    #   c75 c76 c85 c86 c77 c78 c87 c88
    prepare     ymm5
    dot_muladd  ymm0,   ymm10
    dot_muladd  ymm1,   ymm11

    # Compute (4/4):
    #   c51 c52 c61 c62 c53 c54 c63 c64
    #   c71 c72 c81 c82 c73 c74 c83 c84
    prepare     ymm6
    dot_muladd  ymm2,   ymm8
    dot_muladd  ymm3,   ymm9

    # Compute (4/4):
    #   c55 c56 c65 c66 c57 c58 c67 c68
    #   c75 c76 c85 c86 c77 c78 c87 c88
    prepare     ymm7
    dot_muladd  ymm2,   ymm10
    dot_muladd  ymm3,   ymm11

    # Store bottom half of C.
    vmovups [rdx+0x80], ymm8
    vmovups [rdx+0xA0], ymm9
    vmovups [rdx+0xC0], ymm10
    vmovups [rdx+0xE0], ymm11

    ret
