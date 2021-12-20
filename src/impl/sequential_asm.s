.intel_syntax noprefix


.macro  bcast   since:req stride:req
    lea             rbx,    [\since]
    vbroadcastss    ymm8,   [rbx]
    add             rbx,    \stride
    vbroadcastss    ymm9,   [rbx]
    add             rbx,    \stride
    vbroadcastss    ymm10,  [rbx]
    add             rbx,    \stride
    vbroadcastss    ymm11,  [rbx]
.endm

.macro  dot_mul r0:req r1:req r2:req r3:req
    vmulps  \r0,    ymm8,   ymm12
    vmulps  \r1,    ymm9,   ymm12
    vmulps  \r2,    ymm10,  ymm12
    vmulps  \r3,    ymm11,  ymm12
.endm

.macro  dot_muladd  r0:req r1:req r2:req r3:req
    vmulps  ymm13,  ymm8,   ymm12
    vaddps  \r0,    \r0,    ymm13
    vmulps  ymm13,  ymm9,   ymm12
    vaddps  \r1,    \r1,    ymm13
    vmulps  ymm13,  ymm10,  ymm12
    vaddps  \r2,    \r2,    ymm13
    vmulps  ymm13,  ymm11,  ymm12
    vaddps  \r3,    \r3,    ymm13
.endm


.global _impl_sequential_asm_x1_x1_x1
_impl_sequential_asm_x1_x1_x1:
    # rdi :: float *ptr_a
    # rsi :: float *ptr_b
    # rdx :: float *ptr_c
    # rcx :: size_t stride_a
    # r8  :: size_t stride_b
    # r9  :: size_t stride_c

    push    rbx

    mov     rax,    rdx
    vmovups ymm0,   [rax]
    add     rax,    r9
    vmovups ymm1,   [rax]
    add     rax,    r9
    vmovups ymm2,   [rax]
    add     rax,    r9
    vmovups ymm3,   [rax]
    add     rax,    r9
    vmovups ymm4,   [rax]
    add     rax,    r9
    vmovups ymm5,   [rax]
    add     rax,    r9
    vmovups ymm6,   [rax]
    add     rax,    r9
    vmovups ymm7,   [rax]

    vmovups     ymm12,  [rsi]
    bcast       rdi,    rcx
    dot_muladd  ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx), rcx
    dot_muladd  ymm4,   ymm5,   ymm6,   ymm7

  _impl_sequential_asm_x1_x1_x1_zfcont:
    lea         rax,    [rsi + r8]
    vmovups     ymm12,  [rax]
    bcast       rdi + 0x04, rcx
    dot_muladd  ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx + 0x04), rcx
    dot_muladd  ymm4,   ymm5,   ymm6,   ymm7

    add         rax,    r8
    vmovups     ymm12,  [rax]
    bcast       rdi + 0x08, rcx
    dot_muladd  ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx + 0x08), rcx
    dot_muladd  ymm4,   ymm5,   ymm6,   ymm7

    add         rax,    r8
    vmovups     ymm12,  [rax]
    bcast       rdi + 0x0C, rcx
    dot_muladd  ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx + 0x0C), rcx
    dot_muladd  ymm4,   ymm5,   ymm6,   ymm7

    add         rax,    r8
    vmovups     ymm12,  [rax]
    bcast       rdi + 0x10, rcx
    dot_muladd  ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx + 0x10), rcx
    dot_muladd  ymm4,   ymm5,   ymm6,   ymm7

    add         rax,    r8
    vmovups     ymm12,  [rax]
    bcast       rdi + 0x14, rcx
    dot_muladd  ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx + 0x14), rcx
    dot_muladd  ymm4,   ymm5,   ymm6,   ymm7

    add         rax,    r8
    vmovups     ymm12,  [rax]
    bcast       rdi + 0x18, rcx
    dot_muladd  ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx + 0x18), rcx
    dot_muladd  ymm4,   ymm5,   ymm6,   ymm7

    add         rax,    r8
    vmovups     ymm12,  [rax]
    bcast       rdi + 0x1C, rcx
    dot_muladd  ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx + 0x1C), rcx
    dot_muladd  ymm4,   ymm5,   ymm6,   ymm7

    mov     rax,    rdx
    vmovups [rax],  ymm0
    add     rax,    r9
    vmovups [rax],  ymm1
    add     rax,    r9
    vmovups [rax],  ymm2
    add     rax,    r9
    vmovups [rax],  ymm3
    add     rax,    r9
    vmovups [rax],  ymm4
    add     rax,    r9
    vmovups [rax],  ymm5
    add     rax,    r9
    vmovups [rax],  ymm6
    add     rax,    r9
    vmovups [rax],  ymm7

    pop     rbx
    ret


.global _impl_sequential_asm_x1_x1_x1_zf
_impl_sequential_asm_x1_x1_x1_zf:
    # rdi :: float *ptr_a
    # rsi :: float *ptr_b
    # rdx :: float *ptr_c
    # rcx :: size_t stride_a
    # r8  :: size_t stride_b
    # r9  :: size_t stride_c

    push        rbx

    vmovups     ymm12,  [rsi]
    bcast       rdi,    rcx
    dot_mul     ymm0,   ymm1,   ymm2,   ymm3
    bcast       (rdi + 4 * rcx), rcx
    dot_mul     ymm4,   ymm5,   ymm6,   ymm7

    jmp     _impl_sequential_asm_x1_x1_x1_zfcont
