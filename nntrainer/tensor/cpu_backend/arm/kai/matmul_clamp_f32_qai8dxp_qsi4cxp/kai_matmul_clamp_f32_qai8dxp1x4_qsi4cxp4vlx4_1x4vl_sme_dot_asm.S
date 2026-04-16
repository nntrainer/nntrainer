//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot)
    stp  x19, x20, [sp, -144]!
    stp  x21, x22, [sp, 16]
    stp  x23, x24, [sp, 32]
    stp  x25, x26, [sp, 48]
    stp  x27, x28, [sp, 64]
    stp  d8,  d9,  [sp, 80]
    stp  d10, d11, [sp, 96]
    stp  d12, d13, [sp, 112]
    stp  d14, d15, [sp, 128]

    // Setup
    KAI_ASM_INST(0xd503477f) // smstart
    mov  x11, #0
    ldr  x16, [x0]
    ldr  x15, [x0, #0x28]
    cntw x19, ALL, MUL #4
    ldr  x21, [x0, #0x8]
    ldr  x8,  [x0, #0x10]
    ldr  x17, [x0, #0x50]
    ldr  x14, [x0, #0x38]
    ldr  x12, [x0, #0x48]
    ldr  x13, [x0, #0x18]
    ldr  x10, [x0, #0x40]
    ldr  x7,  [x0, #0x58]
    ptrue  p0.b
    mov  w9, #0xf0
    dup  z28.b, w9
    // predicate to load nr words for the RHS sums and scaling factors (should be exactly all true)
    whilelt p6.s, x11, x19
    KAI_ASM_INST(0x8558c01e) // ld1rw   { z30.s }, p0/z, [x0, #0x60]
    KAI_ASM_INST(0x8559c01f) // ld1rw   { z31.s }, p0/z, [x0, #0x64]

    // lhs matrix row loop
KAI_ASM_LABEL(label_1)
    // Reset rhs matrix ptr
    mov x26, x8
    // Reset dst_ptr to dst of next GEMV result
    mov x27, x16
    // Reset n index
    mov x22, #0
    whilelt p3.s, x22, x15

    // rhs matrix row loop (transposed so theoretical columns)
KAI_ASM_LABEL(label_2)

    // Reset rhs block ptr to start of row
    mov     x24, x26
    add     x25, x26, x17
    whilelt p4.b, x24, x25
    addvl   x28, x24, #4
    whilelt p5.b, x28, x25
    mov     x23, #0
    whilelt p1.b, x23, x14
    // Zero for sdot accumulation in inner loop
    dup     z18.s, #0
    dup     z19.s, #0
    dup     z20.s, #0
    dup     z21.s, #0

    // before k loop
KAI_ASM_LABEL(label_3)

    // Load lhs
    KAI_ASM_INST(0xa41706a0) // ld1rqb  { z0.b }, p1/z , [x21, x23]
    add     x23, x23, #16
    whilelt p1.b, x23, x14
    KAI_ASM_INST(0xa41706a1) // ld1rqb  { z1.b }, p1/z , [x21, x23]

    // Load w
    ld1b z2.b, p4/z, [x24]
    ld1b z3.b, p4/z, [x24, #1, mul vl]
    ld1b z4.b, p4/z, [x24, #2, mul vl]
    ld1b z5.b, p4/z, [x24, #3, mul vl]
    ld1b z6.b, p5/z, [x24, #4, mul vl]
    ld1b z7.b, p5/z, [x24, #5, mul vl]
    ld1b z8.b, p5/z, [x24, #6, mul vl]
    ld1b z9.b, p5/z, [x24, #7, mul vl]
    addvl x24, x24, #8
    whilelt p4.b, x24, x25
    addvl   x28, x24, #4
    whilelt p5.b, x28, x25
    ld1b z10.b, p4/z, [x24]
    ld1b z11.b, p4/z, [x24, #1, mul vl]
    ld1b z12.b, p4/z, [x24, #2, mul vl]
    ld1b z13.b, p4/z, [x24, #3, mul vl]
    ld1b z14.b, p5/z, [x24, #4, mul vl]
    ld1b z15.b, p5/z, [x24, #5, mul vl]
    ld1b z16.b, p5/z, [x24, #6, mul vl]
    ld1b z17.b, p5/z, [x24, #7, mul vl]
    addvl x24, x24, #8

    // rhs i4 to i8 and sdot
    lsl  z22.b, z2.b, #4
    lsl  z23.b, z3.b, #4
    lsl  z24.b, z4.b, #4
    lsl  z25.b, z5.b, #4
    sdot z18.s, z22.b, z0.b[0]
    sdot z19.s, z23.b, z0.b[0]
    sdot z20.s, z24.b, z0.b[0]
    sdot z21.s, z25.b, z0.b[0]
    lsl  z22.b, z6.b, #4
    lsl  z23.b, z7.b, #4
    lsl  z24.b, z8.b, #4
    lsl  z25.b, z9.b, #4
    sdot z18.s, z22.b, z0.b[1]
    sdot z19.s, z23.b, z0.b[1]
    sdot z20.s, z24.b, z0.b[1]
    sdot z21.s, z25.b, z0.b[1]
    lsl  z22.b, z10.b, #4
    lsl  z23.b, z11.b, #4
    lsl  z24.b, z12.b, #4
    lsl  z25.b, z13.b, #4
    sdot z18.s, z22.b, z0.b[2]
    sdot z19.s, z23.b, z0.b[2]
    sdot z20.s, z24.b, z0.b[2]
    sdot z21.s, z25.b, z0.b[2]
    lsl  z22.b, z14.b, #4
    lsl  z23.b, z15.b, #4
    lsl  z24.b, z16.b, #4
    lsl  z25.b, z17.b, #4
    sdot z18.s, z22.b, z0.b[3]
    sdot z19.s, z23.b, z0.b[3]
    sdot z20.s, z24.b, z0.b[3]
    sdot z21.s, z25.b, z0.b[3]
    and  z2.d, z2.d, z28.d
    and  z3.d, z3.d, z28.d
    and  z4.d, z4.d, z28.d
    and  z5.d, z5.d, z28.d
    and  z6.d, z6.d, z28.d
    and  z7.d, z7.d, z28.d
    and  z8.d, z8.d, z28.d
    and  z9.d, z9.d, z28.d
    sdot z18.s, z2.b, z1.b[0]
    sdot z19.s, z3.b, z1.b[0]
    sdot z20.s, z4.b, z1.b[0]
    sdot z21.s, z5.b, z1.b[0]
    sdot z18.s, z6.b, z1.b[1]
    sdot z19.s, z7.b, z1.b[1]
    sdot z20.s, z8.b, z1.b[1]
    sdot z21.s, z9.b, z1.b[1]
    and  z10.d, z10.d, z28.d
    and  z11.d, z11.d, z28.d
    and  z12.d, z12.d, z28.d
    and  z13.d, z13.d, z28.d
    and  z14.d, z14.d, z28.d
    and  z15.d, z15.d, z28.d
    and  z16.d, z16.d, z28.d
    and  z17.d, z17.d, z28.d
    sdot z18.s, z10.b, z1.b[2]
    sdot z19.s, z11.b, z1.b[2]
    sdot z20.s, z12.b, z1.b[2]
    sdot z21.s, z13.b, z1.b[2]
    sdot z18.s, z14.b, z1.b[3]
    sdot z19.s, z15.b, z1.b[3]
    sdot z20.s, z16.b, z1.b[3]
    sdot z21.s, z17.b, z1.b[3]

    // End K block loop
    whilelt p4.b, x24, x25
    addvl   x28, x24, #4
    whilelt p5.b, x28, x25
    add     x23, x23, #16
    whilelt p1.b, x23, x14
    b.mi label_3    //b.first label_3

    // Finish of accumulators with scaling factors and zero points

    // Load lhs zero point
    add  x28, x21, x14
    KAI_ASM_INST(0x8540c382) // ld1rw   { z2.s }, p0/z , [x28]
    // Load lhs scaling factor
    KAI_ASM_INST(0x8541c383) // ld1rw   { z3.s }, p0/z , [x28, #4]
    // Load rhs sums
    add  x28, x26, x17
    ld1w z4.s, p6/z, [x28]
    ld1w z5.s, p6/z, [x28, #1, mul vl]
    ld1w z6.s, p6/z, [x28, #2, mul vl]
    ld1w z7.s, p6/z, [x28, #3, mul vl]
    // Load rhs scaling factors
    ld1w z8.s,  p6/z, [x28, #4, mul vl]
    ld1w z9.s,  p6/z, [x28, #5, mul vl]
    ld1w z10.s, p6/z, [x28, #6, mul vl]
    ld1w z11.s, p6/z, [x28, #7, mul vl]
    // Load biases
    addvl x28, x28, #8
    ld1w  z12.s, p6/z, [x28]
    ld1w  z13.s, p6/z, [x28, #1, mul vl]
    ld1w  z14.s, p6/z, [x28, #2, mul vl]
    ld1w  z15.s, p6/z, [x28, #3, mul vl]

    // zero point * rhs row sum
    mla z18.s, p0/m, z4.s, z2.s
    mla z19.s, p0/m, z5.s, z2.s
    mla z20.s, p0/m, z6.s, z2.s
    mla z21.s, p0/m, z7.s, z2.s

    // Convert to float
    scvtf z18.s, p0/m, z18.s
    scvtf z19.s, p0/m, z19.s
    scvtf z20.s, p0/m, z20.s
    scvtf z21.s, p0/m, z21.s

    // lhs scaling factor * rhs scaling factor
    fmul z8.s,  z8.s,  z3.s
    fmul z9.s,  z9.s,  z3.s
    fmul z10.s, z10.s, z3.s
    fmul z11.s, z11.s, z3.s

    // Bias + combined scaling factor * combined accumulator
    fmla z12.s, p0/m, z8.s,  z18.s
    fmla z13.s, p0/m, z9.s,  z19.s
    fmla z14.s, p0/m, z10.s, z20.s
    fmla z15.s, p0/m, z11.s, z21.s

    // Clamp
    fmax z12.s, p0/M, z12.s, z30.s
    fmin z12.s, p0/M, z12.s, z31.s
    fmax z13.s, p0/M, z13.s, z30.s
    fmin z13.s, p0/M, z13.s, z31.s
    fmax z14.s, p0/M, z14.s, z30.s
    fmin z14.s, p0/M, z14.s, z31.s
    fmax z15.s, p0/M, z15.s, z30.s
    fmin z15.s, p0/M, z15.s, z31.s

    // Store
    add  x9,  x27, x22, lsl #2
    mov  x6,  x22
    whilelt p3.s, x6,  x15
    st1w  z12.s, p3, [x9]
    incw x6
    whilelt p3.s, x6,  x15
    st1w  z13.s, p3, [x9, #1, mul vl]
    incw x6
    whilelt p3.s, x6,  x15
    st1w z14.s, p3, [x9, #2, mul vl]
    incw x6
    whilelt p3.s, x6,  x15
    st1w z15.s, p3, [x9, #3, mul vl]

    // End rhs row loop
    add  x26, x26, x12
    add     x22, x22, x19
    whilelt p3.s, x22, x15
    b.lt    label_2

    // End lhs row loop
    add  x16, x16, x13
    add  x21, x21, x10
    cmp  x21, x7
    b.lt label_1

    KAI_ASM_INST(0xd503467f) // smstop
    ldp  d14, d15, [sp, 128]
    ldp  d12, d13, [sp, 112]
    ldp  d10, d11, [sp, 96]
    ldp  d8,  d9,  [sp, 80]
    ldp  x27, x28, [sp, 64]
    ldp  x25, x26, [sp, 48]
    ldp  x23, x24, [sp, 32]
    ldp  x21, x22, [sp, 16]
    ldp  x19, x20, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot)

    KAI_ASM_END
