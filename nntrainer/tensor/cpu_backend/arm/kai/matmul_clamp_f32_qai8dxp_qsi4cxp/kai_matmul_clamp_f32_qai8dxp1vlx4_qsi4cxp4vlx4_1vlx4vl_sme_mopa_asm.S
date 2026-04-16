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
    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa)
    stp x19, x20, [sp, -160]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp x27, x28, [sp, 64]
    stp d8, d9,   [sp, 80]
    stp d10, d11, [sp, 96]
    stp d12, d13, [sp, 112]
    stp d14, d15, [sp, 128]
    KAI_ASM_INST(0xd503477f) //smstart
    ldr x19, [x0]                                // dst
    ldr x20, [x0, #0x8]                          // lhs_packed
    ldr x21, [x0, #0x18]                         // dst_stride_row
    ldr x23, [x0, #0x30]                         // lhs_stride
    ldr x22, [x0, #0x38]                         // rhs_stride
    ldr x24, [x0, #0x40]                         // m_blk
    ldr x26, [x0, #0x48]                         // dst_inc
    mov w9, #0xf0
    dup z28.b, w9
    cntw x7
    lsl x27, x7, #2
    ptrue p2.b
    KAI_ASM_INST(0x8554c81e) // ld1rw { z30.s }, p2/z, [x0, #0x50] // clamp_min
    KAI_ASM_INST(0x8555c81f) // ld1rw { z31.s }, p2/z, [x0, #0x54] // clamp_max
    // M loop head
    ldr  x12, [x0, #0x20]                        // m
    KAI_ASM_INST(0x25ac17e0) //whilelt p0.s, xzr, x12
KAI_ASM_LABEL(label_1)
    ldr  x8, [x0, #0x10]                         // rhs_packed
    mov  x9, x19
    ldr  x13, [x0, #0x28]                        // n
    cmp  x7, x12
    csel x16, x7, x12, lt
    lsl  x16, x16, #2
    // N loop head
    whilelt p3.h, xzr, x13
KAI_ASM_LABEL(label_2)
    mov x10, x20
    mov x11, x8
    mov x17, x9
    whilelt p5.s, xzr, x13
    // K loop
    KAI_ASM_INST(0xc00800ff) //zero    {za}
    add x14, x10, x24
KAI_ASM_LABEL(label_3)
    KAI_ASM_INST(0xa540a150) // ld1w { z16.s }, p0/z, [x10]
    KAI_ASM_INST(0xa544a154) // ld1w { z20.s }, p0/z, [x10, #0x4, mul vl]
    ld1h z0.h, p3/z, [x11]
    ld1h z1.h, p3/z, [x11, #1, mul vl]
    ld1h z2.h, p3/z, [x11, #2, mul vl]
    ld1h z3.h, p3/z, [x11, #3, mul vl]
    lsl  z24.b, z0.b, #4
    lsl  z25.b, z1.b, #4
    lsl  z26.b, z2.b, #4
    lsl  z27.b, z3.b, #4
    KAI_ASM_INST(0xa0984a00) // smopa za0.s, p2/m, p2/m, z16.b, z24.b
    KAI_ASM_INST(0xa0994a01) // smopa za1.s, p2/m, p2/m, z16.b, z25.b
    KAI_ASM_INST(0xa09a4a02) // smopa za2.s, p2/m, p2/m, z16.b, z26.b
    KAI_ASM_INST(0xa09b4a03) // smopa za3.s, p2/m, p2/m, z16.b, z27.b
    and z0.d, z0.d, z28.d
    and z1.d, z1.d, z28.d
    and z2.d, z2.d, z28.d
    and z3.d, z3.d, z28.d
    KAI_ASM_INST(0xa0804a80) // smopa za0.s, p2/m, p2/m, z20.b, z0.b
    KAI_ASM_INST(0xa0814a81) // smopa za1.s, p2/m, p2/m, z20.b, z1.b
    KAI_ASM_INST(0xa0824a82) // smopa za2.s, p2/m, p2/m, z20.b, z2.b
    KAI_ASM_INST(0xa0834a83) // smopa za3.s, p2/m, p2/m, z20.b, z3.b
    KAI_ASM_INST(0xa541a151) // ld1w { z17.s }, p0/z, [x10, #0x1, mul vl]
    KAI_ASM_INST(0xa545a155) // ld1w { z21.s }, p0/z, [x10, #0x5, mul vl]
    ld1h z4.h, p3/z, [x11, #4, mul vl]
    ld1h z5.h, p3/z, [x11, #5, mul vl]
    ld1h z6.h, p3/z, [x11, #6, mul vl]
    ld1h z7.h, p3/z, [x11, #7, mul vl]
    addvl x11, x11, #8
    lsl z24.b, z4.b, #4
    lsl z25.b, z5.b, #4
    lsl z26.b, z6.b, #4
    lsl z27.b, z7.b, #4
    KAI_ASM_INST(0xa0984a20) // smopa za0.s, p2/m, p2/m, z17.b, z24.b
    KAI_ASM_INST(0xa0994a21) // smopa za1.s, p2/m, p2/m, z17.b, z25.b
    KAI_ASM_INST(0xa09a4a22) // smopa za2.s, p2/m, p2/m, z17.b, z26.b
    KAI_ASM_INST(0xa09b4a23) // smopa za3.s, p2/m, p2/m, z17.b, z27.b
    and z4.d, z4.d, z28.d
    and z5.d, z5.d, z28.d
    and z6.d, z6.d, z28.d
    and z7.d, z7.d, z28.d
    KAI_ASM_INST(0xa0844aa0) // smopa za0.s, p2/m, p2/m, z21.b, z4.b
    KAI_ASM_INST(0xa0854aa1) // smopa za1.s, p2/m, p2/m, z21.b, z5.b
    KAI_ASM_INST(0xa0864aa2) // smopa za2.s, p2/m, p2/m, z21.b, z6.b
    KAI_ASM_INST(0xa0874aa3) // smopa za3.s, p2/m, p2/m, z21.b, z7.b
    KAI_ASM_INST(0xa542a152) // ld1w { z18.s }, p0/z, [x10, #0x2, mul vl]
    KAI_ASM_INST(0xa546a156) // ld1w { z22.s }, p0/z, [x10, #0x6, mul vl]
    ld1h z8.h,  p3/z, [x11]
    ld1h z9.h,  p3/z, [x11, #1, mul vl]
    ld1h z10.h, p3/z, [x11, #2, mul vl]
    ld1h z11.h, p3/z, [x11, #3, mul vl]
    lsl  z24.b, z8.b, #4
    lsl  z25.b, z9.b, #4
    lsl  z26.b, z10.b, #4
    lsl  z27.b, z11.b, #4
    KAI_ASM_INST(0xa0984a40) // smopa za0.s, p2/m, p2/m, z18.b, z24.b
    KAI_ASM_INST(0xa0994a41) // smopa za1.s, p2/m, p2/m, z18.b, z25.b
    KAI_ASM_INST(0xa09a4a42) // smopa za2.s, p2/m, p2/m, z18.b, z26.b
    KAI_ASM_INST(0xa09b4a43) // smopa za3.s, p2/m, p2/m, z18.b, z27.b
    and z8.d, z8.d, z28.d
    and z9.d, z9.d, z28.d
    and z10.d, z10.d, z28.d
    and z11.d, z11.d, z28.d
    KAI_ASM_INST(0xa0884ac0) // smopa za0.s, p2/m, p2/m, z22.b, z8.b
    KAI_ASM_INST(0xa0894ac1) // smopa za1.s, p2/m, p2/m, z22.b, z9.b
    KAI_ASM_INST(0xa08a4ac2) // smopa za2.s, p2/m, p2/m, z22.b, z10.b
    KAI_ASM_INST(0xa08b4ac3) // smopa za3.s, p2/m, p2/m, z22.b, z11.b
    KAI_ASM_INST(0xa543a153) // ld1w { z19.s }, p0/z, [x10, #0x3, mul vl]
    KAI_ASM_INST(0xa547a157) // ld1w { z23.s }, p0/z, [x10, #0x7, mul vl]
    ld1h z12.h, p3/z, [x11, #4, mul vl]
    ld1h z13.h, p3/z, [x11, #5, mul vl]
    ld1h z14.h, p3/z, [x11, #6, mul vl]
    ld1h z15.h, p3/z, [x11, #7, mul vl]
    lsl  z24.b, z12.b, #4
    lsl  z25.b, z13.b, #4
    lsl  z26.b, z14.b, #4
    lsl  z27.b, z15.b, #4
    KAI_ASM_INST(0xa0984a60) // smopa za0.s, p2/m, p2/m, z19.b, z24.b
    KAI_ASM_INST(0xa0994a61) // smopa za1.s, p2/m, p2/m, z19.b, z25.b
    KAI_ASM_INST(0xa09a4a62) // smopa za2.s, p2/m, p2/m, z19.b, z26.b
    KAI_ASM_INST(0xa09b4a63) // smopa za3.s, p2/m, p2/m, z19.b, z27.b
    and z12.d, z12.d, z28.d
    and z13.d, z13.d, z28.d
    and z14.d, z14.d, z28.d
    and z15.d, z15.d, z28.d
    KAI_ASM_INST(0xa08c4ae0) // smopa za0.s, p2/m, p2/m, z23.b, z12.b
    KAI_ASM_INST(0xa08d4ae1) // smopa za1.s, p2/m, p2/m, z23.b, z13.b
    KAI_ASM_INST(0xa08e4ae2) // smopa za2.s, p2/m, p2/m, z23.b, z14.b
    KAI_ASM_INST(0xa08f4ae3) // smopa za3.s, p2/m, p2/m, z23.b, z15.b
    addvl x11, x11, #8
    addvl x10, x10, #8
    cmp  x10, x14
    b.lt label_3
    // RHS row sum, scale factor & bias
    ld1w z0.s, p5/z, [x11]
    ld1w z1.s, p5/z, [x11, #1, mul vl]
    ld1w z2.s, p5/z, [x11, #2, mul vl]
    ld1w z3.s, p5/z, [x11, #3, mul vl]
    ld1w z4.s, p5/z, [x11, #4, mul vl]
    ld1w z5.s, p5/z, [x11, #5, mul vl]
    ld1w z6.s, p5/z, [x11, #6, mul vl]
    ld1w z7.s, p5/z, [x11, #7, mul vl]
    addvl x11, x11, #0x8
    ld1w z8.s, p5/z, [x11]
    ld1w z9.s, p5/z, [x11, #1, mul vl]
    ld1w z10.s, p5/z, [x11, #2, mul vl]
    ld1w z11.s, p5/z, [x11, #3, mul vl]
    addvl	x11, x11, #0x4
    // Store loop
    mov    x14, #0
    addvl  x15, x10, #1
KAI_ASM_LABEL(label_4)
    // Load LHS Row-offset & SF
    KAI_ASM_INST(0x8540c950) // ld1rw {z16.s}, p2/z, [x10]
    KAI_ASM_INST(0x8540c9f1) // ld1rw {z17.s}, p2/z, [x15]
    add   x10, x10, #4
    add   x15, x15, #4
    // Scaling factors
    fmul z20.s, z17.s, z4.s
    fmul z21.s, z17.s, z5.s
    fmul z22.s, z17.s, z6.s
    fmul z23.s, z17.s, z7.s
    // Load inner accumulation & convert
    KAI_ASM_INST(0xc002480c)	//    mov z12.b, p2/m, za0h.b[w14, 0]
    KAI_ASM_INST(0xc002482d)	//    mov z13.b, p2/m, za0h.b[w14, 1]
    KAI_ASM_INST(0xc002484e)	//    mov z14.b, p2/m, za0h.b[w14, 2]
    KAI_ASM_INST(0xc002486f)	//    mov z15.b, p2/m, za0h.b[w14, 3]
    // Result += iacc + offset x Row-sum
    mla z12.s, p2/m, z16.s, z0.s
    mla z13.s, p2/m, z16.s, z1.s
    mla z14.s, p2/m, z16.s, z2.s
    mla z15.s, p2/m, z16.s, z3.s
    scvtf z12.s, p2/m, z12.s
    scvtf z13.s, p2/m, z13.s
    scvtf z14.s, p2/m, z14.s
    scvtf z15.s, p2/m, z15.s
    // Result = (iacc + offset x Row-sum) x SFs
    fmul z24.s, z12.s, z20.s
    fmul z25.s, z13.s, z21.s
    fmul z26.s, z14.s, z22.s
    fmul z27.s, z15.s, z23.s
    // Add the bias
    fadd z24.s, p2/m, z24.s, z8.s
    fadd z25.s, p2/m, z25.s, z9.s
    fadd z26.s, p2/m, z26.s, z10.s
    fadd z27.s, p2/m, z27.s, z11.s
    // CLAMP and store
    fmax z24.s, p2/M, z24.s, z30.s
    fmin z24.s, p2/M, z24.s, z31.s
    fmax z25.s, p2/M, z25.s, z30.s
    fmin z25.s, p2/M, z25.s, z31.s
    fmax z26.s, p2/M, z26.s, z30.s
    fmin z26.s, p2/M, z26.s, z31.s
    fmax z27.s, p2/M, z27.s, z30.s
    fmin z27.s, p2/M, z27.s, z31.s
    mov     x28, xzr
    whilelt p5.s, x28, x13
    st1w    z24.s, p5, [x17]
    add     x28, x28, x7
    whilelt p5.s, x28, x13
    st1w    z25.s, p5, [x17, #1, mul vl]
    add     x28, x28, x7
    whilelt p5.s, x28, x13
    st1w    z26.s, p5, [x17, #2, mul vl]
    add     x28, x28, x7
    whilelt p5.s, x28, x13
    st1w    z27.s, p5, [x17, #3, mul vl]
    add  x17, x17, x21
    add  x14, x14, #4
    cmp  x14, x16
    b.lt label_4
    // N loop tail
    add x8, x8, x22
    KAI_ASM_INST(0x04295089) // addvl x9, x9, #4
    sub x13, x13, x27
    whilelt p3.h, xzr, x13
    b.mi label_2
    // M loop tail
    add x20, x20, x23
    add x19, x19, x26
    sub x12, x12, x7
    whilelt p0.s, xzr, x12
    b.mi label_1
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0xd503467f) //smstop
    ldp d14, d15, [sp, 128]
    ldp d12, d13, [sp, 112]
    ldp d10, d11, [sp, 96]
    ldp d8,  d9,  [sp, 80]
    ldp x27, x28, [sp, 64]
    ldp x25, x26, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 160
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa)
    KAI_ASM_END
