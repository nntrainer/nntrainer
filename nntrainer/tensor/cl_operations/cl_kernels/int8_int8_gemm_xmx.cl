// =============================================================================
// int8_int8_gemm_xmx.cl  --  Xe2 (Lunar Lake) w4a8 FC GEMM, XMX/DPAS path.
//
// DROP-IN replacement for nntrainer's v8c_gemm_int8_int4_buf (Intel buffer-path
// w4a8 FC GEMM, the M>4 prefill kernel). Same host SetKernelArguments order,
// same v8c weight packing, BYTE-IDENTICAL epilogue.
//
//   Y[M][N] (fp16) where, per element (m,n):
//     acc_raw   = sum_k  act_int8[m][k] * nibble_raw(n,k)      (i8_u8 DPAS)
//     corrected = acc_raw - 8*row_sum_act[m] - zp_act[m]*row_sum_w_int4[n]
//     Y[m][n]   = (half)((float)corrected * scale_act[m] * scale_wgt[n])
//
// Device : Intel Arc Graphics, Lunar Lake Xe2, NEO 25.18, IGC 2.11, SG16 only.
// Compile: -cl-std=CL3.0 -cl-intel-256-GRF-per-thread
//
// -----------------------------------------------------------------------------
// WEIGHT LAYOUT (v8c packing, NOT a plain [N][K/2]).
//   Per output channel n the row starts at byte offset n*(K/2). K is split into
//   32-K blocks of 16 bytes. Within a 32-K block, byte (c*4+b), c in 0..3,
//   b in 0..3:
//        lo nibble (byte & 0xF)     = uint4 at K = c*8 + b
//        hi nibble ((byte>>4)&0xF)  = uint4 at K = c*8 + b + 4
//   The stored nibble is the RAW uint4 in [0,15]; decoded weight = nibble-8.
//
//   DERIVED VNNI mapping for the i8_u8 DPAS B operand of a 32-K slab: dword
//   g (g=0..7) holds the 4 RAW nibbles for natural K = 4g, 4g+1, 4g+2, 4g+3
//   packed little-endian. With c=g>>1, useHi=g&1 the 4 nibbles come from bytes
//   [c*4+0 .. c*4+3] of the slab, low nibble if useHi==0 else high nibble:
//     g even (useHi=0): K = c*8 + {0,1,2,3} = 4g + {0,1,2,3}     [natural]
//     g odd  (useHi=1): K = c*8 + {4,5,6,7} = 4g + {0,1,2,3}     [natural]
//
// WEIGHT READ: intel_sub_group_2d_block_read_transpose_32b_16r8x1c reads a
//   16(row) x 8(col) tile of 32-bit words TRANSPOSED so lane L gets row L's 8
//   words = 32 contiguous bytes of channel (n0+L) = TWO 32-K slabs (KSUB=64).
//   The surface is the byte weight buffer reinterpreted as 32-bit words:
//   width = (K/2) bytes, height = N rows, pitch = (K/2) bytes,
//   coord x = (k0/2)/4 = k0/8  (in 32-bit elements), coord y = n0.
//
// A READ: intel_sub_group_2d_block_read_8b_8r32x1c reads an 8(row)x32(col)
//   tile of bytes -> ushort[8] == short8; lane L holds row L's 32 K-bytes
//   (VNNI: lane l contributes K 2l, 2l+1 of each of the 8 rows). Surface =
//   act byte buffer, width = K bytes, height = M rows, pitch = K, coord
//   x = kk (byte), y = m0.
//
//   IMPORTANT: surface height is the REAL M / N (not the padded grid) so the
//   2D block messages clamp OOB rows to 0 -- padded M rows read as 0 and the
//   M_valid store guard suppresses their writes.
// -----------------------------------------------------------------------------

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
// FIX 1 (defense in depth): this kernel calls the DPAS matrix-MAD builtin
// (intel_sub_group_i8_u8_matrix_mad_k32, below) which requires the actual
// systolic array, not just cl_intel_subgroups (advertised by every Intel GPU
// since Gen9). Declaring the extension explicitly makes a non-DPAS compiler
// fail this build cleanly instead of silently software-emulating the DPAS
// builtin -- the host-side gate (caps().dpas in blas_kernels.cpp) should
// already keep this kernel off such devices, but this is a second line of
// defense at the compiler.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
// Same defense for the 2D block-io builtins used for the weight/activation
// reads (intel_sub_group_2d_block_read_*): declare the extension so a compiler
// without it rejects the build cleanly.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tuned on Intel Arc Lunar Lake Xe2 (NEO 25.18) at M1024/N4096/K4096:
// MT=4,NT=4,SG_M=1 -> ~30.6 TOP/s (one subgroup per WG maximizes the number
// of concurrent workgroups = occupancy; larger SG_M serialized hard here).
#ifndef MT
#define MT 4          // DPAS M-tiles per subgroup: rows = MT*8
#endif
#ifndef NT
#define NT 4          // N-blocks per subgroup: cols = NT*16
#endif
#ifndef SG_M
#define SG_M 1        // subgroups stacked along M within a WG (share N-block)
#endif

#define M_PER_SG (MT * 8)
#define N_PER_SG (NT * 16)
#define M_PER_WG (M_PER_SG * SG_M)

// One transpose_32b_16r8x1c read = 8 uints = 32 bytes = 64 K-values / lane.
#define KSUB 64

// --------------------------------------------------------------------------
// SWAR: turn one 32-K slab's 16 weight-bytes (uint4) into the 8-dword VNNI B
// operand of RAW UNSIGNED nibbles (0..15) -- no "-8" here (deferred to epilogue
// via the host-supplied row_sum_act / row_sum_w_int4).
//
// Input w4 holds bytes b[0..15] LE across 4 dwords (w4.x=b3 b2 b1 b0, ...).
//   slab byte (c*4+b): K(c*8+b)=low nibble, K(c*8+b+4)=high nibble.
// Output VNNI dword g (0..7), with c=g>>1, useHi=g&1, packs from input dword
//   x = w4[c] (its 4 bytes are slab bytes c*4+{0,1,2,3}):
//     useHi==0 -> the 4 LOW  nibbles of x  (K 4g..4g+3)
//     useHi==1 -> the 4 HIGH nibbles of x  (K 4g..4g+3)
// Each input dword x therefore yields TWO output dwords: lo-nibble pack then
// hi-nibble pack. Pure mask/shift, branch-free, unsigned (for i8_u8 DPAS).
// --------------------------------------------------------------------------
static inline uint8 swar_unpack_unsigned(uint4 w4) {
  uint8 out;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint x = (c == 0) ? w4.x : (c == 1) ? w4.y : (c == 2) ? w4.z : w4.w;
    const uint lo = x & 0x0F0F0F0Fu;        // 4 low nibbles, byte-aligned
    const uint hi = (x >> 4) & 0x0F0F0F0Fu; // 4 high nibbles, byte-aligned
    out[2 * c]     = lo; // g = 2c   (useHi=0) -> K 4*(2c)   .. +3
    out[2 * c + 1] = hi; // g = 2c+1 (useHi=1) -> K 4*(2c+1) .. +3
  }
  return out;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, SG_M, 1)))
__kernel void gemm_xmx_i4(
    __global const char *act,            // arg0: int8 act [M][K] row-major
    __global const uchar *wgt,           // arg1: int4 weight, v8c-packed
    __global const float *scale_act,     // arg2: [M] fp32
    __global const float *scale_wgt,     // arg3: [N] fp32
    __global const int *row_sum_act,     // arg4: [M] int = sum_k act_int8
    __global const int *zp_act,          // arg5: [M] int (act zero point)
    __global const int *row_sum_w_int4,  // arg6: [N] int = sum_k(nibble-8)
    __global half *Y,                    // arg7: [M][N] fp16 output
    const int M, const int N, const int K,        // arg8,9,10
    const int W_act, const int W_wgt,             // arg11,12 (unused; byte off)
    const int M_valid) {                          // arg13: store guard
  const int lane = get_sub_group_local_id();      // 0..15
  const int sg   = get_sub_group_id();            // 0..SG_M-1 (M stacking)

  const int n0 = (int)get_group_id(0) * N_PER_SG;                  // first col
  const int m0 = (int)get_group_id(1) * M_PER_WG + sg * M_PER_SG;  // first row

  // acc_raw[m-tile][n-block].s{r} = sum_k A * nibble_raw (i8_u8 DPAS).
  int8 acc[MT][NT];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j] = (int8)(0);

  // ------------------ K loop (KSUB=64 -> two 32-K DPAS slabs) ----------------
  for (int k0 = 0; k0 < K; k0 += KSUB) {
    // Weight reads: 32 bytes (64 K) per N-block channel. Issue all NT messages
    // first; IGC overlaps them with the independent MMA chain below.
    uint8 wraw[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      uint draw[8];
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
          (__global void *)wgt, K / 2, N, K / 2,
          (int2)(k0 >> 3, n0 + j * 16), draw);
      wraw[j] = *(uint8 *)draw;
    }

    // ---- two 32-K DPAS slabs ----
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int kk = k0 + s * 32; // K base for this DPAS slab

      // SWAR-unpack this slab's weights (16 bytes/channel) for all NT.
      uint8 b[NT];
#pragma unroll
      for (int j = 0; j < NT; ++j)
        b[j] = swar_unpack_unsigned((s == 0) ? wraw[j].lo : wraw[j].hi);

      // Per M-tile: load A (8x32), then NT raw-nibble DPAS.
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        ushort a_raw[8];
        intel_sub_group_2d_block_read_8b_8r32x1c(
            (__global void *)act, K, M, K, (int2)(kk, m0 + i * 8), a_raw);
        const short8 a = *(short8 *)a_raw;

#pragma unroll
        for (int j = 0; j < NT; ++j)
          acc[i][j] = intel_sub_group_i8_u8_matrix_mad_k32(a, b[j], acc[i][j]);
      }
    }
  }

  // ------------------ epilogue (byte-identical to dp4a v8c) ------------------
  // corrected = acc_raw - 8*row_sum_act[m] - zp_act[m]*row_sum_w_int4[n]
  // Y[m][n]   = (half)((float)corrected * scale_act[m] * scale_wgt[n])
  // Each lane owns 8 rows of one 16-wide N-block: lane -> column n0+j*16+lane,
  // the int8 components s0..s7 -> the 8 M-rows row_base+0..+7.
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int col = n0 + j * 16 + lane;
    if (col >= N) continue;
    const float sw  = scale_wgt[col];
    const int   rsw = row_sum_w_int4[col];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      const int row_base = m0 + i * 8;
      const int8 r = acc[i][j];
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int m = row_base + rr;
        if (m >= M_valid) continue;
        const int acc_raw = r[rr];
        const int corrected =
            acc_raw - 8 * row_sum_act[m] - zp_act[m] * rsw;
        const float v = (float)corrected * scale_act[m] * sw;
        Y[(long)m * N + col] = (half)v;
      }
    }
  }
}
