
#define TOPK_LS 64
#define TOPK_EPT 16

inline ulong topk_key(ushort h, uint i) {
  const uint k = (h & 0x8000u) ? ((~(uint)h) & 0xffffu) : ((uint)h | 0x8000u);
  return ((ulong)k << 32) | (ulong)(0xffffffffu - i);
}

__kernel void lmhead_topk_f16_part(__global const ushort *logits,
                                   __global ulong *part, const int N,
                                   const int K) {
  const int lid = get_local_id(0);
  const int base = get_group_id(0) * (TOPK_LS * TOPK_EPT);
  ulong c[TOPK_EPT];
  for (int e = 0; e < TOPK_EPT; ++e) {
    const int i = base + e * TOPK_LS + lid;
    c[e] = (i < N) ? topk_key(logits[i], (uint)i) : 0ul;
  }
  __local ulong lv[TOPK_LS];
  ulong prev = 0xfffffffffffffffful;
  for (int r = 0; r < K; ++r) {
    ulong b = 0ul;
    for (int e = 0; e < TOPK_EPT; ++e)
      if (c[e] < prev && c[e] > b)
        b = c[e];
    lv[lid] = b;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = TOPK_LS / 2; off > 0; off >>= 1) {
      if (lid < off && lv[lid + off] > lv[lid])
        lv[lid] = lv[lid + off];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    prev = lv[0];
    if (lid == 0)
      part[get_group_id(0) * K + r] = prev;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
