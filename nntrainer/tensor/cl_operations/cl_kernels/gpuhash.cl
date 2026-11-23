
__kernel void gpuhash_u16(__global const ushort *src, uint n,
                          __global uint *out, uint slot, uint off,
                          uint row_stride, uint rows) {
  __local uint part[256];
  const uint lid = get_local_id(0);
  uint h = 2166136261u;
  for (uint r = 0; r < rows; ++r) {
    const uint base = off + r * row_stride;
    for (uint i = lid; i < n; i += 256u) {
      h ^= (uint)src[base + i];
      h *= 16777619u;
    }
  }
  part[lid] = h;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    uint acc = 2166136261u;
    for (uint j = 0; j < 256u; ++j) {
      acc ^= part[j];
      acc *= 16777619u;
    }
    out[slot] = acc;
  }
}
