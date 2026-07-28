#!/usr/bin/env python3
"""
Convert a FastViT FP32 safetensors file to Q8_0 format.

For each conv weight tensor that is eligible (out_ch % 32 == 0 and
(in_ch/groups * k * k) % 32 == 0), the FP32 data is quantized to Q8_0
(block_q8_0: fp16 scale + 32 int8 per block of 32 elements) and then
repacked to the block_q8_0x4 layout (4 weight columns interleaved per
super-block) that the FP16 q8_0×q8_0 indirect-conv kernel consumes.

Non-eligible weights (bias, depthwise convs, small channels) are kept
as FP32.

The output safetensors uses nntrainer's extension fields:
  - dtype: "U8", shape: [byte_size]  (opaque byte blob)
  - nntr_dtype: "Q8_0"
  - nntr_shape: [1, 1, K=CRS, N=out_ch]  (logical matmul shape)

Usage:
  python3 convert_to_q8_0.py input.safetensors output.safetensors
"""

import json
import struct
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def quantize_q8_0_row(src: np.ndarray) -> bytes:
    """Quantize one row of 32 float32 values to a single block_q8_0.

    block_q8_0 = { uint16_t d (fp16); int8_t qs[32] }
    d = max(|x|) / 127, qs = round(x / d) clipped to [-128, 127]
    """
    assert len(src) == 32
    amax = float(np.max(np.abs(src)))
    d = amax / 127.0 if amax > 0 else 0.0
    # fp16 scale
    d_fp16 = np.array([d], dtype=np.float16)
    d_bytes = d_fp16.tobytes()  # 2 bytes
    if d > 0:
        inv_d = 1.0 / d
        qs = np.clip(np.round(src * inv_d), -128, 127).astype(np.int8)
    else:
        qs = np.zeros(32, dtype=np.int8)
    return d_bytes + qs.tobytes()  # 2 + 32 = 34 bytes


def quantize_q8_0(weight: np.ndarray, N: int, K: int) -> bytes:
    """Quantize [N, K] FP32 weight to plain block_q8_0 stream.

    weight must be row-major [N rows, K cols], K must be divisible by 32.
    Output: N * (K/32) * 34 bytes.
    """
    assert K % 32 == 0, f"K={K} not divisible by 32"
    assert weight.shape == (N, K), f"weight shape {weight.shape} != ({N}, {K})"
    nb = K // 32
    out = bytearray()
    for n in range(N):
        row = weight[n]
        for b in range(nb):
            block = row[b * 32 : (b + 1) * 32]
            out.extend(quantize_q8_0_row(block))
    return bytes(out)


def repack_q8_0(plain: bytes, N: int, K: int) -> bytes:
    """Repack plain block_q8_0 to block_q8_0x4 layout.

    Mirrors C++ repack_q8_0 in conv_indirect.h:
      - plain: [N][nb] block_q8_0, each 34 bytes
      - out:   [N/4][nb] block_q8_0x4, each 136 bytes

    block_q8_0x4 = { uint16_t d[4]; int8_t qs[128] }
    where qs[32*sub + 8*row + lane], sub=0..3, row=0..3, lane=0..7
    """
    assert N % 4 == 0, f"N={N} not divisible by 4 (implied by N%32==0)"
    nb = K // 32
    block_size = 34  # sizeof(block_q8_0)

    # Parse plain blocks into numpy arrays
    plain_arr = np.frombuffer(plain, dtype=np.uint8).reshape(N, nb, block_size)
    # Extract d (fp16) and qs (int8) for each block
    d_plain = plain_arr[:, :, :2].copy()  # [N, nb, 2]
    qs_plain = plain_arr[:, :, 2:].copy()  # [N, nb, 32] as uint8

    out = bytearray()
    for sc in range(N // 4):
        for j in range(nb):
            # d[4]: scales from 4 rows
            d_block = bytearray()
            for r in range(4):
                d_block.extend(d_plain[sc * 4 + r, j])
            # qs[128]: 4 subs * 8 bytes per row * 4 rows
            qs_block = bytearray(128)
            for r in range(4):
                for sub in range(4):
                    src_offset = sub * 8
                    dst_offset = 32 * sub + 8 * r
                    chunk = qs_plain[sc * 4 + r, j, src_offset : src_offset + 8]
                    qs_block[dst_offset : dst_offset + 8] = bytes(chunk)
            out.extend(d_block)
            out.extend(qs_block)
    return bytes(out)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.safetensors output.safetensors")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    # Read all tensors from the input safetensors
    tensors = {}
    with safe_open(in_path, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    print(f"Loaded {len(tensors)} tensors from {in_path}")

    # Process each tensor
    out_tensors = {}
    out_meta = {}
    quantized_count = 0
    kept_fp32_count = 0

    for name, weight in tensors.items():
        if weight.dtype != np.float32:
            # Non-float32 (e.g. int tensors): keep as-is
            out_tensors[name] = weight
            kept_fp32_count += 1
            continue

        # Determine if this is a conv weight eligible for Q8_0
        # Conv weights are 4D: [out_ch, in_ch, kh, kw]
        if weight.ndim != 4:
            # Bias or other 1D/2D tensor: keep FP32
            out_tensors[name] = weight
            kept_fp32_count += 1
            continue

        out_ch, in_ch, kh, kw = weight.shape
        # For grouped conv, effective in_ch per group = in_ch / groups
        # We don't know groups here, but the graph builder checks
        # in_ch/groups * k * k. For standard conv (groups=1), it's in_ch * k * k.
        # For depthwise (groups=in_ch), it's k * k.
        # We quantize if out_ch % 32 == 0 AND (in_ch * kh * kw) % 32 == 0.
        # The runtime will skip Q8_0 for depthwise convs (their effective
        # width k*k won't be divisible by 32), so we only quantize standard
        # convs here. But we can't distinguish groups from the weight shape
        # alone. The safe approach: quantize if out_ch%32==0 and
        # (in_ch*kh*kw)%32==0. Depthwise convs have in_ch==out_ch and
        # in_ch*kh*kw is usually divisible by 32, so they'd be quantized
        # here too. But the runtime graph builder will NOT set weight_dtype=Q8_0
        # for depthwise convs (the divisibility check uses in_ch/groups*k*k),
        # so the runtime tensor will be FP32, and the file will have Q8_0 data
        # with nntr_dtype set, which will be skipped by the loader.
        #
        # Actually, this is a problem: if the file has Q8_0 data but the
        # runtime expects FP32, the loader will skip it (nntr_dtype set)
        # and the raw read will have already written Q8_0 bytes into the
        # FP32 tensor (garbage).
        #
        # Solution: only quantize weights where the runtime will also use Q8_0.
        # The runtime checks: out_ch%32==0 AND (in_ch/groups*k*k)%32==0.
        # For standard convs (groups=1): in_ch*kh*kw must be %32==0.
        # For depthwise (groups=in_ch): kh*kw must be %32==0 (never true for
        # 3x3=9 or 7x7=49).
        # So we should only quantize if out_ch%32==0 AND (in_ch*kh*kw)%32==0.
        # But depthwise convs with in_ch==out_ch and in_ch*kh*kw%32==0 would
        # also be quantized here, yet the runtime won't use Q8_0 for them.
        #
        # The key insight: depthwise convs have groups=in_ch, so the runtime
        # check (in_ch/groups * k * k) = (1 * k * k) = k*k, which is NOT %32==0.
        # So the runtime won't set Q8_0 for depthwise. But we'd quantize it
        # here because in_ch*kh*kw IS %32==0 (e.g. 64*7*7=3136, 3136%32==0).
        #
        # To avoid this, we need to know which convs are depthwise.
        # Depthwise convs have in_ch == out_ch (for FastViT).
        # So: skip quantization if in_ch == out_ch (depthwise pattern).
        #
        # Actually, stem1 has in_ch=64, out_ch=64, groups=64 (depthwise).
        # But stem0 has in_ch=3, out_ch=64, groups=1 (standard).
        # And conv1x1 has in_ch==out_ch possible but groups=1.
        #
        # Better approach: match the exact graph builder logic.
        # The graph builder uses groups parameter. We can infer groups from
        # the weight shape: for depthwise, weight is [ch, 1, kh, kw].
        # For standard conv, weight is [out_ch, in_ch, kh, kw] with in_ch > 1.
        # For 1x1 conv, weight is [out_ch, in_ch, 1, 1].
        #
        # So: if in_ch == 1, it's depthwise (groups=out_ch).
        # If in_ch > 1, it's standard (groups=1) or grouped.
        # For FastViT, all grouped convs are depthwise (in_ch=1 per group).

        CRS = in_ch * kh * kw
        is_depthwise = (in_ch == 1)  # depthwise: [ch, 1, kh, kw]

        if (not is_depthwise and out_ch % 32 == 0 and CRS % 32 == 0):
            # Eligible for Q8_0 quantization
            # Reshape to [N=out_ch, K=CRS] row-major
            weight_2d = weight.reshape(out_ch, CRS)
            plain = quantize_q8_0(weight_2d, out_ch, CRS)
            repacked = repack_q8_0(plain, out_ch, CRS)

            # Store as U8 blob with nntr_dtype/nntr_shape extensions
            out_tensors[name] = np.frombuffer(repacked, dtype=np.uint8)
            out_meta[name] = {
                "nntr_dtype": "Q8_0",
                "nntr_shape": [1, 1, CRS, out_ch],
            }
            quantized_count += 1
            print(f"  [Q8_0] {name}: [{out_ch}, {in_ch}, {kh}, {kw}] -> "
                  f"K={CRS}, N={out_ch}, {len(repacked)} bytes")
        else:
            out_tensors[name] = weight
            kept_fp32_count += 1

    print(f"\nQuantized: {quantized_count}, Kept FP32: {kept_fp32_count}")

    # Build the safetensors file with custom header
    # safetensors library doesn't support nntr_dtype/nntr_shape natively,
    # so we build the file manually.
    header = {"__metadata__": {"format": "nntrainer"}}

    # Serialize all tensors and build header entries
    data_buf = bytearray()
    entries = []

    for name, tensor in out_tensors.items():
        tensor_bytes = tensor.tobytes()
        offset_start = len(data_buf)
        data_buf.extend(tensor_bytes)
        offset_end = len(data_buf)

        if name in out_meta:
            meta = out_meta[name]
            entries.append({
                "name": name,
                "dtype": "U8",
                "shape": [len(tensor_bytes)],
                "nntr_dtype": meta["nntr_dtype"],
                "nntr_shape": meta["nntr_shape"],
                "data_offsets": [offset_start, offset_end],
            })
        else:
            # FP32 tensor
            if tensor.dtype == np.float32:
                dtype_str = "F32"
            elif tensor.dtype == np.int64:
                dtype_str = "I64"
            else:
                dtype_str = str(tensor.dtype)
            entries.append({
                "name": name,
                "dtype": dtype_str,
                "shape": list(tensor.shape),
                "data_offsets": [offset_start, offset_end],
            })

    # Build header JSON
    for e in entries:
        header[e["name"]] = {
            "dtype": e["dtype"],
            "shape": e["shape"],
        }
        if "nntr_dtype" in e:
            header[e["name"]]["nntr_dtype"] = e["nntr_dtype"]
        if "nntr_shape" in e:
            header[e["name"]]["nntr_shape"] = e["nntr_shape"]
        header[e["name"]]["data_offsets"] = e["data_offsets"]

    header_json = json.dumps(header)
    # Pad to 8-byte alignment
    header_len = len(header_json)
    padding = (8 - header_len % 8) % 8
    header_json_padded = header_json + " " * padding

    # Write file: [u64 header_len][header_json][data]
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json_padded)))
        f.write(header_json_padded.encode("utf-8"))
        f.write(data_buf)

    print(f"\nWrote {out_path} ({len(data_buf)} bytes data, "
          f"{len(header_json_padded)} bytes header)")


if __name__ == "__main__":
    main()
