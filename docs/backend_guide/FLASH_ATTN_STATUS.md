# Flash Attention Implementation Status

**Update (2026-08-03, later same day):** The "current issue" below (garbled
output, dlopen/dlsym never firing) was a stale snapshot from mid-debugging.
Flash-attn dispatches correctly and produces correct output. The actual
end-to-end problem turned out to be a performance crossover, not a
correctness bug - see `HEXAGON_NPU_OBSERVATION_LOG.md` §31 for the full
hands-on profiling writeup. Summary: below ~150-160 tokens the per-layer DSP
round trip (measured 1.5-1.8 ms/layer, 95-96% of flash-attn's own cost) isn't
amortized by the O(n^2) CPU attention cost it replaces, so short prompts
regressed; above ~200 tokens it wins decisively (+48% over GEMM-only offload
at 410 tokens). Fixed with a `step_size`-based gate in
`should_use_flash_attn()` (`Applications/CausalLM/layers/mha_core.cpp`),
env-overridable via `NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS` (default 160). The
two prerequisites below (`NNTR_USE_HEXAGON_CDSP=1` for the rpcmem-backed KV
cache, `NNTR_HEXAGON_FLASH_ATTN=1` to enable the path) both need to be set -
missing the former makes every flash-attn call fail closed and silently fall
back to CPU (§31 finding 2), which can look identical to "flash-attn is slow"
if you're not watching the bridge's stderr.

## Summary (original, kept for history)

This document describes the current status of the flash attention implementation for Hexagon DSP offloading in the nntrainer CausalLM application.

## Changes Made

### 1. Added `nntr_htp_bridge_flash_attn` function
**File:** `../ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp`

Added a new C-ABI function that:
- Accepts Q, K, V, mask, and output buffers
- Supports FP16 and FP32 for Q and output
- K/V are always FP16 (from KVCacheManager's rpcmem pool)
- Builds tensor descriptors for the DSP kernel
- Dispatches `GGML_OP_FLASH_ATTN_EXT` to the cDSP
- Includes output permutation to convert DSP's `[head_dim, n_heads, n_tokens]` layout to caller's expected `[n_tokens, n_heads, head_dim]` layout

**Key implementation details:**
- Output tensor descriptor uses layout `[head_dim, n_heads, n_tokens]` to match DSP kernel write pattern
- After DSP execution, a permutation loop reorders output from DSP layout to caller layout
- Debug logging added with `GGML_LOG_INFO` showing parameters when called
- Call counter `nntr_htp_bridge_get_flash_attn_call_count()` for debugging

### 2. verify_flash_attn Test
**Location:** `../ggml-hexagon/` (test binary deployed to device)

The test compares DSP output against CPU reference:
- **Status: PASSING** for all test modes (small, full, chunked)
- max_abs_err ≈ 0.040606, max_rel_err ≈ 10.9896
- This confirms the DSP kernel and permutation logic are correct in isolation

## Current Issue

### Problem: CausalLM Output is Garbled

When running CausalLM with `NNTR_HEXAGON_FLASH_ATTN=1`:
```
adb shell "cd /data/local/tmp/nntrainer/causallm && export NNTR_HEXAGON_FLASH_ATTN=1 && ./nntrainer_causallm models/qwen3-0.6b"
```

**Symptoms:**
1. Output text is garbled (nonsense characters)
2. Performance is identical to CPU path (~69ms prefill, ~1354ms generation for 128 tokens)
3. No debug log message "nntr-htp-bridge: flash_attn called: ..." appears

**Evidence that flash_attn is NOT being called:**
- The debug log inside `nntr_htp_bridge_flash_attn()` never prints
- Symbol `nntr_htp_bridge_flash_attn` IS exported in libggml-hexagon.so (confirmed via llvm-nm)
- The function pointer resolution in mha_core.cpp uses dlopen/dlsym

### Root Cause Analysis

The issue is that `get_flash_attn_bridge()` in `Applications/CausalLM/layers/mha_core.cpp` is returning `nullptr`, causing `should_use_flash_attn()` to return `false`, which falls back to CPU path.

**Possible reasons:**
1. **dlopen failure**: `libggml-hexagon.so` may not be found at runtime
2. **dlsym failure**: Symbol name mismatch or not exported properly
3. **Library path issue**: The LD_LIBRARY_PATH may not include the directory containing libggml-hexagon.so

**What works:**
- verify_flash_attn test PASSES - this uses the same library and function
- The GEMM bridge functions work (model inference runs)

**What doesn't work:**
- CausalLM doesn't seem to resolve the flash_attn function pointer
- Output is garbled even when flash_attn should be used

## Files Modified

1. `../ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp`
   - Added `nntr_htp_bridge_flash_attn()` function
   - Added `g_flash_attn_calls` atomic counter
   - Added `nntr_htp_bridge_get_flash_attn_call_count()` for debugging
   - Added debug logging at function entry

2. `Applications/CausalLM/layers/mha_core.cpp`
   - No modifications needed - already has flash_attn integration code
   - `should_use_flash_attn()` checks: env flag, is_prefill, head_dim==128, function pointer

## Next Steps for Debugging

1. **Rebuild Android CausalLM binary**: The local build is x86-64, not Android arm64. Need to cross-compile or build on Android target environment.
2. **Check dlopen/dlsym errors**: Once Android binary is rebuilt with debug logging, capture stderr to see if dlopen or dlsym is failing
3. **Verify library loading**: Check if libggml-hexagon.so is already loaded (might need RTLD_NOLOAD check)
4. **Check mha_core.cpp conditions**: Verify `should_use_flash_attn()` conditions are met:
   - `NNTR_HEXAGON_FLASH_ATTN=1` env var set
   - `is_prefill` is true (step_size > 1)
   - `head_dim == 128` (Qwen3-0.6B has head_dim=128 ✓)
   - Function pointer is non-null

## Build Notes

- Local build produces x86-64 binary (not deployable to Android device)
- Device has pre-built `nntrainer_causallm` from 2026-08-03 13:09 (no debug logging)
- To debug: need Android arm64 cross-compile or build on device/CI environment
- verify_flash_attn test binary IS arm64 and works correctly (proves DSP path is functional)

## Test Commands

```bash
# Build
cd ../ggml-hexagon/build-hexagon-android
ninja -j4 ggml-hexagon

# Deploy
adb push bin/libggml-hexagon.so /data/local/tmp/nntrainer/causallm/

# Test verify_flash_attn (should PASS)
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=/system/lib64:. ./verify_flash_attn"

# Test CausalLM (currently produces garbled output)
adb shell "cd /data/local/tmp/nntrainer/causallm && export NNTR_HEXAGON_FLASH_ATTN=1 && LD_LIBRARY_PATH=/system/lib64:. ./nntrainer_causallm models/qwen3-0.6b"

# Check if flash_attn symbol exists
llvm-nm bin/libggml-hexagon.so | grep flash_attn
# Should show: T nntr_htp_bridge_flash_attn
```

## Model Configuration (Qwen3-0.6B)

From `Applications/CausalLM/res/qwen3/qwen3-0.6b/config.json`:
- `head_dim`: 128 ✓ (matches flash_attn requirement)
- `hidden_size`: 1024
- `num_attention_heads`: 16
- `num_key_value_heads`: 8
- GQA_SIZE = 16/8 = 2
