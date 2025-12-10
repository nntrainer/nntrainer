#!/usr/bin/env python3
"""
Script to overwrite lm_head.weight in existing nntrainer binary file.
Uses file end offset for safety - lm_head is the LAST weight in the file.

Usage: python patch_lm_head.py <original_bin> <new_lm_head_bin> [--output <output_bin>]
"""

import sys
import os

# DeepseekV2-Lite config
HIDDEN_SIZE = 2048
VOCAB_SIZE = 102400
DTYPE_SIZE = 4  # float32

LM_HEAD_SIZE = VOCAB_SIZE * HIDDEN_SIZE * DTYPE_SIZE  # 838,860,800 bytes

def patch_lm_head(original_bin_path, new_lm_head_path, output_path=None):
    """Patch lm_head.weight at the END of the binary file."""
    
    if output_path is None:
        output_path = original_bin_path
    
    # Get original file size
    original_size = os.path.getsize(original_bin_path)
    print(f"Original file size: {original_size:,} bytes")
    
    # lm_head offset = file_size - lm_head_size (since it's the LAST weight)
    lm_head_offset = original_size - LM_HEAD_SIZE
    print(f"lm_head offset (from start): {lm_head_offset:,} bytes")
    print(f"lm_head size: {LM_HEAD_SIZE:,} bytes ({VOCAB_SIZE} x {HIDDEN_SIZE} x {DTYPE_SIZE})")
    
    # Verify new lm_head file size
    new_lm_head_size = os.path.getsize(new_lm_head_path)
    if new_lm_head_size != LM_HEAD_SIZE:
        print(f"ERROR: New lm_head file size ({new_lm_head_size:,}) doesn't match expected ({LM_HEAD_SIZE:,})")
        return False
    
    print(f"New lm_head file size: {new_lm_head_size:,} bytes ✓")
    
    # Read new lm_head weights
    print(f"Reading new lm_head from: {new_lm_head_path}")
    with open(new_lm_head_path, 'rb') as f:
        new_lm_head_data = f.read()
    
    # Copy original file if output is different
    if output_path != original_bin_path:
        print(f"Copying {original_bin_path} to {output_path}")
        import shutil
        shutil.copy2(original_bin_path, output_path)
    
    # Patch the file at the end
    print(f"Patching lm_head at offset {lm_head_offset:,} in: {output_path}")
    with open(output_path, 'r+b') as f:
        f.seek(lm_head_offset)
        f.write(new_lm_head_data)
    
    # Verify file size unchanged
    new_size = os.path.getsize(output_path)
    if new_size != original_size:
        print(f"WARNING: File size changed! {original_size:,} -> {new_size:,}")
    else:
        print(f"File size unchanged: {new_size:,} bytes ✓")
    
    print("Patch complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python patch_lm_head.py <original_bin> <new_lm_head_bin> [--output <output_bin>]")
        print(f"\nlm_head.weight size: {LM_HEAD_SIZE:,} bytes ({VOCAB_SIZE} x {HIDDEN_SIZE} x 4)")
        print("\nExample:")
        print("  python patch_lm_head.py nntr_deepseek_v2_lite_moe.bin new_lm_head_transposed.bin")
        sys.exit(1)
    
    original_bin = sys.argv[1]
    new_lm_head = sys.argv[2]
    output_bin = None
    
    if len(sys.argv) > 3 and sys.argv[3] == "--output":
        output_bin = sys.argv[4]
    
    success = patch_lm_head(original_bin, new_lm_head, output_bin)
    sys.exit(0 if success else 1)

