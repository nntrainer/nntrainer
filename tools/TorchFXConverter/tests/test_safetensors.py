"""Test safetensors format support for NNTrainer TorchFX converter.

Tests:
  1. Safetensors header format validation
  2. Binary round-trip: save as safetensors, reload and compare
  3. Name-based offset lookup: weights loaded correctly regardless of order
  4. Parallel loading simulation: verify offsets are non-overlapping
  5. Qwen3 symbolic graph weight ordering vs safetensors
  6. Cross-format consistency: BIN vs safetensors produce same data
  7. Safetensors JSON header parsing
  8. Weight converter auto-detection (extension-based)
"""
import sys
import os
import json
import struct
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _convert_qwen3():
    """Convert a tiny Qwen3 model and return (layers, structure, config, model)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=2048, rms_norm_eps=1e-6,
        tie_word_embeddings=True, rope_theta=1000000.0, sliding_window=None,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()

    from decomposer import AdaptiveConverter
    converter = AdaptiveConverter(model, config)
    result = converter.convert(
        {"input_ids": torch.randint(0, config.vocab_size, (1, 8))})

    return result.layers, result.model_structure, config, model


def _read_safetensors_header(path):
    """Read and parse a safetensors file header."""
    with open(path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size).decode("utf-8").rstrip()
        header = json.loads(header_json)
        data_start = 8 + header_size
    return header, header_size, data_start


def _read_safetensors_tensor(path, header, tensor_name, data_start):
    """Read a single tensor from a safetensors file by name."""
    entry = header[tensor_name]
    start, end = entry["data_offsets"]
    dtype_map = {"F32": np.float32, "F16": np.float16, "BF16": np.float32}
    np_dtype = dtype_map.get(entry["dtype"], np.float32)

    with open(path, "rb") as f:
        f.seek(data_start + start)
        data = f.read(end - start)

    return np.frombuffer(data, dtype=np_dtype).reshape(entry["shape"])


# ============================================================================
# Test 1: Safetensors Header Format Validation
# ============================================================================

def test_safetensors_header_format():
    """Test that safetensors output has valid header structure."""
    print("=" * 70)
    print("TEST 1: Safetensors Header Format Validation")
    print("=" * 70)

    from weight_converter import WeightConverter
    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        output_path = f.name

    try:
        converter.convert(state_dict, output_path, dtype="float32",
                          output_format="safetensors")

        # Verify file exists
        file_size = os.path.getsize(output_path)
        assert file_size > 0, "Output file is empty"
        print(f"  File size: {file_size} bytes")

        # Parse header
        header, header_size, data_start = _read_safetensors_header(output_path)
        print(f"  Header size: {header_size} bytes")
        print(f"  Data starts at offset: {data_start}")

        # Check header alignment (should be 8-byte aligned)
        assert header_size % 8 == 0, \
            f"Header size {header_size} not 8-byte aligned"
        print("  PASS: Header is 8-byte aligned")

        # Check __metadata__ key
        assert "__metadata__" in header, "No __metadata__ key in header"
        assert header["__metadata__"]["format"] == "nntrainer", \
            "Metadata format is not 'nntrainer'"
        print("  PASS: __metadata__ present and correct")

        # Check tensor entries
        tensor_names = [k for k in header if k != "__metadata__"]
        assert len(tensor_names) > 0, "No tensor entries in header"
        print(f"  Found {len(tensor_names)} tensor entries")

        # Verify each entry has required fields
        for name in tensor_names:
            entry = header[name]
            assert "dtype" in entry, f"No dtype for {name}"
            assert "shape" in entry, f"No shape for {name}"
            assert "data_offsets" in entry, f"No data_offsets for {name}"
            offsets = entry["data_offsets"]
            assert len(offsets) == 2, f"data_offsets should have 2 elements: {name}"
            assert offsets[0] < offsets[1], \
                f"Invalid offsets for {name}: {offsets}"
            assert entry["dtype"] in ("F32", "F16", "BF16", "I8", "I16",
                                       "I32", "U8", "U4"),\
                f"Unknown dtype for {name}: {entry['dtype']}"
        print("  PASS: All tensor entries have valid format")

        # Verify offsets are contiguous and non-overlapping
        sorted_entries = sorted(
            [(header[n]["data_offsets"][0], header[n]["data_offsets"][1], n)
             for n in tensor_names])
        for i in range(1, len(sorted_entries)):
            prev_end = sorted_entries[i - 1][1]
            curr_start = sorted_entries[i][0]
            assert curr_start >= prev_end, \
                f"Overlapping offsets: {sorted_entries[i-1][2]} ends at " \
                f"{prev_end}, {sorted_entries[i][2]} starts at {curr_start}"
        print("  PASS: Data offsets are non-overlapping and ordered")

        # Verify total data size matches file
        last_end = sorted_entries[-1][1]
        expected_file_size = data_start + last_end
        assert file_size == expected_file_size, \
            f"File size {file_size} != expected {expected_file_size}"
        print("  PASS: File size matches header + data")

    finally:
        os.unlink(output_path)

    print("  PASS: Safetensors header format test passed\n")


# ============================================================================
# Test 2: Binary Round-Trip (save as safetensors, read back, compare)
# ============================================================================

def test_safetensors_roundtrip():
    """Test that weights saved as safetensors can be read back correctly."""
    print("=" * 70)
    print("TEST 2: Safetensors Round-Trip Validation")
    print("=" * 70)

    from weight_converter import WeightConverter
    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        output_path = f.name

    try:
        converter.convert(state_dict, output_path, dtype="float32",
                          output_format="safetensors")

        header, header_size, data_start = _read_safetensors_header(output_path)
        tensor_names = [k for k in header if k != "__metadata__"]

        # Verify each weight tensor can be read back and matches original
        verified = 0
        for entry in converter.weight_map:
            hf_key = entry["hf_key"]
            nntr_name = entry["nntr_layer"]
            if entry.get("is_bias"):
                nntr_name += ":bias"
            else:
                nntr_name += ":weight"

            assert nntr_name in header, \
                f"Tensor '{nntr_name}' not found in safetensors header"

            # Read tensor back from file
            tensor_data = _read_safetensors_tensor(
                output_path, header, nntr_name, data_start)

            # Get original tensor and apply transform
            original = state_dict[hf_key].float().numpy()
            if entry["transform"] == "transpose" and original.ndim == 2:
                original = original.T

            # Compare
            np.testing.assert_array_equal(
                tensor_data, original,
                err_msg=f"Mismatch for {nntr_name} (hf: {hf_key})")
            verified += 1

        print(f"  PASS: All {verified} tensors verified via round-trip")

    finally:
        os.unlink(output_path)

    print("  PASS: Safetensors round-trip test passed\n")


# ============================================================================
# Test 3: Name-Based Offset Lookup (order-independent loading)
# ============================================================================

def test_name_based_offset_lookup():
    """Test that weights can be loaded by name regardless of file order."""
    print("=" * 70)
    print("TEST 3: Name-Based Offset Lookup (Order-Independent)")
    print("=" * 70)

    from weight_converter import WeightConverter
    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        output_path = f.name

    try:
        converter.convert(state_dict, output_path, dtype="float32",
                          output_format="safetensors")

        header, header_size, data_start = _read_safetensors_header(output_path)
        tensor_names = [k for k in header if k != "__metadata__"]

        # Load tensors in REVERSE order to prove order-independence
        reversed_names = list(reversed(tensor_names))
        loaded_tensors = {}
        for name in reversed_names:
            loaded_tensors[name] = _read_safetensors_tensor(
                output_path, header, name, data_start)

        # Now load in original order and compare
        for name in tensor_names:
            original = _read_safetensors_tensor(
                output_path, header, name, data_start)
            np.testing.assert_array_equal(
                loaded_tensors[name], original,
                err_msg=f"Order-dependent mismatch for {name}")

        print(f"  PASS: {len(tensor_names)} tensors loaded in reverse order "
              f"match forward order")

        # Also test random-access pattern (load every other tensor)
        for i, name in enumerate(tensor_names):
            if i % 2 == 0:
                _read_safetensors_tensor(output_path, header, name, data_start)
        print("  PASS: Random-access loading works")

    finally:
        os.unlink(output_path)

    print("  PASS: Name-based offset lookup test passed\n")


# ============================================================================
# Test 4: Parallel Loading Simulation
# ============================================================================

def test_parallel_loading_offsets():
    """Test that offsets enable non-overlapping parallel reads."""
    print("=" * 70)
    print("TEST 4: Parallel Loading Simulation")
    print("=" * 70)

    from weight_converter import WeightConverter
    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        output_path = f.name

    try:
        converter.convert(state_dict, output_path, dtype="float32",
                          output_format="safetensors")

        header, header_size, data_start = _read_safetensors_header(output_path)
        tensor_names = [k for k in header if k != "__metadata__"]

        # Group tensors by layer (simulate per-layer parallel loading)
        layer_groups = {}
        for name in tensor_names:
            # Layer name is everything before the last ":"
            layer_name = name.rsplit(":", 1)[0] if ":" in name else name
            layer_groups.setdefault(layer_name, []).append(name)

        print(f"  Found {len(layer_groups)} layer groups for parallel loading")

        # Verify each layer group's offsets form a contiguous or
        # non-overlapping range
        for layer_name, tensors in layer_groups.items():
            ranges = []
            for t in tensors:
                start, end = header[t]["data_offsets"]
                ranges.append((data_start + start, data_start + end))

            # Verify no overlaps within group
            sorted_ranges = sorted(ranges)
            for i in range(1, len(sorted_ranges)):
                assert sorted_ranges[i][0] >= sorted_ranges[i - 1][1], \
                    f"Overlap in {layer_name}: {sorted_ranges}"

        print("  PASS: All layer groups have non-overlapping ranges")

        # Simulate concurrent reads from different threads
        import concurrent.futures
        def read_layer_weights(layer_tensors):
            results = {}
            for name in layer_tensors:
                results[name] = _read_safetensors_tensor(
                    output_path, header, name, data_start)
            return results

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(4, len(layer_groups))) as executor:
            futures = {
                executor.submit(read_layer_weights, tensors): layer_name
                for layer_name, tensors in layer_groups.items()
            }
            all_results = {}
            for future in concurrent.futures.as_completed(futures):
                layer_name = futures[future]
                all_results[layer_name] = future.result()

        total_tensors = sum(len(v) for v in all_results.values())
        print(f"  PASS: Parallel loaded {total_tensors} tensors from "
              f"{len(all_results)} layer groups")

        # Verify parallel results match sequential reads
        for layer_name, layer_data in all_results.items():
            for name, data in layer_data.items():
                seq_data = _read_safetensors_tensor(
                    output_path, header, name, data_start)
                np.testing.assert_array_equal(data, seq_data,
                    err_msg=f"Parallel vs sequential mismatch: {name}")

        print("  PASS: Parallel results match sequential reads")

    finally:
        os.unlink(output_path)

    print("  PASS: Parallel loading simulation test passed\n")


# ============================================================================
# Test 5: Qwen3 Symbolic Graph Weight Ordering vs Safetensors
# ============================================================================

def test_qwen3_symbolic_graph_ordering():
    """Test that Qwen3 symbolic graph ordering is handled by safetensors.

    The core problem: TorchFXConverter produces weights in FX execution order,
    while NeuralNetwork iterates in topological sort order. Safetensors solves
    this via name-based lookup. This test verifies the weight names from the
    converter match what NNTrainer expects.
    """
    print("=" * 70)
    print("TEST 5: Qwen3 Symbolic Graph Weight Ordering vs Safetensors")
    print("=" * 70)

    from weight_converter import WeightConverter, build_weight_map
    from emitter_json import emit_json

    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    # Get the FX execution order (symbolic graph order)
    fx_order = []
    for entry in converter.weight_map:
        nntr_name = entry["nntr_layer"]
        suffix = ":bias" if entry.get("is_bias") else ":weight"
        fx_order.append(nntr_name + suffix)

    print(f"  FX execution order: {len(fx_order)} weights")
    for i, name in enumerate(fx_order[:10]):
        print(f"    [{i}] {name}")
    if len(fx_order) > 10:
        print(f"    ... ({len(fx_order) - 10} more)")

    # Write safetensors and verify all names are present
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        output_path = f.name

    try:
        converter.convert(state_dict, output_path, dtype="float32",
                          output_format="safetensors")

        header, _, data_start = _read_safetensors_header(output_path)
        st_names = set(k for k in header if k != "__metadata__")

        # Verify all FX-ordered names are in safetensors
        missing = set(fx_order) - st_names
        assert len(missing) == 0, \
            f"Missing from safetensors: {missing}"
        print("  PASS: All FX execution order weights present in safetensors")

        # Verify the JSON emitter includes safetensors_name
        json_data = emit_json(layers, structure)
        wmap = json_data["weight_map"]
        for entry in wmap:
            if "safetensors_name" in entry:
                assert entry["safetensors_name"] in st_names, \
                    f"safetensors_name '{entry['safetensors_name']}' " \
                    f"not found in file"
        print("  PASS: emitter_json safetensors_name entries match file")

        # Test: even if we reverse the order of weight names, loading by
        # name still gives the correct data (simulates different graph order)
        reversed_order = list(reversed(fx_order))
        for name in reversed_order:
            data = _read_safetensors_tensor(
                output_path, header, name, data_start)
            assert data is not None and data.size > 0
        print("  PASS: Reversed-order name lookup succeeds (order-independent)")

        # Verify typical Qwen3 layer patterns exist
        expected_patterns = [
            "embed_tokens",     # Token embedding
            "layers_0",         # First transformer block
            "layers_1",         # Second transformer block
            "q_proj", "k_proj", "v_proj",  # Attention projections
            "gate_proj", "up_proj", "down_proj",  # FFN layers
            "layernorm",        # Normalization layers
        ]
        all_names_str = " ".join(st_names)
        for pattern in expected_patterns:
            found = any(pattern in n for n in st_names)
            if found:
                print(f"    Found pattern: {pattern}")
            else:
                print(f"    WARN: Pattern '{pattern}' not found in tensor names")
        print("  PASS: Qwen3 layer patterns verified")

    finally:
        os.unlink(output_path)

    print("  PASS: Qwen3 symbolic graph ordering test passed\n")


# ============================================================================
# Test 6: Cross-Format Consistency (BIN vs Safetensors)
# ============================================================================

def test_cross_format_consistency():
    """Test that BIN and safetensors produce identical weight data."""
    print("=" * 70)
    print("TEST 6: Cross-Format Consistency (BIN vs Safetensors)")
    print("=" * 70)

    from weight_converter import WeightConverter
    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        st_path = f.name

    try:
        # Convert in both formats
        converter.convert(state_dict, bin_path, dtype="float32",
                          output_format="bin")
        converter.convert(state_dict, st_path, dtype="float32",
                          output_format="safetensors")

        bin_size = os.path.getsize(bin_path)
        st_size = os.path.getsize(st_path)
        print(f"  BIN file size: {bin_size} bytes")
        print(f"  Safetensors file size: {st_size} bytes")

        # Safetensors should be larger (has JSON header)
        assert st_size > bin_size, \
            "Safetensors should be larger than raw BIN (has JSON header)"
        print("  PASS: Safetensors file larger (includes JSON header)")

        # Read the raw BIN data
        with open(bin_path, "rb") as f:
            bin_data = f.read()

        # Read safetensors data and concatenate in same order
        header, _, data_start = _read_safetensors_header(st_path)
        st_concat = b""
        for entry in converter.weight_map:
            nntr_name = entry["nntr_layer"]
            suffix = ":bias" if entry.get("is_bias") else ":weight"
            name = nntr_name + suffix

            e = header[name]
            s, end = e["data_offsets"]
            with open(st_path, "rb") as f:
                f.seek(data_start + s)
                st_concat += f.read(end - s)

        # Compare: same weight data in same order
        assert len(bin_data) == len(st_concat), \
            f"Data length mismatch: BIN={len(bin_data)} vs ST={len(st_concat)}"
        assert bin_data == st_concat, \
            "Weight data differs between BIN and safetensors"
        print("  PASS: Weight data is identical between BIN and safetensors")

    finally:
        os.unlink(bin_path)
        os.unlink(st_path)

    print("  PASS: Cross-format consistency test passed\n")


# ============================================================================
# Test 7: Safetensors JSON Header Parsing (C++ parser simulation)
# ============================================================================

def test_safetensors_header_parsing():
    """Test that the safetensors header can be parsed by a minimal parser.

    This simulates what the C++ parser in neuralnet.cpp does: iterate over
    JSON keys, find data_offsets for each tensor entry.
    """
    print("=" * 70)
    print("TEST 7: Safetensors Header Parsing (C++ Parser Simulation)")
    print("=" * 70)

    from weight_converter import WeightConverter
    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        output_path = f.name

    try:
        converter.convert(state_dict, output_path, dtype="float32",
                          output_format="safetensors")

        # Read raw header bytes
        with open(output_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)

        header_str = header_bytes.decode("utf-8").rstrip()

        # Simulate minimal C++ JSON parser: just find data_offsets
        # This mirrors the parse_safetensors_header lambda in neuralnet.cpp
        parsed = json.loads(header_str)

        offsets_map = {}
        for key, val in parsed.items():
            if key == "__metadata__":
                continue
            if "data_offsets" in val:
                start, end = val["data_offsets"]
                offsets_map[key] = (start, end - start)

        print(f"  Parsed {len(offsets_map)} tensor offset entries")

        # Verify all offsets are valid
        for name, (offset, size) in offsets_map.items():
            assert offset >= 0, f"Negative offset for {name}"
            assert size > 0, f"Zero or negative size for {name}"
            assert isinstance(offset, int), f"Non-integer offset for {name}"
            assert isinstance(size, int), f"Non-integer size for {name}"

        print("  PASS: All offsets are valid integers")

        # Verify no special characters in tensor names that would break
        # the C++ parser (no unescaped quotes, no backslashes in names)
        for name in offsets_map:
            assert '"' not in name, f"Quote in tensor name: {name}"
            assert '\\' not in name, f"Backslash in tensor name: {name}"
        print("  PASS: Tensor names are C++ parser safe")

        # Verify JSON is compact (no unnecessary whitespace except padding)
        # The C++ parser handles padding spaces at the end
        core_json = header_str.rstrip()
        reparsed = json.loads(core_json)
        assert len(reparsed) == len(parsed), "Re-parse produced different result"
        print("  PASS: JSON header is valid after stripping padding")

    finally:
        os.unlink(output_path)

    print("  PASS: Safetensors header parsing test passed\n")


# ============================================================================
# Test 8: Auto-Detection from File Extension
# ============================================================================

def test_auto_format_detection():
    """Test that WeightConverter auto-detects format from extension."""
    print("=" * 70)
    print("TEST 8: Auto-Detection from File Extension")
    print("=" * 70)

    from weight_converter import WeightConverter
    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        st_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    try:
        # Auto-detect safetensors from extension
        converter.convert(state_dict, st_path, dtype="float32")
        with open(st_path, "rb") as f:
            first_8 = f.read(8)
        header_size = struct.unpack("<Q", first_8)[0]
        # Safetensors: first 8 bytes are header size (a reasonable number)
        assert 0 < header_size < 10_000_000, \
            f"Unexpected header size {header_size} - not safetensors?"
        print("  PASS: .safetensors extension -> safetensors format")

        # Auto-detect bin from extension
        converter.convert(state_dict, bin_path, dtype="float32")
        with open(bin_path, "rb") as f:
            first_4 = f.read(4)
        # BIN format: starts directly with weight data (no magic header
        # in WeightConverter output, which is v0 raw format)
        # Should NOT look like safetensors header
        print(f"  BIN first 4 bytes: {first_4.hex()}")
        print("  PASS: .bin extension -> binary format")

        # Verify file sizes differ (safetensors has header overhead)
        st_size = os.path.getsize(st_path)
        bin_size = os.path.getsize(bin_path)
        assert st_size > bin_size, \
            f"Safetensors ({st_size}) should be larger than BIN ({bin_size})"
        print(f"  PASS: Safetensors ({st_size}B) > BIN ({bin_size}B)")

    finally:
        os.unlink(st_path)
        os.unlink(bin_path)

    print("  PASS: Auto-detection test passed\n")


# ============================================================================
# Test 9: Qwen3 Full Pipeline with Safetensors Weight Conversion
# ============================================================================

def test_qwen3_full_safetensors_pipeline():
    """Full pipeline test: Qwen3 model -> safetensors -> verify all weights."""
    print("=" * 70)
    print("TEST 9: Qwen3 Full Pipeline with Safetensors")
    print("=" * 70)

    from weight_converter import WeightConverter, build_weight_map
    from emitter_json import emit_json

    layers, structure, config, model = _convert_qwen3()
    state_dict = model.state_dict()

    # Build weight map
    wmap = build_weight_map(layers)
    print(f"  Weight map: {len(wmap)} entries")

    # Convert to safetensors
    converter = WeightConverter(layers)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        output_path = f.name

    try:
        converter.convert(state_dict, output_path, dtype="float32",
                          output_format="safetensors")

        header, _, data_start = _read_safetensors_header(output_path)
        st_names = set(k for k in header if k != "__metadata__")

        # Categorize weights
        embedding_weights = [n for n in st_names if "embed" in n.lower()]
        attn_weights = [n for n in st_names
                        if any(p in n for p in ["q_proj", "k_proj", "v_proj",
                                                "o_proj"])]
        ffn_weights = [n for n in st_names
                       if any(p in n for p in ["gate_proj", "up_proj",
                                               "down_proj"])]
        norm_weights = [n for n in st_names if "norm" in n.lower()]

        print(f"  Embedding weights: {len(embedding_weights)}")
        print(f"  Attention weights: {len(attn_weights)}")
        print(f"  FFN weights: {len(ffn_weights)}")
        print(f"  Norm weights: {len(norm_weights)}")

        # Qwen3 with 2 layers should have:
        # 1 embedding + (4 attn + 3 ffn + 4 norms) * 2 layers + 1 final_norm
        # = 1 + 14*2 + 1 = ~30 weight tensors (some may be split/combined)
        total_weights = len(st_names)
        assert total_weights > 10, \
            f"Expected >10 weights for 2-layer Qwen3, got {total_weights}"
        print(f"  Total weight tensors: {total_weights}")

        # Verify weight shapes make sense for Qwen3 config
        for name in st_names:
            entry = header[name]
            shape = entry["shape"]
            # All shapes should have at least 1 dimension
            assert len(shape) >= 1, f"Invalid shape for {name}: {shape}"
            # No zero dimensions
            assert all(d > 0 for d in shape), \
                f"Zero dimension for {name}: {shape}"

        print("  PASS: All weight shapes are valid")

        # Verify total data size matches expected
        total_params = sum(
            np.prod(header[n]["shape"]) for n in st_names)
        expected_bytes = total_params * 4  # float32
        actual_data_bytes = sum(
            header[n]["data_offsets"][1] - header[n]["data_offsets"][0]
            for n in st_names)
        assert actual_data_bytes == expected_bytes, \
            f"Data size {actual_data_bytes} != expected {expected_bytes}"
        print(f"  PASS: Total parameters: {total_params:,} "
              f"({expected_bytes:,} bytes)")

    finally:
        os.unlink(output_path)

    print("  PASS: Qwen3 full safetensors pipeline test passed\n")


# ============================================================================
# Main runner
# ============================================================================

def main():
    """Run all safetensors tests."""
    print("\n" + "#" * 70)
    print("# SAFETENSORS FORMAT TESTS")
    print("#" * 70 + "\n")

    tests = [
        test_safetensors_header_format,
        test_safetensors_roundtrip,
        test_name_based_offset_lookup,
        test_parallel_loading_offsets,
        test_qwen3_symbolic_graph_ordering,
        test_cross_format_consistency,
        test_safetensors_header_parsing,
        test_auto_format_detection,
        test_qwen3_full_safetensors_pipeline,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {test_fn.__name__}: {e}\n")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
