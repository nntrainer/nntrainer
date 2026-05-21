#!/usr/bin/env python3
"""
TorchFX-to-NNTrainer Converter CLI.

Converts HuggingFace models to NNTrainer format:
  - C++ model construction code
  - INI configuration file
  - JSON model config + weight map
  - Binary weight file (.bin)

Usage:
  python converter.py --model Qwen/Qwen3-0.6B --output ./output/
  python converter.py --model bert-base-uncased --output ./output/ --format ini
  python converter.py --model google/mt5-small --output ./output/ --format all

Phase 6 of the TorchFX converter pipeline (DESIGN.md).
"""

import argparse
import os
import sys
import json

import torch


def convert_model(model_name_or_path, output_dir, formats=None,
                  batch_size=1, seq_len=8, dtype="float32",
                  convert_weights=False, verbose=True,
                  model_name=None, plugin_config=None,
                  external_kv_cache=False,
                  inherit_transformer=False):
    """Run the full conversion pipeline.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        output_dir: Output directory for generated files.
        formats: List of output formats ("cpp", "ini", "json", "weights").
                 None means all except weights.
        batch_size: Batch size for INI config.
        seq_len: Sequence length for tracing.
        dtype: Weight dtype ("float32" or "float16").
        convert_weights: Whether to convert and save weights.
        verbose: Print progress messages.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from decomposer import AdaptiveConverter
    from emitter_cpp import emit_cpp, emit_cpp_header, emit_cpp_source, get_output_filenames
    from emitter_ini import emit_ini
    from emitter_json import emit_json_string
    from weight_converter import WeightConverter

    if formats is None:
        formats = ["cpp", "ini", "json"]
    if convert_weights and "weights" not in formats:
        formats.append("weights")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load model and config
    if verbose:
        print(f"Loading model: {model_name_or_path}")

    # Try standard AutoConfig first; fall back for custom model types
    # (e.g. model_type values not registered in HuggingFace transformers).
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
    except ValueError:
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
            config = argparse.Namespace(**config_dict)
            if verbose:
                print(f"  Loaded custom config (model_type="
                      f"{getattr(config, 'model_type', 'unknown')})")
        else:
            raise

    model_type = getattr(config, "model_type", "")

    # Choose model class based on architecture
    causal_types = {
        "qwen3", "qwen2", "llama", "mistral", "gpt2", "gpt_neo", "gpt_neox",
        "phi", "gemma", "gemma2", "gemma3_text", "gemma3n_text", "starcoder2", "codegen",
        "lfm2", "granitemoehybrid",
        "mamba", "mamba2",
    }
    encoder_decoder_types = {
        "t5", "mt5", "bart", "mbart", "pegasus", "marian",
    }

    # Detect if this is an embedding/feature-extraction model
    architectures = getattr(config, "architectures", []) or []
    is_embedding_model = any(
        not arch.endswith(("ForCausalLM", "ForConditionalGeneration",
                           "ForSeq2SeqLM", "ForMaskedLM",
                           "ForSequenceClassification",
                           "ForTokenClassification"))
        for arch in architectures
    ) if architectures else False

    is_causal = model_type in causal_types and not is_embedding_model
    is_encoder_decoder = model_type in encoder_decoder_types

    # Check for custom model types (handled by custom_models.py)
    from custom_models import CUSTOM_LOADERS
    is_custom = model_type in CUSTOM_LOADERS

    if is_custom:
        from custom_models import load_custom_model
        model, config, input_kwargs = load_custom_model(
            model_type, model_name_or_path, config, seq_len, verbose)
    elif is_causal:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32)
        model.eval()
    else:
        model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32)
        model.eval()

    if verbose:
        print(f"Model type: {model_type}, Parameters: "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Step 2: Prepare trace inputs (for non-custom models)
    if not is_custom:
        vocab_size = getattr(config, "vocab_size", 30000)
        input_kwargs = {"input_ids": torch.randint(0, vocab_size, (1, seq_len))}
        if not is_causal and not is_encoder_decoder:
            input_kwargs["attention_mask"] = torch.ones(1, seq_len, dtype=torch.long)
        if is_encoder_decoder:
            input_kwargs["decoder_input_ids"] = torch.randint(
                0, min(vocab_size, 1000), (1, max(1, seq_len // 2)))

    # Step 2.5: Load plugin registry (custom layer mappings)
    plugin_registry = None
    if plugin_config:
        from plugin_registry import PluginRegistry
        plugin_registry = PluginRegistry.from_config(plugin_config)
        if verbose:
            print(f"Loaded {len(plugin_registry)} custom layer plugin(s) "
                  f"from {plugin_config}")

    # Step 3: Run conversion pipeline
    if verbose:
        print("Running conversion pipeline...")

    converter = AdaptiveConverter(model, config,
                                 plugin_registry=plugin_registry)
    result = converter.convert(input_kwargs)

    layers = result.layers
    structure = result.model_structure
    if structure:
        structure.external_kv_cache = external_kv_cache

    if verbose:
        print(f"Converted: {len(layers)} layers, "
              f"{len(result.lazy_chains)} LazyTensor chains")
        if structure:
            print(f"Architecture: {structure.arch_type} ({structure.model_type})")
            print(f"Blocks: {structure.num_layers}")

    # Step 4: Emit outputs
    outputs = {}

    # Derive filenames
    effective_name = model_name or model_name_or_path
    filenames = get_output_filenames(
        structure.model_type if structure else model_type,
        structure.arch_type if structure else "decoder_only",
        model_name=effective_name,
    )

    if "cpp" in formats:
        header_code = emit_cpp_header(
            layers, structure, model_name=effective_name,
            inherit_transformer=inherit_transformer)
        source_code = emit_cpp_source(
            layers, structure, model_name=effective_name,
            inherit_transformer=inherit_transformer)
        header_path = os.path.join(output_dir, filenames["header"])
        source_path = os.path.join(output_dir, filenames["source"])
        with open(header_path, "w") as f:
            f.write(header_code)
        with open(source_path, "w") as f:
            f.write(source_code)
        outputs["cpp_header"] = header_path
        outputs["cpp_source"] = source_path
        if verbose:
            print(f"  C++ header: {header_path} ({len(header_code)} bytes)")
            print(f"  C++ source: {source_path} ({len(source_code)} bytes)")

    if "ini" in formats:
        ini_text = emit_ini(layers, structure, batch_size=batch_size,
                            mode="structured")
        ini_path = os.path.join(output_dir, filenames["ini"])
        with open(ini_path, "w") as f:
            f.write(ini_text)
        outputs["ini"] = ini_path
        if verbose:
            print(f"  INI config: {ini_path} ({len(ini_text)} bytes)")

    if "json" in formats:
        json_str = emit_json_string(layers, structure, indent=2)
        json_path = os.path.join(output_dir, filenames["json"])
        with open(json_path, "w") as f:
            f.write(json_str)
        outputs["json"] = json_path
        if verbose:
            print(f"  JSON config: {json_path} ({len(json_str)} bytes)")

    if "weights" in formats:
        wc = WeightConverter(layers)
        state_dict = model.state_dict()

        # Save in safetensors format (primary, name-based parallel loading)
        st_path = os.path.join(output_dir, "model.safetensors")
        wc.convert(state_dict, st_path, dtype=dtype,
                   output_format="safetensors")
        outputs["weights_safetensors"] = st_path
        if verbose:
            size_mb = os.path.getsize(st_path) / 1024 / 1024
            print(f"  Weights (safetensors): {st_path} ({size_mb:.1f} MB)")

        # Also save in legacy BIN format for backward compatibility
        bin_path = os.path.join(output_dir, "model.bin")
        wc.convert(state_dict, bin_path, dtype=dtype, output_format="bin")
        outputs["weights"] = bin_path
        if verbose:
            size_mb = os.path.getsize(bin_path) / 1024 / 1024
            print(f"  Weights (bin): {bin_path} ({size_mb:.1f} MB)")

        # Also save standalone weight conversion script
        script = wc.generate_script()
        script_path = os.path.join(output_dir, "convert_weights.py")
        with open(script_path, "w") as f:
            f.write(script)
        outputs["weight_script"] = script_path
        if verbose:
            print(f"  Weight script: {script_path}")

    # Emit visualizer-ready JSON (conversion_result.json) with FX graph,
    # node mapping, and generated C++/INI sources for the VS Code plugin
    from vscode_bridge import (serialize_fx_graph, serialize_layer,
                               serialize_structure, build_node_mapping)

    viz_data = {
        "nntrainerLayers": [serialize_layer(l) for l in layers],
        "fxGraph": serialize_fx_graph(result.graph),
        "modelStructure": serialize_structure(structure),
        "nodeMapping": build_node_mapping(layers, result.graph),
        "unsupportedOps": [l.name for l in result.unsupported_ops],
        "unknownLayers": [l.name for l in result.unknown_layers],
        "decomposedModules": list(result.decomposed_module_types),
        "inheritTransformer": inherit_transformer,
    }

    # Include generated sources so visualizer can show them
    if "cpp" in formats:
        viz_data["cppHeader"] = emit_cpp_header(
            layers, structure, model_name=effective_name,
            inherit_transformer=inherit_transformer)
        viz_data["cppSource"] = emit_cpp_source(
            layers, structure, model_name=effective_name,
            inherit_transformer=inherit_transformer)
    if "ini" in formats:
        viz_data["iniConfig"] = emit_ini(
            layers, structure, batch_size=batch_size, mode="structured")

    # Weight map for cross-referencing HF keys to NNTrainer layers
    weight_map = []
    for layer in layers:
        if layer.has_weight and layer.weight_hf_key:
            weight_map.append({
                "layer_name": layer.name,
                "layer_type": layer.layer_type,
                "weight_key": layer.weight_hf_key,
                "transpose_weight": layer.transpose_weight,
            })
    viz_data["weightMap"] = weight_map

    viz_path = os.path.join(output_dir, "conversion_result.json")
    with open(viz_path, "w") as f:
        json.dump(viz_data, f, indent=2, default=str)
    outputs["visualizer_json"] = viz_path
    if verbose:
        print(f"  Visualizer JSON: {viz_path}")

    if verbose:
        print(f"\nConversion complete! Output: {output_dir}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to NNTrainer format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model Qwen/Qwen3-0.6B --output ./qwen3/
  %(prog)s --model bert-base-uncased --output ./bert/ --format ini json
  %(prog)s --model google/mt5-small --output ./mt5/ --format all
  %(prog)s --model ./local_model/ --output ./out/ --weights --dtype float16
""")

    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--format", nargs="+",
                        choices=["cpp", "ini", "json", "all"],
                        default=["all"],
                        help="Output formats (default: all)")
    parser.add_argument("--weights", action="store_true",
                        help="Convert and save model weights")
    parser.add_argument("--dtype", choices=["float32", "float16"],
                        default="float32",
                        help="Weight dtype (default: float32)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for INI config (default: 1)")
    parser.add_argument("--seq-len", type=int, default=8,
                        help="Sequence length for tracing (default: 8)")
    parser.add_argument("--model-name", default=None,
                        help="Override output file naming (default: derived "
                             "from --model). e.g. --model-name KaLM-embedding")
    parser.add_argument("--plugin-config",
                        help="JSON/YAML config file for custom layer plugins")
    parser.add_argument("--external-kv-cache", action="store_true",
                        help="Generate code with external KV cache buffers "
                             "(owned by the generated class instead of "
                             "internal MHA tensors)")
    parser.add_argument("--inherit-transformer", action="store_true",
                        help="Generate C++ class inheriting from "
                             "CausalLM/Transformer base class "
                             "(Applications/CausalLM)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress messages")

    args = parser.parse_args()

    formats = args.format
    if "all" in formats:
        formats = ["cpp", "ini", "json"]

    convert_model(
        model_name_or_path=args.model,
        output_dir=args.output,
        formats=formats,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=args.dtype,
        convert_weights=args.weights,
        verbose=not args.quiet,
        model_name=args.model_name,
        plugin_config=args.plugin_config,
        external_kv_cache=args.external_kv_cache,
        inherit_transformer=args.inherit_transformer,
    )


if __name__ == "__main__":
    main()
