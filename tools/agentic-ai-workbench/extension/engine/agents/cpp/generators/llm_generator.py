"""
LLM-driven C++ code generation from nntrainer_graph_ir.

This generator uses Cline (LLM) to write C++ code from scratch,
using the nntrainer_graph_ir as the source of truth for which
layers to use and how to configure them.
"""

import json
import math
import os
import re
from typing import Dict, List, Any, Optional

from ..harness.base import CppGeneratorHarness, GeneratedFiles
from ..harness.layer_catalog import LAYER_CATALOG, get_available_layers
from ...events import bus


def _values_match(candidate, value) -> bool:
    """Fuzzy compare a constant's value against a graph_ir attribute value.

    Exact equality is too brittle here: the two sides reach us by different
    routes. `epsilon` arrives as a JSON float (1e-06) while NORM_EPS comes
    from the raw config; `rope_theta` is int()-ed into the constants table
    (10000) but the graph_ir keeps it as 10000.0; ints may arrive as the
    strings "2048". A missed match silently degrades to emitting the literal
    number, which is exactly the bug this mapping exists to prevent -- so
    compare numerically with tolerance, and fall back to a normalised
    string compare.
    """
    try:
        a, b = float(candidate), float(value)
        # rel_tol covers dims and rope_theta; abs_tol covers epsilon-scale
        # values where relative error is meaningless near zero.
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)
    except (TypeError, ValueError):
        pass
    return str(candidate).strip().lower() == str(value).strip().lower()


class LLMGenerator(CppGeneratorHarness):
    """
    LLM-driven C++ code generation.
    
    FLOW:
    1. Read nntrainer_graph_ir from state
    2. Build detailed prompt with layer catalog and graph structure
    3. Call LLM to generate C++ code
    4. Parse and validate generated code
    5. Return GeneratedFiles
    """
    
    def __init__(self, nntrainer_root: str, cline_ums_token: Optional[str] = None):
        self.nntrainer_root = nntrainer_root
        self.layers_dir = os.path.join(nntrainer_root, "Applications/CausalLM/layers")
        self.models_dir = os.path.join(nntrainer_root, "Applications/CausalLM/models")
        self.cline_ums_token = cline_ums_token
    
    def get_generator_id(self) -> str:
        return "llm"
    
    def get_capabilities(self) -> List[str]:
        return ["causallm_component", "custom_decoder", "sliding_window", "gqa", "moe"]
    
    def generate(self, state: dict) -> GeneratedFiles:
        """
        Generate C++ code using LLM from nntrainer_graph_ir.
        """
        architecture = state.get("architecture", "")
        hf_config = state.get("hf_config", {})
        graph_ir = state.get("nntrainer_graph_ir") or state.get("graph_ir")
        
        if not graph_ir:
            return GeneratedFiles(
                header="", source="",
                header_filename="", source_filename="",
                architecture=architecture,
                transformer_class="", causal_lm_class="",
                required_layers=[],
                constants={},
                success=False,
                errors=["No nntrainer_graph_ir available for code generation"]
            )
        
        # Validate graph_ir
        validation = self.validate_nntrainer_graph_ir(graph_ir)
        if not validation["valid"]:
            return GeneratedFiles(
                header="", source="",
                header_filename="", source_filename="",
                architecture=architecture,
                transformer_class="", causal_lm_class="",
                required_layers=[],
                constants={},
                success=False,
                errors=validation["errors"]
            )
        
        # Extract constants
        constants = self.extract_constants_from_config(hf_config)
        # Keep an LLM-produced component isolated from same-named handwritten
        # classes. The architecture string itself remains unchanged in state;
        # only C++ symbols and filenames receive the Workbench suffix.
        constants['ARCHITECTURE_NAME'] = architecture.replace('ForCausalLM', '') + 'Workbench'
        
        # Build prompt for LLM
        prompt = self._build_generation_prompt(
            architecture=architecture,
            hf_config=hf_config,
            graph_ir=graph_ir,
            constants=constants
        )
        
        # Call LLM (Cline) for generation
        llm_response = self._call_llm(prompt, state)
        
        # Parse LLM response
        header = llm_response.get("header", "")
        source = llm_response.get("source", "")
        
        if not header or not source:
            return GeneratedFiles(
                header="", source="",
                header_filename="", source_filename="",
                architecture=architecture,
                transformer_class="", causal_lm_class="",
                required_layers=[],
                constants={},
                success=False,
                errors=["LLM failed to generate complete code"]
            )
        
        # Extract metadata from generated code
        transformer_class = self._extract_class_name(header, r"class\s+(\w+Transformer)")
        causal_lm_class = self._extract_class_name(header, r"class\s+(\w+CausalLM)")
        required_layers = list(set(re.findall(r'createLayer\s*\(\s*"(\w+)"', source)))
        
        # Validate generated code syntax using CompilationValidator
        syntax_valid = True
        syntax_errors = []
        try:
            from ..harness.validator import CompilationValidator
            causallm_root = state.get("causallm_project_root", "")
            nntrainer_root = state.get("nntrainer_root", self.nntrainer_root)
            
            if causallm_root and nntrainer_root:
                validator = CompilationValidator(causallm_root=causallm_root, nntrainer_root=nntrainer_root)
                validation_result = validator.validate_syntax(
                    header=header,
                    source=source,
                    architecture=constants['ARCHITECTURE_NAME']
                )
                syntax_valid = validation_result.success
                syntax_errors = validation_result.errors
                if not syntax_valid:
                    bus.log(f"Generated code syntax validation failed: {len(syntax_errors)} errors", "error")
                else:
                    bus.log("Generated code syntax validation passed", "info")
        except ImportError:
            bus.log("CompilationValidator not available - skipping syntax validation", "debug")
        except Exception as e:
            bus.log(f"Syntax validation failed: {e}", "warn")
        
        # Support timestamp filenames (Option B from Option D)
        base_name = constants['ARCHITECTURE_NAME'].lower()
        timestamp = state.get("filename_timestamp", "")
        
        if timestamp:
            header_filename = f"{base_name}_{timestamp}_causallm.h"
            source_filename = f"{base_name}_{timestamp}_causallm.cpp"
        else:
            header_filename = f"{base_name}_causallm.h"
            source_filename = f"{base_name}_causallm.cpp"
        
        return GeneratedFiles(
            header=header,
            source=source,
            header_filename=header_filename,
            source_filename=source_filename,
            architecture=architecture,
            transformer_class=transformer_class,
            causal_lm_class=causal_lm_class,
            required_layers=required_layers,
            constants=constants,
            success=syntax_valid,
            errors=syntax_errors if not syntax_valid else []
        )
    
    def _symbolize_attributes(self, node: dict, constants: dict,
                              hf_config: dict = None) -> Dict[str, str]:
        """Re-express a node's concrete graph_ir attribute values as the
        symbolic C++ the CausalLM base class already exposes -- the
        createAttention/createMlp parameters (n_heads, head_dim, dim,
        hidden_dim) and the generated constants (DIM, GQA_SIZE, NORM_EPS...).

        The graph_ir carries this model's resolved numbers (unit=2048,
        epsilon=1e-06). Handing those to the LLM verbatim is what makes it
        emit `withKey("unit", "2048")`, which silently hardcodes one config
        into a file that is supposed to be driven by cfg/nntr_cfg. Showing
        the symbol instead keeps the generated component reusable.
        """
        def _int(key, default=0):
            try:
                return int(constants.get(key) or default)
            except (TypeError, ValueError):
                return default

        n_heads = _int("NUM_HEADS")
        dim = _int("HIDDEN_SIZE")
        gqa = max(1, _int("GQA_SIZE", 1))
        hidden_dim = _int("INTERMEDIATE_SIZE")
        vocab = _int("NUM_VOCAB")

        # head_dim is NOT always hidden_size / num_heads. Qwen3 and Gemma3 set
        # `head_dim` explicitly in config (Qwen3-0.6B: hidden=1024, heads=16,
        # head_dim=128, so hidden/heads=64 would be wrong and every q/k/v
        # `unit` would fail to match its expression). Trust the config first.
        head_dim = 0
        if hf_config:
            try:
                head_dim = int(hf_config.get("head_dim") or 0)
            except (TypeError, ValueError):
                head_dim = 0
        if not head_dim and n_heads:
            head_dim = dim // n_heads

        name = node.get("name", "")
        in_mlp = ".mlp" in name or "mlp" in (node.get("semantic_type") or "")

        # `unit` cannot be resolved by value alone. In GQA models the
        # arithmetic collides: for Qwen3-0.6B head_dim*n_heads/GQA_SIZE and
        # DIM are both 1024, so k_proj (which wants the GQA expression) and
        # o_proj (which wants DIM) look identical numerically. Pick the
        # candidate order from the projection's ROLE, matched loosely against
        # the HF node name, and only fall back to value order if the role is
        # unrecognised.
        role_haystack = f"{name} {node.get('semantic_type') or ''}".lower()

        def _has(*fragments):
            return any(fragment in role_haystack for fragment in fragments)

        def unit_candidates():
            if in_mlp:
                if _has("down_proj", "down", "o_proj"):
                    return [(dim, "dim"), (hidden_dim, "hidden_dim")]
                return [(hidden_dim, "hidden_dim"), (dim, "dim")]
            if _has("lm_head", "logits"):
                return [(vocab, "NUM_VOCAB")]
            if _has("q_proj", "query", "_wq"):
                return [(head_dim * n_heads, "head_dim * n_heads")]
            if _has("k_proj", "v_proj", "key", "value", "_wk", "_wv"):
                return [(head_dim * n_heads // gqa, "head_dim * n_heads / GQA_SIZE")]
            if _has("o_proj", "out_proj", "attention_out", "_wo"):
                return [(dim, "DIM")]
            return [
                (head_dim * n_heads, "head_dim * n_heads"),
                (head_dim * n_heads // gqa, "head_dim * n_heads / GQA_SIZE"),
                (dim, "DIM"),
                (vocab, "NUM_VOCAB"),
            ]

        per_key = {
            "unit": unit_candidates,
            "out_dim": lambda: [(vocab, "NUM_VOCAB"), (dim, "DIM")],
            "feature_size": lambda: [(head_dim, "head_dim")],
            "num_heads": lambda: [(n_heads, "n_heads"), (n_heads // gqa, "n_heads / GQA_SIZE")],
            "num_heads_kv": lambda: [(n_heads // gqa, "n_heads / GQA_SIZE"), (n_heads, "n_heads")],
            "num_kv_heads": lambda: [(n_heads // gqa, "n_heads / GQA_SIZE")],
            "epsilon": lambda: [(constants.get("NORM_EPS"), "NORM_EPS")],
            "rope_theta": lambda: [(constants.get("ROPE_THETA"), "ROPE_THETA")],
            "max_position_embeddings": lambda: [
                (constants.get("MAX_POSITION_EMBEDDINGS"), "MAX_POSITION_EMBEDDINGS")],
            "sliding_window": lambda: [(constants.get("SLIDING_WINDOW"), "SLIDING_WINDOW")],
            "max_timestep": lambda: [(None, "MAX_SEQ_LEN")],
            "max_new_tokens": lambda: [(None, "NUM_TO_GENERATE")],
            "attn_logit_softcapping": lambda: [
                (constants.get("ATTN_LOGIT_SOFTCAPPING"), "ATTN_LOGIT_SOFTCAPPING")],
        }

        symbolic = {}
        for key, value in (node.get("attributes") or {}).items():
            if isinstance(value, bool):
                # bools are already config-independent; IS_CAUSAL is the one
                # that has a real constant behind it.
                symbolic[key] = "IS_CAUSAL" if key == "is_causal" else ("true" if value else "false")
                continue

            candidates = per_key.get(key)
            if candidates is None:
                symbolic[key] = value
                continue

            resolved = None
            for candidate_value, expression in candidates():
                if candidate_value is None:
                    resolved = expression
                    break
                if _values_match(candidate_value, value):
                    resolved = expression
                    break
            symbolic[key] = resolved if resolved is not None else value

        return symbolic

    def _build_generation_prompt(self, architecture: str, hf_config: dict,
                                  graph_ir: dict, constants: dict) -> str:
        """
        Build comprehensive prompt for LLM code generation.
        """
        available_layers = get_available_layers()
        constants_cpp = self.format_constants_cpp(constants)

        # Build layer usage summary from graph_ir
        layer_usage = {}
        for node in graph_ir.get("nodes", []):
            node_type = node.get("node_type", "")
            layer_usage[node_type] = layer_usage.get(node_type, 0) + 1

        # Full layer reference: names alone (the old behaviour) don't tell
        # the model which properties each layer needs, so anything not
        # covered by the one worked createAttention example below (lm_head,
        # tie_word_embedding, scalar_multiply, activation, swiglu, ...) had
        # no usage guidance at all.
        layer_ref_lines = []
        for lname in available_layers:
            info = LAYER_CATALOG.get(lname, {})
            props = info.get("properties", {})
            req = props.get("required", [])
            opt = props.get("optional", [])
            note = info.get("note", "")
            line = f"- `{lname}`"
            if req:
                line += f" (required: {', '.join(req)})"
            if opt:
                line += f" (optional: {', '.join(opt)})"
            if note:
                line += f" -- {note}"
            layer_ref_lines.append(line)
        layer_reference = "\n".join(layer_ref_lines)

        # Node-detail sampling. Truncating to the first N nodes in sequential
        # graph order silently hides everything past the first 2-3 decoder
        # layers for any model with more than ~15-20 nodes/layer -- exactly
        # where per-layer variation (sliding-window alternation, differing
        # norm placement) shows up for non-uniform architectures. Instead,
        # always include every non-per-layer node (embedding, final norm,
        # etc.) plus ALL nodes from a handful of representative layers
        # spanning the full depth (first, second, middle, last), so a
        # structural change anywhere in the stack is visible rather than
        # only in the first layer or two.
        nodes = graph_ir.get("nodes", [])
        layer_re = re.compile(r"model\.layers\.(\d+)\.")
        by_layer: Dict[int, List[dict]] = {}
        other_nodes: List[dict] = []
        for node in nodes:
            m = layer_re.search(node.get("name", ""))
            if m:
                by_layer.setdefault(int(m.group(1)), []).append(node)
            else:
                other_nodes.append(node)

        representative: List[int] = []
        sampling_note = ""
        if by_layer:
            layer_indices = sorted(by_layer)
            if len(layer_indices) <= 4:
                representative = layer_indices
            else:
                representative = sorted({
                    layer_indices[0], layer_indices[1],
                    layer_indices[len(layer_indices) // 2],
                    layer_indices[-1],
                })
                sampling_note = (
                    f"\n(showing decoder layers {representative} out of "
                    f"{len(layer_indices)} total -- a sample spanning the "
                    f"full depth, not just the first ones. If "
                    f"`uniform_layers` below is False, inspect these samples "
                    f"carefully for what differs between them.)"
                )

        sample_nodes = list(other_nodes)
        for idx in representative:
            sample_nodes.extend(by_layer[idx])

        node_details = []
        for node in sample_nodes:
            symbolic = self._symbolize_attributes(node, constants, hf_config)
            raw = node.get("attributes", {}) or {}
            # Both views are shown on purpose: `Attributes` stays the verbatim
            # graph_ir record (nothing is hidden, including keys the
            # symbolizer doesn't know), while `Emit as` is the config-driven
            # expression to actually write into withKey(...).
            emit_pairs = ", ".join(
                f"{key}={expression}" for key, expression in symbolic.items()
            )
            node_details.append(f"""
    - {node['name']} ({node['node_type']}):
      Attributes: {json.dumps(raw)}
      Emit as: {emit_pairs or '(no properties)'}
      Semantic type: {node.get('semantic_type', 'N/A')}
""")

        # Structural metadata computed upstream (nntrainer_lowering) that was
        # previously discarded before reaching the prompt -- in particular
        # `uniform_layers`, the exact signal for "the sampling/pattern
        # guessing above is not safe here, check every layer shown".
        graph_metadata = graph_ir.get("metadata", {}) or {}
        uniform_layers = graph_metadata.get("uniform_layers", True)
        metadata_lines = [f"- uniform_layers: {uniform_layers}"]
        if uniform_layers is False:
            metadata_lines.append(
                "  <-- WARNING: decoder layers are NOT identical across this "
                "model. Do not assume every layer matches the samples below. "
                "If the block structure itself differs per layer (not just "
                "attention parameters), override createTransformerDecoderBlock "
                "(see AVAILABLE OVERRIDE HOOKS) instead of createAttention/"
                "createMlp alone."
            )
        if "num_layers" in graph_metadata:
            metadata_lines.append(f"- num_layers: {graph_metadata['num_layers']}")
        if "mlp_is_standard_gated_swiglu" in graph_metadata:
            metadata_lines.append(
                f"- mlp_is_standard_gated_swiglu: "
                f"{graph_metadata['mlp_is_standard_gated_swiglu']}"
            )
        metadata_block = "\n".join(metadata_lines)

        prompt = f"""
# C++ Code Generation for nntrainer CausalLM Model

## TASK
Generate C++ header and source files for the model: **{architecture}**

You are writing code from SCRATCH. Do NOT copy existing code. Use the nntrainer_graph_ir
as your source of truth for which layers to use and how to configure them.

## OUTPUT FILES
Generate TWO files:
1. `{constants['ARCHITECTURE_NAME'].lower()}_causallm.h` - Header with class declarations
2. `{constants['ARCHITECTURE_NAME'].lower()}_causallm.cpp` - Implementation

## MODEL CONFIGURATION (HuggingFace)
```json
{json.dumps(hf_config, indent=2)}
```

## CONSTANTS TO DEFINE
Insert these constants in the source file after `namespace causallm {{`:

```cpp
{constants_cpp}
```

## AVAILABLE LAYERS (from Applications/CausalLM/layers/)
You MUST use ONLY these layers in your generated code. Each entry lists the
`withKey(...)` properties that layer needs -- properties not listed here do
not exist and will not compile:

{layer_reference}

## LAYER USAGE IN THIS MODEL
Based on the nntrainer_graph_ir, this model uses:
{json.dumps(layer_usage, indent=2)}

## MODEL STRUCTURE METADATA
{metadata_block}

## MODEL FAMILY PATTERNS (Reference Examples)

Different model families use different patterns. Match your model's architecture:

### Qwen/Qwen2/Qwen3 Family
- Uses QK normalization (reshaped_rms_norm after Q and K projections)
- GQA (Grouped Query Attention) with separate K/V heads
- SwiGLU MLP (gate_proj, up_proj, down_proj + swiglu)
- May have sliding window attention
- Example layer names: `model.layers.{id}.self_attn.q_proj`, `model.layers.{id}.mlp.gate_proj`

### Gemma/Gemma2/Gemma3 Family  
- Uses pre-attention RMSNorm (before attention, not after)
- GQA with QK normalization
- GeGLU or SwiGLU MLP variants
- May have per-layer sliding window alternation (Gemma3)
- Example layer names: `model.layers.{id}.input_layernorm`, `model.layers.{id}.mlp.gate_proj`

### Llama/Llama2/Llama3 Family
- Uses RMSNorm (not reshaped)
- RoPE without QK normalization
- SwiGLU MLP (gate, up, down projections)
- May use GQA in newer versions
- Example layer names: `model.layers.{id}.self_attn.q_proj`, `model.layers.{id}.mlp.gate_proj`

### Phi/Phi2/Phi3 Family
- Often uses parallel attention (QKV fused)
- May use GQA
- MLP variants vary by version
- Example layer names: `model.layers.{id}.self_attn.qkv_proj`

### Mistral/Mixtral Family
- Sliding window attention (Mistral)
- MoE (Mixture of Experts) with sparse routing (Mixtral)
- SwiGLU MLP
- Example layer names: `model.layers.{id}.block_sparse_moe.gate`, `model.layers.{id}.self_attn.q_proj`

### Common Patterns
- **Embedding**: All models start with token embeddings
- **LM Head**: All models end with LM head for logits
- **Residual connections**: Use addition layer for residual around attention and MLP
- **Normalization**: RMSNorm, reshaped_rms_norm, or layer_norm depending on family

## NODE DETAILS (from nntrainer_graph_ir)
Here are the key nodes from the graph:{sampling_note}
{''.join(node_details)}

## AVAILABLE OVERRIDE HOOKS (from Applications/CausalLM/models/transformer.h)
These are ALL of the protected virtuals Transformer exposes for
customization -- this is the complete list, not a subset. Override ONLY the
ones this model actually needs; the default base implementation is used for
anything you don't override.

- `Tensor createAttention(const int layer_id, int seq_len, int n_heads, int head_dim, Tensor query, Tensor key, Tensor value)` -- almost always needed; builds Q/K/V projections + mha_core + output projection for one layer. See the worked example below.
- `Tensor createMlp(const int layer_id, int dim, int hidden_dim, Tensor input)` -- override for non-default MLP variants (SwiGLU, GeGLU, etc). See the worked example below.
- `Tensor createTransformerDecoderBlock(const int layer_id, Tensor input)` -- override ONLY if the decoder block's shape itself differs from norm->attention->residual->norm->mlp->residual (e.g. Gemma-style "sandwich" norms that re-normalize AFTER attention/mlp and before the residual add, on top of the usual pre-norms). If you override this, you are responsible for calling createAttention/createMlp yourself and building the full block -- see the commented example below.
- `void registerCustomLayers()` -- REQUIRED whenever you use any layer type not built into nntrainer core (reshaped_rms_norm, swiglu, etc). See the REQUIRED registration code below -- this one is boilerplate, not a design choice.
- `std::pair<Tensor, Tensor> constructModel()` -- rarely needed; only if the overall embedding->decoder->final-norm graph shape itself differs from the standard CausalLM assembly (e.g. non-standard embedding scaling done outside embed, or an extra head). Prefer createTransformerDecoderBlock first; this is a last resort.
- `void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg)` -- rarely needed; only if a constant can't be derived from the config the standard way. Prefer using the provided CONSTANTS block first; this is a last resort.

## REQUIRED CODE STRUCTURE

### Header File Structure
```cpp
// SPDX-License-Identifier: Apache-2.0
#ifndef __{constants['ARCHITECTURE_NAME'].upper()}_CAUSALLM__
#define __{constants['ARCHITECTURE_NAME'].upper()}_CAUSALLM__

#include <nntrainer/causal_lm.h>

namespace causallm {{

class {constants['ARCHITECTURE_NAME']}Transformer : virtual public Transformer {{
public:
  static constexpr const char *architectures = "{architecture}";

  // Constructor -- definition is REQUIRED boilerplate, see the
  // "CONSTRUCTORS" section of the source file below before writing this.
  {constants['ARCHITECTURE_NAME']}Transformer(json &cfg, json &generation_cfg, json &nntr_cfg);

  virtual ~{constants['ARCHITECTURE_NAME']}Transformer() = default;

  // Override createAttention for custom attention
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  // Optional: Override createMlp for custom MLP
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  // Optional -- ONLY declare this if the model needs it (see
  // AVAILABLE OVERRIDE HOOKS above): sandwich-norm architectures whose
  // decoder block shape itself differs from the default. Delete this line
  // entirely if not needed -- do not declare it without a body in source.
  // Tensor createTransformerDecoderBlock(const int layer_id, Tensor input) override;

  // Register custom layers
  void registerCustomLayers() override;
}};

class {constants['ARCHITECTURE_NAME']}CausalLM : public CausalLM, public {constants['ARCHITECTURE_NAME']}Transformer {{
public:
  static constexpr const char *architectures = "{architecture}";
  
  {constants['ARCHITECTURE_NAME']}CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);
  
  virtual ~{constants['ARCHITECTURE_NAME']}CausalLM() = default;
  
  void registerCustomLayers() override;
}};

}} // namespace causallm
#endif
```

### Source File Structure
```cpp
#include <nntrainer/llm_util.hpp>
#include <nntrainer/model.h>
#include <nntrainer/{constants['ARCHITECTURE_NAME'].lower()}_causallm.h>

// Include any custom layers used based on your model architecture
#include <nntrainer/reshaped_rms_norm.h>
#include <nntrainer/mha_core.h>
#include <nntrainer/fully_connected_layer.h>
#include <nntrainer/embedding_layer.h>
#include <nntrainer/lm_head.h>
#include <nntrainer/swiglu.h>
#include <nntrainer/multiply.h>
#include <nntrainer/addition.h>

namespace causallm {{

// Constants (insert generated constants here)
{constants_cpp}

// ============================================================================
// CONSTRUCTORS -- REQUIRED, copy this pattern EXACTLY, only renaming the
// class. Do not shorten, reorder, or omit any part of the initializer lists.
// ============================================================================
// {constants['ARCHITECTURE_NAME']}CausalLM inherits from BOTH CausalLM and
// {constants['ARCHITECTURE_NAME']}Transformer, and both of THOSE inherit
// Transformer VIRTUALLY -- so there is only one Transformer sub-object in
// the whole hierarchy. C++ rule for virtual bases: only the MOST-DERIVED
// class's own constructor initializer list can initialize a virtual base;
// any "Transformer(...)" written inside {constants['ARCHITECTURE_NAME']}Transformer's
// or CausalLM's constructor is SILENTLY IGNORED when they are used as a base
// of {constants['ARCHITECTURE_NAME']}CausalLM. Getting this wrong does not
// fail to compile -- Transformer's default constructor runs instead, so
// NUM_VOCAB/DIM/NUM_LAYERS/etc. are never set and the model is broken at
// runtime with no compiler warning.
{constants['ARCHITECTURE_NAME']}Transformer::{constants['ARCHITECTURE_NAME']}Transformer(
    json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {{}}

{constants['ARCHITECTURE_NAME']}CausalLM::{constants['ARCHITECTURE_NAME']}CausalLM(
    json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),  // MUST be first, explicit, and pass ModelType::CAUSALLM
    CausalLM(cfg, generation_cfg, nntr_cfg),
    {constants['ARCHITECTURE_NAME']}Transformer(cfg, generation_cfg, nntr_cfg) {{}}

// ============================================================================
// EMBEDDING / FINAL NORM / LM HEAD -- DO NOT WRITE THESE
// ============================================================================
// Transformer::constructModel() in the base class ALREADY builds, in order:
//   1. the token embedding ("embedding0" -- it picks "tie_word_embeddings" or
//      "embedding_layer" based on TIE_WORD_EMBEDDINGS, and sets vocab/dim/
//      dtype itself via buildEmbeddingLayerProperties)
//   2. the NUM_LAYERS decoder loop, calling YOUR createAttention/createMlp
//   3. the final "output_norm" rms_norm, and the LM head
//
// So the graph_ir's embedding node, final-norm node and lm_head node are
// ALREADY IMPLEMENTED for you. They appear in NODE DETAILS because they are
// part of the model, not because you must emit them.
//
// There is NO `embed()` virtual on Transformer -- do not define or declare
// one. Writing `Tensor {constants['ARCHITECTURE_NAME']}Transformer::embed(...)` is a
// compile error (no such declaration in the base class).
//
// Only if the embedding assembly itself genuinely differs from the standard
// pattern do you override `constructModel()` -- and then you are responsible
// for the whole embedding -> decoder -> output_norm graph. This is a last
// resort; see AVAILABLE OVERRIDE HOOKS.

// ============================================================================
// ATTENTION LAYER - Custom attention implementation
// ============================================================================
Tensor {constants['ARCHITECTURE_NAME']}Transformer::createAttention(
    const int layer_id, int seq_len, int n_heads,
    int head_dim, Tensor query, Tensor key, Tensor value) {{
  
  // Q projection - use fully_connected layer
  // NOTE: withKey() accepts string values - nntrainer converts them at runtime.
  // The reference implementation (models/qwen3/qwen3_causallm.cpp) uses strings like "true", "false", std::to_string().
  LayerHandle wq(createLayer(
      "fully_connected",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
       withKey("unit", std::to_string(head_dim * n_heads)),
       withKey("disable_bias", "true"),
       withKey("weight_initializer", "ones")}}));
  Tensor q = wq(query);
  
  // Q normalization (if your model uses QK normalization like Qwen3)
  LayerHandle q_norm(createLayer(
      "reshaped_rms_norm",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
       withKey("packed", "false"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("feature_size", std::to_string(head_dim))}}));
  Tensor q_normed = q_norm(q);
  
  // K projection
  LayerHandle wk(createLayer(
      "fully_connected",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
       withKey("unit", std::to_string(head_dim * n_heads / GQA_SIZE)),
       withKey("disable_bias", "true"),
       withKey("weight_initializer", "ones")}}));
  Tensor k = wk(key);
  
  // K normalization (if your model uses QK normalization)
  LayerHandle k_norm(createLayer(
      "reshaped_rms_norm",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
       withKey("packed", "false"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("feature_size", std::to_string(head_dim))}}));
  Tensor k_normed = k_norm(k);
  
  // V projection
  LayerHandle wv(createLayer(
      "fully_connected",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
       withKey("unit", std::to_string(head_dim * n_heads / GQA_SIZE)),
       withKey("disable_bias", "true"),
       withKey("weight_initializer", "ones")}}));
  Tensor v = wv(value);
  
  // External KV cache placeholders (per-layer)
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
  
  // Attention core - supports sliding window, RoPE, GQA
  // NOTE: Use string values - nntrainer's withKey() converts them at runtime
  LayerHandle mha(createLayer(
      "mha_core",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
       withKey("num_heads", std::to_string(n_heads)),
       withKey("num_heads_kv", std::to_string(n_heads / GQA_SIZE)),
       withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
       withKey("sliding_window", SLIDING_WINDOW),
       withKey("rope_theta", std::to_string(ROPE_THETA)),
       withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
       withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
       withKey("is_causal", IS_CAUSAL ? "true" : "false")}}));
  Tensor a = mha({{q_normed, k_normed, v, cache_k, cache_v}});
  
  // O projection (output projection)
  LayerHandle wo(createLayer(
      "fully_connected",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
       withKey("unit", std::to_string(HIDDEN_SIZE)),
       withKey("disable_bias", "true"),
       withKey("weight_initializer", "ones")}}));
  return wo(a);
}}

// ============================================================================
// MLP LAYER - Custom MLP implementation (if your model needs it)
// ============================================================================
// Example for SwiGLU-based MLP (adapt to your model):
// IMPORTANT: withKey values must match their C++ types (see ATTENTION LAYER example above)
/*
Tensor {constants['ARCHITECTURE_NAME']}Transformer::createMlp(
    const int layer_id, int dim, int hidden_dim, Tensor input) {{
  
  // Gate projection - disable_bias is bool, use true NOT "true"
  LayerHandle gate_proj(createLayer(
      "fully_connected",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_gate_proj"),
       withKey("unit", hidden_dim),
       withKey("disable_bias", true),
       withKey("weight_initializer", "ones")}}));
  Tensor gate = gate_proj(input);
  
  // Up projection
  LayerHandle up_proj(createLayer(
      "fully_connected",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_up_proj"),
       withKey("unit", hidden_dim),
       withKey("disable_bias", true),
       withKey("weight_initializer", "ones")}}));
  Tensor up = up_proj(input);
  
  // SwiGLU activation (no properties needed)
  LayerHandle swiglu(createLayer(
      "swiglu",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_swiglu")}}));
  Tensor gated = swiglu({{gate, up}});
  
  // Down projection
  LayerHandle down_proj(createLayer(
      "fully_connected",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_down_proj"),
       withKey("unit", dim),
       withKey("disable_bias", true),
       withKey("weight_initializer", "ones")}}));
  return down_proj(gated);
}}
*/

// ============================================================================
// TRANSFORMER DECODER BLOCK - ONLY implement if the block shape itself
// differs from the default (see AVAILABLE OVERRIDE HOOKS above). Do NOT
// uncomment/declare this if createAttention + createMlp overrides alone are
// enough -- the base Transformer::createTransformerDecoderBlock already
// does norm->attention->residual->norm->mlp->residual correctly.
// Example for a Gemma-style "sandwich norm" block (extra norms AFTER
// attention/mlp and before each residual add, on top of the usual pre-norms):
/*
Tensor {constants['ARCHITECTURE_NAME']}Transformer::createTransformerDecoderBlock(
    const int layer_id, Tensor input) {{
  LayerHandle input_norm(createLayer(
      "rms_norm",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_input_norm"),
       withKey("epsilon", std::to_string(NORM_EPS))}}));
  Tensor normed = input_norm(input);

  Tensor attn_out = createAttention(layer_id, /* seq_len */ MAX_SEQ_LEN,
                                    NUM_HEADS, HEAD_DIM, normed, normed, normed);

  // Sandwich norm: re-normalize the attention output BEFORE the residual add
  LayerHandle post_attn_norm(createLayer(
      "rms_norm",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_post_attention_norm"),
       withKey("epsilon", std::to_string(NORM_EPS))}}));
  Tensor attn_normed = post_attn_norm(attn_out);

  LayerHandle attn_residual(createLayer(
      "addition",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_attention_residual")}}));
  Tensor after_attn = attn_residual({{input, attn_normed}});

  LayerHandle mlp_input_norm(createLayer(
      "rms_norm",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_post_attention_layernorm"),
       withKey("epsilon", std::to_string(NORM_EPS))}}));
  Tensor mlp_normed = mlp_input_norm(after_attn);

  Tensor mlp_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, mlp_normed);

  LayerHandle post_mlp_norm(createLayer(
      "rms_norm",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_post_feedforward_layernorm"),
       withKey("epsilon", std::to_string(NORM_EPS))}}));
  Tensor mlp_final = post_mlp_norm(mlp_out);

  LayerHandle mlp_residual(createLayer(
      "addition",
      {{withKey("name", "layer" + std::to_string(layer_id) + "_mlp_residual")}}));
  return mlp_residual({{after_attn, mlp_final}});
}}
*/

// ============================================================================
// CUSTOM LAYER REGISTRATION
// ============================================================================
void {constants['ARCHITECTURE_NAME']}Transformer::registerCustomLayers() {{
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  
  // Register custom layers used by this model
  // Only register layers that are NOT built-in to nntrainer
  // Common custom layers: ReshapedRMSNormLayer, SwiGLULayer, etc.
  try {{
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
    // Add more custom layer registrations as needed:
    // app_context->registerFactory(
    //   nntrainer::createLayer<causallm::SwiGLULayer>);
  }} catch (std::invalid_argument &e) {{
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }}
}}

void {constants['ARCHITECTURE_NAME']}CausalLM::registerCustomLayers() {{
  // IMPORTANT: Call CausalLM base class first (not Transformer directly)
  CausalLM::registerCustomLayers();
  // Then register transformer-specific custom layers
  {constants['ARCHITECTURE_NAME']}Transformer::registerCustomLayers();
}}

}} // namespace causallm
```

## IMPORTANT RULES

1. **Use ONLY layers from the available layers list**
2. **Use the exact attribute KEYS from the graph_ir** (e.g. `num_heads`,
   `sliding_window`) -- but NEVER the resolved numeric VALUES. For every
   property, emit the expression given on that node's `Emit as:` line
   (`head_dim * n_heads`, `NORM_EPS`, `ROPE_THETA`, `hidden_dim`, ...).
   Writing `withKey("unit", "2048")` or `withKey("epsilon", "1e-06")` is
   WRONG: it bakes this one checkpoint's shape into a file that must work
   for every config of this architecture. `withKey("unit", std::to_string(head_dim * n_heads))`
   is right. The numbers after `// this model resolves to` are for your
   sanity-check only -- they must not appear in the output.
3. **Name emitted layers with the CausalLM convention, NOT the graph_ir
   node name.** graph_ir node names are HuggingFace *weight identities*
   (`model.layers.0.self_attn.q_proj`) used for weight mapping -- they are
   not layer names. Emitted names must be
   `"layer" + std::to_string(layer_id) + "_<role>"` where `<role>` is the
   CausalLM role: `wq`, `wk`, `wv`, `q_norm`, `k_norm`, `attention`,
   `attention_out`, `gate_proj`, `up_proj`, `down_proj`, `swiglu`. Using the
   HF dotted name breaks weight loading at runtime.
4. **Follow the layer usage patterns** shown in the examples
4b. **Pass every mha_core property the model has**, including
   `sliding_window` (use the `SLIDING_WINDOW` constant -- it is `UINT_MAX`
   when the model has no sliding window, which is the correct "disabled"
   value). Omitting it silently changes attention behaviour.
5. **Include all required headers** for layers you use
6. **Define all constants** from the HuggingFace config
7. **Use proper C++ syntax** - this code will be compiled
8. **Copy the CONSTRUCTORS exactly as shown** -- only rename the class, never shorten or reorder the initializer lists. This is the single most common way generated code silently fails at runtime while still compiling cleanly (see the CONSTRUCTORS section for why).
9. **Only override hooks from AVAILABLE OVERRIDE HOOKS** that this model actually needs -- if `createTransformerDecoderBlock` isn't needed, do not declare it in the header (a declaration with no matching definition is a link error).
10. **If `uniform_layers` is False** (see MODEL STRUCTURE METADATA), do not assume every decoder layer is identical -- check the sampled layers under NODE DETAILS for what changes between them (attention parameters, sliding window, or the block structure itself).

## REFERENCE IMPLEMENTATIONS -- READ FOR CONVENTION, NEVER COPY

Hand-written, known-good CausalLM components live in:
  `{self.models_dir}/`
(e.g. `qwen3/qwen3_causallm.cpp`, `gemma3/gemma3_causallm.cpp`)

Use them **only** to match conventions -- how layers are named, how
`withKey` values are expressed as constants/parameters instead of literals,
how `registerCustomLayers` chains to `CausalLM::registerCustomLayers()`, how
the constructor initializer lists are ordered.

Do NOT copy one and rename it, and do NOT reproduce another
architecture's structure. Your output must be derived from THIS model's
nntrainer_graph_ir above. Where the graph_ir and a reference file disagree
about which layers exist or how they connect, **the graph_ir wins** -- it
describes the model you are generating; the reference file describes a
different model that merely shares the house style.

Quality bar: the generated file should be indistinguishable in style and
rigour from those reference files, while being structurally faithful to the
graph_ir.

## GENERATE THE CODE NOW

Return ONLY the generated code in this format:

```cpp
// === HEADER FILE ===
<complete header content>

// === SOURCE FILE ===
<complete source content>
```
"""
        return prompt
    
    def _call_llm(self, prompt: str, state: dict) -> Dict[str, str]:
        """
        Call an LLM for code generation DIRECTLY using Claude CLI.

        Does NOT delegate to llm_codegen to avoid the old prompts_causallm.py.
        Uses our harness prompt directly.

        Backend: Claude CLI (requires local `claude login`)
        """
        from ...events import bus

        system_prompt = ("You are an expert C++ code generator for nntrainer "
                         "CausalLM models.")
        raw_text = None

        # Use Claude CLI backend (requires local `claude login`)
        try:
            from ..claude_cli_backend import _make_claude_cli_llm
            cli_llm = _make_claude_cli_llm(state)
            if cli_llm is not None:
                bus.log("Calling Claude CLI for code generation", "info")
                raw_text = cli_llm(system_prompt, prompt)
            else:
                bus.log("Claude CLI backend returned None", "error")
        except ImportError:
            bus.log("Claude CLI backend not available (claude_cli_backend module not found)", "error")
        except Exception as exc:
            bus.log(f"Claude CLI backend failed: {exc}", "error")

        if not raw_text:
            return {"header": "", "source": "",
                    "error": "Claude CLI not available (ensure claude CLI is installed and authenticated with 'claude login')"}

        try:
            bus.log(f"LLM response length: {len(raw_text)} chars", "info")

            # Models (even capable ones) often add prose after the requested
            # code block ("Notes on the design: ..."). The whole response is
            # supposed to be a single ```cpp fence (see the prompt's "GENERATE
            # THE CODE NOW" section) -- if one is present, narrow to its
            # contents FIRST so the marker regexes below can't run past the
            # closing fence into trailing commentary.
            fenced = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)\n```', raw_text, re.DOTALL)
            parse_text = fenced.group(1) if fenced else raw_text

            # Strategy 1: Look for our marker format
            header_match = re.search(r'// === HEADER FILE ===\s*\n(.*?)\n// === SOURCE FILE ===', parse_text, re.DOTALL)
            source_match = re.search(r'// === SOURCE FILE ===\s*\n(.*)', parse_text, re.DOTALL)

            if header_match and source_match:
                bus.log("Parsed using marker format", "info")
                return {
                    "header": header_match.group(1).strip(),
                    "source": source_match.group(1).strip()
                }
            
            # Strategy 2: Extract C++ code blocks
            code_blocks = re.findall(r'```cpp\s*(.*?)\s*```', raw_text, re.DOTALL)
            if len(code_blocks) >= 2:
                bus.log(f"Parsed {len(code_blocks)} code blocks", "info")
                return {"header": code_blocks[0].strip(), "source": code_blocks[1].strip()}
            
            # Strategy 3: Single code block - assume it's the source
            if len(code_blocks) == 1:
                bus.log("Single code block found, generating minimal header", "info")
                return {
                    "header": f"// Generated header for {state.get('architecture', 'model')}",
                    "source": code_blocks[0].strip()
                }
            
            # Strategy 4: Look for .h and .cpp file markers
            h_match = re.search(r'\.h[^\n]*\n(.*?)(?=\.cpp|\Z)', raw_text, re.DOTALL)
            cpp_match = re.search(r'\.cpp[^\n]*\n(.*?)$', raw_text, re.DOTALL)
            
            if h_match and cpp_match:
                bus.log("Parsed using file markers", "info")
                return {"header": h_match.group(1).strip(), "source": cpp_match.group(1).strip()}
            
            # Strategy 5: Return raw text as source with minimal header
            bus.log("Could not parse - returning raw text as source", "warn")
            return {
                "header": f"// Generated header for {state.get('architecture', 'model')}",
                "source": raw_text,
                "warning": "Could not parse header/source separation - raw LLM response returned"
            }
                    
        except Exception as e:
            bus.log(f"LLM call failed: {e}", "error")
            return {"header": "", "source": "", "error": str(e)}
    
    def _extract_class_name(self, content: str, pattern: str) -> str:
        """Extract class name matching pattern."""
        match = re.search(pattern, content)
        return match.group(1) if match else ""
