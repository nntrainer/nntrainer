"""
NOOA-Based Layer Generation Agent

Generates missing layer implementations using LLM (via NOOA/Cline).
For layers not found in the registry, this agent:

1. Analyzes layer requirements
2. Uses LLM to generate proper NNTrainer layer implementations
3. Follows existing layer patterns from the codebase
4. Stores generated layers in artifact directory

This agent is called only for unknown layers that cannot be found
in the standard registry.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..converters.layer_registry import LayerInfo, get_layer_registry

logger = logging.getLogger(__name__)


class NOOALayerGeneratorAgent:
    """Generate missing layer implementations using LLM."""

    LAYER_GENERATION_SYSTEM_PROMPT = """You are an expert NNTrainer C++ developer.
Your task is to generate custom layer implementations that follow NNTrainer patterns.

Requirements:
1. Follow NNTrainer Layer API conventions
2. Inherit from Layer base class
3. Implement required virtual methods: Finalize, InitParameters, Forward, Backward
4. Use proper tensor operations
5. Include comprehensive documentation
6. Match the coding style of existing layers (rms_norm, mha_core, swiglu)

Generate ONLY valid, compilable C++ code following NNTrainer conventions."""

    def __init__(self):
        self.registry = get_layer_registry()
        self.logger = logger
        self.artifact_layers: Dict[str, Dict[str, str]] = {}  # name -> {header, source}

    def generate_layer(
        self,
        layer_name: str,
        layer_spec: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """
        Generate a missing layer implementation.

        Args:
            layer_name: Name of the layer to generate
            layer_spec: Layer specification with:
                - description: What the layer does
                - inputs: Input tensor names
                - outputs: Output tensor names
                - properties: Configuration properties
                - reference_layers: Similar layers to use as reference

        Returns:
            Dictionary with "header" and "source" keys, or None if generation failed
        """
        self.logger.info(f"Generating layer: {layer_name}")

        try:
            # Build prompt
            prompt = self._build_generation_prompt(layer_name, layer_spec)

            # For now, return a template since NOOA integration needs authentication
            # In production, this would call:
            # llm_response = await nooa_agent.generate(prompt)
            # return self._parse_generated_code(llm_response)

            return self._generate_template(layer_name, layer_spec)

        except Exception as e:
            self.logger.error(f"Failed to generate layer {layer_name}: {e}")
            return None

    def _build_generation_prompt(
        self,
        layer_name: str,
        layer_spec: Dict[str, Any],
    ) -> str:
        """Build the prompt for LLM layer generation."""
        prompt = f"""Generate a custom NNTrainer layer implementation for:

Layer Name: {layer_name}
Description: {layer_spec.get('description', 'Custom layer')}

Inputs: {', '.join(layer_spec.get('inputs', []))}
Outputs: {', '.join(layer_spec.get('outputs', []))}

Properties:
"""
        for prop_name, prop_desc in layer_spec.get('properties', {}).items():
            prompt += f"  - {prop_name}: {prop_desc}\n"

        if layer_spec.get('reference_layers'):
            prompt += f"\nReference similar implementations: {', '.join(layer_spec['reference_layers'])}\n"

        prompt += """
Generate TWO files:

1. HEADER FILE ({layer_name}.h):
   - Class definition inheriting from Layer
   - Property getters/setters
   - Forward declaration of helper methods

2. SOURCE FILE ({layer_name}.cpp):
   - Full implementation of all virtual methods
   - Forward pass computation
   - Backward pass computation
   - Parameter initialization

Follow NNTrainer conventions exactly as seen in Applications/CausalLM/layers/
"""
        return prompt

    def _generate_template(
        self,
        layer_name: str,
        layer_spec: Dict[str, Any],
    ) -> Dict[str, str]:
        """Generate a template implementation."""
        class_name = self._to_class_name(layer_name)
        guard_name = f"__{layer_name.upper()}_H__"

        header = f'''/**
 * {class_name} Layer for NNTrainer
 *
 * Auto-generated layer implementation
 * {layer_spec.get('description', 'Custom layer')}
 */

#ifndef {guard_name}
#define {guard_name} {guard_name}

#include <layer.h>
#include <tensor.h>
#include <weight.h>

namespace causallm {{

/**
 * @class {class_name}
 * @brief {layer_spec.get('description', 'Custom layer')}
 */
class {class_name} : public Layer {{
 public:
  static constexpr const char *type_str = "{layer_name}";

  /**
   * @brief Construct a new {class_name} Layer object
   */
  {class_name}(const PropsType &props = PropsType()) : Layer(props) {{}}

  /**
   * @brief Destroy the {class_name} Layer object
   */
  ~{class_name}() override = default;

  /**
   * @copydoc Layer::finalize
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding
   */
  void forwarding(RunLayerContext &context, bool training = false) override;

  /**
   * @copydoc Layer::incremental_forwarding
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training = false) override;

  /**
   * @copydoc Layer::calcDerivative
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty
   */
  void setProperty(const std::string &key, const std::string &value) override;

 private:
  // Layer-specific properties
  // TODO: Add properties from {layer_spec.get('properties', {})}
}};

}}  // namespace causallm

#endif  /* {guard_name} */
'''

        source = f'''/**
 * {class_name} Layer Implementation
 *
 * Auto-generated implementation
 */

#include "{layer_name}.h"
#include <nntrainer_log.h>

namespace causallm {{

void {class_name}::finalize(InitLayerContext &context) {{
  // Get input/output dimensions
  // Initialize weights and biases
  // Validate tensor shapes

  LOGI("{layer_name} layer finalized");
}}

void {class_name}::forwarding(RunLayerContext &context, bool training) {{
  // Forward pass computation
  // Inputs: {', '.join(layer_spec.get('inputs', []))}
  // Outputs: {', '.join(layer_spec.get('outputs', []))}

  // TODO: Implement forward pass logic
}}

void {class_name}::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {{
  // Incremental forward pass (for inference with KV cache)
  // This is important for efficient generation

  // TODO: Implement incremental forward pass
  // For most layers, this is the same as normal forwarding
}}

void {class_name}::calcDerivative(RunLayerContext &context) {{
  // Backward pass computation
  // Calculate gradients with respect to inputs

  // TODO: Implement backward pass
}}

void {class_name}::setProperty(const std::string &key,
                              const std::string &value) {{
  // Parse and set layer properties
  // Common patterns:
  //   if (key == "epsilon") {{ epsilon_ = std::stof(value); }}
  //   else if (key == "units") {{ units_ = std::stoi(value); }}

  // TODO: Implement property setting based on layer_spec
}}

}}  // namespace causallm
'''

        return {
            "header": header,
            "source": source,
            "header_filename": f"{layer_name}.h",
            "source_filename": f"{layer_name}.cpp",
        }

    @staticmethod
    def _to_class_name(layer_name: str) -> str:
        """Convert layer_name to ClassName."""
        parts = layer_name.split("_")
        return "".join(p.capitalize() for p in parts) + "Layer"

    def save_generated_layer(
        self,
        layer_name: str,
        generated_code: Dict[str, str],
        artifact_dir: Path,
    ) -> Dict[str, str]:
        """Save generated layer to artifact directory."""
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        header_file = artifact_dir / generated_code["header_filename"]
        source_file = artifact_dir / generated_code["source_filename"]

        header_file.write_text(generated_code["header"], encoding="utf-8")
        source_file.write_text(generated_code["source"], encoding="utf-8")

        self.logger.info(f"Saved generated layer to:")
        self.logger.info(f"  Header: {header_file}")
        self.logger.info(f"  Source: {source_file}")

        # Register in artifact layers
        self.artifact_layers[layer_name] = {
            "header": str(header_file),
            "source": str(source_file),
        }

        # Register in global registry
        layer_info = LayerInfo(
            name=layer_name,
            type="generated",
            file_path=str(source_file),
            header_file=f"<{generated_code['header_filename'].replace('.h', '')}.h>",
            usage_string=f'createLayer("{layer_name}")',
            properties={"name": "Layer name"},
            inputs=[],
            outputs=[],
            description="AI-generated custom layer",
            reference=str(artifact_dir),
        )
        self.registry.register_generated_layer(layer_info)

        return {
            "header": str(header_file),
            "source": str(source_file),
        }

    def generate_build_integration(
        self,
        artifact_dir: Path,
        layers: List[str],
    ) -> str:
        """Generate meson.build snippet for artifact layers."""
        snippet = """# Artifact Layer Sources
artifact_layer_sources = [
"""
        for layer in layers:
            if layer in self.artifact_layers:
                source = self.artifact_layers[layer]["source"]
                snippet += f'  "{source}",\n'

        snippet += """]

# Include artifact layers in build
model_sources += artifact_layer_sources
"""
        return snippet

    def get_layer_usage_example(self, layer_name: str) -> Optional[str]:
        """Get C++ usage example for a generated layer."""
        layer = self.registry.get_layer(layer_name)
        if not layer or layer.type != "generated":
            return None

        return self.registry.get_layer_usage_code(layer_name)

    def print_generation_summary(self):
        """Print summary of generated layers."""
        if not self.artifact_layers:
            print("No generated layers yet")
            return

        print("\n" + "=" * 70)
        print("GENERATED LAYERS SUMMARY")
        print("=" * 70)

        for layer_name, files in self.artifact_layers.items():
            print(f"\n✨ {layer_name}")
            print(f"  Header: {files['header']}")
            print(f"  Source: {files['source']}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    agent = NOOALayerGeneratorAgent()

    # Example: Generate a missing layer
    layer_spec = {
        "description": "Custom normalization layer",
        "inputs": ["input"],
        "outputs": ["normalized"],
        "properties": {
            "epsilon": "Normalization epsilon (default: 1e-6)",
            "feature_size": "Feature dimension",
        },
        "reference_layers": ["rms_norm", "layer_norm"],
    }

    generated = agent.generate_layer("custom_norm", layer_spec)

    if generated:
        print("Generated layer:")
        print(f"  Header: {generated['header_filename']}")
        print(f"  Source: {generated['source_filename']}")
        print(f"\nHeader preview ({len(generated['header'])} bytes):")
        print(generated['header'][:500] + "...\n")
