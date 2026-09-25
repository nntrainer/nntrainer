"""
Enhanced C++ Code Generator with Layer Detection

Extends the basic C++ generator to:
1. Detect available layers from registry
2. Use correct layer patterns from instructions
3. Identify missing layers
4. Integrate with NOOA for generating missing layers
5. Manage artifact layers
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .nntrainer_cpp_generator import NNTrainerCppGenerator, ModelConfig, NNTrainerConfig, GenerationConfig
from .layer_registry import LayerRegistry, get_layer_registry
from .artifact_layer_manager import ArtifactLayerManager

logger = logging.getLogger(__name__)


class EnhancedCppGenerator:
    """
    Enhanced C++ generator with layer detection and artifact management.

    Process:
    1. Analyze model architecture to determine required layers
    2. Check which layers are available in registry
    3. Identify missing layers
    4. Generate C++ code using available layers with correct patterns
    5. For missing layers: trigger NOOA generation
    6. Store generated layers in artifact directory
    7. Integrate artifact layers into build system
    """

    def __init__(
        self,
        architecture: str,
        model_config: ModelConfig,
        nntrainer_config: NNTrainerConfig,
        generation_config: Optional[GenerationConfig] = None,
        artifact_dir: Optional[Path] = None,
    ):
        self.architecture = architecture
        self.model_config = model_config
        self.nntrainer_config = nntrainer_config
        self.generation_config = generation_config or GenerationConfig()

        self.layer_registry = get_layer_registry()
        self.base_generator = NNTrainerCppGenerator(
            architecture=architecture,
            model_config=model_config,
            nntrainer_config=nntrainer_config,
            generation_config=generation_config,
        )

        # Artifact management
        self.artifact_dir = artifact_dir or Path(".") / "artifacts" / self._extract_slug(architecture).lower()
        self.artifact_manager = ArtifactLayerManager(self.artifact_dir)

        self.required_layers = self._determine_required_layers()
        self.missing_layers = self.layer_registry.find_missing_layers(self.required_layers)
        self.available_layers = [l for l in self.required_layers if l not in self.missing_layers]

    def analyze(self) -> Dict[str, any]:
        """Analyze layer requirements and availability."""
        print("\n" + "=" * 70)
        print("LAYER ANALYSIS")
        print("=" * 70)

        print(f"\nArchitecture: {self.architecture}")
        print(f"Required layers: {len(self.required_layers)}")
        print(f"Available layers: {len(self.available_layers)}")
        print(f"Missing layers: {len(self.missing_layers)}")

        print("\n" + "-" * 70)
        print("AVAILABLE LAYERS:")
        print("-" * 70)
        for layer_name in sorted(self.available_layers):
            layer_info = self.layer_registry.get_layer(layer_name)
            status_icon = {
                "builtin": "📦",
                "custom": "🔧",
                "generated": "✨",
            }.get(layer_info.type, "❓")

            print(f"{status_icon} {layer_name:<20} [{layer_info.type}] - {layer_info.description}")

        if self.missing_layers:
            print("\n" + "-" * 70)
            print("MISSING LAYERS (will generate with NOOA):")
            print("-" * 70)
            for layer_name in sorted(self.missing_layers):
                print(f"⚠️  {layer_name}")

        print("\n" + "=" * 70)

        return {
            "required_layers": self.required_layers,
            "available_layers": self.available_layers,
            "missing_layers": self.missing_layers,
            "analysis_complete": True,
        }

    def generate_all(self, include_missing: bool = False) -> Dict[str, str]:
        """
        Generate all output files.

        Args:
            include_missing: If True, return info about missing layers
                            If False, generate only with available layers

        Returns:
            Dictionary with generated files
        """
        # Generate base files using standard generator
        files = self.base_generator.generate_all()

        # Add artifact layer information if there are missing layers
        if self.missing_layers:
            if include_missing:
                files["_MISSING_LAYERS.txt"] = self._generate_missing_layers_report()

            # Add note about NOOA generation
            files["_LAYER_GENERATION_NOTES.md"] = self._generate_layer_generation_notes()

        return files

    def generate_with_layer_detection(self, output_dir: Path) -> Dict[str, any]:
        """
        Full generation pipeline with layer detection.

        1. Analyze layers
        2. Generate C++ code
        3. Create artifact layer stubs
        4. Generate build integration
        5. Return comprehensive report

        Returns:
            Complete generation report
        """
        print("\n" + "=" * 70)
        print("ENHANCED C++ CODE GENERATION WITH LAYER DETECTION")
        print("=" * 70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Analyze
        analysis = self.analyze()

        # Step 2: Generate base files
        print("\n" + "-" * 70)
        print("Generating C++ code...")
        print("-" * 70)

        files = self.generate_all(include_missing=True)

        # Save all files
        for filename, content in files.items():
            filepath = output_dir / filename
            filepath.write_text(content, encoding="utf-8")

            if not filename.startswith("_"):
                print(f"  ✓ {filename}")

        # Step 3: Generate artifact layer stubs (if missing layers exist)
        if self.missing_layers:
            print("\n" + "-" * 70)
            print("Generating artifact layer stubs (for NOOA generation)...")
            print("-" * 70)

            artifact_stubs = self._generate_artifact_stubs()
            for stub_filename, stub_content in artifact_stubs.items():
                stub_path = self.artifact_dir / stub_filename
                stub_path.parent.mkdir(parents=True, exist_ok=True)
                stub_path.write_text(stub_content, encoding="utf-8")
                print(f"  ✓ {stub_filename}")

        # Step 4: Generate build integration
        print("\n" + "-" * 70)
        print("Generating build integration...")
        print("-" * 70)

        build_snippet = self._generate_build_integration()
        build_integration_file = output_dir / "_BUILD_INTEGRATION.md"
        build_integration_file.write_text(build_snippet, encoding="utf-8")
        print(f"  ✓ _BUILD_INTEGRATION.md")

        # Step 5: Generate report
        print("\n" + "-" * 70)
        print("GENERATION COMPLETE")
        print("-" * 70)

        report = {
            "success": True,
            "architecture": self.architecture,
            "analysis": analysis,
            "files_generated": len(files),
            "output_dir": str(output_dir),
            "artifact_dir": str(self.artifact_dir),
            "missing_layers_requiring_nooa": self.missing_layers,
            "next_steps": self._get_next_steps(),
        }

        self._print_report(report)

        return report

    def _determine_required_layers(self) -> List[str]:
        """Determine which layers are required for the model."""
        return [
            "embedding",
            "fully_connected",
            "rms_norm",
            "reshaped_rms_norm",
            "mha_core",
            "swiglu",
            "lm_head",
            "addition",
            "activation",
            "kv_cache_placeholders",
        ]

    def _generate_artifact_stubs(self) -> Dict[str, str]:
        """Generate stub files for missing layers to be implemented."""
        stubs = {}

        for layer_name in self.missing_layers:
            # Header stub
            class_name = self._to_class_name(layer_name)
            guard = f"__{layer_name.upper()}_H__"

            header = f'''/**
 * {class_name} Layer
 *
 * TODO: Implement this layer
 *
 * Use NOOA to generate the full implementation:
 * 1. Run NOOA layer generator with layer specifications
 * 2. Review generated code
 * 3. Integrate into build system
 */

#ifndef {guard}
#define {guard}

#include <layer.h>

namespace causallm {{

class {class_name} : public Layer {{
 public:
  // TODO: Implement layer
}};

}}  // namespace causallm

#endif  /* {guard} */
'''
            stubs[f"{layer_name}.h"] = header

            # Source stub
            source = f'''/**
 * {class_name} Implementation
 *
 * TODO: Generated by NOOA
 */

#include "{layer_name}.h"

namespace causallm {{

// TODO: Implement methods

}}  // namespace causallm
'''
            stubs[f"{layer_name}.cpp"] = source

        return stubs

    def _generate_build_integration(self) -> str:
        """Generate build integration instructions."""
        integration = """# Build Integration Instructions

## Layer Status

### Available Layers
These layers are available in the registry and will be used:
"""

        for layer_name in sorted(self.available_layers):
            layer_info = self.layer_registry.get_layer(layer_name)
            integration += f"- `{layer_name}` ({layer_info.type})\n"

        if self.missing_layers:
            integration += """
### Missing Layers (To Be Generated)
These layers will be generated using NOOA:
"""
            for layer_name in sorted(self.missing_layers):
                integration += f"- `{layer_name}`\n"

        integration += f"""
## Steps

### 1. Check Available Layers
All available layers are documented in layer_registry.py

### 2. Generate Missing Layers (if any)
If there are missing layers:

```bash
# Run NOOA layer generator
python3 nooa_layer_generator.py --layers {' '.join(self.missing_layers)}
```

Generated layers will be stored in:
```
{self.artifact_dir}
```

### 3. Build Integration
Add to meson.build:
```meson
# Add artifact layers if they exist
if fs.exists('{self.artifact_dir}')
  subdir('{self.artifact_dir}')
endif
```

### 4. Compile
```bash
meson build -Denable-transformer=true
ninja -C build Applications/CausalLM/nntr_causallm
```

## Layer Usage Pattern

All layers follow this pattern:

```cpp
LayerHandle layer = createLayer(
    "layer_type",
    {{
        withKey("name", "layer_name"),
        withKey("property1", "value1"),
        withKey("property2", "value2"),
    }}
);
Tensor output = layer(input);
```

## References

- Layer Registry: layer_registry.py
- Available Patterns: Instructions in cppcodegenerationinstructions
- Examples: Applications/CausalLM/models/qwen3/qwen3_causallm.cpp
"""
        return integration

    def _generate_missing_layers_report(self) -> str:
        """Generate detailed report on missing layers."""
        report = """# Missing Layers Report

The following layers are required but not found in the registry:

"""
        for layer_name in sorted(self.missing_layers):
            report += f"- {layer_name}\n"

        report += """
## Action Required

To use these layers, they must be generated or added to the registry.

### Option 1: Use NOOA Layer Generator
```bash
python3 nooa_layer_generator.py \\
    --layer-name {layer_name} \\
    --output-dir {artifact_dir}
```

### Option 2: Implement Manually
1. Copy reference implementation from Applications/CausalLM/layers/
2. Modify for your specific layer
3. Add to artifact directory
4. Update layer_registry.py

### Option 3: Add to Registry
If the layer exists somewhere in the codebase:
1. Locate the implementation
2. Add to layer_registry.py CUSTOM_LAYERS
3. Re-run generation

## References

See generated layer stubs in artifact/ directory for templates.
"""
        return report

    def _generate_layer_generation_notes(self) -> str:
        """Generate notes about layer generation process."""
        return f"""# Layer Generation Notes

## Missing Layers
{len(self.missing_layers)} layers need to be generated:
{', '.join(sorted(self.missing_layers))}

## Generation Options

### 1. NOOA-Based Generation (Recommended)
Use the NOOA agent to generate missing layers with AI assistance:

```python
from agents.nooa_layer_generator import NOOALayerGeneratorAgent

agent = NOOALayerGeneratorAgent()
generated = agent.generate_layer("layer_name", {{
    "description": "...",
    "inputs": [...],
    "outputs": [...],
    "properties": {{...}},
}}))
```

### 2. Artifact Layer Directory
Generated layers are stored in:
```
{self.artifact_dir}
```

### 3. Build Integration
The build system will automatically include artifact layers.

## Next Steps
1. Review artifact layer stubs
2. Run NOOA generation
3. Review generated code
4. Integrate into build
5. Test compilation
"""

    def _get_next_steps(self) -> List[str]:
        """Generate next steps based on generation results."""
        steps = [
            "Review generated C++ files",
            "Check configuration files",
            "Verify layer usage patterns",
        ]

        if self.missing_layers:
            steps.append(f"Generate {len(self.missing_layers)} missing layers using NOOA")
            steps.append("Add generated layers to artifact directory")
            steps.append("Integrate artifact layers into build system")

        steps.extend([
            "Run build: meson build -Denable-transformer=true",
            "Compile: ninja -C build Applications/CausalLM/nntr_causallm",
            "Run tests and validate",
        ])

        return steps

    def _print_report(self, report: Dict):
        """Print comprehensive generation report."""
        print(f"\n✅ Generation Complete!")
        print(f"   Architecture: {report['architecture']}")
        print(f"   Files: {report['files_generated']}")
        print(f"   Output: {report['output_dir']}")

        if report['missing_layers_requiring_nooa']:
            print(f"\n⚠️  Missing Layers ({len(report['missing_layers_requiring_nooa'])}):")
            for layer in report['missing_layers_requiring_nooa']:
                print(f"   - {layer}")
            print(f"\n   Generated stubs in: {report['artifact_dir']}")

        print(f"\n📝 Next Steps:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"   {i}. {step}")

    @staticmethod
    def _extract_slug(architecture: str) -> str:
        """Extract model slug from architecture name."""
        slug = architecture.replace("ForCausalLM", "").replace("CausalLM", "").replace("Model", "")
        if slug and len(slug) > 0:
            return slug[0].upper() + slug[1:] if len(slug) > 1 else slug.upper()
        return architecture.lower()

    @staticmethod
    def _to_class_name(layer_name: str) -> str:
        """Convert layer_name to ClassName."""
        parts = layer_name.split("_")
        return "".join(p.capitalize() for p in parts) + "Layer"
