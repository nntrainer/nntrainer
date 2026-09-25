"""
Artifact Layer Manager

Manages generated/artifact layers:
- Stores generated layer implementations
- Integrates into build system
- Tracks layer dependencies
- Manages meson.build integration
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict


@dataclass
class ArtifactLayer:
    """Information about an artifact layer."""
    name: str
    header_file: Path
    source_file: Path
    generated_at: str
    description: str
    dependencies: List[str]  # Other layers this depends on


class ArtifactLayerManager:
    """Manages artifact layer directory and build integration."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.layers: Dict[str, ArtifactLayer] = {}
        self.metadata_file = self.artifact_dir / "manifest.json"

        # Load existing manifest
        self._load_manifest()

    def add_layer(
        self,
        name: str,
        header_content: str,
        source_content: str,
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """Add a generated layer to artifacts."""
        dependencies = dependencies or []

        # Write files
        header_file = self.artifact_dir / f"{name}.h"
        source_file = self.artifact_dir / f"{name}.cpp"

        header_file.write_text(header_content, encoding="utf-8")
        source_file.write_text(source_content, encoding="utf-8")

        # Track in memory
        from datetime import datetime
        self.layers[name] = ArtifactLayer(
            name=name,
            header_file=header_file,
            source_file=source_file,
            generated_at=datetime.now().isoformat(),
            description=description,
            dependencies=dependencies,
        )

        # Save manifest
        self._save_manifest()

        return {
            "header": header_file,
            "source": source_file,
        }

    def get_layer(self, name: str) -> Optional[ArtifactLayer]:
        """Get an artifact layer by name."""
        return self.layers.get(name)

    def list_layers(self) -> List[str]:
        """List all artifact layer names."""
        return sorted(self.layers.keys())

    def get_layer_files(self, name: str) -> Optional[Dict[str, Path]]:
        """Get header and source files for a layer."""
        layer = self.get_layer(name)
        if not layer:
            return None

        return {
            "header": layer.header_file,
            "source": layer.source_file,
        }

    def get_all_sources(self) -> List[Path]:
        """Get all artifact layer source files."""
        return [layer.source_file for layer in self.layers.values()]

    def get_all_headers(self) -> List[Path]:
        """Get all artifact layer header files."""
        return [layer.header_file for layer in self.layers.values()]

    def generate_meson_snippet(self) -> str:
        """Generate meson.build snippet for artifact layers."""
        if not self.layers:
            return "# No artifact layers"

        sources = [str(layer.source_file) for layer in self.layers.values()]
        headers = [str(layer.header_file) for layer in self.layers.values()]

        snippet = """# ====================================================================
# ARTIFACT LAYERS (AI-Generated)
# ====================================================================

artifact_layer_sources = [
"""
        for source in sources:
            snippet += f'  "{source}",\n'

        snippet += """]

artifact_layer_headers = [
"""
        for header in headers:
            snippet += f'  "{header}",\n'

        snippet += """]

# Add artifact layer sources to model
model_sources += artifact_layer_sources

# Install artifact layer headers
install_headers(artifact_layer_headers,
  subdir: join_paths('nntrainer', 'causallm', 'layers'),
)

"""
        return snippet

    def generate_build_integration(self, model_meson_dir: Path) -> str:
        """Generate integration commands for existing meson.build."""
        if not self.layers:
            return "# No artifact layers to integrate"

        return f"""
# Integrate artifact layers
subdir('{self.artifact_dir.relative_to(model_meson_dir)}')
"""

    def get_dependencies(self, layer_name: str) -> Set[str]:
        """Get all dependencies (recursive) for a layer."""
        layer = self.get_layer(layer_name)
        if not layer:
            return set()

        deps = set(layer.dependencies)
        for dep in layer.dependencies:
            deps.update(self.get_dependencies(dep))

        return deps

    def validate_dependencies(self) -> Dict[str, any]:
        """Validate that all layer dependencies are available."""
        issues = []
        all_layer_names = set(self.layers.keys())

        for layer_name, layer in self.layers.items():
            for dep in layer.dependencies:
                if dep not in all_layer_names:
                    # Check if it's a standard layer
                    from .layer_registry import get_layer_registry
                    registry = get_layer_registry()
                    if not registry.get_layer(dep):
                        issues.append(
                            f"Layer '{layer_name}' depends on '{dep}' which is not available"
                        )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }

    def export_manifest(self, filepath: Path) -> None:
        """Export manifest to JSON file."""
        manifest = {
            "layers": {
                name: {
                    "name": layer.name,
                    "header_file": str(layer.header_file),
                    "source_file": str(layer.source_file),
                    "generated_at": layer.generated_at,
                    "description": layer.description,
                    "dependencies": layer.dependencies,
                }
                for name, layer in self.layers.items()
            }
        }

        filepath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _save_manifest(self) -> None:
        """Save layers manifest to metadata file."""
        self.export_manifest(self.metadata_file)

    def _load_manifest(self) -> None:
        """Load layers manifest from metadata file."""
        if not self.metadata_file.exists():
            return

        try:
            manifest = json.loads(self.metadata_file.read_text())
            for name, layer_data in manifest.get("layers", {}).items():
                self.layers[name] = ArtifactLayer(
                    name=layer_data["name"],
                    header_file=Path(layer_data["header_file"]),
                    source_file=Path(layer_data["source_file"]),
                    generated_at=layer_data["generated_at"],
                    description=layer_data["description"],
                    dependencies=layer_data.get("dependencies", []),
                )
        except Exception as e:
            print(f"Warning: Failed to load artifact manifest: {e}")

    def print_summary(self):
        """Print summary of artifact layers."""
        print("\n" + "=" * 70)
        print("ARTIFACT LAYERS SUMMARY")
        print("=" * 70)

        if not self.layers:
            print("No artifact layers")
            print("=" * 70)
            return

        print(f"\nTotal layers: {len(self.layers)}")
        print(f"Artifact directory: {self.artifact_dir}\n")

        for name, layer in sorted(self.layers.items()):
            print(f"📦 {name}")
            print(f"   Description: {layer.description}")
            print(f"   Header: {layer.header_file.name}")
            print(f"   Source: {layer.source_file.name}")
            if layer.dependencies:
                print(f"   Dependencies: {', '.join(layer.dependencies)}")
            print()

        # Validate
        validation = self.validate_dependencies()
        print("-" * 70)
        if validation["valid"]:
            print("✓ All dependencies valid")
        else:
            print("✗ Dependency issues found:")
            for issue in validation["issues"]:
                print(f"  - {issue}")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ArtifactLayerManager(Path(tmpdir) / "artifacts")

        # Add a sample layer
        header = """
#ifndef __SAMPLE_H__
#define __SAMPLE_H__
// Sample header
#endif
"""
        source = """
// Sample source
"""

        manager.add_layer(
            "sample_layer",
            header,
            source,
            description="Sample layer for testing",
            dependencies=["rms_norm"],
        )

        print("\nGenerated meson snippet:")
        print(manager.generate_meson_snippet())

        manager.print_summary()
