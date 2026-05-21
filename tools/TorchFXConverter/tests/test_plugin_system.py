"""Tests for the Plugin System (Tier 3: LayerPluggable support).

Tests:
  1. PluginRegistry: register, lookup, map_module
  2. PluginRegistry.from_config: JSON config loading
  3. Plugin codegen: C++ header, source, meson.build generation
  4. E2E: custom module conversion through full pipeline
  5. E2E: config-file-driven custom module conversion
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from plugin_registry import (
    PluginRegistry, CustomLayerSpec, get_global_registry,
    register_custom_layer, _global_registry,
)
from plugin_codegen import (
    generate_header, generate_source, generate_meson_build,
    generate_plugin_code, _to_class_name, _to_header_guard,
)


# ============================================================================
# Test helpers: custom PyTorch modules
# ============================================================================

class MyPowLayer(nn.Module):
    """Custom element-wise power layer."""
    def __init__(self, exponent=2):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        return x ** self.exponent


class MyScaleLayer(nn.Module):
    """Custom scaling layer with trainable weight."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


class MyActivation(nn.Module):
    """Custom activation for testing string-based matching."""
    def forward(self, x):
        return torch.clamp(x, min=0, max=6)  # ReLU6-like


# ============================================================================
# Unit tests: PluginRegistry
# ============================================================================

def test_register_by_class():
    """Register and lookup by PyTorch class."""
    registry = PluginRegistry()
    spec = CustomLayerSpec(
        nntrainer_type="custom_pow",
        property_mapper=lambda m: {"exponent": m.exponent},
    )
    registry.register(MyPowLayer, spec)

    mod = MyPowLayer(exponent=3)
    result = registry.lookup(mod)
    assert result is not None
    assert result.nntrainer_type == "custom_pow"

    # Non-matching module returns None
    assert registry.lookup(nn.Linear(4, 4)) is None
    print("  PASS: register/lookup by class")


def test_register_by_name():
    """Register and lookup by class name string."""
    registry = PluginRegistry()
    spec = CustomLayerSpec(nntrainer_type="my_activation")
    registry.register("MyActivation", spec)

    mod = MyActivation()
    result = registry.lookup(mod)
    assert result is not None
    assert result.nntrainer_type == "my_activation"
    print("  PASS: register/lookup by class name")


def test_register_by_predicate():
    """Register and lookup by callable predicate."""
    registry = PluginRegistry()
    spec = CustomLayerSpec(nntrainer_type="pow_variant")
    registry.register(lambda m: hasattr(m, 'exponent'), spec)

    assert registry.lookup(MyPowLayer(2)) is not None
    assert registry.lookup(nn.Linear(4, 4)) is None
    print("  PASS: register/lookup by predicate")


def test_register_simple():
    """Convenience register_simple method."""
    registry = PluginRegistry()
    registry.register_simple(MyPowLayer, "custom_pow",
                             property_mapper=lambda m: {"exponent": m.exponent})
    assert len(registry) == 1
    assert registry.registered_types == ["custom_pow"]
    print("  PASS: register_simple")


def test_map_module():
    """map_module returns correct NNTrainerLayerDef."""
    registry = PluginRegistry()
    spec = CustomLayerSpec(
        nntrainer_type="custom_pow",
        property_mapper=lambda m: {"exponent": m.exponent},
        has_weight=False,
    )
    registry.register(MyPowLayer, spec)

    mod = MyPowLayer(exponent=3)
    layer_def = registry.map_module(mod, "model.pow_layer", ["input_0"])

    assert layer_def is not None
    assert layer_def.layer_type == "custom_pow"
    assert layer_def.name == "model_pow_layer"
    assert layer_def.properties == {"exponent": 3}
    assert layer_def.input_layers == ["input_0"]
    assert not layer_def.has_weight
    print("  PASS: map_module produces correct LayerDef")


def test_map_module_with_weights():
    """map_module with weight key templates."""
    registry = PluginRegistry()
    spec = CustomLayerSpec(
        nntrainer_type="my_scale",
        has_weight=True,
        weight_keys={"weight": "{name}.weight"},
    )
    registry.register(MyScaleLayer, spec)

    mod = MyScaleLayer(dim=16)
    layer_def = registry.map_module(mod, "model.scale", ["fc_out"])

    assert layer_def.has_weight
    assert layer_def.weight_hf_key == "model.scale.weight"
    print("  PASS: map_module with weight keys")


def test_from_config():
    """Load registry from JSON config file."""
    config = {
        "custom_layers": [
            {
                "match_class_name": "MyPowLayer",
                "nntrainer_type": "custom_pow",
                "properties": {"exponent": 2},
                "has_weight": False,
                "description": "Element-wise power",
            },
            {
                "match_class_name": "MyScaleLayer",
                "nntrainer_type": "my_scale",
                "has_weight": True,
                "has_bias": False,
                "weight_keys": {"weight": "{name}.weight"},
            },
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        registry = PluginRegistry.from_config(config_path)
        assert len(registry) == 2
        assert "custom_pow" in registry.registered_types
        assert "my_scale" in registry.registered_types

        # Verify static property mapper works
        mod = MyPowLayer(exponent=5)
        layer_def = registry.map_module(mod, "pow", ["x"])
        assert layer_def is not None
        assert layer_def.properties == {"exponent": 2}  # from config, not module
        print("  PASS: from_config loads JSON correctly")
    finally:
        os.unlink(config_path)


# ============================================================================
# Unit tests: Plugin codegen
# ============================================================================

def test_class_name_generation():
    """layer_type -> CamelCase class name."""
    assert _to_class_name("custom_pow") == "CustomPowLayer"
    assert _to_class_name("my_norm") == "MyNormLayer"
    assert _to_class_name("rms_norm") == "RmsNormLayer"
    print("  PASS: class name generation")


def test_header_guard_generation():
    """class name -> header guard macro."""
    assert _to_header_guard("CustomPowLayer") == "__CUSTOM_POW_LAYER_H__"
    print("  PASS: header guard generation")


def test_generate_header():
    """Generate C++ header for custom layer."""
    header = generate_header(
        layer_type="custom_pow",
        properties={"exponent": "float"},
    )
    assert "class CustomPowLayer" in header
    assert 'type = "custom_pow"' in header
    assert "float exponent" in header
    assert "void finalize" in header
    assert "void forwarding" in header
    assert "void calcDerivative" in header
    assert "#ifndef __CUSTOM_POW_LAYER_H__" in header
    print("  PASS: generate_header")


def test_generate_source():
    """Generate C++ source for custom layer."""
    source = generate_source(
        layer_type="custom_pow",
        properties={"exponent": "float"},
    )
    assert "CustomPowLayer::setProperty" in source
    assert "CustomPowLayer::finalize" in source
    assert "CustomPowLayer::forwarding" in source
    assert "CustomPowLayer::calcDerivative" in source
    assert '#ifdef PLUGGABLE' in source
    assert "ml_train_layer_pluggable" in source
    assert 'key == "exponent"' in source
    assert "std::stof(val)" in source
    print("  PASS: generate_source")


def test_generate_source_no_props():
    """Generate C++ source with no custom properties."""
    source = generate_source(layer_type="my_identity")
    assert "MyIdentityLayer::setProperty" in source
    assert "(void)values;" in source
    print("  PASS: generate_source (no properties)")


def test_generate_source_with_weight():
    """Generate C++ source with weight support."""
    source = generate_source(
        layer_type="my_norm",
        has_weight=True,
    )
    assert "requestWeight" in source
    print("  PASS: generate_source (with weight)")


def test_generate_meson_build():
    """Generate meson.build for plugin library."""
    meson = generate_meson_build(layer_type="custom_pow")
    assert "custom_pow_layer" in meson
    assert "shared_library" in meson
    assert "PLUGGABLE" in meson
    assert "nntrainer_dep" in meson
    print("  PASS: generate_meson_build")


def test_generate_plugin_code_to_dir():
    """Generate complete plugin code to directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = generate_plugin_code(
            layer_type="custom_pow",
            properties={"exponent": "float", "base": "int"},
            has_weight=False,
            output_dir=tmpdir,
        )
        assert len(files) == 3
        for fname, fpath in files.items():
            assert os.path.isfile(fpath), f"File not found: {fpath}"
            with open(fpath) as f:
                content = f.read()
            assert len(content) > 0

        # Verify header exists with correct content
        header_path = files["custom_pow_layer.h"]
        with open(header_path) as f:
            header = f.read()
        assert "float exponent" in header
        assert "int base" in header

        print("  PASS: generate_plugin_code writes files")


# ============================================================================
# E2E tests: full pipeline with custom plugins
# ============================================================================

def test_custom_module_e2e():
    """Custom module converted via plugin registry through full pipeline."""
    from decomposer import AdaptiveConverter

    # Reset global registry
    _global_registry._entries.clear()

    # Create a model with custom module
    class ModelWithCustom(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.pow = MyPowLayer(exponent=3)
            self.fc2 = nn.Linear(32, 8)

        def forward(self, x):
            h = self.fc1(x)
            h = self.pow(h)
            return self.fc2(h)

    model = ModelWithCustom()
    model.eval()

    # Register custom layer in plugin registry
    registry = PluginRegistry()
    registry.register(MyPowLayer, CustomLayerSpec(
        nntrainer_type="custom_pow",
        property_mapper=lambda m: {"exponent": m.exponent},
    ))

    converter = AdaptiveConverter(model, training=False,
                                  plugin_registry=registry)
    result = converter.convert({"x": torch.randn(1, 16)})

    result.summary()

    # The custom_pow layer should be present
    custom = [l for l in result.layers if l.layer_type == "custom_pow"]
    assert len(custom) == 1, f"Expected 1 custom_pow, got {len(custom)}"
    assert custom[0].properties["exponent"] == 3

    fc = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc) == 2

    # No unknowns
    assert len(result.unknown_layers) == 0, \
        f"Has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    print("  PASS: Custom module E2E conversion")
    print(f"    custom_pow: {len(custom)} (exponent={custom[0].properties['exponent']})")
    print(f"    fully_connected: {len(fc)}")
    print(f"    Total layers: {len(result.layers)}")

    # Cleanup
    _global_registry._entries.clear()


def test_config_driven_e2e():
    """Custom module conversion driven by JSON config file."""
    from decomposer import AdaptiveConverter

    _global_registry._entries.clear()

    # Write config
    config = {
        "custom_layers": [
            {
                "match_class_name": "MyPowLayer",
                "nntrainer_type": "custom_pow",
                "properties": {"exponent": 2},
            },
            {
                "match_class_name": "MyActivation",
                "nntrainer_type": "custom_relu6",
            },
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        registry = PluginRegistry.from_config(config_path)

        class ConfigModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 16)
                self.act = MyActivation()
                self.pow = MyPowLayer(exponent=5)

            def forward(self, x):
                return self.pow(self.act(self.fc(x)))

        model = ConfigModel()
        model.eval()

        converter = AdaptiveConverter(model, training=False,
                                      plugin_registry=registry)
        result = converter.convert({"x": torch.randn(1, 16)})

        result.summary()

        assert len(result.unknown_layers) == 0, \
            f"Has unknowns: {[l.layer_type for l in result.unknown_layers]}"

        custom_pow = [l for l in result.layers if l.layer_type == "custom_pow"]
        custom_act = [l for l in result.layers if l.layer_type == "custom_relu6"]
        assert len(custom_pow) == 1
        assert len(custom_act) == 1

        print("  PASS: Config-driven E2E conversion")
        print(f"    custom_pow: {len(custom_pow)}")
        print(f"    custom_relu6: {len(custom_act)}")

    finally:
        os.unlink(config_path)
        _global_registry._entries.clear()


def test_codegen_roundtrip():
    """Generate plugin code, verify it references the correct type."""
    from decomposer import AdaptiveConverter

    _global_registry._entries.clear()

    # Register and convert
    registry = PluginRegistry()
    registry.register(MyScaleLayer, CustomLayerSpec(
        nntrainer_type="my_scale",
        has_weight=True,
        weight_keys={"weight": "{name}.weight"},
    ))

    class ScaleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)
            self.scale = MyScaleLayer(16)

        def forward(self, x):
            return self.scale(self.fc(x))

    model = ScaleModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False,
                                  plugin_registry=registry)
    result = converter.convert({"x": torch.randn(1, 16)})

    scale = [l for l in result.layers if l.layer_type == "my_scale"]
    assert len(scale) == 1
    assert scale[0].has_weight
    assert scale[0].weight_hf_key == "scale.weight"

    # Generate C++ plugin code for the custom layer
    files = generate_plugin_code(
        layer_type="my_scale",
        properties={},
        has_weight=True,
    )
    header = files["my_scale_layer.h"]
    source = files["my_scale_layer.cpp"]

    assert 'type = "my_scale"' in header
    assert "requestWeight" in source
    assert "ml_train_layer_pluggable" in source

    print("  PASS: Codegen roundtrip (convert + generate C++)")

    _global_registry._entries.clear()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TIER 3 PLUGIN SYSTEM TESTS")
    print("=" * 70)

    print("\n--- Plugin Registry Unit Tests ---")
    test_register_by_class()
    test_register_by_name()
    test_register_by_predicate()
    test_register_simple()
    test_map_module()
    test_map_module_with_weights()
    test_from_config()

    print("\n--- Plugin Codegen Unit Tests ---")
    test_class_name_generation()
    test_header_guard_generation()
    test_generate_header()
    test_generate_source()
    test_generate_source_no_props()
    test_generate_source_with_weight()
    test_generate_meson_build()
    test_generate_plugin_code_to_dir()

    print("\n--- Plugin System E2E Tests ---")
    test_custom_module_e2e()
    test_config_driven_e2e()
    test_codegen_roundtrip()

    print("\n" + "=" * 70)
    print("ALL TIER 3 PLUGIN SYSTEM TESTS PASSED!")
    print("=" * 70)
