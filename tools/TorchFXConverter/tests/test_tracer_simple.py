"""Test the tracer with a simple model to verify basic functionality."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from tracer import Tracer


class SimpleModel(nn.Module):
    """Simple model: Linear -> ReLU -> Linear"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ResidualModel(nn.Module):
    """Model with a residual connection (common in transformers)."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.fc1 = nn.Linear(32, 32)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(32, 32)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x + residual  # residual connection
        return x


def test_simple_model():
    print("=" * 60)
    print("TEST 1: Simple Linear -> ReLU -> Linear")
    print("=" * 60)

    model = SimpleModel()
    model.eval()

    tracer = Tracer(model)
    x = torch.randn(1, 32)

    with tracer:
        with torch.no_grad():
            out = model(x)

    print("\nGraph nodes:")
    tracer.graph.print_tabular()
    tracer.print_graph_summary()

    # Verify expected nodes
    nodes = list(tracer.graph.nodes)
    ops = [n.op for n in nodes]
    assert "placeholder" in ops, "Missing placeholder node"
    assert "output" in ops, "Missing output node"

    call_modules = [n for n in nodes if n.op == "call_module"]
    module_names = [n.target for n in call_modules]
    print(f"Leaf modules found: {module_names}")

    assert "fc1" in module_names, "Missing fc1"
    assert "relu" in module_names, "Missing relu"
    assert "fc2" in module_names, "Missing fc2"
    print("PASSED!\n")


def test_residual_model():
    print("=" * 60)
    print("TEST 2: Model with Residual Connection")
    print("=" * 60)

    model = ResidualModel()
    model.eval()

    tracer = Tracer(model)
    x = torch.randn(1, 32)

    with tracer:
        with torch.no_grad():
            out = model(x)

    print("\nGraph nodes:")
    tracer.graph.print_tabular()
    tracer.print_graph_summary()

    nodes = list(tracer.graph.nodes)
    call_modules = [n for n in nodes if n.op == "call_module"]
    module_names = [n.target for n in call_modules]
    print(f"Leaf modules found: {module_names}")

    assert "norm" in module_names, "Missing norm"
    assert "fc1" in module_names, "Missing fc1"
    assert "act" in module_names, "Missing act"
    assert "fc2" in module_names, "Missing fc2"

    # Check that there's an add operation (residual)
    call_functions = [n for n in nodes if n.op == "call_function"]
    func_names = [str(n.target) for n in call_functions]
    print(f"Functions found: {func_names}")

    # The residual add should appear as a call_function or call_method
    has_add = any("add" in str(n.target) for n in nodes
                   if n.op in ("call_function", "call_method"))
    assert has_add, "Missing residual add operation"
    print("PASSED!\n")


if __name__ == "__main__":
    test_simple_model()
    test_residual_model()
    print("All tracer tests PASSED!")
