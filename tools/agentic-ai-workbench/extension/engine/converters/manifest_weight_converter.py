"""Convert verified Hugging Face safetensors using a generated tensor map.

This tool is copied beside every generated CausalLM component.  It intentionally
has no model-family conditionals: all layout decisions live in tensor_map.json,
where they are inspectable and validated before bytes are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _open_tensors(model_path: Path):
    from safetensors import safe_open

    readers = []
    index = {}
    for path in sorted(model_path.rglob("*.safetensors")):
        reader = safe_open(str(path), framework="pt")
        readers.append(reader)
        for key in reader.keys():
            if key in index:
                raise ValueError(f"duplicate tensor in checkpoint shards: {key}")
            index[key] = reader
    if not index:
        raise ValueError("only safetensors checkpoints are supported by the generated converter")
    return readers, index


def _apply_operations(tensor, entry, tensors):
    import torch

    value = tensor
    for operation in entry.get("operations", []):
        op = operation.get("op") if isinstance(operation, dict) else operation
        if op in ("identity", "tied_with_embedding"):
            continue
        if op == "cast_fp16_to_fp32":
            value = value.to(torch.float32)
        elif op == "transpose":
            axes = operation.get("axes", [1, 0]) if isinstance(operation, dict) else [1, 0]
            value = value.permute(*axes)
        elif op == "split":
            axis = operation.get("axis", 0)
            sections = operation.get("sections")
            index = operation.get("index")
            if not sections or index is None:
                raise ValueError(f"split entry for {entry['source']} lacks sections/index")
            value = torch.split(value, sections, dim=axis)[index]
        elif op == "concat":
            axis = operation.get("axis", 0)
            extra = operation.get("sources", [])
            value = torch.cat([value, *tensors(extra)], dim=axis)
        else:
            raise ValueError(f"unsupported tensor-map operation: {op}")
    return value.contiguous()


def convert(model_path: str, tensor_map_path: str, output_path: str) -> None:
    import torch

    tensor_map = json.loads(Path(tensor_map_path).read_text(encoding="utf-8"))
    readers, tensors_by_name = _open_tensors(Path(model_path))
    del readers  # safe_open objects remain referenced by tensor index values

    def resolve(names):
        missing = [name for name in names if name not in tensors_by_name]
        if missing:
            raise ValueError(f"checkpoint is missing tensor(s): {', '.join(missing)}")
        return [tensors_by_name[name].get_tensor(name) for name in names]

    expected_bytes = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for entry in tensor_map.get("entries", []):
            if entry.get("tied") or not entry.get("write", True):
                continue
            source = entry["source"]
            tensor = resolve([source])[0]
            source_shape = list(tensor.shape)
            if source_shape != entry.get("source_shape"):
                raise ValueError(f"{source}: expected source shape {entry.get('source_shape')}, got {source_shape}")
            value = _apply_operations(tensor, entry, resolve)
            if list(value.shape) != entry.get("target_shape"):
                raise ValueError(f"{source}: expected target shape {entry.get('target_shape')}, got {list(value.shape)}")
            value = value.to(torch.float32)
            data = value.detach().cpu().numpy().tobytes(order="C")
            handle.write(data)
            expected_bytes += len(data)
    actual_bytes = output.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"output size mismatch: expected {expected_bytes}, wrote {actual_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-map", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--data-type", default="float32")  # CausalLM CLI compatibility
    args = parser.parse_args()
    convert(args.model_path, args.tensor_map, args.output_name)


if __name__ == "__main__":
    main()
