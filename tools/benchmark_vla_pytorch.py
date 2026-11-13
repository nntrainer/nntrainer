#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the same preprocessed VLA requests as the NNTrainer drivers.

Run under validate_vla_ports.py to collect process RSS and sampled device
memory. Each fresh process performs the requested number of timed requests.
CUDA uses FP32 vision, FP16 language weights, and FP32 pi0.5 expert weights.
The pi0.5 embedding table stays on CPU, as in the NNTrainer implementation.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def compare(label, actual, reference, atol, rtol=0):
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if actual.shape != reference.shape or actual.size == 0:
        raise RuntimeError(f"{label}: shape mismatch")
    error = np.abs(actual-reference)
    bad = (~np.isfinite(actual) | ~np.isfinite(reference) |
           (error > atol + rtol*np.abs(reference)))
    print(f"[numeric] {label} count={actual.size} max_abs={error.max():.8g} "
          f"rmse={np.sqrt(np.mean(error.astype(np.float64)**2)):.8g} "
          f"atol={atol} rtol={rtol} failed={bad.sum()}", flush=True)
    if bad.any():
        raise RuntimeError(f"{label}: numeric mismatch")


def paligemma(args):
    from transformers import PaliGemmaForConditionalGeneration
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.weights, dtype=torch.float32, attn_implementation=args.attention)
    if args.device == "cuda":
        model.model.language_model.half()
        model.lm_head.half()
    model.to(args.device).eval()
    ref = args.reference
    ids = torch.tensor([json.loads((ref/"input_ids.json").read_text())],
                       dtype=torch.long, device=args.device)
    types = torch.tensor([json.loads((ref/"token_type_ids.json").read_text())],
                         dtype=torch.long, device=args.device)
    pixels = torch.from_numpy(np.fromfile(ref/"pixel_values.bin", np.float32)
                              .reshape(1, 3, 224, 224)).to(args.device)
    expected = json.loads((ref/"mm_tokens.json").read_text())

    def run():
        return model.generate(input_ids=ids, token_type_ids=types,
                              pixel_values=pixels,
                              attention_mask=torch.ones_like(ids),
                              max_new_tokens=len(expected), do_sample=False,
                              logits_to_keep=1, use_cache=True)

    def validate(output):
        generated = output[0, ids.shape[1]:].cpu().tolist()
        print(f"[numeric] greedy_tokens count={len(generated)} "
              f"exact_match={generated == expected}", flush=True)
        if generated != expected:
            raise RuntimeError(f"token mismatch: {generated} != {expected}")

    return model, run, validate


def pi05(args):
    sys.path.insert(0, str(args.openpi_dir/"src"))
    from openpi.models import pi0_config
    from openpi.models_pytorch import pi0_pytorch
    from safetensors.torch import load_file

    config = pi0_config.Pi0Config(
        pi05=True, dtype="float32", pytorch_compile_mode=None,
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    model = pi0_pytorch.PI0Pytorch(config)
    state = load_file(str(args.weights))
    expected_state = model.state_dict()
    for key, value in expected_state.items():
        if key not in state:
            aliases = [name for name, tensor in expected_state.items()
                       if name in state and tensor.data_ptr() == value.data_ptr()]
            if len(aliases) != 1:
                raise RuntimeError(f"unresolved missing weight {key}")
            state[key] = state[aliases[0]]
    model.load_state_dict(state, strict=True)
    del state, expected_state
    # Neither vocabulary head is called by openpi action inference.
    model.paligemma_with_expert.paligemma.lm_head = None
    model.paligemma_with_expert.gemma_expert.lm_head = None
    language = model.paligemma_with_expert.paligemma.language_model
    if args.device == "cuda":
        embedding = language.embed_tokens
        language.embed_tokens = None
        language.layers.half()
        language.norm.half()
        # Do not downcast the FP32 RoPE frequency buffer during conversion.
        language.rotary_emb.float()
        print(f"[model] device_parameter_mib={sum(p.numel()*p.element_size() for p in model.parameters())/2**20:.3f}", flush=True)
        model.to(args.device)
        torch.cuda.empty_cache()

        def embed_tokens(tokens):
            return embedding(tokens.cpu()).to(args.device)
        model.paligemma_with_expert.embed_language_tokens = embed_tokens
    model.eval()
    gc.collect()
    # The constructor chooses "high"; use full FP32 math for the expert.
    torch.set_float32_matmul_precision("highest")
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    model.paligemma_with_expert.paligemma.vision_tower.config._attn_implementation = args.attention
    ref = args.reference
    meta = json.loads((ref/"meta.json").read_text())
    # Remove masked trailing language padding exactly as the C++ driver does.
    lang = []
    for value in meta["lang_ids_unpadded"]:
        if value == 0:
            break
        lang.append(value)
    ids = torch.tensor([lang], device=args.device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    pixels = torch.from_numpy(np.fromfile(ref/"pixel_values.bin", np.float32)
                              .reshape(1, 3, 224, 224)).to(args.device)
    noise = torch.from_numpy(np.fromfile(ref/"noise.bin", np.float32)
                             .reshape(1, meta["action_horizon"], meta["action_dim"]))
    noise = noise.to(args.device)
    velocities = []

    def run():
        velocities.clear()
        prefix, padding, attention = model.embed_prefix(
            [pixels], [torch.ones(1, dtype=torch.bool, device=args.device)], ids, mask)
        attn = pi0_pytorch.make_att_2d_masks(padding, attention)
        attn = model._prepare_attention_masks_4d(attn)
        positions = torch.cumsum(padding, dim=1)-1
        if args.device == "cuda":
            prefix = prefix.half()
        _, cache = model.paligemma_with_expert.forward(
            inputs_embeds=[prefix, None], attention_mask=attn,
            position_ids=positions, past_key_values=None, use_cache=True)
        if args.device == "cuda":
            cache.key_cache = [x.float() for x in cache.key_cache]
            cache.value_cache = [x.float() for x in cache.value_cache]
        x = noise.clone()
        state = torch.zeros(1, meta["action_dim"], device=args.device)
        dt = torch.tensor(-1/meta["num_steps"], dtype=torch.float32, device=args.device)
        timestep = torch.tensor(1.0, dtype=torch.float32, device=args.device)
        for _ in range(meta["num_steps"]):
            velocity = model.denoise_step(state, padding, cache, x, timestep.expand(1))
            velocities.append(velocity)
            x = x + dt*velocity
            timestep = timestep+dt
        return x

    def validate(output):
        if args.dump_dir:
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            torch.stack(velocities).float().cpu().numpy().tofile(args.dump_dir/"v_t.bin")
            output.float().cpu().numpy().tofile(args.dump_dir/"actions.bin")
        compare("per_step_velocity", torch.stack(velocities).float().cpu().numpy().ravel(),
                np.fromfile(ref/"v_t.bin", np.float32), .005)
        compare("actions", output.float().cpu().numpy().ravel(),
                np.fromfile(ref/"actions.bin", np.float32), .01)

    return model, run, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["paligemma", "pi05"])
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--openpi-dir", type=Path, default=Path("/tmp/openpi"))
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("--repeats must be in [1,100]")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    start = time.perf_counter()
    with torch.inference_mode():
        model, run, validate = (paligemma if args.model == "paligemma" else pi05)(args)
        torch.set_float32_matmul_precision("highest")
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        load_s = time.perf_counter()-start
        for request in range(args.repeats):
            print(f"[request] index={request}", flush=True)
            start = time.perf_counter()
            output = run()
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            print(f"[timing] load_s={load_s:.8g} total_inference_s={elapsed:.8g}", flush=True)
            validate(output)
        import transformers
        print(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__,
                              attention=args.attention, threads=args.threads,
                              peak_torch_allocated_mib=(torch.cuda.max_memory_allocated()/2**20
                                                        if args.device == "cuda" else 0),
                              peak_torch_reserved_mib=(torch.cuda.max_memory_reserved()/2**20
                                                       if args.device == "cuda" else 0))), flush=True)


if __name__ == "__main__":
    main()
