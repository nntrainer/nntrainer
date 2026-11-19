# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# Dump an openpi PI0Pytorch golden for the pi0.5 3B e2e: load the
# lerobot/pi05_base checkpoint into openpi's PyTorch model (reference venv:
# /home/seungbaek/venvs-openpi, transformers 4.53.2 + jaxtyping==0.2.36),
# run sample_actions on a fixed synthetic observation, and record per-step
# v_t plus the final action chunk.
#
# Output (e2e_golden_pi05/ next to this script):
#   meta.json      observation + config scalars + per-step v_t + actions
#   pixel_values.bin  the synthetic image, CHW float32, [0,1] normalized
#
# The nntrainer driver (nntr_pi05_e2e) reproduces the same numbers with the
# C++ pipeline: SiglipVisionModel -> projector rows -> backbone prefix pass
# -> K/V transplant -> expert Euler loop.

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--openpi_dir", default="/tmp/openpi")
ap.add_argument("--checkpoint", default="/home/seungbaek/pi05-base-hf/model.safetensors")
ap.add_argument("--tokenizer", default="/home/seungbaek/Downloads/paligemma/tokenizer.json")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--prompt", default="pick up the object")
ap.add_argument("--seed", type=int, default=20261112)
args = ap.parse_args()
sys.path.insert(0, os.path.join(args.openpi_dir, "src"))
sys.path.insert(0, os.path.join(args.openpi_dir, "src/openpi/models_pytorch"))
import pi0_pytorch
from openpi.models import pi0_config
from openpi.models import gemma as _gemma
OUT = args.out_dir
os.makedirs(OUT, exist_ok=True)
CKPT = args.checkpoint
PROMPT = args.prompt


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_num_threads(4)

# ---- build the openpi model and load the checkpoint ----
config = pi0_config.Pi0Config(
    pi05=True, dtype="float32", pytorch_compile_mode=None,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m")
model = pi0_pytorch.PI0Pytorch(config)
model = model.to(torch.float32)
model.eval()
torch.set_float32_matmul_precision("highest")

from safetensors.torch import load_file  # noqa: E402

state = load_file(CKPT)
# Safetensors omits duplicated tied weights. Restore only aliases whose
# parameter storage is actually shared in the reference model.
expected = model.state_dict()
for key, value in expected.items():
    if key not in state:
        aliases = [name for name, tensor in expected.items()
                   if name in state and tensor.data_ptr() == value.data_ptr()]
        if len(aliases) != 1:
            raise RuntimeError(f"unresolved missing weight {key}: {aliases}")
        state[key] = state[aliases[0]]
        print(f"resolved tied weight {key} <- {aliases[0]}")
missing, unexpected = model.load_state_dict(state, strict=True)
print(f"state dict: {len(state)} tensors, "
      f"missing={len(missing)}, unexpected={len(unexpected)}")
if missing or unexpected:
    raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
if missing:
    print("  missing (first 5):", missing[:5])
if unexpected:
    print("  unexpected (first 5):", unexpected[:5])

# ---- fixed observation ----
IMG = 224  # pi05 input resolution (fixed in preprocessing; Pi0Config
           # carries no image_resolution field)
image = np.random.rand(3, IMG, IMG).astype(np.float32) * 0.9 + 0.05
pixel_values = torch.from_numpy(image)[None]  # [1, 3, H, W], [0,1]

# paligemma tokenizer (same assets as the paligemma-3b-pt-224 port)
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(
    args.tokenizer)
lang_ids = tok.encode(PROMPT, add_special_tokens=True).ids
print("prompt ids:", lang_ids, "->", PROMPT)

max_token_len = config.max_token_len  # 200 for pi05
mask = [1] * len(lang_ids) + [0] * (max_token_len - len(lang_ids))
lang_ids = lang_ids + [0] * (max_token_len - len(lang_ids))
tokenized_prompt = torch.tensor([lang_ids])
tokenized_prompt_mask = torch.tensor([mask], dtype=torch.bool)

# pi05 uses DISCRETE state tokens inside the prompt; the continuous state
# input stays zero (see openpi _preprocess_observation with pi05=True).

# openpi works with a batch dim: [bsize, horizon, action_dim]
noise = torch.randn(1, config.action_horizon, config.action_dim)

with torch.no_grad():
    observation = {
        "images": [pixel_values],
        "img_masks": [torch.ones(1, dtype=torch.bool)],
        "lang_tokens": tokenized_prompt,
        "lang_masks": tokenized_prompt_mask,
        # pi05 ignores the continuous state (discretized into the prompt
        # tokens upstream); embed_suffix's `if not self.pi05` guard skips it.
        "state": torch.zeros(1, config.action_dim),
    }
    # --- prefix pass (records v_t per step by monkey-patching nothing:
    #     reimplement the 10-step loop calling denoise_step directly so we
    #     can record v_t per step, exactly like sample_actions) ---
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        observation["images"], observation["img_masks"],
        observation["lang_tokens"], observation["lang_masks"])
    # golden vision output for the C++ vision-graph comparison
    # embed_image -> get_image_features expects [B, 3, H, W];
    # observation["images"][0] is ALREADY [1, 3, H, W] (pixel_values got its
    # batch dim at the top of this script) — pass it as-is.
    img_emb_golden = model.paligemma_with_expert.embed_image(
        observation["images"][0])
    img_emb_golden[0].reshape(-1).float().numpy().tofile(
        os.path.join(OUT, "image_emb.bin"))
    print("dumped golden image_emb:",
          tuple(img_emb_golden.shape))
    prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(
        prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(
        prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_kv = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    N_STEPS = 10  # openpi sample_actions default (num_steps is a method
                  # parameter, not a Pi0Config field)
    dt = -1.0 / N_STEPS
    dt_t = torch.tensor(dt, dtype=torch.float32)
    x_t = noise.clone()
    time = torch.tensor(1.0, dtype=torch.float32)
    vts = []
    conds = []
    first = True
    while time >= -dt_t / 2:
        # capture the step's adaRMS cond for cross-checking the C++ side
        if first:
            half = _gemma.get_config(
                "gemma_300m").width // 2  # expert width 1024
            frac = torch.linspace(0.0, 1.0, half, dtype=torch.float64)
            period = 4e-3 * (4.0 / 4e-3) ** frac
            ang = (2 * math.pi / period) * float(time)
            t_emb = torch.cat([torch.sin(torch.tensor(ang)),
                               torch.cos(torch.tensor(ang))]).float()
            cond0 = torch.nn.functional.silu(
                model.time_mlp_out(
                    torch.nn.functional.silu(model.time_mlp_in(t_emb))))
            conds.append(cond0.reshape(-1))
            suf0 = model.action_in_proj(x_t)
            suf0[0].float().numpy().tofile(os.path.join(OUT, "suffix0.bin"))
            print("dumped golden suffix0:", tuple(suf0.shape))
            # capture the expert L0 POST-ATTENTION hidden state (same point
            # as the C++ after1 dump: residual + attn_out * gate, before the
            # MLP). The layer hook returned the post-MLP output instead, so
            # hook self_attn (o_proj output) + input_layernorm (gate) and
            # recompose.
            cap_attn = {}
            cap_ln = {}
            cap_n1 = {}
            suffix_emb = suf0[0]  # [50, 1024] — the residual input
            def attn_hook(mod, inp, out):
                if 0 not in cap_attn:
                    cap_attn[0] = out[0][0].detach().clone()
                    cap_attn[0].float().numpy().tofile(
                        os.path.join(OUT, "l0_attn_out.bin"))
                    print("dumped golden l0_attn_out:", tuple(cap_attn[0].shape))

            def attn_pre_hook(mod, args, kwargs):
                if 0 not in cap_n1:
                    # hidden_states is passed as a kwarg by the decoder layer
                    hs = kwargs.get("hidden_states", args[0] if args else None)
                    if hs is not None:
                        n1 = hs[0].detach().clone()
                        n1.float().numpy().tofile(
                            os.path.join(OUT, "l0_attn_in.bin"))
                        cap_n1[0] = True
            def ln_hook(mod, inp, out):
                if 0 not in cap_ln:
                    cap_ln[0] = out[1][0].detach().clone()  # gate
            ha = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.register_forward_hook(attn_hook)
            hn = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)
            hl = model.paligemma_with_expert.gemma_expert.model.layers[0].input_layernorm.register_forward_hook(ln_hook)
            v_t = model.denoise_step(
                observation["state"], prefix_pad_masks, past_kv, x_t,
                time.expand(1))
            ha.remove()
            hn.remove()
            hl.remove()
            after_attn = suffix_emb + cap_attn[0] * cap_ln[0][:, None, :]
            after_attn[0].float().numpy().tofile(
                os.path.join(OUT, "l0_after_attn.bin"))
            print("dumped golden l0_after_attn:", tuple(after_attn.shape))
            first = False
        else:
            v_t = model.denoise_step(
                observation["state"], prefix_pad_masks, past_kv, x_t,
                time.expand(1))
        vts.append(v_t.clone())
        x_t = x_t + dt_t * v_t
        time = time + dt_t
    final_actions = x_t

    # ---- prefix K/V dump (post-rope, valid rows only) ----
    n_valid = int(prefix_pad_masks[0].sum())
    n_layers = len(past_kv)
    print("past_kv[0][0] shape:", tuple(past_kv[0][0].shape))
    for li in range(n_layers):
        k_l, v_l = past_kv[li]
        knp = k_l[0, :, :n_valid, :].reshape(-1).float().numpy()
        vnp = v_l[0, :, :n_valid, :].reshape(-1).float().numpy()
        knp.tofile(os.path.join(OUT, f"bb{li}_k.bin"))
        vnp.tofile(os.path.join(OUT, f"bb{li}_v.bin"))
    conds[0].numpy().tofile(os.path.join(OUT, "cond0.bin"))
    print(f"dumped prefix K/V for {n_layers} layers, {n_valid} valid rows")

# ---- write the golden ----
pixel_values.numpy().tofile(os.path.join(OUT, "pixel_values.bin"))
# flat binaries for the C++ driver (batch dim stripped: [horizon, dim])
noise.numpy().tofile(os.path.join(OUT, "noise.bin"))
np.concatenate([v.reshape(-1) for v in vts]).astype(
    np.float32).tofile(os.path.join(OUT, "v_t.bin"))
final_actions.reshape(-1).numpy().tofile(
    os.path.join(OUT, "actions.bin"))
with open(os.path.join(OUT, "ids.txt"), "w") as f:
    f.write(" ".join(str(i) for i in lang_ids))
meta = {
    "prompt": PROMPT,
    "checkpoint": CKPT, "seed": args.seed, "torch": torch.__version__,
    "construction_dtype": config.dtype,
    "lang_ids_unpadded": lang_ids,
    "image_resolution": IMG,
    "action_dim": int(config.action_dim),
    "action_horizon": 50,
    "num_steps": N_STEPS,
    "max_token_len": max_token_len,
    "noise": noise.tolist(),
    "per_step_v_t": [v.tolist() for v in vts],
    "final_actions": final_actions.tolist(),
    "atol_v_t": 5e-3,
    "atol_actions": 1e-2,
}
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("wrote", OUT)
fa = final_actions.reshape(-1).tolist()
print("final_actions[0][:3]:", [round(v, 6) for v in fa[:3]])
print("step0 v_t[0][:3]:", [round(v, 6) for v in vts[0].reshape(-1)[:3].tolist()])
