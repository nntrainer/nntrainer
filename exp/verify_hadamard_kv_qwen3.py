#!/usr/bin/env python3
"""
verify_hadamard_kv_qwen3.py

Hypothesis:
  1. Extrapolating KV cache in the original space suffers from outliers → large error.
  2. Applying FWHT (Fast Walsh-Hadamard Transform) to smooth the KV space,
     extrapolating, then inverse-FWHT back, yields better extrapolation accuracy.
  3. Injecting the Hadamard-extrapolated KV cache produces logits closer to the
     ground-truth logits (lower KL-Divergence) than original-space extrapolation.

Usage:
  python3 verify_hadamard_kv_qwen3.py
"""

import math
import os
import sys
import warnings
import copy

# ── SSL workaround for environments with broken CA bundles ──
os.environ.setdefault("CURL_CA_BUNDLE", "")
try:
    import httpx
    _orig_init = httpx.Client.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_init
except Exception:
    pass
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────
# 1. FWHT (Fast Walsh-Hadamard Transform)
# ──────────────────────────────────────────────

def fwht(x: torch.Tensor) -> torch.Tensor:
    """In-place-style FWHT.  Hadamard is self-inverse, so the same function
    serves as forward and inverse transform (up to the 1/√N scaling).

    x shape: (..., head_dim)  — last dim must be a power of two.
    """
    original_shape = x.shape
    N = original_shape[-1]
    x = x.reshape(-1, N).clone()
    M = int(math.log2(N))
    for i in range(M):
        stride = 2 ** i
        x = x.view(-1, N // (stride * 2), 2, stride)
        s = x[:, :, 0, :] + x[:, :, 1, :]
        d = x[:, :, 0, :] - x[:, :, 1, :]
        x = torch.stack([s, d], dim=2)
    x = x.view(original_shape)
    return x / math.sqrt(N)


# ──────────────────────────────────────────────
# 2. Extrapolation helpers
# ──────────────────────────────────────────────

def linear_extrap(t2, t1, t0):
    """1st-order:  X̂_{t+1} = 2 X_t - X_{t-1}"""
    return 2.0 * t0 - t1


def quadratic_extrap(t2, t1, t0):
    """2nd-order Taylor:  X̂_{t+1} = 2.5 X_t - 2 X_{t-1} + 0.5 X_{t-2}"""
    return 2.5 * t0 - 2.0 * t1 + 0.5 * t2


def hard_threshold(x, ratio=0.1):
    """Zero out the smallest-magnitude coefficients (by ratio) in the Hadamard
    domain.  This is the non-linear 'smoothing' step.

    ratio: fraction of coefficients to zero out (0.1 = keep top 90%).
    """
    if ratio <= 0:
        return x
    flat = x.reshape(-1)
    k = max(1, int(flat.numel() * (1.0 - ratio)))
    if k >= flat.numel():
        return x
    thresh = torch.topk(flat.abs(), k, largest=True).values.min()
    mask = x.abs() >= thresh
    return x * mask


# ──────────────────────────────────────────────
# 3. Metrics
# ──────────────────────────────────────────────

def l2_error(pred, target):
    return (pred - target).norm(p=2).item() / target.numel()


def cosine_sim(pred, target):
    p = pred.flatten()
    t = target.flatten()
    return F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).item()


# ──────────────────────────────────────────────
# 4. KV cache helpers (transformers 5.x DynamicCache)
# ──────────────────────────────────────────────

def get_kv_at_pos(cache, layer_idx, pos):
    """Extract K, V at a specific sequence position from a DynamicCache layer."""
    layer = cache.layers[layer_idx]
    k = layer.keys[:, :, pos, :].clone()
    v = layer.values[:, :, pos, :].clone()
    return k, v


def set_kv_at_pos(cache, layer_idx, pos, k_val, v_val):
    """Overwrite K, V at a specific sequence position in a DynamicCache layer."""
    layer = cache.layers[layer_idx]
    layer.keys[:, :, pos, :] = k_val
    layer.values[:, :, pos, :] = v_val


# ──────────────────────────────────────────────
# 5. Extrapolation pipeline
# ──────────────────────────────────────────────

def extrapolate_kv(layers_k, layers_v, num_layers, method, extrap_fn, thresh_ratio):
    """Returns dict of {layer_idx: (k_pred, v_pred)} for the given method."""
    preds = {}
    for li in range(num_layers):
        dk = layers_k[li]
        dv = layers_v[li]

        if method == "hadamard":
            k_pred_h = extrap_fn(fwht(dk["tm2"]), fwht(dk["tm1"]), fwht(dk["t"]))
            k_pred_h = hard_threshold(k_pred_h, ratio=thresh_ratio)
            k_pred = fwht(k_pred_h)
            v_pred_h = extrap_fn(fwht(dv["tm2"]), fwht(dv["tm1"]), fwht(dv["t"]))
            v_pred_h = hard_threshold(v_pred_h, ratio=thresh_ratio)
            v_pred = fwht(v_pred_h)
        else:
            k_pred = extrap_fn(dk["tm2"], dk["tm1"], dk["t"])
            v_pred = extrap_fn(dv["tm2"], dv["tm1"], dv["t"])

        preds[li] = (k_pred, v_pred)
    return preds


# ──────────────────────────────────────────────
# 6. Main experiment
# ──────────────────────────────────────────────

def main():
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map=device
    )
    model.eval()

    prompt = "The quick brown fox jumps over the lazy dog and then"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # --- Run model once to get KV cache at every position ---
    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    past_kv = out.past_key_values
    num_layers = len(past_kv.layers)
    seq_len = past_kv.layers[0].keys.shape[2]
    print(f"Layers: {num_layers}  Seq len: {seq_len}")

    if seq_len < 5:
        print("Need at least 5 tokens for t-2, t-1, t, t+1 extraction.")
        sys.exit(1)

    # Indices: t-2, t-1, t, t+1 (ground truth)
    idx_tm2 = seq_len - 4
    idx_tm1 = seq_len - 3
    idx_t   = seq_len - 2
    idx_tp1 = seq_len - 1   # ground-truth t+1

    # --- Collect per-layer KV slices at the four positions ---
    layers_k = {}
    layers_v = {}
    for li in range(num_layers):
        k_tm2, v_tm2 = get_kv_at_pos(past_kv, li, idx_tm2)
        k_tm1, v_tm1 = get_kv_at_pos(past_kv, li, idx_tm1)
        k_t,   v_t   = get_kv_at_pos(past_kv, li, idx_t)
        k_tp1, v_tp1 = get_kv_at_pos(past_kv, li, idx_tp1)
        layers_k[li] = {"tm2": k_tm2, "tm1": k_tm1, "t": k_t, "tp1": k_tp1}
        layers_v[li] = {"tm2": v_tm2, "tm1": v_tm1, "t": v_t, "tp1": v_tp1}

    # --- Determine the actual next token (ground truth for t+2) ---
    # The model's prediction at position idx_tp1 is what it thinks comes next
    with torch.no_grad():
        next_token_id = out.logits[:, idx_tp1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
    gt_token_str = tokenizer.decode(next_token_id[0])
    print(f"Ground-truth next token (t+2): '{gt_token_str}' (id={next_token_id.item()})")

    # Build attention mask for the next-token forward pass
    next_attn_mask = torch.ones(
        (1, seq_len + 1), dtype=inputs["attention_mask"].dtype, device=device
    )

    # --- Ground-truth logits: original cache + next token ---
    gt_kv = copy.deepcopy(past_kv)
    with torch.no_grad():
        out_gt = model(
            input_ids=next_token_id,
            past_key_values=gt_kv,
            use_cache=True,
            attention_mask=next_attn_mask,
        )
        gt_logits = out_gt.logits[:, -1, :]  # (1, vocab)
        gt_pred_token = gt_logits.argmax(dim=-1)
    gt_pred_str = tokenizer.decode(gt_pred_token[0])
    print(f"Ground-truth prediction for t+3: '{gt_pred_str}' (id={gt_pred_token.item()})")

    # Print ground-truth top-5 tokens
    gt_top5 = torch.topk(gt_logits[0], 5)
    print(f"\n  Ground-truth top-5 predictions:")
    for i in range(5):
        tid = gt_top5.indices[i].item()
        prob = F.softmax(gt_logits[0], dim=-1)[tid].item()
        print(f"    {i+1}. '{tokenizer.decode(tid)}' (id={tid}, prob={prob:.4f})")


    # --- Test multiple configurations ---
    configs = [
        ("original",  "linear",    linear_extrap,    0.0),
        ("original",  "quadratic", quadratic_extrap, 0.0),
        ("hadamard",  "linear",    linear_extrap,    0.1),
        ("hadamard",  "linear",    linear_extrap,    0.3),
        ("hadamard",  "linear",    linear_extrap,    0.5),
        ("hadamard",  "quadratic", quadratic_extrap, 0.1),
        ("hadamard",  "quadratic", quadratic_extrap, 0.3),
        ("hadamard",  "quadratic", quadratic_extrap, 0.5),
    ]

    all_results = []

    for method, order_name, extrap_fn, thresh_ratio in configs:
        # Compute L2 / Cosine metrics
        l2_list = []
        cos_list = []
        preds = extrapolate_kv(layers_k, layers_v, num_layers, method, extrap_fn, thresh_ratio)

        for li in range(num_layers):
            for kv_store in (layers_k, layers_v):
                d = kv_store[li]
                tp1_gt = d["tp1"]
                k_pred, v_pred = preds[li]
                pred = k_pred if kv_store is layers_k else v_pred
                l2_list.append(l2_error(pred, tp1_gt))
                cos_list.append(cosine_sim(pred, tp1_gt))

        # Build modified cache and compute KL + token prediction
        new_kv = copy.deepcopy(past_kv)
        for li in range(num_layers):
            k_pred, v_pred = preds[li]
            set_kv_at_pos(new_kv, li, idx_tp1, k_pred, v_pred)

        with torch.no_grad():
            out_mod = model(
                input_ids=next_token_id,
                past_key_values=new_kv,
                use_cache=True,
                attention_mask=next_attn_mask,
            )
            mod_logits = out_mod.logits[:, -1, :]
            mod_pred_token = mod_logits.argmax(dim=-1)

        mod_pred_str = tokenizer.decode(mod_pred_token[0])

        # KL(truth || mod)
        p = F.softmax(gt_logits, dim=-1)
        q = F.softmax(mod_logits, dim=-1)
        kl = F.kl_div(q.log(), p, reduction="sum").item()

        # Top-5 token match
        gt_top5 = torch.topk(gt_logits[0], 5).indices
        mod_top5 = torch.topk(mod_logits[0], 5).indices
        top5_overlap = len(set(gt_top5.tolist()) & set(mod_top5.tolist()))

        all_results.append({
            "method": method,
            "order": order_name,
            "thresh": thresh_ratio,
            "l2": sum(l2_list) / len(l2_list),
            "cos": sum(cos_list) / len(cos_list),
            "kl": kl,
            "pred_token": mod_pred_str,
            "pred_token_id": mod_pred_token.item(),
            "top5_overlap": top5_overlap,
        })

    # --- Print results table ---
    print("\n" + "=" * 110)
    print("  Hadamard vs Original KV-Cache Extrapolation — Results")
    print("=" * 110)

    header = f"{'Method':<10} {'Order':<10} {'Thresh':>6} {'L2 Error':>10} {'Cosine':>10} {'KL Div':>10} {'Top5∩':>6} {'Pred Token':>15}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(f"{r['method']:<10} {r['order']:<10} {r['thresh']:>6.1f} {r['l2']:>10.6f} {r['cos']:>10.6f} {r['kl']:>10.6f} {r['top5_overlap']:>6} '{r['pred_token']}':>15")

    print("-" * len(header))
    print(f"  Ground-truth prediction: '{gt_pred_str}' (id={gt_pred_token.item()})")
    print("=" * 110)

    # --- Summary ---
    print("\n📋 Summary:")
    best_kl = min(all_results, key=lambda x: x["kl"])
    print(f"  Best KL Divergence: {best_kl['method']}/{best_kl['order']} (thresh={best_kl['thresh']}) → KL={best_kl['kl']:.4f}")
    print(f"  Best L2 Error:      {min(all_results, key=lambda x: x['l2'])['method']}/{min(all_results, key=lambda x: x['l2'])['order']}")
    print(f"  Best Cosine Sim:    {max(all_results, key=lambda x: x['cos'])['method']}/{max(all_results, key=lambda x: x['cos'])['order']}")

    # Check if Hadamard actually predicts the correct token
    gt_id = gt_pred_token.item()
    print(f"\n  Token prediction match (vs ground-truth '{gt_pred_str}'):")
    for r in all_results:
        match = "✅" if r["pred_token_id"] == gt_id else "❌"
        print(f"    {match} {r['method']}/{r['order']}/thresh={r['thresh']:.1f} → '{r['pred_token']}'")

    print()


if __name__ == "__main__":
    main()
