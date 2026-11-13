# SPDX-License-Identifier: Apache-2.0
## @file golden_dump.py
## @brief Dump HF PaliGemma-3B golden outputs for the nntrainer e2e check.
##
## Loads the real google/paligemma-3b-pt-224 checkpoint (FP32) and produces,
## in --out_dir:
##   pixel_values.bin   raw FP32 [1,3,224,224] from the HF PIL processor;
##                      the driver also verifies file preprocessing.
##   input_ids.json     [bos, 256 x image_token, text...] for the mm case
##   text_logits.json   last-row logits, text-only prompt, all-prefix mask
##                      (matches the C++ whole-prompt-bidirectional run())
##   mm_logits.json     last-row logits of the image+text prompt (prefix-LM)
##   mm_tokens.json     greedy tokens for the mm prompt (max n_gen)
##   top5 sanity lines on stdout.
##
## Usage:
##   python golden_dump.py --model_dir ~/Downloads --out_dir ./e2e_golden

import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


def make_sample_image(path: pathlib.Path, seed: int = 7) -> None:
    """Deterministic synthetic test image (no network, no assets needed)."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((336, 336, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:336, 0:336]
    arr[..., 0] = (xx * 255 // 336).astype(np.uint8)
    arr[..., 1] = (yy * 255 // 336).astype(np.uint8)
    arr[..., 2] = ((xx + yy) * 127 // 672).astype(np.uint8)
    for cy, cx, r, v in [(84, 84, 40, 255), (252, 252, 40, 0), (168, 252, 24, 128)]:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        arr[mask] = v
    arr = (arr.astype(np.float32) + rng.randn(336, 336, 3) * 4).clip(0, 255)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def last_row_logits(model, out) -> np.ndarray:
    return out.logits[0, -1, :].detach().float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=pathlib.Path,
                    default=pathlib.Path.home() / "Downloads")
    ap.add_argument("--out_dir", type=pathlib.Path,
                    default=pathlib.Path("./e2e_golden"))
    ap.add_argument("--text_prompt", type=str,
                    default="a photo of")
    ap.add_argument("--mm_prompt", type=str,
                    default="<image>a wide shot of a large amount of "
                            "vehicles stuck in a traffic jam")
    ap.add_argument("--n_gen", type=int, default=16)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--image_seed", type=int, default=7)
    ap.add_argument("--suffix_tokens", type=int, default=0)
    ap.add_argument("--image_backend", choices=["pil", "torchvision"], default="pil")
    ap.add_argument("--text_dtype", choices=["float32", "float16"], default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = args.out_dir / "sample.png"
    if not sample_path.exists():
        make_sample_image(sample_path, args.image_seed)
    image = Image.open(sample_path).convert("RGB")

    print("[1/4] loading model (FP32, ~12.4 GB RAM)...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=getattr(torch, args.dtype),
        attn_implementation="eager")
    if args.text_dtype:
        model.model.language_model.to(getattr(torch, args.text_dtype))
        model.lm_head.to(getattr(torch, args.text_dtype))
    model.to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, backend=args.image_backend)
    bos = model.config.text_config.bos_token_id

    print("[2/4] text-only forward (all-prefix token_type_ids)...")
    text_ids = [bos] + processor.tokenizer.encode(
        args.text_prompt, add_special_tokens=False)
    ids = torch.tensor([text_ids], dtype=torch.long, device=args.device)
    with torch.no_grad():
        out = model(input_ids=ids,
                    token_type_ids=torch.zeros_like(ids),
                    use_cache=False, logits_to_keep=1)
    text_logits = last_row_logits(model, out)
    json.dump(text_logits.tolist(),
              open(args.out_dir / "text_logits.json", "w"))
    print("  text argmax:",
          int(np.argmax(text_logits)),
          processor.tokenizer.decode([int(np.argmax(text_logits))]))

    with torch.no_grad():
        gen_t = model.generate(input_ids=ids,
                               token_type_ids=torch.zeros_like(ids),
                               max_new_tokens=args.n_gen, do_sample=False)
    text_tokens = gen_t[0, ids.shape[1]:].tolist()
    json.dump(text_ids, open(args.out_dir / "text_ids.json", "w"))
    json.dump(text_tokens, open(args.out_dir / "text_tokens.json", "w"))
    print("  text greedy:", text_tokens,
          processor.tokenizer.decode(text_tokens))

    print("[3/4] multimodal forward + greedy generation...")
    proc = processor(images=image, text=args.mm_prompt, return_tensors="pt")
    proc = {k: v.to(device=args.device, dtype=(model.dtype if v.is_floating_point() else v.dtype))
            for k, v in proc.items()}
    pixel_values = proc["pixel_values"]  # [1,3,224,224]
    input_ids = proc["input_ids"]
    img_tok = model.config.image_token_index
    # Prefer the processor's own token_type_ids (its canonical prefix mask);
    # fall back to the image-block convention if the processor omits it.
    if "token_type_ids" in proc:
        token_type_ids = proc["token_type_ids"]
    else:
        token_type_ids = (input_ids != img_tok).long()  # 0=image block, 1=text
    if args.suffix_tokens:
        token_type_ids[:, -args.suffix_tokens:] = 1
    attention_mask = proc.get("attention_mask", torch.ones_like(input_ids))
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    use_cache=False, logits_to_keep=1)
    mm_logits = last_row_logits(model, out)
    json.dump(mm_logits.tolist(), open(args.out_dir / "mm_logits.json", "w"))
    print("  mm argmax:", int(np.argmax(mm_logits)),
          processor.tokenizer.decode([int(np.argmax(mm_logits))]))

    with torch.no_grad():
        gen = model.generate(input_ids=input_ids,
                             pixel_values=pixel_values,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             max_new_tokens=args.n_gen,
                             do_sample=False, return_dict_in_generate=True,
                             output_scores=True)
    gen_tokens = gen.sequences[0, input_ids.shape[1]:].tolist()
    decode_top5 = []
    for scores in gen.scores:
        values, indices = scores[0].float().topk(5)
        decode_top5.append(list(zip(indices.cpu().tolist(), values.cpu().tolist())))
    json.dump(decode_top5, open(args.out_dir / "decode_top5.json", "w"))
    json.dump(gen_tokens, open(args.out_dir / "mm_tokens.json", "w"))
    print("  greedy:", gen_tokens)
    print("  text:", processor.tokenizer.decode(gen_tokens))

    with torch.no_grad():
        features = model.get_image_features(pixel_values=pixel_values)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        if isinstance(features, (tuple, list)):
            features = features[0]
        features.detach().float().cpu().numpy().tofile(args.out_dir / "image_emb.bin")
    json.dump({"torch": torch.__version__, "device": args.device,
               "dtype": args.dtype, "image_backend": args.image_backend, "text_dtype": args.text_dtype, "model_dir": str(args.model_dir)},
              open(args.out_dir / "provenance.json", "w"))
    print("[4/4] saving inputs for the C++ driver...")
    pixel_values.float().cpu().numpy().astype(np.float32).tofile(
        args.out_dir / "pixel_values.bin")
    json.dump(input_ids[0].tolist(), open(args.out_dir / "input_ids.json", "w"))
    json.dump(token_type_ids[0].tolist(),
              open(args.out_dir / "token_type_ids.json", "w"))

    t5 = np.argsort(text_logits)[::-1][:5]
    m5 = np.argsort(mm_logits)[::-1][:5]
    print(f"done -> {args.out_dir}")
    print("text top5:", [(int(i), float(text_logits[i])) for i in t5])
    print("mm   top5:", [(int(i), float(mm_logits[i])) for i in m5])


if __name__ == "__main__":
    main()
