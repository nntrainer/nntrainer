#!/usr/bin/env python3
"""
Integration tests: convert multiple model architectures and verify the
generated C++ code compiles and initializes with NNTrainer.

For each model config:
  1. Create a tiny HuggingFace model locally
  2. Run converter.py to generate C++ header + source
  3. Build with meson/ninja
  4. Run the test executable and verify exit code 0

Usage:
  python -m pytest tests/test_build_and_run.py -v
  python tests/test_build_and_run.py           # standalone
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# Ensure the converter package is importable
CONVERTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CONVERTER_DIR)

# Path to the nntrainer source root and build output
NNTRAINER_ROOT = os.path.abspath(os.path.join(CONVERTER_DIR, "..", ".."))
BUILD_DIR = os.path.join(NNTRAINER_ROOT, "builddir")
JNI_DIR = os.path.join(CONVERTER_DIR, "jni")

# ---- Tiny model configurations for offline testing ----

# --- Decoder-only (CausalLM) models ---

QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}

LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}

QWEN3_TIED_CONFIG = {
    **QWEN3_CONFIG,
    "tie_word_embeddings": True,
}

# Qwen3-0.6B: standard Qwen3 CausalLM (same arch, slightly different config)
QWEN3_06B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 192,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1500,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
}

# Qwen3-1.7B: larger Qwen3 CausalLM (more heads, more intermediate)
QWEN3_17B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 128,
    "intermediate_size": 384,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "vocab_size": 2000,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}

# Granite 4.0: GraniteMoeHybrid in dense mode (num_local_experts=0)
GRANITE_40_CONFIG = {
    "architectures": ["GraniteMoeHybridForCausalLM"],
    "model_type": "granitemoehybrid",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    # Granite-specific: dense mode (no MoE), no Mamba
    "num_local_experts": 0,
    "num_experts_per_tok": 0,
    "layer_types": ["attention", "attention"],
    # Granite scaling multipliers
    "embedding_multiplier": 1.0,
    "logits_scaling": 1.0,
    "residual_multiplier": 1.0,
    "attention_multiplier": 1.0,
}

# LFM-700M: Liquid Foundation Model (lfm2), hybrid conv + attention
LFM_700M_CONFIG = {
    "architectures": ["Lfm2ForCausalLM"],
    "model_type": "lfm2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "norm_eps": 1e-5,
    "tie_word_embeddings": True,
    # lfm2 requires at least one conv layer for HybridConvCache
    "layer_types": ["conv", "full_attention"],
    "conv_L_cache": 3,
}

# --- Embedding models (base models without LM head) ---

# Qwen3-Embedding-0.6B: Qwen3 base model for sentence embeddings
QWEN3_EMBEDDING_CONFIG = {
    "architectures": ["Qwen3Model"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}

# EmbeddingGemma-300M: Gemma2 base model for embeddings
EMBEDDING_GEMMA_CONFIG = {
    "architectures": ["Gemma2Model"],
    "model_type": "gemma2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
}

# KaLM-Embedding-v2.5: Qwen2 base model for embeddings
KALM_EMBEDDING_CONFIG = {
    "architectures": ["Qwen2Model"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}

# --- Encoder-only models ---

# Multilingual-E5-tiny-instruct: XLM-RoBERTa encoder
MULTILINGUAL_E5_CONFIG = {
    "architectures": ["XLMRobertaModel"],
    "model_type": "xlm-roberta",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "hidden_act": "gelu",
    "type_vocab_size": 1,
}

# --- Custom models ---

# GLiNER2-multi-v1: custom "extractor" model_type (NER model)
# Uses custom_models.py loader; requires synthetic weights
GLINER2_CONFIG = {
    "architectures": ["ExtractorModel"],
    "model_type": "extractor",
    "hidden_size": 64,
    "max_width": 4,
    "num_rnn_layers": 1,
    "span_mode": "markerV0",
}

# --- Non-HuggingFace models (conversion-only tests) ---
# These use custom loaders and don't go through _create_local_model.
# They are tested for successful FX tracing and full layer mapping only.

# FLUX: Diffusion Transformer (Black Forest Labs)
FLUX_CONFIG = {
    "model_type": "flux",
    "architectures": ["FluxTransformer2DModel"],
    "in_channels": 4,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 32,
    "num_attention_heads": 2,
    "joint_attention_dim": 64,
    "pooled_projection_dim": 32,
    "guidance_embeds": False,
    "axes_dims_rope": [4, 14, 14],
    "patch_size": 1,
    "hidden_size": 64,
}

# SigLIP: Vision-Language contrastive model (Google)
SIGLIP_CONFIG = {
    "model_type": "siglip",
    "architectures": ["SiglipModel"],
    "vision_hidden_size": 64,
    "vision_intermediate_size": 128,
    "vision_num_hidden_layers": 1,
    "vision_num_attention_heads": 2,
    "vision_image_size": 32,
    "vision_patch_size": 16,
    "text_hidden_size": 64,
    "text_intermediate_size": 128,
    "text_num_hidden_layers": 1,
    "text_num_attention_heads": 2,
    "text_vocab_size": 100,
    "text_max_position_embeddings": 16,
}

# Conformer: Speech recognition model (torchaudio)
CONFORMER_CONFIG = {
    "model_type": "conformer",
    "architectures": ["Conformer"],
    "input_dim": 64,
    "num_attention_heads": 2,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "depthwise_conv_kernel_size": 3,
    "hidden_size": 64,
}

# Zipformer: Speech recognition model (k2-fsa/icefall)
ZIPFORMER_CONFIG = {
    "model_type": "zipformer",
    "architectures": ["Zipformer2"],
    "hidden_size": 64,
    "num_attention_heads": 2,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_stacks": 2,
    "downsampling_factor": [1, 2],
    "cnn_module_kernel": [3, 3],
    "query_head_dim": 8,
    "pos_head_dim": 2,
    "value_head_dim": 4,
    "pos_dim": 16,
    "encoder_unmasked_dim": [32, 32],
}

# Gemma3n: Google's efficient multimodal model (text-only for testing)
# Features: AltUp, Laurel (low-rank residual), per-layer input embeddings
GEMMA3N_CONFIG = {
    "model_type": "gemma3n_text",
    "architectures": ["Gemma3nTextModel"],
    "vocab_size": 500,
    "vocab_size_per_layer_input": 500,
    "hidden_size": 64,
    "hidden_size_per_layer_input": 32,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "sliding_window": 32,
    "altup_num_inputs": 2,
    "altup_active_idx": 0,
    "num_kv_shared_layers": 1,
    "laurel_rank": 8,
}

# --- Object Detection models ---

# YOLOv2: 13-layer CNN backbone with branching (reorg + concat)
YOLOV2_CONFIG = {
    "model_type": "yolov2",
    "num_classes": 5,
    "num_anchors": 5,
    "image_size": 416,
}

# YOLOv3: Darknet53 backbone + Feature Pyramid Network + 3 detection heads
YOLOV3_CONFIG = {
    "model_type": "yolov3",
    "num_classes": 5,
    "image_size": 416,
}

# ResNet18: BasicBlock residual CNN (from nntrainer Applications/Resnet)
RESNET18_CONFIG = {
    "model_type": "resnet18",
    "num_classes": 100,
    "image_size": 32,
}

# ResNet50: Bottleneck residual CNN (from torchvision)
RESNET50_CONFIG = {
    "model_type": "resnet50",
    "num_classes": 100,
    "image_size": 224,
}

# --- Vision Transformer models ---

# ViT-B/16: Vision Transformer (torchvision, 12 encoder layers)
VIT_CONFIG = {
    "model_type": "vit",
    "num_classes": 100,
    "image_size": 224,
}

# DeiT: Data-efficient Image Transformer (HuggingFace, tiny config)
DEIT_CONFIG = {
    "model_type": "deit",
    "hidden_size": 192,
    "num_hidden_layers": 4,
    "num_attention_heads": 3,
    "intermediate_size": 768,
    "image_size": 224,
    "patch_size": 16,
}

# --- GAN models ---

# DCGAN Generator: ConvTranspose2d stack (Radford et al., 2015)
DCGAN_GEN_CONFIG = {
    "model_type": "dcgan_gen",
    "nz": 100,
    "ngf": 64,
    "nc": 3,
}

# DCGAN Discriminator: Conv2d stack with LeakyReLU
DCGAN_DISC_CONFIG = {
    "model_type": "dcgan_disc",
    "nc": 3,
    "ndf": 64,
    "image_size": 64,
}

# Pix2Pix: U-Net Generator with skip connections (Isola et al., 2017)
PIX2PIX_CONFIG = {
    "model_type": "pix2pix",
    "ngf": 64,
    "image_size": 256,
}

# --- torchvision CNN / Hybrid models ---

# MobileNetV2: Inverted residual + hardtanh (ReLU6)
MOBILENET_V2_CONFIG = {
    "model_type": "mobilenet_v2",
    "image_size": 224,
}

# MobileNetV3-Small: Inverted residual + hardswish/hardsigmoid (SE blocks)
MOBILENET_V3_CONFIG = {
    "model_type": "mobilenet_v3_small",
    "image_size": 224,
}

# DenseNet121: Dense connections (each layer concatenates all previous)
DENSENET121_CONFIG = {
    "model_type": "densenet121",
    "image_size": 224,
}

# InceptionV3: Multi-branch parallel convolutions
INCEPTION_V3_CONFIG = {
    "model_type": "inception_v3",
    "image_size": 299,
}

# ConvNeXt-Tiny: Modernized ConvNet (LayerNorm + GELU + permute)
CONVNEXT_CONFIG = {
    "model_type": "convnext",
    "image_size": 224,
}

# Swin Transformer Tiny: Shifted window attention + torch.roll
SWIN_CONFIG = {
    "model_type": "swin",
    "image_size": 224,
}

# MaxViT-T: Multi-axis attention (block + grid) + swapaxes
MAXVIT_CONFIG = {
    "model_type": "maxvit",
    "image_size": 224,
}

# --- Encoder-Decoder models ---

# T5Gemma2-270M: Gemma2-based encoder-decoder (multimodal)
# NOTE: T5Gemma2 has a complex nested config (encoder + decoder + vision tower)
# that requires special handling. This config is for conversion testing only.
T5GEMMA2_CONFIG = {
    "architectures": ["T5Gemma2ForConditionalGeneration"],
    "model_type": "t5gemma2",
    "is_encoder_decoder": True,
    "tie_word_embeddings": True,
    "vocab_size": 1000,
    # Sub-configs are set programmatically in _create_local_model
}


# ---- Registry of all test models ----

MODEL_CONFIGS = {
    # Decoder-only CausalLM
    "qwen3": QWEN3_CONFIG,
    "llama": LLAMA_CONFIG,
    "qwen3_tied": QWEN3_TIED_CONFIG,
    "qwen3_06b": QWEN3_06B_CONFIG,
    "qwen3_17b": QWEN3_17B_CONFIG,
    "granite_40": GRANITE_40_CONFIG,
    "lfm_700m": LFM_700M_CONFIG,
    # Embedding models (base without LM head)
    "qwen3_embedding": QWEN3_EMBEDDING_CONFIG,
    "embedding_gemma": EMBEDDING_GEMMA_CONFIG,
    "kalm_embedding": KALM_EMBEDDING_CONFIG,
    # Encoder-only
    "multilingual_e5": MULTILINGUAL_E5_CONFIG,
    # Custom models
    "gliner2": GLINER2_CONFIG,
    # Encoder-decoder
    "t5gemma2": T5GEMMA2_CONFIG,
    # External KV cache mode (reuses qwen3 config)
    "qwen3_ext": QWEN3_CONFIG,
}

# Mapping: config_name -> meson executable target name.
# Must match the entries in TorchFXConverter/jni/meson.build.
MODEL_BUILD_TARGETS = {
    "qwen3": "converter_qwen3_gen_test",
    "llama": "converter_llama_test",
    "qwen3_tied": "converter_qwen3_tied_test",
    "qwen3_06b": "converter_qwen3_06b_test",
    "qwen3_17b": "converter_qwen3_17b_test",
    "granite_40": "converter_granite_40_test",
    "lfm_700m": "converter_lfm_700m_test",
    "qwen3_embedding": "converter_qwen3_embedding_test",
    "embedding_gemma": "converter_embedding_gemma_test",
    "kalm_embedding": "converter_kalm_embedding_test",
    "multilingual_e5": "converter_multilingual_e5_test",
    "gliner2": "converter_gliner2_test",
    "qwen3_ext": "converter_qwen3_ext_test",
}


# ---- Model creation helpers ----

def _create_local_model(config, model_dir):
    """Create a tiny local model using transformers."""
    import torch
    os.makedirs(model_dir, exist_ok=True)
    model_type = config["model_type"]
    architectures = config.get("architectures", [])
    arch = architectures[0] if architectures else ""

    try:
        # --- Decoder-only CausalLM ---
        if model_type == "qwen3" and arch == "Qwen3ForCausalLM":
            from transformers import Qwen3Config, Qwen3ForCausalLM
            cfg = Qwen3Config(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            model = Qwen3ForCausalLM(cfg)

        elif model_type == "llama" and arch == "LlamaForCausalLM":
            from transformers import LlamaConfig, LlamaForCausalLM
            cfg = LlamaConfig(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            model = LlamaForCausalLM(cfg)

        elif model_type == "granitemoehybrid":
            from transformers import (GraniteMoeHybridConfig,
                                      GraniteMoeHybridForCausalLM)
            cfg = GraniteMoeHybridConfig(**{
                k: v for k, v in config.items()
                if k not in ("architectures",)})
            cfg.architectures = config["architectures"]
            model = GraniteMoeHybridForCausalLM(cfg)

        elif model_type == "lfm2":
            from transformers import Lfm2Config, Lfm2ForCausalLM
            cfg = Lfm2Config(**{k: v for k, v in config.items()
                                if k not in ("architectures",)})
            cfg.architectures = config["architectures"]
            model = Lfm2ForCausalLM(cfg)

        # --- Embedding / base models ---
        elif model_type == "qwen3" and arch == "Qwen3Model":
            from transformers import Qwen3Config, Qwen3Model
            cfg = Qwen3Config(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            cfg.architectures = ["Qwen3Model"]
            model = Qwen3Model(cfg)

        elif model_type == "gemma2" and arch == "Gemma2Model":
            from transformers import Gemma2Config, Gemma2Model
            cfg = Gemma2Config(**{k: v for k, v in config.items()
                                  if k not in ("architectures",)})
            cfg.architectures = ["Gemma2Model"]
            model = Gemma2Model(cfg)

        elif model_type == "qwen2" and arch == "Qwen2Model":
            from transformers import Qwen2Config, Qwen2Model
            cfg = Qwen2Config(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            cfg.architectures = ["Qwen2Model"]
            model = Qwen2Model(cfg)

        # --- Encoder-only ---
        elif model_type == "xlm-roberta":
            from transformers import XLMRobertaConfig, XLMRobertaModel
            cfg = XLMRobertaConfig(**{k: v for k, v in config.items()
                                      if k not in ("architectures",)})
            cfg.architectures = config["architectures"]
            model = XLMRobertaModel(cfg)

        # --- Custom: GLiNER2 (extractor) ---
        elif model_type == "extractor":
            return _create_gliner2_model(config, model_dir)

        # --- Encoder-Decoder: T5Gemma2 ---
        elif model_type == "t5gemma2":
            return _create_t5gemma2_model(config, model_dir)

        else:
            raise ValueError(f"Unknown model type/arch: {model_type}/{arch}")

        model.save_pretrained(model_dir)
        return True

    except ImportError as e:
        print(f"  Import error: {e}")
        return False
    except Exception as e:
        print(f"  Error creating {model_type}: {e}")
        return False


def _create_gliner2_model(config, model_dir):
    """Create a synthetic GLiNER2 extractor model with fake weights."""
    import torch
    os.makedirs(model_dir, exist_ok=True)

    hidden_size = config["hidden_size"]
    max_width = config.get("max_width", 4)

    # Save config.json
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save gliner_config.json
    gliner_cfg = {
        "span_mode": config.get("span_mode", "markerV0"),
        "max_width": max_width,
        "num_rnn_layers": config.get("num_rnn_layers", 1),
    }
    with open(os.path.join(model_dir, "gliner_config.json"), "w") as f:
        json.dump(gliner_cfg, f, indent=2)

    # Create synthetic weights matching extractor structure
    # LSTM: bidirectional, 1 layer
    state = {}
    lstm_h = hidden_size
    # LSTM weight shapes: (4*hidden_size, input_size) for ih, (4*hidden_size, hidden_size) for hh
    state["rnn.lstm.weight_ih_l0"] = torch.randn(4 * lstm_h, hidden_size)
    state["rnn.lstm.bias_ih_l0"] = torch.randn(4 * lstm_h)
    state["rnn.lstm.weight_hh_l0"] = torch.randn(4 * lstm_h, lstm_h)
    state["rnn.lstm.bias_hh_l0"] = torch.randn(4 * lstm_h)
    # Reverse direction
    state["rnn.lstm.weight_ih_l0_reverse"] = torch.randn(4 * lstm_h, hidden_size)
    state["rnn.lstm.bias_ih_l0_reverse"] = torch.randn(4 * lstm_h)
    state["rnn.lstm.weight_hh_l0_reverse"] = torch.randn(4 * lstm_h, lstm_h)
    state["rnn.lstm.bias_hh_l0_reverse"] = torch.randn(4 * lstm_h)

    # Span marker: project_start, project_end, out_project
    # Each is a 2-layer MLP: Linear(D, D) -> ReLU -> Linear(D, D)
    D = hidden_size
    for proj in ["project_start", "project_end"]:
        state[f"span_rep.{proj}.0.weight"] = torch.randn(D, D)
        state[f"span_rep.{proj}.0.bias"] = torch.randn(D)
        state[f"span_rep.{proj}.3.weight"] = torch.randn(D, D)
        state[f"span_rep.{proj}.3.bias"] = torch.randn(D)
    # out_project: Linear(2*D, D) — output per-span features that get reshaped
    state["span_rep.out_project.0.weight"] = torch.randn(D, 2 * D)
    state["span_rep.out_project.0.bias"] = torch.randn(D)

    # Classifier variant (GLiNER2 uses classifier, not prompt)
    state["classifier.0.weight"] = torch.randn(32, D)
    state["classifier.0.bias"] = torch.randn(32)
    state["classifier.3.weight"] = torch.randn(1, 32)
    state["classifier.3.bias"] = torch.randn(1)

    torch.save(state, os.path.join(model_dir, "pytorch_model.bin"))
    return True


def _create_t5gemma2_model(config, model_dir):
    """Create a tiny T5Gemma2 encoder-decoder model."""
    try:
        from transformers import (T5Gemma2Config,
                                  T5Gemma2ForConditionalGeneration)
    except ImportError:
        return False

    vocab_size = config.get("vocab_size", 1000)

    cfg = T5Gemma2Config(
        vocab_size=vocab_size,
        tie_word_embeddings=True,
    )
    # Override encoder to be tiny (text-only, no vision)
    cfg.encoder.hidden_size = 64
    cfg.encoder.intermediate_size = 128
    cfg.encoder.num_hidden_layers = 2
    cfg.encoder.num_attention_heads = 4
    cfg.encoder.num_key_value_heads = 2
    cfg.encoder.head_dim = 16
    cfg.encoder.vocab_size = vocab_size
    # Decoder
    cfg.decoder.hidden_size = 64
    cfg.decoder.intermediate_size = 128
    cfg.decoder.num_hidden_layers = 2
    cfg.decoder.num_attention_heads = 4
    cfg.decoder.num_key_value_heads = 2
    cfg.decoder.head_dim = 16
    cfg.decoder.vocab_size = vocab_size

    try:
        model = T5Gemma2ForConditionalGeneration(cfg)
        model.save_pretrained(model_dir)
        return True
    except Exception as e:
        print(f"  T5Gemma2 creation failed: {e}")
        return False


# ---- Test infrastructure ----

def _run_converter(model_dir, output_dir, model_name=None,
                    external_kv_cache=False):
    """Run converter.py and return (success, output_files)."""
    cmd = [
        sys.executable, os.path.join(CONVERTER_DIR, "converter.py"),
        "--model", model_dir,
        "--output", output_dir,
        "--format", "cpp",
    ]
    if model_name:
        cmd += ["--model-name", model_name]
    if external_kv_cache:
        cmd += ["--external-kv-cache"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0, result.stdout, result.stderr


def _build_test(build_dir, target):
    """Build a specific meson target."""
    cmd = ["ninja", "-C", build_dir, target]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0, result.stdout, result.stderr


def _run_test(build_dir, executable):
    """Run a test executable."""
    env = os.environ.copy()
    lib_paths = [
        os.path.join(build_dir, "nntrainer"),
        os.path.join(build_dir, "Applications", "CausalLM", "layers"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)
    cmd = [os.path.join(build_dir, executable)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=60, env=env)
    return result.returncode == 0, result.stdout, result.stderr


def _meson_reconfigure(build_dir):
    """Reconfigure meson to pick up new/removed source files."""
    cmd = ["meson", "setup", "--reconfigure", build_dir]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=120, cwd=NNTRAINER_ROOT)
    return result.returncode == 0, result.stdout, result.stderr


class TestConverterBuildAndRun(unittest.TestCase):
    """Test that converter output compiles and runs with NNTrainer."""

    @classmethod
    def setUpClass(cls):
        """Check prerequisites."""
        if not os.path.isdir(BUILD_DIR):
            raise unittest.SkipTest(
                f"Build directory not found: {BUILD_DIR}. "
                f"Run: meson setup builddir -Denable-transformer=true")

    # ---- Decoder-only CausalLM ----

    def test_qwen3_tiny_build_and_run(self):
        """Qwen3 tiny model: convert -> build -> initialize."""
        self._run_model_test("qwen3")

    def test_llama_tiny_build_and_run(self):
        """LLaMA tiny model: convert -> build -> initialize."""
        self._run_model_test("llama")

    def test_qwen3_tied_embeddings(self):
        """Qwen3 with tied word embeddings."""
        self._run_model_test("qwen3_tied")

    def test_qwen3_06b(self):
        """Qwen3-0.6B style: tied embeddings, high rope_theta."""
        self._run_model_test("qwen3_06b")

    def test_qwen3_17b(self):
        """Qwen3-1.7B style: more heads, larger intermediate."""
        self._run_model_test("qwen3_17b")

    def test_granite_40(self):
        """Granite 4.0: GraniteMoeHybrid in dense mode (no MoE/Mamba)."""
        self._run_model_test("granite_40")

    def test_lfm_700m(self):
        """LFM-700M: Liquid Foundation Model (lfm2) with hybrid conv+attention."""
        self._run_model_test("lfm_700m")

    # ---- Embedding models ----

    def test_qwen3_embedding(self):
        """Qwen3-Embedding-0.6B: Qwen3 base model for embeddings."""
        self._run_model_test("qwen3_embedding")

    def test_embedding_gemma(self):
        """EmbeddingGemma-300M: Gemma2 base model for embeddings."""
        self._run_model_test("embedding_gemma")

    def test_kalm_embedding(self):
        """KaLM-Embedding-v2.5: Qwen2 base model for embeddings."""
        self._run_model_test("kalm_embedding")

    # ---- Encoder-only ----

    def test_multilingual_e5(self):
        """Multilingual-E5-tiny-instruct: XLM-RoBERTa encoder."""
        self._run_model_test("multilingual_e5")

    # ---- Custom models ----

    def test_gliner2(self):
        """GLiNER2-multi-v1: custom extractor model (NER)."""
        self._run_model_test("gliner2")

    # ---- External KV cache mode ----

    @unittest.expectedFailure
    def test_qwen3_external_kv_cache(self):
        """Qwen3 with --external-kv-cache: convert -> build -> initialize.
        Expected failure: MHA layer accepts 3-4 inputs, external KV needs 5."""
        self._run_model_test("qwen3_ext", external_kv_cache=True)

    # ---- Encoder-Decoder ----

    @unittest.skip("T5Gemma2 is multimodal (text+vision) with deeply nested "
                   "configs; cannot create a memory-efficient tiny model")
    def test_t5gemma2(self):
        """T5Gemma2-270M: Gemma2-based encoder-decoder."""
        self._run_model_test("t5gemma2")

    # ---- Non-HuggingFace models (conversion-only) ----

    def test_flux_conversion(self):
        """FLUX DiT: FX trace and full layer mapping."""
        self._run_custom_conversion_test("flux", FLUX_CONFIG)

    def test_siglip_conversion(self):
        """SigLIP: FX trace and full layer mapping."""
        self._run_custom_conversion_test("siglip", SIGLIP_CONFIG)

    def test_conformer_conversion(self):
        """Conformer: FX trace and full layer mapping."""
        self._run_custom_conversion_test("conformer", CONFORMER_CONFIG)

    def test_zipformer_conversion(self):
        """Zipformer: FX trace and full layer mapping."""
        self._run_custom_conversion_test("zipformer", ZIPFORMER_CONFIG)

    def test_gemma3n_conversion(self):
        """Gemma3n: FX trace and full layer mapping (AltUp + Laurel)."""
        self._run_custom_conversion_test("gemma3n", GEMMA3N_CONFIG)

    def test_yolov2_conversion(self):
        """YOLOv2: FX trace and full layer mapping (CNN + reorg + concat)."""
        self._run_custom_conversion_test("yolov2", YOLOV2_CONFIG)

    def test_yolov3_conversion(self):
        """YOLOv3: FX trace and full layer mapping (Darknet53 + FPN)."""
        self._run_custom_conversion_test("yolov3", YOLOV3_CONFIG)

    def test_resnet18_conversion(self):
        """ResNet18: FX trace and full layer mapping (BasicBlock residual)."""
        self._run_custom_conversion_test("resnet18", RESNET18_CONFIG)

    def test_resnet50_conversion(self):
        """ResNet50: FX trace and full layer mapping (Bottleneck residual)."""
        self._run_custom_conversion_test("resnet50", RESNET50_CONFIG)

    def test_vit_conversion(self):
        """ViT-B/16: FX trace and full layer mapping (patch embed + MHA)."""
        self._run_custom_conversion_test("vit", VIT_CONFIG)

    def test_deit_conversion(self):
        """DeiT: FX trace and full layer mapping (distillation token + SDPA)."""
        self._run_custom_conversion_test("deit", DEIT_CONFIG)

    def test_dcgan_gen_conversion(self):
        """DCGAN Generator: FX trace and full layer mapping (ConvTranspose2d)."""
        self._run_custom_conversion_test("dcgan_gen", DCGAN_GEN_CONFIG)

    def test_dcgan_disc_conversion(self):
        """DCGAN Discriminator: FX trace and full layer mapping (LeakyReLU)."""
        self._run_custom_conversion_test("dcgan_disc", DCGAN_DISC_CONFIG)

    def test_pix2pix_conversion(self):
        """Pix2Pix U-Net: FX trace and full layer mapping (skip connections)."""
        self._run_custom_conversion_test("pix2pix", PIX2PIX_CONFIG)

    def test_mobilenet_v2_conversion(self):
        """MobileNetV2: FX trace and full layer mapping (inverted residual + hardtanh)."""
        self._run_custom_conversion_test("mobilenet_v2", MOBILENET_V2_CONFIG)

    def test_mobilenet_v3_conversion(self):
        """MobileNetV3-Small: FX trace (hardswish + hardsigmoid + SE blocks)."""
        self._run_custom_conversion_test("mobilenet_v3", MOBILENET_V3_CONFIG)

    def test_densenet121_conversion(self):
        """DenseNet121: FX trace and full layer mapping (dense connections)."""
        self._run_custom_conversion_test("densenet121", DENSENET121_CONFIG)

    def test_inception_v3_conversion(self):
        """InceptionV3: FX trace (multi-branch parallel convolutions)."""
        self._run_custom_conversion_test("inception_v3", INCEPTION_V3_CONFIG)

    def test_convnext_conversion(self):
        """ConvNeXt-Tiny: FX trace (LayerNorm + GELU + permute function)."""
        self._run_custom_conversion_test("convnext", CONVNEXT_CONFIG)

    def test_swin_conversion(self):
        """Swin Transformer: FX trace (shifted window attention + roll)."""
        self._run_custom_conversion_test("swin", SWIN_CONFIG)

    def test_maxvit_conversion(self):
        """MaxViT-T: FX trace (multi-axis attention + swapaxes)."""
        self._run_custom_conversion_test("maxvit", MAXVIT_CONFIG)

    def _run_custom_conversion_test(self, name, config_dict):
        """Run conversion-only test for custom (non-HF) models.

        Creates a model via the custom loader, runs AdaptiveConverter,
        and verifies all layers are fully mapped (no unknowns).
        """
        import argparse
        import torch
        from decomposer import AdaptiveConverter
        from custom_models import CUSTOM_LOADERS

        model_type = config_dict["model_type"]
        config = argparse.Namespace(**config_dict)

        # Special handling per model type (uses HF classes, not custom loader)
        if model_type == "siglip":
            self._test_siglip_conversion(config)
            return
        if model_type == "gemma3n_text":
            self._test_gemma3n_conversion(config)
            return

        if model_type not in CUSTOM_LOADERS:
            self.skipTest(f"No custom loader for {model_type}")

        # Generic custom loader path
        loader = CUSTOM_LOADERS[model_type]
        try:
            model, config, input_kwargs = loader(
                "/tmp/test_" + name, config, seq_len=16, verbose=True)
        except ImportError as e:
            self.skipTest(f"Missing dependency for {name}: {e}")

        converter = AdaptiveConverter(model, config)
        result = converter.convert(input_kwargs)

        print(f"\n[{name}] Conversion result:")
        print(f"  Total layers: {len(result.layers)}")
        print(f"  Unknown layers: {len(result.unknown_layers)}")
        print(f"  Fully mapped: {result.is_fully_mapped}")

        self.assertTrue(result.is_fully_mapped,
                        f"{name} has {len(result.unknown_layers)} unknown "
                        f"layers: {[l.layer_type for l in result.unknown_layers]}")
        self.assertGreater(len(result.layers), 0,
                           f"{name} produced no layers")

    def _test_siglip_conversion(self, config):
        """SigLIP needs transformers SiglipModel, not custom loader."""
        try:
            from transformers import SiglipConfig, SiglipModel
        except ImportError:
            self.skipTest("transformers SiglipModel not available")

        import torch
        from decomposer import AdaptiveConverter

        siglip_config = SiglipConfig(
            vision_config=dict(
                hidden_size=config.vision_hidden_size,
                intermediate_size=config.vision_intermediate_size,
                num_hidden_layers=config.vision_num_hidden_layers,
                num_attention_heads=config.vision_num_attention_heads,
                image_size=config.vision_image_size,
                patch_size=config.vision_patch_size,
                num_channels=3,
            ),
            text_config=dict(
                hidden_size=config.text_hidden_size,
                intermediate_size=config.text_intermediate_size,
                num_hidden_layers=config.text_num_hidden_layers,
                num_attention_heads=config.text_num_attention_heads,
                vocab_size=config.text_vocab_size,
                max_position_embeddings=config.text_max_position_embeddings,
            ),
        )
        model = SiglipModel(siglip_config)
        model.eval()

        input_kwargs = {
            "input_ids": torch.randint(0, config.text_vocab_size, (1, 8)),
            "pixel_values": torch.randn(
                1, 3, config.vision_image_size, config.vision_image_size),
        }

        converter = AdaptiveConverter(model, siglip_config)
        result = converter.convert(input_kwargs)

        print(f"\n[siglip] Conversion result:")
        print(f"  Total layers: {len(result.layers)}")
        print(f"  Unknown layers: {len(result.unknown_layers)}")
        print(f"  Fully mapped: {result.is_fully_mapped}")

        self.assertTrue(result.is_fully_mapped,
                        f"siglip has {len(result.unknown_layers)} unknown "
                        f"layers: {[l.layer_type for l in result.unknown_layers]}")
        self.assertGreater(len(result.layers), 0,
                           "siglip produced no layers")

    def _test_gemma3n_conversion(self, config):
        """Gemma3n needs Gemma3nTextModel from transformers."""
        try:
            from transformers.models.gemma3n.modeling_gemma3n import (
                Gemma3nTextModel,
            )
            from transformers import Gemma3nTextConfig
        except ImportError:
            self.skipTest("transformers Gemma3nTextModel not available")

        import torch
        from decomposer import AdaptiveConverter

        cfg = Gemma3nTextConfig(
            vocab_size=config.vocab_size,
            vocab_size_per_layer_input=config.vocab_size_per_layer_input,
            hidden_size=config.hidden_size,
            hidden_size_per_layer_input=config.hidden_size_per_layer_input,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            hidden_activation=config.hidden_activation,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=config.sliding_window,
            altup_num_inputs=config.altup_num_inputs,
            altup_active_idx=config.altup_active_idx,
            num_kv_shared_layers=config.num_kv_shared_layers,
            laurel_rank=config.laurel_rank,
        )
        model = Gemma3nTextModel(cfg)
        model.eval()

        input_kwargs = {
            "input_ids": torch.randint(0, config.vocab_size, (1, 8)),
        }

        converter = AdaptiveConverter(model, cfg)
        result = converter.convert(input_kwargs)

        print(f"\n[gemma3n] Conversion result:")
        print(f"  Total layers: {len(result.layers)}")
        print(f"  Unknown layers: {len(result.unknown_layers)}")
        print(f"  Fully mapped: {result.is_fully_mapped}")

        self.assertTrue(result.is_fully_mapped,
                        f"gemma3n has {len(result.unknown_layers)} unknown "
                        f"layers: {[l.layer_type for l in result.unknown_layers]}")
        self.assertGreater(len(result.layers), 0,
                           "gemma3n produced no layers")

    # ---- Common pipeline ----

    def _run_model_test(self, config_name, external_kv_cache=False):
        """Run full pipeline: create model -> convert -> build -> run.

        Steps:
          1. Create a tiny HuggingFace model in a temp directory
          2. Run converter.py to generate C++ header + source
          3. Copy generated files to jni/ directory
          4. Reconfigure meson to detect the new source files
          5. Build the model-specific test executable with ninja
          6. Run the executable and verify NNTrainer initializes it
        """
        config = MODEL_CONFIGS[config_name]
        generated_files = []

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_dir = os.path.join(tmpdir, "model")
                output_dir = os.path.join(tmpdir, "output")

                # Step 1: Create local model
                if not _create_local_model(config, model_dir):
                    self.skipTest("transformers not available or model "
                                  "creation failed")

                # Step 2: Convert
                ok, stdout, stderr = _run_converter(
                    model_dir, output_dir, model_name=config_name,
                    external_kv_cache=external_kv_cache)
                self.assertTrue(ok, f"Converter failed:\n{stderr}")

                # Step 3: Verify output files exist
                cpp_files = [f for f in os.listdir(output_dir)
                             if f.endswith(('.h', '.cpp'))]
                self.assertGreaterEqual(len(cpp_files), 2,
                                        f"Expected .h and .cpp, got: "
                                        f"{cpp_files}")

                print(f"\n[{config_name}] Converter output:")
                print(stdout)

                # Step 4: Copy generated C++ files to jni/ for meson build
                if config_name not in MODEL_BUILD_TARGETS:
                    return  # no build target defined, conversion-only test

                for f in cpp_files:
                    src = os.path.join(output_dir, f)
                    dst = os.path.join(JNI_DIR, f)
                    shutil.copy2(src, dst)
                    generated_files.append(dst)

            # Step 5: Reconfigure meson to pick up the new files
            ok, stdout, stderr = _meson_reconfigure(BUILD_DIR)
            self.assertTrue(ok,
                            f"meson reconfigure failed:\n{stderr[-500:]}")

            # Step 6: Build the test executable
            target_name = MODEL_BUILD_TARGETS[config_name]
            target = (f"tools/TorchFXConverter/jni/"
                      f"{target_name}")
            ok, stdout, stderr = _build_test(BUILD_DIR, target)
            self.assertTrue(ok, f"Build failed for {config_name}:\n"
                            f"stdout: {stdout[-1000:]}\n"
                            f"stderr: {stderr[-1000:]}")

            # Step 7: Run the executable
            ok, stdout, stderr = _run_test(BUILD_DIR, target)
            self.assertTrue(ok,
                            f"Run failed for {config_name}:\n"
                            f"stdout: {stdout}\nstderr: {stderr}")
            self.assertIn("Model initialized successfully", stdout)

        finally:
            # Cleanup: remove generated files from jni/
            for f in generated_files:
                if os.path.exists(f):
                    os.remove(f)

    def test_prebuilt_qwen3_executable(self):
        """Run the pre-built converter_qwen3_test executable."""
        target = "tools/TorchFXConverter/jni/converter_qwen3_test"
        exe_path = os.path.join(BUILD_DIR, target)

        if not os.path.isfile(exe_path):
            # Try building it
            ok, _, stderr = _build_test(BUILD_DIR, target)
            if not ok:
                self.skipTest(f"Could not build {target}: {stderr[-200:]}")

        ok, stdout, stderr = _run_test(BUILD_DIR, target)
        self.assertTrue(ok,
                        f"Executable failed:\nstdout: {stdout}\nstderr: {stderr}")
        self.assertIn("Model initialized successfully", stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
