import torch
import sys
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig

print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)

data_dtype = "float32"
model_path = "./deepseek"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "deepseek"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype="float32")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
config = AutoConfig.from_pretrained(model_name)
print("model load done")


def save_deep_seek_v2_lite_chat_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer format for multi-head attention model"""

    n_layers = config.num_hidden_layers
    n_experts = config.n_routed_experts

    def save_weight(weight_name, is_transpose=False):
        print(weight_name, params[weight_name].shape, "dtype = ", dtype )
        if is_transpose:
            np.array(params[weight_name].permute(1,0), dtype=dtype).tofile(file)
        else:
            np.array(params[weight_name], dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        save_weight(f"{layer_name}{proj_name}.weight", True)

    def save_attention(layer_name):
        """Save attention layer weights"""

        save_weight(f"{layer_name}self_attn.kv_a_layernorm.weight")

        # Save Q/K/V/O projections using helper
        for proj in ["q_proj", "o_proj", "kv_a_proj_with_mqa", "kv_b_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")


    def save_feed_forward(layer_name):
        """Save feed forward layer weights"""

        if layer_name == "model.layers.0.":
            for proj in ["up_proj", "gate_proj", "down_proj"]:
                save_projection(layer_name, f"mlp.{proj}")

        else:
            save_weight(f"{layer_name}mlp.gate.weight", True)

            #Save Shared Experts per Layer
            for proj in ["up_proj", "gate_proj", "down_proj"]:
                save_projection(layer_name, f"mlp.shared_experts.{proj}")

                # Save MoE projections using helper
            for expert_id in range(n_experts):
                for proj in ["up_proj", "gate_proj", "down_proj"]:
                    save_projection(layer_name, f"mlp.experts.{expert_id}.{proj}")


                    ##### START HERE FROM INITIAL LAYER ################################
    ####################################################################
    # Save embedding layer
    save_weight("model.embed_tokens.weight")

    # Process all layers
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_weight(f"{layer_prefix}input_layernorm.weight")
        save_attention(layer_prefix)
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")
        save_feed_forward(layer_prefix)

        # Save Norm Weights
    save_weight("model.norm.weight")
    save_weight("lm_head.weight")

    ##### SAVE END HERE ################################################
    ####################################################################

########################################################################################



with open(f"./nntr_deepseek_v2_lite_moe.bin", "wb") as f_model :
    save_deep_seek_v2_lite_chat_for_nntrainer(model.state_dict(), config, data_dtype, f_model)

print("Save Done")