## @file weight_converter.py
## @brief weight conversion script for qwen3_moe model

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

total_size = 0
def save_qwen3_moe_for_nntrainer(params, config, dtype, file):  
    """Convert and save weights as nntrainer format for multi-head attention model"""  

    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
      
    def save_weight(weight_name, is_transpose=False):
        print(weight_name, params[weight_name].shape)
        if is_transpose:
            np.array(params[weight_name].permute(1,0), dtype=dtype).tofile(file)  
        else:
            np.array(params[weight_name], dtype=dtype).tofile(file)  

    def save_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight saving"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(f"{layer_name}{proj_name}.base_layer.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True)  
        else:  
            save_weight(f"{layer_name}{proj_name}.weight", True)  

    def save_attention(layer_name):  
        """Save attention layer weights"""  
          
        # Save Q/K/V/O projections using helper  
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:  
            save_projection(layer_name, f"self_attn.{proj}")  
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(proj_norm_name)

    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""  
        
        save_weight(f"{layer_name}mlp.gate.weight", True)  
          
        # Save MoE projections using helper  
        for expert_id in range(n_experts):
            for proj in ["up_proj", "gate_proj", "down_proj"]:  
                save_projection(layer_name, f"mlp.experts.{expert_id}.{proj}")  

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

    # Save final layers  
    save_weight("model.norm.weight")  
    save_weight("lm_head.weight", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Qwen3-4b")
    parser.add_argument("--output_name", type=str, default="./nntr_qwen3_4b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float", trust_remote_code=True)
    model.eval()

    with open(output_name, "wb") as f_model :
        save_qwen3_moe_for_nntrainer(model.state_dict(), config, data_dtype, f_model)
