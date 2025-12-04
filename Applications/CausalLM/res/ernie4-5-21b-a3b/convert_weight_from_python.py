def save_ernie_moe_for_nntrainer(params, config, dtype, file):  
    """Convert and save weights as nntrainer format for multi-head attention model"""  

    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
      
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
          
        # Save Q/K/V/O projections using helper  
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:  
            save_projection(layer_name, f"self_attn.{proj}")  
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(proj_norm_name)

    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""
        
        # First Dense Layer
        if layer_name == "model.layers.0.":
            for proj in ["up_proj", "gate_proj", "down_proj"]:  
                save_projection(layer_name, f"mlp.{proj}")  

        else:
            save_weight(f"{layer_name}mlp.gate.weight", True)
            save_weight(f"{layer_name}mlp.moe_statics.e_score_correction_bias", True)
            
                        
            for proj in ["up_proj", "gate_proj", "down_proj"]:  
                    save_projection(layer_name, f"mlp.shared_experts.{proj}")  
                
            # Save MoE projections using helper  
            for expert_id in range(n_experts):
                for proj in ["up_proj", "gate_proj", "down_proj"]:  
                    save_projection(layer_name, f"mlp.experts.{expert_id}.{proj}")  


    ##### START HERE FROM INITIAL LAYER ################################
    ####################################################################
    
    # 1. Save embedding layer  
    save_weight("model.embed_tokens.weight")

    # 2. Process all layers  
    for layer_idx in range(n_layers):  
        layer_prefix = f"model.layers.{layer_idx}."  
        save_weight(f"{layer_prefix}input_layernorm.weight")  
        save_attention(layer_prefix)  
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")  
        save_feed_forward(layer_prefix)  

    # 3. Save Norm Weights
    save_weight("model.norm.weight")  
    
    print("mode save Done")
    
    ##### SAVE END HERE ################################################
    ####################################################################

########################################################################################
with open(output_name, "wb") as f_model :
        save_ernie_moe_for_nntrainer(model.state_dict(), config, data_dtype, f_model)

print("model save Success")
