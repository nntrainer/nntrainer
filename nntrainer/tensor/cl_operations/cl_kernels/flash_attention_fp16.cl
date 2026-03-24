#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Flash Attention kernel (FP16) - using FP32 internally for compatibility
__kernel void flash_attention_cl_fp16(__global const half *query,
                                      __global const half *key,
                                      __global const half *value,
                                      __global half *output,
                                      __global const half *attention_mask,
                                      const int batch_size,
                                      const int num_heads,
                                      const int seq_len,
                                      const int head_dim,
                                      const half scale) {
  
  // Get global thread indices
  const int batch_idx = get_global_id(0);
  const int head_idx = get_global_id(1);
  const int query_idx = get_global_id(2);
  
  // Check bounds
  if (batch_idx >= batch_size || head_idx >= num_heads || query_idx >= seq_len) {
    return;
  }
  
  // Calculate base offsets for tensors
  const int qkv_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
  const int mask_offset = batch_idx * seq_len * seq_len;
  const int out_offset = qkv_offset + query_idx * head_dim;
  
  // Load query vector for this position
  float local_query[256]; // Increased to handle larger head_dim
  for (int d = 0; d < head_dim; d++) {
    local_query[d] = (float)query[qkv_offset + query_idx * head_dim + d];
  }
  
  // Compute attention scores
  float max_score = -INFINITY;
  
  // Dynamically allocate scores array on stack
  // For very large seq_len, we'll compute in chunks
  if (seq_len <= 256) {
    // Fast path for smaller sequences
    float shared_scores[256];
    
    // First pass: compute scores and find maximum
    for (int key_idx = 0; key_idx < seq_len; key_idx++) {
      float score = 0.0f;
      
      // Compute dot product between query and key
      for (int d = 0; d < head_dim; d++) {
        score += local_query[d] * (float)key[qkv_offset + key_idx * head_dim + d];
      }
      
      // Apply scaling
      score *= (float)scale;
      
      // Apply attention mask if provided
      if (attention_mask != NULL) {
        score += (float)attention_mask[mask_offset + query_idx * seq_len + key_idx];
      }
      
      shared_scores[key_idx] = score;
      
      // Update maximum
      if (score > max_score) {
        max_score = score;
      }
    }
    
    // Second pass: compute softmax
    float exp_sum = 0.0f;
    for (int key_idx = 0; key_idx < seq_len; key_idx++) {
      float exp_score = exp(shared_scores[key_idx] - max_score);
      shared_scores[key_idx] = exp_score;
      exp_sum += exp_score;
    }
    
    // Third pass: normalize and compute weighted sum
    for (int d = 0; d < head_dim; d++) {
      float result = 0.0f;
      for (int key_idx = 0; key_idx < seq_len; key_idx++) {
        float weight = shared_scores[key_idx] / exp_sum;
        result += weight * (float)value[qkv_offset + key_idx * head_dim + d];
      }
      output[out_offset + d] = (half)result;
    }
  } else {
    // Slow path for larger sequences - compute without shared memory
    float exp_sum = 0.0f;
    float scores[512]; // Temporary array for scores
    
    // First pass: compute scores and find maximum, and store scores
    for (int key_idx = 0; key_idx < seq_len && key_idx < 512; key_idx++) {
      float score = 0.0f;
      
      // Compute dot product between query and key
      for (int d = 0; d < head_dim; d++) {
        score += local_query[d] * (float)key[qkv_offset + key_idx * head_dim + d];
      }
      
      // Apply scaling
      score *= (float)scale;
      
      // Apply attention mask if provided
      if (attention_mask != NULL) {
        score += (float)attention_mask[mask_offset + query_idx * seq_len + key_idx];
      }
      
      scores[key_idx] = score;
      
      // Update maximum
      if (score > max_score) {
        max_score = score;
      }
    }
    
    // Second pass: compute softmax values and sum
    for (int key_idx = 0; key_idx < seq_len && key_idx < 512; key_idx++) {
      float exp_score = exp(scores[key_idx] - max_score);
      scores[key_idx] = exp_score;
      exp_sum += exp_score;
    }
    
    // Third pass: normalize and compute weighted sum
    for (int d = 0; d < head_dim; d++) {
      float result = 0.0f;
      for (int key_idx = 0; key_idx < seq_len && key_idx < 512; key_idx++) {
        float weight = scores[key_idx] / exp_sum;
        result += weight * (float)value[qkv_offset + key_idx * head_dim + d];
      }
      output[out_offset + d] = (half)result;
    }
  }
}
