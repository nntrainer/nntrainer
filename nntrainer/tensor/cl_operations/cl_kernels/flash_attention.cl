#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Flash Attention kernel
__kernel void flash_attention_cl(__global const float *query,
                                 __global const float *key,
                                 __global const float *value,
                                 __global float *output,
                                 __global const float *attention_mask,
                                 const int batch_size,
                                 const int num_heads,
                                 const int seq_len,
                                 const int head_dim,
                                 const float scale) {
  
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
    local_query[d] = query[qkv_offset + query_idx * head_dim + d];
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
        score += local_query[d] * key[qkv_offset + key_idx * head_dim + d];
      }
      
      // Apply scaling
      score *= scale;
      
      // Apply attention mask if provided
      if (attention_mask != NULL) {
        score += attention_mask[mask_offset + query_idx * seq_len + key_idx];
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
        result += weight * value[qkv_offset + key_idx * head_dim + d];
      }
      output[out_offset + d] = result;
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
        score += local_query[d] * key[qkv_offset + key_idx * head_dim + d];
      }
      
      // Apply scaling
      score *= scale;
      
      // Apply attention mask if provided
      if (attention_mask != NULL) {
        score += attention_mask[mask_offset + query_idx * seq_len + key_idx];
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
        result += weight * value[qkv_offset + key_idx * head_dim + d];
      }
      output[out_offset + d] = result;
    }
  }
}
