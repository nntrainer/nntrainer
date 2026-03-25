#define SOFTMAX_MIN -1e30f

__kernel void flash_attention_fp32(__global const float *query,
                                          __global const float *key,
                                          __global const float *value,
                                          __global float *output,
                                          const int seqlen_q,
                                          const int seqlen_k,
                                          const int head_dim,
                                          const int num_heads_q,
                                          const int num_heads_kv,
                                          const int batch,
                                          const float scale) {
  
  const int total_work_items = batch * num_heads_q * seqlen_q;
  const int global_id = get_global_id(0);
  
  if (global_id >= total_work_items) return;
  
  // Calculate indices
  const int batch_id = global_id / (num_heads_q * seqlen_q);
  const int head_batch_id = global_id % (num_heads_q * seqlen_q);
  const int head_id = head_batch_id / seqlen_q;
  const int q_id = head_batch_id % seqlen_q;
  
  // For GQA, map query head to corresponding key/value head
  const int kv_head_id = head_id * num_heads_kv / num_heads_q;
  
  // Calculate offsets for query (per head)
  const int query_batch_offset = batch_id * num_heads_q * seqlen_q * head_dim;
  const int query_head_offset = query_batch_offset + head_id * seqlen_q * head_dim;
  const int q_offset = query_head_offset + q_id * head_dim;
  
  // Calculate offsets for key/value (per kv_head)
  const int kv_batch_offset = batch_id * num_heads_kv * seqlen_k * head_dim;
  const int kv_head_offset = kv_batch_offset + kv_head_id * seqlen_k * head_dim;
  
  float max_val = SOFTMAX_MIN;
  
  // First pass: compute max for numerical stability
  // Process in chunks of 4 for better vectorization on Adreno
  int k_id = 0;
  for (; k_id < seqlen_k - 3; k_id += 4) {
    float4 sums = (float4)(0.0f);
    const int k_offset_0 = kv_head_offset + (k_id + 0) * head_dim;
    const int k_offset_1 = kv_head_offset + (k_id + 1) * head_dim;
    const int k_offset_2 = kv_head_offset + (k_id + 2) * head_dim;
    const int k_offset_3 = kv_head_offset + (k_id + 3) * head_dim;
    
    // Process in chunks of 4 for better vectorization on Adreno
    int d = 0;
    for (; d < head_dim - 3; d += 4) {
      float4 q_vals = vload4((q_offset + d) >> 2, query);
      float4 k_vals_0 = vload4((k_offset_0 + d) >> 2, key);
      float4 k_vals_1 = vload4((k_offset_1 + d) >> 2, key);
      float4 k_vals_2 = vload4((k_offset_2 + d) >> 2, key);
      float4 k_vals_3 = vload4((k_offset_3 + d) >> 2, key);
      
      sums.s0 += dot(q_vals, k_vals_0) * scale;
      sums.s1 += dot(q_vals, k_vals_1) * scale;
      sums.s2 += dot(q_vals, k_vals_2) * scale;
      sums.s3 += dot(q_vals, k_vals_3) * scale;
    }
    
    // Handle remaining elements
    for (; d < head_dim; d++) {
      float q_val = query[q_offset + d];
      sums.s0 += q_val * key[k_offset_0 + d] * scale;
      sums.s1 += q_val * key[k_offset_1 + d] * scale;
      sums.s2 += q_val * key[k_offset_2 + d] * scale;
      sums.s3 += q_val * key[k_offset_3 + d] * scale;
    }
    
    max_val = fmax(max_val, fmax(fmax(fmax(sums.s0, sums.s1), sums.s2), sums.s3));
  }
  
  // Handle remaining k_ids
  for (; k_id < seqlen_k; k_id++) {
    float sum = 0.0f;
    const int k_offset = kv_head_offset + k_id * head_dim;
    
    // Process in chunks of 4 for better vectorization on Adreno
    int d = 0;
    for (; d < head_dim - 3; d += 4) {
      float4 q_vals = vload4((q_offset + d) >> 2, query);
      float4 k_vals = vload4((k_offset + d) >> 2, key);
      sum += dot(q_vals, k_vals) * scale;
    }
    
    // Handle remaining elements
    for (; d < head_dim; d++) {
      float q_val = query[q_offset + d];
      float k_val = key[k_offset + d];
      sum += q_val * k_val * scale;
    }
    
    max_val = fmax(max_val, sum);
  }
  
  // Second pass: compute attention weights and output
  float4 exp_sums = (float4)(0.0f);
  
  // Process in chunks of 4 for better vectorization on Adreno
  k_id = 0;
  for (; k_id < seqlen_k - 3; k_id += 4) {
    float4 sums = (float4)(0.0f);
    const int k_offset_0 = kv_head_offset + (k_id + 0) * head_dim;
    const int k_offset_1 = kv_head_offset + (k_id + 1) * head_dim;
    const int k_offset_2 = kv_head_offset + (k_id + 2) * head_dim;
    const int k_offset_3 = kv_head_offset + (k_id + 3) * head_dim;
    
    // Process in chunks of 4 for better vectorization on Adreno
    int d = 0;
    for (; d < head_dim - 3; d += 4) {
      float4 q_vals = vload4((q_offset + d) >> 2, query);
      float4 k_vals_0 = vload4((k_offset_0 + d) >> 2, key);
      float4 k_vals_1 = vload4((k_offset_1 + d) >> 2, key);
      float4 k_vals_2 = vload4((k_offset_2 + d) >> 2, key);
      float4 k_vals_3 = vload4((k_offset_3 + d) >> 2, key);
      
      sums.s0 += dot(q_vals, k_vals_0) * scale;
      sums.s1 += dot(q_vals, k_vals_1) * scale;
      sums.s2 += dot(q_vals, k_vals_2) * scale;
      sums.s3 += dot(q_vals, k_vals_3) * scale;
    }
    
    // Handle remaining elements
    for (; d < head_dim; d++) {
      float q_val = query[q_offset + d];
      sums.s0 += q_val * key[k_offset_0 + d] * scale;
      sums.s1 += q_val * key[k_offset_1 + d] * scale;
      sums.s2 += q_val * key[k_offset_2 + d] * scale;
      sums.s3 += q_val * key[k_offset_3 + d] * scale;
    }
    
    float4 exp_vals;
    exp_vals.s0 = exp(sums.s0 - max_val);
    exp_vals.s1 = exp(sums.s1 - max_val);
    exp_vals.s2 = exp(sums.s2 - max_val);
    exp_vals.s3 = exp(sums.s3 - max_val);
    
    exp_sums += exp_vals;
    
    // Accumulate weighted values
    // Initialize output on first iteration
    if (k_id == 0) {
      for (int d = 0; d < head_dim; d++) {
        output[q_offset + d] = 0.0f;
      }
    }
    
    // Process in chunks of 4 for better vectorization on Adreno
    d = 0;
    for (; d < head_dim - 3; d += 4) {
      float4 v_vals_0 = vload4((k_offset_0 + d) >> 2, value);
      float4 v_vals_1 = vload4((k_offset_1 + d) >> 2, value);
      float4 v_vals_2 = vload4((k_offset_2 + d) >> 2, value);
      float4 v_vals_3 = vload4((k_offset_3 + d) >> 2, value);
      float4 out_vals = vload4((q_offset + d) >> 2, output);
      
      out_vals.s0 += exp_vals.s0 * v_vals_0.s0;
      out_vals.s1 += exp_vals.s0 * v_vals_0.s1;
      out_vals.s2 += exp_vals.s0 * v_vals_0.s2;
      out_vals.s3 += exp_vals.s0 * v_vals_0.s3;
      
      out_vals.s0 += exp_vals.s1 * v_vals_1.s0;
      out_vals.s1 += exp_vals.s1 * v_vals_1.s1;
      out_vals.s2 += exp_vals.s1 * v_vals_1.s2;
      out_vals.s3 += exp_vals.s1 * v_vals_1.s3;
      
      out_vals.s0 += exp_vals.s2 * v_vals_2.s0;
      out_vals.s1 += exp_vals.s2 * v_vals_2.s1;
      out_vals.s2 += exp_vals.s2 * v_vals_2.s2;
      out_vals.s3 += exp_vals.s2 * v_vals_2.s3;
      
      out_vals.s0 += exp_vals.s3 * v_vals_3.s0;
      out_vals.s1 += exp_vals.s3 * v_vals_3.s1;
      out_vals.s2 += exp_vals.s3 * v_vals_3.s2;
      out_vals.s3 += exp_vals.s3 * v_vals_3.s3;
      
      vstore4(out_vals, (q_offset + d) >> 2, output);
    }
    
    // Handle remaining elements
    for (; d < head_dim; d++) {
      float v_val_0 = value[k_offset_0 + d];
      float v_val_1 = value[k_offset_1 + d];
      float v_val_2 = value[k_offset_2 + d];
      float v_val_3 = value[k_offset_3 + d];
      float out_val = output[q_offset + d];
      output[q_offset + d] = out_val + 
        exp_vals.s0 * v_val_0 + 
        exp_vals.s1 * v_val_1 + 
        exp_vals.s2 * v_val_2 + 
        exp_vals.s3 * v_val_3;
    }
  }
  
  // Handle remaining k_ids
  float exp_sum_remaining = exp_sums.s0 + exp_sums.s1 + exp_sums.s2 + exp_sums.s3;
  for (; k_id < seqlen_k; k_id++) {
    float sum = 0.0f;
    const int k_offset = kv_head_offset + k_id * head_dim;
    
    // Process in chunks of 4 for better vectorization on Adreno
    int d = 0;
    for (; d < head_dim - 3; d += 4) {
      float4 q_vals = vload4((q_offset + d) >> 2, query);
      float4 k_vals = vload4((k_offset + d) >> 2, key);
      sum += dot(q_vals, k_vals) * scale;
    }
    
    // Handle remaining elements
    for (; d < head_dim; d++) {
      float q_val = query[q_offset + d];
      float k_val = key[k_offset + d];
      sum += q_val * k_val * scale;
    }
    
    float exp_val = exp(sum - max_val);
    exp_sum_remaining += exp_val;
    
    // Accumulate weighted values
    // Process in chunks of 4 for better vectorization on Adreno
    d = 0;
    for (; d < head_dim - 3; d += 4) {
      float4 v_vals = vload4((k_offset + d) >> 2, value);
      float4 out_vals = vload4((q_offset + d) >> 2, output);
      
      out_vals.s0 += exp_val * v_vals.s0;
      out_vals.s1 += exp_val * v_vals.s1;
      out_vals.s2 += exp_val * v_vals.s2;
      out_vals.s3 += exp_val * v_vals.s3;
      
      vstore4(out_vals, (q_offset + d) >> 2, output);
    }
    
    // Handle remaining elements
    for (; d < head_dim; d++) {
      float v_val = value[k_offset + d];
      float out_val = output[q_offset + d];
      output[q_offset + d] = out_val + exp_val * v_val;
    }
  }
  
  // Normalize by exp_sum
  float total_exp_sum = exp_sum_remaining;
  // Process in chunks of 4 for better vectorization on Adreno
  int d_norm = 0;
  for (; d_norm < head_dim - 3; d_norm += 4) {
    float4 out_vals = vload4((q_offset + d_norm) >> 2, output);
    
    out_vals.s0 /= total_exp_sum;
    out_vals.s1 /= total_exp_sum;
    out_vals.s2 /= total_exp_sum;
    out_vals.s3 /= total_exp_sum;
    
    vstore4(out_vals, (q_offset + d_norm) >> 2, output);
  }
  
  // Handle remaining elements
  for (; d_norm < head_dim; d_norm++) {
    float out_val = output[q_offset + d_norm];
    output[q_offset + d_norm] = out_val / total_exp_sum;
  }
}
