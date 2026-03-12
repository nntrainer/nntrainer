#ifndef _LAYER_NORM_CL_
#define _LAYER_NORM_CL_

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


__kernel void layer_norm_fp32(
    __global const float* input,
    __global const float* gamma,
    __global const float* beta,
    __global float* output,
    __private const int batch_size,
    __private const int feature_size,
    __private const float epsilon)
{
    const int batch_idx = get_global_id(0);
    const int vec_idx = get_global_id(1);  // Vectorized index

    if (batch_idx >= batch_size) {
        return;
    }

    const int vec_width = 4;
    const int feature_vec_idx = vec_idx * vec_width;

    if (feature_vec_idx >= feature_size) {
        return;
    }

    // Calculate statistics once per batch
    __local float stats[2];  // mean, inv_stddev
    __local float local_sum[128];
    __local float local_sum_sq[128];

    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);

    if (local_id == 0) {
        // Only one work-item calculates statistics
        float sum = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            sum += input[batch_idx * feature_size + i];
        }
        float mean = sum / (float)feature_size;

        float sum_sq = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            float diff = input[batch_idx * feature_size + i] - mean;
            sum_sq += diff * diff;
        }
        float variance = sum_sq / (float)feature_size;

        stats[0] = mean;
        stats[1] = rsqrt(variance + epsilon);
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    float mean = stats[0];
    float inv_stddev = stats[1];

    // Vectorized processing
    if (feature_vec_idx + vec_width <= feature_size) {
        // Process 4 elements at once
        int base_idx = batch_idx * feature_size + feature_vec_idx;

        float4 input_vec = vload4(vec_idx, input + batch_idx * feature_size);
        float4 gamma_vec = vload4(vec_idx, gamma);  // Load gamma once per feature vector
        float4 beta_vec = vload4(vec_idx, beta);   // Load beta once per feature vector

        float4 normalized = (input_vec - (float4)mean) * (float4)inv_stddev;
        float4 result = gamma_vec * normalized + beta_vec;

        vstore4(result, vec_idx, output + batch_idx * feature_size);
    } else {
        // Handle remainder elements
        for (int i = 0; i < vec_width && (feature_vec_idx + i) < feature_size; i++) {
            int idx = batch_idx * feature_size + feature_vec_idx + i;
            float normalized = (input[idx] - mean) * inv_stddev;
            output[idx] = gamma[feature_vec_idx + i] * normalized + beta[feature_vec_idx + i];
        }
    }
}

__kernel void layer_norm_fp16(
    __global const half* input,      // Input tensor (batch_size x feature_size)
    __global const half* gamma,      // Scale parameter (feature_size)
    __global const half* beta,       // Shift parameter (feature_size)
    __global half* output,           // Output tensor (batch_size x feature_size)
    __private const int batch_size,  // Number of samples in the batch
    __private const int feature_size,// Number of features
    __private const float epsilon)    // Epsilon for numerical stability (in FP32)
{
    const int batch_idx = get_global_id(0);  // Batch index
    const int vec_idx = get_global_id(1);    // Vectorized index

    if (batch_idx >= batch_size) {
        return;
    }

    const int vec_width = 4;  // Process 4 elements at a time
    const int feature_vec_idx = vec_idx * vec_width;

    if (feature_vec_idx >= feature_size) {
        return;
    }

    // Calculate statistics once per batch
    __local float stats[2];  // mean, inv_stddev (in FP32)
    __local float local_sum[128];
    __local float local_sum_sq[128];

    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);

    if (local_id == 0) {
        // Only one work-item calculates statistics
        float sum = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            sum += as_float(input[batch_idx * feature_size + i]);  // Convert half to float
        }
        float mean = sum / (float)feature_size;

        float sum_sq = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            float diff = as_float(input[batch_idx * feature_size + i]) - mean;
            sum_sq += diff * diff;
        }
        float variance = sum_sq / (float)feature_size;

        stats[0] = mean;
        stats[1] = rsqrt(variance + epsilon);  // Compute inverse standard deviation in FP32
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    float mean = stats[0];
    float inv_stddev = stats[1];

    // Vectorized processing
    if (feature_vec_idx + vec_width <= feature_size) {
        // Process 4 elements at a time
        int base_idx = batch_idx * feature_size + feature_vec_idx;

        float4 input_vec = as_float4(vload4(vec_idx, input + batch_idx * feature_size));  // Convert half4 to float4
        float4 gamma_vec = as_float4(vload4(vec_idx, gamma));  // Convert half4 to float4
        float4 beta_vec = as_float4(vload4(vec_idx, beta));   // Convert half4 to float4

        float4 normalized = (input_vec - mean) * inv_stddev;  // Normalize in FP32
        float4 result = gamma_vec * normalized + beta_vec;    // Scale and shift in FP32

        vstore4(as_half4(result), vec_idx, output + batch_idx * feature_size);  // Convert back to half4 and store
    } else {
        // Handle remainder elements
        for (int i = 0; i < vec_width && (feature_vec_idx + i) < feature_size; i++) {
            int idx = batch_idx * feature_size + feature_vec_idx + i;
            float input_val = as_float(input[idx]);  // Convert half to float
            float normalized = (input_val - mean) * inv_stddev;  // Normalize in FP32
            float result = as_float(gamma[idx]) * normalized + as_float(beta[idx]);  // Scale and shift in FP32
            output[idx] = as_half(result);  // Convert back to half and store
        }
    }
}

#endif // _LAYER_NORM_ADRENO_CL_
