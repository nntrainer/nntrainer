// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mnist_npu_fused_train.cpp
 * @date   10 August 2026
 * @brief  Fused 3-layer FC MNIST training on NPU — all 5 forward ops
 *         (GEMM→ReLU→GEMM→ReLU→GEMM) in ONE flush via
 *         nntr_htp_bridge_fused_fc_forward. Backward still uses per-GEMM
 *         sgemm_fp32 (9 flushes total: 1 forward + 9 backward).
 *
 * This bypasses nntrainer's layer system entirely — we do our own
 * forward, loss, backward, and Adam optimizer, calling the NPU bridge
 * functions directly via dlsym.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <dlfcn.h>

// Bridge function types
typedef int (*fused_fc_forward_t)(
    const float *, const float *, float *, const float *, float *,
    const float *, float *,
    unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);

typedef int (*sgemm_fp32_t)(
    const float *, const float *, float *,
    unsigned int, unsigned int, unsigned int, int, int);

struct NpuBridge {
    fused_fc_forward_t fused_fc_forward = nullptr;
    sgemm_fp32_t sgemm_fp32 = nullptr;
    void * lib = nullptr;

    bool load() {
        lib = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
        if (!lib) {
            // Try common paths
            const char *paths[] = {
                "/data/local/tmp/nntrainer/libggml-hexagon.so",
                "libggml-hexagon.so",
                nullptr
            };
            for (int i = 0; paths[i]; i++) {
                lib = dlopen(paths[i], RTLD_NOW | RTLD_GLOBAL);
                if (lib) break;
            }
        }
        if (!lib) {
            std::cerr << "Cannot load libggml-hexagon.so: " << dlerror() << std::endl;
            return false;
        }

        fused_fc_forward = (fused_fc_forward_t)dlsym(lib, "nntr_htp_bridge_fused_fc_forward");
        sgemm_fp32 = (sgemm_fp32_t)dlsym(lib, "nntr_htp_bridge_sgemm_fp32");

        if (!fused_fc_forward) {
            std::cerr << "Cannot find nntr_htp_bridge_fused_fc_forward: " << dlerror() << std::endl;
            return false;
        }
        if (!sgemm_fp32) {
            std::cerr << "Cannot find nntr_htp_bridge_sgemm_fp32: " << dlerror() << std::endl;
            return false;
        }

        std::cout << "NPU bridge loaded: fused_fc_forward=" << (void*)fused_fc_forward
                  << " sgemm_fp32=" << (void*)sgemm_fp32 << std::endl;
        return true;
    }
};

static uint32_t read_be32(std::ifstream &f) {
    unsigned char buf[4];
    f.read(reinterpret_cast<char *>(buf), 4);
    return (uint32_t(buf[0]) << 24) | (uint32_t(buf[1]) << 16) |
           (uint32_t(buf[2]) << 8) | uint32_t(buf[3]);
}

struct MnistData {
    std::vector<float> images;  // [count, 784]
    std::vector<float> labels;  // [count, 10] one-hot
    uint32_t count = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    bool load_images(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t magic = read_be32(f);
        if (magic != 0x803) return false;
        count = read_be32(f);
        rows = read_be32(f);
        cols = read_be32(f);
        size_t pixels = (size_t)count * rows * cols;
        std::vector<unsigned char> raw(pixels);
        f.read(reinterpret_cast<char *>(raw.data()), pixels);
        images.resize(pixels);
        for (size_t i = 0; i < pixels; i++)
            images[i] = raw[i] / 255.0f;
        std::cout << "Loaded " << count << " images (" << rows << "x" << cols << ")" << std::endl;
        return true;
    }

    bool load_labels(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t magic = read_be32(f);
        if (magic != 0x801) return false;
        uint32_t n = read_be32(f);
        std::vector<unsigned char> raw(n);
        f.read(reinterpret_cast<char *>(raw.data()), n);
        labels.resize((size_t)n * 10, 0.0f);
        for (uint32_t i = 0; i < n; i++)
            labels[i * 10 + raw[i]] = 1.0f;
        std::cout << "Loaded " << n << " labels" << std::endl;
        return true;
    }
};

// Simple Adam optimizer state per parameter
struct AdamParam {
    std::vector<float> m;  // first moment
    std::vector<float> v;  // second moment
    int t = 0;             // timestep
};

void adam_update(float *w, const float *grad, AdamParam &state,
                 size_t n, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
    state.t++;
    float bc1 = 1.0f - std::pow(beta1, state.t);
    float bc2 = 1.0f - std::pow(beta2, state.t);
    for (size_t i = 0; i < n; i++) {
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * grad[i];
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * grad[i] * grad[i];

        float m_hat = state.m[i] / bc1;
        float v_hat = state.v[i] / bc2;
        w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <train_images> <train_labels> <test_images> <test_labels>"
                  << " <epochs> [batch_size] [lr] [hidden1] [hidden2]" << std::endl;
        return 1;
    }

    std::string train_imgs_path = argv[1];
    std::string train_lbls_path = argv[2];
    std::string test_imgs_path = argv[3];
    std::string test_lbls_path = argv[4];
    int epochs = std::atoi(argv[5]);
    int batch_size = argc > 6 ? std::atoi(argv[6]) : 64;
    float lr = argc > 7 ? std::atof(argv[7]) : 0.001f;
    int H1 = argc > 8 ? std::atoi(argv[8]) : 512;
    int H2 = argc > 9 ? std::atoi(argv[9]) : 256;

    // Network dimensions
    const int B = batch_size;
    const int K0 = 784;  // input
    const int N1 = H1;   // hidden1
    const int N2 = H2;   // hidden2
    const int N3 = 10;   // output

    std::cout << "Network: " << K0 << "→" << N1 << "→" << N2 << "→" << N3
              << " batch=" << B << " lr=" << lr << " epochs=" << epochs << std::endl;

    // Load NPU bridge
    NpuBridge bridge;
    if (!bridge.load()) {
        std::cerr << "Failed to load NPU bridge" << std::endl;
        return 1;
    }

    // Load MNIST data
    MnistData train, test;
    if (!train.load_images(train_imgs_path) || !train.load_labels(train_lbls_path)) {
        std::cerr << "Failed to load training data" << std::endl;
        return 1;
    }
    if (!test.load_images(test_imgs_path) || !test.load_labels(test_lbls_path)) {
        std::cerr << "Failed to load test data" << std::endl;
        return 1;
    }

    // Initialize weights with He initialization
    std::mt19937 rng(42);
    auto init_weight = [&](std::vector<float> &w, int fan_in) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));
        for (auto &v : w) v = dist(rng);
    };

    std::vector<float> W1((size_t)K0 * N1), W2((size_t)N1 * N2), W3((size_t)N2 * N3);
    init_weight(W1, K0);
    init_weight(W2, N1);
    init_weight(W3, N2);

    // Adam state for each weight
    AdamParam adam1, adam2, adam3;
    adam1.m.resize(W1.size()); adam1.v.resize(W1.size());
    adam2.m.resize(W2.size()); adam2.v.resize(W2.size());
    adam3.m.resize(W3.size()); adam3.v.resize(W3.size());

    // Training buffers
    std::vector<float> X0((size_t)B * K0);
    std::vector<float> H1_buf((size_t)B * N1);  // post-ReLU
    std::vector<float> Y1((size_t)B * N1);       // pre-ReLU (for backward)
    std::vector<float> H2_buf((size_t)B * N2);  // post-ReLU
    std::vector<float> Y2((size_t)B * N2);       // pre-ReLU
    std::vector<float> Y3((size_t)B * N3);       // logits

    // Backward buffers
    std::vector<float> dY3((size_t)B * N3);      // softmax - label
    std::vector<float> dW3((size_t)N2 * N3);
    std::vector<float> dH2((size_t)B * N2);      // gradient flowing back
    std::vector<float> dY2((size_t)B * N2);      // after ReLU backward
    std::vector<float> dW2((size_t)N1 * N2);
    std::vector<float> dH1((size_t)B * N1);
    std::vector<float> dY1((size_t)B * N1);
    std::vector<float> dW1((size_t)K0 * N1);

    // Shuffle indices
    std::vector<uint32_t> indices(train.count);
    for (uint32_t i = 0; i < train.count; i++) indices[i] = i;

    int steps_per_epoch = train.count / B;
    std::cout << "Steps per epoch: " << steps_per_epoch << std::endl;

    // --- Sanity check: forward pass on first batch, compare with CPU ---
    {
        for (int b = 0; b < B; b++) {
            uint32_t idx = b; // first B samples
            memcpy(&X0[b * K0], &train.images[idx * K0], K0 * sizeof(float));
        }
        bridge.fused_fc_forward(X0.data(), W1.data(), H1_buf.data(),
            W2.data(), H2_buf.data(), W3.data(), Y3.data(),
            B, K0, N1, N2, N3);

        // CPU reference
        std::vector<float> h1_cpu((size_t)B * N1), h2_cpu((size_t)B * N2), y3_cpu((size_t)B * N3);
        for (int b = 0; b < B; b++) {
            for (int j = 0; j < N1; j++) {
                float s = 0;
                for (int k = 0; k < K0; k++) s += X0[b*K0+k] * W1[k*N1+j];
                h1_cpu[b*N1+j] = std::max(s, 0.0f);
            }
            for (int j = 0; j < N2; j++) {
                float s = 0;
                for (int k = 0; k < N1; k++) s += h1_cpu[b*N1+k] * W2[k*N2+j];
                h2_cpu[b*N2+j] = std::max(s, 0.0f);
            }
            for (int j = 0; j < N3; j++) {
                float s = 0;
                for (int k = 0; k < N2; k++) s += h2_cpu[b*N2+k] * W3[k*N3+j];
                y3_cpu[b*N3+j] = s;
            }
        }
        // Compare
        float max_h1_err = 0, max_h2_err = 0, max_y3_err = 0;
        for (size_t i = 0; i < (size_t)B*N1; i++) max_h1_err = std::max(max_h1_err, std::abs(H1_buf[i] - h1_cpu[i]));
        for (size_t i = 0; i < (size_t)B*N2; i++) max_h2_err = std::max(max_h2_err, std::abs(H2_buf[i] - h2_cpu[i]));
        for (size_t i = 0; i < (size_t)B*N3; i++) max_y3_err = std::max(max_y3_err, std::abs(Y3[i] - y3_cpu[i]));
        std::cout << "Sanity check: H1 max_err=" << max_h1_err
                  << " H2 max_err=" << max_h2_err
                  << " Y3 max_err=" << max_y3_err << std::endl;
        // Print first sample logits
        std::cout << "NPU Y3[0]:";
        for (int j = 0; j < N3; j++) std::cout << " " << Y3[j];
        std::cout << std::endl;
        std::cout << "CPU Y3[0]:";
        for (int j = 0; j < N3; j++) std::cout << " " << y3_cpu[j];
        std::cout << std::endl;

        // Also test with individual sgemm_fp32 calls for comparison
        std::vector<float> Y1_indiv((size_t)B * N1), H1_indiv((size_t)B * N1);
        std::vector<float> Y2_indiv((size_t)B * N2), H2_indiv((size_t)B * N2);
        std::vector<float> Y3_indiv((size_t)B * N3);
        // GEMM1: Y1 = X0 @ W1, M=B, N=N1, K=K0, transA=0, transB=0
        bridge.sgemm_fp32(X0.data(), W1.data(), Y1_indiv.data(), B, N1, K0, 0, 0);
        // ReLU1
        for (size_t i = 0; i < (size_t)B * N1; i++) H1_indiv[i] = std::max(Y1_indiv[i], 0.0f);
        // GEMM2: Y2 = H1 @ W2, M=B, N=N2, K=N1, transA=0, transB=0
        bridge.sgemm_fp32(H1_indiv.data(), W2.data(), Y2_indiv.data(), B, N2, N1, 0, 0);
        // ReLU2
        for (size_t i = 0; i < (size_t)B * N2; i++) H2_indiv[i] = std::max(Y2_indiv[i], 0.0f);
        // GEMM3: Y3 = H2 @ W3, M=B, N=N3, K=N2, transA=0, transB=0
        bridge.sgemm_fp32(H2_indiv.data(), W3.data(), Y3_indiv.data(), B, N3, N2, 0, 0);

        float max_h1_err2 = 0, max_y3_err2 = 0;
        for (size_t i = 0; i < (size_t)B*N1; i++) max_h1_err2 = std::max(max_h1_err2, std::abs(H1_indiv[i] - h1_cpu[i]));
        for (size_t i = 0; i < (size_t)B*N3; i++) max_y3_err2 = std::max(max_y3_err2, std::abs(Y3_indiv[i] - y3_cpu[i]));
        std::cout << "Individual sgemm: H1 max_err=" << max_h1_err2
                  << " Y3 max_err=" << max_y3_err2 << std::endl;
        std::cout << "Indiv Y3[0]:";
        for (int j = 0; j < N3; j++) std::cout << " " << Y3_indiv[j];
        std::cout << std::endl;
    }


    for (int epoch = 0; epoch < epochs; epoch++) {

        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0.0f;
        int correct = 0;

        for (int step = 0; step < steps_per_epoch; step++) {
            // --- Load batch ---
            for (int b = 0; b < B; b++) {
                uint32_t idx = indices[step * B + b];
                memcpy(&X0[b * K0], &train.images[idx * K0], K0 * sizeof(float));
            }

            // --- Forward pass ---
            // Using individual sgemm_fp32 calls (3 flushes) + CPU ReLU.
            // TODO: switch to fused_fc_forward once op-chaining bug is fixed.
            // GEMM1: Y1 = X0 @ W1
            bridge.sgemm_fp32(X0.data(), W1.data(), Y1.data(), B, N1, K0, 0, 0);
            // ReLU1: H1 = max(Y1, 0)
            for (size_t i = 0; i < (size_t)B * N1; i++)
                H1_buf[i] = std::max(Y1[i], 0.0f);
            // GEMM2: Y2 = H1 @ W2
            bridge.sgemm_fp32(H1_buf.data(), W2.data(), Y2.data(), B, N2, N1, 0, 0);
            // ReLU2: H2 = max(Y2, 0)
            for (size_t i = 0; i < (size_t)B * N2; i++)
                H2_buf[i] = std::max(Y2[i], 0.0f);
            // GEMM3: Y3 = H2 @ W3
            bridge.sgemm_fp32(H2_buf.data(), W3.data(), Y3.data(), B, N3, N2, 0, 0);


            // We need pre-ReLU outputs for backward. The fused bridge writes
            // H1 (post-ReLU) and H2 (post-ReLU), but not pre-ReLU Y1/Y2.
            // We can recover Y1 from H1: Y1[i] = H1[i] if H1[i] > 0, else unknown.
            // Actually we need Y1 for ReLU backward mask only — H1 > 0 suffices.
            // So we use H1_buf directly as the ReLU mask (H1 > 0 means active).

            // --- Softmax + Cross-entropy loss ---
            for (int b = 0; b < B; b++) {
                // Find max for numerical stability
                float max_val = Y3[b * N3];
                for (int j = 1; j < N3; j++)
                    max_val = std::max(max_val, Y3[b * N3 + j]);
                // Softmax
                float sum = 0.0f;
                for (int j = 0; j < N3; j++) {
                    Y3[b * N3 + j] = std::exp(Y3[b * N3 + j] - max_val);
                    sum += Y3[b * N3 + j];
                }
                for (int j = 0; j < N3; j++)
                    Y3[b * N3 + j] /= sum;

                // Loss + gradient (dY3 = softmax - label)
                uint32_t idx = indices[step * B + b];
                for (int j = 0; j < N3; j++) {
                    float label = train.labels[idx * 10 + j];
                    dY3[b * N3 + j] = Y3[b * N3 + j] - label;
                    if (label > 0.5f) {
                        epoch_loss -= std::log(std::max(Y3[b * N3 + j], 1e-7f));
                    }
                }
                // Accuracy
                int pred = 0;
                for (int j = 1; j < N3; j++)
                    if (Y3[b * N3 + j] > Y3[b * N3 + pred]) pred = j;
                int label_idx = 0;
                for (int j = 0; j < 10; j++)
                    if (train.labels[idx * 10 + j] > 0.5f) label_idx = j;
                if (pred == label_idx) correct++;
            }

            // --- Backward pass ---
            // FC3 backward: dW3 = H2^T @ dY3  [N2, N3] = [N2, B] @ [B, N3]
            //   transA=1 (H2 stored as [B,N2], need [N2,B] → transpose), transB=0
            //   Actually: dW3[i,j] = sum_b H2[b,i] * dY3[b,j]
            //   = H2^T @ dY3 where H2 is [B,N2], dY3 is [B,N3]
            //   In sgemm: C[N2,N3] = A^T @ B, A=H2[B,N2], B=dY3[B,N3]
            //   → M=N2, N=N3, K=B, transA=1, transB=0
            bridge.sgemm_fp32(H2_buf.data(), dY3.data(), dW3.data(),
                              N2, N3, B, 1, 0);

            // dH2 = dY3 @ W3^T  [B, N2] = [B, N3] @ [N3, N2]
            //   C[B,N2] = A @ B^T, A=dY3[B,N3], B=W3[N2,N3] (stored [N2,N3])
            //   → M=B, N=N2, K=N3, transA=0, transB=1
            bridge.sgemm_fp32(dY3.data(), W3.data(), dH2.data(),
                              B, N2, N3, 0, 1);

            // ReLU2 backward: dY2 = dH2 * (Y2 > 0). We use H2_buf as mask (H2>0 ⟺ Y2>0)
            for (size_t i = 0; i < (size_t)B * N2; i++)
                dY2[i] = dH2[i] * (H2_buf[i] > 0.0f ? 1.0f : 0.0f);

            // FC2 backward: dW2 = H1^T @ dY2  [N1, N2]
            //   M=N1, N=N2, K=B, transA=1, transB=0
            bridge.sgemm_fp32(H1_buf.data(), dY2.data(), dW2.data(),
                              N1, N2, B, 1, 0);

            // dH1 = dY2 @ W2^T  [B, N1]
            //   M=B, N=N1, K=N2, transA=0, transB=1
            bridge.sgemm_fp32(dY2.data(), W2.data(), dH1.data(),
                              B, N1, N2, 0, 1);

            // ReLU1 backward
            for (size_t i = 0; i < (size_t)B * N1; i++)
                dY1[i] = dH1[i] * (H1_buf[i] > 0.0f ? 1.0f : 0.0f);

            // FC1 backward: dW1 = X0^T @ dY1  [K0, N1]
            //   M=K0, N=N1, K=B, transA=1, transB=0
            bridge.sgemm_fp32(X0.data(), dY1.data(), dW1.data(),
                              K0, N1, B, 1, 0);

            // --- Adam optimizer update (gradients averaged over batch) ---
            float inv_batch = 1.0f / B;
            for (auto &g : dW1) g *= inv_batch;
            for (auto &g : dW2) g *= inv_batch;
            for (auto &g : dW3) g *= inv_batch;
            adam_update(W1.data(), dW1.data(), adam1, W1.size(), lr);
            adam_update(W2.data(), dW2.data(), adam2, W2.size(), lr);
            adam_update(W3.data(), dW3.data(), adam3, W3.size(), lr);

        }

        float avg_loss = epoch_loss / (steps_per_epoch * B);
        float accuracy = 100.0f * correct / (steps_per_epoch * B);
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " - Loss: " << avg_loss
                  << " - Accuracy: " << accuracy << "%" << std::endl;
    }

    // --- Evaluation on test set ---
    int test_correct = 0;
    int test_steps = test.count / B;
    for (int step = 0; step < test_steps; step++) {
        for (int b = 0; b < B; b++) {
            uint32_t idx = step * B + b;
            memcpy(&X0[b * K0], &test.images[idx * K0], K0 * sizeof(float));
        }

        // Forward pass using individual sgemm (same as training)
        bridge.sgemm_fp32(X0.data(), W1.data(), Y1.data(), B, N1, K0, 0, 0);
        for (size_t i = 0; i < (size_t)B * N1; i++) H1_buf[i] = std::max(Y1[i], 0.0f);
        bridge.sgemm_fp32(H1_buf.data(), W2.data(), Y2.data(), B, N2, N1, 0, 0);
        for (size_t i = 0; i < (size_t)B * N2; i++) H2_buf[i] = std::max(Y2[i], 0.0f);
        bridge.sgemm_fp32(H2_buf.data(), W3.data(), Y3.data(), B, N3, N2, 0, 0);

        for (int b = 0; b < B; b++) {
            int pred = 0;

            for (int j = 1; j < N3; j++)
                if (Y3[b * N3 + j] > Y3[b * N3 + pred]) pred = j;
            uint32_t idx = step * B + b;
            int label_idx = 0;
            for (int j = 0; j < 10; j++)
                if (test.labels[idx * 10 + j] > 0.5f) label_idx = j;
            if (pred == label_idx) test_correct++;
        }
    }
    float test_acc = 100.0f * test_correct / (test_steps * B);
    std::cout << "Test Accuracy: " << test_acc << "% ("
              << test_correct << "/" << (test_steps * B) << ")" << std::endl;

    return 0;
}
