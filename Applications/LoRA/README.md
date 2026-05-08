# LoRA Implementation in NNTrainer

This directory contains implementations of LoRA (Low-Rank Adaptation) training and standard backpropagation training to demonstrate the performance differences between full fine-tuning and parameter-efficient fine-tuning.

## Files

- `backprop_train.cpp`: Standard backpropagation training (baseline - full fine-tuning)
- `lora_train.cpp`: LoRA fine-tuning implementation (loads pre-trained model and trains only adapters)
- `inference.cpp`: Inference pipeline for trained models (both standard and LoRA models)
- `mnist_loader.cpp`: MNIST dataset loader utility

## Implementation Details

### Standard Backpropagation Training (`backprop_train.cpp`)
- Trains all model parameters from scratch
- Full fine-tuning approach
- Serves as the baseline for performance comparison
- Saves trained model for potential use as pre-trained model

### LoRA Fine-tuning (`lora_train.cpp`)
- Implements parameter-efficient fine-tuning using LoRA adapters
- Loads a pre-trained model and freezes original weights
- Trains only the low-rank adapter matrices (significantly fewer parameters)
- Demonstrates faster training with reduced memory requirements
- Saves LoRA-adapted model

### Inference Pipeline (`inference.cpp`)
- Loads trained models (both standard and LoRA models)
- Runs inference on test data
- Evaluates model accuracy
- Demonstrates how LoRA models can be used for inference

## Building and Running

To build the LoRA examples:

```bash
meson setup build
ninja -C build
```

To run the different implementations:

```bash
# Standard training (baseline - full fine-tuning)
./build/Applications/LoRA/jni/backprop_train

# LoRA fine-tuning (parameter-efficient)
./build/Applications/LoRA/jni/lora_train

# Inference with trained model
./build/Applications/LoRA/jni/inference
```

## Key Differences in Implementation

### Standard Training Approach
- Trains all model parameters
- Higher memory requirements
- More computational overhead
- Full fine-tuning of the entire model

### LoRA Fine-tuning Approach
- Freezes original weights during training
- Only trains low-rank adapter matrices
- Significantly fewer parameters to train
- True parameter-efficient fine-tuning
- Lower memory requirements
- Faster training times

## Performance Expectations

When properly implemented, LoRA should be:
- Faster to train than full fine-tuning
- Use less memory
- Require fewer trainable parameters
- Achieve comparable accuracy to full fine-tuning

The LoRA implementation demonstrates these benefits by freezing the base model weights and training only the adapter matrices, resulting in significant parameter efficiency.
