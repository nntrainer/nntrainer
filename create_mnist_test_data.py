#!/usr/bin/env python3
"""
Create minimal MNIST IDX format test data for NPU training validation.
Generates 100 synthetic samples (10 per digit) with simple patterns.
"""

import struct
import numpy as np

def create_idx3(filename, data):
    """Write IDX format image file."""
    with open(filename, 'wb') as f:
        magic = 0x00000803
        f.write(struct.pack('>I', magic))
        f.write(struct.pack('>I', data.shape[0]))  # num images
        f.write(struct.pack('>I', data.shape[1]))  # rows
        f.write(struct.pack('>I', data.shape[2]))  # cols
        f.write(data.astype(np.uint8).tobytes())

def create_idx1(filename, labels):
    """Write IDX format label file."""
    with open(filename, 'wb') as f:
        magic = 0x00000801
        f.write(struct.pack('>I', magic))
        f.write(struct.pack('>I', len(labels)))
        f.write(np.array(labels, dtype=np.uint8).tobytes())

# Create 100 samples (10 per digit), each 28x28
num_samples = 100
images = np.zeros((num_samples, 28, 28), dtype=np.float32)
labels = []

for digit in range(10):
    for i in range(10):
        idx = digit * 10 + i
        labels.append(digit)
        # Create simple pattern: digit value determines which region is bright
        # Add some noise to make it look more realistic
        img = np.random.uniform(0.1, 0.3, (28, 28)).astype(np.float32)
        
        # Make a simple pattern based on digit
        if digit < 5:
            # Horizontal bands
            for band in range(digit + 1):
                y = (band + 1) * 4
                img[y:y+3, :] = np.random.uniform(0.7, 1.0, (3, 28)).astype(np.float32)
        else:
            # Vertical bands
            for band in range(digit - 4):
                x = (band + 1) * 3
                img[:, x:x+3] = np.random.uniform(0.7, 1.0, (28, 3)).astype(np.float32)
        
        images[idx] = img

# Scale to 0-255 for IDX format
images_uint8 = (images * 255).astype(np.uint8)

# Create train and test sets (use same data for simplicity)
create_idx3('/home/anirudh/nntrainer/mnist_train_images.idx3-ubyte', images_uint8)
create_idx1('/home/anirudh/nntrainer/mnist_train_labels.idx1-ubyte', labels)
create_idx3('/home/anirudh/nntrainer/mnist_test_images.idx3-ubyte', images_uint8)
create_idx1('/home/anirudh/nntrainer/mnist_test_labels.idx1-ubyte', labels)

print("Created MNIST test data:")
print(f"  - mnist_train_images.idx3-ubyte: {num_samples} images")
print(f"  - mnist_train_labels.idx1-ubyte: {num_samples} labels")
print(f"  - mnist_test_images.idx3-ubyte: {num_samples} images")
print(f"  - mnist_test_labels.idx1-ubyte: {num_samples} labels")
