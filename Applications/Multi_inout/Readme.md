# Multi-Input/Output Example

This example demonstrates how to build and train a model with multiple inputs and multiple outputs using NNTrainer. It specifically addresses how to handle data feeding for such architectures.

## Model Structure

The model consists of two branches that process different inputs and merge them before producing two separate outputs.

```text
          +----------+      +----------+
          | Output 0 |      | Output 1 |
          +----------+      +----------+
                |                 |
    +---------------------------------------------------+
    |                      flatten                      |
    +---------------------------------------------------+
                              |
    +---------------------------------------------------+
    |                      concat0                      |
    +---------------------------------------------------+
        |                     |
    +-----------+       +-----------+
    | shared_fc |       |shared_lstm|
    +-----------+       +-----------+
        |                     |
    +-----------+       +-----------+
    |  Input 0  |       |  Input 1  |
    +-----------+       +-----------+
```

## Purpose
The purpose of this example is to demonstrate how to build a model with multiple inputs and multiple outputs using the NNTrainer C++ API.
It also shows how to align the C++ implementation with a Python (PyTorch) equivalent, ensuring that data feeding, model structure, and inference results are consistent.
