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

## Data Feeding Order

**CRITICAL**: In NNTrainer, the order in which you must provide input and label data is determined **during compilation**. It does not always match the order in which layers were added to the model.

To determine the correct order:
1.  Compile the model (`model->compile()`).
2.  Check the printed model summary (`model->summarize()`).
3.  Identify the order of layers with type `input` and `output` layers

In this specific example, the compilation order typically requires **Input 1 (1:4:2)** to be fed before **Input 0 (1:1:2)**, despite `Input 0` being defined first in the code.
