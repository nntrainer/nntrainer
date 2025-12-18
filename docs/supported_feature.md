### Supported Optimizers

NNTrainer Provides

| Keyword | Optimizer Name | Description |
 |:-------:|:---:|:---:|
| sgd | Stochastic Gradient Decent | - |
| adam | Adaptive Moment Estimation | - |
| adamw | Adam with decoupled weight decay regularization | - |

| Keyword | Learning Rate | Description |
 |:-------:|:---:|:---:|
| exponential | exponential learning rate decay | - |
| constant | constant learning rate | - |
| step | step learning rate | - |

### Supported Loss Functions

NNTrainer provides

| Keyword | Class Name | Description |
 |:-------:|:---:|:---:|
| cross_sigmoid | CrossEntropySigmoidLossLayer | Cross entropy sigmoid loss layer |
| cross_softmax | CrossEntropySoftmaxLossLayer | Cross entropy softmax loss layer |
| constant_derivative | ConstantDerivativeLossLayer | Constant derivative loss layer |
| mse | MSELossLayer | Mean square error loss layer |
| kld | KLDLossLayer | Kullback-Leibler Divergence loss layer |

### Supported Activation Functions

NNTrainer provides

| Keyword | Loss Name | Description |
 |:-------:|:---:|:---|
| tanh | tanh function | set as layer property |
| sigmoid | sigmoid function | set as layer property |
| softmax | softmax function | set as layer property |
| relu | relu function | set as layer property |
| leaky_relu | leaky_relu function | set as layer property |
| swish | swish function | set as layer property |
| gelu | gelu function | set as layer property |
| quick_gelu | quick gelu function | set as layer property |
| elu | elu function | set as layer property |
| selu | selu function | set as layer property |
| softplus | softplus function | set as layer property |
| mish | mish function | set as layer property |

### Tensor

Tensor is responsible for calculation of a layer. It executes several operations such as addition, division, multiplication, dot production, data averaging and so on. In order to accelerate  calculation speed, CBLAS (C-Basic Linear Algebra: CPU) and CUBLAS (CUDA: Basic Linear Algebra) for PC (Especially NVIDIA GPU) are implemented for some of the operations. Later, these calculations will be optimized.
Currently, we support lazy calculation mode to reduce complexity for copying tensors during calculations.

| Keyword | Description |
 |:-------:|:---:|
| 4D Tensor | B, C, H, W|
| Add/sub/mul/div | - |
| sum, average, argmax | - |
| Dot, Transpose | - |
| normalization, standardization | - |
| save, read | - |

### Others

NNTrainer provides

| Keyword | Loss Name | Description |
 |:-------:|:---:|:---|
| weight_initializer | Weight Initialization | Xavier(Normal/Uniform), LeCun(Normal/Uniform),  HE(Normal/Uniform) |
| weight_regularizer | weight decay ( L2Norm only ) | needs set weight_regularizer_param & type |