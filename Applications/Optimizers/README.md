# Optimizers Validation

## Overview
- A minimal example that trains on either random data or the MNIST dataset to quickly compare optimizers.
- Assumes FP32-only build/runtime.
- Merges functionality of previous independent Random and MNIST optimizer apps.

## Build
1. From repository root:
```bash
meson setup build -Dbuildtype=release -Denable-app=true
meson compile -C build -j"$(nproc)"
```

## Run Examples

**With Random Data**

Same configuration; switch only the optimizer:

```bash
cd build/Applications/Optimizers/jni

# Lion (Default, varied Weight Decay)
./nntrainer_optimizers --dataset=random --opt=lion  --wd=0     --bs=16 --db=32 --epochs=5 --lr=0.001
./nntrainer_optimizers --dataset=random --opt=lion  --wd=0.01  --bs=16 --db=32 --epochs=5 --lr=0.001

# Adam
./nntrainer_optimizers --dataset=random --opt=adam             --bs=16 --db=32 --epochs=5 --lr=0.001

# AdamW
./nntrainer_optimizers --dataset=random --opt=adamw            --bs=16 --db=32 --epochs=5 --lr=0.001

# SGD (Varied Learning Rate)
./nntrainer_optimizers --dataset=random --opt=sgd              --bs=16 --db=32 --epochs=5 --lr=0.001
./nntrainer_optimizers --dataset=random --opt=sgd              --bs=16 --db=32 --epochs=5 --lr=0.0005
```

**With MNIST Data**

Compare optimizers with the same settings (requires MNIST resources):

```bash
cd build/Applications/Optimizers/jni

./nntrainer_optimizers --dataset=mnist --opt=lion  --lr=0.001 --wd=0.01 --epochs=100 --bs=32
./nntrainer_optimizers --dataset=mnist --opt=adam  --lr=0.001              --epochs=100 --bs=32
./nntrainer_optimizers --dataset=mnist --opt=adamw --lr=0.001 --wd=0.01     --epochs=100 --bs=32
./nntrainer_optimizers --dataset=mnist --opt=sgd   --lr=0.001              --epochs=100 --bs=32
```

Run from an arbitrary directory with explicit resource paths:

```bash
./build/Applications/Optimizers/jni/nntrainer_optimizers \
  --dataset=mnist \
  --config=Applications/MNIST/res/mnist.ini \
  --data=Applications/MNIST/res/mnist_trainingSet.dat \
  --opt=lion --lr=0.001 --wd=0.01 --epochs=100 --bs=32
```

## Options

**General**
- `--dataset=random|mnist` : dataset type (default: random)
- `--opt=lion|adam|adamw|sgd` : optimizer (default: lion)
- `--wd=<float>` : weight decay (used by Lion/AdamW)
- `--epochs=<int>` : number of epochs
- `--bs=<int>` : batch size
- `--lr=<float>` : learning rate

**Random Dataset Options**
- `--db=<int>` : number of batches (iterations) per epoch

**MNIST Dataset Options**
- `--config=<path>` : INI path (auto-discovered if not provided)
- `--data=<path>` : dataset path (auto-discovered if not provided)
- `--train_size=<uint>` : number of training samples (default: 100)
- `--val_size=<uint>` : number of validation samples (default: 100)

## Output
- Prints L2 norm of weights before/after training.
- Prints Delta L2 (magnitude of weight updates).
- Prints per-epoch training loss logs.

## Notes
- Because data is random in Random Mode, loss curves are better suited to compare update magnitude and consistency rather than convergence.
- MNIST Mode exposes optimizer differences in loss curves (convergence speed/quality) more clearly than random data.
