# nntrainer Benchmark Tool

A benchmark tool for nntrainer CausalLM models on Android devices.

## Installation

### Requirements

- Python 3.6+
- ADB (Android Debug Bridge) installed and device connected
- Canonical native artifacts deployed with `../build_android.sh --install`
  to `/data/local/tmp/Quick.AI/`
- Model files and `nntr_config.json` available on device

### Python Dependencies

```bash
pip install tabulate
```

## Usage

### Basic Benchmark

Run a single trial benchmark:

```bash
python3 benchmark_android.py \
  -m /data/local/tmp/Quick.AI/models/qwen3-0.6b \
  -p 512 \
  -n 128 \
  -t 4
```

### Options

- `-m, --model`: Model directory path on device (required)
- `-p, --n-prompt`: Number of prompt tokens (default: 512)
- `-n, --n-gen`: Number of generation tokens (default: 0)
- `-t, --n-threads`: Number of OMP threads (default: 4)
- `-b, --batch-size`: Batch size (default: 1)

## Output

The tool outputs TPS (tokens per second) metrics for both prefill and generation phases.

```bash
python3 benchmark_android.py \
  -m <device_model_path> \       # Model directory path on device (required)
  -p <n_prompt> \                # Number of prompt tokens (default: 512)
  -n <n_gen> \                   # Number of generation tokens (default: 0)
  -r <n_trials> \                # Number of trials (default: 5)
  -t <n_threads> \               # Number of OMP threads (default: 4)
  -b <batch_size> \              # Batch size (default: 1)
  --device-info <string> \       # Device info (auto-detect if not specified)
```

Can list arguments accept comma-separated values:
- `-p, --n-prompt`: Comma-separated prompt token counts
- `-n, --n-gen`: Comma-separated generation token counts
- `-t, --n-threads`: Comma-separated thread counts

The script runs benchmarks for all combinations of the specified parameters (Cartesian product).


### Examples

#### Prefill benchmark 
```bash
# Test 512 tokens, no generation
python3 benchmark_android.py \
  -m /data/local/tmp/Quick.AI/models/qwen3-0.6b \
  -p 512 -n 0 -r 5 -t 4

# Test 128,256,512,1024 tokens, no generation
python3 benchmark_android.py \
  -m /data/local/tmp/Quick.AI/models/qwen3-0.6b \
  -p 128,256,512,1024 -n 0 -r 5 -t 4
```

#### Generation benchmark 
```bash
# Test 128 token generation
python3 benchmark_android.py \
  -m /data/local/tmp/Quick.AI/models/qwen3-0.6b \
  -p 512 -n 128 -r 5 -t 4

# Test 128,256,512,1024 token generation
python3 benchmark_android.py \
  -m /data/local/tmp/Quick.AI/models/qwen3-0.6b \
  -p 512 -n 128,256,512,1024 -r 5 -t 4
```

#### Test different thread counts
```bash
# Test with 4 threads
python3 benchmark_android.py \
  -m /data/local/tmp/Quick.AI/models/qwen3-0.6b \
  -t 4

# Test with 2,4,8,16 threads
python3 benchmark_android.py \
  -m /data/local/tmp/Quick.AI/models/qwen3-0.6b \
  -t 2,4,8,16
```

### Output Format

The sweep script outputs a pretty table:

**Pretty Table:**
```
BENCHMARK SWEEP RESULTS (Not Real Result)
Model: qwen3-0.6b | Size: 2.30 GiB | Type: CausalLM | Dtype: FP32-FP32 | Device: S25U
+-----------+---------+------+----------------+-----------------+
| Threads   | Prompt  | Gen  | Prefill TPS    | Gen TPS         |
+===========+=========+======+================+=================+
| 1         | 512     | 128  | 200.50 ± 5.25  | 30.10 ± 2.10    |
| 2         | 512     | 128  | 350.25 ± 8.40  | 55.30 ± 3.20    |
| 4         | 512     | 128  | 620.80 ± 12.50 | 95.40 ± 4.80    |
| 8         | 512     | 128  | 750.30 ± 15.20 | 120.50 ± 5.80   |
+-----------+---------+------+----------------+-----------------+
```

## How It Works

1. **Load Configuration**: Pulls `nntr_config.json` from the device via ADB
2. **Backup & Modify**: Creates a backup of the original config on device, modifies it with test parameters, and pushes back to device
3. **Run Trials**: Executes the C++ benchmark binary on the device multiple times via ADB
4. **Collect Metrics**: Parses output to extract TPS values and temperatures
5. **Calculate Statistics**: Computes mean and standard deviation across trials
6. **Restore Configuration**: Restores the original `nntr_config.json` on the device
7. **Output Results**: Prints results in specified format


## Troubleshooting

### ADB device not found
```bash
adb devices
```
Make sure your device is connected and ADB debugging is enabled.

### Model file not found on device
The script requires all model files to be on the device. Deploy the native
artifacts with `../build_android.sh --install`, then copy each model directory
under `/data/local/tmp/Quick.AI/models/`.

### nntr_config.json not found on device
The script reads `nntr_config.json` directly from the device. Make sure it exists at the specified model path.

Example device structure:
```
/data/local/tmp/Quick.AI/
├── models/qwen3-0.6b/
│   ├── nntr_config.json
│   ├── config.json
│   ├── generation_config.json
│   └── nntr_qwen3_0.6b_fp32.bin
├── libc++_shared.so  
├── libccapi-nntrainer.so  
├── libnntrainer.so
├── libcausallm.so
├── nntr_causallm
└── run_causallm.sh
```



## License

Same as nntrainer project license.
