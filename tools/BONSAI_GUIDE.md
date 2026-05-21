# Running Bonsai Q1_0 Models on NNTrainer

## Prerequisites

```bash
pip install gguf numpy huggingface-hub
```

## Step 1: Download Bonsai GGUF Model

```bash
# Bonsai-1.7B (~250MB)
huggingface-cli download prism-ml/Bonsai-1.7B-gguf \
  --local-dir ./bonsai-1.7b-gguf

# Or Bonsai-4B (~570MB)
# huggingface-cli download prism-ml/Bonsai-4B-gguf --local-dir ./bonsai-4b-gguf

# Or Bonsai-8B (~1.16GB)
# huggingface-cli download prism-ml/Bonsai-8B-gguf --local-dir ./bonsai-8b-gguf
```

## Step 2: Inspect GGUF Metadata (Optional)

```bash
python tools/gguf_to_nntrainer.py \
  --gguf ./bonsai-1.7b-gguf/Bonsai-1.7B-Q1_0-g128.gguf \
  --output /tmp/bonsai-inspect \
  --dump-only
```

This prints all metadata (architecture, layer count, dimensions) and tensor list.

## Step 3: Convert GGUF to NNTrainer Format

```bash
python tools/gguf_to_nntrainer.py \
  --gguf ./bonsai-1.7b-gguf/Bonsai-1.7B-Q1_0-g128.gguf \
  --output ./models/bonsai-1.7b \
  --tokenizer ./bonsai-1.7b-gguf/tokenizer.json
```

This will:
- Dequantize Q1_0 weights to FP32 (for compatibility)
- Save `model_float32.bin` in NNTrainer binary format
- Generate `config.json`, `nntr_config.json`, `generation_config.json`

## Step 4: Extract Tokenizer

If the tokenizer is embedded in the GGUF, extract it separately:

```bash
# If tokenizer.json exists in the download directory, copy it
cp ./bonsai-1.7b-gguf/tokenizer.json ./models/bonsai-1.7b/

# Or download from the unpacked model if available
# huggingface-cli download prism-ml/Bonsai-1.7B-unpacked tokenizer.json \
#   --local-dir ./models/bonsai-1.7b
```

Then update `nntr_config.json` to point to the tokenizer:
```json
{
  "tokenizer_file": "/absolute/path/to/models/bonsai-1.7b/tokenizer.json"
}
```

## Step 5: Build NNTrainer CausalLM

```bash
meson setup builddir -Denable-test=true \
  -Denable-tflite-backbone=false \
  -Denable-tflite-interpreter=false

ninja -C builddir
```

## Step 6: Run Inference

```bash
./builddir/Applications/CausalLM/nntrainer_causal_lm ./models/bonsai-1.7b
```

## Architecture Notes

Bonsai models use Q1_0_g128 quantization:
- 1 bit per weight + FP16 scale per 128 weights
- Effective: 1.125 bits per weight
- Memory: ~14x smaller than FP32

| Model | GGUF Size | FP32 Dequantized |
|-------|-----------|-----------------|
| Bonsai-1.7B | ~250 MB | ~6.5 GB |
| Bonsai-4B | ~570 MB | ~15 GB |
| Bonsai-8B | ~1.16 GB | ~30 GB |

Note: The dequantized FP32 model is large. For production, native Q1_0 loading
(without dequantization) would be needed -- this requires additional integration
with the CausalLM inference pipeline to use Q1_0_Tensor directly in FC layers.

## Troubleshooting

### Unknown architecture
Run `--dump-only` to check `general.architecture`. If it's not `qwen3`,
the model may need a different CausalLM backend (the converter will still
extract weights, but `config.json` may need manual adjustment).

### Tokenizer not found
GGUF files embed tokenizer data. If `tokenizer.json` is not available
separately, you may need to extract it from the GGUF metadata or download
from the unpacked model variant (e.g., `prism-ml/Bonsai-1.7B-unpacked`).

### Memory issues with large models
Bonsai-8B dequantized to FP32 is ~30GB. Use `--no-dequantize` flag for
raw Q1_0 output (experimental), or consider the 1.7B variant first.
