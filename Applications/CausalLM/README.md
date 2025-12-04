# ☄️ CausalLM Inference with NNTrainer

- This application provides examples to run causal llm models using nntrainer.
- This example only provides *inference* mode, not *training* mode yet.
 
## Supported models

| Model | Model Param Size |
| :--- | :--- |
| **Llama** | - |
| **Qwen3** | 0.6B, 1.7B, 4B, 7B, 14B |
| **Qwen3MoE** | 30B-A3B |
| **Gpt-Oss** | 20B, 120B |
| **Ernie 4.5** | 21B-A3B |
| **Deepseek V2 Lite** | 16B |
- You can try your own model with custom layers! 
- Feel free to contribute! 😊

## How to run

- download and copy the model files from hugingface to `res/{model}` directory.
- The folder should contain
    - config.json
    - generation_config.json
    - tokenizer.json
    - tokenizer_config.json
    - vocab.json
    - nntr_config.json
    - nntrainer weight binfile (matches with the name in nntr_config.json)
    - which are usuallyl included in HF model deployment.
- compile the Application
- If you test CausalLM on your PC, build with `-Denable-transformer=true`
- run the model with the following command

```
$ cd build/Applications/CausalLM
$ ./nntr_causallm {your model config folder}
```

e.g.,

```
$ ./nntr_causallm /tmp/nntrainer/Applications/CausalLM/res/qwen3-4b/
```

### Recommended Configuration 

- PC test
```
$ meson build -Denable-fp16=true -Dthread-backend=omp -Denable-transformer=true -Domp-num-threads=4
$ export OMP_THREAD_LIMIT=16 && export OMP_WAIT_POLICY=active && export OMP_PROC_BIND=true && export OMP_PLACES=cores && export OMP_NUM_THREADS=4
```

- Android test
```
$ ./tools/package_android.sh -Domp-num-threads=4 -Dthread-backend=omp
```

## Supported Models

- Qwen3 (0.6B, 1.7B, 4B, 8B, 14B, 32B) [[link](https://huggingface.co/Qwen/Qwen3-4B)]
- Qwen3-MoE (30B-A3B) [[link](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)]
- GPT-OSS (MoE: 20B, 120B) [[link](https://huggingface.co/openai/gpt-oss-20b)]

For more details, please refer to the [Model Documentation](models/README.md).