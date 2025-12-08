# ☄️ CausalLM Inference with NNTrainer

- This application provides examples to run causal llm models using nntrainer.
- This example only provides *inference* mode, not *training* mode yet.

## Supported models

- Llama
- Qwen3 (1.7b/4b/7b/14b)
- Qwen3MoE (30b-A3b)
- Gpt-Oss-20b 
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
$ meson build -Denable-fp16=true -Dggml-thread-backend=omp -Denable-transformer=true -Domp-num-threads=4
$ export OMP_THREAD_LIMIT=16 && export OMP_WAIT_POLICY=active && export OMP_PROC_BIND=true && export OMP_PLACES=cores && export OMP_NUM_THREADS=4
```

- Android test
```
$ ./tools/package_android.sh -Domp-num-threads=4 -Dggml-thread-backend=omp
```

## Model Explanations

- qwen3_causallm : basic implementation of qwen3 model
- qwen3_moe_causallm : basic implementation of qwen3 moe model
- qwen3_slim_moe_causallm : nntrainer's FSU-scheme-activated qwen3 moe model
- qwen3_cached_slim_moe_causallm : nntrainer's MoE-specific FSU-based qwen3 moe model
- gptoss_causallm : basic implementation of gpt oss model
- gptoss_cached_slim_causallm : nntrainer's MoE-specific FSU-based gpt-oss moe model

