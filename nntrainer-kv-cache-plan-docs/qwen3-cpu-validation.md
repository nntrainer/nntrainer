# Qwen3 0.6B: Linux CPU 기준 모델과 검증

사용자 결정에 따라 이후 KV Manager의 실제 모델 검증은 **Qwen3 0.6B**를 기준으로
한다. Gemma4는 별도 브랜치의 호환성 수정 범위로 유지한다. Linux CPU를 먼저
검증하고 Android Arm CPU로 이동하며, 이후 backend는 추후 결정한다.

## 원본과 변환 규약

원본은 공개 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)이며 revision을
`c1899de289a04d12100db370d81485cdf75e47ca`로 고정했다. `trust_remote_code`에 의존하는
별도 모델 코드는 다운로드하지 않는다. 원본 checkpoint SHA256은
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`다.

| 항목 | 값 |
|---|---|
| 구조 | 28 layers, Q head 16, KV head 8, head dimension 128, tied embedding |
| HF 원본 | `build-kv-cpu/models/hf/qwen3-0.6b/` |
| FP32 모델 | `build-kv-cpu/models/qwen3-0.6b-fp32/` |
| 양자화 모델 | `build-kv-cpu/models/qwen3-0.6b-q40/` |
| 변환 metadata | `build-kv-cpu/models/qwen3-cpu-manifest.json` |
| 설정 | FP32 activation, batch 1, init plane 128, logical KV capacity 512, skip_prefill false |
| 실행 고정값 | NNTR CPU threads 8, OpenBLAS threads 1, OMP threads 8, greedy, non-thinking |

기존 numpy `weight_converter_layergraph.py`의 FFN 순서가 `gate, up`으로 남아 있어
현재 graph와 torch converter의 `up, gate` 순서에 맞췄다. 두 행렬의 shape가 같아
잘못된 순서도 파일 크기 검사로는 검출되지 않는다. 수정 후 numpy와 torch converter로
전체 모델을 각각 변환해 **출력 파일 SHA256의 완전 일치**를 확인했다.

| 모델 | 크기(bytes) | SHA256 |
|---|---:|---|
| FP32 `.bin` | 2,384,199,680 | `d60649a75e7faf3a4d50f046dbea6607d53e645fe4999ab181e4a651b84da7e9` |
| Q4_0 FC + Q6_K embedding/lm_head, X86 `.bin` | 375,614,464 | `eed53478ffd0d72241efde66bf505de74b3c63ff12f5bf0e363a0329f8c0ee2a` |

양자화 파일은 FP32 파일의 약 15.75%다. 이는 **weight 양자화**이며 KV cache 양자화를
구현했다는 뜻이 아니다. 두 모델 모두 이 CPU profile의 KV는 UINT16에 half bit를
보관한다. X86 packing 결과를 그대로 Android Arm에서 사용하지 않는다.

## 재현 명령

NNTrainer build는 [Linux CPU 실행 기록](linux-cpu-execution.md)의 명령을 사용한다.
아래 Python 환경은 기존 설치 중 정상 동작하는 별도 환경이다. 시스템 Python의
transformers/huggingface_hub 버전 충돌을 피하기 위해 `PYTHONNOUSERSITE=1`을 쓴다.
확인 버전은 torch `2.13.0+cpu`, transformers `4.55.4`, huggingface_hub `0.36.2`다.

```bash
PYTHONNOUSERSITE=1 /home/jayden/miniforge3/envs/rkv-hf/bin/python \
  tools/kv_cache/prepare_qwen3_cpu.py --download --quantize \
  --check-torch-converter \
  --torch-python /home/jayden/miniforge3/envs/rkv-hf/bin/python
```

스크립트는 원본 revision과 변환 hash/크기를 manifest에 남긴다. 실제 변환·양자화의
핵심 명령은 다음과 같다. 원래 `~/TEST_MODELS` 파일은 사용하거나 수정하지 않는다.

```bash
python3 Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter_layergraph.py \
  --model_path build-kv-cpu/models/hf/qwen3-0.6b \
  --output_name build-kv-cpu/models/qwen3-0.6b-fp32/nntr_qwen3_0.6b_fp32.bin \
  --data_type float32
NNTR_ENGINE=cpu NNTR_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  build-kv-cpu/Applications/CausalLM/nntr_quantize \
  build-kv-cpu/models/qwen3-0.6b-fp32 \
  --output build-kv-cpu/models/qwen3-0.6b-q40 \
  --fc_dtype Q4_0 --embd_dtype Q6_K --lmhead_dtype Q6_K \
  --isa X86 --no_sidecar --output_bin nntr_qwen3_0.6b_q40_embdq6k.bin
```

LogitsProcessor를 읽기 전용 관찰 hook으로 쓰는 probe는 root build를 변경하지 않고
기존 Meson compiler/linker 옵션으로 별도 컴파일한다. linked CausalLM ABI가 바뀌면
probe도 다시 빌드해야 한다.

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py \
  --output build-kv-cpu/validation/qwen3-post/qwen_cpu_probe
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  /home/jayden/miniforge3/envs/rkv-hf/bin/python tools/kv_cache/validate_qwen3_cpu.py \
  --output build-kv-cpu/validation/qwen3-post/default \
  --probe build-kv-cpu/validation/qwen3-post/qwen_cpu_probe \
  --include-session --trace
```

강제 flash / 장문 검증은 `--force-flash --long-prompt`를 추가하고 output을 분리한다.
`--force-flash`는 `NNTR_MIN_PREFILL=1`을 설정해 짧은 prefill과 token decode에서도
해당 consumer를 검사한다. 장문은 256-row activation plane을 사용한다.
`--trace`는 KV Manager/producer/consumer의 주소 및 byte 계측을 stderr에 보관한다.
trace/probe의 추가 작업이 있어 이 실행의 timing을 성능 개선 수치로 사용하지 않는다.

## 변경 전 모델 baseline

CPU CausalLM 동작을 core facade로 전환하기 전의 라이브러리를
`build-kv-cpu/validation/qwen3-baseline/baseline-libs/`에 보존했다. 실제로 로드한
라이브러리 경로는 `ldd`로 확인했으며 각 파일 hash도 보관한다. baseline probe는
`9a9662f69`의 product header를 별도로 추출해 사용했다. 새 header로 만든 probe에
옛 라이브러리를 섞지 않는다.

일반 prefill의 네 prompt × FP32/Q4_0, 총 **8회**에서 HF FP32 eager reference와
생성 token ID sequence가 모두 일치했다. 매 generation step의 전체 logits가 finite였다.
입력 문자열은 HF tokenizer로 non-thinking template을 만든 뒤 양쪽에 동일하게 전달한다.

| prompt | HF / FP32 / Q4_0 결과 |
|---|---|
| `What is 2 + 2? Answer with only the number.` | 세 경로 모두 `2<\|im_end\|>` |
| `Compute 2 + 2.` | 세 경로 모두 `2 + 2 = 4.<\|im_end\|>` |
| 대한민국 수도 질문 | 세 경로 모두 `대한민국의 수도는 서울입니다.<\|im_end\|>` |
| 1부터 5까지 쉼표로 구분 | 세 경로 모두 `1, 2, 3, 4, 5<\|im_end\|>` |

**첫 prompt의 `2`는 산술적으로 틀린 출력이다.** 공식 HF 원본도 동일한 입력 token
IDs, non-thinking, greedy 조건에서 같은 `[17, 151645]`를 생성했다. 따라서 이 관찰을
nntrainer 변환 오류나 새 KV Manager 회귀로 단정하지 않는다. 일반적인 산술 표현에서는
정답을 생성했다. 문장 유창성이나 네 개 prompt만으로 모델 전체 정확성을 주장하지 않는다.

HF는 FP32 KV, nntrainer는 half-bit KV를 쓰므로 logits의 bitwise 일치를 기대하지 않는다.
네 prompt의 첫 prefill logits 최대 절대 오차는 FP32 모델에서 약 `0.01097–0.02187`,
Q4_0 모델에서 약 `6.15–9.55`다. 양자화는 logits를 바꾸며, 이 작은 prompt 집합에서의
token 일치는 포괄적인 양자화 품질 검증을 대신하지 않는다.

동일 allocation의 두 turn 검증에서도 FP32/Q4_0가 각각 `hello`, `hello`를 생성하고
HF token IDs와 일치했다. 첫 turn 종료 position 21에서 이어서 prefill/decode한 뒤
position은 49였다. 마지막 예측 token은 아직 KV에 쓰이지 않았으므로 다음 turn의
입력 앞에 그 token을 포함해 HF의 이어 붙인 prefix와 같은 history를 만들었다.

증거는 `build-kv-cpu/validation/qwen3-baseline/` 아래의 converter log/hash,
`parity/report.json`, 개별 logits `.f32`, `*-multiturn.json`과 `multiturn-hf.json`이다.
강제 flash baseline은 `forced-flash/report.json`에 따로 보관한다. 장문 Q4_0 결과는
`France`이고 HF/FP32는 `France.`로, 양자화에 따른 마침표 token 차이가 관찰됐다.
의미상 국가 답변은 같지만 token 불일치를 숨기지 않는다.

## CPU KV 변경 후

CPU facade/step 연결과 x86 flash의 prefix deinterleave 제거 후 새 ABI로 probe를
다시 컴파일했다. baseline과 post의 shared library snapshot을 분리했으며 report에
Git HEAD, dirty diff SHA256, probe/header revision 및 library hash를 남긴다.

| 비교 조건 | 결과 | 증거 (`build-kv-cpu/validation/qwen3-post/` 아래) |
|---|---|---|
| 기본 경로: 네 prompt × FP32/Q4_0 | 모든 생성 token IDs 및 매 step 전체 logits bytes가 baseline과 완전 일치 | `default-comparison.json` |
| 기본 경로: 동일 session의 두 turn, FP32/Q4_0 | 양 turn token/logits 완전 일치, 위치 `0 → 21 → 49` | 같은 report의 `sessions` |
| `NNTR_MIN_PREFILL=1`: 다섯 prompt(164-token 장문 포함) × 두 dtype | token/logits 완전 일치 | `forced-flash-comparison.json` |
| 강제 flash: 동일 session 두 turn | token/logits 완전 일치 | 같은 report의 `sessions` |
| `VJEPA_BK=64`, `VJEPA_BQ=32`, 164-token 장문 | 두 번째 이후 key-block offset을 포함해 FP32/Q4_0 token/logits 완전 일치 | `tiles64-comparison.json` |
| `NNTR_MIN_PREFILL=999999`: per-row consumer, 두 dtype | token/logits 완전 일치, consumer trace 확인 | `per-row-comparison.json`, `per-row-trace-audit.json` |

이 host의 기본 x86 flash threshold는 1이므로 기본 경로도 flash를 사용한다.
별도의 per-row consumer를 검사하려면 `--min-prefill 999999`로 threshold를 올린다.
타일 경계 재현은 다음 옵션으로 수행한다.

```bash
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  /home/jayden/miniforge3/envs/rkv-hf/bin/python tools/kv_cache/validate_qwen3_cpu.py \
  --output build-kv-cpu/validation/qwen3-post/tiles64 \
  --probe build-kv-cpu/validation/qwen3-post/qwen_cpu_probe \
  --prompt long --force-flash --key-block 64 --query-block 32 --trace
python3 tools/kv_cache/compare_qwen_cpu_runs.py \
  --baseline build-kv-cpu/validation/qwen3-baseline/tiles64 \
  --current build-kv-cpu/validation/qwen3-post/tiles64 \
  --output build-kv-cpu/validation/qwen3-post/tiles64-comparison.json
```

CPU storage witness는 **58,720,256 backing bytes, 28 owner layers**를 보고했다.
장문 prefill부터 decode까지 각 step의 28개 allocation pair가 한 번씩 모두 나타나는지,
producer의 key/value 주소가 manager allocation 주소와 같은지 기계적으로 확인했다.
각 call은 `read_alias=1`, `write_alias=1`, `append_copy_bytes=0`이고 실제 consumer는
`flash-strided`, `history_copy_bytes=0`이다. FP32→half 변환/RoPE의 출력 byte 수는
별도 `transform_bytes`로 남기며 copy와 혼동하지 않는다.

```bash
python3 tools/kv_cache/audit_qwen_cpu_trace.py \
  build-kv-cpu/validation/qwen3-post/forced-flash \
  --output build-kv-cpu/validation/qwen3-post/forced-flash-trace-audit.json
```

audit는 owner 누락/중복, compatibility profile, layout history-copy를 거절한다.
실제 log에서 owner 하나와 대응 consumer 한 줄을 제거한 경우와 nonzero layout-copy
line을 추가한 경우도 거절되는지 확인했다. 이 계측은 소스의 application operation과
backing identity에 대한 증거이며, 하드웨어 메모리 트래픽 측정은 아니다.

준비 스크립트도 별도 output에서 FP32 변환 → torch converter hash parity → Q4_0
양자화까지 한 번 재실행해 두 weight 파일의 SHA256가 최초 결과와 같은지 확인했다.
증거는 `build-kv-cpu/preparation-check/pipeline.log`와 해당 `models/qwen3-cpu-manifest.json`이다.

이 검증의 직접 쓰기는 FP32 activation의 최종 RoPE/half 변환 destination을 대상으로
한다. native FP16 activation에서 남아 있는 V append copy, quantized KV, GPU/NPU,
Android Arm 실기기는 이 Linux CPU 결과의 완료 범위가 아니다.

## 추가 검증: chunked prefill과 파일 복원

`--prefill-chunk 17`로 164-token 장문을 17-token step들로 나누어 실행했다.
FP32와 Q4_0 모두 기존과 같은 답변을 생성하고 모든 logits가 finite였다. 각 step의
28개 owner가 빠짐없이 같은 manager allocation으로 접근했는지 audit를 통과했다.
이 조건에서도 baseline/post의 token IDs 및 매 generation step 전체 logits bytes가
완전 일치했다. 비교 결과는 `chunk17-comparison.json`에 기록했다.

`--session-only --snapshot-resume`는 첫 turn 종료 position 21의 cache를 파일로 저장하고,
**새 모델 인스턴스**를 초기화·weight load한 뒤 `resumeFromKVCacheFile(path, 0)`로
복원한다. 같은 두 번째 turn을 기존 live session과 복원 session 양쪽에서 실행한다.
FP32/Q4_0 각각 저장 파일은 **2,409,136 bytes**였고, 복원 position 21과 다음 turn 종료
position 49가 live session과 같았다. 생성 token IDs와 각 generation step 전체 logits가
**bitwise 동일**했다. 보고서의 `snapshot_resume`에 이를 별도로 표시한다.

```bash
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  /home/jayden/miniforge3/envs/rkv-hf/bin/python tools/kv_cache/validate_qwen3_cpu.py \
  --output build-kv-cpu/validation/qwen3-post/chunk17 \
  --probe build-kv-cpu/validation/qwen3-post/qwen_cpu_probe \
  --prompt long --prefill-chunk 17 --trace
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  /home/jayden/miniforge3/envs/rkv-hf/bin/python tools/kv_cache/validate_qwen3_cpu.py \
  --output build-kv-cpu/validation/qwen3-post/snapshot-resume \
  --probe build-kv-cpu/validation/qwen3-post/qwen_cpu_probe \
  --session-only --snapshot-resume --trace
```

Snapshot 검증은 원래 모델과 복원 모델의 allocation을 동시에 유지한다. audit는 이
경우 두 storage instance를 구분하고 각 step을 실제 key/value allocation pair에 연결해
owner 누락/중복과 copy 유무를 검사한다. 증거는 `snapshot-resume/report.json`,
`snapshot-resume-trace-audit.json`, `chunk17/report.json`, `chunk17-trace-audit.json`이다.
weight 모델과 `.kv` 파일은 모두 build 아래의 실행 산출물로만 남기며 커밋하지 않는다.
