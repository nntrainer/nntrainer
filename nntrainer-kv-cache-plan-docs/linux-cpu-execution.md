# Linux CPU 개발·검증 기록

사용자 결정: **Linux x86 CPU → Android Arm CPU → 이후 backend는 추후 결정**.
추가 결정: 이후 모델 정확성·성능 검증 기준은 **Qwen3 0.6B**다.
Gemma4 호환성 수정은 별도 `fix/gemma4-skip-prefill-lmhead` 브랜치에서 관리한다.

후속 Android CPU 작업은 기기 `bacd420f`에서 진행한다.
[Android 빌드·단위 검증](android-build-checks.md)과
[Android Qwen 실기기 검증](android-cpu-validation.md)에 Linux 결과와 구분해 기록한다.
`AGENTS.md`는 참고하며 수정하지 않는다. 구현·명령·검증·미완료 사항을 단계별로
이 문서에 갱신하고, 통과한 단위로 DCO/agent trailer를 갖춘 커밋을 남긴다.

## P0: 계획에 대한 비판적 검토

기준 소스는 계획과 같은 `141871a58ff439e3caeac40c1632ff8ac3543d20`이다.
따라서 소스 버전 차이보다 실행 환경과 모델 패키지의 차이를 먼저 해결한다.

| 계획의 가정 | 실제 관찰 / 적용 결정 |
|---|---|
| Ubuntu 24.04, GCC 13 | Ubuntu 22.04, GCC/G++ 11.4, Meson 0.61.2, Ninja 1.10.1, clang-format 14.0.0. 첫 기준은 비FP16 CPU build. native FP16 대신 wrapper도 별도 확인한다. |
| CPU 다음 OpenCL/QNN probe | 사용자 요청에 따라 Linux CPU를 먼저 완료하고 Android CPU로 진행한다. 장치 probe를 CPU 착수의 선행 조건으로 두지 않는다. |
| x86 tokenizer archive를 그대로 link | `esaxx.o`가 GCC 11 libstdc++에 없는 `std::ios_base_library_init()`을 요구한다. in-tree Rust crate를 현재 컴파일러로 별도 build하고 `causallm-tokenizer-lib`를 지정한다. 추적 archive를 변경하지 않는다. |
| 서브모듈 전체 초기화 | 이 Linux profile에 필요한 xgrammar(+하위 모듈), googletest를 고정 SHA로 초기화했다. OpenBLAS/iniparser는 시스템 라이브러리를 사용한다. vendored 소스는 편집하지 않는다. |
| encoder 준비를 Meson에 맡김 | 기존 `prepare_encoder.sh`가 재configure마다 다운로드할 수 있다. 최초 header 준비 후 build의 `encoder/` marker를 만든다. 다운로드 header는 기존 ignore 규칙을 따른다. |
| manager 새 API만 넣으면 직접 접근 | 현재 FP32 활성값 경로는 이미 외부 cache를 읽고 K RoPE/V half 변환 결과를 최종 cache에 쓴다. 이를 보존하면서 소유권과 step publication을 core로 옮긴다. native FP16의 순수 V copy는 별도 미완료 항목이다. |
| core MVP batch=1 | 기존 batch=2 FP32 manager 테스트를 유지한다. contiguous CPU storage가 batch offset을 명시적으로 검사하도록 한다. session 간 공유는 구현 범위가 아니다. |
| 모든 model 테스트가 실행됨 | 최초 baseline은 93 PASS + deprecated QINT4 fixture 1 SKIP. Meson의 suite SKIP=0 표기와 개별 gtest SKIP를 구분한다. |
| 기존 Gemma4 패키지를 바로 사용 | `skip_prefill=true`일 때 초기 layer property 오류가 발생한다. 이를 끈 별도 설정에서 FP32 `.bin`은 로드/생성이 가능하지만, 기존 양자화 `.bin`은 현재 graph와 byte 수가 다르다. 원본 파일은 수정하지 않는다. |

추가 확인: `ml-api-support=disabled`는 이 checkout의 Meson에서 완전한 dependency
제거를 보장하지 않는다. 설치된 ML-API package가 발견되면 `required:false`여도
`.found()` 조건에 의해 `ML_API_COMMON=1`, `NNSTREAMER_AVAILABLE=1` 및 해당 library
link가 활성화된다. 위 옵션은 요청값이고 실제 compile/link 결과는 이와 구분한다.
새 core 모듈의 header/source에는 application, GPU/NPU SDK 의존이 없다.

Host: Intel Core i9-14900, x86_64, AVX2/FMA/F16C 지원, RAM 62GiB
(착수 시 available 약 46GiB), 작업 filesystem 여유 약 69GiB.
`~/TEST_MODELS`는 읽기만 한다. 모델 검증용 설정과 symlink는 build 아래 생성한다.
외부에 설치된 nntrainer layer plugin이 기본 AppContext 초기화에서 탐색될 수 있다는
점도 확인했다. 실제 executable의 core/CausalLM 라이브러리는 `ldd`로 이 build의
경로를 확인했다. 시스템 설치를 덮어쓰지 않는다.

## 재현 가능한 빌드

저장소 루트에서 실행한다. 스크립트는 tokenizer를 현재 compiler로 먼저 build한다.
Cargo.lock을 사용하며, cached crate만 쓰려면 `KV_TOKENIZER_OFFLINE=1`을 지정한다.

```bash
git submodule update --init --recursive --depth 1 \
  Applications/CausalLM/xgrammar subprojects/googletest
KV_TOKENIZER_OFFLINE=1 CC=gcc CXX=g++ \
  tools/kv_cache/build_cpu.sh build-kv-cpu
```

실제 NNTrainer 설정 명령은 다음과 같다.

```bash
CC=gcc CXX=g++ meson setup build-kv-cpu --buildtype=debugoptimized \
  -Dplatform=none -Denable-app=true -Denable-transformer=true \
  -Denable-test=true -Denable-api=false -Denable-fp16=false \
  -Denable-opencl=false -Denable-cuda=false -Denable-npu=false \
  -Denable-htp=false -Denable-tflite-interpreter=false \
  -Denable-tflite-backbone=false -Denable-nnstreamer-backbone=false \
  -Dml-api-support=disabled -Denable-capi=disabled \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled -Dopenblas-num-threads=1 \
  -Dcausallm-tokenizer-lib="$PWD/build-kv-cpu/tokenizer/x86_64-unknown-linux-gnu/libtokenizers_c.a"
ninja -C build-kv-cpu -j8
```

재configure는 `meson configure build-kv-cpu -D옵션=값` 또는 스크립트 마지막에
`--reconfigure`를 전달한다. 스크립트의 추가 인자로 `-Denable-fp16=true
-Dfp16-impl=wrapper` 등 profile을 덮어쓸 수 있다. 서로 다른 profile은 build
디렉터리를 분리한다. runtime thread 수는 아래처럼 고정한다.

```bash
NNTR_ENGINE=cpu NNTR_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  meson test -C build-kv-cpu --print-errorlogs \
  unittest_kv_cache_manager unittest_kv_ring unittest_causallm_models
python3 tools/kv_cache/validate_cpu_models.py \
  --build build-kv-cpu --output build-kv-cpu/validation/baseline-compatible
```

검증 스크립트는 상속된 `NNTR_*`, `LD_*` tuning을 제거하고 greedy decoding을 쓴다.
원래 설정의 `skip_prefill`은 KV prefill 자체 생략이라는 뜻이 아니라 graph의
마지막-token 최적화다. 기존 패키지의 해당 최적화 오류를 피하기 위해 smoke 기본은
false이며 `--skip-prefill`로 원래 true 조건을 재현한다.

## 변경 전 baseline

| 검증 | 결과 | 증거 |
|---|---|---|
| 전체 Linux CPU build | 로컬 tokenizer 지정 후 통과 | `/tmp/kv-baseline-build-local-tokenizer.log` |
| 기존 manager | 20 PASS | `build-kv-cpu/validation/baseline/tests.txt` |
| 기존 ring | 22 PASS | 같은 log와 XML |
| CausalLM tiny model / reference | 93 PASS, deprecated QINT4 1 SKIP | 같은 log와 XML; fixture가 전부 빠진 통과가 아님 |
| 두 Gemma4, skip_prefill=true | 각 산술/한국어 시도에서 초기화 실패: `[LayerImpl] Unknown Layer Properties count 3` | `build-kv-cpu/validation/baseline/*.stderr.txt` |
| FP32, skip_prefill=false, 산술 | `2 + 2`에 `4<turn\|>` 생성, exit 0 | `/tmp/kv-fp32-noskip.stdout`, `.stderr` |
| FP32, skip_prefill=false, 산술/한국어 재현 | 각각 `4<turn\|>`, `서울<turn\|>`; 두 실행 exit 0 | `build-kv-cpu/validation/baseline-compatible/report.json` |
| 기존 Q4_0/Q6_K, skip_prefill=false | weight 크기 불일치로 load 거절 | `/tmp/kv-model-noskip.stderr` |

양자화 파일의 실제 크기는 **2,708,362,380 bytes**, 현재 540-weight graph가 요구하는
크기는 **3,313,915,020 bytes**이며 차이는 **605,552,640 bytes**다.
이는 `262144 × (35 × 256)` PLE lookup table을 Q4_0 대신 Q6_K로 해석할 때의
크기 차이와 정확히 같다. 현재 소스는 PLE에 `embedding_dtype=Q6_K`를 적용한다.
따라서 옛 패키지의 PLE dtype 규약 차이라는 강한 가설이 있지만, 파일에 dtype
metadata가 없어 이것만으로 weight 내용을 증명한 것은 아니다. 강제로 읽거나
파일을 padding하지 않는다. 작동이 확인된 FP32 모델을 우선 사용한다.

이후 core 변경 전후의 실제 모델 비교는 사용자 결정대로 Qwen3 0.6B로 진행한다.
정상적인 짧은 문장만으로 HF logits 일치나 장문 정확성을 주장하지 않는다.

Gemma 초기화 오류의 후속 원인 확인: GDB stack은 `buildLmHeadOutput` →
`TieWordEmbedding::setProperty` → `LayerImpl::setProperty`다.
공통 builder가 FC 전용 `plan_last_row_only`를 tied/custom LM head에도 전달하는
**현재 소스의 버그**이며 모델 설정 자체의 결함으로 단정하지 않는다.
별도 branch의 `21d03c477`은 해당 속성을 FC에만 전달한다. 신규 회귀 6개와 전체
model suite 99 PASS/1 SKIP, 기존 FP32 Gemma의 `skip_prefill=true` 산술/한국어 smoke
모두 통과했다. 상세 기록은 해당 branch의 `docs/gemma4-skip-prefill-fix.md`에 있다.

## P1: 내부 CPU storage와 step lease

`nntrainer/kv_cache/`에 C++17 내부 모듈을 추가했다. 초기 지원은 동기 CPU,
token-major contiguous FP32 또는 IEEE half payload(FP16/UINT16 backing)이다.
기존 MemAllocator를 사용하고 device-visible/SVM/등록 필요 allocator는 거절한다.
네이티브 backend handle과 비동기 completion을 지원한다고 선언하지 않는다.

- 모든 layer의 geometry/encoding/alias/byte overflow와 alignment rounding을
  allocation 전에 검사한다. owner마다 K/V를 할당하고 alias layer는 이를 공유한다.
- `KvStorage`가 실제 allocation과 allocator를 소유한다. aliasing shared_ptr로
  내보낸 MemoryData/Tensor가 manager보다 오래 살아도 allocation을 유지한다.
- `StepLease`는 한 번에 하나만 발행하며 모든 owner 준비 후 commit해야 위치를
  전진시킨다. 실패 시 RAII abort, linear tail은 미공개, ring overwrite는 reset까지
  session invalid다. ring seam은 두 segment로 표현한다.
- reset/reallocate/다음 step 시작 시 이전 view를 무효화한다. Tensor bridge의
  raw mutable alias는 이 검사를 우회하는 명시적인 기존 graph 호환 경계다.
- source를 core include 경로에 직접 추가하지 않아 기존 app의 같은 이름 header와
  충돌하지 않으며, header는 설치/public ABI로 노출하지 않는다.

리뷰에서 이전 ring read view의 다음 step 재사용, mutable MemoryData 주소를
allocation 해제에 사용하는 문제, 기본/이동된 view의 bufferId 접근을 발견해 고쳤다.
각각 재현 테스트를 추가했다.

검증: 신규 core **18 PASS**; 독립 ASan/UBSan/**LeakSanitizer 18 PASS**, 오류·누수 없음.
변경 전 범위의 기존 manager 20, ring 22, model 93 PASS/1 deprecated SKIP도 유지했다.
`enable-app=false`, `enable-transformer=false` core-only sanitizer target을 빌드했다.
증거: `/tmp/kv-core-review/tests.log`, `/tmp/kv-core-review-asan/tests.log`,
`/tmp/kv-p1-tests-host.log`, `build-kv-core-asan/`.

일부 sandbox Meson 실행은 gtest XML상 완료 후에도 종료 감시 timeout이 발생했다.
이를 통과로 세지 않고 같은 검사를 일반 실행 권한으로 반복해 exit 0을 확인했다.
sanitizer도 leak 검사를 끄는 우회로 최종 판정을 대신하지 않았다.

## P2: CausalLM CPU 연결과 직접 읽기·쓰기

CPU facade가 allocation, encoding, geometry, committed position을 core에 위임한다.
기존 input Tensor placeholder에는 allocation을 소유하는 MemoryData alias를 전달한다.
모델별 hook으로 KV head 수를 전달해 Gemma4의 variable geometry도 보존한다.
기존 width-only 호출, layer alias, snapshot 파일 형식은 호환 경계로 유지한다.

`incrementalInference()`는 graph 실행 전에 `[from,to)` lease를 발행하고 성공 직후
commit한다. MHA의 `cache_index = _from` 동작은 upstream에 이미 존재하던 것이며,
새로 구현한 cursor 최적화가 아니다. 당시 readiness 일괄 처리는 실제 producer 완료를
강제하지 못했다. 2026-10-29 변경에서 실제 MHA의 checked write/read와 완료 통지를
연결했다. 활성 step 중 reset, 재allocation, save/load를 거절하며 현재 CPU facade는
ring을 명시적으로 거절한다. core의 ring 검사는 독립 계약 실험 범위다.
기존 raw Tensor bridge 자체에 generation 검사를 추가한 것은 아니므로 동기 graph
실행 범위 안에서 사용해야 한다. token마다 input name/pointer를 찾는 기존 adapter의
정리는 후속 최적화로 남긴다.

리뷰에서 x86 flash attention의 `Ka`/`Va` 전체 prefix packing을 찾아 제거했다.
기존 AVX2 half/float kernel이 지원하는 leading dimension을 KV width로 전달한다.
QK와 AV 모두 `head * head_dim + token * KV_width`에서 manager cache를 직접 읽는다.
Q packing, score/accumulator tile 및 레지스터의 half→float 변환은 필요 연산이다.
K RoPE와 V FP32→half 변환은 최종 manager cache slice에 결과를 쓴다.

초기 direct-access 완료 범위는 **Linux x86, Qwen3 0.6B, batch=1, FP32 activation,
RoPE, contiguous UINT16 half-bit KV**, weight FP32 또는 X86 Q4_0/Q6_K다.
native FP16 activation의 V 순수 copy, ARM prefix packing, no-RoPE, INT8 KV,
GPU/NPU adapter는 이 완료 판단에 포함하지 않는다. FP16 wrapper build 통과를
native FP16 producer 직접 쓰기 완료로 해석하지 않는다.

`NNTR_KV_CPU_TRACE=1`은 다음 operation의 증거를 stderr에 기록한다.

| trace | 기록 |
|---|---|
| `kv-cpu-storage` | 실제 aligned backing bytes, owner 수, layer별 K/V 주소·크기·source |
| `kv-cpu-access` | producer slice와 read Tensor의 storage alias, append 순수 copy와 필요한 변환 byte 수 |
| `kv-cpu-read` | 실제 consumer(`per-row`/`flash-strided`/호환 packing)의 history copy |
| `kv-cpu-layout` | OHWI→concat 호환 gather가 있으면 별도의 history copy |

consumer 내부 copy가 0이어도 layout operation을 빼고 전체 copy가 0이라고 선언하지
않는다. Qwen 지원 profile의 audit는 allocation 주소 대응·전체 layer witness·변환
byte 수·append/history copy를 검사한다. 물리 DRAM traffic, cache miss, 장치 migration
없음을 profiler로 증명한 수치는 아니다. timing 측정에 trace/probe 비용을 섞지 않는다.

검증 결과:

- 비FP16 CPU build: core **18**, facade **25**, ring **22**, tiny model **99 PASS**,
  deprecated QINT4 fixture **1 SKIP**. 모든 Meson target exit 0.
- FP16 wrapper CPU build: core **18**, facade **25**, ring **22 PASS**.
  broad model target은 저장소 기존 규칙에 따라 이 profile에서 생성되지 않는다.
- core-only `enable-app=false`, `enable-transformer=false`: ASan/UBSan/LeakSanitizer
  **18 PASS**, 오류·누수 없음.
- Qwen FP32/Q4_0: 일반·강제 flash·164-token 장문·같은 session 두 turn에서
  변경 전후 모든 생성 token과 **모든 step logits가 bitwise 일치**한다.
  BK64/BQ32로 세 key block을 통과하는 조건도 일치한다.
- 지원 profile trace의 manager/producer/consumer 주소가 일치하며 append/history
  application copy는 0이다. 512-token geometry의 unique backing은
  **58,720,256 bytes / 28 owner layers**다.

상세 명령·정확성 한계·모델 hash와 원본 reference 비교는
[Qwen3 CPU 검증](qwen3-cpu-validation.md), 리뷰 근거는 [P2 리뷰](p2-review.md)에 있다.
최종 local log는 `/tmp/kv-p2-release-tests.log`, `/tmp/kv-fp16-tests.log`,
`/tmp/kv-core-final-asan-tests.log`이며 Qwen의 JSON/개별 logits/trace는
`build-kv-cpu/validation/qwen3-baseline/` 및 `qwen3-post/`에 보존한다.

Android source 목록은 Meson File 대신 절대경로 문자열을 사용하도록 바로잡았다.
Android app의 private core include 경로도 연결했다. 이는 build 연결 준비이며
Linux 단계 종료 시점에는 Android cross build·실기기 검증 전이었다. 이후 Arm CPU
검증 결과는 위 Android 전용 기록을 참고한다.

### Qwen을 이용한 추가 CPU 상태 검증

모델 준비에서 멈추지 않고 두 weight profile 모두 다음 조건을 추가 확인했다.

- 164-token prefill을 **17-token chunk**로 나눈 뒤 decode한다. 변경 전과 새 CPU
  manager 구현의 모든 token·각 step의 전체 logits bytes가 완전히 일치한다.
- position **21**에서 KV snapshot을 저장하고 **새 모델 인스턴스**에서 weight와
  snapshot을 로드한 뒤 같은 다음 turn을 실행한다. live continuation과 token,
  전체 logits bytes 및 최종 position **49**가 일치한다. snapshot 크기는 두 profile
  모두 **2,409,136 bytes**다.
- 여러 모델 인스턴스의 allocation witness를 구분해 각 28 owner의 누락/중복,
  manager 주소 대응, chunk별 byte 수, compatibility/layout copy 여부를 audit한다.

이는 동일 세션의 저장·복원 검증이다. 두 세션의 물리 KV 공유나 prefix COW를
구현했다는 의미는 아니다. 원본/변환 모델은 build 아래에만 보관하며 commit하지 않는다.
재현 옵션은 `--prefill-chunk 17`, `--session-only --snapshot-resume`다.
증거는 `build-kv-cpu/validation/qwen3-post/`의 `chunk17-comparison.json`,
`snapshot-resume/report.json`, 해당 trace audit JSON에 보존한다.

### Android 검증에서 발견한 재초기화 mode 누락

Android attention의 inference 전용 조건을 검사하면서 기존 graph의 mode 전달
누락을 발견했다. `NetworkGraph::finalizeContext()`는 실행 mode를 전달하지만,
`refinalizeContext()`는 이를 생략하여 `LayerNode::refinalize()`가 `InitLayerContext`를
기본 `TRAIN`으로 만들었다. 모델을 `INFERENCE`로 compile해도 재초기화 callback은
학습 mode를 받았다. graph의 현재 mode를 refinalize까지 전달하고, 직접 호출의 기본
`TRAIN` 동작과 기존 tensor type/loss scale 처리는 유지했다.

실제 `NeuralNetwork::reinitialize()`의 layer callback을 관찰하는 회귀 테스트를
추가했다. 수정 전에는 `INFERENCE`·`VALIDATE` 2개가 실패했고, 수정 후에는 세 mode와
기본값 4개가 모두 통과했다. 전체 graph **16 PASS**, KV **65 PASS**, 기존 모델
**99 PASS / deprecated QINT4 1 SKIP**이며 직접 모델 실행의 종료 코드도 0이다.
재현 증거는 `build-kv-android/validation/checks/refinalize-mode.*`와
`host-final-models.xml`, `/tmp/kv-android-refinalize-host-model-tests.log`에 보관했다.
앞선 sandbox Meson 실행기는 assertion 완료 뒤 종료 대기가 남아, 동일 바이너리의
직접 실행으로 정상 종료를 확인했다.

## Android Arm CPU 후속 검증

사용자 지정 기기 `bacd420f`(SM8650, Android 15, arm64-v8a)에서 NDK r28c로
CPU 엔진과 CausalLM을 빌드했다. Qwen FP32 checkpoint를 재사용하고, Q4_0/Q6_K
weight는 `--isa ARM`으로 다시 packing했다. 빌드·기기 단위 검사 명령은
[Android 빌드 기록](android-build-checks.md), 모델 hash와 상세 결과는
[Android Qwen 검증](android-cpu-validation.md)에 있다.
최종 runtime의 기기 단위 검사는 core 18 + facade 25 + ring 22 + attention profile
15 = **80 PASS, 0 FAIL, 0 SKIP**이며, 증거는 `checks/final-cpu.validation.json`이다.

모델 검증은 baseline·기본 half 수치 호환·FP32 attention·chunk17·강제 per-row·
snapshot을 포함한 **56회**와, 최종 수정 runtime의 **8회**로 총 **64회 통과**했다.
최종 8회는 앞선 FP32 attention 구현과 생성 token 및 모든 logits bytes가 일치한다.
초기 지원 조건 검사의 10회 초기화 거절은 별도 증거로 보관했다. 조사 중 위 graph
mode 누락을 회귀 테스트로 재현했고, 기기 진단에서는 engine-neutral MHA의 GPU
residency 표기를 실제 실행 backend로 오해한 새 검사 오류를 확인했다. 후자는
runtime을 선택하는 `NNTR_ENGINE=cpu`로 검사하고 중립 context를 허용하도록 바로잡았다.

지원 profile은 `NNTR_CPU_ATTENTION_FP32=1`, inference, batch 1, FP32 Q/K/V/O와
linear FP16 KV다. per-row 경로는 28 owner의 동일 backing 주소와 read/write alias,
append/history/tile copy 모두 0을 확인했다. ARM flash 경로는 full-prefix 복사를
제거했으나 계산용 tile packing이 남는다. 기본 geometry의 worker별 scratch는
65,536 bytes이며, 이 결과를 총 copy traffic 감소나 GPU/NPU 물리 메모리 공유의
증거로 해석하지 않는다. 기본 Android half 수치 경로는 유지한다.

FP32 출력 token은 HF와 일치하고 모든 logits가 finite다. Q4 장문 출력의
`France`와 HF/FP32의 `France.` 차이는 Linux에서 확인한 양자화 차이와 같다.
새로운 checkpoint/변환 오류나 비정상 출력은 확인하지 않았다. 최종 모델 증거는
`build-kv-android/validation/final-cpu/`, 전체 집계는 `android-summary.json`에 있다.

## 단계 경계

P0는 환경과 변경 전 baseline, P1은 내부 core storage/lease와 독립 테스트,
P2는 기존 CausalLM CPU allocation 및 graph 실행 연결이다. 각 단계에서 구현 범위,
실행한 검사, 남은 한계를 추가한다. GPU/NPU/CUDA direct access, quantized KV,
session 간 공유, Android 실기기 완료를 Linux CPU 결과로 대신하지 않는다.

| 단계별 commit | 내용 |
|---|---|
| `9a9662f69` | P0 환경 검토, 기존 모델 호환성, 빌드·검증 절차 |
| `21c87982b` | P1 core geometry/storage/step lease와 수명·실패 테스트 |
| `267c1b62d` | Gemma4 LM head 속성 버그 수정의 현재 브랜치 반영 |
| `2b9bf7890` | P2 CPU facade/graph transaction 연결, x86 KV prefix copy 제거 |
| `fc06bb03f` | 고정 revision Qwen 변환·weight 양자화·HF/기존 구현 비교 도구 |
| `15136f828` | Qwen chunk17 및 새 모델의 snapshot 복원 검증 |
| `504c49b05` | Android KV 단위 검사와 Linux/Android probe 빌드 지원 |
| `6964fb8c9` | graph 재초기화 시 실행 mode 보존과 회귀 검사 |
| `d3cf7049a` | Android CPU 직접 KV 쓰기, bounded tile 읽기와 profile 검사 |

Gemma4 독립 브랜치는 `fix/gemma4-skip-prefill-lmhead`, 해당 commit은
`21d03c477`이며 worktree는 `/tmp/nntrainer-gemma4-fix`다.
모든 commit은 기존 인간 author/committer identity와 DCO sign-off를 유지하고
`Co-authored-by: Codex (OpenAI)`를 포함한다. checkpoint, 변환 모델, 로그, logits,
KV snapshot 및 build 산출물은 commit하지 않는다. 사용자 기존 untracked
`Applications/Android/kotlin/.gradle/`도 변경·추가하지 않는다.
