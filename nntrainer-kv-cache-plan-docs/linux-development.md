# Linux에서 KV cache manager 개발 시작하기

> 이 문서는 최초 제안 절차다. 현재 Linux host의 실제 build 명령, tokenizer ABI
> 차이, 모델 호환성 및 실행 증거는 [Linux CPU 실행 기록](linux-cpu-execution.md)을
> 우선 참고한다. 개발 순서는 Linux CPU → Android Arm CPU → 이후 추후 결정이다.

이 문서는 [구현 계획](README.md)의 P0–P5를 실행하기 위한 절차다.
명령은 **Linux에서 실행할 제안**이며, Windows 분석 환경에서 빌드/테스트한 결과가 아니다.
소스 기준은 `141871a58ff439e3caeac40c1632ff8ac3543d20`이다.

## 1. Windows 작업 결과와 Linux 인계

이번 분석에서는 다음과 같이 작업했다.

- 원래 `C:/nntrainer`의 `gemma4-ple-fsu` 브랜치와 기존 untracked patch를 유지했다.
- `origin`이 upstream nntrainer/nntrainer임을 확인하고 `git fetch origin main`을 실행했다.
- `C:/nntrainer-kv-cache-plan`에 local main의 worktree를 만들고
  `git pull --ff-only origin main`을 실행했다.
- main이 `cf2b653b`에서 `141871a5`로 fast-forward된 후
  `docs/kv-cache-manager-plan` 브랜치를 만들었다.
- 이번 작성물은 `docs/kv-cache/` 아래 Markdown 3개이며 commit/push는 하지 않았다.

Linux로 옮길 때 Windows worktree 전체를 복사하지 않는다.
worktree의 `.git` 파일은 원래 Windows 저장소 경로를 가리킨다.
Linux에 새 clone을 만들고 Markdown 디렉터리만 복사하면 된다.

첨부용 `nntrainer-kv-cache-plan-docs.zip`을 사용하는 경우, 압축 내부의
`kv-cache/`를 새 clone의 `docs/` 밑에 푼다.

```bash
git clone https://github.com/nntrainer/nntrainer.git nntrainer-kv
cd nntrainer-kv
git switch main
git pull --ff-only origin main
git log -1 --format='%H %cs %s'

# 재현 가능한 첫 baseline은 분석 SHA로 고정한다.
KV_BASE_SHA=141871a58ff439e3caeac40c1632ff8ac3543d20
git switch -c feature/unified-kv-cache "$KV_BASE_SHA"
git submodule sync --recursive
git submodule update --init --recursive --depth 1

# 다운로드한 zip 경로에 맞춰 실행
unzip /path/to/nntrainer-kv-cache-plan-docs.zip -d docs
```

시작 시점의 최신 main으로 개발하려면 새 SHA로 차이 분석을 먼저 수행한다.
위 baseline 검증을 마친 뒤 최신 main 반영과 메모리 구조 변경을 서로 다른 변경으로
검토한다. `subprojects/` 초기화는 필요하지만 vendored 파일을 편집하지 않는다.

## 2. Linux 개발 환경

권고 host는 Ubuntu 24.04 x86_64다. repository가 x86에서 AVX2/FMA/F16C 옵션을
사용하므로 host CPU도 해당 instruction을 지원해야 한다.
Android 실기기 검증은 Linux host에서 cross build한다.

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential gcc-13 g++-13 clang clang-format-14 \
  cmake meson ninja-build pkg-config python3 python3-dev python3-numpy \
  libopenblas-dev libiniparser-dev libjsoncpp-dev \
  libgtest-dev libgmock-dev libunwind-dev \
  libcurl4-openssl-dev unzip wget

gcc-13 --version
meson --version
ninja --version
clang-format-14 --version
git submodule status --recursive
```

기본 개발 profile은 TFLite/NNStreamer/C API 의존성을 꺼서 범위를 줄인다.
전체 CI 재현은 [.github/workflows/ubuntu_clean_meson_build.yml](../.github/workflows/ubuntu_clean_meson_build.yml)
및 [getting-started.md](../docs/getting-started.md)의 dependency 구성을 사용한다.
배포판 package 이름이나 fallback dependency는 Meson log로 확인한다.

x86 Linux tokenizer archive는 tree에 있다. 새 archive가 필요하거나 aarch64 Linux를
사용한다면 [build_tokenizers_c.sh](../Applications/CausalLM/tools/build_tokenizers_c.sh)로
별도 output 디렉터리에 build하고 `-Dcausallm-tokenizer-lib=...`로 지정한다.
추적 중인 archive를 덮어쓰지 않는다.

## 3. 변경 전 CPU baseline

아래 변수는 같은 Bash session에서 사용한다. core/model baseline은 비FP16과 FP16
build를 분리한다. **현재 broad model unittest는 enable-fp16=false에서만 생성된다.**

```bash
KV_COMMON_OPTS=(
  -Dplatform=none
  -Denable-app=true
  -Denable-transformer=true
  -Denable-test=true
  -Denable-api=false
  -Denable-opencl=false
  -Denable-cuda=false
  -Denable-npu=false
  -Denable-htp=false
  -Denable-tflite-interpreter=false
  -Denable-tflite-backbone=false
  -Denable-nnstreamer-backbone=false
  -Dml-api-support=disabled
  -Denable-capi=disabled
  -Denable-nnstreamer-tensor-filter=disabled
  -Denable-nnstreamer-tensor-trainer=disabled
  -Dopenblas-num-threads=1
)

CC=gcc-13 CXX=g++-13 meson setup build-kv-cpu \
  --buildtype=debugoptimized \
  -Denable-fp16=false "${KV_COMMON_OPTS[@]}"
ninja -C build-kv-cpu

meson test -C build-kv-cpu --list
NNTR_ENGINE=cpu meson test -C build-kv-cpu --print-errorlogs \
  unittest_kv_cache_manager unittest_kv_ring unittest_causallm_models

CC=gcc-13 CXX=g++-13 meson setup build-kv-fp16 \
  --buildtype=debugoptimized \
  -Denable-fp16=true "${KV_COMMON_OPTS[@]}"
ninja -C build-kv-fp16
NNTR_ENGINE=cpu meson test -C build-kv-fp16 --print-errorlogs \
  unittest_kv_cache_manager unittest_kv_ring
```

확인할 것은 exit code만이 아니다.

- test 목록에 요청한 target이 실제로 있는지 확인한다.
- fixture가 없어서 model tests가 모두 SKIP한 경우 baseline 완료가 아니다.
- 현재 Meson은 fixture tarball을 source의 reference 디렉터리에 풀 수 있다.
  build 뒤 `git status --short`를 확인하고 생성물과 의도한 수정 파일을 구분한다.
- `NNTR_QUANTIZE_BIN`은 model test의 Meson env에서 설정된다.
  binary를 직접 실행한다면 직접 지정한다.
- INT8이나 device 관련 환경 변수가 baseline에 섞이지 않도록 `NNTR_*` 설정을 기록한다.
  특히 presence를 검사하는 flag는 `=0`도 켜진 것으로 처리될 수 있으므로 baseline에서는 unset한다.
- 기존 helper 단위 테스트는 작은 batch=2 FP32 geometry도 사용한다.
  신규 batch=1 API를 만들면서 기존 테스트를 제거하지 않는다.

기존 [run_unittest_x86.sh](../Applications/CausalLM/run_unittest_x86.sh)는 모델 테스트를
빠르게 실행하는 대안이다. 위 전용 build와 다른 `builddir_x86_test`를 사용하고
기본 dependency 옵션도 다르다. 위 baseline과 섞어서 같은 build라고 보고하지 않는다.

## 4. 메모리 오류 및 플랫폼 독립성

P1에서 새로운 테스트 target을 등록하기 전에는 기존 target만 실행한다.
`unittest_kv_storage` 같은 이름은 구현 계획의 제안 이름이며 아직 존재하지 않는다.

```bash
CC=gcc-13 CXX=g++-13 meson setup build-kv-asan \
  --buildtype=debug -Db_sanitize=address,undefined -Db_lundef=false \
  -Denable-fp16=false "${KV_COMMON_OPTS[@]}"
ninja -C build-kv-asan
NNTR_ENGINE=cpu meson test -C build-kv-asan --print-errorlogs \
  unittest_kv_cache_manager unittest_kv_ring

# core가 CausalLM/application에 역으로 의존하지 않는지 별도 확인
CC=gcc-13 CXX=g++-13 meson setup build-kv-core \
  --buildtype=debugoptimized \
  -Denable-app=false -Denable-transformer=false -Denable-test=true \
  -Denable-opencl=false -Denable-cuda=false -Denable-npu=false \
  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false \
  -Dml-api-support=disabled -Denable-capi=disabled \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled
ninja -C build-kv-core
```

sanitizer는 CPU에서 owner/bounds/abort를 검증하는 데 사용한다.
vendor GPU/NPU driver가 포함된 sanitizer 결과는 별도로 해석한다.
fake backend로 늦게 끝나는 completion, 등록 실패, allocation 실패를 주입하고
owner가 마지막 작업 전에 해제되지 않는지 검증한다.

Clang build, `-Denable-fp16=true -Dfp16-impl=wrapper` build도 회귀 범위다.
core header에 Linux fd/ioctl, OpenCL/QNN/CUDA SDK header가 노출되지 않도록 확인한다.
실제 device adapter의 POSIX 코드는 platform guard 아래 둔다.

## 5. OpenCL 개발과 실기기 검증

```bash
sudo apt-get install -y opencl-headers ocl-icd-opencl-dev clinfo

# 앞 절의 KV_COMMON_OPTS와 같은 세션에서 실행.
# 중복 -D 옵션을 넘기지 않도록 OpenCL 항목을 바꾼 배열을 만든다.
KV_CL_OPTS=()
for opt in "${KV_COMMON_OPTS[@]}"; do
  if [[ "$opt" == "-Denable-opencl=false" ]]; then
    KV_CL_OPTS+=("-Denable-opencl=true")
  else
    KV_CL_OPTS+=("$opt")
  fi
done

CC=gcc-13 CXX=g++-13 meson setup build-kv-cl \
  --buildtype=debugoptimized -Denable-fp16=true "${KV_CL_OPTS[@]}"
ninja -C build-kv-cl
clinfo > build-kv-cl/device-capabilities.txt
meson test -C build-kv-cl --list
```

ICD loader/header만 설치하면 compile은 가능하지만 GPU 실행을 검증한 것은 아니다.
실제 device ICD가 필요하다. 현재 OpenCL CI도 compile/link 확인용이다.
POCL CPU device에서 통과한 결과를 Android GPU의 zero-copy 근거로 쓰지 않는다.

새 probe executable은 다음 결과를 JSON으로 출력하도록 P0/P3에서 구현한다.

| probe | 기록할 것 |
|---|---|
| device | vendor, device, driver, OpenCL version, extension list |
| allocation | 요청 및 실제 memory kind, SVM grain, alignment, host fallback |
| read/write | CPU pattern → GPU 검증/수정 → CPU 검증, offset/경계/guard 영역 |
| synchronization | map/unmap, producer/consumer event, 지연 완료, 반복 handoff |
| cache binding | source view와 kernel backing id/offset 일치 |
| copy | enqueue read/write/copy, scatter/gather, identity kernel byte 수 |
| import | QNN용 buffer의 GPU import 성공 여부와 정확한 exporter/handle type |

초기 profile은 현행 FP16 buffer/SVM 경로로 제한한다.
Adreno image profile은 source 분석의 mirror 경로를 별도 계측하고,
선택한 engine/config 및 `NNTR_*` 값을 결과에 함께 저장한다.
단순히 `NNTR_GPU_SVM_POOL=1`이라고 쓴 것만으로 direct access를 통과시키지 않는다.

## 6. Android 및 QNN

repository CI와 같은 NDK r26d를 첫 baseline으로 사용한다.
QNN SDK 버전, HTP architecture, device firmware/driver, graph binary/exporter 버전을
서로 호환되는 조합으로 고정한다.

```bash
export ANDROID_NDK=/path/to/android-ndk-r26d
export PATH="$ANDROID_NDK:$PATH"
adb devices

./Applications/CausalLM/build_android.sh --help
./Applications/CausalLM/build_tokenizer_android.sh
./Applications/CausalLM/build_android.sh
./Applications/CausalLM/run_unittest_android.sh

# OpenCL native build는 별도 작업 디렉터리/checkout 사용을 권고한다.
./Applications/CausalLM/build_android.sh -Denable-opencl=true

# QNN build: SDK를 확보한 환경에서 실행
export QNN_SDK_ROOT=/path/to/compatible-qnn-sdk
./Applications/CausalLM/build_android.sh --qnn
```

`build_android.sh`의 기본 variant는 cpu이고 `--qnn`이 NPU 옵션을 관리한다.
`-Denable-npu=true`를 직접 덧붙이지 않는다.
`--install` 없이도 build할 수 있으며 실제 probe/test 배포는 별도로 수행한다.

위 Android unittest script는 기존 모델 회귀용이다. 새 storage/lease/QNN probe를
자동으로 포함하지 않는다. P1/P4에서 Android test target과 deploy/run 경로를 추가한다.
현재 CausalLM canonical build는 `app_build/meson.build`에도 source 목록이 있으므로
root Meson 수정만으로 Android에 새 manager가 들어간다고 가정하지 않는다.

### 6.1 QNN 검증 순서

1. 작은 graph로 기존 input RPC 등록/execute baseline을 확인한다.
2. manager allocation을 같은 input에 연결하고 복사 없이 값이 전달되는지 검증한다.
3. nonzero offset의 output view를 등록한다. guard bytes를 두어 인접 영역 오염을 잡는다.
4. K/V 각각 output layout과 cache layout의 모든 stride를 비교한다.
5. graph/exporter 변경을 적용해 최종 producer가 persistent destination에 쓰게 한다.
6. prefill→decode 및 generation→다음 turn prefill을 같은 storage로 실행한다.
7. register/unregister 수, init/재load 실패, context 2개 사용, cancellation을 검증한다.
8. 마지막에 실제 공개 Gemma4 QNN model의 append helper 사용이 사라졌는지 확인한다.

동일 메모리 input/output overlap은 문서/SDK 계약과 작은 실행 실험이 모두 필요하다.
memRegister 성공만으로 허용됐다고 판정하지 않는다.
graph 변환 source가 없다면 input direct read까지 진행할 수 있으나
output direct write는 미완료 항목으로 남긴다.

CPU/GPU/NPU 세 장치 공유 probe는 동일 **backing identity**로 순차 write/read 검증한다.
CPU virtual address가 모든 runtime에서 동일할 필요는 없지만 physical backing과
offset의 동일성을 추적할 수 있어야 한다. profiler가 physical copy를 관측할 수 없으면
그 항목은 `unknown`으로 기록한다.

## 7. CUDA 및 추가 build

CUDA는 지원 Linux GPU와 toolkit을 갖춘 환경에서 아래 구성으로 compile한다.
이 명령은 실행에 필요한 model/profile 선택까지 자동으로 보장하지 않는다.

```bash
CC=gcc-13 CXX=g++-13 meson setup build-kv-cuda \
  --buildtype=debugoptimized \
  -Denable-transformer=true -Denable-app=true -Denable-test=true \
  -Denable-fp16=true -Denable-cuda=true -Dcuda-path=/usr/local/cuda \
  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false \
  -Dml-api-support=disabled -Denable-capi=disabled \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled
ninja -C build-kv-cuda
meson test -C build-kv-cuda --list
```

현재 main에는 CUDA 전용 `unittest_cuda_gemma4` target도 있다.
device memory와 managed memory 결과를 분리하고 page migration 및
host fallback 여부를 함께 측정한다. Windows compile, Android cross build,
Tizen/GBS, Yocto 회귀는 각 repository CI 설정을 따른다.

## 8. 결과 기록 양식

아래 양식은 **아직 실행하지 않은** 결과의 템플릿이다.
빈 칸을 통과로 간주하지 않는다. 예: `artifacts/kv-cache/<date>/<profile>/`에
build log, test XML, capability JSON, copy trace, benchmark CSV를 함께 보관한다.

```yaml
source_sha: ""
baseline_sha: "141871a58ff439e3caeac40c1632ff8ac3543d20"
build:
  os: ""
  compiler: ""
  meson_options: []
  sdk_versions: {}
device:
  soc: ""
  gpu: ""
  npu: ""
  driver: ""
model:
  weights_sha256: ""
  graph_sha256: ""
  config_sha256: ""
  batch: 1
  prompt_tokens: 0
  decode_tokens: 0
cache:
  logical_layout: ""
  physical_k_layout: ""
  physical_v_layout: ""
  encoding: ""
  actual_memory_kind: ""
  unique_backing_bytes: 0
  mirror_bytes: 0
  append_copy_bytes: 0
  history_copy_bytes: 0
  layout_conversion_bytes: 0
  physical_migration: unknown
result:
  executed: false
  direct_read: unverified
  direct_write: unverified
  cross_backend_sharing: unverified
  correctness: unverified
  skipped_tests: []
  unsupported_reason: ""
  ttft_ms_median: null
  token_latency_ms_p95: null
```

timing은 instrumentation on/off를 분리한다. 새 코드와 baseline에서 같은 model,
prompt, thread 수, sampling seed, engine profile, thermal 조건을 사용한다.
profile은 “OpenCL” 한 단어보다 구체적으로 SoC/layout/encoding/consumer를 기록한다.

## 9. 첫 주 실행 목록

| 일차 | 작업 | 남겨야 할 산출물 |
|---|---|---|
| 1 | SHA 고정, submodule, CPU 비FP16/FP16 baseline | configure/test log, fixture 실행 확인 |
| 2 | Tensor ownership와 실제 K/V copy 계측 | pointer/backing 지도, byte counter baseline |
| 3 | GPU capability, QNN fd/offset 및 K stride probe | 지원/미지원 행렬, graph 변경 요구 |
| 4 | core spec/storage/lease와 fake allocator 구현 시작 | checked sizing 및 수명 테스트 |
| 5 | CausalLM thin facade와 direct read 연결 | 기존 model parity, 다음 주 direct producer 범위 |

device가 준비되지 않았다면 3일차의 실기기 항목을 미실행으로 남기고 portable
core/fake backend 작업을 계속한다. 테스트 mock은 실기기 gate의 대체가 아니다.

## 10. commit/PR 전 점검

- repository [AGENTS.md](../AGENTS.md)를 다시 확인한다.
- 변경 C/C++에 clang-format 14를 적용하고 `git diff --check`를 실행한다.
- 해당 단계의 focused test와 영향 있는 build matrix를 실행한다.
- formatter가 건드린 무관한 줄이나 vendored 파일을 PR에 포함하지 않는다.
- commit은 component prefix, DCO sign-off와 agent co-author trailer를 갖춘다.
- [.github/PULL_REQUEST_TEMPLATE.md](../.github/PULL_REQUEST_TEMPLATE.md)를 그대로 사용한다.
  Build/Run test의 미실행 항목은 Skipped로 정직하게 표시한다.
- 새 테스트 등록과 check_count 영향을 확인한다.
- 요구 1 완료 판단에는 실기기 copy 증거를 첨부한다. compile 성공만으로 대체하지 않는다.
