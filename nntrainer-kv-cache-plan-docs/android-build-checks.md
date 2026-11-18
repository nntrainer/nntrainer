# Android CPU 빌드와 KV 단위 검증

2026-10-28, 사용자 지정 기기 **bacd420f**에서 Linux CPU 작업을 이어 검증했다.
모델 실행 결과와 직접 접근 범위는 [Android Qwen 검증](android-cpu-validation.md)에
별도로 기록한다. `AGENTS.md`, 원본 모델과 vendored source는 수정하지 않는다.

## 실제 환경과 선택

| 항목 | 확인값 |
|---|---|
| 기기 | `23116PN5BC`, SM8650, `arm64-v8a` |
| OS | Android 15, API 35 |
| NDK | `/opt/android-ndk-r28c`, `28.2.13676358` |
| 빌드 API / STL | android-29 / c++_shared |
| CPU ISA | `armv8.2-a+fp16+dotprod+i8mm` |
| 런타임 | `NNTR_ENGINE=cpu`, NNTrainer threads 4, OpenBLAS threads 1 |
| 전용 기기 경로 | `/data/local/tmp/nntr_kv_cpu_bacd420f` |

기기의 `/proc/cpuinfo`에서 FP16, dot product, i8mm 기능을 확인했다. 계획서의
NDK r26d를 이 환경에서 실행한 것처럼 기록하지 않고 실제 설치된 r28c를 사용한다.
엔진은 FP16을 지원하지만 모델의 activation dtype은 FP32다. 기존 Android attention은
이 입력을 내부에서 FP16으로 좁히므로 Linux와의 bitwise 일치를 전제로 하지 않는다.

## 재현 빌드

초기화한 dependency는 기존 xgrammar/googletest와 고정 revision의 iniparser다.
OpenBLAS는 저장소가 지정한 `openblas-0.2.20.tar.gz`를 builddir에 준비했다.
tokenizer는 in-tree Cargo.lock과 설치된 Android Rust target으로 별도 빌드했다.

```bash
git submodule update --init --recursive --depth 1 \
  Applications/CausalLM/xgrammar subprojects/googletest subprojects/iniparser
ANDROID_NDK=/opt/android-ndk-r28c \
  Applications/CausalLM/tools/build_tokenizers_c.sh --target=android --offline \
  --out-dir="$PWD/build-kv-android/tokenizer"
cp build-kv-android/tokenizer/aarch64-linux-android/libtokenizers_c.a \
  Applications/CausalLM/lib/libtokenizers_android_c.a

ANDROID_NDK=/opt/android-ndk-r28c ANDROID_SERIAL=bacd420f \
  Applications/CausalLM/build_android.sh --nntr-threads=4 \
  -Denable-app=false -Denable-test=false \
  -Denable-opencl=false -Denable-cuda=false -Denable-htp=false \
  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false \
  -Denable-capi=disabled -Dml-api-support=disabled \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled
```

`enable-app/test=false`는 엔진 패키징 단계의 옵션이다. CausalLM 앱은 이어지는
canonical `app_build/meson.build` graph로 별도 cross compile된다. 생성물은
`builddir/android_build_result/`와 `builddir_app/cpu/`에 있다. 앱 소스 수정 후에는
`ninja -C builddir_app/cpu -j8`로 갱신할 수 있다.

기존 prepare_encoder의 재다운로드를 피하기 위해 이미 header가 준비된 builddir의
`encoder/` marker를 유지했다. tokenizer와 모델·runtime·로그는 ignore되는 생성물이며
추적 중인 x86 tokenizer archive를 교체하지 않았다.

## ABI와 probe

생성된 `nntrainer-abi.ini`는 다음 flags를 앱과 테스트에 전달한다.

```text
-march=armv8.2-a+fp16+dotprod+i8mm -DENABLE_FP16=1 -DUSE__FP16=1
```

root와 standalone Meson graph의 main object 이름이 달라 기존 probe builder를
일반화했다. 실제 `compile_commands.json`의 `-o` object로 target과 link 치환을
결정하며 Android에는 host 절대경로 RUNPATH를 추가하지 않는다.
관찰용 probe TU는 `-fno-fast-math`로 컴파일해 `std::isfinite` 검사가 상수 true로
최적화되지 않게 한다. 엔진/모델의 연산 flags를 바꾼 것은 아니다.

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py --build builddir_app/cpu \
  --output build-kv-android/validation/baseline/qwen_cpu_probe
```

Linux root graph와 Android standalone graph 양쪽에서 probe compile/link를 확인했다.
baseline의 네 library, probe, ABI, source/hash metadata는
`build-kv-android/validation/baseline/`에 보존했다. 다른 ABI의 header와 library를
섞지 않는다.

## 기기 KV 단위 검증

JNI harness에 core/facade/ring executable을 추가하고 private `<kv_cache/...>`
include를 연결했다. 각 테스트가 자체 main을 가지므로 gtest-all과 gtest_main을
분리했다. snapshot fixture는 Android에 없는 `/tmp` 대신 `TMPDIR` 또는 cwd를 쓴다.

```bash
ANDROID_NDK=/opt/android-ndk-r28c /opt/android-ndk-r28c/ndk-build \
  -C Applications/CausalLM/jni NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs \
  NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk \
  NDK_APPLICATION_MK=./Application.mk \
  CAUSALLM_PREBUILT_LIB=../../../builddir_app/cpu/libcausallm.so \
  unittest_kv_cache_core unittest_kv_cache_manager unittest_kv_ring -j4
```

NDK의 googletest source를 기존 harness의 ignored `jni/googletest/` 위치에 준비했다.
모든 ADB 호출은 `adb -s bacd420f`를 사용하며, 배포와 실행은 전용 `checks/` 경로에서
진행했다. `TMPDIR`와 `LD_LIBRARY_PATH`도 해당 절대경로로 고정했다.

| 초기 baseline runtime 기기 실행 | 결과 |
|---|---|
| core geometry/storage/lease | 18 PASS |
| CausalLM facade/snapshot | 25 PASS |
| ring geometry | 22 PASS |
| 초기 합계 | **65 PASS, 0 FAIL, 0 SKIP** |

변경된 fixture의 Linux facade 25개도 통과했다. Android XML, stdout, executable hash와
ABI 기록은 `build-kv-android/validation/checks/validation.json`과 같은 디렉터리에 있다.
이 단위 검증 통과만으로 attention 직접 접근 완료를 판단하지 않는다. Qwen 실제 모델의
주소·copy 계측과 출력 검증은 별도 단계다.

## 실제 모델 초기화에서 발견한 통합 문제

batch=1 / inference guard를 추가한 첫 실제 Qwen FP32·Q4 smoke 2개가 추론 전에
초기화에서 거절되었다. 조사 과정에서 `LayerNode::refinalize()`가
`InitLayerContext`를 만들 때 graph의 execution mode를 전달하지 않아 기본 `TRAIN`을
사용하는 기존 오류를 확인했다. `compile(INFERENCE)`를 호출해도 재초기화 context는
달라질 수 있었다. graph mode를 refinalize까지 전달하도록 엔진 코드를 수정했으며,
앱만 교체해서 반영되는 변경이 아니므로 Android 엔진과 앱을 함께 다시 빌드했다.

이미 구성된 동일 build 디렉터리에서는 다음 명령을 실제로 사용했다.

```bash
ninja -C builddir -j8
meson install -C builddir --no-rebuild --skip-subprojects
ninja -C builddir_app/cpu -j8
```

이 mode 오류는 host regression으로 재현했다. 수정 전 INFERENCE·VALIDATE 두 case가
실패했고 수정 후 mode 보존 및 기존 training 기본 동작 4개가 모두 통과했다.
증거는 `checks/refinalize-mode.before.log`, `refinalize-mode.after.log/.xml`에 있다.
첫 실제 모델 초기화 거절 2개는 `final-guarded/`의 per-model stderr·report에 보존했다.
첫 로그에는 실제 context mode를 출력하지 않았으므로 두 거절 모두의 직접 원인이
mode였다고 단정하지 않는다.

mode를 보존한 다음 진단 runtime에서는 실제 Qwen 8개 case가 여전히 초기화에서
거절되었다. 이번 메시지는 `process_engine=cpu`, `layer_engine=1`,
`execution_mode=1`, `batch=1`이었다. CausalLM의 residency policy는 `mha_core`를
engine-neutral type으로 선언하며 `LayerNode::finalize()`는 이 context에 GPU stamp를
부여한다. 이 값은 CPU attention의 실제 실행 backend를 뜻하지 않는다. 잘못 추가한
context-engine CPU 검사를 제거하고 실제 runtime 선택인 `NNTR_ENGINE=cpu` 조건을
유지했다. 두 번째 진단의 8개 실패는 `final-verified/`에 보존했다. 이 10개 거절은
추론 출력을 생성하지 않았으며 성공 모델 검증 횟수에 포함하지 않는다.

## 최종 final-cpu runtime 재검증

Android FP32 attention opt-in은 CPU process, batch=1, inference, 외부 FP16 cache와
FP32 Q/K/V activation을 요구한다. profile suite는 CPU context의 지원 profile,
CPU process에서 engine-neutral GPU context 허용, 명시적 `0`의 legacy 유지 3개와
training/validation, batch=2, 비CPU process, half Q/K/V, FP32 cache, 내부 cache,
RoPE 없음, INT8, sink 거절 12개로 구성한다. **15개가 실제 기기에서 모두 통과**했다.
이 target은 production `mha_core.cpp`를 직접 컴파일한다. 테스트용 context만으로
전체 graph 통합을 대신하지 않으며 실제 Qwen 검증은 별도 문서에 기록한다.

mode 전달 수정과 engine-neutral context 허용을 모두 반영한 최종 runtime을
`build-kv-android/validation/final-cpu/runtime/`에 고정했다. 해당 디렉터리의 네 library를
기기의 전용 `checks/`에 배포하고 네 test target을 다시 빌드·실행했다. 로컬 frozen
파일과 기기 library·실행 파일의 SHA256이 모두 일치한다.
`final-cpu/build.json`은 반영된 source 파일 및 runtime hash를 함께 보존한다.

```text
libnntrainer.so: 4,960,352 bytes
SHA256: 0b7204882344ffe917cd1fd4e991e84e7e0930f0ad5611fbf31a1727d7479ad8
libcausallm.so: 17,591,568 bytes
SHA256: 78502cc34a2ee950e3d177186a0bb3c43950376e4df3f6daa1aea21ca24f0cd7
```

| 최종 final-cpu 기기 단위 검증 | 결과 |
|---|---|
| core geometry/storage/lease | 18 PASS |
| CausalLM facade/snapshot | 25 PASS |
| ring geometry | 22 PASS |
| Android FP32 profile 승인·거부 | 15 PASS |
| 합계 | **80 PASS, 0 FAIL, 0 SKIP** |

네 실행 파일 모두 exit code `0`이며 disabled·error도 `0`이다. 증거는
`build-kv-android/validation/checks/final-cpu.validation.json`,
`final-cpu.device.sha256.txt`, `final-cpu.ndk-build.log`와
`final-cpu.unittest_*.xml/.log`에 있다. manifest는 네 suite의 실제 결과와 exit code,
source/executable/runtime/device hash, XML·log hash 및 ADB 명령을 기록한다.
기존 profile 15개 결과와 다른 runtime의 65개 결과를 합산한 수치가 아니라 네 suite를
같은 최종 runtime으로 다시 실행한 결과다.

재현용 build와 기기 실행 명령은 다음과 같다. 여기서 단위 테스트는 모델을 로드하지
않고 전용 `checks/`만 사용한다.

```bash
ANDROID_NDK=/opt/android-ndk-r28c /opt/android-ndk-r28c/ndk-build \
  -C Applications/CausalLM/jni NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs \
  NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk \
  NDK_APPLICATION_MK=./Application.mk \
  CAUSALLM_PREBUILT_LIB=../../../builddir_app/cpu/libcausallm.so \
  unittest_kv_cache_core unittest_kv_cache_manager unittest_kv_ring \
  unittest_mha_cpu_profile -j4

adb -s bacd420f push \
  build-kv-android/validation/final-cpu/runtime/libnntrainer.so \
  build-kv-android/validation/final-cpu/runtime/libccapi-nntrainer.so \
  build-kv-android/validation/final-cpu/runtime/libcausallm.so \
  build-kv-android/validation/final-cpu/runtime/libc++_shared.so \
  /data/local/tmp/nntr_kv_cpu_bacd420f/checks/

for suite in unittest_kv_cache_core unittest_kv_cache_manager unittest_kv_ring \
  unittest_mha_cpu_profile; do
  adb -s bacd420f push \
    "Applications/CausalLM/jni/obj/local/arm64-v8a/$suite" \
    /data/local/tmp/nntr_kv_cpu_bacd420f/checks/
  adb -s bacd420f shell \
    "cd /data/local/tmp/nntr_kv_cpu_bacd420f/checks && chmod 755 $suite && \
    LD_LIBRARY_PATH=/data/local/tmp/nntr_kv_cpu_bacd420f/checks \
    TMPDIR=/data/local/tmp/nntr_kv_cpu_bacd420f/checks NNTR_ENGINE=cpu \
    ./$suite --gtest_color=no --gtest_output=xml:final-cpu.$suite.xml" \
    > "build-kv-android/validation/checks/final-cpu.$suite.log" 2>&1
  adb -s bacd420f pull \
    "/data/local/tmp/nntr_kv_cpu_bacd420f/checks/final-cpu.$suite.xml" \
    build-kv-android/validation/checks/
done
```

이전 baseline 65개, 최초 profile 15개, 중간 guarded 65개는 기존 파일에 유지하며,
잘못된 context guard가 있던 중간 80개는
`checks/history/final-verified-context-guarded/final-verified.validation.json`과
인접 XML·로그에 별도 보존했다. 최종 결과는 `final-cpu.*`만 가리킨다.

## 2026-10-29 리뷰 반영: runtime 선택과 opt-in 기본값

위 `final-cpu` 기록은 2026-10-28에 고정한 runtime의 결과다. 이후 리뷰에서 Android
CPU 변경의 기본 동작 범위를 줄이고 engine 판정을 통합했다. 새 빌드에서는 다음 값을
명시한다. 모델·기기 회귀 결과는 해당 source/runtime hash로 별도 기록한다.

| 옵션 | 기본값 | 활성화 효과 |
|---|---|---|
| `NNTR_CPU_ATTENTION_FP32` | `0` | 지원되는 CPU external-cache profile에서 Q/K/V/O를 FP32로 유지 |
| `NNTR_CPU_KV_TILED` | `0` | ARM CPU flash가 전체 Ka/Va history 대신 bounded K/V tile을 pack |
| `NNTR_CPU_KV_LEGACY` | `0` | 비교·회귀 검증을 위해 기존 CPU facade allocation 경로 선택 |
| `NNTR_KV_CPU_TRACE` | `0` | 선택된 CPU runtime의 allocation/access/history·tile copy 증거 출력 |

FP32와 tiled 옵션은 독립적이며 `=0`으로 명시적으로 끌 수 있다. 두 옵션을 모두
설정하지 않은 Android CPU는 기존 half attention과 full-history packing을 유지한다.
따라서 trace의 `flash-packed`는 실제 Ka/Va 복사량을 `history_copy_bytes`로 보고한다.
`tiled=1`일 때만 `flash-tiled`, `history_copy_bytes=0`, 실제 `tile_pack_bytes`와
worker별 `tile_scratch_bytes`를 보고한다. full-history copy와 bounded kernel scratch를
하나의 zero-copy 수치로 합치지 않는다.

`runtime_config.h`의 공통 선택 함수는 `NNTR_ENGINE`의 대소문자를 정규화하고 live
context registry로 backend를 확인한다. 명시한 backend가 등록되어 있으면 사용하고,
미등록이면 core `Engine::parseComputeEngine`과 같이 CPU로 fallback한다. 변수가 unset이면
등록된 GPU, CUDA, CPU 순서로 선택한다. CPU-only 환경의 unset과 명시적 `cpu`는
동일한 engine, cache allocation, FP32 승인 정책과 trace를 사용한다. `llm_util.hpp`,
KV facade allocation, MHA finalize가 이 함수를 공유한다.

MHA는 finalize에서 engine과 CPU trace/FP32/tiled/device-attention 설정을 snapshot으로
보존한다. 실행 중인 각 layer/step은 같은 결정을 사용하며 반복적인 env/string 판정을
피한다. 새 객체 또는 재finalize는 env를 다시 읽으므로 서로 다른 설정을 같은 test
process에서 비교해도 process-static cache의 첫 값에 묶이지 않는다. engine-neutral
context의 GPU residency stamp는 여전히 CPU 실행을 거절하는 기준이 아니다.

공통 admission 정책은 Android macro 밖의 순수 함수로 추출했다.
`test/unittest/meson.build`의 `unittest_causallm_runtime`은 transformer/app 활성화와
독립적으로 등록되며 engine 선택, 옵션 기본값·독립성·snapshot, FP32 profile의
승인·거부를 **Linux 23개 실제 test**로 검증한다. 로컬 executable 직접 실행은 exit `0`,
23 PASS였고 source hash·XML·stdout·compile log는
`build-kv-cpu/validation/review-runtime-policy/validation.json`과 인접 파일에 보존했다.
Sandbox 안의 Meson runner는 PASS 출력 후 subprocess 종료를 기다리다 timeout되었으나,
정상 환경에서 같은 `unittest_causallm_runtime`과 legacy facade target을 함께 실행해
**23 + 20 PASS, Meson exit 0**을 확인했다. 실행기 현상을 CI 정책 테스트 실패와
구분한다. 정상 runner 로그도 `review-runtime-policy/meson-unsandboxed.log`에 보존한다.
Android 전용 finalize suite는 unset CPU case를 추가해 16개다.

기존 facade fixture의 공통 20개를 legacy allocation에서도 실행하도록 Meson에
`unittest_kv_cache_manager_legacy`를 등록했다. 기본 CPU-core 실행은
`NNTR_CPU_KV_LEGACY=0`, legacy 실행은 `=1`을 고정하며 core-only transaction case는
legacy 호출에서 제외한다. 추가한 두 CI target은 fake SKIP에 의존하지 않는다.

```bash
ninja -C build-kv-cpu test/unittest/unittest_causallm_runtime \
  Applications/CausalLM/unittest_kv_cache_manager
build-kv-cpu/test/unittest/unittest_causallm_runtime
NNTR_ENGINE=cpu NNTR_CPU_KV_LEGACY=1 \
  build-kv-cpu/Applications/CausalLM/unittest_kv_cache_manager \
  '--gtest_filter=-*.cpu_*:KVCacheManagerCpu.*:*.active_step*'
```


## 2026-10-29 최종 candidate의 native 85 + 실제 graph 7

성능 비교에 사용한 고정 runtime
`build-kv-perf/runtime/android-candidate/`와 동일한 네 library로 `bacd420f`에서
최종 경계 검증을 실행했다. production source 9개의 hash가 candidate `build.json`과
일치하고, 기기에 배포한 library 및 executable hash도 로컬과 일치한다.

```text
libnntrainer.so SHA256:
37d5d8cfb4b6adf41ed3cea19a822c7981a441e3d45055fdd969c1e941f99d1a
libcausallm.so SHA256:
80657e37ed93d97908443d7ace29c2be679a5ce4ad3de9bd5c5964cf12b32ea4
```

| 기기 검증 그룹 | 결과 |
|---|---|
| native core | 22 PASS |
| native facade | 25 PASS |
| native ring | 22 PASS |
| native MHA profile | 16 PASS |
| native 소계 | **85 PASS** |
| 실제 Qwen graph 경계 | 5 PASS |
| 실제 LFM2 hybrid graph 및 logits pool | 2 PASS |
| actual graph 소계 | **7 PASS** |
| 합계 | **92 PASS, 0 FAIL, 0 SKIP** |

다섯 executable 호출 모두 exit `0`이며 error·disabled도 `0`이다. host pure policy
23개와 legacy facade 20개는 별도 결과이며 위 native 92개에 더하지 않는다.

Qwen graph는 capacity까지 commit/reset 반복, 실행되지 않은 선언 producer 때문에
commit 거절, foreign K/V의 write 이전 거절, 잘못된 interval 후 정상 실행 복구를
검증했다. LFM2 graph는 attention+conv 경로의 producer 선언, 연속 decode,
logits pool 재사용과 overflow 시 position·이전 logits 보존을 검증했다.
실제 model wrapper와 MHA hook을 호출하는 fixture를 사용했으며 기존 raw graph helper의
숫자 비교만으로 대체하지 않았다. Android broad golden 전체는 실행하지 않았다.

NDK model target을 기존과 같은 j4 옵션으로 빌드한 뒤 아래 filter만 사용했다.
환경은 `NNTR_ENGINE=cpu`, `NNTR_CPU_KV_LEGACY=0`,
`NNTR_CPU_ATTENTION_FP32=0`, `NNTR_CPU_KV_TILED=0`, `NNTR_KV_CPU_TRACE=0`으로 고정했고,
각 test의 scoped-env override는 fixture가 복구한다. `TMPDIR`와 `LD_LIBRARY_PATH`는
모두 `/data/local/tmp/nntr_kv_cpu_bacd420f/checks`다.

```bash
ANDROID_NDK=/opt/android-ndk-r28c /opt/android-ndk-r28c/ndk-build \
  -C Applications/CausalLM/jni NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs \
  NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk \
  NDK_APPLICATION_MK=./Application.mk \
  CAUSALLM_PREBUILT_LIB=../../../builddir_app/cpu/libcausallm.so \
  unittest_causallm_models -j4
# 기기의 checks 경로에서 해당 환경으로 실행한 model filter:
./unittest_causallm_models \
  '--gtest_filter=*QwenCpuKVGraphTest*:*Lfm2CpuKVGraphTest*' \
  --gtest_output=xml:review-final.actual-graph.xml
```

증거는 `build-kv-android/validation/checks/`의
`review-final.native.validation.json`(85), `review-final.actual-graph.validation.json`(7),
`review-final.validation.json`(합계), `review-final.device.sha256.txt`,
각 `review-final.*.xml/.log`에 있다. manifest에는 실제 test count·exit code,
source/fixture/runtime/executable/device hash, XML·log hash와 ADB 명령을 기록했다.
NDK 준비 기록은 `review-candidate.build.json`, `review-candidate.ndk-build.log`,
`review-final.model-ndk-build.log`에 있다. 이전 단계의 모든 기록은 보존했다.
