# Gemma4 native TL2 / affine WH 작업 인수인계

작성: 2026-10-15, Asia/Seoul. 이 문서는 **작동 완료 보고가 아닌 중간 작업
체크포인트**다. 다음 세션에서는 먼저 이 문서를 읽고, 전체 모델을 반복 실행하기
전에 작은 재현 프로그램으로 decode 다운 원인을 좁혀야 한다.

## 1. 요청과 현재 상태

사용자가 승인한 최종 방향:

- Gemma4 26B-A4B의 expert는 native TL2로 한 번만 저장한다.
- Prefill은 TL2를 DSP scratch의 WH int4로 펼쳐 HMX에서 계산한다.
- Decode는 native TL2 LUT GEMV를 사용한다.
- FC와 untied LM head는 WH int4다. 원본 채널 offset을 없애고 대칭 int4로
  재양자화하는 대신 **WH와 커널에 offset 지원을 추가**하도록 사용자가 선택했다.
- Embedding은 Q4_0, router/norm/scalar는 FP32다.
- 한 개의 model binary를 사용하며 expert sidecar나 head 변환 캐시는 필요 없다.

현재 확인한 결과:

- 변환 완료, 전문가 7,680개와 FC/head 206개를 검증했다.
- Android ARM core/app 및 v81 DSP skel 빌드 완료.
- 전체 모델 로딩과 30개 레이어 prefill 완료.
- LM head의 262144 x 2816 weight가 4096채널씩 64개 WH slice로 NPU에 등록된다.
- **전체 모델 첫 decode는 성공하지 못했다.** Expert pool 교체 중 오류 및
  기기 연결 단절이 반복됐고, 사용자가 휴대폰 다운을 확인하고 재시작했다.
- 전체 모델의 generated text, decode TPS, 원본 대비 PPL은 아직 없다.

마지막 사용자 요청은 다른 세션에서 계속할 수 있는 문서와 현재 작업 커밋이다.
기기 실행은 중단하고 이 체크포인트를 남긴다.

## 2. 브랜치와 upstream

작업 시작 HEAD는 `343224be2`, 브랜치는 `integrate/pr4416-quickdotai`다.
PR [#4428](https://github.com/nntrainer/nntrainer/pull/4428)의 다음 커밋을 fetch하고
그 변경을 로컬 수정과 병합했다. 최초 체크포인트 `194eced79` 작성 후 PR push
준비 과정에서 이 커밋을 `21893b185` 위로 rebase했다. 아래 upstream 커밋들은
최종 체크포인트의 부모 이력에 포함되므로 다음 세션에서 중복 적용하지 않는다.

- `ffcbb7a5a`: untied QS4CX head를 NPU WH slice로 실행.
- `887aedc2c`: QS4CX head의 WH 변환 캐시.
- `21893b185`: [LM head 문서](gemma4-htp-lmhead-npu.md).

Upstream의 QS4CX head는 로딩 때 WH로 변환하고 `.lmhead_wh`를 캐시한다.
이번 pack은 이미 affine WH이므로 `registerFcWh`가 파일의 tile 행과 offset을
slice별로 직접 모은다. 두 경우 모두 prefill 마지막 행의
`lm_head_shared_fp32` / `lm_head_wh_f32`와 decode `LM_HEAD`가 같은 handle을 쓴다.

**최종 graph wire는 upstream과 같은 v3 / 64 parts / 144 words다.** 중간에
사용했던 256-part v3 빌드는 폐기했다. ARM stub, ARM core, DSP skel을 같이
갱신해야 한다. 새 `lm_head_wh_f32` RPC가 추가됐으므로 옛 stub/skel 혼용 금지.

## 3. 로컬 weight와 파일 형식

저장소 루트의 산출물:

```text
gemma4-26b-a4b-ternary-tl2-npu/
  nntr_gemma4_ternary_tl2_qs4cx_wh_affine.bin
  nntr_config.json
  conversion_report.json
  config.json / tokenizer.json / generation_config.json / chat_template.jinja
  validation/s26_validation.json
```

- binary 크기: **6,604,559,480 bytes**.
- SHA256: `c7e978a68c026b9bd54f7e84594701fe63dd3be5b4a62348d2a6afc3a9b9514b`.
- records: 7680 `TL2CX_WH`, 206 `QS4CX_WH_AFFINE`, 1 `Q4_0`, 391 `FP32`.
- 총 8278 record의 runtime size/offset을 실제 Tensor factory와 대조했다.
- 원본 BF16 checkpoint: `gemma4-26b-pure-ternary/ckpt/model/`.
- 검증된 QS2/Q4 중간 source:
  `Applications/CausalLM/res/gemma4/gemma4-26b-pure-ternary-npu/`.
  Source binary SHA256:
  `57c9818bc791f4b18cfdc31fdb792c34604a062fd662d0a46a9ebbe56647edab`.

이 대형 디렉터리/weight, SDK, 빌드 산출물은 커밋하지 않는다. 같은 워크스페이스에
남아 있으며, 다른 머신에서는 별도로 복사하거나 재생성해야 한다.

### Native TL2 record

```text
[ceil(N/128)][ceil(K/24)][5][128] bytes
[N FP32 scales]
[N FP32 signed-weight column sums]
```

3개 trit를 magnitude 4bit와 sign 1bit로 저장한다. 실제 source의
`{-1,0,1}`을 유지하며 palette는 implicit이다. **±7 확장이나 scale /7 보정 없음.**
DSP arena에서는 기존 QS2 capacity `K*N/4`를 예약하고 그 뒤에 scales/sums를 둔다.
파일에서는 compact native payload 바로 뒤에 tail이 있다. 이 차이는 pool
load/rebind 진단 시 특히 주의한다. Miss wire의 native marker는 `0x102`다.

### Affine WH record

```text
[K*N/2 WH int4 bytes]
[N FP32 scales]
[N FP32 plain signed-code column sums]
[N INT32 offsets]
```

`w = scale[n] * (q[k,n] + offset[n])`이며 `q`는 signed int4다.
실제 source BF16 값은 `BF16_round(w)`로 모두 정확하게 복원한다. Source에는
원래 FP32 scale이 없으므로 BF16 rounding interval 안에서 대표 scale을 복원한다.
**커널은 FP32 grid를 사용한다.** BF16 round-trip 오차 0과 FP32 grid 대 BF16의
작은 차이를 혼동하지 않는다. 보고서에 두 지표가 별도로 있다.

Activation을 다시 quantize하지 않는다. 실제 AH activation에 대해
`acc += offset[n] * sum_k(u8[k] - zp)`를 적용한 뒤 기존 dequant를 수행한다.
Registration의 bias sequence를 N 또는 2N float로 확장해 뒤 N개에 offset을
전달한다. DSP weight metadata의 네 번째 배열에 INT32 offset을 보관한다.

`QS4CX_WH_AFFINE_Tensor`의 size와 scale pointer는 legacy QS4CX의 N/2 padding을
상속하면 안 된다. 이번 작업에서 실제 파일 read-back으로 찾아 수정한 부분이다.

## 4. 코드 위치와 이미 수정한 문제

| 위치 | 변경 |
|---|---|
| `tools/convert_tl2_model.cpp` | QS2 source → native TL2, serialized trit/tail 전수 검증 |
| `tools/convert_gemma4_fc_wh.py` | HF BF16 int4 grid 복원, affine WH packing, read-back 검증 |
| `Applications/CausalLM/res/gemma4/convert_gemma4_tl2.py` | 두 변환 단계를 연결하고 임시 Q4 pack을 제거하는 wrapper |
| `tensor_dim.h`, tensor factories, `qs4cx_tensor.h` | `TL2CX_WH`, `QS4CX_WH_AFFINE` 타입과 record layout |
| `htp_compute_ops.cpp` | direct WH source, affine slicing/registration, native expert loading, WH head binding |
| `hmx/hexkl_wh_affine.h`, DMA/MoE kernels | 동일 activation의 offset 보정 |
| `hvx/hvx_gemv_u8tl2.c` | native TL2 → WH prefill 역변환, unit palette 유지 |
| `hmx/hexkl_mm_u8i4_moe.c` | native panel DMA/확장 및 기존 TL2 M=1 경로 연결 |
| `htp_graph_desc.h`, graph/IDL/skel | 64-part WH head와 standalone head RPC |
| Gemma model, QKV/dense/residual layers | 새 dtype 연결 및 load/decode hook 처리 |

기기 테스트 중 수정한 두 문제:

1. Untied LM head에도 `lmhead_engine=htp`를 적용한다. 기존에는 tied 쪽에만
   engine property가 붙어 repack에서 `WH FC/head requires HTP`로 실패했다.
2. WH QKV의 M=1 fast projection이 decode 준비 훅보다 먼저 return하지 않게
   했다. 잘못된 fast path는 첫 decode의 RMSNorm hook을 건너뛰어
   `the HTP attention hook asked for a RoPE table it has`로 실패했다.

이 두 오류는 수정됐지만 이후의 첫-token pool 교체/기기 다운은 해결되지 않았다.

## 5. 기기 결과와 TPS

기기: Galaxy S26 Ultra `R3CY70LV96T`, `SM-S948U`, `SM8850`, Android 16, v81.
사용자 기존 배포 디렉터리는 그대로 두고 아래 테스트 디렉터리만 사용했다.

```text
/data/local/tmp/nntr_gemma4_tl2_check/runtime/
/data/local/tmp/nntr_gemma4_tl2_check/model/
```

Model SHA256는 기기에서도 대조했다. 테스트 config는 greedy, 32-token 요청,
`skip_prefill=true`, cache 16 experts/layer, prefill capacity 512, context 2048다.
질문은 `What is 1+1? Answer briefly.`이며 template의 empty thought channel을
붙였다. **배포 pack의 기본 sampling/config는 이 테스트용 복사본과 다르다.**

### 전체 모델

| 실행 | Prefill 입력행 | 합산 레이어 시간 | Prefill TPS | Decode TPS |
|---|---:|---:|---:|---|
| 12:37 수정본, layer profile | 23 | 69.987 s | 0.329 | 측정 불가 |
| 13:18 재측정, profile=2 + DMA trace | 23 | 70.444 s | 0.326 | 측정 불가 |

이는 **`NNTR_LAYER_PROFILE`의 M>1 행 합계로 계산한 진단 TPS**다. 정상 종료 시
앱이 출력하는 wall-clock TPS나 TTFT가 아니며 모델 로딩 시간도 제외한다.
특히 `NNTR_HTP_PROFILE=2`는 MoE norm/epilogue 일부를 host로 옮기는 진단
모드다. 이 수치를 production throughput으로 제시하지 않는다.

두 번째 실행의 prefill 합계:

| 구간 | 시간 | 합계 비중 |
|---|---:|---:|
| MoE, 30 layers | 68.517 s | 97.26% |
| QKV | 0.717 s | 1.02% |
| Dense FFN | 0.438 s | 0.62% |
| Attention | 0.040 s | 0.06% |
| Residual/norm 및 나머지 | 약 0.732 s | 약 1.04% |

Prefill 지연의 가장 유력한 원인은 `hvx_tl2_expand_wh96`의 스칼라 중첩 루프다.
현재는 trit별 decode와 WH byte read-modify-write를 수행한다. Function 이름과
달리 이 prefill 역변환에는 HVX vector intrinsic이 없다. 많은 expert가 선택되면
M이 작아도 각 선택 expert의 전체 weight를 펼친다. 정확성 우선 구현으로,
기존 QS2의 vectorized `hvx_expand_i2i4`와 같은 성능을 기대할 수 없다.

다만 **expansion만의 시간을 직접 분리 측정한 것은 아니다.** Native DMA
descriptor 수, worker 동기화, uncached memory/flash read도 추가로 조사해야 한다.
FC/head의 affine 지원만으로 70초가 생겼다고 볼 근거는 없다.

첫 decode에서는 `pool_sync`와 KV seed 30개까지 진행했다. 앞선 실행의 오류는
`token driver: S1 did not rebind a pool load`, 후속 RPC는 `AEE_ERPC (0x200)`다.
반복 실행에서도 같은 단계에서 ADB 연결이 사라졌고 사용자도 기기 다운을 확인했다.
**AEE_ERPC는 transport 오류이며 원인을 특정하는 DSP exception code가 아니다.**
Expert rebind, DSPQ 문맥의 DMA, native feed 경로, cache/address-space/동시성은
후보일 뿐이다. 원인 확정 없이 재부팅 사유를 단정하지 않는다.

### 통과한 작은 기기 테스트

- Affine FC: M=1,2,64,65, K=96,N=128에서 equivalent int4 reference와 bit-identical.
- 새 WH LM head RPC: K=96,N=128에서 FC 결과와 bit-identical.
- Native TL2 integer GEMV: K=96,N=128 및 실제 GU K=2816,N=1408,
  DN K=704,N=2816, mode 0/3/512, workers 1/8 모두 오차 0.
- 위 standalone TL2 probe에서 GU 8-worker 약 6–8 µs, DN 약 3–4 µs.
  Weight가 VTCM에 준비된 kernel 시간이며 전체 token TPS로 환산하면 안 된다.
- 기본 session 옵션의 작은/실제 expert M=1 계산 및 arena swap 4회는 통과했다.
  실제 expert는 host round trip 약 45–47 ms였다. 이 실행은 모델의
  `0xcf03e1` 옵션을 설정하지 않았으므로 production decode 측정이 아니다.

**다음 실행 직전에 중단된 항목:** `tools/tl2_swap_device_check.cpp`에
`real fast` 옵션으로 `moe_set_opts(0xcf03e1)`를 적용하도록 추가하고 로컬 binary를
빌드했다. 이 새 binary의 기기 push와 fast 실행은 확인하지 않았다.
기기에 남은 swap test executable은 fast 옵션 추가 전 버전일 수 있으므로 다시 빌드/전송한다.

## 6. 빌드 환경과 재현 명령

```sh
export HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1
export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04
export HEXKL_ROOT=/tmp/nntr-gemma4-tl2-unit/hexkl-sdk
export HEXKL_SDK_VER=6.4.0.2
export HEX_ARCH=v81
export ANDROID_NDK=/home/jwon/Android/Sdk/ndk/26.3.11579264
export PATH=$ANDROID_NDK:$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH
```

`HEXKL_ROOT`는 repo의 `hexkl-1.0-beta.2/hexkl_addon`에서 include와 필요한
`6.4.0.2/armv8_android26`, `hexagon_toolv19_v81` libs를 복사한 것이다.
Meson은 source tree 내부를 가리키는 절대 include 경로를 거부해서 /tmp를 썼다.
다른 세션에서는 이 디렉터리 존재부터 확인한다.

```sh
# IDL 변경 시 ARM stub와 DSP를 둘 다 재생성
bash nntrainer/tensor/htp_backend/generate_stub.sh
bash test/htp/build.sh

# 기존 build directory를 이용하는 incremental ARM build
taskset -c 0-7 ninja -C builddir -j8
meson install -C builddir --no-rebuild --skip-subprojects
taskset -c 0-7 ninja -C builddir_app/htp -j8
```

처음부터 빌드하려면 `Applications/CausalLM/build_android.sh --htp`를 사용한다.
단, 이 스크립트의 native package 단계는 builddir를 wipe할 수 있다. 현재
`builddir/openblas`에 복구한 Android OpenBLAS와 dependency 경로를 먼저 보존한다.
BLAS를 끄면 Android build 설정이 실패한다. xgrammar는 repo gitlink와 동일한
`d2c94ef2d514865f4721cc87c0ec633ca63e0489`의 source/dependencies를 인접
워크스페이스에서 복구했으며 해당 dependency를 수정한 것은 아니다.

변환 재생성(현재 산출물을 덮어쓰지 말고 새 output 사용):

```sh
gcc-13 -O2 -I nntrainer/tensor/htp_backend/hvx -c \
  nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c -o /tmp/tl2_scalar.o
g++ -std=c++17 -O2 -pthread -I nntrainer/tensor \
  -I nntrainer/tensor/htp_backend -I nntrainer/tensor/htp_backend/hvx \
  tools/convert_tl2_model.cpp /tmp/tl2_scalar.o -o /tmp/convert_tl2_model
python3 Applications/CausalLM/res/gemma4/convert_gemma4_tl2.py \
  Applications/CausalLM/res/gemma4/gemma4-26b-pure-ternary-npu \
  --output /tmp/gemma4-tl2-affine-new \
  --converter /tmp/convert_tl2_model \
  --hf-source gemma4-26b-pure-ternary/ckpt/model
```

Python에는 `numpy`, `torch`, `safetensors`가 필요하다. Source/intermediate/output을
동시에 둘 충분한 공간도 필요하다. 기존 output은 덮어쓰지 않고 거부한다.

### 小さな device check の build

```sh
O=/tmp/nntr-gemma4-tl2-unit
CC=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang
CXX=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++
INC="-I$HEXAGON_SDK_ROOT/incs -I$HEXAGON_SDK_ROOT/incs/stddef -I$HEXAGON_SDK_ROOT/ipc/fastrpc/incs -Inntrainer/tensor/htp_backend/generated -Inntrainer/tensor/htp_backend/hvx -Inntrainer/tensor/htp_backend"
$CC -O2 -fPIC $INC -c nntrainer/tensor/htp_backend/generated/nntr_hvx_stub.c -o "$O/s26_rpc_stub.o"
$CC -O2 $INC -c nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c -o "$O/s26_tl2_scalar.o"
$CXX -DENABLE_HEXKL -std=c++17 -O2 -fPIE -pie -static-libstdc++ $INC \
  tools/tl2_swap_device_check.cpp "$O/s26_tl2_scalar.o" "$O/s26_rpc_stub.o" \
  -L$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64 \
  -lcdsprpc -llog -ldl -o "$O/tl2_swap_device_check"
```

동일 include/link 방식으로 `tools/tl2_device_check.cpp`와
`tools/wh_affine_device_check.cpp`도 빌드한다. Native TL2 checker만 scalar object가
필요하다. /tmp에 host용 fake `nntr_hvx.h`도 있으므로 include 검색 충돌에 주의한다.
새 tool들은 SDK의 generated header를 사용해야 한다.

## 7. 다음 세션의 권장 순서

1. `adb devices -l`과 기기 상태 확인. 전체 모델로 다운을 반복 재현하지 않는다.
2. 올바른 ARM/DSP pair를 테스트 디렉터리에 다시 배포한다.
3. 방금 빌드한 swap tool로 작은 기본 옵션 case, `real`, 마지막에 `real fast`를
   비교한다. 마지막 case는 아직 검증되지 않았다. 기기 상태를 보면서 순차 실행한다.
4. 기본 FastRPC swap은 통과했으므로 모델 옵션 적용 후 whole-MoE M=1 경로,
   DSPQ token 문맥, 첫 miss의 slot/arena/offset/native flag를 분리한다.
   단독 GEMV 통과가 이 경로의 정상 동작을 보장하지 않는다.
5. 그 다음에만 전체 모델을 다시 시도한다. 연결이 살아 있을 때부터 DSP/driver
   로그를 스트리밍해야 한다. 이전의 사후 logcat은 재부팅 뒤 boot log여서
   원인 자료가 없었다. pstore는 permission denied, SYSTEM_LAST_KMSG는 없었다.
6. Decode가 안정되면 계측을 끄고 TTFT/steady decode TPS를 측정한다.
7. `skip_prefill=false`에서도 head의 마지막 행 실행을 전체 모델로 검증한다.
8. Prefill scalar TL2→WH를 HVX로 구현하고 native DMA descriptor/worker scheduling을
   조사한다. Layout/수치 결과는 현재 host reference와 같아야 한다.

진단 실행 예(현재 **기기 다운 가능성이 있는 미해결 재현**, 바로 실행하지 말 것):

```sh
# runtime에서 실행. LD_LIBRARY_PATH에 vendor/system 경로를 강제로 넣지 않는다.
LD_LIBRARY_PATH=$PWD ADSP_LIBRARY_PATH=$PWD NNTR_NUM_THREADS=8 \
NNTR_LAYER_PROFILE=1 NNTR_HTP_PROFILE=2 NNTR_HTP_DMA_TRACE=1 \
timeout -s KILL 240 ./nntr_causallm /data/local/tmp/nntr_gemma4_tl2_check/model
```

`LD_LIBRARY_PATH=/vendor/lib64:/system/lib64`를 붙였을 때 Android library ABI
충돌이 있었다. 정상 앱에는 runtime 경로만 지정하고 기본 linker 검색을 사용한다.
또한 현재 브랜치의 `NNTR_HTP_PREFILL_ROWS`는 주로 conv path용이며 MoE를
자동으로 1행씩 쪼개는 해결책이 아니다. E2E 기본값도 `htp_fixed_opts.h`에
고정되어 있으므로 오래된 env knob가 먹는다고 가정하지 않는다.

## 8. 검증과 증거 보관 위치

통과한 host checks: affine FC/dense (M=1,2,64,65, workers 0/3), 실제 Gemma
shape의 native prefill tails, converter exact/rejection tests, graph v3 WH head,
10000-token driver identity와 pool/failure paths, weight swap.
Host stand-in은 scheduling/layout 검사이며 실제 DSP 검증과 구분한다.

전체 host suite는 중간 단계에서 수행했고, 최신 LM head 통합 후에는 해당
graph/token/swap와 converter checks를 별도로 다시 통과했다. 최신 tree의 전체
suite/모든 CI platform을 통과했다고 주장하지 않는다. Host HVX emulation의
`_Float16` 때문에 이 환경에서는 `gcc-13`을 사용했다.

```sh
CC=gcc-13 bash test/htp/host/run_host_checks.sh
python3 test/htp/host/wh_affine_conversion_check.py
```

핵심 local logs는 `/tmp/nntr-gemma4-tl2-unit/` 아래에 있다:

- `s26_tps_baseline.log`: 13:18 전체 진단 실행, 실패 직전까지의 layer/DMA trace.
- `s26_smoke_pr4428_hook.log`: 앞선 전체 실행과 S1 rebind 오류.
- `s26_affine_probe.log`, `s26_tl2_probe.log`: 통과한 기기 수치 검증.
- `s26_tl2_swap_small.log`, `s26_tl2_swap_real.log`: 기본 옵션 교체 테스트.
- `pr4428_{arm,app,dsp}_build.log`, `s26_qkv_hook_build.log`: 빌드 결과.
- `pr4428_{graph,token,swap}_check.log`, `pr4428_conversion_check.log`: 최신 host checks.

일부 로그/설정은 ignored model directory의 `validation/`에도 복사했다.
이 문서 옆의 [요약 JSON](validation/gemma4-tl2-s26-20261015.json)은 커밋에 포함한다.
원시 Android logcat 전체에는 unrelated device 정보가 있어 커밋하지 않는다.
