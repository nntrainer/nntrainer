# Gemma4 native TL2 / affine WH 작업 인수인계

**Prefill 최적화 후속 검증은 [9절](#9-2048토큰-prefill-최적화-후속-검증)을
참조한다.** 1~8절은 최적화 전 체크포인트이며, 전체 decode 문제는 별도 미해결이다.

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

## 9. 2048토큰 prefill 최적화 후속 검증

PR #4428의 `a97d383a4ef1fe94bf3b6d53f47e50b9c6624530`을 기준으로
S26 Ultra에서 측정했다. 기존 작업 디렉터리의 미커밋 변경을 보존하기 위해
`codex/pr4428-tl2-prefill-20261015` 브랜치와 별도 worktree를 사용했다.
모델 파일과 ARM runtime은 체크포인트 그대로이며 DSP 커널을 변경했다.
실제 모델 dtype은 expert `TL2CX_WH`, FC/head `QS4CX_WH_AFFINE`이다.

### 측정 조건

- 입력 **2048토큰**, cache **16 experts/layer**, `NNTR_HTP_PAGE_CACHE=0`.
  이 옵션은 expert miss 읽기에 `O_DIRECT`를 사용한다.
- `init_seq_len=2048`, `max_seq_len=2304`, `skip_prefill=false`,
  `num_to_generate=1`, `NNTR_NUM_THREADS=8`, `NNTR_MOE_CACHE_GROW=0`.
- [고정 UTF-8 입력](validation/gemma4-tl2-prefill-p2048.txt)은 기존 Ardley
  벤치마크 문장을 2048토큰으로 늘렸다. Tokenizer와 실제 API 양쪽에서
  2048토큰을 확인했다. `use_chat_template=false`로 중복 template을 막는다.
- `tools/gemma4_prefill_device_check.cpp`가 공개 API로 prefill과 마지막 행
  affine LM head까지 실행한다. 주기적인 취소 요청은 generation loop에서
  처리되어 decode forward를 시작하지 않는다. 모든 측정의
  `decode_tokens=0`을 확인한다. `num_to_generate=1`만으로는 decode가 생기므로
  기존 CLI를 그대로 사용하면 이 측정과 다르다.
- TPS는 API의 `prefill_tokens / prefill_duration`이다. 모델 로딩 시간은
  제외한다. 취소된 streaming의 `ttft_ms=0`은 TTFT 측정값으로 사용하지 않는다.
- 최종 반복 측정은 `NNTR_LAYER_PROFILE`, `NNTR_HTP_PROFILE`을 설정하지 않는다.
  병목 분석은 profile 1로 수행했다. Profile 2는 host norm 경로와 복사량도
  바꾸므로 최종 속도 비교에서 제외한다.

### 원인과 수정

1. Native TL2→WH 변환은 가중치마다 scalar lookup/bit packing을 수행했다.
   HVX lookup, sign 처리, WH transpose로 바꾸고 96행 블록의 고정 인덱스를
   전개해 반복적인 나눗셈과 임시 vector 저장을 없앴다. Magnitude 1/7과
   부분 K/N 타일의 값은 유지한다.
2. In-place 변환 전 packed 패널을 복사하는 `memcpy`도 VTCM의 scalar 읽기를
   발생시켰다. 정렬된 HVX vector 복사로 변경했다.
3. Native staging은 첫 chunk에서 전체 expert를 기다렸고, background 변환도
   꺼져 있었다. Gate/up chunk 순서로 DMA를 발행하고 공유되는 128열 패널은
   한 번만 전송·변환한다. FIFO background job으로 변환과 HMX를 겹친다.
4. Affine activation 합계와 accumulator offset 보정이 VTCM을 scalar로
   읽고 썼다. HVX row sum과 정수 multiply-accumulate로 바꿨다. 등록 시
   검증되는 `[-15,15]` offset을 signed halfword로 곱하되 합계는 int32로
   유지한다. 보정은 FC/FFN 후처리 worker의 담당 행/타일에서 수행한다.
   비동기 job이 끝날 때까지 합계를 자체 보관해 다음 블록과 겹치지 않는다.

TL2 변환만 벡터화한 상태에서도 2048토큰은 **159.669초 / 12.827 TPS**였다.
이때 QKV 56.692초, output projection을 포함한 post-attention norm
61.699초, dense FFN 35.348초가 소요되어 affine 경로가 다음 병목임을 확인했다.
Affine 벡터화 후 6.969초, VTCM 복사 개선 후 3.965초로 줄었다.
이는 같은 고정 입력을 사용한 중간 단계 측정이며 최종 결과와 구분한다.

### 최종 결과

| 상태 | Prefill 시간 | TPS |
|---|---:|---:|
| PR head, 수정 전 (profile 1) | 310.313초 | 6.60 |
| 최종 빌드, 계측 off 5회 중앙값 | 3.191초 | 641.81 |
| 최종 빌드, 계측 off 5회 평균 TPS | — | 630.00 |

최종 5회는 3507 / 3187 / 3189 / 3203 / 3191ms였다. 첫 회
583.97 TPS를 제외하지 않았으며, 나머지 4회는 639.40~642.61 TPS였다.
중앙값 기준으로 최신 PR의 회귀 상태보다 약 97.25배 빨라졌다.
과거 600+ TPS 기록과는 weight pack이 다르므로 과거 모든 pack보다
빠르다고 해석하지 않는다. 같은 입력의 첫 prefill 출력 ID는 수정 전후
모두 `818`이며, 전체 decode를 실행한 결과는 아니다.

최종 반복 측정, artifact hash와 검증 범위는
[prefill 결과 JSON](validation/gemma4-tl2-prefill-s26-20261015.json)에 기록한다.
원시 로그는 `/tmp/pr4428-tl2-prefill-20261015/`, 격리된 device runtime은
`/data/local/tmp/nntr_tl2_prefill_20261015/`에 보관한다.

### 재현

검사 도구는 NDK clang++의 C++17로 다음 include/library를 사용해 빌드한다.
`--allow-shlib-undefined`는 Android 실행 시 함께 로드되는 runtime 의존성용이다.

```sh
"$NDK/aarch64-linux-android28-clang++" -std=c++17 -O2 -Wall -Wextra -Werror \
  -IApplications/CausalLM/api \
  -IApplications/CausalLM/Android/QuickDotAI/src/main/cpp/include \
  tools/gemma4_prefill_device_check.cpp -L"$RUNTIME" \
  -lquick_dot_ai_api -Wl,--allow-shlib-undefined -o gemma4_prefill_device_check
```

Device의 `$BENCH/gemma4-26b-a4b-htp-w4a32/`에 위 조건의 설정과 모델을
배치한다. `$RUNTIME`은 일치하는 ARM 라이브러리와 빌드한 v81 skel이 있는
디렉터리다. 고정 입력 파일도 device에 복사한 뒤 실행한다.

```sh
LD_LIBRARY_PATH="$RUNTIME" ADSP_LIBRARY_PATH="$RUNTIME" \
NNTR_NUM_THREADS=8 NNTR_HTP_PAGE_CACHE=0 NNTR_MOE_CACHE_GROW=0 \
./gemma4_prefill_device_check "$BENCH" "$RUNTIME" gemma4-tl2-prefill-p2048.txt
```

Native TL2 prefill은 실제 DSP에서 독립 int4 reference와 bit-identical이다:
작은/실제 Gemma shape, magnitude 1/7, expert 1/3개, M=2/23/65/129,
각 3회 반복, gather/scatter와 공유 패널 경계를 포함한다. Affine FC는
M=1/2/64/65/128, affine FFN은 M=2/64/65/129, 작은/실제 shape에서 같은
검증을 통과했다. V79/V81 simulator에서도 TL2 layout 192 case와 affine
sum/correction 210 case가 정확히 일치한다.

`CC=gcc bash test/htp/host/run_host_checks.sh` 전체 실행도 exit 0으로
완료했다. 선택적인 libnative 및 AArch64/qemu 검사는 환경 도구 부재로
skip되었고, 실제 HVX 경로는 위 simulator와 device 검증으로 별도 확인했다.
ASan/UBSan, v79/v81 DSP 빌드, clang-format 14 변경 줄 검사도 통과했다.
Expert swap의 `real fast` 옵션도 실제 기기에서 4회 교체 결과가 일치했다.

Prefill 첫 출력 ID의 일치는 전체 생성 품질 검증이 아니다. 이 절의 속도
검증은 decode를 제외한다. 이후의 전체 생성 검증은 아래 10절에 기록한다.
원본 checkpoint 대비 PPL과 모든 CI platform은 이번 검증 범위 밖이다.

## 10. Affine FFN decode 수정과 전체 생성 확인

`62fdb7145760e37b4f58e54c38b10d77af26993e`에서 전체 생성을 실행하면 첫
decode의 `nntr_dspq` thread가 `hexkl_micro_hmx_acc_clear_int32`에서 DSP
exception을 일으켰다. Host에는 이후 `S1 did not rebind a pool load`가
보고됐다. Affine weight를 M=1 GEMV에서 제외하는 조건 때문에 dense FFN이
FastRPC caller의 HMX 경로로 들어간 것이 문제였다.

Gate/up과 requantized down activation의 `sum(u-zp)`를 expert별로 보관하고,
GEMV의 int32 accumulator에 `offset * sum(u-zp)`를 더한 뒤 기존 dequant를
실행하도록 수정했다. Affine weight도 기존 M=1~4 GEMV 경로를 사용한다.
큰 M의 affine tail 경로 제한과 prefill 산술은 유지한다.

### 검증

- Host: M=1/2/3/4, worker 0/3, VTCM feed off/on에서 독립 int4 HMX
  reference와 bit-identical. GEMV 경로 선택도 assert해 HMX fallback을 잡는다.
- S26 Ultra: 작은 shape `K=96,I=192,N=128`과 실제 Gemma shape
  `K=2816,I=704,N=2816`, M=1/2/3/4/64/65/129, flags
  `0x0`(HMX), `0x21`(GEMV feed off), `0xcf03e1`(운영 옵션)을 각 3회 실행해
  모든 affine FFN 출력이 독립 int4 reference와 bit-identical이다.
  기존 affine FC와 마지막 행 LM head 검사도 통과했다.
- 전체 HTP host suite가 exit 0으로 완료됐다. 선택 항목인 libnative 및
  AArch64/qemu 검사는 도구 부재로 skip됐다. Affine ASan/UBSan, v79/v81
  DSP 빌드, clang-format 14 변경 줄 검사, Doxygen 검사도 통과했다.

2048토큰 고정 입력, C=16, page cache off, `num_to_generate=128`로 CLI 전체
생성을 실행했다. 수정 직후 profile 1 실행과 최종 빌드의 계측 off 실행 모두
prefill ID `818`과 decode 101개를 포함한 **102개 출력 ID가 완전히 일치**하고
EOS에서 끝났다. 최종 실행의 로그와 artifact hash, ID 전체는
[생성 로그](validation/gemma4-tl2-affine-decode-s26-20261015.log)와
[검증 JSON](validation/gemma4-tl2-affine-decode-s26-20261015.json)에 보관한다.
CLI의 decode 출력은 공백으로 시작하며, prefill 출력 ID는 별도 ID 파일에
포함된다. 생성 문장은 그대로 보관했다. 저장소의 whitespace 검사를 위해
e2e 측정 줄의 끝 공백 한 글자만 제거했고, 원본과 보관본 hash는 JSON에
각각 기록했다.

| 전체 생성 실행 | Prefill | Decode |
|---|---:|---:|
| 수정 직후, profile 1 | 3693ms / 554.563 TPS | 101토큰 / 11.7101 TPS |
| 최종 빌드, 계측 off | 3296ms / 621.359 TPS | 101토큰 / 11.6939 TPS |

생성된 3문장은 원문의 어업·농업, 역사, lifeboat와 공동체를 일관되게
요약한다. 이는 한 prompt의 정상 생성 확인이며 일반적인 품질이나 PPL
검증을 의미하지 않는다. **700 TPS 목표는 아직 달성하지 못했다.** 9절의
641.81 TPS는 이전 커밋의 prefill-only 5회 중앙값이며 위 단일 전체 생성
실행과 합쳐서 집계하지 않는다.

LM head는 prefill의 마지막 행 하나에 대해 NPU GEMV를 수행한다.
262144개 출력을 4096열씩 64개 WH slice로 처리하며, 8개 lane과 lane별
double-buffer DMA를 사용한다. 기존 641.81 TPS 측정에도 포함된 경로다.
최종 decode의 LM head 평균은 6.354ms/token이며 prefill 전체의 주 병목은
별도로 분석해야 한다.

라우터의 cached scratch 실험은 profile 1 비교에서 3293ms → 3299ms로,
전체 prefill 개선을 확인하지 못했다. 이 실험은 후속 검토용 패치로 보관하고
이번 수정에는 포함하지 않았다.
