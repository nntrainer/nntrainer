# 다음 KV backend를 위한 환경 조사

2026-10-29 기준 후속 backend 선정 자료다. 모델/GPU KV 구현, 새 SDK 다운로드,
GPU memory allocation·공유 검증은 수행하지 않았다. 성능 측정 담당자가 기기 window를
반환한 뒤 `bacd420f`의 전용 `capability/` 경로에서 조회 probe를 **정확히 1회** 실행했다.
조회 종료 후 window를 즉시 반환했으며 추가 기기 실행은 하지 않았다.

## 호스트에서 확인한 사실

| 항목 | 확인값 |
|---|---|
| Android NDK | `/opt/android-ndk-r28c`, `28.2.13676358` |
| probe target | arm64-v8a, Android API 29 |
| QNN SDK 환경변수 | `QNN_SDK_ROOT=/home/jayden/qairt/2.47.0.260601` |
| 실제 QNN headers | `include/QNN/QnnCommon.h`, `QnnInterface.h`, `System/QnnSystemInterface.h` 존재 |
| header API version | QNN `2.36.0`, System `1.11.0` |
| Android QNN runtime | Htp/System/Prepare/NetRunExtensions library 존재 |
| HTP Stub/Skel | SDK 안에 V75, V79, V81의 complete pair 존재 |
| Hexagon/HexKL | `HEXAGON_SDK_ROOT` unset; 조사한 `/opt` 및 QAIRT 경로에서 `libsdkl.so` 미확인 |

QNN SDK는 `/opt`에 있지 않지만 환경변수가 가리키는 사용자 경로에 실제로 설치되어
있다. SDK package version과 header API version을 혼동하지 않는다. 파일 존재 확인은
기기 DSP runtime/driver, 선택할 HTP architecture, 실제 QNN graph 실행의 호환성 검증을
의미하지 않는다.

SDK metadata에는 `version: 2.47.0`, `build_id: 260601114230`,
`qnn_backend_api_version: 2.18.0`이 적혀 있고 HTP header의 별도 API version은
`5.47.0`이다. 서로 다른 version 필드를 같은 값으로 취급하지 않는다. 실제 runtime의
provider API version과 기기 호환성은 아직 조회하지 않았다. Android Htp library의
embedded build string은 `v2.47.0.260601114230`으로 SDK package와 일치한다.

상세 파일 경로·크기·SHA256은 ignored 생성물
`build-kv-android/validation/backend-capability/host-environment.json`에 보존한다.
현재 CPU 빌드의 `enable-opencl=false`, `enable-npu=false`, `enable-htp=false` 설정은
변경하지 않았다.

## 저장소의 기존 연결점

OpenCL loader는 일반 `libOpenCL.so` 로딩 후 Android에서
`/vendor/lib64/libOpenCL.so`, `/system/vendor/lib64/libOpenCL.so` 및 32-bit 대응 경로를
차례로 시도한다. 이 조사용 probe도 같은 목록을 사용한다. 기기 전체의
`LD_LIBRARY_PATH`를 vendor 디렉터리로 변경할 필요는 없다.
[loader](../nntrainer/opencl/opencl_loader.cpp)는 이 접근을 이미 구현하고 있다.

[device info](../nntrainer/opencl/opencl_device_info.cpp)는 SVM capability와 extension을
조회하며, [SVM allocator](../nntrainer/cl_svm_allocator.cpp)는 SVM allocation 및 host
fallback을 가지고 있다. [queue manager](../nntrainer/opencl/opencl_command_queue_manager.cpp)는
coarse/fine-grain capability에 따른 동기화를 구분한다. GPU attention에는 SVM pointer를
직접 받는 분기와 non-SVM upload 분기가 모두 있다. 따라서 실제 allocation이 SVM인지,
CPU producer와 GPU consumer가 같은 owner와 layout을 쓰는지는 후속 단계에서 별도로
입증해야 한다. driver의 capability bit만으로 manager 공간 공유 완료를 선언하지 않는다.

QNN은 [RPC manager](../nntrainer/qnn/jni/qnn_rpc_manager.cpp)의 fd 기반
`QNN_MEM_TYPE_ION` 등록과 `memRegister` 연결이 있다. 공개 QNN CausalLM 경로는
별도의 prefill/decode KV layout과 append/copy 처리를 사용한다. 현재 CPU Qwen checkpoint가
있다는 사실만으로 QNN graph/model artifact까지 준비되었다고 보지 않는다.
[기존 분석의 QNN 범위](repository-analysis.md#5-qnnnpu-경로)를 이어서 검토한다.
QNN graph backend와 HexKL HTP ComputeOps backend는 별도 후보로 다룬다.

## 준비한 OpenCL 조회 probe

ignored `build-kv-android/validation/backend-capability/`에
`opencl_capability_probe.c`와 Android executable을 준비했다. `clang-format-14`와
NDK cross compile을 통과했으며 dependency는 `libc.so`, `libdl.so`뿐이다.
NNTrainer/GPU model library와 tokenizer를 배포하지 않는다.

조회 범위는 platform/device 이름·vendor·version, driver/OpenCL C version,
SVM capability mask와 bit별 값, extension 목록이다. SVM entrypoint의 존재도
`dlsym`으로만 확인한다. context, command queue, OpenCL buffer, SVM region, kernel을
만들거나 실행하지 않는다. API query error와 loader별 오류는 JSON에 그대로 남긴다.

```bash
/opt/android-ndk-r28c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang \
  -O2 -Wall -Wextra -Werror -DCL_TARGET_OPENCL_VERSION=200 \
  -Inntrainer/opencl -fPIE -pie \
  build-kv-android/validation/backend-capability/opencl_capability_probe.c \
  -ldl -o build-kv-android/validation/backend-capability/opencl_capability_probe
```

현재 executable SHA256은
`45e101e8ce99f246b23b52f2940fa30933fd6e3744756f1ee47bc3c0e7434976`이다.
기기 바이너리 SHA256이 로컬과 일치함을 확인한 뒤 아래 명령을 1회 실행했고
exit `0`으로 JSON 결과를 회수했다. vendor `LD_LIBRARY_PATH` 설정 없이 첫 시도의
`libOpenCL.so` soname 로드에 성공했다. 이번 결과는 shell-launched probe의 접근성이다.
Android APK의 linker namespace는 별도로 검증해야 한다.

```text
cd /data/local/tmp/nntr_kv_cpu_bacd420f/capability &&
chmod 755 opencl_capability_probe && ./opencl_capability_probe
```

## bacd420f 실제 조회 결과

| 항목 | 조회값 |
|---|---|
| platform | `QUALCOMM Snapdragon(TM)`, vendor `QUALCOMM` |
| platform/device version | OpenCL `3.0` |
| device | `QUALCOMM Adreno(TM) 750` |
| OpenCL C | `3.0 Adreno(TM) 750` |
| driver | `OpenCL 3.0 QUALCOMM build: commit unknown Compiler E031.45.02.19` |
| SVM capability query | success, mask `11` (`0xB`) |
| coarse-grain buffer | 지원 광고 |
| fine-grain buffer | 지원 광고 |
| fine-grain system | 미지원 광고 |
| SVM atomics | 지원 광고 |
| SVM entrypoint | alloc/free/map/unmap/kernel-arg-pointer 5개 모두 존재 |

관련 extension 광고는 `cl_khr_fp16`, `cl_khr_image2d_from_buffer`,
`cl_qcom_ext_host_ptr`, `cl_qcom_ext_host_ptr_iocoherent`,
`cl_qcom_dmabuf_host_ptr`, `cl_qcom_android_ahardwarebuffer_host_ptr`,
`cl_qcom_recordable_queues`를 포함한다. 기존 분석에서 조사 후보로 적었던
`cl_qcom_ion_host_ptr`, `cl_arm_import_memory`, `cl_khr_external_memory`는 이 목록에
없었다. extension 이름의 존재와 실제 import 성공은 구분한다.

전체 extension 문자열, API query error, loader 시도 및 driver 원문은
`build-kv-android/validation/backend-capability/opencl.stdout.json`에 보존했다.
`opencl.validation.json`에는 명령·exit code·실행 횟수 1·source/binary/device hash가,
`opencl.stderr.txt`에는 stderr가 있다. context/queue/allocation/kernel 호출은 없었다.

## 결과에 근거한 설계 제안

이번 기기에서 loader와 SVM query가 모두 성공했으므로 **다음 작은 adapter 후보는
OpenCL/SVM을 우선 제안**한다. 기존 token-major KV 및 SVM attention 경로와 연결하는
범위부터 잡을 수 있다. 다음 단계는 allocation별 실제 memory kind와 owner lifetime,
producer/consumer 동기화, fallback의 정직한 보고를 검증하는 것이다. OHWI image mirror
제거와 Qwen GPU 전체 실행 성능은 각각 별도 범위다. 이번 조회는 그 구현을 시작하거나
공유 완료를 입증한 결과가 아니다.

구체적으로 현재 [SVM region 생성](../nntrainer/opencl/opencl_context_manager.cpp)은
`CL_MEM_READ_WRITE`만 넘긴다. 반면 queue의 `coarseGrainSVMOnly()`는 device의
fine-grain capability가 있으면 false를 돌려준다. device의 지원 bit와 실제 allocation
flags를 같은 것으로 취급하지 않도록 후속 adapter에서 검토해야 한다. Khronos 문서상
fine-grain flag를 주지 않은 allocation은 coarse-grain일 수 있고, allocation 자체가
host/device memory consistency를 보장하지 않는다. [Khronos clSVMAlloc](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clSVMAlloc.html)

QNN/HTP도 SDK 설치가 장애물이 되지는 않는 상태다. 선택하려면 Qwen3-0.6B에 적합한
실제 graph artifact와 KV input/output layout·encoding·append 경계를 먼저 확정해야 한다.
ION fd 등록의 존재만으로 graph output이 persistent KV slot에 직접 쓰인다고 판단하지
않는다. 다음 구현 backend의 확정과 메모리 공유 검증 착수는 사용자 결정에 따른다.
