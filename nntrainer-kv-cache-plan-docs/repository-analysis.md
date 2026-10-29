# KV cache manager: 현재 저장소 분석

- 분석일: 2026-10-28
- upstream: `https://github.com/nntrainer/nntrainer.git`
- 분석 기준: `main`, `141871a58ff439e3caeac40c1632ff8ac3543d20`
- 기준 커밋: 2026-10-19, `[test] Regenerate the Qwen2, Qwen3 and LFM2 fixtures with up, gate order`
- 환경: Windows에서 소스 정적 분석. Linux 빌드, GPU/NPU 실행, 성능 측정은 수행하지 않았다.
- 관련 문서: [구현 계획](README.md), [Linux 개발·검증 절차](linux-development.md)

## 1. 출발점 정정

최신 main에는 KV cache manager가 이미 있다. 따라서 이번 작업은 기존 구현의
공용화와 직접 접근 계약의 완성으로 설계해야 한다.

| 요구 | 현재 main에서 확인한 것 | 아직 필요한 것 |
|---|---|---|
| 1. CPU/GPU/NPU가 manager 공간을 직접 사용 | CausalLM manager의 Tensor alias, OpenCL SVM 및 CUDA allocation 선택, QNN RPC buffer 등록 | 공용 소유권·layout·동기화 계약, producer 직접 쓰기, QNN append 제거, GPU mirror 정리 |
| 2. 다양한 KV 양자화 | MHA 내부 INT8 + FP16 scale 경로, QNN의 고정소수점 KV 표현 | manager가 payload와 quant metadata를 함께 관리; backend별 지원 검증 |
| 3. 양자화 상태 저장·복원 | CausalLM versioned KV 파일, 별도 QNN byte dump manager | 통합 형식, scale/zero-point/layout/model identity, 호환성 및 완전한 상태 검증 |
| 4. batch/session 공유 | batch 차원, Gemma4의 **모델 내 layer 간** KV alias, ring helper, QNN speculative helper | session allocator, paged attention, prefix index, refcount/COW, scheduler |

batch 차원이 있거나 같은 모델의 여러 layer가 K/V를 공유하는 것은 session 간
prefix 공유를 구현했다는 뜻이 아니다. QNN helper의 tree-tail API 역시 공용
multi-session manager의 존재를 의미하지 않는다.

## 2. 소스 근거 지도

아래 링크는 모두 분석 기준 SHA에 고정했다. 구현 착수 시 main이 전진했다면 symbol과
호출 경로를 다시 확인한다. 코드 주석에 기록된 속도 수치는 이번 분석의 측정 결과로 쓰지 않았다.

| ID | 위치 / 핵심 symbol | 관찰 |
|---|---|---|
| E01 | [CausalLM/kv_cache_manager.h](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/kv_cache_manager.h#L45), [allocate](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/kv_cache_manager.cpp#L51) | layer별 Tensor, 위치, width/capacity/alias, save/load를 소유하는 application 모듈 |
| E02 | [allocateAndBindKVCache](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/models/causal_lm.cpp#L404) | manager의 MemoryData와 offset을 graph Tensor에 setData |
| E03 | [incrementalInference](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/models/causal_lm.cpp#L549) | raw pointer 재포장으로 SVM 속성이 소실되지 않도록 실제 cache Tensor 전달; graph input 이름으로 layer 식별 |
| E04 | [createKVCachePlaceholders / wireAttentionKVCache](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/models/transformer.cpp#L857) | input layer 기반 external FP16/UINT16 cache; INT8은 별도의 3-input 내부 cache |
| E05 | [MHACoreLayer::forwarding](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/layers/mha_core.cpp#L1293) | external cache는 input 3/4; 3/4-input legacy 경로와 5/6-input external 경로 공존 |
| E06 | [MHA step view와 쓰기](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/layers/mha_core.cpp#L1973), [K/V append](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/layers/mha_core.cpp#L3014) | cache slice에 K RoPE 출력; V copy/변환, no-RoPE K copy가 남음 |
| E07 | [MHA GPU mirror / sync_kv_slab](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/layers/mha_core.cpp#L3300) | OHWI/image 저장소와 slab 사이 scatter/gather, authoritative plane 상태가 layer 안에 존재 |
| E08 | [OpenCL attention](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/tensor/cl_operations/attention_kernels.cpp#L2764) | SVM 직접 kernel 인자 또는 non-SVM K/V upload 경로 |
| E09 | [ClSVMAllocator](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/cl_svm_allocator.cpp#L60), [MemoryPool::getMemory](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/tensor/memory_pool.cpp#L262) | SVM 실패 시 host fallback; MemoryData 속성은 allocator predicate에서 설정 |
| E10 | [MemoryData](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/tensor/memory_data.h#L53), [MemoryPool::deallocate](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/tensor/memory_pool.cpp#L290) | MemoryData와 실제 allocation 수명은 구분해야 함; pool이 실제 buffer를 해제 |
| E11 | [Tensor::setData](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/tensor/tensor.cpp#L1290), [getSharedDataTensor](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/tensor/tensor_base.cpp#L229), [fillPlaceholder](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/tensor/tensor_pool.cpp#L495) | alias/offset와 dependent 갱신 재사용 가능; 임의 stride 표현으로 일반화할 수는 없음 |
| E12 | [MemAllocator](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/mem_allocator.h#L27), [Engine](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/engine.h#L110) | backend별 allocator와 plugin context가 이미 있음 |
| E13 | [TensorBacking / ViewSpec](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/tensor/cl_operations/cl_tensor_view.h#L80) | 하나의 cl_mem을 buffer/image view로 사용하는 기반이 있음 |
| E14 | [QnnKvCacheManager](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/qnn/qnn_kv_cache_manager.h#L24) | 외부 uint8_t 포인터를 보관하는 별도 helper; allocation 자체를 소유하지 않음 |
| E15 | [QNN append / process_key / process_value](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/qnn/generate_qnn_utils.cpp#L205) | generation→prefill copy, window shift memmove, step output memcpy |
| E16 | [Gemma4 QNN mapping](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/qnn/gemma4-e2b-qnn/gemma4_e2b_qnn.cpp#L815), [decode append](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/qnn/gemma4-e2b-qnn/gemma4_e2b_qnn.cpp#L1459) | 공개 QNN 모델은 자체 kvs/prefill_kvs 배열과 공용 copy helper를 사용 |
| E17 | [QNNGraph::populateTensor](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/qnn/jni/QNNGraph.cpp#L340), [registerQnnTensor](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/qnn/jni/qnn_rpc_manager.cpp#L140) | RPC fd를 QNN_MEM_TYPE_ION으로 등록하고 graphExecute에 전달 |
| E18 | [Android application allocator](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/qnn/android_memory_allocator.cpp#L36), [core RpcMem](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/qnn/jni/rpc_mem.h#L13) | application과 core의 allocation 진입점이 별개; 수명/해제 책임 통합 필요 |
| E19 | [kv_ring.h](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/kv_ring.h#L630), [layer source/cap hook](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/models/transformer.h#L350) | ring capacity, absolute position, 공유 source 규칙을 보존해야 함 |
| E20 | [CausalLM 파일 포맷](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/kv_cache_manager.cpp#L539), [QNN 파일 포맷](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/qnn/qnn_kv_cache_manager.cpp#L380) | 서로 다른 v1 형식과 검증 규칙 |
| E21 | [core MultiHeadAttentionLayer](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/layers/multi_head_attention_layer.cpp#L545), [LLaMA custom layer](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/LLaMA/jni/custom_multi_head_attention_layer.cpp#L594) | CausalLM 외에도 내부 cache 구현이 있음 |
| E22 | [manager tests](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/test/unittest/layers/unittest_kv_cache_manager.cpp#L30), [CausalLM test targets](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/Applications/CausalLM/meson.build#L434) | manager·ring·model 회귀 테스트 기반 |
| E23 | [HTP context](https://github.com/nntrainer/nntrainer/blob/141871a58ff439e3caeac40c1632ff8ac3543d20/nntrainer/htp_context.h#L15) | HexKL ComputeOps 경로도 존재; QNN graph backend와 구분 필요 |

### 검색 범위

`Applications/CausalLM`, `Applications/LLaMA`, `nntrainer`, `api/ccapi`,
`test/unittest`, Meson/Android build 및 CI 구성을 조사했다.
`subprojects/`는 수정하지 않았다. 아래 사실은 전체 소스의 재검색으로도 확인한다.

```bash
rg -n 'KVCacheManager|QnnKvCacheManager' Applications nntrainer test
rg -n 'copy_kv_cache_window|append_outputs_to_kv_cache' Applications/CausalLM
rg -n 'setData|fillPlaceholder|syncDependents' nntrainer/tensor
rg -n 'cache_key|cache_value' nntrainer/layers/multi_head_attention_layer.cpp
```

## 3. CPU 경로

현재 주 경로:

```text
CausalLM::allocateAndBindKVCache
  -> KVCacheManager::allocate
  -> cache_k_lN / cache_v_lN input placeholder
  -> Tensor::setData(shared MemoryData, element offset)
  -> incrementalInference(shared Tensor 전달)
  -> TensorPool::fillPlaceholder / syncDependents
  -> MHACoreLayer input[3], input[4]
  -> cache slice 쓰기 및 attention 읽기
```

- cache 전체를 매 토큰 input/output 배열에 복제하는 구조로만 설명하면 부정확하다.
  graph input이라는 표기와 실제 메모리 복사는 별개다.
- concat 경로의 K RoPE는 cache slice를 출력 목적지로 삼을 수 있다.
- FP16 V는 `copyData`가 남고, FP32 activation 경로는 cache 표현으로 변환해 쓴다.
  no-RoPE 경로도 K/V copy가 있다. 필요 연산인 변환과 단순 복사를 구분해서 측정해야 한다.
- FP16 비활성 build의 external placeholder는 `UINT16`이다. 이는 이 경로에서
  half bit 저장 용도이며, 일반 unsigned 고정소수점 KV와 혼동하면 안 된다.
  manager 단위 테스트의 FP32 cache와 모델 추론의 cache dtype도 동일하지 않다.
- 위치는 manager `cache_pos_`, MHA `cache_index`, inference `from/to`에 나뉜다.
  `incrementalInference` 종료 시 위치 동기화와 `setKVCachePosition`의 layer 순회가 있다.
- variable width, Gemma4 layer alias, LFM2의 attention 없는 block을 고려해야 한다.
  현재 allocation이 모든 model layer를 나열한다고 해서 모두 독립 cache owner는 아니다.

core `MultiHeadAttentionLayer`는 내부 cache slice에 projection을 직접 쓰는 부분도
있지만, 이후 cached K/V를 transpose하는 경로가 있다. 공용 manager만 이식한다고
그 read-side 전체 변환이 사라지지는 않는다. CausalLM first로 적용한 다음 별도
adapter로 전환한다. 학습 경로의 cache/gradient 계약은 유지한다.

## 4. GPU 경로

GPU를 하나의 allocator로 보면 안 된다.

| 경로 | 현재 저장 방식 | 직접 접근 / 복사 경계 |
|---|---|---|
| OpenCL SVM | manager의 MemoryPool + gpu-svm allocator | 적합한 attention kernel은 SVM 포인터 직접 사용 |
| OpenCL non-SVM fallback | host cache + kernel scratch | K/V를 clEnqueueWriteBuffer로 업로드하는 경로 존재 |
| OpenCL OHWI/image | MHA 소유 mirror/image와 concat slab 공존 가능 | scatter/gather 및 sync_kv_slab; 포인터 alias만으로 제거 안 됨 |
| CUDA | host, cuda-uvm 또는 cuda device allocation | 선택된 kernel에 따라 direct read/write; V copy kernel 및 fallback 가능 |

기존 `TensorBacking`의 buffer/image view는 **같은 physical layout**에 대해
재사용할 기반이다. image view 생성 자체가 concat→OHWI 전치를 해주지는 않는다.
zero-copy view와 layout conversion을 다른 operation으로 모델링해야 한다.

`MemoryData::ResidencyClass`는 설치되는 API이며 값 변경/재번호화가 금지된
계약으로 주석에 명시되어 있다. 새 manager의 backend descriptor 때문에 이 enum을
임의로 바꾸거나 CUDA/QNN을 OpenCL SVM으로 위장하면 안 된다.

현재 SVM allocator가 host fallback을 할 수 있지만 MemoryPool은 allocator의
`isSVM()`로 속성을 찍는다. **개별 allocation 결과와 advertised capability가
어긋날 수 있는 정적 위험**이다. 장애 재현 결과는 아니며, 공용화 첫 단계의
allocation 실패 주입 테스트 대상으로 삼는다.

## 5. QNN/NPU 경로

### 5.1 manager 코드 존재와 실제 모델 연결은 별개

`QnnKvCacheManager`의 source는 Meson target에 등록되어 있다. 그러나 분석 범위의
공개 tree에서는 클래스 정의 외의 직접 사용 call site를 찾지 못했다.
공개 Gemma4 QNN 구현은 자체 `kvs`, `fresh_kvs`, `prefill_kvs`,
`kv_row_lengths`, `kv_columns`와 `generate_qnn_utils`를 사용한다.
다운스트림 비공개 모델의 helper 사용 여부는 이번 저장소로 확인할 수 없다.

따라서 helper 클래스만 고쳐서 “QNN 적용 완료”로 판정하지 않는다. 실제 공개 모델의
allocation → graph binding → append → turn 전환 경로도 이관해야 한다.

### 5.2 현재 복사 경로

```text
application rpcmem allocation / model input memory
  -> QNNGraph::populateTensor
  -> QNNRpcManager::registerQnnTensor(fd, context)
  -> graphExecute(input KV, 별도 new-K/V output)
  -> append_outputs_to_kv_cache
       K: channel별 memcpy
       V: token row별 memcpy
       window 초과: memmove
  -> 다음 generation
  -> multi-turn prefill: copy_kv_cache_window
```

CPU↔HTP shared I/O 등록이 이미 있어도, **QNN output→persistent KV append**는
다른 복사다. GPU로 memcpy를 옮겨도 이 복사가 없어졌다고 볼 수 없다.

### 5.3 output을 cache offset에 바인딩하는 것만으로 안 되는 이유

현재 key helper의 source/destination 주소는 다음과 같다.

```text
K source(d, t)      = source_base + d * step_stride + t
K destination(d,t) = cache_base  + d * capacity    + position + t
V destination(t,d) = cache_base  + (position + t) * head_width + d
```

K의 `step_stride != capacity`이면 output base pointer를 변경해도 channel별
stride 차이는 해결되지 않는다. QNN graph가 destination stride 또는 state update를
지원하도록 export 계약을 바꾸거나, compatible native layout으로 producer/consumer를
함께 바꿔야 한다. V도 dtype, packing, padding, output shape, alias 규칙을 검증해야 한다.

현재 core 등록 코드는 `QNN_MEM_TYPE_ION`과 fd를 사용한다. 여러 tensor를 한 arena의
임의 offset으로 옮기려면 기존 API가 자동으로 처리한다고 가정할 수 없다.
공식 shared-buffer descriptor 지원 여부를 SDK·backend 버전별로 검증해야 한다.
현재 pointer→context/handle map도 여러 context와 여러 view를 같은 allocation에
등록하는 일반적인 registry로 취급할 수 없다.

### 5.4 NPU 전체에 대한 범위

우선 Android Qualcomm QNN/HTP를 실제 adapter 대상으로 권고한다.
`HtpContext`/HexKL은 별도의 compute offload 경로다. CPU layer를 이용하는 부분은
공용 CPU cache를 재사용할 수 있지만, NPU attention의 직접 KV 접근을 확인한 것은 아니다.
다른 NPU vendor는 공통 descriptor/수명 계약을 바탕으로 별도 adapter를 추가한다.

## 6. 메모리·수명·ring에서 반드시 보존할 조건

1. `shared_ptr<MemoryData>` 공유와 allocation owner 공유는 다르다.
   pool-backed MemoryData는 주소/속성 wrapper이며 pool이 buffer를 해제한다.
   manager 파괴 뒤 view가 남거나 device 작업이 남는 경우를 별도 보호해야 한다.
2. Tensor offset의 단위는 element다. 새 native descriptor의 byte offset과 변환할 때
   dtype 크기, packing, overflow를 검증한다.
3. manager의 기존 read/write view helper는 linear `cache_pos * width` 방식이다.
   MHA는 별도 `cacheRow()`로 ring을 처리한다. 두 API를 그대로 통일하면 wrap에서 틀린다.
4. `kvRingCap`은 chunk `C`에 정렬되고 `W+C` 이상이 되도록 설계되어 있다.
   단순 `capacity=W`로 축소하면 chunk prefill 중 필요한 history를 덮어쓸 수 있다.
5. layer alias는 source geometry와 읽기/쓰기 역할이 일치해야 한다.
   공유 layer를 두 번째 writer로 만들거나 source를 다시 zero-fill하지 않는다.
6. 실패 시 position만 되돌리는 것은 이미 덮어쓴 ring 데이터를 복구하지 못한다.
   초기 버전은 해당 실패 시 session을 invalid로 만들고 reset/rebuild를 요구하는 정책이 안전하다.
7. persist할 때 현재 authoritative plane의 완료된 데이터를 읽어야 한다.
   manager slab만 읽으면 GPU mirror에만 있는 최신 값이 빠질 수 있는 경로를 점검해야 한다.

## 7. 양자화와 persistence의 실제 범위

MHA INT8 경로는 token/head별 FP16 scale과 signed INT8 payload를 별도 내부 Tensor로
둔다. `wireAttentionKVCache(..., use_int8=true)`는 external cache를 연결하지 않는다.
따라서 manager의 dtype enum만 확장해서는 기존 INT8을 공용화할 수 없다.

CausalLM v1 파일에는 magic, version, dtype, geometry digest, model tag, 길이와
payload 크기가 있다. ring physical rows를 저장하고 alias layer payload는 중복 기록한다.
QNN v1은 architecture, tensor 이름/크기/row length/is_key와 raw bytes를 기록한다.
QNN quant parameter 전체를 독립적으로 복원하는 공통 파일 포맷은 아니다.

이들은 유용한 migration 입력이다. “저장 기능이 없다”는 전제로 새 형식 하나만
도입해 기존 파일을 읽지 못하게 만들면 안 된다.

## 8. 외부 규격 확인과 남은 실험

설계 판단에 필요한 범위만 공식 자료를 확인했다.

- SVM은 capability와 coherence 규칙에 따라 map/unmap 및 동기화가 필요하다.
  `CL_MEM_USE_HOST_PTR`는 driver의 별도 cache를 허용한다.
  [Khronos OpenCL API](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html)
- QNN은 tensor별 ION 등록과 fd+offset 기반 multi-tensor shared buffer를 구분한다.
  이는 **stride가 다른 K output의 직접 쓰기를 보장하는 기능은 아니다**.
  [Qualcomm HTP Shared Buffer Tutorial](https://docs.qualcomm.com/nav/home/htp_shared_buffer_tutorial.html?product=924033590759186372)
- Qualcomm OpenCL ION import는 확인할 후보이며 대상 driver의 extension 지원을 확인해야 한다.
  [cl_qcom_ion_host_ptr](https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_ion_host_ptr.txt),
  [cl_qcom_ext_host_ptr](https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_ext_host_ptr.txt)
- dma-buf 공유에는 비동기 device access와 CPU cache 동기화 계약이 따른다.
  RPC fd를 무조건 import 가능한 dma-buf라고 가정하지 않는다.
  [Linux dma-buf 문서](https://docs.kernel.org/driver-api/dma-buf.html)
- CUDA managed memory의 같은 가상 주소는 physical migration 부재를 의미하지 않는다.
  [NVIDIA Unified Memory](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)

| 미확인 항목 | 왜 필요한가 | 해소 방법 |
|---|---|---|
| 실제 GPU/NPU SoC, SDK, driver | import·layout·coherence 가능 여부 결정 | P0 capability report와 실제 read/write probe |
| QNN export source와 graph 변경 권한/도구 | K/V 직접 쓰기와 prefill/decode 공통 layout에 필요 | 작은 update+attention graph로 재현 후 실제 모델 재export |
| target별 driver 내부 copy | API 호출 0만으로 physical zero-copy 증명 불가 | device profiler/trace + vendor contract, 미측정은 unknown |
| pooled view가 owner보다 오래 사는 사용 | source 분석으로 모든 소비자 수명 보장 불가 | fake allocator, ASan, delayed-completion 테스트 |
| GPU image 경로의 최신 cache 저장 | authoritative plane과 snapshot 경계가 분리됨 | prefill 직후/전환 후 save-resume 비교 |
| 기존 helper의 batch>1 및 ring view 일관성 | 최초 MVP는 batch=1이나 회귀 금지 | 기존 batch 테스트 유지, 별도 bounds/stride 테스트 |

모든 “직접 접근 가능” 판정은 target/profile별 실행 증거가 있어야 한다.
이번 문서는 그 검증을 위한 설계와 작업 순서를 제공한다.
