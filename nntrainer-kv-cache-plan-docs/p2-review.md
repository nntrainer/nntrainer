# P2 CPU facade·attention 연결 코드 리뷰

> 이 문서는 Linux P2 당시의 기록이다. 당시 facade는 graph 종료 뒤 모든 owner를
> 일괄 ready 처리했고, 실제 MHA는 checked view를 사용하지 않았다. 아래 core 계약을
> 당시 실행 경로가 모두 강제했다는 뜻으로 읽으면 안 된다. 2026-10-29 후속 변경은
> 실제 MHA 쓰기·읽기 경계에 checked view와 readiness를 연결한다.
> `cache_index = _from`은 upstream에 이미 존재하던 동작이며 새 최적화가 아니다.
> CPU facade ring은 현재 명시적으로 거절한다. 상세는
> [리뷰 후속 기록](review-followup.md)과 [이식 가능한 검증 절차](../docs/kv-cache-cpu-validation.md)를 참고한다.

리뷰 범위는 core storage를 기존 CausalLM allocation/graph 실행에 연결하는 변경이다.
기준은 P1 core commit `21c87982b`이며, 모델 실행 기준은 사용자 결정에 따라
Qwen3 0.6B다. `AGENTS.md`는 수정하지 않았다. 이 문서는 소스 리뷰와 작은 재현의
결과이며, P2 전체 runtime 검증 결과는 [실행 기록](linux-cpu-execution.md)에 남긴다.

## 발견과 반영

| 발견 | 근거·재현 | 반영한 해결 |
|---|---|---|
| CPU producer view가 alias라는 사실만으로 history copy 0을 선언할 수 없음 | 기존 `MHACoreLayer::gemm_attention()`은 전체 prefix를 `Ka`/`Va`에 head별 `memcpy`로 모은 뒤 연산한다. Qwen3는 기본 `use_flash_attention=true`여서 긴 prefill에서 이 경로에 들어갈 수 있다. | x86 비INT8 flash 경로는 기존 AVX2 half/float kernel의 `ldb`를 활용해 token-major manager allocation을 직접 읽는다. K/V prefix packing을 제거하고 Q packing 및 bounded attention scratch는 유지한다. producer와 실제 consumer trace를 나눠 검증한다. |
| facade 재allocation이 core의 active-step 거절을 우회함 | `beginStep()` 후 `allocateCpu()`가 새 core manager를 생성·교체할 수 있었다. 기존 core의 `initialize()` guard만으로 facade 교체까지 막을 수 없다. | `hasActiveStep()` query를 사용해 facade `allocate()` 진입에서 거절한다. 기존 storage identity 유지와 원래 lease commit 가능 여부를 negative test로 검증한다. |
| load가 active cache를 먼저 변경한 뒤 마지막 position 갱신에서 거절될 수 있음 | `load()`는 cache payload를 읽은 뒤 `setPosition()`을 호출한다. 그 시점에만 active-step 상태를 확인하면 committed prefix를 이미 손상시킬 수 있다. ring abort로 invalid한 session의 snapshot도 거절해야 한다. | `save()`/`load()` 진입점에서 CPU session이 valid하고 idle인지 확인한다. active/invalid session은 파일 접근이나 payload 변경 전에 거절한다. |
| non-CPU facade의 `commitStep()` 방어 부족 | 다른 CPU facade에서 발행한 lease를 전달하면 null `cpu_cache_`를 역참조할 수 있었다. 정상 graph 경로에서는 CPU lease만 발행하지만 public wrapper도 잘못된 호출을 안전하게 거절해야 한다. | `beginStep()`과 같이 CPU storage 존재 여부를 먼저 검사한다. |
| Android source 목록의 Meson 타입 불일치 | 신규 core source를 `files()`로 등록했지만 Android의 `jni/meson.build`는 `' '.join(nntrainer_sources)`를 사용한다. Meson 0.61.2의 최소 프로젝트에서 `str.join argument 1 was of type File but should have been str`를 재현했다. | 기존 모듈과 같이 `meson.current_source_dir() / filename` 절대경로 문자열을 source 목록에 추가한다. 재현 로그: `/tmp/kv-meson-file-join/result.txt`. Android cross build 통과를 이 재현으로 대신하지 않는다. |

## 확인한 데이터·수명 계약

CPU facade는 FP32, native FP16, UINT16 half-bit backing을 구분한다. UINT16은
일반 정수 양자화 encoding이 아니다. geometry의 head 수와 head dimension을 core에
넘기며, alias layer는 source와 같은 실제 allocation을 공유한다. 기존 width-only
호출은 head geometry를 알 수 없는 호환 경로로 유지한다.

graph 실행 직전에 committed `[from,to)` interval의 lease를 발행하고, 동기 graph가
성공한 뒤 commit한다. graph 예외는 lease destructor가 abort한다. linear append의
미완료 tail은 publish하지 않으며 ring overwrite가 있었다면 reset 전 session을
거절한다. MHA의 실행 cursor는 동일한 `from`에서 유도한다.

Tensor의 `MemoryData` alias는 allocation 수명을 유지하지만 raw mutable Tensor
접근 자체에 generation 검사를 추가하지는 않는다. 이 호환 bridge를 사용하는 graph
실행과 외부 접근은 같은 동기 step 안으로 제한해야 한다. core `KvView`와 raw Tensor
bridge의 검증 수준을 혼동하지 않는다.

x86 flash direct-read의 offset은 `head * head_dim + token * KV_width`다. 기존
`mha_hsgemm_avx2()`의 QK(`TransB=true`)와 AV(`TransB=false`) 구현 모두 `ldb`를
사용하므로, 두 연산에 동일한 `KV_width` leading dimension을 전달하는 변경은
token-major layout과 일치한다. 실제 출력 일치 여부는 Qwen3 실행 비교로 확인한다.

## trace 해석과 검증 범위

producer alias·append byte 수와 consumer의 history/layout byte 수는 별도로 기록한다.
OHWI gather가 발생하면 consumer 내부에서 추가 packing을 하지 않더라도 manager에서
scratch로 옮긴 byte 수가 남는다. producer profile이 compatibility로 표시된 실행을
완료된 direct-access profile로 세지 않는다.

FP32 activation의 K RoPE 및 V half 변환은 최종 cache slice를 destination으로
사용하는 필요한 연산이다. native FP16 V의 순수 append copy와 ARM의 기존 prefix
packing은 호환 경로로 남는다. 이번 Linux x86 수정만으로 Android CPU, GPU/NPU,
INT8 KV 또는 장치 간 공유가 검증되었다고 표시하지 않는다.

재검증 시 core/facade/ring 및 기존 tiny model 회귀, Qwen3 FP32·weight-quantized
출력 비교, consumer trace를 함께 확인한다. CPU facade 테스트의 실행 환경은
`NNTR_ENGINE=cpu`로 고정해야 OpenCL이 포함된 build에서도 CPU 계약을 검증한다.
