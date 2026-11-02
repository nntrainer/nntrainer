# HTP MoE 확장을 위한 아키텍처 검토와 리팩터링 계획

기존 Gemma4와 LFM2의 HTP 정확도·성능을 유지하면서, 다른 MoE 모델을 추가할 때
모델별 backend 분기를 계속 늘리지 않는 구조를 목표로 한다. 변경 영향은 HTP
경로로 제한한다. 기존 nntrainer의 모델·그래프·Context·ComputeOps를 활용하고,
HTP 내부의 실행·가중치·캐시·메모리 수명 관리 책임을 분리한다.

검토 대상은 [PR 4428][pr]의 `a3640d20e09a91183fc69cae04165a72e24be213`이다.
아래 구조 검토의 코드 근거는 해당 revision의 파일과 symbol로 표시한다.
2026-11-02 구현 및 검증 결과는 문서 끝의 별도 항목에 기록한다.
문서를 담는 브랜치와 최초 검토 revision은 다르다.

## 검토 결론

미사용 코드 제거, 중복 계약 통합, HTP 내부 책임 분리는 적절하다. 다만 독립적인
`HtpModelSpec`, 별도 모델 프레임워크, 새로운 범용 실행 엔진, 모델별 독립 DSP
session을 만드는 방향은 수정해야 한다. 기존 추상화와 같은 역할을 다시 만들면
모델과 실행 상태의 원본이 두 개가 된다.

[기존 dispatch 문서][architecture]는
`Engine → Context → ContextData → ComputeOps → kernels` 연결을 설명한다.
[아키텍처 규칙][rules]은 중립적인 모델 그래프, 기존 레이어의 whole-op dispatch,
backend별 decode 확장 지점을 지향한다. 문서의 목표와 실제 구현 상태는 구분한다.
특히 `ExecPlan`은 아직 일부가 shadow이며, 완성된 HTP 실행 정책으로 사용할 수 없다.

이번 작업의 승인 기준은 HTP 코드량 감소뿐 아니라, 새 모델 때문에 일반 레이어·공통
실행기·다른 backend에 HTP 전용 분기가 추가되지 않는 것이다.

## 변경 범위

| 구분 | 범위 |
| --- | --- |
| 포함 | HTP backend, DSP/RPC, HTP 전용 모델 연결 코드·레이어·변환 도구·테스트 |
| 제한적으로 포함 | HTP 전용 API 연결과 빌드 목록, 기존 hook의 내부 구현 위임 |
| 제외 | OpenCL/CUDA/CPU 최적화 변경, 일반 MHA 통합, 공통 메모리 planner 재설계 |
| 유지 | 기존 공개 API, tensor enum과 파일 형식, 비HTP 실행 동작 |
| 별도 과제 | 다중 HTP 모델 동시 실행, 여러 DSP PD 지원, 공통 모델·레이어의 전면 통합 |

공유 파일 변경이 필요하면 HTP 연결 지점으로 한정한다. 공통 `ComputeOps` virtual
삭제나 순서 변경을 코드 정리와 함께 수행하지 않는다. 호출자와 ABI 영향 검토가
필요한 변경은 분리한다. `subprojects/`는 수정하지 않는다.

기존 CPU reference와 fallback은 검증·호환 계약으로 보존한다. 새로운 모델을 위해
자동 CPU 전환이나 임의의 근사 연산을 추가하는 것은 이번 범위에 포함하지 않는다.

## 현재 구조의 주요 문제

PR 전체는 622개 파일에 182,086줄 추가, 1,267줄 삭제 규모다. 선행 39개 커밋의
끝인 `4ccf8e113`과 비교하면 523개 파일에 152,614줄 추가, 602줄 삭제다.
전체 변경량을 HTP 신규 구현으로 해석하지 않으며, 선행 OpenCL 영역의 정리는
이번 계획에서 제외한다.

| 문제 | 검토 revision의 근거 | 정리 방향 |
| --- | --- | --- |
| 미사용 pool layer | `Lfm2MoePoolLayer`의 구현·헤더 합계 2,222줄. 저장소 내 생성·factory 등록 경로 없이 빌드 목록과 타입 adapter가 남음 | 외부 C++ 사용 계약 확인 후 클래스·adapter·빌드 항목 제거 |
| 미사용 artifact reader | `preparedFcSourceHash`, `readPreparedFc`, `readPreparedQs2`, `readPreparedQs2Cached`의 저장소 내 호출 없음 | TL2가 사용하는 endian·CRC·파일 I/O 유틸을 추출한 뒤 reader 제거 |
| 고정 옵션 이후의 잔여 분기 | `htp_fixed::E2E=true`, `HtpBackend::e2eRequested`, 항상 참인 `forwardSwitch` | 초기화 상태를 구분하는 조건은 유지하고 도달 불가능한 선택만 제거 |
| 책임 집중 | `htp_compute_ops.cpp` 약 12,000줄에 dispatch·registry·arena·worker·decode·profiling 혼재 | 데이터·mutex·수명까지 내부 구성요소로 분리 |
| 모델 간 잘못된 상속 관계 | `Lfm2HtpMoECausalLM`이 `Gemma4HtpCausalLM` 상속 | 공통 HTP 준비 서비스를 조합하고 모델 의미는 각각 소유 |
| 모델 구조의 이중 정의 | Gemma weight loading에서 이름과 `norm++`, `fc++` 순서로 DSP binding 구성 | 기존 모델에서 정보를 읽는 typed binding adapter 도입 |
| 제어 프로토콜의 공통 인터페이스 유입 | `ComputeOps::set_decode_graph_desc`, `set_decode_graph_param`, token ID 관련 메서드 | 기존 entry는 유지하면서 HTP 내부 runtime으로 위임 |
| 메모리와 상태 추적 분산 | staging의 `holds`, `copied_out`, `IonAlias`, lost 기록, static KV 상태 | 역할별 소유자와 명시적 invalidate·materialize 계약 |
| 불필요한 scratch 선할당 | HTP MHA가 accelerator 성공 시 사용하지 않는 u16 변환 버퍼를 미리 확장 | fallback·비교 경로 진입 시 지연 할당 |
| RPC 중복 | Host/DSP profiling 인덱스와 timed/untimed wrapper 중복 | SDK 비의존 공유 계약과 operation별 내부 helper |
| runtime과 진단 혼재 | production skel에 probe 상시 링크, 별도 ARM/DSP 생성 디렉터리 | source manifest 분리와 일관된 IDL 생성 절차 |

미사용 판단은 저장소 내부 호출 관계에 대한 것이다. 외부 사용 가능성이 있는 export는
추가 확인 없이 삭제하지 않는다. probe에는 실제 테스트 호출자가 있으므로 진단 빌드로
분리하는 작업과 삭제를 구분한다.

## 기존 아키텍처와 맞추는 설계

### 모델 정보와 DSP 표현의 경계

기존 `NeuralNetwork`, `NetworkGraph`, `LayerNode`, layer property가 모델 구조를
정의한다. `RunLayerContext`가 제공하는 weight·shape를 HTP adapter가 읽고,
HTP 내부 변환기가 DSP 실행 descriptor를 만든다.

```text
기존 모델과 layer property
          |
          v
기존 graph와 RunLayerContext
          |
          v
비소유 HTP binding adapter
          |
          v
HTP 내부 실행 계획과 DSP descriptor
```

DSP op list와 wire format는 backend에 필요한 표현이므로 유지한다. adapter는
독립적인 topology나 host weight 저장소를 만들지 않는다. 기존 graph에서 얻을 수
있는 정보를 모델별 설정으로 다시 선언하지 않는다. 부족한 metadata는 HTP 연결
코드에 제한적으로 전달하고, 공통 모델 인터페이스 확장은 별도 과제로 다룬다.

LFM과 Gemma의 공통 weight preparation은 서비스로 추출한다. 이를 새로운
`HtpCausalLM` 모델 계층으로 대체하지 않는다. Gemma 이름의 loader 안에 있는
convolution 처리는 현재 LFM에서 사용하므로, 이름만 보고 제거하지 않는다.

### 연산 dispatch와 decode 실행의 경계

레이어별 연산은 기존 `RunLayerContext::getComputeOps()` 연결을 유지한다.
HTP의 resident decode 상태 머신은 내부 `DecodeRuntime`으로 추출한다.
최종적인 실행 진입점은 이미 존재하는 `HtpContext::runDecode()` 확장 지점을
활용하는 방향으로 수렴시킨다.

현재 HTP는 `runDecode()`를 override하지 않고 graph walk 안의 layer hook으로
parameter·KV를 준비하고 token driver를 실행한다. 따라서 이행은 단순 함수 이동이
아니다. 다음 계약을 먼저 보존해야 한다.

- prefill과 첫 decode에서의 weight binding 및 KV seed
- graph 준비 완료 여부와 한 토큰 prompt 처리
- mixed-engine graph의 기존 실행 경로
- logits 요청, greedy token ID 직접 반환, sampling과 logits processor
- prompt 반복, KV 복원·rewind, cancellation

현재 `runDecode()`의 반환형은 tensor vector다. token ID 직접 반환을 그대로 담을 수
있다고 가정하지 않는다. 우선 기존 hook을 내부 runtime의 facade로 유지하고,
반환 계약을 보존할 수 있는 구간부터 이동한다. 인터페이스를 맞추기 위해 매 token마다
전체 logits를 host에 복원하는 변경은 피한다.

DSP graph 구조는 준비 단계에서 결정하고 weight 주소는 로딩 후 binding한다.
token 실행 때마다 모델 구조를 다시 해석하지 않는다. 동적인 KV 위치·유효 길이는
실행 parameter로 갱신한다. 공통 graph의 node를 실제로 치환하는 작업은 기존
compile/realizer 경계를 따라야 하며 이번 내부 분리와 별개로 설계한다.

### ComputeOps와 HTP 제어 기능의 경계

GEMM, attention, fused operation은 연산 dispatch다. DSP graph word, op 번호,
token transport, expert miss 처리와 pool growth는 HTP 실행 제어다.

후자를 내부 `DecodeRuntime`, weight registry, expert I/O 구성요소로 이동한다.
기존 `ComputeOps` 메서드는 초기에는 전달 함수로 남긴다. 새 모델을 추가할 때마다
공통 인터페이스에 모델별 메서드를 추가하지 않는다. generation policy는 `CausalLM`에
남기고 backend는 요청한 출력 형태를 실행·반환한다.

### 자원 소유권과 수명

| 수명 | 책임 |
| --- | --- |
| backend와 process | 기존 `HtpBackend`가 DSP 연결과 transport 관리 |
| 로드된 모델 | DSP 실행 계획, 등록 weight, expert cache의 소유 및 해제 |
| 실행과 대화 세션 | KV 위치·유효 범위, staging 상태, 진행 중 작업의 reset·save·restore |

현재 HTP는 process-wide backend와 단일 활성 모델을 전제로 한다. 새 RAII 객체는
독립 DSP session이 아니라 기존 runtime의 독점 사용권과 모델 자원의 수명을
표현한다. 객체를 분리했다는 이유로 동시 다중 모델 지원을 주장하지 않는다.

host tensor와 weight는 기존 `Manager`와 `RunLayerContext`의 관리 체계를 따른다.
DSP representation, ION staging, VTCM, DMA ring, expert residency arena는
backend-private 자원으로 유지할 수 있다. 모든 scratch를 일반 tensor planner로
옮기거나 host tensor를 HTP registry가 중복 소유하지 않는다.

상위 종료 절차는 새 작업 중지와 진행 중 작업 완료를 보장한 뒤 model, 등록 메모리,
runtime, DSP 연결을 정해진 순서로 해제한다. 현재 API의
`stop_workers → models.clear → htp_runtime.reset` 관계와 backend close hook의
queue·arena 해제 순서를 보존한다. `ContextData`의 `ComputeOps*`는 비소유 포인터이므로
객체 분해만으로 수명이 안전해지는 것은 아니다.

## 모델별 최적화와 신규 MoE 대응

### 모델 의미와 커널 조건과 튜닝의 분리

| 구분 | 예 | 관리 원칙 |
| --- | --- | --- |
| 모델 의미 | softmax/sigmoid router, GeGLU/SwiGLU, norm·rotation 순서 | 기존 모델과 checkpoint의 의미를 보존 |
| 커널 조건 | head dimension·정렬, TL2/WH layout, static activation scale | 실제 지원 조건을 검증하고 적합한 경로 선택 |
| 성능 튜닝 | expert cache 크기, reader 수, arena 예산, prefetch | 모델·형식·장치·메모리 조건별 검증값 사용 |

기존 `ModelFeatures × DeviceCaps → ExecPlan` 방향을 따른다. 다만 현재 resolver는
HTP 실행 조건을 충분히 표현하지 못하고, `HtpContext::caps()` override도 없다.
기본 CPU capability와 shadow plan을 HTP의 authoritative 정책으로 사용하지 않는다.

이번 단계에서는 HTP 내부 설정을 immutable하게 정리하고 실제 하드웨어 조건과
튜닝값을 분리한다. 기존 neutral metadata를 재사용하며, 새로운 선택 정책은 기존
결과와 shadow 비교한 뒤 항목별로 적용한다. 전역 resolver의 동작은 바꾸지 않는다.

최적화의 적용 여부는 모델 이름이 아니라 연산 의미·형상·형식으로 판단한다.
예측 prefetch는 지원 router와 layout, fused FFN은 연산 순서, static quantization은
해당 위치의 유효 scale, Hadamard는 checkpoint의 회전 규칙을 검증한다. 선택 결과와
미적용 이유를 로딩 시 확인할 수 있게 한다.

특정 배포 pack의 hidden size·레이어 수 검증은 앱의 pack validator에 남긴다.
backend의 일반 지원 범위를 해당 pack의 숫자로 제한하지 않는다.

### 새로운 모델의 추가 절차

1. 기존 모델·레이어로 구조와 연산 의미를 표현하고 weight 역할을 연결한다.
2. HTP adapter가 해당 graph와 metadata를 읽어 지원 조건을 검사한다.
3. 기존 연산·형식으로 표현 가능하면 backend 수정 없이 binding을 추가한다.
4. 형상·자원 조건만 다르면 지원하는 HTP 경로와 튜닝값을 선택한다.
5. 새로운 연산 의미가 필요하면 해당 HTP 기능과 검증을 별도 변경으로 추가한다.

최적화를 적용할 수 없으면 같은 의미의 지원 HTP 경로를 선택한다. 그런 경로도 없으면
로딩 단계에서 원인을 반환한다. 명세가 다양한 expert 수와 top-k를 표현할 수 있다는
사실이 커널의 지원 범위를 자동으로 확장하지는 않는다.

레이어별 형식·expert 수, dense/MoE 혼합을 표현할 수 있도록 하되, 사용 사례가 없는
shared-expert 커널이나 새로운 router를 미리 구현하지 않는다. 기존 Gemma/LFM의
커널·연산·DMA 순서와 튜닝값을 먼저 보존하고 일반화와 성능 개선은 분리한다.

## 구현 계약

구현 기준은 정리 커밋 `3514d3247854e8d10616ea1a63a10df0216be409`이다.
기존 단일 prompt의 kernel, quantization, DMA 순서와 튜닝값을 유지한다.
동일 입력의 token ID와 생성 문자열은 완전히 일치해야 하며, logits와 파일
record는 기존 경로와 동일해야 한다. 동일 장치 비교의 prefill/decode 속도는
각각 2% 이상 느려지면 원인을 조사한다. 단일 측정은 회귀 탐지 참고값이며
통계적 동등성의 근거로 사용하지 않는다. 기존 측정과 같은 page-cache 정책을
사용하고 peak host/ION/DSP 메모리를 함께 확인한다.

실행 상태는 초기 prefill → decode → 추가 prefill → decode를 허용해야 한다.
추가 prefill은 여러 토큰을 받을 수 있으며, DSP가 소유한 attention KV와 LFM2
convolution state를 그대로 이어받아야 한다. 독립 prompt reset과 continuation은
다른 동작이다. host KV가 stale하거나 해제된 상태에서 이를 다시 seed하지 않는다.
지원 구간은 기존 `Context::runDecode` 진입점에서 선택하고, 아직 준비되지 않은
첫 decode와 mixed-engine graph는 기존 layer hook을 유지한다.
Speculative decoding의 draft 모델, 검증, rollback 정책은 이번 작업에 추가하지 않는다.

## 단계별 이행 계획

| 단계 | 변경 | 단계 완료 조건 |
| --- | --- | --- |
| 1 | 모델 정보 원본, 실행 진입점, 수명·fallback·결과 계약 확정 | 기존 graph와 HTP descriptor의 대응 및 종료 순서가 명확함 |
| 2 | 미사용 pool·reader, 고정 분기, scratch 선할당 정리 | 외부 사용 확인, 관련 빌드·기본/fallback 경로 검증 |
| 3 | RPC 인덱스·wrapper, HTP 옵션 파싱·converter 공통 유틸 정리 | wire 값, record bytes, 기존 설정 의미 유지 |
| 4 | 기존 hook을 유지하며 DecodeRuntime·registry·staging·expert I/O 추출 | 상태와 mutex의 소유권 명확화, lifecycle 테스트 통과 |
| 5 | LFM→Gemma 상속 해소와 비소유 binding adapter 도입 | 기존 weight 등록·descriptor 순서 유지, 새 작은 모델 표현 가능 |
| 6 | 지원 구간의 실행 책임을 HtpContext 확장 지점으로 이동 | KV 준비·mixed graph·token/logits 계약과 성능 유지 |
| 7 | production/diagnostic manifest와 테스트 연결 정리 | ARM/DSP 생성물 일치, RPC ABI와 진단 빌드 보존 |

각 단계는 독립적으로 검증할 수 있는 커밋으로 나눈다. 파일 이동과 연산 변경,
구조 변경과 튜닝 변경을 같은 커밋에 섞지 않는다. 기존 virtual 제거, shared layer의
중립화, 일반 allocator 통합이 필요해지면 이번 HTP 한정 작업 밖으로 분리한다.

converter는 checkpoint reader와 affine WH writer를 내부 공통 모듈로 추출한다.
모델별 tensor plan은 유지한다. Q4_0처럼 scale 반올림 시점이 다른 codec은 기존 입력
계약과 byte 동일성을 확인하기 전까지 하나의 정책으로 합치지 않는다.

## 검증과 완료 기준

| 영역 | 확인 사항 |
| --- | --- |
| 정확도 | 기존 Gemma/LFM token·logits, router 선택, quantized record, KV window와 calibration 계약 |
| 특수 실행 | 한 토큰 prompt, chunked prefill, greedy/sampling, logits processor, 강제 fallback·check 모드 |
| 수명 | 반복 prompt, cancellation, 초기화·I/O 실패 후 재시도, unload/reload, CPU handle 공존 |
| 프로토콜 | IDL method ordinal, profiling 인덱스, DSPQ layout, stale stub/skel 감지 |
| 메모리 | peak host·ION·DSP 메모리, scratch 해제, cache budget과 주소 재사용 |
| 성능 | 같은 장치·pack·prompt·생성 길이·cache 조건의 prefill/decode 비교 |
| 확장성 | 지원 연산 안에서 expert 수·top-k·레이어 구성이 다른 작은 모델을 backend 수정 없이 추가 |
| 범위 | HTP 비활성 빌드와 CPU/OpenCL/CUDA 경로의 동작 유지, SDK 의존성 유출 방지 |

동일 조건의 baseline과 반복 측정을 남기고 성능·메모리 허용 범위는 구현 시작 시
명시한다. host test 통과만으로 장치의 수치·속도 동일성을 판단하지 않는다.
양자화나 연산 순서 변경은 별도의 품질 검증이 필요하다.

테스트가 cpp의 문자열 경계를 잘라 `.inc`를 생성하는 구조는, 내부 모듈 추출 후
직접 include/link하도록 전환한다. 독립 reference를 production 구현과 합쳐
검증의 독립성을 잃지 않도록 한다. DSP probe를 분리할 때 기존 IDL ordinal을
삭제하지 않으며 호환 stub 또는 버전 분리를 별도로 설계한다.

완료 시 기존 Gemma/LFM의 검증된 동작을 보존하고, 신규 모델의 차이가 모델 의미와
필요한 HTP 기능에 국한되어야 한다. 기존 HTP 전용 레이어를 모두 중립 레이어로
통합하는 것은 이 단계의 완료 조건이 아니다. 이번 단계는 추가 fork를 만들지 않고
기존 nntrainer 아키텍처로 수렴할 수 있는 경계를 확보한다.

## 2026-11-02 구현 결과

정리 기준 `3514d324` 이후 단계 1–7의 구현을 적용했다. 수치 kernel, IDL,
공통 `ComputeOps` virtual 목록과 순서, descriptor builder는 변경하지 않았다.

| 단계 | 반영 내용 |
| --- | --- |
| 1 | 실행·수명·결과·성능 계약 명시 |
| 2 | 항상 참인 `forwardSwitch` 제거, 사용하지 않는 MHA scratch 지연 할당 |
| 3 | profiling wire enum, affine WH writer와 checkpoint reader, HTP 설정 파싱, u8i4/u8i8 layer timed/untimed dispatch 공유 |
| 4 | `HtpDecodeRuntime`, `HtpWeightRegistry`, `HtpStaging`, `HtpPrefillReaders`, profiling·worker policy 추출; staging/reader 테스트 직접 연결 |
| 5 | LFM2의 Gemma 상속 제거, 공통 weight preparation 조합, graph weight를 빌리는 typed binding |
| 6 | `HtpContext::runDecode`에서 decode 이후 연속된 다중 토큰 입력 처리 |
| 7 | runtime/diagnostic DSP source manifest 분리, qaic 생성 공통화, 진단 ordinal 호환 stub 유지 |

추가 입력은 기존 resident M=1 연산을 토큰 순서대로 실행한다. DSP KV와
convolution state를 유지하며 중간 token ID는 소비하고 마지막 logits를 반환한다.
초기 prefill과 일반 한 토큰 decode는 기존 graph walk를 사용한다. batch=1,
FP32 token 입력, 단일 vocabulary 출력, 준비된 전체 resident graph, 연속된 위치와
이미 확보한 KV 용량 안에서 지원한다. host의 stale KV를 재사용하지 않는다.
독립 prompt의 `run()` reset 의미, 저장·복원 API, sampling 정책은 바꾸지 않았다.
추가 입력용 HMX batch kernel, KV 용량 자동 확장, speculative decoding은 포함하지 않는다.

`htp_compute_ops.cpp`는 11,660줄에서 9,847줄로 분리되었다. 구현 커밋까지의
전체 차이는 51개 파일, +4,766/-3,427줄로 **순증 1,339줄**이다. 분리한 header,
명시적인 상태 검사, 회귀 테스트와 RPC 호환 stub이 포함된 수치이며, 선행 정리
커밋의 삭제량은 포함하지 않는다.

### 검증 결과와 한계

[기계 판독 결과](../measurements/htp_refactor_20261102/validation.json)에
binary hash, 실행 조건, token/logits 비교와 메모리 관측값을 기록했다.
Gemma4 26B A4B TL2/affine WH를 S26 Ultra(v81)에서 측정했으며,
`NNTR_HTP_PAGE_CACHE=0`, 8 threads, prompt 1,555 tokens, 생성 112 IDs
(측정 decode 111 tokens) 조건을 유지했다. 최종 검증 쌍은 각 binary 1회 실행이다.
개발 중 앞선 측정은 별도로 보관했으며 평균에 섞지 않았다.

| 항목 | 변경 전 | 변경 후 |
| --- | ---: | ---: |
| prefill | 643.360 tok/s | 635.992 tok/s (-1.15%) |
| decode | 18.3350 tok/s | 18.5619 tok/s (+1.24%) |
| CLI peak footprint (RssAnon + accelerator) | 3,675,092 KiB | 3,656,540 KiB |
| arena allocation | 2,496 MiB | 2,496 MiB |
| 종료 시 DSP heap | 360,864 KiB | 360,864 KiB |

112개 token ID와 생성 문자열이 완전히 일치했다. 단일 측정으로 통계적 성능
동등성을 확정하지 않는다. CLI footprint는 `RssAnon + accelerator`의 합을
sampling한 최대값이다. 독립적인 peak ION/DSP sampling은 수행하지 않았으므로
위 allocation 및 종료 counter를 peak memory로 해석하지 않는다.

Gemma pack과 별도의 LFM2 synthetic SR15 fixture에서 초기 prefill 8 tokens 이후
`decode 1 → 추가 3 → decode 1 → 추가 2 → decode 1`을 수행했다. 각 모델의
6개 지점 전체 FP32 logits가 한 토큰씩 입력한 경로 및 변경 전 binary와 바이트
단위로 일치했다. LFM fixture는 conv/attention/conv 3 layers, 32 experts,
top-k 2이며 기존 64-expert predictor preset은 끈 상태다. LFM 실제 배포 pack의
생성 품질·속도 회귀 검증은 pack 확보 후 남아 있다.

전체 HTP host suite, 수정 후 RPC·registry·staging 검사, ASan/UBSan staging·reader
수명 검사, WH record 3종 byte fixture, 작은 LFM converter 검사가 통과했다.
Android engine/app과 v81 production/diagnostic DSP 빌드가 통과했고, model source는
host HTP on/off와 Android HTP에서 컴파일했다. HTP를 비활성화한 전체 CPU engine
빌드도 통과했다. ARM/DSP 생성 header가 같으며
IDL과 146개 공개 RPC symbol도 유지된다. Production skel은 904,592 bytes,
diagnostic skel은 968,592 bytes다.

### 재현 진입점

- `test/htp/host/run_host_checks.sh`: SDK 없는 host 검사.
- `test/htp/host/htp_weight_utils_check.py`: NumPy가 설치된 Python에서 WH byte 검사.
- `test/htp/build.sh`: 기본 diagnostic 빌드. `NNTR_HTP_DIAGNOSTICS=0`은 production
  빌드이며 독립 probe RPC는 `AEE_EUNSUPPORTED`를 반환한다. Android app 빌드는
  production을 기본으로 선택하며 `NNTR_HTP_DIAGNOSTICS=1`로 진단 구현을 포함한다.
- CausalLM `app_build`의 `-Denable-htp-device-check=true`:
  `htp_continuation_check PACK gemma|lfm rows|chunks OUTPUT_PREFIX`를 빌드한다.
  각 mode는 새 process로 실행하고 8/9/12/13/15/16 위치의 `.f32` 파일을 비교한다.
  모델 수치 검증용으로 원래 pack의 init/max sequence를 64/128로 설정한다.

## 2026-11-02 추가 구조 정리

기준은 `b51223427`이며, 구현·테스트 정리 3개 커밋은 53개 파일에서
+2,903/-3,351줄로 **순감 448줄**이다. 아래 검증 기록 추가분은 제외한다.

- `htp_qkv_layer.cpp`를 core `QKVLayer`에 통합했다. 일반 QKV의 forwarding,
  export와 weight 순서를 유지하며 기존 HTP type 이름은 얇은 호환 클래스로 남겼다.
- `MHACoreLayer`가 FP16 cache 구현을 조합하고, DSP handle·보정·resident 상태는
  `HtpAttentionState`가 소유한다. CPU/OpenCL/CUDA와 다른 계산·cache layout을
  하나의 수치 구현으로 합친 것은 아니다. 같은 속성 선언 10개만 공유했다.
- `htp_act_quant`와 `htp_q4_0_convert`의 소스·헤더를 `htp_backend`로 옮겼다.
  SDK 없는 기본 빌드와 설치 헤더 이름, 수치 본문은 유지했다.
- AVX2 TQ2CX 선언, expert preload 정책, LFM2 변환 보조 코드, prepared-format
  endian reader를 기존 파일에 통합해 파일 4개와 62줄을 줄였다. AVX2 커널 CPP는 유지했다.
- 오래된 Hadamard patch 생성기와 불필요한 실행 wrapper를 삭제했다.
  실제 회귀 검사는 유지·수정하고 attention/QKV 검사를 Meson에 연결했다.

[검증 기록](../measurements/htp_refactor_20261102/layer-refactor.json):
CPU FP16 off/on과 Android engine/app 빌드, 관련 gtest 89건, 변환 pack 동일성,
prepared-format 및 수명 검사가 통과했다. SDK-free host suite의 마지막 include
오류를 수정한 뒤 영향받은 abs-max/quant 구간을 다시 실행해 통과했다.
선행 구간과 26개 오류 주입 검사는 통과했으며, SDK native/QEMU NEON 검사는 생략했다.

S26 Ultra에서 같은 DSP·model·prompt로 각 버전을 1회 측정했다.

| 항목 | 변경 전 | 변경 후 |
| --- | ---: | ---: |
| prefill | 630.065 tok/s | 634.694 tok/s (+0.73%) |
| decode | 18.2927 tok/s | 18.3108 tok/s (+0.10%) |

112개 생성 ID·문자열이 같았고 Gemma와 LFM synthetic fixture의 6개 continuation
지점 전체 logits가 변경 전 및 한 토큰씩 입력한 경로와 바이트 단위로 같았다.
단일 측정은 통계적 성능 동등성을 확정하지 않으며, 실제 LFM pack의 품질·성능과
전체 원격 플랫폼 CI 검증은 포함하지 않는다. Speculative decoding은 추가하지 않았다.

## 구현 검토 시 참고할 소스

| 파일 | 주요 symbol 또는 확인 내용 |
| --- | --- |
| [context.h][context] | `Context::runDecode`, `DeviceCaps`, shadow `ExecPlan` |
| [model_features.h][features] | 기존 모델 특성 표현과 현재 표현 범위 |
| [neuralnet.cpp][neuralnet] | `getDecodeContext`, `runDecode` 호출, compile/realizer 연결 |
| [htp_context.cpp][htp-context] | 기존 factory 재사용과 ops 연결 |
| [compute_ops.h][ops] | 연산 dispatch와 HTP 제어 메서드의 현재 경계 |
| [htp_compute_ops.cpp][htp-ops] | graph hook, staging, worker, weight registry, runtime teardown |
| [htp_model_runtime.h][runtime] | 단일 활성 모델과 runtime 독점 사용 |
| [gemma4_htp_moe_causallm.cpp][gemma] | weight 이름·순서 기반 decode binding |
| [lfm2_htp_moe_causallm.h][lfm] | Gemma HTP 클래스 상속 |
| [test/htp/build.sh][build] | runtime·probe 링크, DSP 생성·검증 절차 |

[pr]: https://github.com/nntrainer/nntrainer/pull/4428
[architecture]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/docs/backend_guide/ARCHITECTURE.md
[rules]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/docs/backend_guide/ARCHITECTURE_REFACTOR.md
[context]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/nntrainer/context.h
[features]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/nntrainer/model_features.h
[neuralnet]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/nntrainer/models/neuralnet.cpp
[htp-context]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/nntrainer/htp_context.cpp
[ops]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/nntrainer/tensor/cpu_backend/compute_ops.h
[htp-ops]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/nntrainer/tensor/htp_backend/htp_compute_ops.cpp
[runtime]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/Applications/CausalLM/api/htp_model_runtime.h
[gemma]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/Applications/CausalLM/models/gemma4/gemma4_htp_moe_causallm.cpp
[lfm]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/Applications/CausalLM/models/lfm2_moe/lfm2_htp_moe_causallm.h
[build]: https://github.com/nntrainer/nntrainer/blob/a3640d20e09a91183fc69cae04165a72e24be213/test/htp/build.sh
