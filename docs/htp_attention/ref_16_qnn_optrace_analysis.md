<!-- SPDX-License-Identifier: Apache-2.0 -->

# ref_16 -- QNN 컴파일러가 수행하는 최적화 정리

> **출처**: 별도 세션에서 수행된 QNN HTP optrace 분석. 원문을 그대로 보존한
> 참조 문서이며, 우리 코드베이스로의 매핑과 구현 순서는
> `36_from_the_qnn_optrace.md`에 있습니다. 이 파일은 36이 인용하는 근거이므로
> 수정하지 마세요 -- 새로운 트레이스를 뜨면 ref_17로 추가합니다.
>
> **주의**: 여기 나오는 블록 4.72 ms(5,666,199 cycles @ 1.2 GHz)와 조직
> 비교표의 QNN 3.21 ms(seq 1024)는 **서로 다른 측정**입니다. 설정이나 기기가
> 다르니 섞어 인용하지 마세요.

> 자체 NPU 커널/컴파일러 개발 시 QNN과 대등한 성능을 내기 위한 참조 문서
>
> **근거 데이터**: `Qwen3-0.6B_prefill.onnx` (Qwen3-0.6B 어텐션 블록 1개, seq=1024, w8 양자화)를
> QAIRT 2.47.0.260601로 컴파일 후 HTP optrace 프로파일링한 결과
> (`Qwen3-0.6B_attn_quant_w8_1024_chromeTrace_opTrace.json`)
>
> 모델 형상: hidden=1024, Q heads=16, KV heads=8 (n_rep=2), head_dim=128, seq=1024

---

## 0. 측정 베이스라인

| 항목 | 값 |
|---|---|
| E2E wall clock | **5,666,199 HTP cycles** (~4.7ms @1.2GHz) |
| ONNX 노드 | 60개 |
| 실행된 QNN 노드 | **33개** (27개 소멸) |
| 발행된 HTP 마이크로커널 인스턴스 | 5,549개 |
| 총 유닛-사이클 (HVX×8 + HMX + DMA) | 23.1M |
| 병렬 압축률 | **4.1×** (23.1M 작업 → 5.67M wall) |
| 완전 idle 구간 | **0.09%** |
| 동기화 오버헤드 | SyncOp 1,644회 / 37,659 cy = **E2E의 0.66%** |

**하드웨어 유닛 구성** (트레이스에서 관측된 스레드):
- `tid 256` — DMA 엔진 ×1
- `tid 512~519` — HVX 벡터 스레드 ×8
- `tid 768` — HMX 행렬 엔진 ×1

**유닛별 점유율**:

| 유닛 | busy cycles | 점유율 |
|---|---|---|
| HVX5 / HVX4 / HVX7 | 4.42M / 3.80M / 3.33M | 78% / 67% / 59% |
| HVX0~3, HVX6 | 1.77M ~ 2.38M | 31~42% |
| DMA | 734,745 | 13.0% |
| HMX | 538,778 | 9.5% |

> **이 그래프는 HMX(행렬 엔진) 바운드가 아니라 HVX(벡터) 바운드입니다.**
> 자체 커널 개발 시 "행렬곱만 빠르면 된다"는 가정은 틀립니다. 실제 병목은
> elementwise / softmax / transpose / 레이아웃 변환입니다.

---

## 1. 그래프 레벨 — Fusion & Elimination

ONNX 60노드 중 **33개가 실행 그래프에서 완전히 사라졌습니다.**

### 1.1 패턴 기반 대형 융합 (Pattern Fusion)

**RMSNorm 융합 — 7 노드 → 1 op**

```
Pow → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul → Mul(weight)
  ⇒ 단일 QNN `RmsNorm` op
```

q_norm / k_norm 두 곳에 적용 → **14개 ONNX 노드가 2개 QNN 노드로**.
융합된 op은 다시 하드웨어에 맞는 3단 커널로 재분해됩니다:

```
q::rmsnorm_8_compute_mean_squared.tcm   (128 tiles)  ← 부분합 reduction
q::rmsnorm_quant_8_normvals.tcm         (128 tiles)  ← rsqrt + 스케일 산출
q::rmsnorm_8_normalize.tcm              (128 tiles)  ← 정규화 + 재양자화
```

> **구현 시사점**: 융합은 "ONNX 노드를 합치는 것"이 아니라
> **의미 단위로 인식 → 하드웨어 친화적 형태로 재분해**하는 2단계 과정입니다.
> 중간 텐서(`x²`, `mean`, `rsqrt`)를 메모리에 쓰지 않는 것이 핵심 이득입니다.

### 1.2 상수 폴딩 (Constant Folding)

`node_Transpose_0`, `node_Transpose_13`, `node_Transpose_24`, `node_Transpose_130` — **4개 전부 소멸**.

MatMul의 weight 쪽 transpose를 컴파일 타임에 미리 적용해서 `MatMul` → **`FullyConnected_w_scale`** 로 매핑했습니다.
가중치는 어차피 상수이므로 런타임에 전치할 이유가 없습니다.

### 1.3 View/Reshape 흡수 (Producer Absorption)

| ONNX 노드 | 처리 |
|---|---|
| `node_view`, `node_view_1` | producer인 linear에 붙어 `node_linear_post_reshape`로 병합 |
| `node_view_2` | 완전 소멸 — `node_linear_2` 내부에서 `q::Reshape` ×32로 실행 |
| `Unsqueeze` ×4 | 전부 소멸 (메타데이터만 변경, 0 cycle) |
| `node_Reshape_117`, `node_Reshape_145`, `node__unsafe_view_1`, `node_transpose_3` | 전부 소멸 |

### 1.4 RoPE rotate_half 병합

`node_slice_2`, `node_slice_4`, `node_cat`, `node_cat_1` 소멸 → 인접 op 내부에서 `q::Concat`이 in-place로 수행됩니다.

### 1.5 Zero-Cost 뷰 연산 — "Non Executed Tensors"

트레이스에 **833개**의 0-cost 이벤트가 별도 스레드(`tid=1`)로 기록됩니다:

| 종류 | 개수 |
|---|---|
| `q::SlicePad_shape_inplace` | 548 |
| `q::Concat` | 107 |
| `$Const` | 83 |
| `q::Reshape` | 70 |
| `$Shape` | 25 |
| **합계** | **833개 / 4,165 cycles (0.07%)** |

실제 데이터 이동 없이 **뷰 디스크립터(base pointer + stride)만 갱신**하도록 컴파일 타임에 해소한 것입니다.

> **구현 시사점**: 텐서를 `(buffer, offset, strides, shape)` 뷰로 표현하고,
> slice/reshape/concat/pad를 가능한 한 뷰 조작으로 처리하는 IR 설계가 필수입니다.
> 이것만으로 전체 op의 13%(833/6382)를 공짜로 만듭니다.

---

## 2. 양자화 파이프라인 — 재양자화를 데이터 경로에서 제거

**데이터 타입 분포**: 액티베이션 `QUInt8`(1,312) / 가중치 `QInt8`(100) / 바이어스·누산 `QInt32`,`Int32`(532)

### 2.1 스케일 준비 체인의 루프 외부 호이스팅

```
conv_scale_from_weights   →  per-channel weight scale 산출
invscale_to_qi32          →  1/scale 을 Q31 고정소수점으로
combine_scales            →  in_scale × w_scale / out_scale 결합
scale_convert             →  정수 multiplier + shift 로 변환
convert_weights_to_signed.shuffled  →  uint8→int8 + HMX 네이티브 셔플
bias_update_and_fused_shuffle       →  zero-point 보정항을 bias에 흡수
```

**핵심**: 이 체인은 타일 루프 **바깥에서 1회만** 실행됩니다.
결과적으로 데이터 경로에는 **float dequant/requant 커널이 하나도 없습니다.**

재양자화 수식이 정수 bias에 미리 접혀 들어가므로, HMX 출력은
`(int32_acc + folded_bias) * multiplier >> shift` 한 번으로 uint8이 됩니다.

### 2.2 Zero-point 보정 사전 계산

`bias_update_and_fused_shuffle`이 하는 일:

```
Σ(x - zx)(w - zw) = Σxw - zx·Σw - zw·Σx + N·zx·zw
                              ^^^^^^^          ^^^^^^^^^
                              상수 → bias로 흡수    상수 → bias로 흡수
```

`zw·Σx` 항만 런타임 계산이 필요하고, 나머지는 전부 컴파일 타임에 bias로 접힙니다.

> **구현 시사점**: 양자화 커널의 성능은 MAC 처리량이 아니라
> **얼마나 많은 스케일 연산을 데이터 경로 밖으로 밀어냈는가**로 결정됩니다.
> 타일마다 float 스케일을 곱하는 구현은 QNN 대비 수 배 느립니다.

### 2.3 가중치 사전 셔플

`convert_weights_to_signed.shuffled`는 HMX가 요구하는 인터리브 레이아웃으로
가중치를 미리 재배치합니다. 런타임 데이터 경로에서 셔플/전치가 발생하지 않습니다.

---

## 3. 연산자 → 하드웨어 유닛 매핑

### 3.1 HMX(행렬 엔진)로 가는 연산

HMX를 사용한 노드는 **6개뿐**이고, 전부 `q::ConvLayer_s1.opt` 단일 커널입니다:

| 노드 | 역할 | HMX 타일 수 | 타일 shape |
|---|---|---|---|
| `node_MatMul_122` | QKᵀ | 256 | `[1,8,32,256]` |
| `node_scaled_dot_product_attention` | P·V | 64 | `[1,8,32,128]` |
| `node_linear` | Q proj | 8 | `[1,8,128,256]` |
| `node_linear_1` | K proj | 4 | `[1,8,128,256]` |
| `node_linear_2` | V proj | 4 | `[1,8,128,256]` |
| `node_linear_3` | O proj | 4 | `[1,8,128,256]` |

**주목할 점**: QKᵀ와 P·V는 activation×activation 동적 행렬곱인데도
정적 가중치 Conv와 **동일한 커널(`ConvLayer_s1.opt`)로 매핑**됐습니다.
동적 피연산자를 "런타임에 생성된 가중치"로 취급해, 스케일 준비 체인을 그대로 재사용합니다.

> **구현 시사점**: 동적 matmul용 별도 커널을 만들지 말고,
> 가중치 준비 단계만 런타임으로 옮긴 conv 커널로 통합하는 편이 유지보수·성능 모두 유리합니다.

### 3.2 HVX(벡터)로 가는 연산

초월함수(exp, rsqrt), 정규화, elementwise, transpose, 레이아웃 변환 전부.
**Softmax는 HMX를 전혀 쓰지 않습니다** — `q::Softmax_Crouton_Scratch` 128타일이 HVX 8스레드에서만 실행됩니다.

### 3.3 커널 효율 벤치마크 (자체 커널 목표치)

트레이스에서 산출한 **cycle/element** 값입니다. 자체 커널 개발 시 목표 기준선으로 사용하세요.

| 커널 | 타일 shape | cy/tile | **cy/elem** | 평가 |
|---|---|---|---|---|
| `ConvLayer_s1.opt` (HMX) | `[1,8,32,256]` | 528 | **0.01** | HMX 최고 효율 |
| `ConvLayer_s1.opt` (HMX) | `[1,8,128,256]` | 14,650 | **0.06** | |
| `Add.tcm` | `[1,8,16,1024]` | 8,737 | **0.07** | 양호 |
| `linearclip` | `[1,8,512,64]` | 17,616 | **0.07** | 양호 |
| `ForceFormat_Crouton` | `[1,1024,16,128]` | 177,934 | **0.08** | 양호 |
| `convert_weights_to_signed.shuffled` | `[1,1,128,256]` | 4,358 | **0.13** | |
| `Concat` | `[1,1,1024,128]` | 26,123 | **0.20** | |
| `Softmax_Crouton_Scratch` | `[1,8,16,1024]` | 31,657 | **0.24** | exp 포함, 타당 |
| `mul_op` | `[1,8,1024,32]` | 67,306 | **0.26** | |
| `rmsnorm_8_normalize.tcm` | `[1,8,16,128]` | 4,748 | **0.29** | |
| `SlicePad_shape_inplace` | `[1,8,1024,32]` | 82,839 | **0.32** | |
| `Transpose_impl` | `[1,8,32,1024]` | 184,973 | **0.71** | 나쁨 |
| `mul_op` | `[1,2,1024,128]` | 399,728 | **1.52** | **병리적** |

HVX는 8-bit 기준 이론 피크가 스레드당 128 B/cycle이므로 `0.008 cy/elem`가 상한입니다.
- 양호한 커널(0.07~0.1)은 피크의 **10~15%** → 메모리 바운드 영역
- `mul_op [1,2,1024,128]`의 1.52는 피크의 **0.5%** → 명백한 구현 결함

**`Cycles per Packet`** 필드도 유용한 지표입니다 (VLIW 패킷당 소요 사이클):
- 정상 커널: **2.0~2.4** (dual-issue 정상 동작)
- `q::Concat`: **7.9**, `rmsnorm_quant_8_normvals`: **6.3** → 스톨 발생 중

---

## 4. 타일링 + 멀티스레드 병렬화

### 4.1 실측 타일링 전략

| 노드 | 타일 분할 | 유닛사이클 → wall | speedup |
|---|---|---|---|
| `node_Softmax_124` | 128 × `[1,8,16,1024]` | 4,052k → 680k | **6.0×** |
| `rms_norm_node___536094` | 128 × 3단계 | 702k → 106k | **6.6×** |
| `node_expand_1` | 8 tiles | 3,882k → 921k | 4.2× |
| `node_Add_123` | 128 × `[1,8,16,1024]` | 1,181k → 695k | 1.7× |
| `node_Transpose_115` | 8 × `[1,8,32,1024]` | 2,143k → 1,912k | **1.1× (실패)** |

### 4.2 타일 크기 결정 규칙 (역산)

**HVX 커널**: 시퀀스 축(1024)을 **16 또는 8**로 쪼개 8스레드에 분배
- `[1,8,16,1024]` QUInt8 = 128KB/타일 → VTCM working set에 안착
- 128타일 / 8스레드 = 스레드당 16타일 → 로드 밸런싱 여유 확보

**HMX 커널**: 출력 채널을 **256**으로, 배치/시퀀스를 **128 또는 32**로 고정
- `[1,8,128,256]` — 가중치 `[1,1,1024,256]` QInt8 = 256KB가 VTCM에 상주
- 1024 출력 채널 → 256씩 4분할 (`node_linear`의 `weights_to_vtcm` 16회 = 4채널그룹 × 4타일)

> **구현 시사점**: 타일 크기는 **VTCM 용량 / 스레드 수 / 커널 shape 제약** 3개의 교집합입니다.
> 특히 "타일 수 ≥ 스레드 수 × 4" 를 만족해야 마지막 스레드 대기(tail effect)를 흡수합니다.
> `node_Transpose_115`가 8타일밖에 안 되어 1.1×에 그친 것이 반례입니다.

### 4.3 병렬화 실패 사례 분석

`node_Transpose_115` — K를 `[1,8,32,1024]`로 전치:
- 타일 8개 = 스레드 8개와 정확히 같음 → 부하 불균형 흡수 불가
- 타일당 184,973 cycles로 지나치게 큼 (0.71 cy/elem)
- 결과: 2.14M 유닛사이클이 1.91M wall로 거의 압축되지 않음

**교훈**: transpose는 타일을 잘게 쪼개고, 가능하면 **producer의 출력 레이아웃을
미리 전치된 형태로 생성**해서 아예 없애야 합니다.

---

## 5. 메모리 계층 — VTCM / DMA

### 5.1 명시적 VTCM 스테이징

```
q::ConvLayer.opt.weights_to_vtcm   [1,1,1024,256] QInt8   ← 가중치 타일 프리페치
q::ConvLayer.opt.bias_to_vtcm      [1,8,1,64]     Int32   ← 바이어스 프리페치
```

`.tcm` 접미사 커널(`Add.tcm`, `rmsnorm_*.tcm`, `Slice_contig.tcm`)은
**VTCM 상주 데이터 전용 변종**입니다. 같은 연산이라도 데이터 위치에 따라
다른 커널을 선택합니다.

### 5.2 비동기 DMA + 체크포인트 태그

```
DMA 발행     : weights_to_vtcm     flags=['dma']
태그 기록    : DmaCheckpointSet    flags=['dma_set']
완료 대기    : weights_to_vtcm     flags=['dma_wait']
```

**실측 오버랩 증거** (`node_linear`):

```
ts=1,204,562  HMX  ConvLayer_s1.opt  (타일 N 계산)     ──────┐ 11,857 cy
ts=1,213,551  DMA  weights_to_vtcm   (타일 N+1 가중치)  ─┐   │
ts=1,219,719  DMA  weights_to_vtcm   (타일 N+2 가중치)   │   │
ts=1,231,613  HMX  ConvLayer_s1.opt  (타일 N+1 계산)  ───┴───┘
```

DMA 유닛 점유율이 **13%**에 불과한데도 HMX가 멈추지 않는 것이
프리페치가 완전히 동작하고 있다는 증거입니다.

### 5.3 자동 Spill/Fill (VTCM 축출/복원)

| 유형 | 횟수 | cycles | 발생 노드 |
|---|---|---|---|
| `@Spill` | 299 | 288,664 (1.2%) | MatMul_122(255), SDPA(42), linear(2) |
| `@Fill` | 318 | 220,657 (1.0%) | MatMul_122(280), SDPA(36), linear(2) |

QKᵀ 중간 결과가 `[1,16,1024,1024]` QUInt8 = **16MB**로 VTCM(통상 8MB) 초과 →
컴파일러가 자동으로 축출/복원 코드를 삽입했습니다. 비용은 **전체의 2.2%**로 억제됐습니다.

> **구현 시사점**: 레지스터 할당의 spill/fill과 동일한 문제를 텐서 단위로 풀어야 합니다.
> 라이브 구간 분석 → VTCM 압력 계산 → 재계산(rematerialization) vs 축출 비용 비교.
> 이걸 수동으로 관리하는 커널은 큰 시퀀스 길이에서 반드시 실패합니다.

### 5.4 초기 가중치 프리로드

`SystemService_ChunkPreload` **13개**가 `ts≈0`에 실행됩니다.
그래프 실행 시작 전에 가중치 청크를 미리 끌어옵니다.

---

## 6. 데이터 레이아웃 — Crouton

HTP 네이티브 타일드 포맷(**Crouton**, depth-major 블록 레이아웃)을
그래프 전체에 **전파**하고, 경계에서만 변환합니다.

**전체 그래프에서 레이아웃 변환은 단 23회**:
- `q::ForceFormat_Crouton` × 7
- `q::ForceFormat_Flat` × 16

Softmax는 아예 `q::Softmax_Crouton_Scratch`라는 **Crouton 네이티브 커널**로 실행되어
de-tiling 없이 타일드 레이아웃 위에서 직접 동작합니다.

> **구현 시사점**: 레이아웃 변환은 데이터 전체를 훑는 순수 오버헤드입니다.
> 관측된 3개의 `ForceFormat_Crouton`만으로도 342k cycles(6%)를 소모합니다.
> - 커널을 **네이티브 레이아웃에서 직접 동작**하도록 작성
> - 레이아웃을 IR의 타입 일부로 만들고, 컴파일 타임에 전파/최소화
> - 변환이 불가피하면 producer/consumer 융합으로 흡수

---

## 7. 스케줄링 — 여기가 가장 큰 차별화 지점

### 7.1 토폴로지 순서를 따르지 않는 실행

노드 span이 대규모로 겹칩니다:

| 노드 | span | 겹치는 노드 수 |
|---|---|---|
| `node_scaled_dot_product_attention` | 718,547 ~ 5,328,591 | **28개** |
| `Input` | 8,800 ~ 4,561,737 | 27개 |
| `node_mul_4` | 77,695 ~ 4,500,706 | 27개 |
| `node_MatMul_122` | 2,888,247 ~ 5,212,183 | 16개 |

**동시 활성 유닛 분포**:

| 동시 활성 | 시간 | 비율 |
|---|---|---|
| 0개 (idle) | 5,278 cy | **0.09%** |
| 1개 | 1,160,695 cy | 20.5% |
| 2~7개 | 3,218,011 cy | 56.8% |
| **8개 이상** | 1,282,215 cy | **22.7%** |

### 7.2 크리티컬 패스 우선 스케줄링 — V projection 사례

**ONNX 순서는 q→k→v 인데, 실행 순서는 v가 압도적으로 먼저입니다.**

| | 노드 | HMX 실제 계산 구간 |
|---|---|---|
| **V proj** | `node_linear_2` | **131,963 ~ 188,558** |
| K proj | `node_linear_1` | 1,167,808 ~ 1,231,600 |
| Q proj | `node_linear` | 1,204,562 ~ 1,410,238 |
| O proj | `node_linear_3` | 5,521,100 ~ 5,612,572 |

**이유**: 의존성 체인 길이가 다릅니다.

```
V:  linear_2 → transpose_2 → expand_1(GQA repeat) → PV matmul
                             ^^^^^^^^^^^^^^^^^^^^
                             전체 최고 비용 3.88M 유닛사이클

Q:  linear   → RMSNorm → transpose → RoPE → QK^T
K:  linear_1 → RMSNorm → transpose_1 → RoPE → expand → QK^T
```

V 경로에는 RMSNorm도 RoPE도 없어서 projection 직후 바로 최고 비용 연산으로 진입합니다.
스케줄러는 이를 인식하고 V를 맨 앞으로 끌어올렸습니다:

```
V proj HMX      131,963 ── 188,558
V transpose_2          155,580 ── 293,653
V expand_1(GQA)             196,255 ──────────────────── 1,117,505   ← HVX 8스레드 점유
K proj HMX                                        1,167,808 ── 1,231,600
Q proj HMX                                           1,204,562 ── 1,410,238
                                                     ^^^^^^^^^^^^^^^^^^^^
                                          expand_1이 HVX를 쓰는 동안 Q/K 가중치 DMA 진행
```

> **구현 시사점**: list scheduling with critical-path priority.
> 각 노드의 "남은 경로 비용(height)"을 비용 모델로 추정 → 큰 것부터 ready list에서 꺼냄.
> 단순 토폴로지 순서 실행 대비 이 그래프에서만 수십만 cycle 차이를 만듭니다.

### 7.3 어텐션 소프트웨어 파이프라이닝 (flash-attention 유사)

4개 노드의 타일 실행 구간이 크게 겹칩니다:

```
QK^T   (HMX, 256 tiles)  4,504,200 ─────────────────── 5,212,183
+mask  (HVX, 128 tiles)          4,628,810 ──────────────── 5,257,093
softmax(HVX, 128 tiles)             4,644,929 ───────────────── 5,317,803
P·V    (HMX,  64 tiles)                4,693,359 ──────────────── 5,323,951
```

50k cycle 버킷별 타일 분포 — 네 단계가 계속 섞여 있습니다:

```
ts~4,700,000: QK^T 21, +mask 12, softmax 12, PV 4
ts~5,000,000: QK^T 32, softmax 16, PV 4
ts~5,150,000: QK^T 15, +mask 13, softmax  8, PV 4
```

앞선 QKᵀ 타일이 나오는 즉시 HVX가 mask+softmax를 처리하고,
완성된 P 타일을 HMX가 V와 곱하는 동안 HMX는 남은 QKᵀ 타일도 계속 처리합니다.
**HVX가 softmax를 하는 동안 HMX를 놀리지 않는 것**이 핵심입니다.

> **구현 시사점**: 노드 단위가 아니라 **타일 단위로 의존성을 추적**해야 이게 가능합니다.
> IR에서 노드 간 의존을 "전체 텐서 준비 완료"가 아니라
> "타일 (i,j) 준비 완료"로 표현하는 설계가 전제 조건입니다.

### 7.4 불변 연산 호이스팅

`node_scaled_dot_product_attention`의 스케일/가중치 준비는 `ts=718,547`,
실제 HMX 계산은 `ts=4,693,359` — **약 400만 cycle 앞당겨 실행**됐습니다.
피연산자에 의존하지 않는 준비 작업을 최대한 앞으로 끌어 다른 계산 뒤에 숨겼습니다.

---

## 8. 동기화 메커니즘

`SyncOp` 1,644회 총 **37,659 cycles = E2E의 0.66%**.

전체 배리어가 아니라 **DMA 체크포인트 태그 기반의 fine-grained 동기화**입니다:
- `DmaCheckpointSet` — 특정 DMA의 완료 태그를 테이블에 기록
- `dma_wait` — 해당 태그만 대기 (다른 DMA/계산은 계속 진행)

> **구현 시사점**: 노드 경계마다 `barrier()`를 거는 구현은
> 위 7.3의 파이프라이닝을 원천적으로 불가능하게 만듭니다.
> 토큰/펜스 기반 의존성 추적이 필수입니다.

---

## 9. QNN도 놓친 부분 — 자체 구현 시 개선 기회

### 9.1 GQA repeat_kv를 곱셈으로 구현 (최대 병목)

`node_expand` / `node_expand_1` (ONNX `Expand`)를
**ones 텐서와의 브로드캐스트 곱셈**(`q::mul_op`)으로 구현했습니다.

| | 값 |
|---|---|
| `q::mul_op` 총 유닛사이클 | **8,011,003 (전체 커널 시간의 34.6%)** |
| 그중 `[1,2,1024,128]` 타일 16개 | 6,395,648 |
| 타일당 | ~400,000 cycles (**1.52 cy/elem**) |
| 상위 15개 최장 커널 | **전부 이 mul_op** |

같은 트레이스의 `q::Concat [1,1,1024,128]`은 26,123 cy (0.20 cy/elem)로 **7.6배 빠릅니다.**

**개선 방향**:
1. KV를 물리적으로 복제하지 말고 **MatMul 커널이 head 축에서 stride 0으로 브로드캐스트 읽기**
2. 불가피하면 Expand를 곱셈이 아닌 **DMA 복사 또는 Concat**으로 lowering
3. 그래프 레벨에서 `Unsqueeze→Expand→Reshape` 패턴을 인식해 no-op 뷰로 처리

이것만 해결하면 이 그래프에서 **수 M cycle 규모의 이득**이 예상됩니다.
부수적으로 V projection 내부의 `q::Reshape [1,1,1024,32]` ×32도 함께 사라집니다.

### 9.2 K transpose 병렬화 실패

`node_Transpose_115`: 2.14M 유닛사이클 → 1.91M wall (**1.1×**), 1,479,786 cy (6.4%).
K projection 출력을 처음부터 전치된 레이아웃으로 생성하면 통째로 제거 가능합니다.

### 9.3 Softmax 절대 비용

`q::Softmax_Crouton_Scratch` 4,052,196 cy (17.5%), 0.24 cy/elem.
1024×1024 어텐션 행렬 전체를 materialize한 뒤 softmax를 도는 구조입니다.
**진짜 flash-attention**(online softmax + 타일 단위 running max/sum)을 구현하면
이 중간 행렬 자체와 §5.3의 spill/fill 535개를 함께 제거할 수 있습니다.
QNN은 파이프라이닝은 했지만 online softmax는 하지 않았습니다.

---

## 10. 구현 우선순위 체크리스트

성능 기여도 순으로 정렬했습니다.

### Tier 1 — 이게 없으면 QNN 대비 수 배 느립니다

- [ ] **뷰 기반 텐서 IR** — slice/reshape/concat/pad/unsqueeze를 `(buffer, offset, stride, shape)` 조작으로 처리 (§1.5)
- [ ] **양자화 스케일의 정수 bias 폴딩** — 데이터 경로에서 float 연산 완전 제거 (§2.1, §2.2)
- [ ] **타일 단위 의존성 추적 + 논블로킹 동기화** — 노드 배리어 금지 (§7.3, §8)
- [ ] **VTCM 스테이징 + 비동기 DMA 프리페치** — double buffering (§5.1, §5.2)
- [ ] **네이티브 타일드 레이아웃 + 컴파일 타임 전파** (§6)

### Tier 2 — QNN 수준 도달에 필요합니다

- [ ] **패턴 융합**: RMSNorm(7노드), LayerNorm, SiLU/GELU, MatMul+Transpose (§1.1, §1.2)
- [ ] **크리티컬 패스 우선 list scheduling** (§7.2)
- [ ] **자동 spill/fill** — 텐서 라이브니스 분석 기반 (§5.3)
- [ ] **가중치 사전 셔플** — 하드웨어 네이티브 레이아웃으로 오프라인 변환 (§2.3)
- [ ] **타일 크기 자동 결정** — VTCM 용량 / 스레드 수 / 커널 제약의 교집합, 타일 수 ≥ 스레드 수 × 4 (§4.2)

### Tier 3 — QNN을 넘어서는 부분

- [ ] **GQA broadcast-read MatMul** — repeat_kv 물리 복제 제거 (§9.1)
- [ ] **Online softmax (flash-attention)** — 중간 행렬 materialize 제거 (§9.3)
- [ ] **Producer 출력 레이아웃 최적화** — transpose 노드 제거 (§9.2)

### 검증 지표

자체 커널이 QNN과 대등한지 판단하는 기준:

| 지표 | QNN 실측값 | 목표 |
|---|---|---|
| 병렬 압축률 (유닛사이클 / wall) | 4.1× | ≥ 4× |
| 완전 idle 비율 | 0.09% | < 1% |
| 동기화 오버헤드 | 0.66% | < 2% |
| Spill/Fill 오버헤드 | 2.2% | < 5% |
| 레이아웃 변환 횟수 | 23회 / 60노드 | 최소화 |
| Zero-cost 뷰 연산 비율 | 833 / 6,382 = 13% | ≥ 13% |
| elementwise 커널 효율 | 0.07 cy/elem | ≤ 0.1 |
| Cycles per Packet | 2.0~2.4 | ≤ 2.5 |

---

## 부록 — 분석 재현 방법

```bash
# 트레이스 생성 (on-device)
#   ondevice/optrace_config.json + ondevice/run_recur_net_optrace.sh 참조
#   리더 라이브러리: libQnnHtpOptraceProfilingReader.so (QAIRT 2.47.0.260601)

# 이벤트 구조
#   pid=0        : Core 0 Overview (유닛별 집계, 분석은 여기서)
#   pid=1..N     : QNN 노드별 뷰 (동일 이벤트 중복)
#   tid=1        : Non Executed Tensors (0-cost 뷰 연산)
#   tid=256      : DMA
#   tid=512..519 : HVX 스레드 ×8
#   tid=768      : HMX

# 핵심 필드
#   args["QNN Op Name"]   : 원본 ONNX 노드 이름 → 융합/소멸 추적
#   args["HTP Op Type"]   : 실제 실행된 마이크로커널
#   args["Flags"]         : dma / dma_set / dma_wait / uses_hvx / uses_hmx / sync
#   args["Dims"]          : 타일 shape → 타일링 전략 역산
#   args["Cycles per Packet"] : VLIW 패킷 효율 (2.0~2.4 정상)
#   args["Dominant Path Cycles"] : 크리티컬 패스 기여도
```

ONNX 노드 대비 소멸 노드 추출:

```python
import onnx, json
m = onnx.load('Qwen3-0.6B_prefill.onnx', load_external_data=False)
d = json.load(open('Qwen3-0.6B_attn_quant_w8_1024_chromeTrace_opTrace.json'))
ex = [e for e in d['traceEvents'] if e.get('ph')=='X' and e['pid']==0 and e['tid']!=1]
qnn = {e['args'].get('QNN Op Name') for e in ex}
print([n.name for n in m.graph.node if n.name not in qnn])   # 융합/소멸된 노드
```
