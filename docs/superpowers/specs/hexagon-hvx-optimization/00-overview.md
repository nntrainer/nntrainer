# Hexagon HVX 성능 최적화 (M6) — 개요

- 날짜: 2026-09-14
- 브랜치: `hvx_impl` (M1–M5 완료, 마지막 커밋 299cfdb1). 현재 상태의 권위 있는 공개 문서는 `docs/backend_guide/HEXAGON.md` (§8 수치, §9 후속).
- 목표: qwen3-0.6b HVX-only e2e를 **decode ≥ 30 tok/s, prefill ≥ 100 tok/s** (S25, 8 Elite cDSP, v75 skel)로 끌어올린다. HMX는 이번 범위에서 사용하지 않는다.
- 환경 제약: **디바이스 없음.** 모든 검증·측정은 x86 테스트와 hexagon-sim(v75, SDK 6.0.0.2, toolchain 8.7.08)에서 한다. sim은 HVX 4 unit → 워커 4개. 디바이스 워커 수는 아직 기록이 없다.
- 참고 코드: `/home/j2z0/Project/llama.cpp/ggml/src/ggml-hexagon/htp` (32행 타일 vrmpy matmul, `hvx-mm-kernels-tiled.h`; DMA·워커 큐 패턴). 기법만 차용하고 코드는 포팅하지 않는다.

이 문서는 전체 설계의 개요이며, 세부는 단계별 문서로 나뉜다:

| 문서 | 단계 | 내용 |
|---|---|---|
| [01-profile-baseline](01-profile-baseline.md) | P1 | op kind별 pcycle 프로파일, 워커 수 강제, 배리어 비용 측정, qwen3 차원 sim 프로파일 테스트 → **baseline 표** |
| [02-tiled-weight-layout](02-tiled-weight-layout.md) | P2 | WEIGHTS 이미지의 int8 projection을 32행×128k 타일로 리팩: `pack_weights`, `ref_ops`, validator, `.hexcfg`, ABI v4 |
| [03-w8a8-tiled-kernel](03-w8a8-tiled-kernel.md) | P3 | 타일 vrmpy W8A8 커널 (decode → prefill 토큰 블로킹 → VTCM 스트립 DMA), quant 병렬화, xq 중복 복사 제거, LOGITS/EMBED 타일 경로 |
| [04-w8a16-down-kernel](04-w8a16-down-kernel.md) | P4 | down_proj: `[N][K]` 유지, 32행 블로킹 fp16×int8 커널 + 벡터 리덕션 + VTCM 스트리밍 |
| [05-attention-eltwise](05-attention-eltwise.md) | P5 | ATTN 균등 분할·KV append/softmax 벡터화·K^T 재사용, decode 단일 워커 op의 열 분할 |
| [06-verification](06-verification.md) | P6 | 정확도 게이트, sim 성능 predicate, 문서 갱신, 디바이스 복귀 시 체크리스트 |
| [07-follow-ups](07-follow-ups.md) | — | 범위 외 후속: cross-op weight prefetch, op 퓨전(ABI v5), HAP_power, v79 skel, down_proj 블록 양자화 |

## 현재 상태와 목표 (HEXAGON.md §8.2, 2026-08-31 S25)

| | 현재 DSP | CPU fp32 (같은 폰) | 목표 | 목표 환산 |
|---|---|---|---|---|
| decode (n=1) | 11.2 tok/s (89 ms/tok), 앱 13.0 | 18.7 | **≥ 30** | ≤ 33 ms/tok, 토큰당 읽는 가중치 507 MB(layer 352 + lm_head 155) → **≥ 15 GB/s** |
| prefill (512 tok, chunk 128) | 25.2 tok/s (~40 ms/tok) | 52 | **≥ 100** | ≤ 10 ms/tok, 0.44 GMAC/tok → **≥ 44 GMAC/s** |

- prefill 목표는 HVX int8 피크(4 unit × 128 MAC/vrmpy × 2.09 GHz ≈ 1 TMAC/s)의 수 % 수준. 커널 재구성으로 도달 가능하며 **sim에서 사이클로 증명 가능**.
- decode 목표는 DDR 대역폭 요구다. sim은 DDR 대역폭을 모델링하지 않으므로 **sim으로 증명 불가**. 이번 작업은 decode의 연산·명령수·중복 트래픽을 제거해 대역폭 상한에 가까워질 조건을 만들고, 수치는 디바이스 복귀 후 측정으로 남긴다 ([06](06-verification.md)).

## 병목 진단 (코드 근거)

| 위치 | 문제 | 영향 |
|---|---|---|
| `hvx-quant.h: hvx_dot_i8` / `hvx-matmul.c: mm_compute_range` | (출력행, 토큰)마다 vrmpy K/128회 + **5단계 `vror` 수평합 + 스칼라 store**. decode(k=1024)는 행당 vrmpy 8 : 리덕션 ~7 | 명령의 절반이 리덕션. prefill은 같은 가중치 행을 토큰마다 재로드 → ~10 GMAC/s |
| `hvx_op_matmul_w8a8` | 활성화 quant가 RPC 스레드에서 **직렬** 실행(워커 idle) | 128 tok × k=3072 스칼라 루프가 매 matmul마다 |
| `mm_worker_vtcm` | 워커마다 xq 전체를 자기 VTCM 슬랩에 `memcpy` | 중복 복사 N_worker배 (prefill 128×3072 = 393 KB × 워커 수) |
| `mm_w8a16_worker` / `hvx_dot_fp16_i8` | (행, 토큰)마다 `hvx_sum_sf_pair` 수평합, VTCM 스트리밍 없음 | down_proj = layer 가중치의 ~25% |
| `attn_worker` | kv head(8) 단위 분할 → 워커 4~6개에서 2:1 불균형; KV append 스칼라(transpose 쓰기 m×128); softmax max/scale/sum 스칼라 3패스; query 행마다 K^T 재스트리밍 | prefill 후반 청크에서 ATTN 비중 상승 (M5에서 10.8 → 6.5 s/청크 @pos 896) |
| `htp_graph_forward_upto` + `wp_run` | 451 op × fork-join 배리어, op 간 겹침 없음. decode에서 RMSNORM/ADD/SILU/ROPE는 행 분할이라 **워커 1개**만 일함 | 33 ms 예산에서 배리어 비용이 수 % 이상일 수 있음 → P1에서 측정 |

**weight pre-compute 현황**: 호스트 팩 시점에 RoPE cos/sin 테이블과 norm fp16 변환만 사전 계산. int8 가중치는 `[N][K]` 바이트 복사이며 리팩·행합 사전계산은 없다. 이번 P2가 첫 가중치측 사전 배치다.

## 핵심 결정

| 결정 | 내용 | 근거 |
|---|---|---|
| 가중치 레이아웃 | int8 projection(wq/wk/wv/wo/gate/up + embed)을 **32행 × 128k 타일**(4 KB)로 리팩. **데이터 타입 불변**: int8 값 + per-output-channel fp32 스케일 `[N]` 그대로, 바이트 순서만 바뀜. 텐서 크기·오프셋 불변 | 1 vrmpy = 32 출력행 × 4k 부분합, 수평 리덕션 0. `Q6_Vw_vrmpyacc_VwVbVb`(signed×signed)는 벡터×벡터형만 있어 활성화 4바이트를 `Q6_V_vsplat_R`로 브로드캐스트 (스칼라형은 ub×b만 존재 → 가중치 부호 오프셋이 필요해 배제, [07](07-follow-ups.md)) |
| down_proj (W8A16) | **`[N][K]` 유지**, fp16×int8 커널을 32행 블로킹 + 벡터 리덕션으로 재작성, VTCM 스트리밍 추가 | 수치 무변경(PPL 유지). vrmpy를 못 쓰므로 타일의 이점 없음. 이미지 안에 두 레이아웃이 공존하지만 커널·ref·팩이 op kind로 갈라지므로 경계는 명확 |
| embed / lm_head | `embed` 텐서도 타일링. `MATMUL_LOGITS`는 타일 커널 공용, `EMBED`는 타일 인덱싱 gather(토큰당 K/4개 4바이트 로드) | lm_head가 decode 가중치의 30%. 미타일 시 옛 커널이 남음 |
| 파이프라이닝 범위 | **저비용 항목만**: quant 워커 분산, xq 중복 복사 제거, ATTN 균등 분할, decode 열 분할, 배리어 비용 측정. cross-op prefetch·op 퓨전은 [07](07-follow-ups.md) | prefill 100은 커널로 충분. decode는 대역폭이라 prefetch 효과를 sim에서 측정 불가 → 디바이스 복귀 후 별도 마일스톤 |
| 프로파일 | `htp_graph`에 op kind별 pcycle 누적 카운터. RPC ABI에 올리지 않고 DSP 내부 API + sim 테스트 printf | sim 전용 계측이라 와이어 포맷 불변 |
| 이미지 호환성 | `.hexcfg`에 `weight_layout=tiled32` 키, op-list 헤더 `reserved2[0]`에 레이아웃 id, `NNTR_HTP_ABI_VERSION` 3 → **4** | 옛 `.hexw`/옛 skel 조합이 조용히 틀린 결과를 내지 않고 `init()`에서 거부 |
| sim 프로파일 모델 | 실제 qwen3 차원(hidden 1024, ffn 3072, heads 16/8, head_dim 128, max_seq 2048, max_chunk 128), **layer 2, vocab 4096**. embed/lm_head는 vocab 선형 → 151936으로 환산 보고 | 풀 vocab(155 MB)은 sim 메모리·시간에서 비현실적 |
| 검증 주체 | 에이전트가 x86/sim 빌드·실행 직접 수행, **커밋 전에만 사용자 승인** | 2026-09-14 사용자 결정 (메모리 `hvx-work-approval-gate` 갱신) |
| 워커 수 | 커널은 워커 수를 상수로 가정하지 않는다. sim 4 / 디바이스 미상 → P1에서 강제 인자로 4·6 모두 측정, 분할은 항상 `(total * wid) / nw` 형태 | sim과 디바이스의 unit 수가 다름 |

## 단계 순서와 게이트

```
P1 profile → P2 layout → P3 W8A8 tiled → P4 W8A16 → P5 attn/eltwise → P6 verification/docs
```

- P1은 측정 도구이자 **baseline 표**를 만든다. 이후 모든 단계는 같은 테스트로 전/후 표를 남긴다.
- P2는 커널을 바꾸지 않는다. 옛 커널을 타일 인덱싱으로만 고쳐(정확도 동일) 레이아웃 전환을 따로 증명한 뒤 P3에서 커널을 교체한다. 정수 경로는 bit-exact이므로 x86 `hexagon_ref_run` PPL은 P2·P3 전후 동일해야 한다.
- P3 완료 시점에 prefill 예산 검사(§[06](06-verification.md) predicate). 미달이면 P4·P5 순서를 프로파일 상위 항목으로 재배열한다.
- 각 단계의 Task는 "실패 테스트 작성 → 통과 → sim 프로파일 전/후 → 사용자 승인 → 커밋" 순서다.
- **모든 단계(P1~P5)의 마지막 Task는 `docs/backend_guide/HEXAGON.md` 갱신이다.** 그 단계가 바꾼 공개 사실(와이어 포맷, 레이아웃, 파일 목록, 테스트 이름, 커널 규칙, 측정 표)을 해당 절에 반영하고, 문서 변경만 있는 별도 커밋(`[docs] ...`)으로 남긴다. 바꿀 공개 사실이 없는 단계는 "HEXAGON.md 변경 없음"을 Task 보고에 명시한다. 절별 대응은 [06](06-verification.md) "문서 갱신" 표 참조.

## 범위 외

[07-follow-ups](07-follow-ups.md) 참조. 이번 작업에서 만들지 않지만 설계 문서에는 남긴다: cross-op weight prefetch, op 퓨전(ABI v5), `HAP_power` 앱 투표, v79-native skel, down_proj per-128-block 활성화 양자화, 가중치 부호 오프셋(ub×b 스칼라 vrmpy).
