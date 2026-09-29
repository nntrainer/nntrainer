# H4 — HMX matmul 커널 (`htp/ops/hmx-matmul.c` 신규)

상위: [00-overview](00-overview.md) · 선행: [03-weight-pipeline](03-weight-pipeline.md)

## 실행 흐름

1. **양자화 (op 진입, 기존과 동일 위치)**: per-token 동적, u8 zero-point 128. `hvx-quant.h`에 u8 변형 추가
2. **activation 배치**: `hexkl_micro_hmx_copy_submatrix_to_8b_activation`으로 M×K 블록을 VTCM의 HMX activation 영역에 배치 (M은 64 블록 단위 패딩, decode는 M=1 → 1블록)
3. **N-타일 루프**: wh 가중치 타일을 DDR HMX 버퍼에서 VTCM으로 DMA (기존 더블버퍼 패턴 재사용) → `hexkl_micro_hmx_mm_u8i8` → `hexkl_micro_hmx_acc_read_int32`. 루프 순서는 **N-타일 바깥, K-스트립 상주, M-블록 재사용** — 가중치 DMA 트래픽을 M과 무관하게 청크당 1회로 만든다 ([06 ①](06-optimizations.md) 필수 반영)
4. **후처리**: `(acc − 128·colsum)·s_w·s_x` → fp16 저장 (`MATMUL_LOGITS`는 마지막 토큰 행만 fp32)

## 스레딩

HMX 유닛은 코어당 하나의 공유 자원 — HVX 커널의 N-슬랩 워커 병렬화를 그대로 쓸 수 없다. HMX 발행(2–3단계)은 단일 스레드, 워커 풀은 DMA 프리페치·양자화·후처리(dequant/저장)를 담당하는 분업으로 재배치.

## VTCM 배치

activation 블록 + 가중치 더블버퍼 2면 + 누산기 출력 영역 + HMX config(`hexkl_micro_hmx_config_size()`, VTCM 끝에 배치 — 예제 규약). 기존 `HTP_GRAPH_VTCM_BYTES` 예산 내 재배치.

## 디스패치

- 기존 HVX w8a8 커널(`hvx_op_matmul_w8a8`, `hvx_op_matmul_logits`)은 삭제하지 않고 **안전망 + 비교 기준**으로 유지
- op 디스패치 플래그로 HMX/HVX 경로 선택

## 범위 (M5 이후 갱신)

- 대상 op: `MATMUL_W8A8`(레이어당 q/k/v/o/gate/up), `MATMUL_LOGITS`. `MATMUL_W8A16`(down_proj, fp16 활성화, `hvx-matmul.c` `mm_w8a16_worker`)은 HVX 유지 — 레이어 MAC의 약 1/4이 남으므로 기대 이득 상한을 계산할 때 반영
- 측정 훅: M5의 `HTP_MM_NO_VTCM` 같은 컴파일 스위치 패턴으로 HVX/HMX를 같은 skel 빌드 규칙에서 갈아 끼우고, `hexagon_e2e_test`의 pcycles/us 라인을 `summ.py` 식으로 집계

## 완료 기준

- 시뮬레이터에서 HVX w8a8 출력과 **비트 동일** (M=1과 M=프리필 청크 양쪽)
