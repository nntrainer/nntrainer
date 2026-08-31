# Hexagon HMX 백엔드 — 개요 (w8a8 matmul → HMX u8i8 전환)

- 날짜: 2026-08-20 (갱신 2026-08-31, M5 완료 반영)
- 전제: hexagon-hvx M1–M5 **완료** (`hvx_impl` e8e8b5b1…8ca80f3a): qwen3-0.6b가 HVX 경로로 e2e 실행·검증되고 앱이 `nntr_config.json "engine": "htp"`로 DSP 경로를 탄다. 현재 상태의 권위 있는 공개 문서는 `docs/backend_guide/HEXAGON.md` (§5.4 앱 통합, §8 수치, §9 후속). hexagon-hvx 스펙 원본은 로컬 전용(git 미추적)이다.
- 목표: 모든 w8a8 matmul을 HexKL NPU Micro API의 HMX u8i8 커널로 교체 — prefill·decode 동일 경로
- 의존성: `~/hexkl_addon/` (HexKL 1.0 Beta.2, `libhexkl_micro.a`) — **2026-08-31 기준 이 머신에 없음.** 착수 전 확보가 첫 단계

이 문서는 전체 설계의 개요이며, 세부는 마일스톤별 문서로 나뉜다:

| 문서 | 마일스톤 | 내용 |
|---|---|---|
| [01-spike-relocation](01-spike-relocation.md) | H1 | 선행 스파이크: wh 타일 재배치 유효성 판정 + 변환 비용 측정 — **설계 확정 게이트** |
| [02-build-integration](02-build-integration.md) | H2 | libhexkl_micro.a 링크, HAP_compute_res HMX 파라미터, 버전 체크 |
| [03-weight-pipeline](03-weight-pipeline.md) | H3 | init 1회 wh 변환(메모리 중립), W8_CX colsum 출력, zero-point 보정 수식 |
| [04-hmx-kernel](04-hmx-kernel.md) | H4 | HMX matmul 커널, u8 양자화, 스레딩·VTCM 배치 |
| [05-e2e-verification](05-e2e-verification.md) | H5 | 비트 비교 → 3-way 정확도 → 성능 게이트, 디스패치 전환 |
| [06-optimizations](06-optimizations.md) | — | 최적화 포인트: 루프 순서 타일링, wh 버퍼 배치 순서, 누산기 stall 천장, 융합 후보 |

## 핵심 결정

| 결정 | 내용 | 근거 |
|---|---|---|
| 적용 범위 | **HMX everywhere** — `MATMUL_W8A8`(레이어당 q/k/v/o/gate/up 6개)와 `MATMUL_LOGITS`. prefill·decode 동일 경로. **`MATMUL_W8A16`(down_proj, fp16 활성화 — M4 정확도 게이트로 도입)은 범위 밖, HVX 유지** | M5 측정(아래): decode도 대역폭이 아니라 HVX 루프 연산 병목이라 HMX 이득이 decode에도 기대됨. 레이아웃 단일화로 이중 레이아웃 문제 소멸, "커널 차이는 M뿐" 유지. M>1 decode 확장 시 자동 수혜. down_proj는 u8i8 불가(fp16 입력) → `hmx_mm_f16` 후속 |
| 가중치 배치 | **init 1회 변환, 메모리 중립** — row-major → wh 레이아웃 변환 후 원본 해제 | wh 레이아웃은 HexKL 비공개 내부 포맷이라 호스트 사전 생성 불가. on-the-fly 변환은 decode에서 토큰당 변환 비용 발생 |
| 활성화 | per-token 동적 양자화 유지, int8 대칭 → **u8 (zero-point 128 고정)** | HMX u8i8은 activation이 unsigned. 보정항은 가중치 열합으로 처리 ([03](03-weight-pipeline.md)) |
| HVX 커널 | 삭제하지 않고 **안전망 + 비교 기준**으로 유지 | op 디스패치 플래그로 선택. 비트 동일 출력이 가능해 단위 테스트 기준으로 사용 |
| 자원 확보 | `hexkl_micro_hw_init`/`hmx_lock` **사용 안 함** | 헤더에 "테스트·예제용, 통합자는 자체 루틴 사용" 명시. 기존 `HAP_compute_res` 경로에 HMX 파라미터 추가 |

## M5 측정으로 갱신된 전제 (2026-08-31, 8 Elite S25, v75 skel — HEXAGON.md §8.2)

| 항목 | 값 | HMX 설계에의 함의 |
|---|---|---|
| decode | 11.3 tok/s (88 ms/token, 512·1024 ctx 동일) | 가중치 598 MB/88 ms ≈ **7 GB/s 실효** — 대역폭 상한(~40 GB/s, ~60 tok/s)에 한참 못 미침 |
| prefill | 23.2 tok/s @1024 (~43 ms/token, 청크 32/64/128/256 무관) | 0.44 GMAC/token → **~10 GMAC/s = 연산 병목** |
| VTCM/DMA 더블버퍼 on/off | prefill 동일, decode +5% | 가중치 반입은 병목 아님 → `hvx_dot_i8`가 (row, token)마다 하는 수평 합이 원인 |
| 앱 (`engine="htp"`) | prefill 23.8 / decode 13.0 tok/s, RSS 758 MB | 같은 폰 **CPU fp32 decode 18.7 tok/s**가 더 빠름 — HMX(또는 HVX 루프 재구성)가 넘어야 할 1차 기준 |
| 어텐션 | M5에서 K^T 벡터화 완료 (위치 무관) | 남은 시간의 대부분이 matmul → 이 스펙이 다음 병목을 정면으로 겨눔 |
| HAP_power | rc 0, 클럭 불변(2.09 Gcyc/s) | 클럭은 이미 최고 코너, 전력 투표 불필요 |
| skel | **v75만** (v79는 IEEE/qf32 체인 오동작, HEXAGON.md §7) | hexkl은 `hexagon_toolv88_v75` 변형을 링크 |

이 수치로 보면 "decode는 HMX 이득 없음"이라는 원래 가정은 틀렸다 — decode도 연산 병목이므로 HMX 전환(혹은 그 전에 HVX 내부 루프만 재구성해도)이 decode·prefill 모두에 이득이다. **HMX 착수 전에 HVX 루프를 먼저 고치는 것**(row 묶음당 가중치 1회 로드, dot마다 수평 합 제거 — 어텐션 커널에 M5에서 적용한 것과 같은 재구성)이 더 싼 선택일 수 있으니 H1과 함께 판단한다.

**H1 스파이크가 게이트다.** wh 타일 재배치가 무효로 판정되면 본 설계 전체가 fallback(prefill-only HMX + on-the-fly 변환, decode는 HVX 유지)으로 전환된다 — [01](01-spike-relocation.md) 참조.

## 마일스톤

1. **스파이크** ([01](01-spike-relocation.md)): wh 타일 재배치 판정 + 변환 비용 측정 → 설계 확정 또는 fallback 전환
2. **빌드 연동** ([02](02-build-integration.md)): libhexkl_micro.a 링크 + HAP_compute_res HMX 파라미터 + 버전 체크
3. **가중치 변환 파이프라인** ([03](03-weight-pipeline.md)): W8_CX colsum 출력 + init 변환 + 원본 해제 — 시뮬레이터 검증
4. **HMX matmul 커널** ([04](04-hmx-kernel.md)): u8 양자화 + 커널 본체 — HVX와 비트 비교 통과
5. **e2e 전환 + 성능** ([05](05-e2e-verification.md)): 디스패치 전환, 3-way 정확도, TPS·init 시간 측정

## 범위 외 (후속 작업)

- `hmx_mm_f16` (SDPA/attention 내부 HMX화)
- `u8i4` (w4a8 — 4bit 가중치)
- QKV / gate-up 융합 (호스트 lowering 변경 — [06 ④](06-optimizations.md))
