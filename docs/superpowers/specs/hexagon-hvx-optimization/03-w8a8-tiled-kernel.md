# P3 — 타일 vrmpy W8A8 커널

상위: [00-overview](00-overview.md) · 선행: [02](02-tiled-weight-layout.md) · 다음: [04-w8a16-down-kernel](04-w8a16-down-kernel.md)

대상 op: `MATMUL_W8A8`(레이어당 q/k/v/o/gate/up), `MATMUL_LOGITS`, `EMBED`(gather만). 파일: `htp/ops/hvx-matmul.c`, `hvx-embed.c`, `hvx/hvx-quant.h`.

## 커널 본체 (한 n-타일 × 토큰 블록)

```
for nt in [nt0, nt1):                         # 워커의 n-타일 범위
  strip = W_tile(nt, 0..k_tiles)              # VTCM(DMA) 또는 DDR 직접
  for tb in 토큰 블록 (TB개, TB = 4 기본, m<4면 m):
    acc[0..TB) = 0
    for kt: for g in 0..31:
      vW = vmem(strip + kt*4096 + g*128)       # 1 vload, TB 토큰에 공유
      for t in tb: acc[t] = Q6_Vw_vrmpyacc_VwVbVb(acc[t], vW,
                              Q6_V_vsplat_R(*(int32*)(xq[t] + kt*128 + 4g)))
    for t in tb:
      f = Q6_Vsf_equals_Vw(acc[t])                          # int32 → fp32, lane = 행
      f = f * vmem(sw + nt*32) * splat(xq_scale[t])         # qf32 곱 → Vsf (기존 규칙: qf 포맷만)
      y[t][nt*32 .. +32) = fp16(f)   (W8A8)  |  fp32 그대로 (LOGITS)
```

- 벡터당 명령: 1 vload(공유) + TB × (스칼라 4B 로드 + vsplat + vrmpyacc). 스칼라 로드는 스칼라 슬롯. 리덕션 0, 스칼라 store 0.
- 출력 32열이 연속이므로 fp16 저장은 64 B(반 벡터). 두 n-타일을 묶어 128 B 풀 벡터 store로 하는 것은 측정 후 선택.
- HVX 규칙(HEXAGON.md §7) 유지: fp32 곱은 `Q6_Vqf32_vmpy_VsfVsf` → `Q6_Vsf_equals_Vqf32`, fp16 narrow는 `Q6_Vhf_equals_Wqf32` 계열. IEEE hf 연산 금지.
- 수치: int32 누적은 순서 무관 정확값 → ref와 **bit-exact**. fp32 스케일 곱 순서 `(float)acc * sw * sx`는 ref와 동일하게 유지(결합 순서 바꾸면 fp16 반올림이 달라질 수 있음).

## 워커 분할과 VTCM

- 워커 wid는 n-타일 `[n_tiles*wid/nw, n_tiles*(wid+1)/nw)`. 타일 단위라 4·6 워커 모두 균등(n_tiles = 32~96).
- (구현에서 폐기 — 결정표 참조: xq는 VTCM에 넣지 않고 `c->xq`를 캐시 경유 4바이트 스칼라 로드로 읽는다. VTCM은 워커별 가중치 더블버퍼 전용.) ~~**xq는 VTCM에 1부**만: op 진입 시 quant 결과를 VTCM 앞부분(공유 영역, `m*k` B)에 쓰고 모든 워커가 읽는다. 기존 워커별 `memcpy` 제거. VTCM 4 MB 중 공유 xq 최대 `128 × 3072 = 384 KB`, 나머지를 워커별 가중치 더블버퍼로 분할.~~
- 가중치 스트리밍: 워커 슬랩에 n-타일 스트립(`32*K` B: K=1024 → 32 KB, K=3072 → 96 KB) 단위 더블버퍼. 기존 `dma_queue` 패턴(kick c+1, wait c) 그대로, 청크 = 정수 개의 스트립. 슬랩이 스트립 2개 미만이면 DDR 직접 경로.
- `HTP_MM_NO_VTCM` 스위치 유지 (측정용).

## 활성화 quant 병렬화

- `htp_quant_row_fp16`을 워커 잡으로: 토큰 행을 워커에 분배, 결과를 공유 VTCM xq(또는 VTCM 없으면 `c->xq`)에 기록. 배리어 1회 추가되지만 직렬 스칼라 루프(prefill 128×3072)가 사라짐.
- quant 자체 벡터화는 별도 Task(3b)로 분리됐다. 수치 규칙·테스트 계약은 [03-1-quant-vectorization](03-1-quant-vectorization.md)이 대체한다(이 문서의 `Q6_Vw_equals_Vsf`(RNE) 안은 §7 규칙 1 위반이라 폐기).

## decode(m=1) 경로

같은 커널의 TB=1. 벡터당 vload + 스칼라 로드 + vsplat + vrmpyacc → 1 packet 수준. decode의 진짜 상한은 DDR이므로 여기서는 명령수 최소화와 DMA 연속성만 확보.

## MATMUL_LOGITS

타일 커널 공용, `y_is_f32`. N=151936 → 4748 n-타일, 워커 균등. VTCM 스트리밍 사용(기존 "m=1이라 무의미" 주석은 리덕션 비용 때문이었고, 이제는 DMA 연속성이 이득).

## EMBED gather

토큰당 `row = tokens[t]`: `nt = row/32, r = row%32`; k-그룹 g마다 `*(int32*)(base + (nt*k_tiles + k/128)*4096 + ((k%128)/4)*128 + r*4)` 4바이트 로드 → int8×4 → fp32 × scale → fp16. K/4 = 256 로드/토큰. 스칼라로 충분(EMBED는 op-list 1회).

## Task (각각 실패 테스트 → 통과 → `profile` 전/후)

1. 벡터화 quant (`test_quant` bit-exact)
2. 타일 커널 DDR 경로, TB=1 (`test_matmul` m=1, `test_logits`)
3. TB=4 토큰 블로킹 (`test_matmul` m=8, 128; ref bit-exact)
4. 공유 VTCM xq + quant 워커 분산 (`test_matmul_dma`)
5. 스트립 단위 DMA 더블버퍼 (`test_matmul_dma`, `HTP_MM_NO_VTCM`과 bit-identical)
6. EMBED 타일 gather (`test_embed`) — P2의 임시 브리지 제거
7. `graph`·`profile` PASS, 표 갱신
8. **HEXAGON.md 갱신** ([06 표](06-verification.md) P3 행): §1.4 op 설명, §2.2 VTCM 공유 xq 배치, §7 타일 커널 규칙, §8.3 전/후 행. `[docs]` 커밋

## 완료 기준

- sim 전 테스트 PASS, x86 `hexagon_ref_run` PPL 불변(커널은 sim에서만 돌지만 이미지는 공용).
- `profile prefill512`: `MATMUL_W8A8` 사이클이 baseline 대비 **≥ 4배 감소**. 28-layer 환산 prefill 청크 예산은 [06](06-verification.md) predicate로 판정.
- `profile decode512`: `MATMUL_W8A8 + LOGITS` 명령수(sim 스레드 Insns) baseline 대비 ≥ 3배 감소.

## 측정 기록 (2026-09-16)

커밋: `cd5ac6b3`(Task 1 타일 커널, 브리지·`wrow_scratch` 제거, LOGITS 공용 + VTCM) → `49b95816`(Task 2 `MM_TB 4`) → `c777f89a`(Task 3 quant 워커 분산) → `3a52c764`(Task 3b quant HVX 벡터화, [03-1](03-1-quant-vectorization.md)) → `343653a5`(최종 리뷰 후속: LOGITS 행 quant도 `quant_worker`로 — RPC 스레드는 HVX 컨텍스트가 없다, HEXAGON.md §7 규칙 7). 계획·결정표: [plans/2026-09-15-hvx-m6-p3-w8a8-tiled-kernel.md](../../plans/2026-09-15-hvx-m6-p3-w8a8-tiled-kernel.md). 로그(세 시나리오 모두 `343653a5`): `logs/hexagon/sim_prof_{prefill0,prefill512,decode512}_w4_p3c.log`, 집계 `logs/hexagon/summ_p3c.txt`.

| 항목 | P1 baseline | P3 HEAD(`343653a5`) | 배수 | 기준 | 판정 |
|---|---|---|---|---|---|
| prefill512 `MATMUL_W8A8` | 938,298,568 | 20,931,069 | 44.8× | ≤ 234,574,642 | PASS |
| prefill512 total | 1,059,812,175 | 141,982,701 | 7.5× | — | — |
| decode512 `W8A8 + LOGITS` | 8,134,596 | 527,389 | 15.4× | ≤ 2,711,532 | PASS |
| decode512 total | 9,265,863 | 1,664,446 | 5.6× | — | — |
| prefill0 total / 환산 ms/tok | 1,028,601,630 / 53.82 | 109,369,374 / 5.66 | 9.4× | — | acc STAT 불변 |
| prefill512 환산 ms/tok | 55.45 | 7.37 | — | ≤ 10 (06) | PASS |
| decode512 환산 ms/tok | 69.35 | 12.47 | — | ≤ 16.5 (06) | PASS |
| thread Insns prefill512 | 6,976,126,635 | 3,551,900,612 | 2.0× | — | — |
| thread Insns decode512 | 5,893,729,661 | 3,113,313,034 | 1.9× | — | — |
| 워커 spread T1..T4 | 80 % / 84 % | 16 % / 17 % | — | < 15 % (06) | 근접 미달\* |

\* sim Insns는 측정 창이 아니라 전체 실행(DMA busy-wait 포함)이라 불균형 상한이다 — 01 §해석 6의 단서 그대로.

- **Task 3b 우회 경위**: Task 4 1차 측정(HEAD `c777f89a`)에서 prefill512 W8A8 226,505,764(4.14×, PASS)이나 decode512 W8A8+LOGITS 7,396,241(1.10×, **FAIL**). 원인은 스칼라 `htp_quant_row_fp16` — k=1024 한 행이 ≈345K pcycles이고 m=1에서는 워커 하나가 혼자 돌린다. 07 ⑧로 미뤘던 벡터화를 P3 안으로 되돌려(사용자 결정 2026-09-16) 게이트를 닫았다. prefill512도 18.19 → 7.37 ms/tok.
- **`MM_TB 4` 판정**: acc per-shape W8A8 per_call이 1.3~3.7 % 감소. sim이 가중치 대역폭을 제대로 과금하지 않아 작게 나온 것으로 보고 채택, 디바이스 재측정은 07 ⑩.
- **정확도**: `profile acc` STAT `max_abs=0.0914676 max_rel=93.3662`이 P2부터 P3 HEAD까지 **불변**. 움직인 것은 `matmul_w8a8_m8` STAT만(max_abs 0 → 0.0078125 = fp16 1 ulp, qf32 에필로그), 단위 테스트 2e-3/1e-3 바운드 안.
- **LOGITS quant 수정 비용**: `343653a5`에서 LOGITS per_call이 59,580 → 65,485(prefill512) / 59,232 → 65,092(decode512)로 늘었다 — 배리어 1회 추가. 그 대신 RPC 스레드에서 HVX를 돌리는 버그(HVX 컨텍스트 미소유; sim은 강제하지 않는다)가 없어졌다.
- **P4 근거**: `MATMUL_W8A16`(3072→1024, row-major, VTCM 없음)이 prefill512의 55.4 %, decode512의 36.7 %. per_call 38,941,117(prefill512) / 316,338(decode512)로 같은 레이어의 W8A8 전체 합보다 크다. ATTN이 장문맥 2위(27.5 % / 16.7 %).
- **디바이스 (2026-09-16, S25 Ultra `R3CY10WM83Y`, v75 skel @`343653a5`)**: `--eval` 386 step PPL **37.5336** / top1 134 (x86 ref 37.4738 / 133) → 실리콘 정상. decode DSP 55.9 M pcycles/step(26.8 ms, host wall 45.8 ms = 21.8 tok/s; M5는 156 M / 77.1 ms / 13.0 tok/s), 생성 모드 ~73 M steady(35 ms, host 76.4 ms), 128-tok prefill chunk 1,413 M / 736 ms(M5 ≈ 5.4 s). 상세·함정은 [06 디바이스 복귀](06-verification.md). **디바이스 decode 병목은 이제 호스트/RPC**(step당 ~40 ms, logits 복사·argmax) → 07 ⑫.
