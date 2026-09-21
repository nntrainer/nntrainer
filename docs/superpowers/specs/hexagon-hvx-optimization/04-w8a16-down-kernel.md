# P4 — down_proj W8A16 커널 (레이아웃 유지)

상위: [00-overview](00-overview.md) · 선행: [03](03-w8a8-tiled-kernel.md) · 다음: [05-attention-eltwise](05-attention-eltwise.md)

대상: `MATMUL_W8A16` (`down`, `[N=1024][K=3072]`, fp16 활성화, 레이어 가중치의 ~25%). 수치 계약 불변: fp16 x × int8 w를 fp32로 누적, `(fp16)(acc * sw[n])`.

## 현재 비용

`hvx_dot_fp16_i8`: 128 B 가중치마다 unpack 1 + hf 변환 2 + `hvx_vec_mpyacc_f32_f16` 2(각각 Wqf32 곱 + 변환·가산 수 개), 끝에 `hvx_sum_sf_pair`(64-lane 수평합). (행, 토큰)마다 수평합 + 스칼라 store. VTCM 없음.

## 2026-09-16 결정 (계획서 `docs/superpowers/plans/2026-09-16-hvx-m6-p4-w8a16-down-kernel.md` 우선, 사용자 승인)

| # | 항목 | 이 문서 원안 | 결정 | 이유 |
|---|---|---|---|---|
| D1 | 수치 계약 | fp16 활성화 × int8, fp32 누적 불변 | **per-token int16 활성화 양자화**(`sx = absmax/32767`), int32 정수 누적(`Q6_Ww_vmpyacc_WwVhVh`), `((float)dot·sw[n])·sx[t]` → fp16. **Task 1이 x86 `hexagon_ref_run` PPL(기준 37.4738, ≤ +0.5 %)로 판정**, 미달 시 D1' | v75의 fp32 누적은 64 MAC당 ≥ 5 벡터 op(§7 규칙 1)라 아래 "설계"로는 ~2배가 한계 → ≥ 3배 미달 예상. int16은 int8(+6 % PPL)보다 256배 촘촘 |
| D1' | 대안 | — | (a) lane별 스케일(토큰당 64개, 블록 = `k ≡ j (mod 64)`) → 재측정; (b) 아래 원안(fp 블로킹, 수치 불변) | |
| D2 | 레이아웃 | `[N][K]` 유지 | 유지 | 타일화 + hi/lo vrmpy는 ABI v5·이미지 재push가 따라옴 → 07 ⑤ |
| D3 | VTCM 스트리밍 | Task 2 | **P4 제외** → 07 ⑬(디바이스 A/B) | sim이 DDR/DMA를 모델링하지 않아 판정 불가 |
| D4 | 블로킹 | R=4/TB=2 측정 후 조정 | R=4/TB=2 고정, 행 꼬리 = 마지막 행 복제 + 부분 store, 토큰 꼬리 tb=1 | 스윕은 07 ⑩(디바이스) |
| D5 | 반올림 | — | P3 매직 방식 `1.5·2^16`, 0행·tie행 동일, 일반 행 ±1 비율 ≤ 2 % | `Vw_equals_Vsf` 반올림 모드 미검증 |
| D6 | 07 포함 | — | ⑪ 포함(Task 4a), ⑨⑩⑫ 제외 | |
| D7 | x86 오라클 | PPL 불변 | `hexagon_ref_run` PPL이 새 정의로 바뀜 → 새 기준값; 디바이스 판정 = DSP vs 새 ref | |

아래 "설계"·"Task"는 원안 기록으로 남긴다. 실행 순서·검증은 계획서를 따른다.

## 설계

1. **행 블로킹 R=8, 토큰 블로킹 TB=2**: 가중치 128 B를 한 번 unpack/hf 변환해서 TB 토큰에 재사용, 활성화 벡터를 R 행에 재사용. 레지스터: R×TB 누산기 VectorPair = 16 pair = 32 벡터는 과다 → **R=4, TB=2** (8 pair) 시작, 측정 후 조정.
2. **리덕션 트리**: R개 누산 VectorPair를 한 번에 줄인다. `hvx_sum_sf_pair`를 R번 부르는 대신, 4개의 pair를 (lo+hi 합 → 4 벡터) → vshuff/vdeal 2단계로 "벡터 1개에 4행 합"으로 접고 마지막에만 수평합 1회. 리덕션 비용 R분의 1. 결과 4행 × 1 토큰 = 4 fp32 → fp16 4개 store (스칼라 store 4회 또는 vscatter 없이 `Q6_V_vror`로 모아 8 B store).
3. **VTCM 스트리밍**: P3의 스트립 더블버퍼를 `[N][K]` 행 청크에 재사용(`mm_worker_vtcm` 일반화 — 기존 `Follow-up:` 노트). 활성화 fp16 `m*k*2` B(128×3072×2 = 768 KB)는 공유 VTCM 영역에 1부.
4. 워커 분할: 행 `[N*wid/nw, N*(wid+1)/nw)`, R 배수로 정렬(N=1024는 4·6 워커 모두 R=4 배수로 나눠짐: 256 / 170.67 → 나머지 행은 R<4 꼬리 루프).

## 수치

fp32 누적 순서가 바뀌므로 ref(`ref_matmul_w8a16`, 스칼라 순차 합)와 bit-exact가 아니다. 기존 sim `matmul` W8A16 검사 tolerance(`|d| <= atol + rtol*|ref|`)를 유지하고, x86 `hexagon_ref_run`은 커널을 쓰지 않으므로 PPL은 불변. **정확도 회귀 게이트는 디바이스 복귀 후 `--eval` PPL(19.98 ± 노이즈 밴드)** — [06](06-verification.md).

## Task

1. R=4/TB=2 블로킹 + 리덕션 트리, DDR 경로 (`test_matmul` W8A16 케이스 m=1, 8, 128)
2. VTCM 스트리밍 일반화 (`test_matmul_dma`에 W8A16 케이스 추가, `HTP_MM_NO_VTCM`과 tolerance 내 동일)
3. `profile` 전/후 표
4. **HEXAGON.md 갱신** ([06 표](06-verification.md) P4 행): §1.4 `MATMUL_W8A16` 설명, §7 누적 순서·디바이스 PPL 게이트, §8.3 전/후 행. `[docs]` 커밋

## 완료 기준

- `profile prefill512`의 `MATMUL_W8A16` 사이클 baseline 대비 **≥ 3배 감소**.
- `MATMUL_W8A16`이 레이어 사이클의 25%를 넘지 않음(타일 W8A8 6개 op 합계 대비). 넘으면 [07 ⑤](07-follow-ups.md) 블록 양자화 전환을 사용자에게 제안.

## 측정 기록 (2026-09-16)

구현 커밋 `82eacd7e` (+ 도구 `7dd3a949`). P3 HEAD는 `8669ed2b`. 설계는 spec 원안(fp 누적)이 아니라 **per-token int16 활성화 + int32 lane-wise 누적**(결정 D1): `htp_quant_row_fp16_i16`(magic `1.5*2^16`, 해상도 2^-6) → `Q6_Wh_vunpack_Vb` + `Q6_Ww_vmpyacc_WwVhVh`(`MM16_R 4` × `MM16_TB 2`) → int32 lo+hi → `Q6_Vsf_equals_Vw` → vshuff 2단 + vror 3단(qf32) → `((float)dot*sw)*sx` → fp16. `k <= 16384`에서 int32 정확, validator가 초과를 rc 5로 거부. VTCM 스트리밍(Task 2)은 **미구현**(D3 → [07 ⑬](07-follow-ups.md)).

**sim `profile acc` (v75, 4 workers, timing off, `logs/hexagon/sim_prof_acc_w4_p4t2.log`, wall 10m39s vs P3 ~16m)**

| 항목 | P3 | P4 | 비 |
|---|---|---|---|
| per-shape `per_call` `MATMUL_W8A16` 3072→1024 | 2,445,246 | **683,954** | **3.58×** |
| acc 전체 op 합 | 6,940,426 | 3,415,503 | 2.03× |
| acc STAT | `max_abs=0.0914676 max_rel=93.3662` | `max_abs=0.077216 max_rel=98.2637` | 게이트 0.1/0.1 PASS |

나머지 kind는 ±5 % 이내. 13개 골든 테스트 13/13 PASS(`logs/hexagon/sim_13tests_p4.log`); `matmul_w8a16_m1 0/0`, `m2 0/0`, `m8 0.0078125/0.000974659`, `m7 0.03125/0.0138741`, `quant16_generic pm1=167/25600`. **sim 긴 시나리오(prefill0/prefill512/decode512)는 P4에서 재실행하지 않았다**(사용자 결정: 디바이스 측정으로 대체).

**PPL (x86 `hexagon_ref_run --eval`, ref 자체가 int16 정의로 바뀜)**

| 프롬프트 | P3 정의 | P4 정의 | DSP(S25 Ultra, v75 skel @`82eacd7e`) | 차 |
|---|---|---|---|---|
| 387 tok (로컬 `/tmp/t.i32`) | 37.4738 / 133 | **37.1629 / 134** (−0.83 %) | 37.6413 / 138 | +1.29 % |
| 512 tok (`eval_prompt512.txt`) | 33.4011 / 181 | **33.0195 / 184** (−1.14 %) | 33.3132 / 189 | +0.89 % |

P3 skel A/B(같은 토큰): 387 → 37.5336/134 (+0.16 %), 512 → 33.5493/183 (+0.44 %). 1-layer 격리(`--dump-op`): down 입력(SILU_MUL 출력, P3 이후 코드 불변)이 이미 rel-RMS 2.65 % 발산, DSP 자기 입력으로 int16 ref를 재계산하면 rel-RMS 1.6e-4 / max_abs 0.00098(fp16 1 ulp) → 커널은 실리콘에서 정의대로 정확. → [07 ⑮](07-follow-ups.md).

**디바이스 성능 (S25 Ultra `R3CY10WM83Y`)**: `--eval` decode 55.9 M → **43.3 M** pcycles/step (host 45.8 → 38.0 ms, 21.8 → 26.3 tok/s); 생성 모드 decode ~73 M → **59.5 M** (host 76.4 → 31.2 ms, 13.1 → 32.0 tok/s); 128-tok prefill chunk 1,413 M / 736 ms → **871.9 M / 478 ms**.

**완료 기준**: acc 3.58× ≥ 3× 로 충족, 디바이스에서 판정; prefill512 sim 행은 미측정이라 "레이어 사이클 25 %" 조건은 이번 측정으로 읽을 수 없고 디바이스 수치로 대체했다.
