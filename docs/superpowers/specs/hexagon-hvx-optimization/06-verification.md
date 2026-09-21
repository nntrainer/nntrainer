# P6 — 검증, 성능 predicate, 문서

상위: [00-overview](00-overview.md) · 선행: [05](05-attention-eltwise.md)

## 정확도 게이트 (매 Task)

| 표면 | 명령 | 통과 조건 |
|---|---|---|
| x86 와이어/레이아웃 | `gcc ... test_oplist_header.c`, `build_x86_hexagon/test_lowering`, `test_w8cx_bin` | PASS |
| x86 모델 정밀도 오라클 | `hexagon_ref_run /tmp/qwen3_full --tokens /tmp/t.i32 --eval` (386 step) | PPL **20.2718 동일**(P2 이후 항상; 정수 경로 bit-exact) |
| sim 커널 골든 | `run_sim_test.sh {smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph profile}`, `profile acc` (M6 P2부터 태스크당 게이트) | 전부 `SIM_TEST <name> PASS` |
| sim 정확도 회귀 | `profile acc`(또는 `prefill0`)이 8-토큰 prefill logits를 ref와 비교 | **atol 0.1 / rtol 0.1** (`find_divergence.py` 기본값과 동일; 2026-09-14 결정). qwen3 차원에서는 per-token int8 re-binning이 fp16 1-ulp 차이를 layer당 ~2배 증폭해 logits max_abs ≈ 0.08(rms 0.77)에 이르며 단일 원인 op는 없음(`logs/hexagon/sim_diverge_qwen3_2l.log`). M6 P2부터는 `profile_prefill_acc STAT max_abs=0.0914676 max_rel=93.3662`(01의 재-baseline 참조) — sim 프로파일 모델이 타일 인덱스를 통해 같은 난수 바이트를 순열해 읽어 값이 이동했을 뿐, 정수 경로는 커널과 스칼라 참조가 bit-exact로 일치하고 0.1/0.1 바운드는 그대로 통과한다. tiny `graph` 테스트는 3e-2/5e-2 유지 |

## 성능 predicate (sim, 디바이스 아님)

P1 `profile` 출력을 `tools/hexagon/summ_prof.py`(신규, `SIM_PROF` 라인 → 28-layer/풀-vocab 환산 표)로 집계한다. 표 머리에 `timing=on/off`, `workers=n`을 항상 적는다.

| predicate | 계산 | 기준 |
|---|---|---|
| **prefill 예산** | `prefill512` layer op 합 × 14 + (EMBED+LOGITS) × 151936/4096, ÷ 2.09 Gcyc/s ÷ 128 tok | **≤ 10 ms/tok** (sim-derived). 통과하면 "prefill 100 tok/s는 sim 사이클 기준 충족, 디바이스 미측정"으로 보고 |
| decode 연산 예산 | `decode512` 같은 환산 | ≤ 33 ms/tok 의 **50 % 이하**(연산이 대역폭 시간과 겹칠 여지). 대역폭 자체는 sim에서 미증명으로 명시 |
| 배리어 | `barrier_empty_x1000` / 1000 × 451 | 토큰당 ≤ 3 ms(sim-derived). 초과 시 [07 ②](07-follow-ups.md) 퓨전 우선순위 상향 근거로 기록 |
| 워커 균형 | sim 스레드별 Insns | ATTN·W8A8 워커 간 편차 < 15 % |

### P3 판정 (2026-09-16, 근거·로그는 [03 측정 기록](03-w8a8-tiled-kernel.md#측정-기록-2026-09-16))

sim 세 시나리오 모두 단일 커밋 `343653a5` 측정 (`logs/hexagon/sim_prof_*_w4_p3c.log`, 집계 `summ_p3c.txt`).

| predicate | P1 | P3 HEAD(`343653a5`) | 기준 | 판정 |
|---|---|---|---|---|
| prefill 예산 | 55.45 ms/tok | **7.37** | ≤ 10 | PASS |
| decode 연산 예산 | 69.35 ms/tok | **12.47** | ≤ 16.5 (33의 50 %) | PASS |
| prefill512 `MATMUL_W8A8` (03 완료 기준 ≥ 4배) | 938,298,568 | 20,931,069 (44.8×) | ≤ 234,574,642 | PASS |
| decode512 `W8A8 + LOGITS` (≥ 3배) | 8,134,596 | 527,389 (15.4×) | ≤ 2,711,532 | PASS |
| 배리어 | 0.935 ms/tok | 0.934 | ≤ 3 | PASS |
| 워커 균형 (sim Insns spread) | 80 % / 84 % | 16 % / 17 % | < 15 % | 근접 미달 — sim Insns는 전체 실행 기준 상한이라 `HTP_MM_NO_VTCM` 비교 또는 디바이스 측정으로만 확정 가능 (prefill0 344 %는 스칼라 ref가 단일 스레드로 도는 것) |

정확도: `profile acc STAT max_abs=0.0914676 max_rel=93.3662`이 P2 재-baseline 이후 P3 전 커밋에서 불변. 움직인 STAT은 `matmul_w8a8_m8`(0 → 0.0078125, fp16 1 ulp)뿐이고 `quant`는 ±1 비율 계약으로 개정됐다([03-1](03-1-quant-vectorization.md)).

### 디바이스 복귀 (S25 Ultra `R3CY10WM83Y`, v75 skel @`343653a5`, 2026-09-16)

위 "디바이스 복귀 시 체크리스트" 중 2·3·4 대체 수행. 항목 1(`wp_size` 출력)·5(앱 tok/s)는 미수행.

| 표면 | 결과 | 판정 |
|---|---|---|
| `run_device_test.sh` | RPC_TEST PASS, round-trip median 259 µs | PASS |
| `hexagon_e2e_test --eval` 386 step | PPL **37.5336** / top1 134 vs x86 `hexagon_ref_run` 37.4738 / 133 (§5.1 로컬 프롬프트) | PASS — 타일 커널·`Q6_Vsf_equals_Vw`·벡터 quant 모두 실리콘에서 정상 |
| `--eval` decode DSP | 55.9 M pcycles/step (26.8 ms @2.09 GHz), host wall 45.8 ms = **21.8 tok/s** (M5 156 M / 77.1 ms / 13.0 tok/s) | 2.8× |
| `--chunk 128 --steps 64` decode | ~73 M steady (35 ms), 초기 ~20 step은 ~92 M; host wall 76.4 ms = 13.1 tok/s (M5 185 M / 89.2 ms) | 2.5× |
| 128-tok prefill chunk @pos 0 | 1,413 M pcycles / host 736 ms (M5 ≈ 5.4 s) | ~7× |

- 체크리스트 3의 "PPL 19.98 ± 노이즈"는 `eval.txt`가 아닌 §5.1 로컬 프롬프트로 대체 측정됐다 — 판정은 **DSP vs x86 reference 차이**(37.5336 vs 37.4738)로 읽는다.
- decode 병목이 DSP에서 **호스트/RPC**로 이동: 생성 step당 ~40 ms가 RPC 왕복 + 151,936 float logits 복사·argmax. [07 ⑫](07-follow-ups.md).
- `--steps` 워밍업(~92 M → ~73 M)은 원인 미규명.
- tie-row 디바이스 검증은 PPL 일치로 간접 확인만 됨 — [07 ⑧](07-follow-ups.md)에 남김.
- **함정**: `run_e2e_test.sh`의 push가 파일 크기만 비교해 P2 레이아웃 변경(크기 동일)을 감지하지 못했고, 디바이스에 남은 row-major 이미지 + tiled32 `.hexcfg` 조합으로 PPL 2e8…nan이 나왔다. 수동 push로 해소. [07 ⑪](07-follow-ups.md).

### P4 판정 (2026-09-16, 근거·로그는 [04 측정 기록](04-w8a16-down-kernel.md#측정-기록-2026-09-16))

구현 커밋 `82eacd7e`. sim은 `profile acc` + 13개 골든 테스트만 재실행했고 **긴 시나리오 3개는 재측정하지 않았다**(사용자 결정: 디바이스 측정으로 대체) — 따라서 prefill/decode 예산 ms/tok 행은 P3 값이 최신이다.

| predicate | P3 | P4 | 기준 | 판정 |
|---|---|---|---|---|
| `MATMUL_W8A16` 감소 (04 완료 기준 ≥ 3배) | acc per_call 2,445,246 | **683,954 (3.58×)** | ≥ 3× | PASS (acc per-shape + 디바이스로 판정, prefill512 미측정) |
| `MATMUL_W8A16` ≤ 레이어 사이클 25 % | 55.4 % / 36.7 % (P3 prefill512 / decode512) | 측정 불가 (prefill512 미실행) | ≤ 25 % | 판정 보류 — 디바이스 수치로 대체 (decode 59.5 M, prefill chunk 871.9 M) |
| 정확도 게이트 `profile acc` | 0.0914676 / 93.3662 | **0.077216 / 98.2637** | 0.1 / 0.1 | PASS |
| 골든 테스트 | 13/13 | **13/13** | 전부 | PASS |
| x86 PPL (387 tok) | 37.4738 | **37.1629** | 회귀 없음 | PASS (−0.83 %; ref 정의가 int16으로 바뀌었다) |
| 디바이스 PPL 갭 (512 tok) | +0.44 % (P3 skel) | **+0.89 %** | 계획서 ≤ 0.2 % | 기준 오설정 — 아래 격리 근거로 사용자 승인 |

계획서의 "≤ 0.2 %" 갭 기준은 잘못 설정된 것이었다: 이 프롬프트들의 PPL은 1e-3 미만 섭동에 ~1 % 반응하고(ref 자체가 int16 변경만으로 1.14 % 이동), 1-layer 격리에서 down **입력**(SILU_MUL 출력, 코드 불변)이 이미 rel-RMS 2.65 % 발산인 반면 DSP 자기 입력 기준 커널 출력은 fp16 1 ulp(rel-RMS 1.6e-4) 정확이다. 0.16 / 0.44 / 0.89 / 1.29 %는 한 밴드이며 원인은 상류 ATTN/W8A8 qf32 체인 — [07 ⑮](07-follow-ups.md).

### 디바이스 (P4) — S25 Ultra `R3CY10WM83Y`, v75 skel @`82eacd7e`, 2026-09-16

skel·이미지 md5를 매 실행 전 디바이스 사본과 대조(`push_if_changed` md5화, [07 ⑪](07-follow-ups.md) 완료).

| 표면 | P3 | P4 | 판정 |
|---|---|---|---|
| `run_device_test.sh` | RPC_TEST PASS | RPC_TEST PASS | PASS |
| `--eval` 387 tok PPL | 37.5336 / 134 (ref 37.4738, +0.16 %) | **37.6413 / 138** (ref 37.1629, +1.29 %) | 상동 밴드 |
| `--eval` 512 tok PPL | 33.5493 / 183 (ref 33.4011, +0.44 %) | **33.3132 / 189** (ref 33.0195, +0.89 %) | DSP −0.70 %, ref −1.14 % |
| `--eval` decode DSP | 55.9 M (26.8 ms), host 45.8 ms = 21.8 tok/s | **43.3 M** mean / median 43.65 M (20.7 ms), host 38.0 ms = **26.3 tok/s** | 1.29× |
| 생성 모드 decode (steady) | ~73 M (35 ms), host 76.4 ms = 13.1 tok/s | **59.5 M** (28.5 ms), host 31.2 ms = **32.0 tok/s** | DSP 1.23× (host 2.4×는 비교 불가 — P3 host 조건 미규명) |
| 128-tok prefill chunk @pos 0 | 1,413 M / 736 ms | **871.9 M / 478 ms** | 1.62× |

컨텍스트별 처리량(생성 모드 `--chunk 128 --steps 64`, host wall 기준; prefill tok/s = 프롬프트 토큰 ÷ 전체 prefill 청크 host wall, decode tok/s = 1000 ÷ 프롬프트 이후 63 step의 host wall 중앙값):

| context | prefill tok/s | DSP Mcyc/tok | decode tok/s | decode median host ms | decode median DSP Mcyc |
|---|---|---|---|---|---|
| 512 | 192.1 | 11 | 27.7 | 36.1 | 68.9 |
| 1024 | 118.2 | 18 | 22.9 | 43.7 | 85.2 |
| 4096 (`--max-seq 4224` 이미지, weights 599,180,800 B) | 27.3 | 77 | 9.4 | 106.1 | 216.9 |

M5 비교: 512 → 25.2 / 11.2 tok/s, 1024 → 23.2 / 11.3. 로그 `logs/hexagon/device_e2e_{eval,gen,gen512,gen1024,gen4096,eval512}_p4.log`, `device_e2e_eval512_p3skel.log`.

sim이 모델링하지 않는 것(문서에 명시): DDR 대역폭·지연, DMA 실제 처리량, L2 미스 비용, 디바이스 HVX unit 수. 따라서 **decode ≥ 30 tok/s는 이번 작업에서 증명되지 않는 목표**이며, 디바이스 복귀 후 아래 체크리스트로 닫는다.

## 디바이스 복귀 시 체크리스트 (범위 밖, 문서만)

1. `hexagon_e2e_test`/FARF에 `wp_size` 출력 추가해 디바이스 워커 수 기록 (메모리·가이드에 반영).
2. `HEX_ARCH=v75 build_skel.sh` → `run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t.i32 --chunk 128 --steps 64` (512·1024 tok) → §8.2 형식 표.
3. `--eval` PPL 19.98 ± 노이즈 밴드 (P4 W8A16 누적 순서 변경의 실측 게이트).
4. `find_divergence.py` 1-layer 이미지로 NO_DIVERGENCE.
5. 앱 `engine="htp"` prefill/decode tok/s.

## 문서 갱신 (단계마다, 그 단계의 마지막 Task)

`docs/backend_guide/HEXAGON.md`는 현재 상태의 권위 문서다. 각 단계는 자기가 바꾼 공개 사실을 그 단계 안에서 반영하고 `[docs]` 커밋으로 남긴다. P6는 전체 정합성 점검과 §8.3·§9 최종본만 담당한다.

| 단계 | HEXAGON.md 갱신 절 |
|---|---|
| P1 | §3(`test_profile.c`, `sim_model.{h,c}`, `summ_prof.py`), §5.2(`profile` 테스트, 시나리오·워커 인자, `SIM_TIMING`), §8.3 신규 "sim 프로파일 (M6)" — baseline 표(sim-derived 명시) |
| P2 | §1.4(ABI v4, `weight_layout` 필드, `n % 32` validator), §2.1(타일 정의·역인덱스·`nntr_htp_tile_off`), §2.4(`.hexcfg` `weight_layout=tiled32`, legacy 이미지 거부), §5.1(이미지 재생성 필요 안내) |
| P3 | **완료 (2026-09-16)**: §0 상태 단문, §1.4 표의 `MATMUL_W8A8`/`MATMUL_LOGITS`, §2.1(타일 스트립 읽기 — 브리지 문단 대체), §2.2(VTCM 워커 분할 — "공유 xq"는 구현에서 폐기), §3 파일 설명, §5.2(`test_matmul`/`test_matmul_dma`/`test_quant`, `summ_prof acc` 주의), §7 규칙 5(타일 커널)·6(벡터 quant), §8.3 P3 행 3개 + per-shape 표 + budget framing, §9(W8A16 최우선 + 디바이스 재측정) |
| P4 | §1.4 `MATMUL_W8A16` 설명(블로킹·VTCM), §7(fp32 누적 순서가 ref와 다름 → 디바이스 PPL 게이트), §8.3 전/후 행 |
| P5 | §1(ATTN 두 패스 구조), §2.2(KV append 블록 transpose), §7(softmax 마스크 규칙), §8.3 전/후 행 |
| P6 | §0 상태 문단(수치·"디바이스 미측정" 명시), §8.3 최종 표, §9(후속: [07](07-follow-ups.md) 항목, 디바이스 복귀 체크리스트), 전 절 교차 참조 점검 |

- `NOTICE`: ggml-hexagon에서 새로 가져온 코드가 있으면 그 단계에서 추가(기법만 차용했으면 변경 없음).

## 커밋 규칙

Task마다 1커밋, 메시지 형식은 메모리 `commit-message-format`(제목 + 바디 + Signed-off-by + Co-Authored-By). `docs/superpowers/`는 gitignore 대상이라 spec은 커밋하지 않는다.
