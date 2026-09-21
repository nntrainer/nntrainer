# P1 — op별 프로파일과 baseline

상위: [00-overview](00-overview.md) · 다음: [02-tiled-weight-layout](02-tiled-weight-layout.md)

## 목적

이후 모든 단계가 같은 도구로 전/후를 비교할 수 있게, sim에서 op kind별 사이클을 표로 뽑는 계측을 만들고 현재 커널의 baseline을 기록한다. 커널은 건드리지 않는다.

## 변경 파일

| 파일 | 변경 |
|---|---|
| `htp/ops/htp_ops.h` | `struct htp_exec_ctx`에 `uint64_t prof_cycles[NNTR_HTP_OP_KIND_COUNT]; uint32_t prof_calls[NNTR_HTP_OP_KIND_COUNT];` 추가 |
| `htp/htp_graph.{h,c}` | `htp_graph_forward_upto` 루프에서 op마다 `HAP_perf_get_pcycles` 차를 kind별 누적. `htp_graph_profile_reset(g)` / `htp_graph_profile_get(g, cycles[], calls[])` DSP 내부 API. `htp_graph_init`에 워커 수 인자(0 = 자동) 추가 — 기존 시그니처는 래퍼로 유지 |
| `htp/worker_pool.{h,c}` | 변경 없음 (`wp_create(n)`가 이미 강제 지원) |
| `test/hexagon/sim/test_profile.c` | 신규 sim 테스트 (아래) |
| `test/hexagon/sim/sim_test_main.c` | `profile` 항목 추가, argv로 워커 수·시나리오 전달 |
| `tools/hexagon/run_sim_test.sh` | 인자 패스스루는 이미 지원. `SIM_TIMING=1`이면 `--timing` 추가 |

RPC ABI(`nntr_htp.idl`, `nntr_htp_common.h`)는 P1에서 바꾸지 않는다.

## sim 프로파일 테스트 (`test_profile`)

- 모델: qwen3 차원 그대로, `n_layers=2`, `vocab=4096`, `max_seq=2048`, `max_chunk=128`. 가중치는 결정적 의사난수 int8, 스케일 상수. 이미지·op-list는 `test_graph.c`와 같은 방식으로 sim 안에서 생성 (`lower_qwen3`는 호스트 C++라 sim에서 못 쓰므로, `test_graph.c`의 수동 op-list 빌더를 qwen3 차원으로 확장한 공용 헬퍼 `sim_model.{h,c}`로 분리).
- 시나리오 (argv):
  - `prefill0`: 128 tok @pos 0
  - `prefill512`: 128 tok 청크 4개로 pos 0..511 채운 뒤 **pos 512의 128 tok 청크만 측정**
  - `decode512`: 위 상태에서 n=1 @pos 512 를 8회 실행, 중앙값
- 워커 수: argv 두 번째 인자(`4`, `6`, `0`=자동). sim은 자동 = 4.
- 출력 형식 (파싱 가능하게 한 줄씩):

```
SIM_PROF scenario=<s> workers=<n> tokens=<m> pos=<p> total_pcycles=<c>
SIM_PROF kind=<name> calls=<k> pcycles=<c> per_call=<c/k>
...
SIM_PROF barrier_empty_x1000=<c>      # 빈 wp_run 1000회의 총 pcycles
SIM_PROF profile PASS
```

- `barrier_empty_x1000`: 빈 잡을 1000번 `wp_run`해서 fork-join 1회 비용을 잰다. 451 op × 이 값 = 토큰당 배리어 비용 추정치.

## 환산 규칙 (보고서 표 작성 시)

- 28 layer 환산: layer 루프 op(EMBED·최종 RMSNORM·LOGITS 제외)의 사이클 × 14.
- lm_head 환산: `MATMUL_LOGITS` 사이클 × (151936 / 4096). **`EMBED`는 환산하지 않는다**(gather라 비용이 n_tokens × hidden에만 비례, vocab 무관 — 2026-09-14 최종 리뷰에서 수정). 최종 RMSNORM이 layer op와 함께 ×14 되는 ~0.1% 오차는 허용.
- 배리어: `barrier_empty_x1000`/1000 × ops(= 1 + 16·layers + 2)를 모든 행에 같은 값으로 더한다(`summ_prof.py --barrier-cyc 4332`). 라인이 있는 로그와 없는 로그를 섞어 계산하지 않는다.
- 시간 환산: 디바이스 클럭 2.09 Gcyc/s (HEXAGON.md §8.2) 로 나눈 값을 "sim-derived ms"로 표기하고 **디바이스 측정치가 아님을 항상 병기**.
- `--timing` 유무를 표 머리에 적는다. 두 모드 수치를 섞지 않는다.

## Task

1. `prof_cycles` 누적 + `profile_get/reset` + 워커 수 인자 (실패 테스트: `test_profile`이 kind 표를 출력하고 합계가 `total_pcycles`의 ±1% 안에 들어야 PASS)
2. `sim_model` 헬퍼 + 세 시나리오 (실패 테스트: 시나리오별 `SIM_PROF` 라인과 ref 비교 — prefill0에서 8-토큰 prefill logits를 `ref_graph_forward`와 비교해 정확도 회귀도 같은 테스트에서 잡음. tolerance는 atol 0.1 / rtol 0.1 — [06](06-verification.md) 정확도 게이트 표 참조)
3. 배리어 측정 라인
4. **baseline 표**를 이 문서 아래 섹션에 기록. `--timing` 실행 시간이 30분 이내면 이후 기본값으로 채택, 아니면 non-timing pcycles를 사용하고 명시
5. **HEXAGON.md 갱신** ([06 표](06-verification.md) P1 행): §3 파일 목록, §5.2 `profile` 테스트 사용법, §8.3 baseline 표(sim-derived 명시). `[docs]` 커밋

## 완료 기준

- `HEX_ARCH=v75 ./tools/hexagon/run_sim_test.sh profile prefill512 0` 등 3 시나리오 × 워커 4(자동)가 PASS하고 표가 남는다. 워커 6 강제 실행은 **건너뜀**(2026-09-14 결정: sim은 HVX unit 4개라 6 워커는 lock 경합만 보여 디바이스 예측력이 없음; 강제 워커 수 기능은 `test_graph`의 2-워커 재실행으로 검증됨). P5 착수 시 ATTN 분할 불균형 근거가 필요하면 `prefill512 6` 한 번만 실행.
- 기존 13개 sim 테스트 PASS 유지.
- HEXAGON.md에 P1의 공개 사실이 반영되어 있다.

## Baseline (2026-09-14 측정)

- 측정: 2026-09-14, Hexagon SDK 6.0.0.2 / hexagon-clang 8.7.08, `HEX_ARCH=v75`, hexagon-sim **timing=off**(명령어 수 기반 pcycles; `--timing` 채택 여부는 아래 "timing 판정" 참조), 워커 4(자동), 모델 qwen3 차원 2-layer / vocab 4096 / max_seq 2048 (`test_profile.c` `QWEN3_2L`).
- 로그: `logs/hexagon/sim_prof_{prefill0,prefill512,decode512}_w4.log` (gitignore). 집계: `python3 tools/hexagon/summ_prof.py logs/hexagon/sim_prof_prefill0_w4.log logs/hexagon/sim_prof_prefill512_w4.log logs/hexagon/sim_prof_decode512_w4.log`.
- 정확도 게이트: `profile_prefill_acc STAT max_abs=0.080923 max_rel=165.186` (atol/rtol 0.1 PASS; fill_tokens UB 수정 후 토큰 id가 바뀐 재실행. 수정 전 토큰(절반이 id 0)에서는 max_abs=0.0798156). **2026-09-15 (M6 P2) 재-baseline**: `profile_prefill_acc STAT max_abs=0.0914676 max_rel=93.3662`(atol/rtol 0.1 PASS 유지) — P2에서 `sim_model.c`의 채움이 동일한 균등 난수 바이트를 타일 인덱스로 읽도록 바뀌어(row-major → tiled32) P1과 같은 시드의 순열된 값이 되었기 때문이며, 정수 경로는 커널과 스칼라 참조가 여전히 bit-exact로 일치한다(사용자 결정, 값은 위 STAT과 별도 재실행 없이 그대로 채택).
- non-timing sim은 결정적: 같은 입력으로 prefill0을 두 번 돌려 `total_pcycles`가 비트 단위로 같았다. 토큰 id를 바꾸면(fill_tokens 수정) total이 1027368585 → 1028601630(+0.12%)로 미세하게 달라진다 — 데이터 의존 분기(스칼라 quant의 amax 비교 등). 전/후 비교는 **같은 바이너리의 토큰 생성 코드를 유지한 채** 수행하고, 0.2% 미만 차이는 노이즈로 본다.
- 실행 시간(호스트 wall-clock): prefill0 23m48s, prefill512 41m35s, decode512 34m42s.

### raw (sim pcycles, 2-layer / vocab 4096 그대로)

| scenario | workers | timing | tokens | total | EMBED | RMSNORM | MATMUL_W8A8 | ROPE | ATTN | SILU_MUL | ADD | MATMUL_LOGITS | MATMUL_W8A16 | barrier/op |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prefill0 | 4 | off | 128 | 1028601630 | 1479141 | 1157993 | 939704687 | 298476 | 6102014 | 1291956 | 83238 | 601029 | 77882241 | 4332 |
| prefill512 | 4 | off | 128 | 1059812175 | 1479915 | 1158034 | 938298568 | 298476 | 38715038 | 1292706 | 85885 | 600504 | 77882194 | (4332)* |
| decode512 | 4 | off | 1 | 9265863 | 50160 | 78894 | 7533759 | 10344 | 288128 | 50531 | 22739 | 600837 | 632788 | (4332)* |
| prefill512 | 6 | — | 미실행 (P5에서 필요 시; sim은 HVX unit 4개) | | | | | | | | | | | |

\* prefill512/decode512 로그는 Task 5 이전에 생성되어 `barrier_empty_x1000` 라인이 없다. 배리어 비용은 시나리오와 무관하므로 prefill0 값(4332 cyc/op)을 적용한다.

### scaled to 28 layers / vocab 151936 (sim-derived ms @2.09 GHz, **디바이스 미측정**; `summ_prof.py --barrier-cyc 4332`, EMBED 미환산 — 2026-09-14 최종 리뷰 후 재계산. 나머지 % 열은 소수 첫째 자리에서 변화 없음)

| scenario | workers | timing | ms/tok | tok/s | barrier ms/tok | EMBED | RMSNORM | MATMUL_W8A8 | ROPE | ATTN | SILU_MUL | ADD | MATMUL_LOGITS | MATMUL_W8A16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prefill0 | 4 | off | 53.82 | 18.6 | 0.007 | 0.01% | 0.1% | 91.0% | 0.0% | 0.6% | 0.1% | 0.0% | 0.2% | 7.6% |
| prefill512 | 4 | off | 55.45 | 18.0 | 0.007 | 0.01% | 0.1% | 88.2% | 0.0% | 3.6% | 0.1% | 0.0% | 0.1% | 7.3% |
| decode512 | 4 | off | 69.35 | 14.4 | 0.93 | 0.04% | 0.8% | 72.8% | 0.1% | 2.8% | 0.5% | 0.2% | 15.4% | 6.1% |

(디바이스 실측 §8.2: prefill 25 tok/s, decode 11–13 tok/s. sim-derived 값이 같은 자릿수라 sim 비율은 병목 순위 판단에 쓸 수 있다.)

### 해석 (P2~P5 우선순위 근거)

1. **MATMUL_W8A8이 prefill 사이클의 91%, decode의 73%.** 2-layer × 128 tok의 W8A8 연산량 ≈ 3.2 GMAC가 938M cyc → **≈ 3.4 MAC/cycle**. HVX 4 unit의 vrmpy 피크(≈ 2048 MAC/cycle)의 0.2% 수준. m=1 decode도 같은 효율(q proj 2 MMAC → 628K cyc)이라 **현재 decode는 대역폭 한계가 아니라 커널 효율 한계**다. P2/P3 타일 커널이 최우선.
2. **MATMUL_W8A16(down)** 7.6% / 6.1%. P4 대상. W8A8이 타일화되면 상대 비중이 커진다.
3. **MATMUL_LOGITS**는 풀 vocab 환산 시 decode의 15.4%로 2위. EMBED/lm_head 타일링(P3 포함)이 decode에 중요.
4. **ATTN**은 L에 선형(6.1M @L=128 → 38.7M @L=640). prefill512에서 3.6%, decode 2.8%. P5는 P3·P4 이후에 의미가 있다.
5. **배리어** 4332 cyc/op → 토큰당 1.95M cyc = 0.93 ms(sim-derived), 06 predicate(≤ 3 ms) 통과. 퓨전([07 ②](07-follow-ups.md)) 우선순위 상향 근거 없음. 단 decode 소형 op는 배리어가 지배적(RMSNORM 8.8K 중 4.3K, ADD 5.7K 중 4.3K) → P5 ⑤ "decode 열 분할"의 기대 효과는 배리어 하한에 막혀 작다.
6. **스레드 균형(sim Insns)**: `summ_prof.py` 스레드 표 — 워커 T1–T4 Insns가 prefill0 277M/126M/206M/182M(spread 76%), prefill512 1528M/706M/1069M/821M(80%), decode512 1250M/555M/851M/664M(84%). 단 sim의 스레드별 Insns는 측정 창이 아니라 **전체 실행**(가중치 채움·ref·채우기 청크 포함)이고 DMA 큐 busy-wait도 명령어로 잡히므로 이 값은 불균형의 **상한**일 뿐이다. 06 predicate(편차 < 15%)는 P3에서 `HTP_MM_NO_VTCM` 우회 실행과 비교해 DMA 대기분을 분리한 뒤 판단한다. — **P3 결과(2026-09-16)**: spread가 prefill512·decode512 모두 16 %로 떨어졌으나 여전히 상한이라 판정 보류. [06 P3 판정](06-verification.md) 참조.

### timing 판정

`SIM_TIMING=1 profile prefill0`(2026-09-14, 사용자 실행): 30분을 넘겨도 `SIM_PROF pool` 라인 이전(초기화·가중치 채움)에 머물러 **중단**. non-timing prefill0가 24분이므로 timing 모드는 수 시간 규모로 추정. **결정: P1~P6 baseline과 전/후 비교는 모두 timing=off(명령어 수 기반 pcycles)로 수행**하고 표 머리에 `timing=off`를 명시한다. timing=off는 결정적이라 비교 노이즈가 없다는 장점이 있고, 대신 메모리 지연·DMA 실제 처리량은 반영되지 않는다(06 참조).
