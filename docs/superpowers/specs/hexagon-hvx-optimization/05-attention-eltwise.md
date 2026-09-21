# P5 — ATTN 균등 분할·벡터화, decode 소형 op 분할

상위: [00-overview](00-overview.md) · 선행: [04](04-w8a16-down-kernel.md) · 다음: [06-verification](06-verification.md)

P1 프로파일이 정한 순서로 진행한다. 아래는 코드에서 확인된 항목과 설계이며, 프로파일 비중이 1% 미만인 항목은 건너뛰고 문서에 "측정상 불필요"로 남긴다.

## ① ATTN 워커 분할 — (kv head × GQA group × 토큰 블록)

현재: kv head(8) 단위 → 워커 4개면 2/2/2/2, 6개면 2/1/1/2/1/1(33% idle).
변경: 잡 단위를 `(h, g)` = 16 q-head로 하고, prefill(m ≥ 16)에서는 `(h, g, 토큰 블록 16)`까지 쪼개 `총잡 * wid / nw` 로 분배. KV append는 잡과 무관하게 **먼저 별도 패스**(아래 ②)로 끝내야 같은 kv head를 여러 워커가 읽어도 안전하다(현재는 append와 SDPA가 같은 워커에 묶여 있어 순서가 보장됨 → 두 `wp_run`으로 분리, 배리어 1회 추가).

## ② KV append 벡터화

- V: 이미 `memcpy`(행 연속). 유지.
- K^T: `kh[i*max_seq + pos+t] = krow[i]` 스칼라 stride 쓰기 m×128회. 변경: 토큰 블록 64개 단위로 K 행 64개(각 128 hf)를 레지스터에서 transpose(`Q6_W_vshuff`/`vdeal` 트리, 64×128 → 128×64)해 K^T 행 i에 64 위치를 **벡터 1개**로 store. m<64 꼬리(decode m=1 포함)는 스칼라 유지. prefill 청크 128 → 두 블록.

## ③ softmax 벡터화

현재 스칼라 3패스(scale+max, sub, sum). 변경: scale 곱·max를 `Q6_Vqf32_vmpy_VsfVsf`/`Q6_Vsf_vmax_VsfVsf`로 64 위치씩, 마지막 벡터만 수평 max 1회; `sub`는 exp 직전 벡터 패스로 흡수; sum은 `hvx_exp_f32` 출력에 벡터 누산 후 수평합 1회. L 꼬리(64 배수 초과분)는 `-inf` 마스크로 lane 무효화(`Q6_V_vmux_QVV`). exp는 기존 `hvx_exp_f32` 유지.

## ④ K^T 재사용 (prefill)

현재 query 행마다 K^T 스트립(128 벡터 × ceil(L/64))을 재로드. 변경: 잡 하나가 토큰 블록 TQ=4 행을 동시에 처리 — K^T 벡터 1 로드에 4 query의 `vsplat(q[t][i])` × mpyacc 4. 누산기 4 pair = 8 벡터. PV도 같은 방식으로 4행 동시(`a0,a1` × 4 = 16 벡터, 한도 안).

## ⑤ decode 소형 op의 열 분할

**P1 baseline 반영(2026-09-14)**: decode에서 RMSNORM 8.8K cyc/호출 중 4.3K, ADD 5.7K 중 4.3K가 fork-join 배리어(4332 cyc)다. 즉 이 op들의 실제 연산은 1.4K~4.5K cyc라 열 분할로 얻을 수 있는 상한이 호출당 수천 cyc, 토큰당 28 layer 기준 ≈ 0.1 ms 미만이다. 기본 결정: **⑤는 구현하지 않고 "측정상 불필요"로 §9에 기록**; P3·P4 이후 decode 프로파일에서 eltwise 합이 5%를 넘으면 재검토.

원안: RMSNORM/ADD/SILU_MUL/ROPE는 행(토큰) 분할이라 m=1이면 워커 1개. 변경: `m < nw`이면 열 방향(64-hf 벡터 단위)으로 분할. RMSNORM은 행 전체 sumsq가 필요하므로 (a) 워커별 부분 sumsq → 공유 배열 → 배리어 → 스케일 적용의 2단계, 또는 (b) m=1은 그냥 단일 워커 유지. **P1 프로파일에서 decode RMSNORM 비중을 보고 결정**; 기본은 (b)+ADD/SILU/ROPE만 열 분할.

## Task (프로파일 순서로 재배열 가능)

1. KV append 별도 패스 + (h,g,토큰블록) 분할 (`test_attn` m=1, 8, 128; pos 0, 512)
2. K^T append 64-블록 transpose (`test_attn` bit-exact — 데이터 이동만)
3. softmax 벡터화 (`test_attn` tolerance — exp·합 순서 변경)
4. K^T/PV 4-query 재사용 (`test_attn`)
5. 열 분할 eltwise (`test_eltwise`, `test_rope` m=1)
6. `profile` 전/후 표
7. **HEXAGON.md 갱신** ([06 표](06-verification.md) P5 행): §1 ATTN 두 패스 구조, §2.2 KV append 블록 transpose, §7 softmax 마스크 규칙, §8.3 전/후 행. 건너뛴 항목은 "측정상 불필요"로 §9에 기록. `[docs]` 커밋

## 완료 기준

- `profile prefill512` ATTN 사이클 baseline 대비 ≥ 2배 감소, 워커 4·6 모두에서 워커별 ATTN 사이클 편차 < 15%(sim 스레드별 Insns로 확인).
- `profile decode512`에서 op당 배리어 외 idle이 줄었는지는 스레드별 Insns 분포로 보고.
