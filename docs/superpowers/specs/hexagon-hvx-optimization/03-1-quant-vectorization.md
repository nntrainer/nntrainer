# P3-1 — 활성화 quant HVX 벡터화 (P3 decode 기준 통과용 추가 범위)

상위: [00-overview](00-overview.md) · 선행: [03-w8a8-tiled-kernel](03-w8a8-tiled-kernel.md) Task 1~3 (커밋 `cd5ac6b3`, `49b95816`, `c777f89a`) · 게이트: [06-verification](06-verification.md)

작성 2026-09-16. P3 Task 4 측정에서 decode512 predicate가 미달했고 원인이 스칼라 quant 한 행이라 확인되어, 브레인스토밍 때 후속(07 ⑧)으로 미뤘던 벡터화를 P3 안으로 되돌린다. 사용자 결정 2026-09-16.

## 왜 지금 하는가 (측정 근거)

| 항목 | 값 | 출처 |
|---|---|---|
| 스칼라 `htp_quant_row_fp16` 1행(k=1024) 비용 | ≈ 345K pcycles | Task 3 acc 델타 2.76M ÷ 8행 |
| decode512 W8A8 op 1회 (m=1) | ≈ 510K, 그중 quant ≈ 345K | `logs/hexagon/sim_prof_decode512_w4_p3.log` |
| 같은 가중치 크기의 W8A16 op (quant 없음) | 316K | 같은 로그 |
| decode512 W8A8+LOGITS | 7,396,241 (기준 ≤ 2,711,532) | Task 4 판정 |
| prefill512 W8A8 | 226,505,764 (기준 통과, 그중 quant ≈ 130M 추정) | 같은 판정 |

스칼라 코어에 fp16 연산이 없어 원소마다 소프트웨어 변환(`(float)x[i]`)과 `lrintf`가 들어간다. m=1이면 워커 하나가 행 전체를 혼자 돌리고 나머지 3개는 배리어에서 기다린다.

## 목표

`htp_quant_row_fp16(const __fp16 *x, int8_t *q, uint32_t k) -> float`의 **시그니처·호출자·의미를 그대로** 두고 본체만 HVX로 바꾼다. 호출자 `quant_worker`(W8A8)와 `hvx_op_matmul_logits`(인라인 1행)는 변경하지 않는다.

출력 계약 (**2026-09-16 사용자 결정 (b)로 개정**): 반환 scale은 스칼라 참조 `ref_quant_row`(`test/hexagon/sim/ref_ops.c`)와 동일. `q[0..k)`는 ref와 동일하되, **원소의 약 1e-5 비율(측정 3.6e-5)에서 ±1 LSB 차이를 허용**한다. 원문의 "바이트 동일" 계약은 이 하드웨어에서 불가능함이 확인됐다(아래 "정확성 위험" 참조): §7이 허용하는 `Vqf32_vmpy`는 Qfloat의 Von Neumann 반올림(V75 HVX PRM §4.6, 가수 LSB가 implicit 1)으로 곱 결과를 sf ulp의 2배 격자로 내리고 half-ulp를 더해 내보내므로, 어떤 반올림 코드도 `lrintf(fl32(x*inv))`를 모든 입력에서 재현할 수 없다. 테스트 (a)~(e)(격자 위 데이터, tie, 0행)는 memcmp 정확 일치를 유지하고, 비격자 무작위 행 (f)에서 "차이는 전부 ±1, 비율 ≤ 2e-4"를 요구하며 비율을 STAT 줄로 출력한다. 커널 출력 STAT, 13개 테스트 STAT, acc STAT(0.0914676 / 93.3662)은 실측에서 모두 불변이었다.

## 수치 규칙 (HEXAGON.md §7 준수)

스칼라 정의:
```
amax = max_i |x[i]|                 (fp16 → fp32 정확 변환 뒤 비교)
inv  = amax > 0 ? 127.f / amax : 0  (fp32 나눗셈 1회)
q[i] = (int8)lrintf((float)x[i] * inv)   (fp32 IEEE 곱 → RNE 반올림)
return amax / 127.f
```

벡터 대응 (qf 포맷 연산만; IEEE-format HVX 명령 `Q6_Vsf_vmpy_VsfVsf`, `Q6_Vw_equals_Vsf`는 §7 규칙 1에 따라 쓰지 않는다. spec 03의 "`Q6_Vw_equals_Vsf`(RNE)" 문장은 이 문서가 대체한다):

1. **absmax**: fp16 비트에서 부호를 지우고(`hvx_vec_abs_f16`, `& 0x7fff`) `Q6_Vuh_vmax_VuhVuh`로 누적. 양수 fp16의 비트 패턴은 값과 단조이므로 정수 max가 곧 |x| max이다. 벡터 내부 리덕션은 `Q6_V_vror_VR` 2,4,…,64바이트 + vmax 6단, lane 0을 `hvx_vec_get_f16`으로 꺼내 `(float)`로 올린다(정확). `inv`와 반환값은 스칼라로 1회 계산 — 나눗셈은 스칼라와 동일 연산.
2. **fp16 → fp32**: `hvx_vec_f16_to_f32(v)` (`Wqf32_vmpy_VhfVhf(vshuff(v), 1.0)` → `Vsf_equals_Vqf32`). ×1.0이라 정확. lo = 원소 0..31, hi = 32..63.
3. **곱**: `Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x32, splat(inv)))`. 스칼라의 fp32 곱 1회에 대응.
4. **RNE 반올림 → int32** (구현에서 개정됨): 원안(1.5·2^23 매직 더하기로 하드웨어 RNE 유도)은 sim에서 실패했다 — qf32 덧셈은 Von Neumann 반올림이라 정수 비트가 사라져 0행이 전부 1이 됐다. 채택안: 매직 1.5·2^8(`0x43C00000`, 384.f)을 `Q6_Vqf32_vadd_VsfVsf` → `Q6_Vsf_equals_Vqf32`로 더하면 p∈(−128,128)이 [2^8, 2^9)에 들어가 sf ulp = 2^-15이고, `bits − magic_bits = 2·floor(p·2^14) + 1`이 된다. 이를 0 방향으로 1비트 시프트(`(d asr 1) + (d lsr 31)`; 음수 tie가 한 칸 내려가는 것을 상쇄)해 `A = sign(p)·floor(|p|·2^14)`을 얻고, 정수 연산 `(A + (2^13−1) + ((A asr 14) & 1)) asr 14`로 ties-to-even. 리뷰어 검산: 0, ±0.5→0, ±1.5→±2, ±2.5→±2, ±127 모두 `lrintf`와 일치. 해상도가 2^-14이므로 참값이 tie 위 2^-14 안에 있는 원소만 even으로 가서 ±1 차이(계약 참조).
5. **int32 → int8**: `Q6_Vh_vpack_VwVw_sat(hi, lo)` → `Q6_Vb_vpack_VhVh_sat(hi_h, lo_h)`. 값은 이미 [-127,127]이라 saturate는 안전장치. 128원소(입력 fp16 벡터 2개)마다 int8 벡터 1개 저장.

루프 단위 128원소; `k % 128 == 0`은 validator가 보장한다(`nntr_htp_oplist_validate`, EMBED 포함). 로드/스토어는 `hvx_vmemu`(ACT 행 오프셋과 테스트 버퍼의 정렬을 가정하지 않는다).

### 정확성 위험과 판정 규칙

3번 곱에서 `Vqf32_vmpy_VsfVsf → Vsf_equals_Vqf32`가 IEEE fp32 곱과 1 ulp 다를 가능성이 있다(Task 1 에필로그에서 W8A8 m8 max_abs가 0 → fp16 1 ulp로 움직인 것과 같은 원인). 그 차이가 반올림 경계(.5)를 넘기는 원소만 q가 ±1 달라진다. **테스트는 이 가능성을 허용하지 않는다** — memcmp로 바이트 일치를 강제한다.

- 일치하면: 계약 그대로, 아래 게이트로 진행.
- 불일치하면: 구현 서브에이전트는 테스트를 완화하지 말고 **불일치 원소 수·위치·차이(±1인지)** 를 보고한다. 그다음은 사용자 결정: (a) 스칼라 정의를 유지하고 벡터 곱을 정확히 맞출 다른 방법을 찾는다, (b) 벡터 quant를 커널 정의로 받아들이고 `ref_quant_row`는 스칼라로 두되 quant 테스트를 "±1 이하, 비율 보고"로 바꾸며 acc STAT를 재기준한다. (b)는 수치 변경이라 HEXAGON.md §7과 06에 기록이 필요하다.

## 테스트 (`test/hexagon/sim/test_quant.c`, sim `quant`)

모든 케이스에서 `htp_quant_row_fp16` 결과 `q`와 `ref_quant_row` 결과 `q_ref`를 `memcmp`, scale은 `==`. 하나라도 다르면 첫 불일치 인덱스·got·ref 출력 후 FAIL.

| 케이스 | 목적 |
|---|---|
| (a) 무작위 행 16개, k=1024, 진폭 8 / 0.01 / 1000 / 1e-4 순환 | 일반 경로, 지수 범위 |
| (b) 전부 0 | scale 0, q 전부 0 (inv=0 분기) |
| (c) tie 행: `x[0]=127`(→ inv = 1.0 정확), `x[1..]` = ±(n+0.5), n=0..126 반복 | ties-to-even이 `lrintf`와 같은지 직접 검증 (2.5→2, 3.5→4, −2.5→−2) |
| (d) k=3072 무작위 1행, k=128 무작위 1행 | gate/up 폭과 최소 폭, 루프 경계 |
| (e) 음수만 있는 행, amax가 마지막 원소인 행 | absmax 리덕션 lane 누락 |

RED: 벡터 본체를 넣기 전에 (c)·(e)를 추가해도 스칼라 본체로는 PASS여야 한다(테스트의 정당성 확인). 벡터 본체로 바꾼 뒤 모든 케이스 PASS가 GREEN.

## 게이트 (06 규칙, sim 컴파일 소스 변경)

1. sim `quant matmul matmul_dma logits graph` PASS, **STAT 전부 Task 3 값과 자리수까지 동일** (`matmul_w8a8_m1` 0/0, `m8` 0.0078125/0.000788644, `m7` 0/0, `m128` 0.03125/0.000749625, `matmul_dma_ref_*` 0.0078125/0.000714286, `logits` 7.62939e-06/2.36832e-07, `graph_prefill` 0.020095/16.4921, `graph_decode` 0.0241753/10.7802, `graph_partial_attn` 0.000488281/0.390625, `graph_prefill_2workers` 0.020095/16.4921).
2. `profile acc` STAT `0.0914676 / 93.3662` 동일, `profile PASS`; per-shape W8A8 per_call이 Task 3(1024→3072 1,089,729; 1024→1024 983,991; 2048→1024 1,954,859; 1024→2048 1,036,280; LOGITS 519,780) 대비 감소.
3. 커밋 후 사용자 실행: `profile decode512`, `profile prefill512` 재측정 → Task 4 predicate 재판정.

## 완료 기준

- decode512 `MATMUL_W8A8 + MATMUL_LOGITS` ≤ 2,711,532 (P1 ÷ 3). 추정치 ≈ 12 × ~165K + ~175K ≈ 2.2M.
- prefill512 `MATMUL_W8A8` ≤ 234,574,642 유지(추정 ≈ 100M).
- 정확도: 위 STAT 전부 불변.

## 범위 밖

- 양자화 방식 변경(per-block, 비대칭 등) — 07 ⑤.
- `ref_quant_row` 변경 — 사용자 결정 (b)일 때만.
- LOGITS quant를 워커에 나누는 것 — 1행이라 의미 없음.
