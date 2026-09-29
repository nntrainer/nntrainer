# 후속 작업 (이번 범위 외)

상위: [00-overview](00-overview.md)

M6에서 만들지 않지만 설계 근거를 남긴다. ①②는 decode 대역폭·배리어 계열로 **디바이스 측정이 가능해진 뒤** 별 마일스톤으로, ③④는 환경 작업, ⑤⑥은 수치가 바뀌는 선택이라 사용자 결정이 필요하다.

## ① cross-op weight prefetch (decode 대역폭) — **완료 (#25, 디바이스 확인 2026-09-18)**

- 문제: 각 matmul이 자기 첫 스트립 DMA를 op 진입 후 kick하고 기다린다. ATTN·RMSNORM·eltwise가 도는 동안 DDR은 idle.
- 설계: `htp_graph_forward_upto`가 op i를 실행하기 전에 op i+1..i+2 중 matmul의 첫 청크(워커별 buf[0])를 미리 kick. 워커 슬랩의 buf[0]을 "다음 op 전용"으로 예약하면 현재 op의 더블버퍼와 충돌하지 않는다(현재 op는 buf[1], buf[2] 사용 → 슬랩을 3분할). `dma_queue`를 op 수명이 아니라 그래프 수명으로 승격(현재는 op마다 memalign/free).
- 기대: decode에서 matmul 사이 non-matmul 시간(P1 프로파일의 ATTN+RMSNORM+ROPE+ADD+SILU 합)만큼 DMA가 앞서 감. sim에서는 효과 측정 불가.
- **구현 (#25, 2026-09-18, `docs/plans/25-decode-prefetch.md`)**: 설계는 위 스케치와 다르다 — 슬랩 3분할·buf[0] 예약 대신 **free-buffer handover**: 워커별 `dma_queue`를 그래프 수명으로 승격(`htp_graph_dma_init`, 파괴 전 워커에서 flush)하고, 마지막 청크 DMA를 pop한 뒤 비어 있는 반쪽 슬랩에 다음 타일 matmul(`ctx.next_mm`, init 때 만든 `next_mm[]` 표, 부분 실행 한계로 클립)의 청크 0을 kick해 `ctx.pf[wid]`에 기록; 다음 op 진입 시 포인터 동일성으로 hit, 불일치·DDR fallback은 flush. 3분할은 decode의 1청크 op를 2청크로 만들고 LOGITS 아래 VTCM 1/3을 놀리므로 기각. 출력은 구성상 bit-identical (`test_matmul_dma` (a)–(d), `graph` STAT 불변). 디바이스 판정은 `docs/measurements/25-decode-prefetch.md` (A `HTP_MM_NO_PREFETCH` 대 B, pass 2 Mcyc 비, 게이트 ≤ 0.90).
- **완료 (2026-09-18, S25 Ultra `R3CY10WM83Y`, v79, SDK 6.4.0.2, `docs/measurements/25-decode-prefetch.md` pass 2)**: B(prefetch) / A(`HTP_MM_NO_PREFETCH`) = 54.575 / 61.735 Mcyc = **0.884**(−11.6 %; `mm8` 24.35 → 16.05 Mcyc, −34 %, stream-only skel의 순수 DMA 대기보다 0.5 Mcyc 아래 — W8A8 MAC은 스트림 뒤에 완전히 숨음). A는 #35 B1 +0.9 %(env PASS), `--eval` 41.4947 / 162와 `E2E gen`이 A/B 양 pass에서 byte-identical(G3 PASS). **prefetch가 기본값**(`HTP_MM_NO_PREFETCH`는 opt-out). 1024 / 4096에서는 B = C(stream-only) 0.1 % / 1.2 % 안 — 그 위 decode 레버는 ATTN(#58). W8 스트림 상한: `mm8`+`mm16`+`lg` = 32.2 Mcyc = 18.5 ms @1743 MHz = 32.3 GB/s → **54 tok/s @512**(HEXAGON_BENCHMARK.md 목표 셀, #51 입력).

## ② op 퓨전 (배리어 수 감소, ABI v5)

- 후보: `RMSNORM → MATMUL_W8A8(q/k/v 셋)` (quant를 norm 출력에서 바로), `SILU_MUL → MATMUL_W8A16`, `MATMUL → ADD`(residual을 store 시점에 합산). q/k/v와 gate/up은 같은 입력이라 N 방향 이어붙여 matmul 1회로 융합 가능(호스트 lowering 변경, 가중치 이미지 순서 변경).
- 비용: `lower_qwen3` op 시퀀스·`ref_graph_forward`·sim `graph` 테스트·`find_divergence.py`의 op 인덱스 전부 변경. ABI v5.
- 판단 근거: P1의 `barrier_empty_x1000` × 451이 토큰당 3 ms(sim-derived)를 넘거나, 디바이스 decode에서 non-matmul 구간이 20 %를 넘을 때.

## ③ HAP_power 투표 (앱)

M5에서 하네스는 이미 최고 클럭(2.09 Gcyc/s)이었지만, 앱은 토큰 사이에 샘플링·출력으로 idle 갭이 있어 클럭이 내려갈 수 있다. `HexagonRunner::init`에 compute apptype + DCVS performance 모드 투표를 넣고 앱 tok/s로 판정. 디바이스 필요.

## ④ v79-native skel — **사용자 요청(2026-09-16): v79 skel로 변경할 것**

HEXAGON.md §7. IEEE hf·qf32 체인 오동작 원인 규명 후 `HEX_ARCH=v79` 기본화. 타일 커널(P3)·int16 lanewise 커널(P4)은 qf 포맷 규칙만 쓰므로 영향 없음. 전제: ⑭ SDK 상향(v79 QuRT sim 이미지는 6.0.0.2에 없어 sim 게이트가 v75에 묶여 있음). 작업 순서: (1) ⑭ 후 v79 sim에서 13개 + `profile acc` 재실행, (2) `HEX_ARCH=v79 build_skel.sh` → 디바이스 e2e PPL이 v75 skel과 일치하는지, (3) `hvx-base.h`의 `__HVX_ARCH__ >= 79` 분기(`Wsf_vmpyacc`, `Vsf_vadd/vmpy` IEEE 경로)가 §7 규칙 1의 실리콘 오동작을 다시 밟지 않도록 qf 경로 강제 여부 결정, (4) 기본값 v79로 전환 + HEXAGON.md §5.3/§7 갱신.

- **(1) 결과 (#23, 2026-09-17, SDK 6.4.0.2 / hexagon-clang 19.0.04, v79 sim)**: 13개 중 9 PASS / 4 FAIL. FAIL = `quant`(tie 행 `x=-1.5`가 `-2` 대신 `-1`, `rand0`/`rand15`/`k3072`에 ±1 원소 1개씩, `quant_generic 13/65536`, `quant16_generic 229/25600`), `matmul`(`w8a8_m1` max_abs 0.03125 / max_rel 0.045677), `matmul_dma`(`ref_4096k` 0.0625 / 0.273998), `logits`(0.0195827 / 0.273973). PASS = smoke pool exp rmsnorm rope eltwise embed attn graph. 즉 실패면은 qf 전용 코드인 `quant_row`(`hvx-quant.h`)와 W8A8 에필로그 `mm_tile`(`hvx-matmul.c`)이고 — 두 파일 모두 `__HVX_ARCH__` 분기가 없다, 양자화기의 유일한 아키텍처 의존 명령은 `hvx-base.h`의 fp16→fp32 확장뿐 — IEEE 헬퍼를 실제로 타는 attn/eltwise는 sim에서 통과. `profile acc` v79: PASS, STAT `max_abs=0.0899355 max_rel=49.2742`(v75 값에서 이동, 0.1 밴드 안), v79 sim은 HVX 6유닛이라 `workers=6`(`total_pcycles` 6,105,728; v75 4워커와 비교 불가). (2)는 `docs/measurements/23-sdk64-baseline.md` 변형 B로 진행.
- **(2) 결과 (#23, 2026-09-17, S25 Ultra, v79 skel = 워크스테이션 SDK 6.4.0.1 / hexagon-clang 19.0.04 빌드)**: 512/1024/4096 여섯 실행 모두 hang·DSP 재시작·FARF fatal 없음. `--eval` PPL/top-1이 v75 skel과 일치(512 41.2623/161 vs 40.7596/162, 1024 5.9509/692 vs 5.9141/698, 4096 1.5687/3758 vs 1.5662/3763). 속도는 v75보다 빠름 — prefill 207.4 / 131.2 / 34.4 tok/s(+9.7 / +12.0 / +26.9 %), decode DSP Mcyc 60.1 / 76.5 / 177.6(−6.5 / −9.1 / −16.6 %); 512 decode 호스트 tok/s는 31.4 vs 31.9로 평탄(호스트 경로 지배). 즉 §7 규칙 1의 실리콘 오동작(inf/0)은 19.0.04 툴체인에서는 재현되지 않았고, sim의 4개 FAIL(±1 LSB)은 디바이스에서 보이지 않음. (3)의 질문은 "qf 경로 강제"에서 "sim 4개 FAIL을 통과/바운드시킨 뒤 기본값 전환"으로 바뀜(HEXAGON.md §7 규칙 1 갱신, §9). #35(v75 기본 고정 + IEEE 헬퍼 빌드 실패)는 이 숫자를 보고 다시 판단.

## ⑤ down_proj 레이아웃 단일화 (속도 문제는 P4가 해소)

- **P4 갱신(2026-09-16)**: 속도 문제는 블록 양자화가 아니라 **per-token int16 + int32 lane-wise 누적**으로 해소됐다(acc per_call 3.58×, [04 측정 기록](04-w8a16-down-kernel.md#측정-기록-2026-09-16)). 정확도도 개선 방향(x86 PPL −0.83 %)이라 int8 블록 양자화로 갈 이유가 없어졌다. 남은 것은 **레이아웃 단일화**뿐이다.
- 남은 내용: 이미지에서 `down`도 타일링해 row-major/tiled32 이중 레이아웃을 없앤다. int16 커널은 행 단위로 128 B씩 읽으므로 타일 인덱스로 바꾸는 것은 주소 계산 변경에 가깝다. 이득은 속도가 아니라 packer·validator·문서의 단순화(HEXAGON.md §2.1 "`down` stays row-major").
- 원안(참고): SwiGLU 출력을 128개 단위 스케일로 int8화(ggml q8_0) → 타일 vrmpy 경로. per-token int8이 +6 % PPL이었고 int16은 −0.83 %라, 이 방향은 수치 손해를 감수해야 정당화된다 — 현재 근거 없음.

## ⑥ 가중치 부호 오프셋 + 스칼라 vrmpy (명령수 감소)

- `Q6_Vw_vrmpyacc_VwVubRb`(unsigned 벡터 × signed 스칼라 4B)를 쓰면 vsplat이 사라져 벡터당 vload + vrmpy만 남는다. 조건: 가중치를 `w + 128`(uint8)로 저장하고 결과에서 `128 × Σ_k xq[t][k]`(토큰당 스칼라, 모든 행 공통)를 뺀다. 정수 경로라 여전히 bit-exact.
- 배제 이유(M6): 저장 타입이 int8 → uint8-with-offset으로 바뀌어 "데이터 타입 불변" 결정과 충돌. vsplat이 vrmpy와 다른 슬롯에서 공발행되면 이득이 없을 수 있음 → P3 프로파일에서 벡터당 packet 수가 1을 크게 넘을 때만 재검토.

## ⑦ decode M>1 (speculative / batch)

타일 커널의 TB 블로킹이 그대로 적용되므로 커널 변경 없음. 앱·RPC 계약(`batch_size > 1` fallback) 쪽 작업.

## ⑧ quant 후속 (P3에서 벡터화 완료, 남은 항목)

당초 "벡터화 보류" 항목이었으나 P3 Task 3b에서 [03-1](03-1-quant-vectorization.md)로 구현했다. 남은 것:

- **디바이스 tie-row 검증(S25 Ultra)**: 음수 tie 경로가 Qfloat 반올림의 부호 대칭성에 기대는데 V75 HVX PRM이 명시하지 않는다. sim은 통과. `test_quant` (c) 케이스를 디바이스에서 한 번 돌린다.
- **NaN 처리 차이**: 벡터 absmax는 NaN을 absmax 후보로 세고 스칼라 `ref_quant_row`는 건너뛴다. 실모델 활성화에 NaN이 없어 방치했다.
- **정확 일치가 필요해지면**: 정수 가수 곱 커널(fp16 가수를 정수로 꺼내 `Vw` 곱 → 시프트). 리뷰어 추정 ~20 vector ops / 32 lane으로 여전히 §7 한계(qf 포맷 연산만) 안에 들어간다. 현재 ±1 비율 3.6e-5(모델, sim 실측 0/65536)가 문제되지 않는 한 불필요.

## ⑨ W8A8 DMA 청크 크기 — **완료 (#25, 2026-09-18: max-fit 유지)**

`mm_worker_vtcm`의 청크는 슬랩 절반에 맞는 최대 타일 수(max-fit)라 decode(m=1, n=1024)는 워커당 청크 1개 = DMA 후 계산의 직렬 구조다. 스트립(32행) 단위 등 작은 청크와의 비교는 DDR 지연을 모델링하는 디바이스에서만 가능 — ① cross-op prefetch와 함께 판정. P3 이후 decode는 W8A16 지배(37 %)라 우선순위는 낮다.

- **#25 (2026-09-18)**: `HTP_MM_CHUNK_ROWS=<n>` 빌드 플래그(0 = max-fit)로 변형 D(64행)를 같은 handoff에서 ①의 B와 비교(D 규칙: 512 decode Mcyc 2 % 이상 낮고 prefill이 B 이상이면 기본값).
- **완료 (2026-09-18, 같은 handoff pass 2)**: D(64행) 59.993 vs B(max-fit) 54.575 Mcyc = **+9.9 %**, prefill도 낮음(181.4 vs 191.1 tok/s); D의 `mm8` 20.66 vs B 16.05 Mcyc — 작은 청크는 decode의 1청크 op를 여러 청크로 쪼개 청크마다 DMA 지연을 낸다. **max-fit 유지**, `HTP_MM_CHUNK_ROWS`는 기본값 없는 측정 플래그로 남김.

## ⑩ `MM_TB` 디바이스 재측정 — **#57로 이관 (2026-09-18)**

sim에서 `MM_TB 4`의 per-shape W8A8 이득은 1.3~3.7 %에 그쳤다(시뮬레이터가 가중치 대역폭을 제대로 과금하지 않기 때문). 디바이스에서 `MM_TB` 1/2/4/8을 재측정해 기본값을 확정한다.

- **#25 (2026-09-18)**: `MM_TB`·`MM16_TB`를 `#ifndef` 기본값으로 바꿔 `HEX_EXTRA_CFLAGS=-DMM_TB=8u`로 변형을 빌드한다. `MM_TB`는 m=1에서 inert(`mm_tiles`가 m < MM_TB이면 tb=1 꼬리만 탄다)라 prefill 전용 노브; H1(prefetch) 판정 뒤 H2 handoff(`MM_TB 2/8`, `MM16_TB 1/4`, 512 토큰, pass 2 Mcyc 최저값, < 2 %면 현행 유지)로 기본값 확정. **`MM16_R`은 4로 고정**(static check): `mm16_block` 에필로그의 shuffle 트리가 정확히 4행만 접으므로 다른 값은 컴파일되지만 잘못된 행을 낸다(code review, 2026-09-18) — 스윕하려면 에필로그 일반화(값별 sim 게이트)가 먼저다.
- **#57로 이관 (supervisor 2026-09-18)**: #25의 H2 sweep은 쓰지 않았다 — `MM_TB`·`MM16_TB`는 m=1에서 inert이고 decode `mm8`은 이미 순수 DMA 대기 아래라 H2는 prefill(m=128)만 움직일 수 있는데, prefill은 목표에서 5×이고 그 레버는 HMX다. `MM_TB 2/8` + `MM16_TB 1/4` sweep과 `MM16_R` 에필로그 일반화(2/4/8)는 **#57**의 디바이스 세션에서 함께 판정한다. ⑩은 #57 아래에서 열려 있다.

P4에서 같은 이유로 W8A16의 `MM16_R`(현재 4) / `MM16_TB`(현재 2)도 sim만 보고 정한 값이다(레지스터 압박 대 재사용의 절충). 디바이스에서 `MM16_R` 2/4/8 × `MM16_TB` 1/2/4 스윕을 decode(m=1)와 prefill(m=128) 양쪽에서 돌려 기본값을 확정한다 — m=1에서는 TB 블로킹이 무효라 R만 의미가 있다. ⑬(VTCM 스트리밍)과 같은 세션에서 함께 측정하는 것이 효율적이다.

## ⑪ `run_e2e_test.sh` 체크섬 push — **완료 (2026-09-16, `7dd3a949`)**

`push_if_changed`가 **파일 크기만** 비교한다. ABI v4는 WEIGHTS 바이트 순서만 바꾸고 크기(598,623,744)를 유지했으므로, 디바이스에 남아 있던 2026-09-01 row-major 이미지가 그대로 쓰였고 `.hexcfg`는 `weight_layout=tiled32`라 P2·P3 skel 전부가 쓰레기 출력(PPL 2e8 … nan)을 냈다. 수동 `adb push`로 해소(2026-09-16). 수정: 크기 대신 `md5sum` 비교(또는 `.hexw`에 레이아웃 id를 포함한 짧은 헤더/사이드카 파일). 600 MB md5는 디바이스에서 수 초 걸리므로 `--force-push` 플래그 병행 검토.

**완료**: `7dd3a949` [tools] Compare checksums before pushing the Hexagon e2e image — `push_if_changed`가 크기 대신 md5를 비교한다. P4 디바이스 측정에서 skel·하네스·1-layer 이미지가 조용히·정확히 push되는 것을 확인했다.

## ⑫ 호스트/RPC decode 경로 (P3 이후 실제 병목)

디바이스 측정(2026-09-16, 06 디바이스 표): `--eval` decode DSP 26.8 ms인데 host wall 45.8 ms, 생성 모드는 DSP 35 ms / host 76.4 ms — step당 ~40 ms가 호스트다. 내역: FastRPC 왕복(~0.26 ms, 무시 가능) + **151,936 float logits 복사** + 호스트 argmax. 설계 후보: (a) `forward()`가 top-k(값+인덱스)만 rout으로 반환 — 샘플링 파라미터를 DSP가 알아야 하므로 greedy/top-k만, (b) argmax를 DSP에서 수행해 토큰 id 1개만 반환(샘플링 불가, 앱 제약), (c) logits를 rpcmem ACT 슬롯에 써서 복사를 없애고 호스트가 zero-copy로 읽기 — ABI 변경 최소. **(c) → (a) 순으로 검토.** P4 이후 디바이스에서 판정.

- 2026-09-17, `docs/plans/24-host-logits.md` / 브랜치 `hvx/24-host-logits`: #23 로그(하네스는 RPC만 계측, argmax·`log_softmax_at`은 구간 밖)를 다시 읽으면 생성 모드의 host−DSP 간격은 512/1024/4096에서 0.5 / 2.9 / 2.2 ms(host 31.3 / 43.2 / 104.1 ms vs DSP 64.3 / 84.2 / 213.0 Mcyc @ 2.09 GHz)로, "~40 ms"는 P3 생성 모드(비교 불가로 이미 표시)와 teacher-forced `--eval` 행(`pcycles÷us` 1.15 GHz — 호스트가 151,936개 `exp()`에 ~10 ms를 쓰는 동안 DSP가 반클록으로 내려가는 idle-gap/DCVS 효과)에서 온 값이다. 구현은 (c)의 최소형: ABI·IDL·이미지 무변경, 호스트 logits 버퍼만 rpcmem(`HexagonBackend::logits_`, 하네스 `--logits-mem malloc|rpcmem|static`, `static`은 `FASTRPC_MAP_STATIC` 1회 매핑). `hexagon_rpc_test`가 8-float 대비 151,936-float 반환 비용을 메모리 종류별로 측정하고(`forward_full_us mem=…`), 하네스가 `E2E decode steps= median_us= median_pcycles= pcycles_per_us=` 요약을 찍는다. (a)/(b)는 static 반환이 8-float 대비 ≥ 2 ms일 때만 필요. 디바이스 수치: `docs/measurements/24-host-logits.md`.
- 2026-09-18 측정(S25 Ultra `R3CY10WM83Y`, v75 skel `cc0f1725…`), **⑫ 종결**: 빈 호출(`hexagon_rpc_test` dummy path, 32회 warm median)에서 151,936 float 반환은 malloc 8919 / rpcmem 8771 / rpcmem+`FASTRPC_MAP_STATIC` 6381 µs — 1회 매핑이 2.5 ms를 없애고, fd 경로만으로는 0.15 ms. 실제 decode step @512(`--chunk 128 --steps 64`, 역순 2회)에서는 static 34.00 / rpcmem 34.06 / malloc 34.07 ms로 세 경로가 70 µs(0.2 %) 안 — 호스트 wall은 이 유닛 클록(1.91 GHz)의 DSP op loop 그 자체이고, 명목 2090 MHz로 계산한 2.85 ms "gap"은 유닛 클록이지 전송이 아니다(HEXAGON.md §7 rule 9(c)). `--eval` PPL 33.0884 / top-1 189, 생성 id는 세 경로와 #23 로그에 대해 byte-identical. 하네스와 `HexagonBackend`의 기본은 `static`; (a)/(b) DSP top-k/argmax(ABI v5)는 열지 않는다. 남은 호스트 측 항목은 idle-gap 동안의 DSP 클록(#41: `--eval` 1167–1178 vs 생성 루프 1905–1915 pcycles/µs), decode 예산은 weight stream(#25, #51).

## ⑬ W8A16 VTCM/DMA 스트리밍 (P4에서 제외, D3) — **#59 (2026-09-18)**

P4의 int16 lanewise `down_proj` 커널은 DDR 직접 읽기(행 4개 × 3 KB 동시 스트림)다. sim은 DDR/DMA를 모델링하지 않아 스트리밍의 이득을 판정할 수 없어 제외했다. 디바이스에서 `HTP_MM_NO_VTCM`식 A/B(같은 세션, decode·prefill 모두)로 판정하고, 이득이 있으면 `mm_worker_vtcm`의 청킹(`rows_per_buf`를 `MM16_R` 배수로)을 row-major용으로 일반화한다. `hvx-matmul.c`의 `Follow-up:` 노트가 자리다.

- **숫자가 생겼다 (#25, 2026-09-18)**: stream-only skel C의 FARF 분할에서 `down`(row-major, 88.1 MB/step)은 `mm16` 7.23 Mcyc = 4.15 ms에 움직여 **21.2 GB/s**, 타일 DMA 링(q/k/v/o/gate/up 352.3 MB, 9.51 ms)은 **37.0 GB/s**. shipping 커널의 `mm16` 8.59 Mcyc(B의 15.7 %)는 스트림 대기보다 1.35 Mcyc 위일 뿐이라 큰 레버는 커널이 아니라 `down`을 DMA 링에 올리는 것(21 → 37 GB/s, ≈ 3 Mcyc/step) — **issue #59**(p2). `mm_worker_vtcm`의 청킹을 row-major용으로 일반화(`rows_per_buf`를 `MM16_R` 배수로)하는 설계는 위 그대로.

## ⑭ Hexagon SDK 6.0.0.2 → 6.4 이상 상향 — **사용자 요청(2026-09-16)** — **완료(#23, 디바이스 확인 2026-09-17)**

- 2026-09-17, `docs/plans/23-sdk64-baseline.md` / 브랜치 `hvx/23-sdk64-baseline`: 컨테이너(`tools/docker/`)에서 SDK 6.4.0.2 / hexagon-clang 19.0.04(`toolv19`)로 v75 13개 + `profile acc` 전부 PASS, STAT은 P4 기록과 비트 동일(HEXAGON.md §8.3). `hexagon_toolv19_v79`·`computev79` 존재, `-mhvx-ieee-fp` v75/v79 모두 허용(스크립트가 프로브해 출력), `Q6_*` 94개 이름 모두 19.0.04 헤더에 존재(리네임 없음). HexKL은 1.0.0-beta1(`lib/hexagon_toolv19_{v73,v75,v79}`)이 `~/Qualcomm/hexkl_addon`에 있어 공존 문제 없음(#27). 디바이스 재측정은 `docs/measurements/23-sdk64-baseline.md`(v75/v79 skel, 512/1024/4096) 핸드오프로 진행; 채워지면 완료.
- 2026-09-17 디바이스 확인(S25 Ultra, `docs/measurements/23-sdk64-baseline.md` 변형 A): Mac이 워크스테이션에서 닿지 않아 skel·하네스는 워크스테이션의 SDK 6.4.0.1(같은 hexagon-clang 19.0.04)로 재빌드, 그 바이너리로 sim 13개 + `profile acc`를 먼저 재실행해 컨테이너와 STAT 동일·pcycles ≤ 0.2 % 확인. v75 skel: prefill 189.0 / 117.1 / 27.1 tok/s(P4 192.1 / 118.2 / 27.3, −1.6 … −0.6 %), decode 31.9 / 23.2 / 9.6 tok/s(27.7 / 22.9 / 9.4), DSP Mcyc 64.3 / 84.2 / 213.0(68.9 / 85.2 / 216.9). 1024·4096은 ±3 % / ±5 % 밴드 안, 512 decode만 빠른 쪽으로 밴드 밖(측정 구성상 ±5 %, HEXAGON.md §8.2). P4 프롬프트 `--eval` 33.0884 / 189 vs x86 33.0195 / 184(+0.21 %, P4 밴드 안). **회귀 없음 → ⑭ 종결.** 남은 편차 하나: 계약 §2 규칙 3(컨테이너 툴체인이 전부 빌드)과 달리 디바이스 바이너리는 워크스테이션 빌드였다 — 컨테이너 skel md5(`b9b1e361…`/`bdfdf143…`)는 디바이스에서 실행되지 않았다.

현재 sim·skel 빌드는 `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.0.0.2`(toolchain 8.7.08)에 고정돼 있고, QuRT sim 이미지가 v75까지만 있어 sim 게이트가 v75다(HEXAGON.md §5.2는 SDK 6.3.0.0/toolchain 8.8에서도 통과했다고 기록). 상향 시 확인할 것: `setup_sdk_env.source` 경로(`build_sim_test.sh`, `run_sim_test.sh`, `build_skel.sh`, 계획서·핸드오프의 하드코딩), `run_main_on_hexagon` 이미지의 `hexagon_toolv*_v79` 존재, `-mhvx-ieee-fp` 플래그 호환, `hvx_hexagon_protos.h` 인트린식 이름 변화, HexKL(6.0.0.2 lib)과의 공존(메모리 `hmx-impl-worktree`). 완료 후 13개 + `profile acc` + 디바이스 e2e를 새 SDK로 재측정하고 HEXAGON.md §5.2/§8.3의 SDK·툴체인 표기를 갱신한다. ④의 전제.

## ⑮ 상류 qf32 발산 (ATTN/W8A8 체인)

P4 디바이스 격리(2026-09-16, 1-layer 이미지 `--dump-op`, 128-tok 청크)에서: `down_proj` **입력**(= SILU_MUL 출력, P3 이후 코드 불변)이 이미 DSP vs x86 ref rel-RMS **2.65 %**(max_abs 0.113, 0.1/0.1 밴드 밖 원소는 0개), 출력은 2.46 %. 반면 DSP 자기 입력으로 int16 ref를 재계산해 DSP 출력과 비교하면 rel-RMS 1.6e-4 / max_abs 0.00098(fp16 1 ulp, 원소의 19.9 % — ⑧의 ±1 LSB 비율과 정합). 즉 **커널은 실리콘에서 정의대로 정확하고, DSP-vs-reference PPL 갭(0.16 / 0.44 / 0.89 / 1.29 % 한 밴드)을 만드는 것은 상류 ATTN/W8A8 qf32 체인**이다. 레이어 0에서 이미 2.65 %라는 것은 ATTN(qf32 score/softmax/PV)과 W8A8 에필로그의 qf32 반올림이 fp32 ref와 다른 지점이 누적된다는 뜻이다.

- 왜 지금 안 고치나: 정확도가 나쁜 방향으로 회귀한 것이 아니고(P4는 x86 ref를 −0.83 % 개선), 이 프롬프트의 PPL이 1e-3 미만 섭동에 ~1 % 반응해 갭 자체가 판정력이 낮다.
- 어떻게 볼 것인가: P5 ATTN 작업(K^T 재사용)에서 커널을 어차피 다시 쓰므로 그때 함께 본다. 후보 절차: (1) 1-layer에서 op별 rel-RMS 프로파일을 떠서 2.65 %가 어느 op에서 생기는지 이분(ATTN 내부는 `forward_debug`로 안 보이므로 중간 dump 추가 필요), (2) 해당 지점만 fp32 누적/`Vsf_equals_Vqf32` 순서를 ref와 맞춰 재측정, (3) 갭이 0.2 % 밴드로 줄면 06의 게이트를 다시 세운다.
- 관련: §7 규칙 1(IEEE 경로 사용 불가)이 qf32를 강제하므로, 완전 일치는 ⑧의 "정수 가수 곱" 계열 수단이 필요할 수 있다.
