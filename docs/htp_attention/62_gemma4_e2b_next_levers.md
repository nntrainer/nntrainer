# 62. Gemma-4 E2B on the HTP — 남은 시간이 어디에 있고 무엇을 할 수 있나 (2026-11-09)

**상태.** PLE tail 융합(`ec325458d`)과 dense FFN의 arena 입출력(`90ced232e`)까지 들어간 상태를
측정했다. 이 문서는 그 측정 결과와, 아직 하지 않은 최적화 후보를 근거·예상 효과·방법과 함께 정리한다.
후보는 §4에 있다. 시도했다가 버린 것은 §5, 측정하면서 밟은 함정은 §6이다.

- 기기: S25 Ultra(SM8750, v79). `R3CY205ZMND`와 다른 한 대에서 쟀다.
- 설정: a16, `cfg_p8193_a16.json`(8192 토큰 prefill, 256행 청크, `skip_prefill`)과 folded PLE export를
  쓴다. 측정 패치는 `gemma-4-E2B-hexkl/tools/folded_ple_meas.patch`다.
- 정확도 기준: prompt nll/token은 10.0292(`NNTR_PPL=1`, `skip_prefill: false`)다. 생성 첫 4토큰은
  "This summary describes the"다.
- 같은 빌드라도 세션(포트 재연결)마다 prefill이 ±60 ms쯤 흔들린다. A/B는 같은 세션에서 번갈아 돌려야
  한다.

## 1. 기준 숫자

| | 8192 prefill | decode |
|---|---|---|
| `ec325458d` (PLE tail 융합) | 2965–3139 ms (약 2,650 tok/s) | 11.3–12.2 tok/s (약 87 ms/토큰) |
| `90ced232e` (dense arena I/O) | 같은 세션에서 −47 ~ −71 ms (−1.5 ~ −2.2%) | 11.7 → 12.3 tok/s |

## 2. prefill 분해 (`ec325458d`, 256행 청크 32개, 청크당 평균 96 ms)

`NNTR_LAYER_PROFILE=1`로 노드별 시간을 잰 결과다.

| 노드 | 청크당 | 비율 |
|---|---|---|
| `ffn` (dense FFN + PLE tail, 레이어 0–13) | 38.8 ms | 40% |
| `attention` | 22–24 ms | 25% |
| `qkv` (레이어 0–14) | 18.5 ms | 19% |
| `post_attention_norm` (o-proj + 잔차) | 12.4 ms | 13% |
| `per_layer_input_lut` | 1.7 ms | 2% |

- **KV 공유 레이어(15–34)는 prefill 비용이 0에 가깝다.** 공유 레이어의 은닉 상태 중 마지막 토큰이 아닌
  것은 아무 출력에도 쓰이지 않기 때문이다. prefill 시간은 거의 전부 레이어 0–14에 있다. 레이어 14는
  K/V만 계산한다(`532d5f37d`).
- **attention은 sliding과 global이 다르다.**
  - sliding 레이어는 호출당 약 1.0 ms로 일정하다.
  - global 레이어 4와 9는 위치에 비례해 첫 청크 2.8 ms에서 마지막 청크 10.6 ms까지 커진다.
  - 그리고 **KV 약 5120행에서 계단이 있다**(§4.4).
- **DSP 안쪽.** 진입점 wrapper를 skel에 넣어 쟀다. 프로브를 끈 상태의 DSP 총시간이다.

  | 호출 (256행) | 호스트 wall | DSP 총시간 | 호출당 호스트 오버헤드 |
  |---|---|---|---|
  | qkv `layer_norm_ld` | 약 1093 µs | 810 µs | 283 µs |
  | o-proj `layer_res_add_u16` | 약 774 µs | 571 µs | 203 µs |
  | dense `dense_ffn_ple` | 2692 µs | 2413 µs | 279 µs |
  | attention `attn_q2_step_fq` (sliding) | 약 1000 µs | 약 650 µs | 약 350 µs |

  - **dense 내부:** gate_up 1173 µs, down 468 µs, post-FFN norm/add 254 µs, PLE gate 130 µs, PLE
    proj+norm+add 311 µs다.
  - **matmul은 이미 HMX 한계 근처다.** a16은 u8 sweep을 두 번 하므로 그만큼 연산이 두 배다. gate_up이
    약 16 TOPS, down이 약 20 TOPS로 돈다.
- **호출당 오버헤드를 합치면 청크당 약 16 ms(17%)다.** 그중 ARM 쪽 스테이징 memcpy는 3.4 ms/청크다
  (`NNTR_HTP_PROFILE=2`의 `arm staging memcpy`).

## 3. FastRPC 호출 수

| | 호출 수 |
|---|---|
| prefill, 청크당 | 약 58회: qkv 15, attention 14, o-proj 15, dense 14 |
| prefill, 1회 꼬리 | 마지막 64행의 공유 레이어 FC 195회 |
| prefill 전체 | 약 2,050회 |
| decode, 토큰당 | FC 186회 + attention 레이어당 1회 = 약 220회 |

decode 토큰당 FC 시간은 아래와 같다(`NNTR_HTP_PROFILE=2`, M==1 호출 수 × 호출당 host µs). FC만
52.6 ms이고, 나머지 약 34 ms는 attention decode 호출과 CPU 연산이다.

| 호출 | 토큰당 |
|---|---|
| 공유 레이어 gate·up (40회 × 344 µs) | 13.8 ms |
| 공유 레이어 down (20 × 538) | 10.8 ms |
| 자기 레이어 dense (15 × 516) | 7.7 ms |
| lm_head (1 × 5634) | 5.6 ms |
| o-proj (28 × 146 + 7 × 196) | 5.5 ms |
| q / qkv (16 × 129 + 4 × 161 + 12 × 204 + 3 × 275) | 6.0 ms |
| 공유 레이어 PLE gate + proj (20 × 87 + 20 × 78) | 3.3 ms |

## 4. 후보

### 4.1 arena 입출력을 qkv·attention·o-proj로 넓히기 (예상 prefill −40 ~ −85 ms, −1.5 ~ −3%)

- **한 것.** `mm_u8i4_dense_ffn_ple_io`(`90ced232e`)다.
  - 스테이징 버퍼를 처음 쓸 때 `fastrpc_mmap(FASTRPC_MAP_FD)` + `arena_attach`한다. 해제는 세션 종료
    hook이 한다.
  - 호출에는 arena id와 float 개수만 넘긴다.
  - DSP는 진입 시 입력과 출력을 invalidate하고, 반환 시 출력을 flush한다.
  - 호스트는 캐시 유지를 하지 않는다. CPU 쪽은 IO-coherent다(§6.1).
- **측정.**
  - dense 호출당 절감은 30–70 µs다. 프로파일에서는 2614 → 2591 µs였다.
  - o-proj도 0.3–0.8 ms/청크 같이 줄었다. dense를 거친 버퍼가 매핑된 채로 넘어가기 때문이다.
  - 매핑만 하고 호출은 sequence로 둔 변형은 그 절반쯤의 효과였다.
- **기대보다 작은 이유.** FastRPC가 sequence 버퍼에 하던 DSP 쪽 캐시 유지를 이제 진입점이 직접 한다.
  1.5 MiB invalidate + flush가 약 46 µs이고, 이 비용은 그대로 남는다. 줄어든 것은 프레임워크 몫뿐이다.
  남은 호출당 오버헤드는 약 120 µs다(FastRPC 고정 80–100 µs + 캐시 유지).
- **방법.** 나머지 세 진입점에 같은 `_io` 쌍둥이를 만든다.
  - `mm_u8i4_layer_norm_ld`(qkv: act in, out_cat out)
  - `mm_u8i4_layer_res_add_u16`(o-proj: u16 act, res, out)
  - `attn_q2_step_fq`(q in, out_u16 out, K/V append 행)
- **주의할 점.**
  - DSP arena 테이블은 `NNTR_HVX_MAX_ARENAS = 32`다. 가중치 arena chunk와 스테이징 풀의 크기 클래스
    수를 합쳐 세어야 한다. 슬롯이 모자라면 `ioArena`가 `kNoIoArena`를 돌려주고 기존 경로로 간다.
    느려질 뿐 틀리지는 않는다.
  - 출력을 invalidate하지 않으면, 이전에 캐시에 있던 줄이 flush될 때 호스트가 나중에 쓴 값을 덮을 수
    있다. 그래서 출력도 진입 시 invalidate한다.
  - 검증은 `DenseFfnPleIoMatchesArgs`와 같은 방식으로 한다. sequence 진입점과 출력 전체를 비트 단위로
    비교하고, 매 호출 새 데이터를 쓴다. 그리고 캐시 유지를 뺀 mutant가 실패하는지 본다.
- **캐시 유지까지 줄이려면.** DSP가 만든 출력을 DSP가 바로 다시 읽는 경우가 있다(out_to_host false:
  dense out → 다음 qkv in). 이때는 flush와 invalidate가 필요 없다. 다만 호스트가 `sampledSame` 창과
  CPU 폴백에서 읽으므로, 버퍼마다 "마지막 writer"를 추적하는 상태가 필요하다. 호출당 약 46 µs 상한이다.

### 4.2 attention + o-proj를 한 호출로 (예상 −45 ms, 레이어 통째로 하면 상한 −135 ms)

- **근거.** 호출 하나를 없애면 남은 고정비 약 100 µs가 사라진다.
  - attention+o-proj는 레이어당 1회, 청크당 14회 × 32 ≈ 45 ms다.
  - qkv+attention+o-proj+FFN을 한 호출로 하면 레이어당 3회, 약 135 ms가 상한이다.
- **방법.** attention의 u16 출력은 이미 ION에 남아 o-proj로 넘어간다(`attn_out16_`, `res_add_u16`).
  - DSP 쪽은 두 진입점을 한 함수 안에서 차례로 부르면 된다. 출력이 DSP 힙/arena에 머물게 한다.
  - 호스트 쪽은 B(PLE tail 융합, `ec325458d`)와 같은 그래프 수준 작업이다. `mha_core`와
    `post_attention_norm`(residual_add + proj)이 서로 다른 레이어이기 때문이다.
  - 선례는 B에서 dense_ffn이 slice를 두 번째 입력으로 받은 방식이다. 다만 attention 레이어는 KV 캐시
    바인딩과 decode 경로까지 함께 다뤄야 해서 B보다 크다.
- **판단.** 이득 대비 작업량은 §4.3이 낫다.

### 4.3 decode: 공유 레이어(15–34)의 FFN + PLE tail 융합 (예상 토큰당 −6 ~ −9 ms, 87 → 약 80 ms)

- **근거.** 공유 레이어 20개는 아직 옛 경로다. FFN이 gate·up·down FC 3회, PLE tail이 gate·proj 2회에
  CPU 연산 5개(gelu, mul, norm, add, scalar)다.
  - 이 FC 5회가 토큰당 27.9 ms다(§3 표). DSP 시간은 그중 약 23 ms다.
  - 자기 레이어 0–14는 `dense_ffn` + `ple_tail` 한 호출이다(M==1에서 516 µs).
- **방법.** `createTransformerDecoderBlock`에서 `ffn_adds`가 `!is_kv_shared_layer`를 요구하는 조건을
  푼다. 공유 레이어도 `dense_ffn`(in_norm = pre_ffn_norm, out_norm, out_add, ple_tail)으로 만든다.
  - 주석에 남은 이유는 "공유 레이어의 attention과 MLP는 skip_prefill을 쓰는 이전 레이어"라는 것이다.
    지금은 `dense_ffn`도 `skip_prefill`을 받는다(`dense_ffn_layer.cpp`의 finalize). 따라서 다음을 확인할
    것이다.
  - 가중치 파일 순서를 확인해야 한다. pre_ffn_norm, gate/up/down(converter의 순서), post_ffn_norm,
    ple gate, ple proj, post_ple_norm, layer_scalar 순서가 `dense_ffn`이 요청하는 순서와 맞아야 한다.
    `Gemma4DifferentialTest.FP32MatchesHFReference`가 잡는다. 레이어 0에서 실제로 잡았다(B).
  - `USE_DOUBLE_WIDE_MLP`이면 공유 레이어의 intermediate는 2배(12288)다. dense a16 경로의
    `DENSE_A16_MAX_ROWS`, heap 크기, down의 열 slice를 확인해야 한다.
  - a16 인코딩 키 이름이 바뀐다(`layer{L}_ffn:gate` 등). enc 파일에 공유 레이어 키를 추가해야 한다.
    B 때처럼 기존 키를 복사하면 된다.
- **예상 근거.** FC 5회 → 1회로 transport 4회분(약 0.3 ms)이 줄고, CPU 연산 5개가 없어진다. 레이어당
  0.3–0.45 ms × 20 ≈ 6–9 ms다. matmul 자체는 그대로다.
- **같이 볼 것.**
  - 공유 레이어의 o-proj(`attention_out` FC)와 post_attention_norm도 자기 레이어처럼
    `residual_add(proj)` 한 호출로 접을 수 있다. 토큰당 20회 더 줄어든다.
  - lm_head는 토큰당 5.6 ms(Q4_0 262144 × 1536)다. vocab이 커서 대역폭으로 결정되는 비용이라 구조로
    줄일 여지는 작다.

### 4.4 global attention의 K/V 스트리밍 (예상 8192 prefill −120 ms, −4%; 프롬프트가 길수록 커짐)

- **근거.** 레이어 4, 9(head_dim 512, KV head 1개, q head 8개)의 kernel 시간이 다음과 같다.

  | 청크 시작 위치 | kernel |
  |---|---|
  | 4608 | 2.72 ms |
  | 4864 | 6.33 ms |
  | 7936 | 9.79 ms |

  - qk + softmax + pv는 위치에 비례해 3.0 → 3.3 → 5.0 ms로만 늘어나고, 나머지 시간이 0에서 3–4.7 ms로
    뛴다(`NNTR_HTP_ATTN_TRACE=1`의 `kernel=` 대 `qk/softmax/pv`).
  - 원인은 `hexkl_attn_q2_plan`이다. KV head 하나의 K^T와 V(u8)가 각각 행 수 × 512 B다. 약 4.8K행부터
    VTCM에 상주하지 못해 스트리밍 레이아웃으로 바뀐다.
  - 스트리밍에서는 **64행 query 블록마다, q head마다 K^T chunk와 V를 다시 DMA한다.** 같은 KV head를
    쓰는 q head 8개가 같은 K/V를 8번 읽는다.
- **예상.** 계단 이전의 기울기(약 0.41 µs/키)로 이어 그리면, 청크 19–31의 두 레이어에서 청크당 평균
  약 4.6 ms를 덜 쓴다. 13청크 × 2 ≈ 120 ms다.
- **방법.** 스트리밍 chunk 하나를 여러 (q head, 블록)이 함께 쓰도록 루프 순서를 바꾼다.
  - 블록의 score 행은 softmax(두 번 훑기) 때문에 전부 VTCM에 있어야 한다. 그 크기가 블록 하나에
    n_blk × 4 KiB, 8192에서 약 1 MiB다. 그래서 한 번에 공유할 수 있는 (head, 블록) 쌍은 2–4개다.
  - 그렇게 해도 K/V DMA는 1/2–1/4로 준다.
  - 다른 방법은 KV를 블록 단위로 끊어 가며 처리하는 flash 방식(온라인 softmax)이다. 그러면 K/V를 한 번만
    읽으면 되지만, 지금의 "softmax는 두 번 훑기, 아무것도 rescale하지 않음"이라는 수치 계약이 바뀐다.
- **판단.** 커널 1206줄 규모의 재구성이다. 512폭 global head가 있는 모델에서 긴 프롬프트일수록 이득이
  크다.

### 4.5 qkv 에필로그 (head별 norm + RoPE) — 상한 5%, 두 번 시도해서 이득이 없었다 (§5.1)

남은 아이디어는 rsqrt를 벡터로 계산하는 것(Newton 3회)이다. head마다 생기는 스칼라 추출, `sqrtf` 호출,
나눗셈을 없앤다. 다만 §5.1의 측정상 이 비용은 프로브를 켠 상태에서도 호출당 77 µs 이하였다. 최대 약
37 ms(1.2%)다.

## 5. 시도했다가 버린 것

### 5.1 qkv 에필로그 (2026-11-09)

- **측정.** 프로브를 켠 상태에서 qkv 호출(256행)의 에필로그 대기가 451 µs로, matmul 155 µs보다 길었다.
  에필로그를 통째로 빼면 prefill이 −169 ms였다(프로브 상태). 내역은 dequant 115, 제곱합·추출·sqrt 77,
  스케일 패스 52, RoPE 130 µs다.
- **시도 1: norm 스케일과 RoPE를 한 패스로(`hvx_norm_rope_row_f32`).**
  - 호스트 에뮬레이터에서 비트 단위로 같았다.
  - 대기 451 → 409 µs, prefill 약 −20 ms라 세션 간 편차에 묻힌다.
- **시도 2: VTCM 누산 타일에서 바로 dequant하기.**
  - 제곱합을 구하고 다시 dequant하면서 norm+RoPE를 한 뒤 DDR에 한 번만 쓰는 방식이다.
  - e2e가 빨라지지 않았다(기준 3038/2994 ms, 이것 3050/3091 ms). 타일을 두 번 dequant하는 비용이 L2
    재읽기를 없앤 이득을 상쇄한다.
- **교훈 1: 프로브 측정은 과대평가한다.** 프로브를 켠 측정(`hexkl_probe_on`)은 타일 루프 안에서
  qtimer를 읽는다. 그래서 에필로그 대기를 부풀린다. 프로브를 켠 baseline 3171 ms 대 프로브 없는 약
  3040 ms다.
- **교훈 2: 컴파일러 버그.** 시도 2의 첫 판은 기기에서 4행 그룹의 첫 행이 약 0으로 normalize됐다.
  - hexagon-clang(`-mhvx`, ieee-fp 없음)이 배열에 든 qf32 누산기 하나를 sf 형식으로 루프 내내 들고
    다녔다. 그런데 뒤의 합산 루프는 그 값을 qf32로 읽었다. 이름 있는 변수 4개로 바꾸자 사라졌다.
  - 호스트 에뮬레이터는 float 형식이 하나뿐이라 이런 문제를 원리상 볼 수 없다.
  - **qf32를 쓰는 HVX 커널은 기기 gtest가 따로 있어야 한다.** `A16QkvNormRopeMatchesHost`
    (`e33e9890d`)를 이 이유로 남겼다. 기준 skel에서 통과하고, 위 버그가 있는 skel에서는 핸들마다 25%가
    틀린다.

### 5.2 SDK 버퍼 속성 (`remote_register_buf_attr`)

측정용 `rpc_bench` 진입점(in/out sequence만 받고 아무것도 하지 않음)으로 쟀다.

- **결과.** `TRY_MAP_STATIC`, `COHERENT`, `FORCE_NOFLUSH|FORCE_NOINVALIDATE`, `KEEP_MAP`, uncached
  rpcmem 모두 cached rpcmem과 같았다(1.5 MiB in/out, DSP는 아무것도 안 함, 310–350 µs).
- **함께 발견한 것.** sequence로 넘긴 버퍼는 DSP에서 쓰기가 느리다. 1.5 MiB memset이 sequence 버퍼에서
  502 µs, arena에서 93 µs였다.

### 5.3 마이크로벤치가 실제 호출 절감을 5배쯤 크게 보였다

`rpc_bench_arena`에서는 sequence 879 µs 대 arena 261 µs였다. 그런데 실제 dense 호출에서는 30–70 µs
절감이었다. 벤치의 sequence 쪽 비용 대부분은, 실제 호출에서는 어느 경로든 치러야 하는 DSP 캐시 유지였다.
**새 경로의 이득은 마이크로벤치가 아니라 실제 호출의 A/B로 판정할 것.**

## 6. 측정 메모와 함정

1. **IO coherency (S25 Ultra).**
   - CPU 쪽: cached rpcmem에 CPU가 쓰고 읽을 때 `dc cvac`/`civac`가 없어도 DSP와 값이 맞았다
     (arena 경로, 35/35회; `DenseFfnPleIoMatchesArgs` 5회 비트 동일).
   - DSP 쪽: coherent하지 않다. invalidate/flush를 빼면 매 호출 틀린다.
   - 호스트에서 캐시 유지를 하면 1.5 MiB에 230–470 µs가 들므로 하지 않는다.
2. **포트가 바뀌면 `/data/local/tmp`가 비워진다.** 같은 기기에서도 그렇다. `stage_nntr.sh`류로 약 15분
   다시 올려야 한다.
3. **DSP FARF를 logcat에서 보려면** `nntrainer_causallm.farf`(내용 `0x1f`)가 바이너리 옆이나
   `ADSP_LIBRARY_PATH`에 있어야 한다. attention 트레이스(`NNTR_HTP_ATTN_TRACE=1`)는 `ml_logi`라 logcat
   버퍼가 넘칠 수 있다. 끝나고 `logcat -d`를 파일로 받는다.
4. **노드별 시간은 `NNTR_LAYER_PROFILE=1`**(stdout의 `LAYERPROF`)로 본다. 진입점별 DSP 시간은 skel에
   임시 wrapper를 넣어 FARF로 찍는다(커밋하지 않음). `NNTR_HTP_PROFILE=2`의 a16 경로는 `dsp=`가 0이라
   DSP 내역을 주지 않는다.
5. **A/B는 같은 skel에 라이브러리 디렉터리만 바꿔서** 같은 세션 안에서 번갈아 돌린다. 기기에 `base/`, `io/`
   식으로 두고 `LD_LIBRARY_PATH`와 `ADSP_LIBRARY_PATH`를 바꾼다. IDL은 끝에만 추가하므로 옛 라이브러리도
   새 skel로 돈다.
