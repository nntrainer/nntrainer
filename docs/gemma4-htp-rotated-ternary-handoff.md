# Gemma4 pre-rotated ternary checkpoint: 런타임 작업 인수인계

작성: 2026-10-16, Asia/Seoul. 이 문서는 **작업 중간 체크포인트**다. 다른
세션이 이 문서만 읽고 이어서 작업할 수 있도록 계획, 완료 범위, 재개 지점,
검증 기기와 경로, 다음 세션의 시작 프롬프트를 적는다.

관련 문서: [TL2 / affine WH 인수인계](gemma4-tl2-affine-handoff.md),
[Gemma4 HTP 사용자 가이드](gemma4-htp-user-guide.md),
[코드 가이드](gemma4-htp-code-guide.md), [lm_head NPU](gemma4-htp-lmhead-npu.md),
[KV cache S25U 기록](gemma4-htp-kv-cache-s25u.md).

## 0. 한 줄 요약

새 Gemma4-26B-A4B 체크포인트(Hadamard로 사전 회전된 ternary, 마지막 두 MoE
레이어만 4-bit)를 다른 세션이 변환해 기기에 올린다. 그 전에 런타임이 그 파일을
그대로 돌릴 수 있어야 한다. 필요한 런타임 작업은 네 가지이고,
**2026-10-16 밤 세션에서 A, B, C, E가 끝나 새 체크포인트가 기기에서 정상 텍스트를 낸다
(7절).** D(static activation scale)만 사이드카 값이 없어 미착수다. 체크포인트 쪽에는
고칠 것이 둘 있다(7.2절: config.json의 두 키, 28/29의 TL2 저장).

## 1. 배경: PR #4428 코멘트와 사용자 결정

[PR #4428](https://github.com/nntrainer/nntrainer/pull/4428)의 2026-10-16 코멘트
세 건이 요구사항이다.

| 시각(UTC) | 내용 |
|---|---|
| 10:15 | "pre-rotated ternary checkpoint → on-device model: nntrainer에 남은 일". (1) MoE 레이어별 expert dtype (0–27 `TL2CX_WH`, 28–29 4-bit `QS4CX_WH_AFFINE`), (2) 온라인 Hadamard 회전을 이 PR 베이스로 포팅 (nnstreamer `codex/gemma4-spinquant-20261014`의 `7ed1a3ffd`), (3) 컨버터(다른 세션 몫), (4) 회전 규약 확인 |
| 10:21 | "scales, formats, and what the runtime must take from the file". 가중치 scale/column sum/offset은 레코드에 있는 값을 그대로 쓴다(재계산·재양자화 금지). 28–29 expert는 offset 0이지만 커널이 offset을 처리해야 한다. 파일에 activation scale은 없다 |
| 사용자 | "dynamic quantization은 이제 필요 없다" = activation scale은 양자화 부서가 보정으로 최적값을 찾아 주고, 런타임은 그 값을 static으로 그대로 쓴다. 이것도 코드 작업에 포함 |

체크포인트 레시피(10:15 코멘트):

| 텐서 | dtype | 비고 |
|---|---|---|
| embedding | `Q4_0` | 기존 팩과 byte 동일 |
| attention / dense FFN / lm_head | `QS4CX_WH_AFFINE` | 원본 int4 grid, 채널 offset 있음 |
| routed experts, layer 0–27 | `TL2CX_WH` | 정확히 {-s, 0, +s} |
| routed experts, layer 28–29 | `QS4CX_WH_AFFINE` | 4-bit 컨테이너 (값은 ternary, offset 0) |
| router / norm / scalar | FP32 | |
| 회전 | `rotation_meta.json`: mode hadamard, `rotation_mode: both`, seed 42 | gate_up 입력 `x ← x R_in`, down 입력 `a ← a R_down` |

## 2. 브랜치, 커밋, 워크트리

- PR head(작업 시작점): `72cca39c3` (`integrate/pr4416-quickdotai`, Jungwon-Lee fork).
  이 아래에 오늘 push한 LFM2 KV 커밋 `ef6bd729e`, `3f518fa83`이 있다.
- 작업 워크트리: `~/nntrainer-4428-had`, 브랜치 `htp/4428-hadamard`.
- 커밋 1건: **`[HTP] Add online Hadamard rotation for Gemma4 MoE`** =
  `7ed1a3ffd`의 cherry-pick(`-x`), 충돌 14군데 해결, 사인오프 추가.
- push 대상: `git push git@github.com:Jungwon-Lee/nntrainer.git HEAD:integrate/pr4416-quickdotai`.
- 다른 워크트리: `~/nntrainer-4428-head`(LFM2 KV 작업, 브랜치 `htp/4428-lfm2-kv`,
  이미 push됨), `~/nntrainer-4428-lmhead`, `~/nntrainer-4428-prefill` 등은 이 작업과 무관.
- 참고로 fetch해 둔 ref: `upstream/codex/gemma4-spinquant-20261014`
  (`910860585` = 회전 측정 기록 docs, `7ed1a3ffd` = 커널). 이 브랜치는 PR
  베이스와 1,489 커밋 차이라 rebase가 아니라 cherry-pick이 맞다.

## 3. 계획과 진행 상태

| # | 작업 | 상태 | 재개 지점 |
|---|---|---|---|
| A | Hadamard 회전 포팅 | **완료**, 기기 검증 완료 (4.6절 세 셀 통과) + 디코드 TL2 down 회전 누락 수정 `5d366e5cb` | — |
| B | 레이어별 expert dtype + `QS4CX_WH_AFFINE` expert | **완료** `d0e32095b`, `59825043b` (슬롯 크기별 free list, offset RPC `weight_set_offsets_batch`) | 7절의 메모리 주의 |
| C | 혼합 모델의 회전 범위 | **완료** `2d9057fb5` (dense FFN 마스킹, `scope=all` = expert 전부; `moe_hadamard`/`moe_hadamard_scope` 설정 키) | — |
| D | static activation scale | **미착수** (사이드카 값이 아직 없음; 설계 4.4절) | 포맷 합의 뒤 |
| E | 회전 규약 | **확정** (7절): 런타임 방향 맞음 + 체크포인트의 입력 차원별 ±1 부호 벡터(seed 42)를 사이드카로 적용 `2edf1c76f` | — |
| F | 기기 검증 | 7절의 표 | — |

## 4. 상세

### 4.1 A. Hadamard 포팅 (완료)

무엇이 들어왔나(`7ed1a3ffd`, 31파일 +6,080):

- 커널: `nntrainer/tensor/htp_backend/hvx/hvx_hadamard_f32.{c,h}` (HVX FP32,
  H44 Paley × Sylvester, decode용), `hmx/hexkl_hadamard_f16.h` (FP16 HMX, prefill용,
  2816/704 폭 전용), `hmx/hexkl_hadamard_coeffs.inc`, `hvx/hvx_hadamard_h44.inc`
  (생성 테이블, `test/htp/generate_hadamard_tables.py`).
- `hmx/hexkl_mm_u8i4_moe.c`: gate_up 입력 회전(`had_up`, 양자화 전),
  GeGLU 뒤 down 입력 회전(`had_down`), decode에서 한 MoE op의 여러 miss-round
  호출이 회전된 행을 공유(`hexkl_moe_rotated_input`, `hexkl_graph.c`가 소유).
- 플래그: `hexkl_mm_u8i4_moe.h`의 `HEXKL_MOE_FLAG_HAD_UP/DOWN/SCALAR/HMX/2BIT_ONLY`
  (0x04000000..0x40000000), ARM 쪽 `htp_moe_opts.h`의 같은 값.
- 환경변수(ARM, `htp_compute_ops.cpp` `sendMoeOptsOnce`):
  `NNTR_MOE_HADAMARD` = off | up | down | both | both_scalar | up_hmx | down_hmx | both_hmx,
  `NNTR_MOE_HADAMARD_SCOPE` = 2bit(기본) | all. **기본은 off.** 켜면 stderr에
  `[HTP] online Hadamard=... scope=... applied=0x..` 한 줄.
- 체크: `test/htp/host/hadamard_check.c`(FWHT 대조), `moe_layer_host_check.c`의
  HADAMARD 셀, `graph_host_check.c`의 회전 행 재사용 셀, `moe_opts_host_check.c`.
- 기기 도구(선택): `test/htp/hadamard_moe_bench.cpp`, `build_hadamard_bench.sh`,
  `run_hadamard_bench_on_device.sh`, `nntr_hvx_hadamard_probe.c`(IDL `hadamard_probe`).
- IDL이 바뀌었다(`hadamard_probe`, `mm_u8i4_moe_layer_norm_timed`). **ARM stub과
  DSP skel을 같이 재생성**해야 한다(옛 skel + 새 ARM = EBADPARM).

충돌 해결(14군데)에서 내가 손으로 합친 부분. 다음 세션은 이 부분만 의심하면 된다.

| 파일 | 합친 내용 |
|---|---|
| `hmx/hexkl_mm_u8i4_moe.c` | HEAD의 `moe_dn_fg_worker`(requant 융합, `fuse_dn_rq`)를 유지하고 `moe_rotate_rows`만 추가. HEAD의 `moe_gu_ctx`(acc_i24, affine_offset)에 `had_vb` 필드를 더하고 `moe_gu_worker` 끝에 upstream의 fp16 패킹 블록을 붙임. 스캔 출력 버퍼는 HEAD의 `use_tl2 ? act_ah : act_u8`를 세 분기 모두에 사용. M=1 AH 공유는 `if (use_tl2) {HEAD} else if (i != 0 && M == 1 && widest == 1) {공유} else {gather}`. 블록 n의 requant 제출은 HMX 회전 블록 뒤에 HEAD의 `if (!fuse_dn_rq)` 조건 유지. **추가 가드**: native TL2(`use_tl2`)에서는 회전 행 재사용(`rotated`)을 끈다 — 재사용 시 스캔이 건너뛰어져 `act_ah`가 채워지지 않기 때문. 비용은 호출당 HVX 행 한 줄 회전 |
| `test/htp/nntr_hvx.idl` | HEAD의 `weight_swap_batch_tl2_compact_arena` + upstream 두 entry 모두 유지 |
| `test/htp/nntr_hvx_mm_u8i4.c` | `moe_set_opts`의 HMX 검사(upstream) + HEAD의 `lm_head_wh_f32`와 upstream의 `moe_layer_norm_timed` 둘 다 |
| `test/htp/host/*_host_check.c` | 양쪽 블록 모두 유지 |
| `test/htp/host/run_host_checks.sh` | upstream이 손대지 않은 두 빌드 줄(`tl2_prefill_host_check`, `wh_affine_host_check`)에 `hvx_hadamard_f32.c` 추가 (첫 실행에서 link 실패로 발견) |

검증 상태:

- `bash test/htp/host/run_host_checks.sh`: 전체 통과 (HADAMARD 셀 포함; 로그 94줄).
- v81 DSP skel + ARM 빌드: **미확인.** `build_android.sh --htp`를 시작했으나 사용자가 확인 생략을
  지시해 결과를 보지 않고 push했다. 다음 세션은 4.6절의 빌드 명령부터 실행해 stub 재생성과
  skel/ARM 링크를 확인한다(host check는 통과했으므로 C 코드는 컴파일되지만 Hexagon 전용 블록
  `#if defined(__hexagon__)`과 IDL 생성 코드는 host check가 보지 못한다).
- 기기: **미실행.** 4.6절의 세 셀(회전 off 바이트 동일, 회전 on 실행, LFM2 회귀)이 그대로 남아 있다.

### 4.2 B. 레이어별 expert dtype + `QS4CX_WH_AFFINE` expert (미착수)

현재 코드가 막는 지점(모두 확인함):

1. `Applications/CausalLM/models/gemma4/gemma4_htp_causallm.cpp:219`
   `MOE_LAYER_DTYPE = nntr_cfg.value("moe_layer_dtype", FC_LAYER_DTYPE)` 하나뿐.
   `createMoe(layer_id)`(`:644-669`)가 모든 레이어에 같은 `weight_dtype`를 넘긴다.
   검증은 `gemma4_htp_moe_causallm.cpp:54-61`.
2. `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp:596-616` `expertDesc()`:
   QS4CX_WH / QS2CX_WH / TL2CX_WH만 허용, `ExpertFileDesc.w_bits`만 있고 affine 표시 없음.
   `moeReadExpertWeight()`(`:1001-1030`, CPU 대조 경로)도 8N 꼬리만 읽음.
3. `nntrainer/tensor/htp_backend/htp_compute_ops.cpp`
   - `expertStride(K,N,w_bits)`(`:8536`) = codes + 8N(page 올림). affine은 12N.
   - `takeExpertSlot()`(`:8500-8505`) "프로세스당 슬롯 크기 하나" — 다른 크기면 throw.
   - direct prefetch(`:8237-8246`) "uniform expert shapes" — `w_bits` 다르면 throw.
   - `readWeight()`(`:9355-9440`) 꼬리 2N floats만 읽어 슬롯에 scale/colsum 기록.
   - `registerStagedBatch()`(`:9036-9050`) 배치 첫 desc의 `w_bits`로 DSP entry 선택.
   - `set_fc_wh_source(..., affine)`(`:2677`)는 FC/head용이고 expert 경로와 무관.
4. DSP: `hexkl_mm_u8i4_dma.c` `hexkl_weight_u8i4_tail_from_arena()`(`:169`)가 슬롯
   꼬리에서 scale/colsum 2N만 DMA. offset은 `hexkl_weight_u8i4_set_offsets()`(`:115`)로
   `w->arrays`의 네 번째 N 배열에 복사되며, 현재는 FC 등록 entry의 `bias` 2N 인자
   (`nntr_hvx_mm_u8i4.c:115-125 finish_affine_registration`)로만 들어온다.
   expert swap(`swap_pair`, `:540`)에는 offset 경로가 없다.
5. MoE 커널은 affine expert를 이미 지원한다(`w_offset`, `affine_sums`,
   `hexkl_mm_u8i4_moe.c:1945`, `:2508`, `:3080`, `:3298`). 단 `allow_tail = !n_affine && all_4bit`
   등 일부 빠른 경로가 꺼진다. offset이 전부 0이면 `w_offset = NULL`(대칭 등록)로
   넣는 것이 가장 빠르고 결과도 동일하다.
6. 메모리: expert 하나가 TL2 ≈ 1.33 MB, 4-bit ≈ 3.0 MB(K 2816, inter 704). 풀 슬롯을
   최대 크기로 통일하면 C=16 × 30층에서 +0.8 GB. **슬롯 크기별 free list**로
   가야 한다. `ExpertLru`는 레이어별 capacity 합(`expert_lru.h:12,66`)이라
   레이어 28/29의 eviction은 같은 크기의 슬롯을 돌려주므로 크기별 free list가 맞다.

제안 구현 순서(작은 커밋 단위):

1. `nntr_config.json` 키 `moe_layer_dtype_per_layer`: `{"28": "QS4CX_WH_AFFINE", "29": "QS4CX_WH_AFFINE"}`.
   `Gemma4HtpCausalLM`에 `std::map<int,std::string> MOE_LAYER_DTYPE_PER_LAYER`,
   `moeLayerDtype(layer_id)` 헬퍼, `createMoe`가 그것을 사용. 검증기는 각 값을 기존
   허용 집합 + `QS4CX_WH_AFFINE`로 확인. 문서 `gemma4-htp-user-guide.md` 키 표에 추가.
2. `ExpertFileDesc`에 `bool affine = false;`. `expertDesc()`가 `QS4CX_WH_AFFINE`를
   `w_bits 4, affine true`로 받음. `gemma4_htp_weights.cpp:355-365`(resident 경로
   `register_qs4cx_weight`)도 같은 dtype 허용.
3. `readWeight()`: affine이면 꼬리 3N(scale f32, colsum f32, offset i32)을 읽어
   슬롯에 scale/colsum 뒤 offset 4N을 추가 기록, `ArenaEntry`에 `std::vector<int32_t> offsets`.
   `expertStride`에 affine 인자(12N). `StagedExpert`/`ExpertResident`에 전파.
4. 슬롯 풀: `free_expert_slots_`를 `std::map<uint32_t stride, std::vector<ExpertSlot>>`로.
   `takeExpertSlot(d)`는 d의 stride 리스트에서 꺼내고 없으면 `place()`.
   `expert_slot_bytes_` 단일 검사 제거(또는 stride별). `reserve_qs4cx_wh_expert_slots`는
   첫 레이어 shape로 전부 예약하므로, Gemma4 로더(`gemma4_htp_weights.cpp` preload 루프 앞)에서
   레이어별 shape 수를 세어 stride별로 예약하도록 바꾼다(아니면 28/29는 첫 miss에서 place).
5. direct prefetch(`:8237`): K/inter/N_out만 균일 검사, 슬롯은 desc의 stride로.
   `registerStagedBatch`: 배치를 (w_bits, affine)별로 쪼개 DSP entry 선택.
6. DSP offset 전달: 새 IDL entry 하나
   `weight_set_offsets_batch(in sequence<uint32> h, in sequence<uint32> n_per, in sequence<float> offsets)`
   → 각 handle에 `hexkl_weight_u8i4_set_offsets()`. 호스트는 swap 성공 뒤, offset이
   전부 0이 아닌 handle에만 호출. 전부 0이면 호출하지 않음(대칭 = 비트 동일, 더 빠름).
   stub/skel 재생성 필수.
7. `moeReadExpertWeight()`(CPU 대조)와 `wh_affine_host_check.c`에 expert affine 셀 추가.
8. E2E decode list: `htp_compute_ops.cpp:3908` `l.bits`는 레이어별이라 그대로.
   `pool_descs_[0][0].w_bits`로 reader 기본값을 정하는 `:8222`는 TL2 기준 유지.

### 4.3 C. 혼합 모델에서의 회전 범위 (미착수)

- `NNTR_MOE_HADAMARD_SCOPE` 기본 `2bit`는 2-bit(TL2 포함, `bits == 2`) 쌍만 회전한다.
  레이어 28/29(4-bit)는 회전되지 않는다 → 체크포인트가 모든 MoE 레이어를 회전했다면 틀림.
- `all`은 dense FFN(prefill `gemm_qs4cx_fused_swiglu_fp32`, decode `graph_op_dense_ffn`)의
  1056폭 chunk 호출에도 적용되고, `hvx_hadamard_supported(1056)`가 0이라 `AEE_EUNSUPPORTED`.
- 제안: dense FFN 호출에서 HAD 비트를 마스킹한다. decode는 `hexkl_graph.c:665 graph_wh_flags()`
  (op kind가 `HTP_OP_DENSE_FFN`이면 `flags &= ~HAD_MASK`), prefill은 dense FFN이 쓰는
  skel entry(호스트 `htp_compute_ops.cpp:4906`에서 호출하는 entry 확인)에서 같은 마스킹.
  그 뒤 `scope=all`이 "expert 전부(2-bit + 4-bit), dense FFN 제외"가 된다.
  추가로 nntr_config 키 `moe_hadamard`(값은 env와 동일)를 두어 앱이 env 없이 켜도록 한다
  (Quick.AI는 env 옵션 7개를 넘기므로 env만으로도 가능).
- 회전이 켜지면 4-bit affine expert 경로에서도 `affine_sums`가 회전·양자화된 행의 합이라
  수식은 그대로 맞는다.

### 4.4 D. static activation scale (미착수)

파일에 activation scale 텐서가 없으므로 사이드카로 받는다. 제안 포맷
(`<model>.actscale.json`, 또는 nntr_config `activation_scale_file`):

```json
{
  "format": "nntr_actscale_v1",
  "quant": "u8_asym: u = clamp(round(x / scale) + zero_point, 0, 255)",
  "per_tensor": true,
  "layers": [
    {"attn_in":  {"scale": 0.0123, "zero_point": 131},
     "o_in":     {"scale": ..., "zero_point": ...},
     "ffn_in":   {"scale": ..., "zero_point": ...},
     "ffn_down_in": {"scale": ..., "zero_point": ...},
     "moe_in":   {"scale": ..., "zero_point": ...},
     "moe_down_in": {"scale": ..., "zero_point": ...}}
  ],
  "lm_head_in": {"scale": ..., "zero_point": ...}
}
```

의미: 각 값은 **회전이 적용된 뒤**(moe_in은 `x R_in` 뒤, moe_down_in은 `a R_down` 뒤)
HMX에 들어가는 텐서의 per-tensor u8 비대칭 scale/zero_point. 런타임은 재유도 없이 그대로 쓴다.
양자화 부서에 이 수식과 "회전 뒤 분포"임을 꼭 전달할 것.

런타임이 바꿀 지점(현재 모두 per-row min/max 스캔):

| 경로 | 위치 |
|---|---|
| MoE gate_up 입력 (prefill/decode) | `hexkl_mm_u8i4_moe.c:2618` `hvx_quant_rows_u8_params_rm`, 회전 분기 `:1988` `moe_rotate_input_worker` |
| MoE down 입력 (GeGLU 뒤 requant) | `:779` `moe_dn_worker`, `:985` tail, `:1454/:1466` M=1 |
| FC M=1 decode (q/k/v/o, dense, lm_head WH) | `:3953` `hexkl_mm_u8i4_fc_m1_run` |
| FC prefill (WH) | `hexkl_mm_u8i4_dma.c:1235, :1295, :1475, :1545` |
| u8i8 경로 | `hexkl_mm_u8i8_dma.c:235` (Gemma4 HTP 경로에서 쓰는지 확인) |
| 일반 entry | `nntr_hvx_mm_u8i4.c:88` |

전달 방식 제안: (1) ARM이 JSON을 읽어 세션에 한 번 올리는 RPC
`act_scale_table_set(in sequence<float> scale, in sequence<int32> zp)`; (2) prefill
호출 전 `act_scale_select(in uint32 entry)` 한 줄 RPC(레이어·텐서 인덱스), 커널은
세션의 "현재 entry"가 있으면 스캔을 건너뛰고 `hvx_quant_row_u8_rm(row, K, scale, zp, u8)`로
양자화; (3) decode graph는 op param(`HTP_GRAPH_PARAM_ACT_SCALE`, f32 2개)로 op마다
지정. 값이 없으면 지금처럼 dynamic. host check: static scale = 해당 행의 dynamic scale로
주면 바이트 동일해야 한다.

### 4.5 E. 회전 규약 (컨버터와 맞출 것)

런타임이 하는 일(`hexkl_hadamard_f16.h` 머리말, `hvx_hadamard_f32.c`):
길이 n = 44·p(2816 = 44×64, 704 = 44×16)인 행 x를 44×p 행렬 X(row-major)로 보고
`X ← H44 · X · Hp / sqrt(n)` 을 계산한다. H44는 Paley(F_43) 구성, 첫 행
`[1, -1, -1, ...]`, **비대칭**(SpinQuant `get_had44`와 동일, `generate_hadamard_tables.py`의
`paley()`), Hp는 Sylvester(대칭). 행벡터 x에 대한 Kronecker 표현으로는
`x ← x · (H44ᵀ ⊗ Hp) / sqrt(n)`.

따라서 y = x·W(W는 [K, N], K = 입력 폭)를 보존하려면 컨버터가 양자화 **전에**
`W' = (H44 ⊗ Hp) · W / sqrt(n)` 을 적용해야 한다. 즉 **W의 각 열(K-벡터)을 44×p로
reshape해서 런타임과 똑같이 `H44 · Wm · Hp / sqrt(n)` 을 적용**하면 된다
(활성과 가중치에 같은 변환; H44가 비대칭이라 전치하면 틀린다).
gate_up은 K = 2816(p = 64), down은 K = 704(p = 16).

확인할 것: 체크포인트의 `rotation_meta.json`은 seed 42를 적는다. SpinQuant/QuaRot류가
seed로 **random sign flip(H·diag(±1))**이나 random orthogonal을 쓰면 런타임의 고정
H44⊗Hp와 맞지 않는다. 컨버터 세션에 (1) 실제 R_in, R_down 행렬(또는 생성 코드)을
받아 위 식과 수치 비교(허용 오차 1e-6), (2) HF reference forward에 같은 변환을 넣고
회전 on/off next-token CE가 같은지 확인을 요청한다. 이 저장소에는 아직 그 스크립트가
없다. `hvx_hadamard_f32.c`의 `hadamard_scalar()`가 가장 읽기 쉬운 참조 구현이다.

### 4.6 F. 기기 검증 절차

기기: **S26 Ultra `R5KL309ALRK` (SM-S948N, v81)** — 팩과 설정이 모두 여기 있다.
`R3CY70LV96T`(SM-S948U)도 v81로 쓸 수 있으나 팩을 다시 push해야 한다.
`R3CN80CW3FY`는 우리 기기가 아니다. `adb devices`로 매번 확인.

폰 경로(`/data/local/tmp/nntrainer/causallm/` 아래):

| 경로 | 내용 |
|---|---|
| `models/gemma4_26b_a4b_tl2_qs4cx_wh_affine/` | 현재 TL2 팩(expert TL2CX_WH, FC/head QS4CX_WH_AFFINE, Q4_0 embedding). 새 체크포인트가 오면 `models/<새 이름>/`에 |
| `models/lfm2-24b-a2b-tl2-4bit/` | LFM2 팩(회귀 확인용) |
| `claude_lfm2kv/bin/` | PR head `3f518fa83` 빌드(LFM2 KV 검증에 쓴 것) |
| `claude_lfm2kv/g4_full/` | Gemma4 2k 프롬프트, max 2304 설정. 기준 id: `claude_lfm2kv/ids_g4_full_mine.txt` |
| `claude_lfm2kv/a2k_full/` | LFM2 2k 설정. 기준 id: `ids_a2k_full.txt` |
| `claude_lfm2kv/run2.sh` | `sh run2.sh <bin dir> <cfg dir 절대경로> <tag> [env...]` → `ids_<tag>.txt`, `log_<tag>.txt` |
| `claude_had/bin/` | 이 문서의 Hadamard 포팅 빌드 |

빌드(워크스테이션):

```sh
cd ~/nntrainer-4428-had
source ~/nntrainer/tools/htp/env.sh
export HEX_ARCH=v81 HEXKL_SDK_VER=6.4.0.2 JAVA_HOME=$HOME/jdk/jdk-17.0.20.1+1 ANDROID_HOME=$HOME/android-sdk
bash Applications/CausalLM/build_android.sh --htp        # stub 재생성 + skel + ARM, 약 4분
# 산출물
#  builddir_app/htp/{nntr_causallm,libcausallm.so,libquick_dot_ai_api.so}
#  builddir/android_build_result/lib/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so,libsdkl.so}
#  test/htp/build/libnntr_hvx_skel.so
# host check
bash test/htp/host/run_host_checks.sh
```

기기 실행:

```sh
D=/data/local/tmp/nntrainer/causallm
adb -s R5KL309ALRK push builddir_app/htp/nntr_causallm builddir_app/htp/libcausallm.so \
  builddir_app/htp/libquick_dot_ai_api.so builddir/android_build_result/lib/arm64-v8a/*.so \
  test/htp/build/libnntr_hvx_skel.so $D/claude_had/bin/
# 회전 off: PR head 빌드와 id가 바이트 동일해야 한다
adb -s R5KL309ALRK shell "sh $D/claude_lfm2kv/run2.sh $D/claude_had/bin $D/claude_lfm2kv/g4_full had_off"
adb -s R5KL309ALRK shell "cmp $D/claude_lfm2kv/ids_had_off.txt $D/claude_lfm2kv/ids_g4_full_mine.txt && echo SAME"
# 회전 on (현재 팩은 회전되지 않은 가중치라 출력은 깨져도 됨; 오류 없이 돌고 stderr에 applied 줄이 떠야 함)
adb -s R5KL309ALRK shell "sh $D/claude_lfm2kv/run2.sh $D/claude_had/bin $D/claude_lfm2kv/g4_full had_on NNTR_MOE_HADAMARD=both_hmx NNTR_MOE_HADAMARD_SCOPE=2bit"
adb -s R5KL309ALRK shell "grep -E 'online Hadamard|prefill:|generation:' $D/claude_lfm2kv/log_had_on.txt"
# LFM2 회귀(4-bit expert, 회전 off)
adb -s R5KL309ALRK shell "sh $D/claude_lfm2kv/run2.sh $D/claude_had/bin $D/claude_lfm2kv/a2k_full had_lfm2 && cmp $D/claude_lfm2kv/ids_had_lfm2.txt $D/claude_lfm2kv/ids_a2k_full.txt && echo SAME"
```

새 체크포인트가 오면: nntr_config는 TL2 팩의 것을 복사하고 `moe_layer_dtype_per_layer`(B)
추가, `NNTR_MOE_HADAMARD=both_hmx NNTR_MOE_HADAMARD_SCOPE=all`(C 뒤), activation scale
사이드카(D). 합격 기준은 10:15 코멘트의 Acceptance: 2048 프롬프트 id가 기준과 같고,
petcare sub/main 시나리오가 돈다. 회전된 가중치 + 회전 on에서만 텍스트가 의미 있다.
AI Studio 앱이 떠 있으면 DSP arena를 잡고 있어 CLI가 `AEE_ENOMEMORY`로 실패한다(앱 종료).

## 7. 2026-10-16 밤 세션 결과 (워크스테이션 `shsh1004-lee`, 워크트리 `~/nntrainer-4428-had`)

커밋(모두 `htp/4428-hadamard`, `Seunghui98/nntrainer`에 push됨; `Jungwon-Lee/nntrainer`에는
이 계정에 push 권한이 없어 올리지 못함 — `integrate/pr4416-quickdotai`로는 권한 있는 쪽에서
`git push git@github.com:Jungwon-Lee/nntrainer.git htp/4428-hadamard:integrate/pr4416-quickdotai`):

| 커밋 | 내용 |
|---|---|
| `d0e32095b` | `[CausalLM/Gemma4]` `moe_layer_dtype_per_layer` 키 (B-1) |
| `59825043b` | `[HTP]` affine 4-bit expert: 슬롯 크기별 free list, 12N 꼬리, `weight_set_offsets_batch` IDL, direct prefetch는 첫 레이어 크기만 (B-2..6) |
| `2d9057fb5` | `[HTP]` `scope=all`은 expert만 회전, dense FFN은 act bit 1 / 그래프 마스크 (C) |
| `5d366e5cb` | `[HTP]` 디코드 M=1 native TL2 down 분기에 빠져 있던 `moe_rotate_rows` (포팅 버그) |
| `2edf1c76f` | `[HTP]` 회전 부호 사이드카 `moe_hadamard_signs` + `moe_set_hadamard_signs` IDL + `tools/htp/recover_hadamard_signs.py` (E) |

### 7.1 회전 규약 (E) — 확정

두 팩의 같은 expert(레이어 28 expert 0 등)를 디코드해 비교: W_new 와 (H44⊗Hp)·W_tl2/√n 의
행 상관은 |r| 평균 0.76(모든 행 > 0.3), 그 중 절반(gate_up 1395/2816, down 358/704)이 음수.
즉 런타임의 H44⊗Hp 방향은 맞고, 체크포인트는 그 뒤에 **입력 차원별 ±1 부호**(seed 42)를
곱한다: `x ← (x H) ⊙ s_in`, `a ← (a H) ⊙ s_down`. 부호 벡터는 expert 0/1, 30개 레이어 전부
동일. (H44ᵀ⊗Hp 후보의 |r| 0.725는 H44ᵀ = 2I − H44 에 의한 반상관으로 설명됨; Hp⊗H44
순서는 r ≈ 0.) 복원 도구: `tools/htp/recover_hadamard_signs.py` (두 팩의 .bin과
`NNTR_MOE_DESC_TRACE=1` 로그로부터 사이드카 생성). 기기 사이드카:
`/data/local/tmp/nntrainer/causallm/claude_had/rotation_signs.json`.
**컨버터 팀에 요청**: `rotation_meta.json`의 부호 벡터(또는 생성 코드)를 팩에 함께 싣고
`nntr_config.json`에 `"moe_hadamard": "both", "moe_hadamard_scope": "all",
"moe_hadamard_signs": "rotation_signs.json"` 을 넣을 것. 데이터 복원본과 일치해야 한다.

### 7.2 체크포인트(바이너리) 쪽 문제 두 가지

1. `config.json`에 `global_head_dim: 512`, `num_global_key_value_heads: 2` 가 없어
   (per_layer_config만 있음) 런타임이 `set_decode_graph_desc: AEE_EINVALIDFORMAT`로 즉시
   실패한다. 기기 설정 복사본(`claude_had/cfg/g4new_*/config.json`)에 두 키를 넣어 우회.
   컨버터가 TL2 팩의 config.json처럼 두 키를 써야 한다.
2. 레이어 28/29를 4-bit 컨테이너에 넣은 값은 ternary다(10:21 코멘트, 디코드로도 확인:
   레벨 {-1,0,1}, offset 전부 0). **TL2CX_WH로 저장하면** B의 모든 런타임 분기가 필요 없고
   메모리도 TL2 팩과 같아진다. 지금은 4-bit 슬롯(3.0 MB)이 프리필 read-ahead 때문에
   28·29 두 레이어분 256개(768 MB)까지 깔려 arena가 8→12 chunk(1792→2624 MiB)다.
   (공유 LRU에 "같은 클래스 먼저 비우기"를 넣어 봤으나 피크는 read-ahead가 결정해 효과 없음,
   디코드에서 대량 eviction 부작용이 있어 되돌림.)

### 7.3 기기 결과 (S26 Ultra R5KL309ALRK, C=16, 바이너리 `claude_had/bin11`)

새 체크포인트, `both` / `scope=all` / 부호 사이드카, 텍스트는 네 셀 모두 Ardley 요약문
(EOS로 82–104 토큰에서 끝남; 512 budget은 채우지 않음):

| 셀 | prefill tok/s | decode tok/s | MemAvailable drop (MiB) | 앱 peak memory (MB) | arena |
|---|---|---|---|---|---|
| p1024 g128 | 520 | 12.6 | 4858 | 4477 | 12 chunk 2624 MiB |
| p2048 g128 | 614 | 12.6 | 4820 | 4969 | 〃 |
| p4096 g128 | 561 | 11.9 | 5344 | 5479 | 〃 |
| p2048 g512 | 573 | 12.4 | 5004 | 4803 | 〃 |

위 표는 연속 측정이라 발열 영향이 있다. 식힌 뒤 두 팩을 번갈아 2회 돌린 p2048 g128
(`bin11`, 60 s 휴식): TL2 팩 704 / 696 prefill, 16.6 / 16.7 decode; 새 팩 654 / 668 prefill,
15.4 / 15.1 decode — 새 팩이 prefill ~6 %, decode ~8 % 느리다(4-bit 슬롯의 28/29 레이어와,
부호 때문에 HVX로 도는 회전). 메모리: TL2 팩 MemAvailable drop 3887 / 4422 MiB (p2048 /
p4096), 앱 peak 4140 / 4176 MB; 새 팩은 arena +832 MiB(7.2절 2).

회귀: 모든 커밋 뒤 TL2 팩(회전 off) 2048 토큰 id가 PR head와 바이트 동일, LFM2 2k id 동일,
TL2 팩 `both_hmx`/`scope=all` 실행 OK. host check 전체 통과(`SET OFFSETS BATCH`,
`HADAMARD SIGNS` 셀 포함).

### 7.4 D (static activation scale) — 미착수, 설계 메모

4.4절 그대로. 구현 시 carrier는 `hexkl_moe_scratch`의 per-call 필드(부호 벡터처럼 세션
전역이 아니라 레이어별 값이므로): prefill은 레이어 호출 전 `moe_act_scale_set` RPC 한 줄,
디코드는 MOE op param. 커널의 스캔 지점은 4.4절 표. HMX fp16 회전 경로(had_mode 3)는
양자화가 융합돼 있어 static scale도 그 안에서 적용해야 한다(부호 적용 때는 HVX로 우회했음).

## 5. 열린 질문

1. ~~체크포인트의 R_in/R_down이 고정 H44⊗Hp인지, seed 42가 sign flip을 뜻하는지(4.5절).~~
   → 7.1절: H44⊗Hp + 입력 차원별 ±1 부호(seed 42). 컨버터의 `rotation_meta.json`과 대조 필요.
2. 레이어 28/29가 정말 4-bit로 남는지 — 값은 ternary(확인). TL2로 저장하면 B 전체와
   +832 MiB가 사라진다 — 컨버터 쪽 결정(7.2절).
3. activation scale 사이드카 포맷(4.4절)과 산출 조건(회전 뒤, per-tensor).
4. LFM2 27k 프롬프트 텍스트 퇴화, Gemma4 chunk-2 q2 커널 문제(오늘 별도 분석)는 이 작업과 무관하게 남아 있다.

## 6. 다음 세션 시작 프롬프트

(2026-10-16 밤 이후: A–C, E 완료. 남은 것은 D와 컨버터 쪽 7.2절. 아래는 그 전의 프롬프트.)

```
~/nntrainer-4428-had (브랜치 htp/4428-hadamard, PR #4428 head 위)에서
docs/gemma4-htp-rotated-ternary-handoff.md 를 먼저 읽고 이어서 작업해줘.
A(Hadamard 포팅)는 끝났으니 4.6절의 기기 검증 셀(R5KL309ALRK)을 먼저 돌려 PR head와
회전 off 바이트 동일을 확인하고, 그 다음 4.2절 B(레이어별 expert dtype + QS4CX_WH_AFFINE
expert, 슬롯 크기별 풀, DSP offset entry)를 1단계부터 작은 커밋으로 구현해줘.
그 뒤 4.3절 C(dense FFN 마스킹으로 scope=all 안전화), 4.4절 D(static activation scale
사이드카)를 진행해. 커밋은 `[<component>] <subject>`, `git commit -s`,
Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>, clang-format-14는 바뀐 줄만,
push는 git push git@github.com:Jungwon-Lee/nntrainer.git HEAD:integrate/pr4416-quickdotai.
기기 결과는 셀마다 바로 보고해줘.
```
