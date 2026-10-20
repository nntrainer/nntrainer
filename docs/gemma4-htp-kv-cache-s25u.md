# Gemma4 26B-A4B HTP KV cache 저장·로드 — S25 Ultra 실행 기록 (2026-10-15)

이 문서는 PR #4428 head `6987da2db` 위에서 Gemma4 26B-A4B native TL2 / affine WH
pack으로 **KV cache dump 저장 → 로드 → 남은 프롬프트 prefill → decode**를 S25
Ultra(v79)에서 처음 실행한 기록이다. S26 Ultra가 없어서 S25 Ultra에서 했다.
이하의 수치는 당시 S25 Ultra 실행 기록이다.

**2026-10-18 상태:** 현재 native CLI는 KV snapshot과 chunked prefill을
지원하며, public QuickDotAI API는 snapshot/resume을 노출하지 않는다.
최신 native TL2/affine 런타임, compact expert 저장, 메모리 회수와 S26 검증
결과는 [TL2 가이드](gemma4-htp-tl2-guide.md#current-validation-coverage)를 따른다.
아래의 S25 장문 측정은 최신 코드로 다시 실행한 결과가 아니다. 이후 S26
snapshot/demo 조사에서는 expert-load rebind 실패가 한 번 발생하고 11회
반복에서 재현되지 않았으며 원인은 미확정이다. int8 KV scale은 calibration에
참여한 행에 의존하므로 snapshot/chunked 실행과 단일 prefill의 생성 ID가
달라질 수 있다. 7절은 이 측정 시점의 후속 과제로 읽는다.

측정은 모두 기기 실측이고, 한 번 또는 두 번 실행한 값이다(번갈아 3회 규칙은
지키지 않았다). 실행하지 않은 것은 "미실행"으로 적는다.

## 1. 목표와 시나리오

에이전트 프롬프트를 둘로 나눈다. 바뀌지 않는 앞부분(static: 시스템 프롬프트,
tool 선언)은 한 번 prefill해서 KV cache 파일로 저장한다. 실행할 때는 그 파일을
로드하고 바뀌는 뒷부분(dynamic: 사용자 요청, 이전 tool 결과)만 prefill한 뒤
생성한다.

| 시나리오 | static | dynamic | 필요 context |
|---|---:|---:|---:|
| sub agent | 1,537 토큰 | 16 토큰 | ~2.1k |
| main agent | 23,063 토큰 | 6,641 토큰 | ~30.1k |

입력은 사내 데모 자산(`demo/ver3/petcare_gemma4/{sub_agent,main_agent}/input_{static,dynamic}.txt`)이며
이 저장소에 넣지 않는다. 토큰 수는 모델 `tokenizer.json`으로 셌다.
`generation_config`의 EOS는 `1, 106, 50`(50 = `<|tool_response>`)이다.

## 2. 환경

| 항목 | 값 |
|---|---|
| 기기 | Galaxy S25 Ultra `R3CY205ZMND`, SM-S938N, SM8750, Hexagon **v79**, RAM 11.1 GB |
| 모델 | `/data/local/tmp/nntrainer/causallm/models/gemma4_26b_a4b_tl2_qs4cx_wh_affine/` (`nntr_gemma4_ternary_tl2_qs4cx_wh_affine.bin`, 6,604,559,480 B; expert `TL2CX_WH`, FC/head `QS4CX_WH_AFFINE`, embedding `Q4_0`) |
| 코드 | PR #4428 `6987da2db` + 이 문서와 함께 올린 커밋 둘(4절) |
| 도구 | Hexagon SDK 6.4.0.2, HexKL `lib/6.4.0.2`, NDK r26d |

### 2.1 빌드 (v79)

`Applications/CausalLM/build_android.sh --htp`는 `HEX_ARCH`가 v81이 아니면 멈춘다.
ARM 쪽은 아키텍처와 무관하므로 v81로 스크립트를 통과시킨 뒤 skel만 v79로 다시
만든다. 스크립트는 고치지 않았다.

```bash
export ANDROID_NDK=$HOME/workspace/android-ndk-r26d ANDROID_NDK_HOME=$ANDROID_NDK
export PATH=$ANDROID_NDK:$PATH
unset HEXAGON_SDK_ROOT; source $HOME/workspace/Hexagon_SDK/6.4.0.2/setup_sdk_env.source >/dev/null
export HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon HEXKL_SDK_VER=6.4.0.2

HEX_ARCH=v81 bash Applications/CausalLM/build_android.sh --htp   # ARM 라이브러리, nntr_causallm
(cd test/htp && HEX_ARCH=v79 ./build.sh)                          # "ARCH OK (V79)" 확인
# 이후 ARM만 바꿨으면: ninja -C builddir_app/htp
```

S26 Ultra면 두 번째 줄이 필요 없다(`HEX_ARCH=v81` 그대로).

### 2.2 기기 배치

```bash
D=/data/local/tmp/nntrainer/causallm; R=$D/s25kv
adb shell mkdir -p $R
for f in builddir_app/htp/{nntr_causallm,libcausallm.so,libquick_dot_ai_api.so} \
         builddir/android_build_result/lib/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so,libsdkl.so} \
         test/htp/build/libnntr_hvx_skel.so; do adb push $f $R/; done
adb shell chmod 755 $R/nntr_causallm
```

### 2.3 config

시나리오마다 디렉터리 하나: 모델의 `nntr_config.json`을 복사해 아래만 바꾸고,
`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`은
모델 디렉터리로 심링크한다. `model_file_name`, `tokenizer_file`은 절대 경로로 둔다.

| 키 | sub (`pc_sub`) | main (`pc_main`) |
|---|---|---|
| `init_seq_len` | 2048 | 2048 |
| `max_seq_len` | 2304 | 30272 (23,063 + 6,641 + 512, 64 배수) |
| `num_to_generate` | 512 | 512 |
| `moe_cache_experts` | 16 | 8 |
| `bad_word_ids` | `[]` | `[]` |
| `sample_input` | dynamic 텍스트 | dynamic 텍스트 |
| `system_prompt.head_prompt` | static 텍스트 | static 텍스트 |
| `system_prompt.tail_prompt` | `""` | `""` |
| `system_prompt.kvcache.pre_computed_cache_path` | `$D/pc_sub/static.kv` | `$D/pc_main/static.kv` |

동작: `pre_computed_cache_path` 파일이 **없으면** SAVE 모드(`head_prompt`만 prefill,
`<path>`와 `<path>.q2` 저장 후 종료, decode 없음), **있으면** USE 모드(로드 →
`sample_input + tail_prompt` prefill → decode). `chat_input`을 쓰면 head/tail이
지워지므로 쓰지 않는다. 비교용 `*_full`은 `system_prompt` 없이
`sample_input = static + dynamic`.

### 2.4 실행

```sh
# $D/petcare.sh <sub|main> <save|use|full>
D=/data/local/tmp/nntrainer/causallm; R=$D/s25kv; C=$D/pc_$1
case "$2" in
  save) rm -f $C/static.kv $C/static.kv.q2 ;;
  use)  [ -f $C/static.kv ] || { echo "no dump"; exit 1; } ;;
  full) C=${C}_full ;;
esac
cd $R && LD_LIBRARY_PATH=$R ADSP_LIBRARY_PATH=$R NNTR_NUM_THREADS=8 NNTR_MOE_CACHE_GROW=0 \
  ./nntr_causallm $C
```

볼 줄: `[TIMING] kv_load_ms`, `prefill:`, `generation:`, `[PERF-SPLIT]`.
`NNTR_MOE_CACHE_GROW=0`은 main에 필수다(5.3절).

## 3. 결과

### 3.1 decode가 v79에서 도는가

| 실행 | prefill | decode | 확인 |
|---|---|---|---|
| 짧은 질문 19토큰, C=16 | 961 ms | 13토큰, 12.2 TPS, EOS 종료 | 기기 다운 없음 |
| `docs/validation/gemma4-tl2-prefill-p2048.txt`, C=16, `NNTR_HTP_PAGE_CACHE=0 NNTR_MOE_CACHE_GROW=0` | 2048토큰 4,035 ms (507.6 TPS) | 101토큰 11.1 TPS | **출력 ID 102개가 S26 Ultra 기록(`gemma4-tl2-affine-decode-s26-20261015.json`)과 전부 동일** |

`6987da2db` 이전 head(`62fdb7145`)는 S26 Ultra에서 첫 decode가 기기 다운으로
끝났다. 그 head로 S25 Ultra를 돌린 적은 없다.

### 3.2 sub agent (1,537 + 16)

| 실행 | 결과 |
|---|---|
| save | dump 346,255,488 B + `.q2` 234,544 B, 전체 6.3 s |
| use | `kv_load_ms` 156–209, prefill 16토큰 1,213–1,219 ms, TTFT 1.29 s, decode 512토큰 15.2–17.3 TPS (steady 55.4 ms) |
| full (cache 없음) | prefill 1,553토큰 3,020 ms (514 TPS), decode 17.4 TPS |

### 3.3 main agent (23,063 + 6,641)

| 실행 | 결과 |
|---|---|
| save | 23,063행, dump 682,045,568 B + `.q2` 234,544 B, 전체 70.2 s |
| use | `kv_load_ms` 292–684, prefill 6,641토큰 21.9–25.0 s (266–304 TPS, 2,048행 청크 4번), TTFT 23.8–26.6 s, decode 512토큰 6.58–6.69 TPS (steady 146–148 ms), peak 5.59 GB (RssAnon + accel) |
| full | 미실행 |

### 3.4 생성 문장

출력 품질은 이번 범위 밖이다(데모 사용자 지시). 다만 관찰한 것은 적는다.

- sub: 망가진다. 첫 토큰이 `<|channel>`이 아니라 `s`이고, 이어서 같은 줄을 반복한다.
  cache 없이(`full`) 돌려도, `attention_engine=cpu`로 int8 HTP attention을 빼도
  똑같다. 그러므로 KV 저장·로드나 attention 양자화 때문은 아니다.
- main: 앞부분은 맞다(`thought` 채널, `petcare` skill 선택). 그 뒤 같은 bullet 두
  줄을 번갈아 반복한다. config가 greedy(`do_sample: false`)이고
  `repetition_penalty`가 없다.
- 위 3.1에서 2k 요약 프롬프트는 S26 Ultra와 ID가 같았으므로 v79 계산 경로는 그
  범위에서 S26과 같다. 남은 후보는 ternary weight 자체와 greedy 설정, 그리고 30k에서의
  int8 decode attention 정밀도(검증 안 됨)다. 원본 ternary checkpoint를 PyTorch로
  같은 프롬프트에 돌려 비교하면 확정된다(미실행: 이 워크스테이션에 checkpoint와 GPU가 없다).

## 4. 코드 수정

PR #4428 head에서는 이 시나리오가 세 군데에서 막혔다. 수정은 모두 host 쪽(ARM)이고
DSP 코드는 바꾸지 않았다.

1. **`Gemma4HtpMoECausalLM::run()`이 config의 KV cache 설정을 지운다.** 매 실행
   `setSessionTokenPosition(0)`을 불러 `USE_KVCACHE`를 끈다. 그래서 save도 load도
   하지 않고 static + dynamic 전체를 prefill했다(sub에서 확인). config가 cache를
   지정했으면 `setPrecomputedKVCache(path, SYS_PROMP_LEN)`으로 유지한다.
2. **init_seq_len보다 긴 프롬프트가 잘린다.** `Gemma4HtpTransformer`는 KV ring을
   쓰지 않으므로 `prefillDriveChunk()`가 0이고, 프롬프트가 `init_seq_len`(2048)에서
   경고와 함께 잘렸다. `chunkLongPrompts()`를 true로 해서 2048행 청크로 넣는다.
   sliding 층의 cache가 window + 청크 한 개(1024 + 2048 = 3072행)로 잡히는 것과
   맞는 동작이다.
3. **긴 static을 저장할 수 없고, 청크 사이에서 window가 움직이지 않는다.**
   SAVE 모드는 프롬프트 전체를 한 번의 `incrementalInference`로 넣었다.
   청크 prefill 루프는 청크마다 `compactKVCache`를 부르지 않아 sliding 층이 3072행을
   넘어 썼을 것이다. SAVE를 같은 청크 루프로 바꾸고, 두 루프 모두 청크 앞에서
   `compactKVCache(청크 시작 위치)`를 부른다. 둘 다 PR #4416 쪽 원본
   (`cc0a155e0`, `e09715acd`)에 있던 것이 port에서 빠진 것이다.

### 4.1 v81(S26 Ultra)에서도 맞는가

S26 Ultra에서는 **실행하지 않았다**. 코드를 읽어 확인한 것:

- 세 수정은 아키텍처와 무관한 host 코드다. v79/v81 차이는 DSP 커널 안의
  `__HEXAGON_ARCH__` 분기뿐이고(5.4절), 수정은 그 경로를 바꾸지 않는다.
- 청크 크기가 0인 모델(Gemma4 HTP 외 CPU/GPU 모델 대부분)은 SAVE가 예전처럼
  한 번에 돈다. `compactKVCache`는 windowed 층이 없으면 아무것도 하지 않는다.
- ring 모델: SAVE 청크가 0부터 C 단위로 나뉘므로 ring의 경계 조건(청크가 C의 배수를
  넘지 않음)을 지킨다.
- QuickDotAI API는 `run(input, …, "", "")`로 system prompt 없이 부르므로 SAVE는
  일어나지 않는다. 앱 pack의 `nntr_config.json`에 `system_prompt.kvcache`가 있으면
  이제 그 파일을 로드하려 한다(파일이 없으면 load가 실패) — 일반 `CausalLM`과 같은
  동작이다. 현재 `htp_chat` preset과 SampleTestAPP pack에는 그 키가 없다.

알려진 한계(코드는 그대로 둠):

- `do_prefill`의 한 블록 경로(프롬프트 ≤ 청크)는 compaction을 부르지 않는다.
  로드 직후에는 `load_kvcache`가 이미 compaction을 하고, Gemma4 HTP의 `run()`은
  매 프롬프트 cache를 새로 잡으므로 현재 도달하는 경로에서는 문제가 없다. 한 세션에서
  여러 턴을 이어 쓰게 되면 다시 본다.
- SAVE 모드는 마지막 청크의 logits 버퍼를 돌려주지 않고 끝난다(수정 전과 같음,
  프로세스가 바로 끝난다).
- host 단위 테스트는 추가하지 않았다. 검증은 3절의 기기 실행이다.

### 4.2 windowed cache의 decode 출력 불일치 (2026-10-19, S26 Ultra)

`max_seq_len` > 3072 (sliding 층이 3072행 window cache로 바뀌는 지점)에서만 decode
토큰이 달라지던 문제(PR #4428 코멘트 10-16, "starts right, then repeats")의 원인과 수정.

- **원인**: decode attention(`attn_m1`)의 int8 cache scale은 층별로 **첫 seed에 들어온
  행들의 |max|** (kv head × dim마다, `q8_set_scales_rows`)로 한 번 정해진다. full
  cache는 프롬프트 전체 행을 seed하지만 windowed cache는 compaction 뒤 남은 window
  (1024행)만 seed하므로 scale이 달라지고, 9번째 토큰부터 갈렸다(p2048, ids md5
  `dcdc84441c` vs `1dd27d8bf5`; 3328과 4096은 서로 같음 = windowed 쪽은 결정적).
  prefill·첫 토큰은 같다.
- **수정**: `KVCacheManager::compact`가 버리는 행들의 열별 |max|를 접어 두고
  (`droppedAmaxK/V`, fp16 비트로 max → 열마다 한 번 f32), `compactKVCache`가 그 값을
  `HtpMHACoreLayer::foldDroppedAmax`로 층에 넘긴다. 층은 decode seed 직전에
  `attn_m1_kv_amax_floor`(새 IDL 메서드)로 DSP에 보내고, DSP는 scale을
  `max(seed 행 |max|, floor)`로 잡는다. max는 순서와 무관하게 정확하므로 full cache의
  seed와 **비트 동일**한 scale이 된다. floor는 `.q2` 사이드카에 calibration 뒤에
  붙어 저장되고(`save_kvcache`가 저장 전 compaction을 한 번 더 해 window 밖 행을
  접는다), USE 실행이 그대로 이어받는다. full cache(아무 행도 버리지 않음)는 floor가
  비어 있어 바이트 하나 바뀌지 않는다.
- **결과** (p2048 요약 프롬프트, C=16, bin = PR head + 수정): `max_seq_len` 2304 /
  4096 / 30272(C=8, `NNTR_MOE_CACHE_GROW=0`) 모두 ids md5 `dcdc84441c`. 30272의
  prefill 622 TPS, decode 9.3 TPS(C=8), peak 5.25 GB — 메모리는 여전히
  `max_seq_len`에 비례해 로드 때 잡힌다(한 config로 합치는 일의 남은 과제).
- **검사**: `test/htp/host/attn_m1_host_check.c`의 `check_amax_floor`(window seed +
  floor = full seed의 scale·출력 바이트, floor 없는 대조군은 달라야 함),
  `unittest_kv_cache_manager`의 `compact_folds_the_dropped_rows_amax`.

**20k prefill에서의 확인** (2026-10-19 밤, 같은 기기, C=8 + `NNTR_MOE_CACHE_GROW=0`
`NNTR_HTP_PAGE_CACHE=0`, 128 토큰 greedy). 20k에서는 full cache가 DSP에 올라가지 않으므로
직접 비교 대신 세 가지로 확인했다.

| 구성 | prefill 토큰 | ids md5 |
|---|---|---|
| 22528 windowed, 수정 후 | 19,270 | `3a6276ddfc` |
| 30272 windowed, 수정 후 | 19,270 | `3a6276ddfc` |
| 22528 windowed, 수정 전(PR head) | 19,270 | `1d3df1379a` (2번째 토큰부터) |
| full 3072 vs windowed 3328, `init_seq_len` 512 (compaction 여러 번) | 1,645 | 둘 다 `110f13888b` (수정 전 `be575fec7e`) |
| save 17,590행 → use +1,680행, floor 있음 / `.q2`에서 floor를 지움 | | `415ec37813` / `d5923d5fb5` |

- 수정 후는 `max_seq_len`에 무관하고, compaction이 여러 번 일어나도 full cache와 비트
  동일하며, USE 실행은 `.q2`의 floor를 실제로 읽어 쓴다(지우면 ids가 바뀐다). 2k에서
  9번째 토큰이던 분기가 20k에서 2번째인 것은 window가 19,270행 중 1,024행만 seed해
  scale 차이가 훨씬 컸기 때문이다. USE가 단일 pass와 다른 것(2번째 토큰부터)은
  prefill 쪽 calibration 차이로, window와 무관한 기존 항목이다.
- 20k에서의 성능: prefill 385~394 TPS, decode 7.3~7.9 TPS, peak 약 4.9 GB (C=8).
- **절대 정확도는 결론 내리지 못했다.** CPU attention을 기준(prefill 423 s)으로 두면
  수정 후는 1번째, 수정 전은 4번째 토큰에서 갈리는데 greedy 한 샘플이고 그 뒤로는 양쪽
  다 거의 맞지 않아(75개 중 2 / 8) 의미가 없다. 요약이 문서의 어느 부분을 다루는지도
  near-tie에서 뒤집힌다(floor 유무에 따라 같은 초점이 양쪽으로 오갔다).
- 20k 덤프(`kvsu20k.bin` + `.q2`)를 뜯어 보면 floor(버린 행 |max|)는 window 행
  |max|의 중앙값 1.25배(일부 열 최대 6배, BOS 이상치 없음)다. 그래서 sliding 층 window
  행의 int8 상대 오차는 window-local scale 0.65 % → floor scale 0.82 %로 1.25배 커진다.
  즉 이 수정은 full cache를 정확히 재현하지만, full cache 자체가 전체 행 |max|로 scale을
  잡는 설계라 수정 전 windowed 경로가 양자화는 조금 더 촘촘했다. 둘 다 1 % 미만이다.
  정확도를 우선한다면 compaction 때 window 기준으로 scale을 다시 잡는 쪽이 둘 다보다
  낫지만, "`max_seq_len`에 무관"이라는 요구와는 다른 선택이라 여기서는 하지 않았다.

프롬프트 예산도 같은 날 고쳤다: 로드한 KV prefix 행(`SYS_PROMP_LEN`, 이어 쓰는 세션의
`global_token_len`)을 `max_seq_len - num_to_generate`에서 빼고 남는 만큼만 프롬프트에
준다. prefix가 있는데 프롬프트가 그 안에 안 들어가면 **cache 로드와 prefill 전에**
`std::runtime_error`로 거부한다(메시지에 prefix 행 수, 프롬프트 토큰 수,
num_to_generate, max_seq_len, 남는 자리). 전에는 예산이 prefix를 몰라 첫 qkv 단계에서
`rows 1537..2482 past the RoPE table's 2304`로 죽었다. prefix 없는 프롬프트는 전처럼
경고와 함께 꼬리를 자른다.

## 5. 운영상 주의

### 5.1 처음 배치에서 본 오류

첫 배치(`62fdb7145` 빌드)는 `Factory::create`에서
`Key is not found for the object. Key: embedding_layer`로 끝났다. `6987da2db`로
다시 빌드하고 ARM 라이브러리와 skel을 다시 올린 뒤에는 재현되지 않았다. 원인은
확인하지 못했다. 다시 나오면 `builddir_app/htp`의 `.so`와 기기의 `.so` md5부터 맞춘다.

### 5.2 C(층당 상주 expert 수)

`NNTR_MOE_CACHE_EXPERTS` 환경 변수는 이 head에서 효과가 없다(miss 수 동일).
`nntr_config.json`의 `moe_cache_experts`로 바꾼다.

| main use, 64토큰 생성 | C=8 | C=16 |
|---|---:|---:|
| MOE us/token (miss 대기 포함) | 44,362 | 38,354 |
| miss/token | 157.7 | 115.1 |
| decode TPS | 6.13–6.20 | 6.29 |
| peak | 5.59 GB | 5.95 GB |

### 5.3 `NNTR_MOE_CACHE_GROW`

기본값(켜짐)이면 main use의 두 번째 토큰에서 pool을 늘리다가 실패한다:
`HTP arena: cannot place an expert slot (1 MiB). fastrpc_mmap(16 MiB) failed: err=1`
(로그: `pool: 8 a layer … room for 29 more a layer`, `+26 experts a layer from the
second token`). 30k context에서는 DSP 매핑 공간이 KV cache에 쓰이므로 `=0`으로 끈다.

## 6. 어디에 시간이 드는가 (main use, S25 Ultra)

`NNTR_LAYER_PROFILE=1 NNTR_HTP_ATTN_TRACE=1`, C=8, 64토큰 생성. 계측이 켜진 실행이다.

### 6.1 decode — 토큰당 141.7 ms

| op | us/token | 비중 |
|---|---:|---:|
| ATTN_M1 | 72,314 | 51 % |
| MOE (그중 miss 대기 21,064) | 44,362 | 31 % |
| FC | 11,099 | 8 % |
| LM_HEAD | 7,257 | 5 % |
| ROUTER_TOPK | 3,348 | 2 % |
| RMSNORM 외 | ~2,400 | 2 % |

같은 커널이 2k(sub, smoke)에서는 ATTN_M1 2.0 ms다. 늘어난 70 ms는 full-attention
층 5개가 29.7k행을 읽는 몫이다: 층당 K·V int8 약 61 MB, 토큰당 약 304 MB → 실효
약 4.2 GB/s. DDR 대역폭에 한참 못 미치므로 커널 쪽 여지가 있다(추정, 미측정).

### 6.2 prefill — 6,641토큰 25.0 s

| 구간 | 호출 | 시간 |
|---|---:|---:|
| full 층 attention (`hexkl_attn_q2`, 24–30k 컬럼, segment merge) | 20 | 7,538 ms (kernel 7,031; 호출당 410–460 ms, 그중 softmax ~190 ms, wait ~193 ms) |
| sliding 층 attention (3,072 컬럼) | 100 | 1,250 ms |
| 나머지 (MoE, QKV, FC 등) | | ~16.2 s (≈ 410 TPS) |

로드 뒤 full 층 mirror 재append(25k행)는 호출당 17 ms로 작다.

## 7. S26 Ultra에서 할 것

1. 위 2절대로 v81로 빌드해(skel 재빌드 불필요) sub/main save → use를 그대로 돌린다.
   S26 Ultra에서는 C, `NNTR_MOE_CACHE_GROW` 한도가 다를 수 있다(DSP 매핑 한도가 다름).
2. 출력 품질: `main use`와 `main full`의 `NNTR_GENERATED_IDS`를 비교한다(같으면
   KV 경로 무관). `repetition_penalty` 또는 sampling으로 반복 루프가 사라지는지 본다.
   원본 ternary checkpoint의 PyTorch greedy 출력과 첫 토큰부터 비교한다.
3. decode ATTN_M1(6.1절): S26 Ultra에서 같은 30k 조건으로 per-kind 시간을 먼저 잰다.
   `hvx_attn_m1_f32.c`의 `NNTR_ATTN_Q8_NARROW_PREFETCH`(v81 이상)와
   `NNTR_ATTN_Q8_GQA8_REUSE`(v81만)는 S26 Ultra에서 이미 켜져 있다. S25 Ultra에서
   켜는 실험은 v79 skel을 `HEX_EXTRA_CFLAGS='-DNNTR_ATTN_Q8_NARROW_PREFETCH=1
   -DNNTR_ATTN_Q8_GQA8_REUSE=1'`로 빌드하는 데까지 했고(컴파일 통과) 측정하지 않았다.
4. prefill full 층 attention(6.2절): softmax와 wait가 각각 절반. segment 크기와
   K/V DMA 대기를 나눠 잰 뒤 손댄다.
5. 메모리: peak 5.6 GB 중 host fp16 KV(full 층 5 × 30,272행 × 1024 × 2 × 2 B ≈
   620 MB, sliding 25 × 3,072행 × 2048 × 2 × 2 B ≈ 630 MB)가 prefill 동안 상주한다.
   decode 시작 때 해제되는지(`rss_mib` 614 MB로 내려감)는 확인했다.
