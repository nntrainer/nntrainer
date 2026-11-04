# NPU KV radix prefix 인수인계 — 2026-11-04

## 현재 상태

NPU Gemma4 MoE의 **opt-in `q8_row` radix 모드**를 구현하고 Android에서
fresh/reused 출력 ID·최종 위치 일치를 검증했다. 기존 `q8` 설정을 그대로
유지하는 투명한 prefix 재사용이나 무성능회귀까지 달성한 것은 아니다.

- 실제 작업 경로: `/tmp/nntr-pr4428-validation`
- 기본 repo `/home/seungbaek/projects/nntrainer`는 별도 checkout이다.
- 브랜치: `kv/pr4428-dsp-flash-20261104`
- 원격: `baek2sm/nntrainer`
- 구현 커밋: `02ef9a424` — `[HTP] Restore causal prefill pages for exact radix reuse`
- 앞선 초기화 수정: `de249d40c` — 독립 요청 calibration 초기화
- 상세 설명: `docs/gemma4-htp-radix-prefix.md`
- 원본 입력·출력 ID·위치·성능·바이너리 해시:
  `docs/validation/gemma4-htp-radix-20261104.json`

## 구현 및 검증한 것

- 압축 radix tree가 공통 64-token 페이지를 검색한다. 1024-token source에서
  768-token 별도 실행/전체 체크포인트 없이 768-token prefix를 재사용한다.
- immutable FP16 **prefill 페이지**와 DSP INT8 **decode 세션 페이지**를
  분리한다. 각 radix 경계는 페이지를 공유하며 전체 KV를 중복 복사하지 않는다.
- prefix hit는 FP16 prefill 상태를 복원하고 새 suffix를 HMX에서 처리한 뒤,
  fresh와 같은 방식으로 DSP INT8 decode KV를 seed한다.
- `q8_row`의 행별 양자화와 고정 attention reduction tile을 사용한다.
  전체 projection/FFN을 64-token 단위로 잘라 실행하는 느린 실험안은 폐기했다.
- 호환되지 않는 설정이나 attention fallback은 명시적으로 실패한다.
- host prefix 페이지는 prefix LRU가 보존한다. decode 세션 때문에 계속
  pin되지 않는다. prefix clear 후 host bytes=0, 활성 DSP KV 유지 검사 통과.
- Android 길이 행렬: 공통 길이 64/256/750/768/960/2048의 두 분기씩,
  총 12개에서 fresh와 출력 ID·위치 일치. 750 공통 길이는 704를 재사용한다.
- 최종 guarded 빌드에서 768/2048의 네 분기, 두 대화 A/B 전환·fork,
  Flash save/park/restore를 다시 검증했다.
- source/fresh/reused의 앞 768-token K/V 해시는 30개 레이어 모두 일치했다.
- Ubuntu KV unit test 36개, Ubuntu/Android prefill 페이지 및 calibration
  회귀 검사, DSP page/session/Flash host 검사와 ASan/UBSan 통과.

## 측정 결과와 미충족 사항

단위 ms, 두 suffix 입력의 평균. 주파수 고정·무작위 교차 실험은 아니다.

| 입력 / hit 길이 | Fresh q8_row | Radix hit | 기존 q8 fresh |
| --- | ---: | ---: | ---: |
| 800 / 768 | 4014 | 2507 | 2821 |
| 2080 / 2048 | 9029 | 3263 | 4779 |

- 같은 새 정책에서의 hit TTFT 이득은 확인했다. 기존 q8과의 무회귀 주장은 하지 않는다.
- q8_row cold prefill은 기존 q8보다 느리다. prefix capture까지 추가하면
  source 1024/3072에서 4811→5002ms, 12798→13215ms가 관측됐다.
- 보존한 추가 FP16 prefix payload: source 1024는 206.25MiB,
  source 3072는 646.25MiB. clear 후 모두 0이다.
- **기존 q8 대비 task regression이 관측됐다.** 긴 RUBY 입력은 q8에서
  코드를 답했지만 q8_row에서는 channel marker를 반복하다 generation cap에
  도달했다. 같은 q8_row의 fresh/reused ID는 일치하나 모델 품질 검증 통과로
  해석하면 안 된다. JSON의 `quality_checks_passed`를 별도로 확인한다.
- corpus/perplexity와 충분히 긴 steady decode TPS 검증은 하지 않았다.

## 다음에 할 작업

1. q8_row cold prefill 비용을 줄인다. 현재 fixed attention tile / FP32
   context handoff의 전송과 커널 시간을 분리 측정할 것.
2. 기존 q8 및 q8_row의 task 품질·perplexity를 비교하고 prefix-consistent
   quantization 정책을 검토한다. 기존 q8의 whole-block calibration이 future
   suffix에 의존하는 문제를 무시한 채 기존 INT8 페이지를 잘라 쓰면 안 된다.
3. prefix DRAM byte budget을 연결한다. 현재는 세션·prefix 개수 제한이며
   byte budget이나 NPU의 자동 Flash demotion은 구현하지 않았다.
4. 필요하면 host prefill radix 페이지·토큰 인덱스의 Flash 보존을 설계한다.
   현재 DSP Flash는 resident 대화 재개용이며 host prefix index를 저장하지 않는다.

## 재현

현재 Android 실행 폴더:
`/data/local/tmp/nntr_pr4428_kv_validation/radix-20261104`

기기: SM-S948N / Android 16 / SM8850 v81, serial `R5KL20NFRCK`.
가중치는 기존 `/data/local/tmp/nntr_pr4428_kv_validation/models/gemma4/`를 쓴다.
로컬 원본 로그·실행 바이너리: `/tmp/nntr-radix-20261104/`.

설정: `attention_kv_dtype=q8_row`, `skip_prefill=true`,
64배수 `init_seq_len`, `htp.kv_session=true`, `htp.kv_paged=true`,
`htp.kv_prefix_cache=true`, `htp.kv_request_reserve=true`.

```bash
cd /tmp/nntr-pr4428-validation
source /home/seungbaek/toolchain/nntrainer-htp-20261103/env.sh
ninja -C builddir install
meson configure builddir_app/htp -Denable-htp-device-check=true
ninja -C builddir_app/htp nntr_causallm htp_page_prefix_check \
  htp_sessions_check htp_prefill_check htp_calibration_check
```

```bash
adb -s R5KL20NFRCK shell '
cd /data/local/tmp/nntr_pr4428_kv_validation/radix-20261104 &&
export LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH=$PWD &&
export NNTR_NUM_THREADS=8 NNTR_HTP_KV_MANAGER=1 &&
./htp_page_prefix_check pack fixture-768.json next-result.json'
```

`NNTR_HTP_PREFIX_HASH`는 진단 전용이다. 그 실행 시간은 성능표에 넣지 않는다.
앞선 실패 조사 JSON/CSV와 이전 부분 prefix 검증은 `docs/validation/`에
보존했다. 새 모드의 검증과 이전 q8 상태 절단 실험을 혼동하지 말 것.
