# NPU KV radix prefix 인수인계 — 2026-11-04

## 현재 상태

사용자 지시에 따라 느린 `q8_row` radix 경로를 **기존 최적화 q8/Q2 경로로
교체**했다. projection·FFN·MoE는 기존 큰 배치를 유지하고 Q16/K8/V8
attention 및 U16 context → folded O projection을 사용한다.

단, 기존 whole-call calibration을 그대로 유지한 투명한 재사용은 아니다.
prefix의 재현성을 위해 attention calibration을 256-query domain으로 나눴다.
같은 정책에서 fresh/reused ID·위치 일치를 검증했지만, 기존 whole-call q8과의
품질 동일성 및 cold 성능 무회귀는 **미달성**이다. 이를 완료로 오인하면 안 된다.

- 실제 작업 경로: `/tmp/nntr-pr4428-validation`
- 기본 repo `/home/seungbaek/projects/nntrainer`는 별도 checkout이다.
- 브랜치: `kv/pr4428-dsp-flash-20261104`, 원격: `baek2sm/nntrainer`
- 현재 구현 커밋: `b60d133b9` — `[HTP] Keep optimized Q2 kernels for radix prefix reuse`
- [현재 구현 및 재현](gemma4-htp-radix-prefix.md)
- [현재 raw 입력/ID/위치/성능/바이너리 해시](validation/gemma4-htp-q8-radix-20261104.json)
- 과거 `q8_row`의 9초 결과는 폐기된 정책의 TTFT다. 과거 JSON/CSV는 이력으로 보존했다.

## 구현 및 검증한 것

- radix tree가 토큰 ID를 비교하고 **복원 가능한 256-token 경계**를 찾는다.
  물리 DSP 페이지는 64-token이다. 1024-token source의 계산에서 768-token
  경계까지 보존하므로 별도 768-token 추론/전체 체크포인트 없이 재사용한다.
- 공통 750-token은 512-token을 재사용한다. 256 미만은 안전하게 miss한다.
  suffix가 1/2-token이면 이전 경계로 물러나고, 3-token이면 현재 경계를 쓴다.
- immutable host FP16 prefill 페이지와 정확한 domain별 calibration을 보존한다.
  DSP INT8 decode 페이지는 별도 세션/COW 상태다. 단순 INT8 snapshot 절단이 아니다.
- attention만 domain별로 실행한다. U16 결과와 encoding을 모아 기존 wide
  O-projection RPC **한 번**으로 처리한다. DSP는 encoding 경계에서 기존 커널을 쓴다.
- Q subview는 부모 ION generation을 검증한다. sampled-only host tensor를
  읽던 문제를 수정했다. Q/K/V RPC staging은 전체 예약 slab 대신 live span을 쓴다.
- Android 64/256/750/768/960/2048 공통 길이의 12개 분기 및 짧은 suffix
  3개에서 fresh/reused 출력 ID·최종 위치가 일치했다.
- 서로 다른 A/B 대화 6턴 전환, fork, Flash save/park/restore, native CLI
  A/B/A/B 4개 결과를 독립 참조와 대조했다. 모두 ID·위치 일치.
- source/fresh/reused의 앞 768-token FP16 K/V hash가 30개 레이어 모두 일치했다.
- prefix clear 후 host payload=0, 활성 DSP KV 유지 확인.
- Ubuntu KV unit test 36개, 실제 staging/handoff 검사 201개, DSP page/COW/
  sliding/Flash host 검사와 ASan/UBSan 통과. Ubuntu/Android prefill 페이지·
  독립 calibration 검사 통과. Ubuntu/Android ARM/v81 DSP 빌드 통과.

## 속도와 품질의 남은 문제

같은 causal q8 정책에서 두 suffix 평균, 단위 ms:

| 입력 / 실제 재사용 | 전체 재계산 prefill | Hit prefill | 전체 재계산 TTFT | Hit TTFT |
| --- | ---: | ---: | ---: | ---: |
| 800 / 768 | 2036 | 1582 | 2423 | 1955 |
| 2080 / 2048 | 4364 | 1538 | 4793 | 2015 |

별도 native CLI 실험: 같은 최종 바이너리로 warmup 후 2080-token 4회씩,
prefix retention 없이 기존 whole-call q8 → causal q8 순으로 실행:

| 정책 | Prefill 평균 | Prefill TPS | TTFT 평균 |
| --- | ---: | ---: | ---: |
| 기존 whole-call q8 | 3800ms | 547 | 4217ms |
| causal q8 | 4506ms | 462 | 4999ms |

- 마지막 대조에서 cold prefill **18.6% 증가**. 같은 최종 radix 바이너리의
  앞선 실행은 4181–4425ms였다. 주파수를 고정하지 않았으므로 정확한 무회귀
  추정으로 해석하지 말 것. 이전 9초보다는 줄었지만 기존 경로와 동등하지 않다.
- **긴 RUBY task regression이 남아 있다.** 기존 whole-call q8은 코드를
  답하지만 causal q8은 `tool_response`를 출력한다. 같은 causal q8의
  fresh/reused ID 일치는 이 품질 문제 해결을 의미하지 않는다.
- 800-token 두 코드 회상은 성공했다. 일부 길이는 EOS/marker를 출력한다.
  원본 결과를 유지했고 `all_task_quality_checks_passed=false`로 기록했다.
- corpus/perplexity와 충분한 steady decode TPS는 검증하지 않았다.
- 추가 host FP16 payload: source 1024는 165MiB, source 3072는 605MiB.
  prefix는 개수/LRU 제한이며 byte budget은 없다. DSP Flash에는 host radix
  페이지/토큰 index를 저장하지 않는다.

## 다음 작업

1. **calibration grouping에 따른 품질 회귀 해결**: whole-call q8과 domain별
   attention의 차이를 층별로 추적할 것. whole-call scale은 미래 suffix에
   의존하므로 기존 INT8 상태를 그냥 절단해 ID 동일성을 주장하면 안 된다.
2. cold overhead의 attention RPC/scale growth/재양자화 비용을 분리해 줄일 것.
   q8_row나 FP32 context로 되돌리거나 projection/FFN을 잘게 쪼개지 말 것.
3. 품질 문제 해결 후 같은 바이너리·고정 조건에서 baseline/hit/cold 교차
   반복 측정하고 corpus/perplexity를 검사할 것. 현재는 완전 무회귀 검증 아님.
4. host prefix byte budget 및 Flash 보존은 별도 후속 범위다.

## 재현 위치와 빌드

Android: `/data/local/tmp/nntr_pr4428_kv_validation/q2-radix-20261104`

기기: SM-S948N / Android 16 / SM8850 v81, serial `R5KL20NFRCK`.
가중치는 기존 `/data/local/tmp/nntr_pr4428_kv_validation/models/gemma4/`.
로컬 최종 원본 로그·바이너리: `/tmp/nntr-q2-radix-20261104/final-device/`.

설정: `attention_kv_dtype=q8`, `skip_prefill=true`, `init_seq_len=4096`,
`htp.kv_session/kv_paged/kv_prefix_cache/kv_request_reserve=true`.
기존 설정의 `kv_prefix_cache=false`이면 whole-call q8을 유지한다.

```bash
cd /tmp/nntr-pr4428-validation
source /home/seungbaek/toolchain/nntrainer-htp-20261103/env.sh
bash nntrainer/tensor/htp_backend/generate_stub.sh
ninja -C builddir install
ninja -C builddir_app/htp nntr_causallm htp_page_prefix_check \
  htp_sessions_check htp_prefill_check htp_calibration_check
export HEXKL_ROOT=/tmp/nntr-pr4428-validation-results/hexkl-device
export HEXKL_SDK_VER=device-v81
export HEXKL_MICRO_DSO=$HEXKL_ROOT/lib/libnntr_hexkl_provider.so
NNTR_HTP_DIAGNOSTICS=0 bash nntrainer/tensor/htp_backend/dsp/build.sh
```

ARM 엔진·앱과 **matching DSP skeleton 모두** 배포해야 한다. 기존 RPC
메서드의 encoding 배열을 확장한 것으로 새 RPC wire method는 없다.

```bash
adb -s R5KL20NFRCK shell '
cd /data/local/tmp/nntr_pr4428_kv_validation/q2-radix-20261104 &&
export LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH=$PWD &&
export NNTR_NUM_THREADS=8 NNTR_HTP_KV_MANAGER=1 &&
./htp_page_prefix_check pack fixture-768.json next-result.json'
```

`NNTR_HTP_PREFIX_HASH`는 진단 전용이다. 해당 실행의 시간은 성능표에 넣지 않는다.
