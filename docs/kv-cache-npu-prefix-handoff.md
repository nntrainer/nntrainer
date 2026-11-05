# NPU KV radix 인수인계 — 2026-11-05

## 현재 구현

- 작업 경로: `/tmp/nntr-pr4428-validation`
- 브랜치: `kv/pr4428-dsp-flash-20261104`, 원격 `baek2sm/nntrainer`
- 구현 커밋: `f2e43d820`
- 기존 q8/Q2 calibration·Q16/K8/V8·U16 O projection 및 projection/FFN/MoE 배치 유지.
- **256-token calibration 정책은 제거했다.** prefix 옵션이 계산 dispatch를 바꾸던
  과거 q8_row 분기도 제거했고 Q2/U16 구성이 아니면 명시적으로 거부한다.
- [구현·설정](gemma4-htp-radix-prefix.md),
  [전체 입력/출력 ID·위치·성능·바이너리 해시](validation/gemma4-htp-native-radix-20261105.json).

## 재사용 의미

Radix는 토큰과 **원래 prefill 호출 경계**가 모두 맞는 상태만 재사용한다.
새로 계산하는 긴 요청은 마지막 완전한 native 호출 경계와 최종 multi-row
prefill endpoint를 DRAM에 보존한다. **새 opt-in 비동기 prefix Flash 모드에서는
더 이른 native 경계의 global/sliding KV도 저장·복원한다.**

- 4608-token source → 공통 4096-token prefix 재사용.
- 8704-token source → 공통 8192-token prefix 재사용. 실제 계산 배치는 기존대로
  4096-token 두 호출이며, 8192는 저장된 위치다.
- 2080-token 동일 입력 반복 → 기존 전체 2079-token prefill 재사용, 마지막 입력은
  원래 DSP decode로 처리.
- 한 호출로 계산한 1024-token source에서 768-token만 자르는 요청은 miss.
  계산을 바꾸어 hit를 만들지 않는다.
- 한 토큰 prefill 꼬리는 DSP window를 seed **전에** 설정한다. 기존 4098-token
  경계의 seed-twice 오류를 고쳤다. 이 경우 host checkpoint는 앞선 4096-token이다.

FP16 RPC slab을 page view로 pin한다. 마지막 sliding window는 memmove 대신
read-only offset으로 seed한다. 원래 compact 결과와 KV bytes·dropped maxima가
같음을 Ubuntu/Android에서 확인했다. 첫 seed 후에는 일반 decode entry를 쓴다.
DSP INT8 page/COW/session/Flash 경로는 유지했다.
Radix의 host ID·release namespace를 DSP 세션과 분리했다. 사용하지 않는
DSP prefix checkpoint retain RPC와 그로 인한 decode tail COW는 제거했다.
실제 session select/fork만 DSP checkpoint를 보존한다.

## 검증 완료

- Android SM-S948N / Android 16 / SM8850 v81, serial `R5KL20NFRCK` 실제 모델 구동.
- 분기 11개와 source 반복 5개, cold source·parent 보존의
  전체 출력 ID 및 위치 일치. suffix 1/2/3-token, 4K/8K 경계, 안전한 miss 포함.
- 긴 RUBY 출력은 기존 working legacy의 전체 출력 ID와 일치. 과거 `tool_response`
  회귀는 해소됐다. 일부 다른 baseline fixture 자체의 EOS는 그대로 기록했다.
- 서로 다른 A/B 6턴 전환, fork, paged Flash save/park/restore 통과.
- native CLI 4098-token cold → repeat: ID/위치 동일, 4096-token 재사용 확인.
- Ubuntu build, 기존 KV unit test 36개, Ubuntu/Android page/view/lifetime 및 독립
  calibration 검사 통과. DSP page/COW/Flash host 검사와 ASan/UBSan 통과
  (`detect_leaks=0`). clang-format 14와 git diff 검사 통과.

동일한 모델 하나를 유지하고 저장 보존만 OFF/ON 교차한 cold 측정이다. 두 길이마다
모드별 4회, hit=0. 단위 ms, 각 셀은 OFF / ON:

| 입력 토큰 | Prefill | TTFT |
| --- | ---: | ---: |
| 2080 | 3886.0 / 3941.8 | 4333.0 / 4375.7 |
| 4608 | 9007.2 / 9026.8 | 9466.6 / 9501.8 |

실제 hit의 독립 재계산 대비 TTFT, 단위 ms:

| 입력 / 재사용 토큰 | 재계산 | 재사용 |
| --- | ---: | ---: |
| 2080 / 2079 | 3959.6 | 316.8 |
| 4128 / 4096 | 8728.2 | 2116.9 |
| 8224 / 8192 | 15848.4 | 2224.6 |

프로세스를 나눈 앞선 측정에서 2K +2.36%가 관측됐고 원본 수치를 JSON에 남겼다.
동일 모델의 storage-only 비교에서도 약 1% 이하 차이가 관측됐다. 사용자는 이 차이를 측정 오차 내 무시 가능한 수준으로 수용했다. 사용하지 않는 DSP prefix
snapshot을 제거하기 전의 same-model 원본도 JSON에 보존했다. CPU/DSP 주파수를 고정하지 않았다.
이 결과를 모든 상황의 절대 0% overhead 또는 sustained decode TPS 보장으로
확대 해석하지 말 것. 과거 18.6%와 9초 수치는 폐기된 다른 계산 정책의 결과다.

## 남은 범위와 주의할 점

- 세션/프리픽스 수와 LRU 제한만 있으며 host prefix **byte budget은 없다**.
  RPC slab 전체 pin으로 작은 pack은 1160 MiB, 긴 pack은 1240 MiB/source를
  보존한다. 여유 DRAM에 맞춰 prefix 수를 설정해야 한다.
- `htp.kv_seed_q=true`의 직접 양자화 seed는 기존 경로를 보존하고 prefix는 miss 처리한다. 기본 검증 pack은 `false`다.
- DSP Flash snapshot은 host FP16 radix index를 저장하지 않는다. 새 prefix Flash
  backing은 별도 경로로, scale/layout/token index는 모델의 DRAM metadata에 남는다.
- 첫 seed에서 tensor의 RPC slab 소유권을 반납하며 radix pin만 유지한다. clear 후 논리 pin은 0이다. 실제 OS MemAvailable 즉시 증가량은 driver/allocator 상태에 따라 다르며 전체 payload 즉시 반환을 주장하지 않는다.
- 비기본 repetition penalty와 `NNTR_RESUME_BLOCK=0`은 재사용 대신 원래 계산.
- corpus 품질 또는 긴 steady decode TPS는 별도 검증 범위다. 현재 결과는 기존
  계산 유지와 검사한 요청의 정확한 ID·위치 및 cold/hit TTFT 검증이다.

## 실행 재현

기기 경로: `/data/local/tmp/nntr_pr4428_kv_validation/native-q8-complete-20261105`
로컬 원본: `/tmp/nntr-original-q8-20261105/complete-raw`
가중치: 기존 `/data/local/tmp/nntr_pr4428_kv_validation/models/gemma4/`.

```sh
source /home/seungbaek/toolchain/nntrainer-htp-20261103/env.sh
ninja -C builddir install
ninja -C builddir_app/htp nntr_causallm htp_page_prefix_check htp_sessions_check \
  htp_prefill_check htp_calibration_check htp_prefix_cold_bench
```

실행은 `LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64`,
`ADSP_LIBRARY_PATH=$PWD`, `NNTR_NUM_THREADS=8`, `NNTR_HTP_KV_MANAGER=1`.
`htp_page_prefix_check pack fixture-4096.json result.json`,
`htp_page_prefix_check long-pack fixture-8192.json result.json`,
`htp_prefix_cold_bench pack paired-fixture.json cold.json`을 사용한다.
fixture 원본은 evidence JSON의 `length_cases` 및 `cold_same_model_raw` 안에 있다.
ARM 엔진·앱과 DSP skeleton을 함께 배포한다.

## 추가 완료: 과거 native 경계의 비동기 Flash 보존

- 구현 커밋 `f2e43d820`, 기존 q8/Q2·배치·DSP 연산 변경 없음.
- `enableDeviceKVPrefixFlash(directory, flash_bytes, staging_bytes)` 또는
  top-level `kv_prefix_flash` JSON으로 opt-in. 기본값은 OFF.
- 8704-token source가 끝나도 과거 4096-token 경계를 Flash에서 복원한다.
  마지막 native 경계/최종 상태는 기존 DRAM 경로를 유지한다.
- earlier boundary의 live global KV + sliding window를 owned staging에 복사하고
  worker가 checksum/write/fdatasync한다. 완료 후 staging 해제. calibration은
  원본 그대로 DRAM frame에 남는다. 현재 file 간 global page dedup은 없다.
- staging byte budget이 차면 추론을 기다리지 않고 그 경계 백업 skip. 파일은
  budget/LRU로 퇴출하고, evicted/missing/corrupt 상태는 이전 경계 또는 재계산.
- native CLI A→B와 독립 재계산의 RUBY 답변 전체 11개 ID/위치 일치.
- cold 8704-token 저장 OFF/ON 모드별 4회: TTFT 17.204 / 17.272초 (+0.39%).
  staging 평균 107.9ms, background write/sync 평균 971.4ms. 모든 ON 표본에서
  요청 종료 시 pending=0, skip/failure=0. 주파수 고정 없이 측정.
- 실제 A→B: 4123-token B 중 4096 재사용. 재계산 8.453초 → 복원 TTFT 3.223초.
  file read/checksum 985.3ms, 세션 전환 7.8ms는 TTFT와 별도.
- synthetic branch 2개, 기존 8192 DRAM 경계, 세션 6턴/fork/DSP Flash 복원 통과.
  synthetic baseline의 EOS를 품질 성공으로 포장하지 않았으며 RUBY로 보강했다.
- Ubuntu 36 unit, Android/Ubuntu owned-stage/view, Flash LRU/budget/checksum/
  cancellation/failed-write 검사 및 host ASan/UBSan 통과 (detect_leaks=0).
- staging/file budget은 기존 latest-prefix RPC slab을 포함한 전체 DRAM 제한이 아니다.
  최신 DRAM prefix 크기 문제 및 cross-file page dedup은 후속 최적화 범위.
- backing은 process-local. 정상 clear/LRU/destruction 시 정리되지만 비정상 종료 후
  private cache directory 잔여물 정리는 호출 애플리케이션이 담당해야 한다.
- [설정·제약·측정 설명](gemma4-htp-radix-prefix.md),
  [전체 원본 데이터](validation/gemma4-htp-prefix-flash-20261105.json).

재현 bundle: `/tmp/nntr-prefix-flash-20261105` (기기에도 같은 basename).
`run-final.sh`는 기능/교차 성능/기존 세션 검증, `run-quality.sh`는 실제 CLI RUBY
재계산과 A→B Flash hit를 실행한다. 로컬 `raw/`에 모든 결과와 로그가 있다.
