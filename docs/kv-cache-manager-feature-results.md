# KV Cache Manager — 핵심 기능과 적용 효과

2026-11-05, 구현 기준 `286abec4e`, 측정 도구 `f258e8629548098bd5de93698d52de097b698285`.
현재 CPU core와 Gemma4 HTP/DSP 연동 범위를 구분했다. 아래 길이별 속도 수치는
**Android SM-S948N / SM8850 v81, Gemma4 26B A4B**의 실제 측정이다.
기존 dynamic-scale q8/Q2, U16 O-projection, projection/FFN/MoE 실행을 유지했다.

## 핵심 기능

| 기능 | 현재 구현 범위 | 적용 효과와 검증 |
| --- | --- | --- |
| KV 세션 관리 | CPU 및 NPU native C++/request API에서 생성·선택·삭제·현재 상태 fork. 한 번에 한 대화 실행 | 대화별 KV·위치·pending token 보존. 기존 Android A/B 6턴·fork·Flash 재개 ID 검사 통과. 아래 신규 prefix 실험은 세션 스케줄러 성능 측정과 구분 |
| Paged KV + 공유/COW | CPU page pool과 DSP native page table. DSP 물리 페이지 64토큰 | 페이지 할당·공유·회수, 수정되는 tail 분리. DSP paged/contiguous 바이트 일치·parent 보존 검사 통과. host 커널의 페이지 자료구조 정합성 검사이며, paged 자체의 독립 속도 향상률은 측정하지 않음 |
| Radix Tree prefix caching | 토큰 prefix 검색과 복원 가능 상태 확인. NPU는 기존 4096-token prefill 경계 및 호환되는 최종 endpoint | 공통 prefix 재계산 생략. 4K–20K에서 DRAM/Flash hit 총 20개 비교 모두 전체 출력 ID·위치 일치 |
| Global / Sliding KV 관리 | NPU global은 필요한 페이지까지 확장, sliding은 window 밖 tile 회수. CPU의 paged sliding은 mask 기반이며 같은 물리 회수 정책은 아님 | 오래된 sliding KV를 활성 메모리에서 회수하고, 필요 시 과거 경계의 window를 복원. 아래 활성 DSP payload 실측 참고 |
| DRAM ↔ Flash 계층 | CPU prefix LRU demotion/restore. NPU는 DSP INT8 세션 snapshot과 별도의 FP16 native-prefix backing | DRAM에 남지 않은 과거 4K/8K/12K/16K/20K 경계도 Flash에서 복원하여 재계산 절감 |
| 비동기 prefix 저장 | NPU native 경계의 live KV를 immutable staging으로 복사한 후 worker가 write/sync. 최신 경계는 기존 DRAM 경로 | 현재 요청과 저장을 겹침. 아래 20K cold OFF/ON으로 노출 비용을 별도 측정 |
| 용량 제한·LRU·실패 복구 | CPU page pool/Flash byte budget. NPU staging/file byte budget·file LRU·prefix 수 제한 | staging 부족 시 백업 skip, 파일 퇴출/손상/읽기 실패 시 이전 유효 경계 또는 재계산. 기존 budget/LRU/checksum/cancel/failed-write 검사 통과. NPU 최신 DRAM prefix 전체 byte budget은 미지원 |
| 관측 지표 | 요청별 TTFT·출력 ID·KV payload·prefix hit/reused tokens·staging/write/read·skip/failure | 캐시 효과, 저장 비용, 실패 및 메모리 사용을 분리해 확인 가능 |

## 길이별 prefix 재사용 효과

K=1024토큰. 공통 prefix 뒤에 동일한 27-token 새 suffix를 붙였으며,
답변은 모두 `RUBY-456`이었다. 전체 11개 출력 ID와 최종 위치가 각 재계산과 일치했다.
단위는 **초**다. Before는 각 길이의 DRAM/Flash 재계산 **전체 4개 표본 중
최솟값**을 골라 공통으로 사용했다. DRAM/Flash hit는 각각 기존 2회 평균을
유지했다. 재계산에 가장 유리한 관측값을 기준으로 비교한 보수적인 표다.
개선 배수는 반올림 전 ms 값에서 계산했다. 원래의 tier별 paired 평균과 모든
표본은 원본 JSON에 보존했고, 새 표의 선택 정책·표본 출처도 함께 기록했다.

| 공통 prefix / 요청 전체 토큰 | 공통 Before: 재계산 최솟값 | DRAM hit 평균 | 개선 | Flash hit 평균 | 개선 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 / 4,123 | 8.648 | 2.362 | 3.66× | 2.512 | 3.44× |
| 8,192 / 8,219 | 16.116 | 2.311 | 6.97× | 2.936 | 5.49× |
| 12,288 / 12,315 | 24.745 | 2.503 | 9.89× | 3.562 | 6.95× |
| 16,384 / 16,411 | 34.584 | 2.630 | 13.15× | 4.206 | 8.22× |
| 20,480 / 20,507 | 46.829 | 2.752 | 17.01× | 5.325 | 8.79× |

두 실험 구간의 재계산은 동일한 입력 token IDs, prefill 토큰 수, 모델 설정과
원래 forward 경로를 사용한다. DRAM 쪽 재계산에 추가 forward나 백업이 들어간
것은 아니다. 실험 구간은 순차 실행됐고, cache producer의 상태와 기기의 온도·
주파수·메모리 상태까지 동일하게 고정하지는 않았다. 더 느린 DRAM 구간의
원인을 이 자료만으로 분리하지 못했으므로 이를 성능 이득으로 사용하지 않는다.
이 정리는 기존 실측치를 재집계한 것으로 새 측정 결과가 아니다.

DRAM은 N+512-token source에서 마지막 완전한 N 경계를 재사용했다.
Flash는 25,088-token source를 먼저 처리하여 24K까지 진행된 뒤에도 과거 N 경계를
복원했다. 캐시 생성 source 요청의 시간은 hit TTFT에 포함하지 않았다. 재계산
control은 같은 모델·runtime·producer 메모리를 유지하면서 lookup만 일시 중단한다.
측정용 후속 요청은 새 prefix를 publish하지 않아 이후 Flash 표본이 DRAM hit로
바뀌지 않도록 했다. 이 제어는 벤치마크 subclass에만 있고 제품 코드는 변경하지 않았다.

## 저장 비용과 overlap

20K를 넘는 **20,992-token source**에서 earlier-boundary 백업만 OFF/ON으로 바꿨다.
기존 DRAM prefix 보존과 모든 연산 설정은 동일하다. OFF/ON/ON/OFF, 모드별 2회다.

| 항목 | 결과 |
| --- | ---: |
| Cold TTFT — 저장 OFF | 57.819초 |
| Cold TTFT — 저장 ON | 60.300초 (+4.29%) |
| ON의 staging 복사 합계 / 요청 | 880.7ms |
| ON의 background write/sync 합계 / 요청 | 8325.1ms |
| ON에서 요청 종료 시 미완료 payload | 0 bytes |
| ON에서 보존된 earlier 경계 / 요청 | 4개 (4K·8K·12K·16K) |
| 백업 skip / 실패 | 0 / 0 |

20K cold에서는 저장 ON이 +4.29% 느렸다. background write가 요청 종료 전에
완료됐다는 사실만으로 추가 지연이 없다고 해석하지 않는다. staging 복사와
CPU/DDR/I/O 경쟁, 기기 온도·주파수의 영향을 이 측정만으로 분리하지 못했다.
이 수치는 후속 최적화 대상으로 유지하며 cache-hit 이득과 함께 평가해야 한다.

별도의 25,088-token producer는 5개 경계를 저장했다. 합계 backing은
2200.000 MiB, peak staging은
600.000 MiB였으며 설정한 768 MiB
staging 및 3 GiB file 예산 이내였다. 파일 쓰기/sync 합계는
3733.0ms, staging 합계는
448.5ms였다.

## Payload와 복원 비용

| 공통 prefix | 활성 DSP INT8 K+V payload (실측) | 해당 경계의 Flash FP16 live KV payload | Flash read/checksum 평균 |
| --- | ---: | ---: | ---: |
| 4,096 | 146.875 MiB | 280 MiB | 498.7 ms |
| 8,192 | 186.875 MiB | 360 MiB | 811.4 ms |
| 12,288 | 226.875 MiB | 440 MiB | 1221.3 ms |
| 16,384 | 266.875 MiB | 520 MiB | 1752.8 ms |
| 20,480 | 306.875 MiB | 600 MiB | 2545.6 ms |

Flash 크기는 현재 모델의 global/sliding shape와 파일 framing을 기준으로 한
live-row 크기이며 파일당 header/checksum 48 bytes가 추가된다. 양자화 scale과
layout/token index는 DRAM metadata에 남는다. DSP INT8 payload와 prefill 복원용
FP16 backing은 서로 다른 표현이므로 같은 메모리 항목으로 합산/비교하지 않는다.
전체 모델 DRAM, 가중치, host prefix RPC slab, staging, scale 및 OS file cache는
활성 DSP payload 수치에 포함되지 않는다. 이 pack의 최신 host-prefix slab pin
카운터는 1560 MiB이며, 전체 DRAM을 줄이는 추가 최적화는 별도 과제다.

## 해석 범위

- 실측 cache-hit 시나리오의 TTFT 절감이며 전체 serving throughput/TPS 수치가 아니다.
  TTFT에는 파일 로드·검증·KV 복원·suffix 계산이 포함되고 모델 로드는 제외한다.
- 주파수는 고정하지 않았다. 모드별 2회는 모든 장치/워크로드의 무회귀 보장이 아니다.
  각 표본과 min/max를 원본 JSON에 남겼다. 공통 최솟값은 관측된 최상의
  재계산 성능이며, 고정된 환경에서의 재현성을 새로 검증한 결과는 아니다. 중간 thermal service 상태는 2,
  HAL SKIN 44.6°C였으며 원문도 JSON에 보존했다. 서로 다른 tier 구간의
  재계산 절대시간 차이를 캐시 효과로 귀속하지 않는다.
- Flash hit 전마다 clean file-cache 회수를 advisory로 요청했다. UFS 내부 cache까지
  비운 보장은 하지 않는다. source 백업은 worker에서 fdatasync 후 완료 처리한다.
- NPU prefix 경계는 기존 동적 scale의 native 호출 단위다. 임의 64-token 경계,
  `kv_seed_q=true`, 프로세스 재시작 후 radix 복원은 현재 범위가 아니다.
- 세션·paged/COW·Global/Sliding·Flash를 결합한 효과다. 이 TTFT를 각 자료구조의
  독립 성능 개선률로 중복해서 귀속하지 않는다.

[전체 입력·출력 ID·표본·바이너리 해시](validation/gemma4-htp-prefix-grid-20261105.json).
이전 [native radix 검증](validation/gemma4-htp-native-radix-20261105.json) 및
[비동기 Flash/A→B 검증](validation/gemma4-htp-prefix-flash-20261105.json)도 함께 참고한다.

## 재현

`tools/kv_cache/benchmark_htp_prefix_grid.cpp`를 `htp_prefix_grid_bench`로 빌드한다.
기기 bundle: `/data/local/tmp/nntr_pr4428_kv_validation/nntr-prefix-grid-20261105`.
로컬 원본: `/tmp/nntr-prefix-grid-20261105`. 가중치는 기존 모델을 재사용했다.

```sh
export LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH=$PWD NNTR_NUM_THREADS=8 NNTR_HTP_KV_MANAGER=1
./htp_prefix_grid_bench pack /data/local/tmp/nntr_pr4428_kv_validation/prefix-grid-cache result-grid.json
```
