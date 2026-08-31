# H5 — e2e 전환, 검증과 성능

상위: [00-overview](00-overview.md) · 선행: [04-hmx-kernel](04-hmx-kernel.md)

## 검증 단계

| 단계 | 내용 | 기준 |
|---|---|---|
| 단위 | 시뮬레이터에서 HMX vs HVX w8a8 출력 비교 (`run_sim_test.sh` 프레임 재사용) | **비트 동일** |
| e2e | M4의 4-way 절차: x86 참조 실행기 `hexagon_ref_run`(같은 packed 이미지) ↔ 디바이스 `hexagon_e2e_test`, 발산 시 `find_divergence.py`로 op 이진 탐색 | 모델 수준 PPL 게이트 — eval.txt 386 tok: HVX 커널 **19.9787**(x86 ref 20.2718) 대비 노이즈 밴드 내. 짧은 텍스트 로짓 게이트는 w8a8 증폭 때문에 쓰지 않음 (HEXAGON.md §8.1) |
| 앱 | `engine="htp"` 앱 실행 (HEXAGON.md §5.4 절차) | 문장 생성 정상 + 폴백 유지 |
| 성능 | `run_e2e_test.sh … --tokens in1024.tokens.i32 --chunk 128 --steps 64` + init 변환 시간 | **기준선(M5, HEXAGON.md §8.2): prefill 23.2 tok/s @1024, decode 11.3 tok/s, 앱 decode 13.0.** 1차 목표: decode가 같은 폰 CPU fp32(18.7 tok/s)를 넘을 것. 이론 상한 ~60 tok/s(대역폭). 앱 e2e(600 MB 패킹 포함) 6.4 s 대비 init 변환 시간 회귀 폭 보고 |

- 디바이스 테스트에서 **프로덕션 DMA 경로**(init 변환·wh 타일 스트리밍)의 캐시 일관성 확인 — H1 스파이크는 자체 memcpy 경로를 디바이스에서 확인했을 뿐, 본 구현의 DMA 경로는 여기서 검증
- 보고는 항상 STAT 정확도 + Pcycles/wall/forward_us 표로 병기

## 전환

- 검증 통과 후 op 디스패치 기본값을 HMX 경로로 전환. HVX 경로는 플래그로 잔존 (안전망)
