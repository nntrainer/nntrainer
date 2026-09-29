# H1 — 선행 스파이크: wh 타일 재배치 (설계 확정 게이트)

상위: [00-overview](00-overview.md)

**질문**: `hexkl_micro_hmx_rm_to_wh_i8`이 VTCM에 생성한 wh 타일을 DDR로 복사해 보관했다가 **다른 VTCM 오프셋**으로 다시 올려도 `hexkl_micro_hmx_mm_u8i8` 결과가 유효한가?

wh 레이아웃이 타일 자기완결적(512B 블록 내부 순열)이면 유효. 절대 VTCM 주소/오프셋 의존이 섞여 있으면 무효. Beta.2 문서에 명시가 없어 실험으로만 판정 가능하며, **본 설계 전체가 이 판정에 걸려 있다.**

## 방법

`~/hexkl_addon/examples/hexkl_micro_hmx_mm_u8i8_i32/`(시뮬레이터 테스트, `run_simulator.sh` 포함)를 스크래치에 복사해 3개 시나리오로 개조:

1. **베이스라인**: 예제 원본 실행 (CPU 레퍼런스 비교 내장)
2. **재배치 (핵심)**: 타일별로 `rm_to_wh_i8` 변환 → DDR로 memcpy → 원래 VTCM 영역을 쓰레기값으로 덮음 → 다른 VTCM 오프셋으로 복사 → `hmx_mm_u8i8` → 베이스라인과 비트 비교. 오프셋을 정렬 최소 단위 배수 여러 값으로 반복해 오프셋 독립성 확인
3. **비용 측정**: `rm_to_wh_i8` 타일당 Pcycles(PMU) → qwen3-0.6b 전체 가중치(≈0.6GB) 환산 init 변환 시간 추정. 수 초 수준이면 설계 그대로, 수십 초면 초기화 전략 재고. 같은 실행에서 **`hmx_mm` 대비 `acc_read` 비중**도 측정 — 누산기 단일 상태 stall의 실제 손실 수치화 ([06 ③](06-optimizations.md))

**2단계 실행**: 시뮬레이터(`run_simulator.sh`)로 빠르게 반복하며 1차 판정 → 통과 시 **같은 테스트를 디바이스(`run_android.sh`)에서 재실행해 확정**. 게이트가 설계 전체를 결정하므로 시뮬레이터 단독 판정으로는 통과 선언하지 않는다 (시뮬레이터가 모델링하지 않는 실기기 동작 — VTCM→DDR 복사 경로의 캐시 일관성, HMX 유닛 실제 동작 — 을 배제하기 위함). 스파이크는 M1–M5 완료 후이므로 디바이스는 이미 확보된 상태 (S25 SM-S931N, 시리얼 R3CY30LD2LK; WSL2에서는 `ADB_SERVER_SOCKET=tcp:<windows-ip>:5037`로 Windows adb 서버를 경유. 실행기 VTCM 예산은 `HTP_GRAPH_VTCM_BYTES` = 4 MB). 단, 프로덕션 DMA 경로의 캐시 일관성은 여전히 본 구현의 디바이스 테스트([05](05-e2e-verification.md)) 몫.

## 판정

- 시뮬레이터·디바이스 양쪽에서 모든 오프셋 비트 일치 → 본 설계 확정, H2로 진행
- 불일치 → **fallback: prefill-only HMX + on-the-fly 변환**. decode는 HVX 유지, 가중치는 row-major 한 본만 유지하고 prefill 커널이 DMA 타일 로드 직후 `rm_to_wh_i8`로 실행 중 변환. [03](03-weight-pipeline.md)·[04](04-hmx-kernel.md)는 이 전제로 재설계

## 산출물

- 스파이크 코드는 throwaway (참고용으로 스크래치에 보존)
- 보고: 판정(일치/불일치) + 타일당 변환 Pcycles + init 시간 추정치 표
