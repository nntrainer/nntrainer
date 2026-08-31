# H2 — 빌드 연동과 HMX 자원 확보

상위: [00-overview](00-overview.md) · 선행: [01-spike-relocation](01-spike-relocation.md)

## 링크와 버전

- `~/hexkl_addon/lib/6.3.0.0/hexagon_toolv88_v75|v79/libhexkl_micro.a`를 `libnntr_htp_skel.so`에 정적 링크. 헤더는 `~/hexkl_addon/include/hexkl_micro.h`
- `tools/hexagon/build_skel.sh`(이미 `HEX_ARCH`, `HEX_EXTRA_CFLAGS`를 받음)에 `HEXKL_ADDON_ROOT` 경로 변수 추가; skel은 meson 밖(HEXAGON.md §4)이므로 meson 변경은 없음. SDK 버전(`HEXAGON_SDK_ROOT`)에서 `lib/<sdk_version>/` 자동 선택 (hexkl_addon 자체 스크립트와 같은 규칙)
- `version.h`로 최소 HexKL 버전 컴파일 타임 체크 + 세션 init 시 `hexkl_micro_get_version` 런타임 로그

## HMX 자원 확보

HMX 사용에는 두 자원이 필요하다: **VTCM**(HMX는 VTCM에 있는 데이터만 읽고 씀)과 **HMX 유닛 사용권**(코어당 하나뿐인 공유 하드웨어라 배타적 확보 필요).

HexKL은 이를 위한 편의 함수를 제공하지만 — `hexkl_micro_hw_init()`(VTCM을 라이브러리가 확보), `hmx_lock()`/`hmx_unlock()`(HMX 사용권 획득/반납) — **사용하지 않는다**:

- 헤더 문서에 "테스트·예제용이며, 통합자는 자체 루틴을 구현하라(may implement their own)"고 명시된 함수들이다
- 우리 실행기는 이미 `htp_graph.c`에서 `HAP_compute_res_acquire()`로 세션 수명 동안 VTCM을 자체 확보하며, 기존 HVX 커널·DMA 큐가 전부 이 VTCM을 쓴다. `hw_init`을 병용하면 자원 관리 주체가 둘이 되어 포인터 소유권·수명이 꼬인다
- 실제 연산 함수(`hmx_mm_u8i8` 등)는 전부 `vtcm_base`를 **인자로** 받으므로, VTCM을 누가 확보했는지와 무관하게 동작한다

대신 기존 확보 코드(`htp_graph.c` `htp_graph_init`, VTCM 확보 블록)에 HMX 파라미터 한 줄만 추가한다:

```c
HAP_compute_res_attr_init(&rattr);
HAP_compute_res_attr_set_vtcm_param(&rattr, HTP_GRAPH_VTCM_BYTES, 1);
HAP_compute_res_attr_set_hmx_param(&rattr, 1);   /* 추가: HMX 사용권 동시 확보 */
id = HAP_compute_res_acquire(&rattr, 10000);
```

세션 init 1회로 VTCM+HMX를 함께 확보하고, 세션 종료 시 기존 `HAP_compute_res_release()`로 함께 반납 — 자원 관리 주체가 하나로 유지되고, 토큰마다 lock/unlock을 반복할 필요도 없다.

## 완료 기준

- v75/v79 크로스 빌드에서 `libnntr_htp_skel.so`가 hexkl 심볼 포함하여 링크 성공
- 시뮬레이터에서 세션 init 시 HMX 포함 compute_res 확보 성공 + 버전 로그 출력
