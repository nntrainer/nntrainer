# P2 — 32행 타일 가중치 레이아웃

상위: [00-overview](00-overview.md) · 선행: [01](01-profile-baseline.md) · 다음: [03-w8a8-tiled-kernel](03-w8a8-tiled-kernel.md)

## 대상 텐서

int8 `[N][K]` projection 전부: `embed`(EMBED gather + LOGITS 공용), 레이어별 `wq wk wv wo gate up`. **`down`은 제외** (`[N][K]` 유지, [04](04-w8a16-down-kernel.md)). 스케일 `[N]` fp32는 위치·형식 불변.

전제: `K % 128 == 0`(기존 validator), **`N % 32 == 0`**(신규). qwen3-0.6b의 N = 1024, 2048, 3072, 151936 모두 만족. 텐서 바이트 수 `N*K`는 불변이므로 `HexWeightOffsets`와 `weights_size`(598,623,744)는 그대로다.

## 타일 정의

```
n_tiles = N / 32, k_tiles = K / 128
tile(nt, kt)  : 4096 B, 오프셋 = (nt * k_tiles + kt) * 4096      # n-타일 바깥, k-타일 안쪽
tile 내부     : 32 벡터 v[g], g = 0..31 (각 128 B)
v[g] 바이트 [4r .. 4r+3] = w[nt*32 + r][kt*128 + 4g .. kt*128 + 4g + 3]   (r = 0..31)
```

- 하나의 벡터 v[g]는 "32행 × 연속 4k". `Q6_Vw_vrmpyacc_VwVbVb(acc, v[g], splat4(x[kt*128 + 4g..+3]))`의 lane r에 행 `nt*32+r`의 4-MAC 부분합이 쌓인다. g=0..31, kt=0..k_tiles-1 누적 후 acc lane r = 행 r의 완전한 dot. 수평 리덕션 없음.
- 한 n-타일의 K 전체 스트립(`k_tiles * 4096 = 32*K` B)은 연속 → DMA 디스크립터 1개, 선형 스트리밍.
- 역인덱스(gather용): 원소 `w[n][k]` 는 `base + ((n/32) * k_tiles + k/128) * 4096 + ((k%128)/4) * 128 + (n%32) * 4 + (k%4)`.

헬퍼를 한 곳에 둔다: `nntr_htp_common.h`에 `static inline uint32_t nntr_htp_tile_off(uint32_t n, uint32_t k, uint32_t K)` — 호스트 팩, DSP EMBED gather, x86/sim 레퍼런스가 **같은 함수**를 쓴다. 세 곳이 각자 계산하지 않게 하는 것이 정확도 게이트의 핵심.

## 변경 파일

| 파일 | 변경 |
|---|---|
| `htp/nntr_htp_common.h` | `NNTR_HTP_ABI_VERSION 4`, `NNTR_HTP_WEIGHT_LAYOUT_TILED32 = 1`을 헤더 `reserved2[0]`(→ `weight_layout`로 이름 변경, 64B 불변)에 기록. validator: `MATMUL_W8A8`·`MATMUL_LOGITS`·`EMBED`의 in1 텐서에 `n % 32 == 0`(EMBED는 vocab), `weight_layout`이 알려진 값이 아니면 rc 4. `nntr_htp_tile_off()` |
| `host/graph_lowering.{h,cpp}` | `pack_weights`: 대상 텐서를 memcpy 대신 `repack_tiled32(dst, src, N, K)`로 (행 단위 순회, 4바이트 청크 복사). 헤더에 `weight_layout` 기록. `HexModelConfig`에는 필드 추가 없음(레이아웃은 lowering 상수) |
| `Applications/CausalLM/hexagon/hex_image.cpp` | `.hexcfg`에 `weight_layout=tiled32` 쓰기/읽기. 키 없으면 "legacy image, regenerate with nntr_hexpack" 예외 |
| `test/hexagon/sim/ref_ops.{h,c}` | `ref_matmul_w8a8` / `ref_matmul_logits` / `ref_embed`가 `nntr_htp_tile_off`로 가중치를 읽음 (`ref_matmul_w8a16`은 불변). `ref_dot_i8`은 row-major 전용으로 남기고 타일용 `ref_dot_i8_tiled(w_base, n, xq, K)` 추가 |
| `htp/ops/hvx-matmul.c`, `hvx-embed.c` | **임시 브리지**: 옛 스칼라/벡터 커널이 타일 오프셋으로 행을 읽도록만 수정 (`hvx_dot_i8`은 연속 행을 전제하므로 P2에서는 W8A8/LOGITS를 `ref_dot_i8_tiled`와 같은 스칼라 경로로 임시 전환해도 됨 — 성능 무의미, 정확도만). P3가 교체 |
| `test/hexagon/test_lowering.cpp` | `pack_weights` 후 `nntr_htp_tile_off` 역변환으로 원본 `[N][K]` 복원 확인(모든 텐서, 바이트 동일). 헤더 `weight_layout` 값 확인 |
| `test/hexagon/test_oplist_header.c` | v4 크기·필드 검사 |
| `test/hexagon/sim/test_matmul.c`, `test_logits.c`, `test_embed.c`, `test_graph.c` | 가중치 생성 후 타일 리팩을 거쳐 커널에 주고, ref는 같은 타일 버퍼를 읽음 |
| `docs/backend_guide/HEXAGON.md` §1.4, §2.1, §2.4 | P6에서 일괄 갱신 |

## 이미지 재생성

`nntr_hexpack`으로 `/tmp/qwen3_full.{hexw,hexcfg}`와 1-layer 이미지를 다시 만든다. 크기는 598,623,744로 동일해야 한다(변하면 리팩 버그). 앱 경로(`HexagonBackend::create`)는 `pack_weights`를 직접 부르므로 자동 반영.

## Task

1. `nntr_htp_tile_off` + 헤더 v4 + validator (`test_oplist_header` 실패→통과)
2. `pack_weights` 리팩 + `test_lowering` 역변환 검사
3. `ref_ops` 타일 읽기 + x86 `hexagon_ref_run --eval` 로 **PPL 동일**(20.2718, 386 step) 확인 — 정수 경로라 bit-exact여야 함. 다르면 리팩/역인덱스 버그
4. sim 테스트 4개 + 옛 커널 임시 브리지 → 13개 sim PASS
5. `.hexcfg` 키 + `nntr_hexpack` 재생성 + 크기 확인
6. **HEXAGON.md 갱신** ([06 표](06-verification.md) P2 행): §1.4 ABI v4·`weight_layout`·validator, §2.1 타일 정의와 역인덱스, §2.4 `.hexcfg` 키와 legacy 거부, §5.1 이미지 재생성 안내. `[docs]` 커밋

## 완료 기준

- x86: `test_lowering`, `test_w8cx_bin`, `test_oplist_header` PASS. `hexagon_ref_run --eval` PPL이 P2 전과 소수점 4자리까지 동일.
- sim: 13개 + `profile` PASS.
- 옛 `.hexcfg`(키 없음)는 앱/하네스에서 명시적 오류로 거부.
- HEXAGON.md의 레이아웃·ABI 설명이 코드와 일치한다.
- **PPL_before/PPL_after 기록** (2026-09-15, Task 8): 이 머신에는 §8.1의 원본 `eval.txt`가 없어 로컬 387-토큰 프롬프트(`/home/j2z0/Project/models/eval_prompt512.txt`, 첫 387 토큰)로 대체 측정. `PPL_before == PPL_after == 37.4738`(386 steps, top1 133) — HEAD 6ff5fd9e(전) / 4c5d84b5(후), 로그 `logs/hexagon/x86_eval_before_p2.log` / `logs/hexagon/x86_eval_after_p2.log`. 별개로, sim 프로파일 `acc` 게이트의 STAT은 사용자 결정(2026-09-15)으로 `max_abs=0.0914676 max_rel=93.3662`로 재-baseline됨(01-profile-baseline.md 참조; P2에서 sim 모델 채움이 타일 인덱스를 통한 순열 값이 되었을 뿐 정수 경로는 bit-exact 유지, 0.1/0.1 바운드 통과).
