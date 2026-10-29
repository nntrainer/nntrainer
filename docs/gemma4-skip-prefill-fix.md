# Gemma4 skip-prefill LM head 호환성 수정

이 변경은 Linux KV Manager 개발과 분리한
`fix/gemma4-skip-prefill-lmhead` 브랜치에서 검증했다.
기준 소스는 `141871a58ff439e3caeac40c1632ff8ac3543d20`이다.
앞으로 KV Manager의 주 모델 검증은 사용자 결정에 따라 Qwen3 0.6B를 사용한다.
여기서의 Gemma4 실행은 이 호환성 수정의 회귀 확인이다.

## 원인과 수정 범위

`CausalLM::buildLmHeadOutput()`은 `skip_prefill=true`일 때 모든 종류의 LM head에
`plan_last_row_only=true`를 전달했다. 이 속성은 `fully_connected`의 출력 계획용
속성이며, `tie_word_embeddings`와 `lm_head`는 지원하지 않는다. 따라서 tied
embedding을 쓰는 Gemma4는 weight를 읽기 전에 다음 오류로 초기화에 실패했다.

```text
[LayerImpl] Unknown Layer Properties count 3
```

두 전용 head는 이미 `finalize()`에서 출력 height를 1로 고정한다. 따라서 FC head에만
`plan_last_row_only`를 전달하고, 전용 head에는 기존 `skip_prefill`만 전달한다.
속성 검증을 완화하거나 layer에 새 API를 추가하지 않는다. `lmhead_untie=true`인
FC head의 height 1 최적화는 유지한다.

별도로 발견된 기존 Gemma4 양자화 `.bin`의 byte 크기 불일치는 이 수정의 범위가 아니다.
실제 2,708,362,380 bytes와 현 graph 요구 3,313,915,020 bytes의 차이
605,552,640 bytes는 PLE table의 Q4_0/Q6_K 크기 차이와 일치하지만, 파일에 tensor
dtype metadata가 없으므로 이 수치만으로 원래 dtype을 확정하지 않는다. 해당 파일을
강제로 읽거나 padding하지 않았다. 원본 `~/TEST_MODELS` 파일도 수정하지 않았다.

## 검증 결과

환경: Ubuntu 22.04, GCC/G++ 11.4, CPU, FP32 activation, 8 NNTrainer threads,
OpenBLAS 1 thread, greedy decoding. 빌드는 이미 검증된 `build-kv-cpu`의 compile/link
명령과 object를 별도 디렉터리로 snapshot하고, 이 브랜치의 변경 C++ translation
unit 2개를 다시 컴파일·재링크했다. 원래 개발 디렉터리의 source/object/library는
변경하지 않았다. 전체 fresh build 및 Android/GPU 검증을 수행한 것으로 해석하지 않는다.

| 검사 | 결과 |
|---|---|
| 새 테스트를 변경 전 library에 실행 | 6개 중 tied/custom head 초기화 2개가 위 오류로 실패. 기존 FC와 음성 테스트 4개는 통과. |
| 새 테스트를 수정 library에 실행 | 6/6 통과. tied/custom/FC head 초기화·실행과 output height 1 확인. |
| 속성 음성 테스트 | 전용 head의 FC 전용 속성, FC의 잘못된 bool 값 거절을 각각 확인. 정상 3개·음성 3개 추가. |
| 전체 CausalLM model test | 총 100개 중 99 PASS, deprecated QINT4 fixture 1 SKIP, exit 0. |
| 원본 Gemma4 FP32, `skip_prefill=true`, 산술 | `What is 2 + 2? Answer with only the number.` → `4<turn\|>`, exit 0. |
| 원본 Gemma4 FP32, `skip_prefill=true`, 한국어 | `대한민국의 수도는 어디인가요? 도시 이름만 답하세요.` → `서울<turn\|>`, exit 0. |

실제 모델 smoke는 별도 설정을 사용했다: `init_seq_len=64`, `max_seq_len=256`,
`num_to_generate=48`, `do_sample=false`, `skip_prefill=true`. 짧은 두 응답의 정상
생성을 확인한 결과이며 HF logits 일치나 장문 정확성 검증은 아니다.

로컬 증거는 `/tmp/nntrainer-gemma4-fix/build-snapshot/`에 남겼다.

- `baseline-regression.log`, `fixed-regression.log`
- `fixed-models-full.log`, `fixed-models-full.xml`
- `gemma4-fp32-smoke.json`, prompt별 `.stdout.txt` / `.stderr.txt`

정규 Meson 빌드에서 동일 테스트는 다음과 같이 실행한다.

```bash
ninja -C build-kv-cpu Applications/CausalLM/unittest_causallm_models
NNTR_ENGINE=cpu NNTR_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  meson test -C build-kv-cpu --print-errorlogs unittest_causallm_models
# 신규 회귀만 선택하려면
NNTR_ENGINE=cpu NNTR_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  build-kv-cpu/Applications/CausalLM/unittest_causallm_models \
  '--gtest_filter=*Gemma4SkipPrefillHeadTest*'
```

변경 C/C++에 clang-format 14를 적용했고 무관한 기존 formatting 변경은 제외했다.
`AGENTS.md`와 vendored dependency source는 변경하지 않았다.
