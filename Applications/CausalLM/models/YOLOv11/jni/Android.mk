LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../../../..
endif

# Prebuilt nntrainer shared libs (produced by the meson-android build under
# build_android/jni/arm64-v8a). Override NNTRAINER_LIBS if they live elsewhere.
ifndef NNTRAINER_LIBS
NNTRAINER_LIBS := $(NNTRAINER_ROOT)/build_android/jni/$(TARGET_ARCH_ABI)
endif

ML_API_COMMON_INCLUDES := $(NNTRAINER_ROOT)/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/ggml_interface \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	$(ML_API_COMMON_INCLUDES)

include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_LIBS)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_LIBS)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_ARM_MODE := arm
LOCAL_MODULE := yolov11_infer
LOCAL_MODULE_TAGS := optional
LOCAL_LDLIBS := -llog -landroid

# Match the library's ISA features so int8 (dotprod/i8mm) and fp16 paths
# compile consistently with libnntrainer. ENABLE_FP16/USE__FP16 mirror the
# CausalLM app mk and meson's enable-fp16 block so app-side / c2psa FP16-gated
# code (and the HalfTensor FP16 quant-conv path) compiles for the W4A16 /
# FP16-activation build.
LOCAL_CFLAGS += -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod+i8mm -pthread \
	-fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_CXXFLAGS += -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod+i8mm -pthread \
	-fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_LDFLAGS += -fexceptions

LOCAL_SRC_FILES := main.cpp c2psa_layer.cpp
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)

# Standalone TDD correctness probe for the hand-written Q8_0xQ8_0 SMMLA asm
# kernel (nntr_gemm_q8_0_4x8_q8_0_fp16), verified against the intrinsics
# oracle nntr_gemm_q8_0_q8_0_4x4_fp16. Not part of the app itself.
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_ARM_MODE := arm
LOCAL_MODULE := gemm_q8_0_probe
LOCAL_MODULE_TAGS := optional
LOCAL_LDLIBS := -llog -landroid

LOCAL_CFLAGS += -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod+i8mm -pthread \
	-fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_CXXFLAGS += -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod+i8mm -pthread \
	-fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_LDFLAGS += -fexceptions

LOCAL_SRC_FILES := gemm_q8_0_probe.cpp
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)

# Timing-only microbenchmark: tests whether the Q8_0 vs Q4_0 GEMM speed gap
# widens with weight working-set size (cache-locality hypothesis), since this
# device denies hardware PMU access to non-root shell (simpleperf stat fails
# with "Event type ... is not supported on the device").
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_ARM_MODE := arm
LOCAL_MODULE := gemm_cache_bench
LOCAL_MODULE_TAGS := optional
LOCAL_LDLIBS := -llog -landroid

LOCAL_CFLAGS += -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod+i8mm -pthread \
	-fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_CXXFLAGS += -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod+i8mm -pthread \
	-fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_LDFLAGS += -fexceptions

LOCAL_SRC_FILES := gemm_cache_bench.cpp
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)
