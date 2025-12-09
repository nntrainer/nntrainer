LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../..
endif

# ARM architecture flags - can be overridden via MESON_ARM_MARCH environment variable
# Default: armv8.2-a+fp16+dotprod+i8mm (for compatibility with existing builds)
ifdef MESON_ARM_MARCH
ARM_MARCH_FLAGS := $(MESON_ARM_MARCH)
else
ARM_MARCH_FLAGS := $(ARM_MARCH_FLAGS)
endif


ML_API_COMMON_INCLUDES := ${NNTRAINER_ROOT}/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/ggml_interface \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	${ML_API_COMMON_INCLUDES}

ifeq ($(MESON_ENABLE_OPENCL), 1)
NNTRAINER_INCLUDES += $(NNTRAINER_ROOT)/nntrainer/opencl \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cl_operations \
	$(NNTRAINER_ROOT)/nntrainer/layers/cl_layers 
endif


LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/jni/$(TARGET_ARCH_ABI)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/jni/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so

include $(PREBUILT_SHARED_LIBRARY)


ifeq ($(MESON_ENABLE_OPENCL), 1)
include $(CLEAR_VARS)

LOCAL_MODULE := opencl
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/jni/$(TARGET_ARCH_ABI)/libOpenCL.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := clblast
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/obj/local/$(TARGET_ARCH_ABI)/libclblast.a

include $(PREBUILT_STATIC_LIBRARY)
endif

include $(CLEAR_VARS)
GTEST_PATH := googletest

LOCAL_MODULE := googletest_main
LOCAL_CFLAGS := -Igoogletest/include -Igoogletest/
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions

LOCAL_SRC_FILES := \
    $(GTEST_PATH)/src/gtest-all.cc

include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := test_util
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) ../include

LOCAL_SRC_FILES := ../nntrainer_test_util.cpp

include $(BUILD_STATIC_LIBRARY)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_activations
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti  -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_activations.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)


LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_exe_order
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti  -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_exe_order.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_internal
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti  -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_internal.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_lazy_tensor
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti  -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_lazy_tensor.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)


LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_tensor
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti  -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_tensor.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)


LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_tensor_nhwc
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti  -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_tensor_nhwc.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)


LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_tensor_fp16
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_tensor_fp16.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)


LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_util_func
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_util_func.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_modelfile
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_modelfile.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_graph
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_graph.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_appcontext
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_appcontext.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_base_properties
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_base_properties.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_common_properties
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_common_properties.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_tensor_neon_fp16
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_tensor_neon_fp16.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_tensor_pool
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_tensor_pool.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_tensor_pool_fp16
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_tensor_pool_fp16.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_lr_scheduler
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../unittest/unittest_nntrainer_lr_scheduler.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_compiler
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/compiler -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DNDK_BUILD=1 -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp

LOCAL_SRC_FILES := \
     ../unittest/compiler/compiler_test_util.cpp \
     ../unittest/compiler/unittest_compiler.cpp \
     ../unittest/compiler/unittest_realizer.cpp \

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_models
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/models -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
	../unittest/unittest_nntrainer_models.cpp \
	../unittest/models/models_test_utils.cpp \
	../unittest/models/models_golden_test.cpp 

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_models
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/models -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DNDK_BUILD=1 -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp

LOCAL_SRC_FILES := \
	 ../unittest/models/models_test_utils.cpp \
	 ../unittest/models/models_golden_test.cpp \
	 ../unittest/models/unittest_models_recurrent.cpp \
	 ../unittest/models/unittest_models_multiout.cpp \
	 ../unittest/models/unittest_models.cpp \

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_datasets
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/datasets -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DNDK_BUILD=1 -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp

LOCAL_SRC_FILES := \
	 ../unittest/datasets/data_producer_common_tests.cpp \
	 ../unittest/datasets/unittest_random_data_producers.cpp \
	 ../unittest/datasets/unittest_func_data_producer.cpp \
	 ../unittest/datasets/unittest_raw_file_data_producer.cpp \
	 ../unittest/datasets/unittest_iteration_queue.cpp \
	 ../unittest/datasets/unittest_databuffer.cpp \
	 ../unittest/datasets/unittest_data_iteration.cpp \
	 ../unittest/datasets/unittest_datasets.cpp \


LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_layers
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/layers -I../../nntrainer/layers/loss -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DNDK_BUILD=1 -DENABLE_FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp
ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_CFLAGS += -DEANBLE_OPENCL=1
endif

LOCAL_SRC_FILES := \
	 ../unittest/layers/layers_dependent_common_tests.cpp \
	 ../unittest/layers/layers_standalone_common_tests.cpp \
	 ../unittest/layers/layers_golden_tests.cpp \
	 ../unittest/layers/unittest_layer_node.cpp \
	 ../unittest/layers/unittest_layers.cpp \
	 ../unittest/layers/unittest_layers_impl.cpp \
	 ../unittest/layers/unittest_layers_input.cpp \
	 ../unittest/layers/unittest_layers_loss.cpp \
	 ../unittest/layers/unittest_layers_fully_connected.cpp \
	 ../unittest/layers/unittest_layers_batch_normalization.cpp \
	 ../unittest/layers/unittest_layers_layer_normalization.cpp \
	 ../unittest/layers/unittest_layers_convolution2d.cpp \
	 ../unittest/layers/unittest_layers_convolution1d.cpp \
	 ../unittest/layers/unittest_layers_pooling2d.cpp \
	 ../unittest/layers/unittest_layers_flatten.cpp \
	 ../unittest/layers/unittest_layers_activation.cpp \
	 ../unittest/layers/unittest_layers_addition.cpp \
	 ../unittest/layers/unittest_layers_multiout.cpp \
	 ../unittest/layers/unittest_layers_rnn.cpp \
	 ../unittest/layers/unittest_layers_rnncell.cpp \
	 ../unittest/layers/unittest_layers_lstm.cpp \
	 ../unittest/layers/unittest_layers_lstmcell.cpp \
	 ../unittest/layers/unittest_layers_gru.cpp \
	 ../unittest/layers/unittest_layers_grucell.cpp \
	 ../unittest/layers/unittest_layers_preprocess_flip.cpp \
	 ../unittest/layers/unittest_layers_split.cpp \
	 ../unittest/layers/unittest_layers_embedding.cpp \
	 ../unittest/layers/unittest_layers_concat.cpp \
	 ../unittest/layers/unittest_layers_permute.cpp \
	 ../unittest/layers/unittest_layers_attention.cpp \
	 ../unittest/layers/unittest_layers_dropout.cpp \
	 ../unittest/layers/unittest_layers_reshape.cpp \
	 ../unittest/layers/unittest_layers_multi_head_attention.cpp \
	 ../unittest/layers/unittest_layers_positional_encoding.cpp \

ifeq ($(MESON_ENABLE_OPENCL), 1)

LOCAL_SRC_FILES += \
	 ../unittest/layers/unittest_layers_transpose_cl.cpp \
	 ../unittest/layers/unittest_layers_concat_cl.cpp \
	 ../unittest/layers/unittest_layers_swiglu_cl.cpp \
	 ../unittest/layers/unittest_layers_fully_connected_cl.cpp \
	 ../unittest/layers/unittest_layers_addition_cl.cpp \
	 ../unittest/layers/unittest_layers_rmsnorm_cl.cpp \
	 ../unittest/layers/unittest_layers_reshape_cl.cpp 
endif

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

ifeq ($(MESON_ENABLE_OPENCL), 1)
include $(CLEAR_VARS)

LOCAL_MODULE := unittest_blas_kernels_cl
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/layers -I../../nntrainer/layers/loss -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DNDK_BUILD=1 -DENABLE_FP16=1 -DENABLE_OPENCL=1 
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp

LOCAL_SRC_FILES := \
	 ../unittest/unittest_blas_kernels_cl.cpp 

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_attention_kernels_cl
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/layers -I../../nntrainer/layers/loss -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DNDK_BUILD=1 -DENABLE_FP16=1 -DENABLE_OPENCL=1 -DUSE__FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp

LOCAL_SRC_FILES := \
	 ../unittest/unittest_attention_kernels_cl.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer 
LOCAL_STATIC_LIBRARIES := googletest_main test_util


ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)
endif

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_cpu_backend
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/layers -I../../nntrainer/layers/loss -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DNDK_BUILD=1 -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp

LOCAL_SRC_FILES := \
	 ../unittest/unittest_nntrainer_cpu_backend.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := unittest_nntrainer_cpu_backend_fp16
LOCAL_CFLAGS := -Igoogletest/include -I../include -I../unittest/layers -I../../nntrainer/layers/loss -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti -DNDK_BUILD=1 -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp

LOCAL_SRC_FILES := \
	 ../unittest/unittest_nntrainer_cpu_backend_fp16.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)

# unittest_ccapi
include $(CLEAR_VARS)

LOCAL_MODULE := unittest_ccapi
LOCAL_CFLAGS := -Igoogletest/include -I../include -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 $(ARM_MARCH_FLAGS) -O3 -frtti
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    ../ccapi/unittest_ccapi.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main test_util

ifeq ($(MESON_ENABLE_OPENCL), 1)
LOCAL_SHARED_LIBRARIES += opencl
LOCAL_STATIC_LIBRARIES += clblast
endif

include $(BUILD_EXECUTABLE)
