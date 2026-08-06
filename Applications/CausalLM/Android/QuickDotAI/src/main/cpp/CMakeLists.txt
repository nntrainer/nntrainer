# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Build the JNI shim against the canonical API header and staged arm64-v8a
# prebuilts. Gradle packages the JNI library and its dependencies.
cmake_minimum_required(VERSION 3.22.1)
project(quickai_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Canonical QuickDotAI API headers.
set(CAUSALLM_API_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../api")

# Staged arm64-v8a native libraries.
set(PREBUILT_LIBS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../prebuilt_libs")

add_library(quick_dot_ai_api SHARED IMPORTED)
set_target_properties(quick_dot_ai_api PROPERTIES
    IMPORTED_LOCATION
        "${PREBUILT_LIBS_DIR}/libquick_dot_ai_api.so")

add_library(quickai_jni SHARED
    quickai_jni.cpp)


target_include_directories(quickai_jni PRIVATE
    ${CAUSALLM_API_DIR})

find_library(log-lib log)

target_link_libraries(quickai_jni
    quick_dot_ai_api
    ${log-lib})
