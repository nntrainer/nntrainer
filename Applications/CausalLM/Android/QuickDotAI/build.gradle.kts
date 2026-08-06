// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
//
// QuickDotAI — reusable AAR that bundles the QuickDotAI interface and
// both concrete implementations (LiteRTLm + NativeQuickDotAI) plus the
// JNI shim (libquickai_jni.so) and the CausalLM prebuilt shared
// libraries. Third-party apps can depend on this AAR to run on-device
// LLMs without linking QuickAIService or any of LauncherApp's REST
// plumbing.

import java.io.File
import java.util.Properties

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
}

val nntrainerNdkPath =
    providers.gradleProperty("nntrainerNdkPath").orNull
        ?.trim()
        ?.takeIf { it.isNotEmpty() }
        ?: throw GradleException(
            "Missing -PnntrainerNdkPath. Run " +
                "Applications/CausalLM/build_android.sh --app, or pass " +
                "the absolute path of the NDK used for the native prebuilts."
        )

val requestedNdkDirectory = File(nntrainerNdkPath)
if (!requestedNdkDirectory.isAbsolute) {
    throw GradleException("nntrainerNdkPath must be absolute: $nntrainerNdkPath")
}
val nntrainerNdkDirectory = requestedNdkDirectory.canonicalFile
val ndkSourcePropertiesFile = nntrainerNdkDirectory.resolve("source.properties")
if (!ndkSourcePropertiesFile.isFile) {
    throw GradleException(
        "Invalid nntrainerNdkPath; source.properties is missing: " +
            ndkSourcePropertiesFile
    )
}
val installedNdkProperties = Properties()
ndkSourcePropertiesFile.inputStream().use { installedNdkProperties.load(it) }
val nntrainerNdkRevision =
    installedNdkProperties.getProperty("Pkg.Revision")
        ?.trim()
        ?.takeIf { it.isNotEmpty() }
        ?: throw GradleException(
            "Pkg.Revision is missing from $ndkSourcePropertiesFile"
        )

// Sync transitive prebuilts into jniLibs, removing stale files.
val prebuiltNativeLibsDir =
    layout.buildDirectory.dir("generated/jniLibs/arm64-v8a")

val copyPrebuiltNativeLibs = tasks.register<Sync>("copyPrebuiltNativeLibs") {
    from(project.file("prebuilt_libs"))
    include("*.so")
    include("htp_backend_ext_config.json")
    // externalNativeBuild already packages the API library and C++ runtime.
    exclude("libquick_dot_ai_api.so", "libc++_shared.so")
    into(prebuiltNativeLibsDir)
}

android {
    namespace = "com.example.quickdotai"
    compileSdk = 36
    ndkVersion = nntrainerNdkRevision
    ndkPath = nntrainerNdkDirectory.path

    packaging {
        jniLibs.useLegacyPackaging = true
    }


    defaultConfig {
        minSdk = 33

        ndk {
            // Only arm64-v8a is supported by the prebuilt CausalLM libraries.
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -frtti -fexceptions"
                // Match prebuilts and package one shared C++ runtime.
                arguments += "-DANDROID_STL=c++_shared"
            }
        }

        consumerProguardFiles("consumer-rules.pro")
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets {
        getByName("main") {
            // Pick up the generated/jniLibs/<abi>/*.so tree produced by
            // copyPrebuiltNativeLibs above, alongside any hand-placed files
            // in src/main/jniLibs/.
            //
            // `buildDir` getter was deprecated in Gradle 8 and removed in
            // Gradle 9; use the Provider-based layout.buildDirectory API
            // so this file is forward-compatible with Gradle 9 if we
            // ever roll the wrapper back up.
            jniLibs.srcDirs(
                "src/main/jniLibs",
                layout.buildDirectory.dir("generated/jniLibs").get().asFile
            )
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "consumer-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        // LiteRT-LM 0.10.0 (the version our mirror serves) was compiled
        // with Kotlin 2.3.0 but our compiler is 2.2.21; the flag tells
        // kotlinc to accept the newer metadata stamp on the LiteRT-LM
        // AAR this module directly links against. See
        // libs.versions.toml for the full story.
        freeCompilerArgs += "-Xskip-metadata-version-check"
    }
}

// The merge*JniLibFolders task reads android.sourceSets.main.jniLibs and
// stages the native libraries for packaging into the AAR, so make it
// depend on the copy task. ExternalNativeBuild also benefits because the
// CMake link step reads libquick_dot_ai_api.so directly from prebuilt_libs.
tasks.matching {
    it.name.startsWith("merge") && it.name.endsWith("JniLibFolders")
}.configureEach {
    dependsOn(copyPrebuiltNativeLibs)
}
tasks.matching { it.name.startsWith("externalNativeBuild") }.configureEach {
    dependsOn(copyPrebuiltNativeLibs)
}

dependencies {
    // kotlinx.serialization is exposed as an `api` dependency because the
    // public types (ModelId, BackendType, LoadModelRequest, …) carry
    // @Serializable annotations so consumers that want to JSON-ify them
    // can do so without pulling the runtime in themselves.
    api(libs.kotlinx.serialization.json)

    // LiteRT-LM is the engine used by LiteRTLm.kt for Gemma-family models.
    // Exposed as `api` so consumers don't have to redeclare it.
    //
    // Pinned to an explicit version via the version catalog instead of
    // `latest.release`: dynamic versions are non-deterministic (they
    // resolve differently depending on what each environment's Maven
    // mirror happens to cache) and they caused a hard failure earlier
    // when one mirror served 0.10.0 as "latest" while our Kotlin
    // compiler was pinned to a version that could not read 0.10.0's
    // metadata stamp. See gradle/libs.versions.toml for the rationale
    // behind the exact pin.
    api(libs.litertlm.android)
    
    // AndroidX Core for createBitmap and other utility functions
    implementation("androidx.core:core-ktx:1.12.0")
}
