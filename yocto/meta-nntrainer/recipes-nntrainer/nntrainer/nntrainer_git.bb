SUMMARY = "NNTrainer: on-device neural network training and inference"
HOMEPAGE = "https://github.com/nntrainer/nntrainer"
SECTION = "libs"
# iniparser (MIT) from subprojects/ is linked statically into libnntrainer.
LICENSE = "Apache-2.0 & MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=095e13fef457e259d3bc155d0ed859f1 \
                    file://subprojects/iniparser/LICENSE;md5=8474d3b745f77e203f1fc82fb0bb7678"

# Used when building outside CI. CI replaces the source with the checked-out
# tree through externalsrc (EXTERNALSRC:pn-nntrainer).
SRC_URI = "gitsm://github.com/nntrainer/nntrainer.git;branch=main;protocol=https"
SRCREV = "${AUTOREV}"
PV = "0.6.0+git"
S = "${WORKDIR}/git"

# iniparser is built from the bundled subprojects/iniparser (meson
# cmake.subproject), so cmake-native is needed and no iniparser recipe is.
# openblas must come from the recipe: if meson cannot find it, meson.build
# builds subprojects/OpenBLAS with the host cmake instead.
DEPENDS = "openblas ml-api cmake-native"

inherit meson pkgconfig

EXTRA_OEMESON = " \
    -Dplatform=yocto \
    -Denable-blas=true \
    -Denable-app=false \
    -Dinstall-app=false \
    -Denable-capi=enabled \
    -Denable-ccapi=true \
    -Dml-api-support=enabled \
    -Denable-test=false \
    -Denable-logging=true \
    -Denable-nnstreamer-tensor-filter=disabled \
    -Denable-nnstreamer-tensor-trainer=disabled \
    -Denable-nnstreamer-backbone=false \
    -Denable-tflite-backbone=false \
    -Denable-tflite-interpreter=false \
"

# nntrainer installs unversioned shared libraries (libnntrainer.so,
# libccapi-nntrainer.so, libcapi-nntrainer.so); ship them in the main package.
FILES_SOLIBSDEV = ""
FILES:${PN} += "${libdir}/*.so"
INSANE_SKIP:${PN} += "dev-so"
