# nntrainer/nntrainer/qnn/meson.build
# Defines qnn_sources_abs and qnn_inc_abs for the ndk-build Android.mk module.
# Gated by the parent (enable-npu). No external -Dqnn-sdk-root: the QNN 2.47
# headers + sample utilities are vendored under jni/vendor/.

qnn_src_dir = meson.current_source_dir()

qnn_sources_abs = [
  qnn_src_dir / 'jni/vendor/Log/Logger.cpp',
  qnn_src_dir / 'jni/vendor/Log/LogUtils.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/common/GetOpt.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/common/StringOp.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/linux/Directory.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/linux/DynamicLoading.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/linux/FileOp.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/linux/Path.cpp',
  qnn_src_dir / 'jni/vendor/Utils/DataUtil.cpp',
  qnn_src_dir / 'jni/vendor/Utils/DynamicLoadUtil.cpp',
  qnn_src_dir / 'jni/vendor/Utils/IOTensor.cpp',
  qnn_src_dir / 'jni/vendor/Utils/QnnDlcUtils.cpp',
  qnn_src_dir / 'jni/vendor/Utils/QnnSampleAppUtils.cpp',
  qnn_src_dir / 'jni/vendor/WrapperUtils/QnnWrapperUtils.cpp',
  qnn_src_dir / 'jni/vendor/qnn-api/dlwrap.cpp',
  qnn_src_dir / 'jni/vendor/qnn-api/BackendExtensions.cpp',
  qnn_src_dir / 'jni/vendor/qnn-api/ResourceManager.cpp',
  qnn_src_dir / 'jni/QNNGraph.cpp',
  qnn_src_dir / 'jni/qnn_properties.cpp',
  qnn_src_dir / 'qnn_context.cpp',
  qnn_src_dir / 'jni/qnn_rpc_manager.cpp',
  qnn_src_dir / 'jni/rpc_mem.cpp',
]

qnn_inc_abs = [
  qnn_src_dir,
  qnn_src_dir / 'jni',
  qnn_src_dir / 'jni/vendor',
  qnn_src_dir / 'jni/vendor/QNN',
  qnn_src_dir / 'jni/vendor/Log',
  qnn_src_dir / 'jni/vendor/PAL/include',
  qnn_src_dir / 'jni/vendor/Utils',
  qnn_src_dir / 'jni/vendor/WrapperUtils',
  qnn_src_dir / 'jni/vendor/qnn-api',
]

qnn_inc = include_directories(
  '.',
  'jni',
  'jni/vendor',
  'jni/vendor/QNN',
  'jni/vendor/Log',
  'jni/vendor/PAL/include',
  'jni/vendor/Utils',
  'jni/vendor/WrapperUtils',
  'jni/vendor/qnn-api',
)

# Expose qnn headers (qnn_context.h 등) to the rest of nntrainer.
nntrainer_inc += qnn_inc
nntrainer_inc_abs += qnn_src_dir
