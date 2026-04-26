# Qualcomm QNN backend integration. Gated entirely behind enable-npu.
# The Qualcomm QNN SDK itself ("Confidential and Proprietary" code) is NOT
# carried in this repository — users must supply it via -Dqnn-sdk-root=<path>.
# See docs/backend_guide/QNN_BUILD.md for instructions.

qnn_sdk_root = get_option('qnn-sdk-root')

if qnn_sdk_root == ''
  error('enable-npu=true requires -Dqnn-sdk-root=<path-to-qcom-qnn-sdk>. ' +
        'Obtain the QNN SDK from https://qpm.qualcomm.com/ and pass its root ' +
        'directory. See docs/backend_guide/QNN_BUILD.md.')
endif

if not import('fs').is_dir(qnn_sdk_root)
  error('qnn-sdk-root does not exist or is not a directory: ' + qnn_sdk_root)
endif

# QNN SDK include directories. Marked is_system to keep -Wall warnings from
# the proprietary headers from polluting the build log.
qnn_sdk_inc = include_directories(
  qnn_sdk_root / 'include',
  qnn_sdk_root / 'include' / 'QNN',
  is_system: true,
)
nntrainer_inc += qnn_sdk_inc

# nntrainer's own QNN integration code (Apache-2.0).
subdir('jni')
