import os
import shutil

VENDOR_DIR = 'qnn/jni/vendor'
SAMPLEAPP_SRC_PREFIX = 'examples/QNN/SampleApp/SampleApp/src'
GENIE_SRC_PREFIX = 'examples/Genie/Genie/src'

def is_from_qnn(file_path):
	if not file_path.endswith(('.cpp', '.hpp', '.h')):
		return False
	with open(file_path, 'r') as f:
		return 'Qualcomm Technologies, Inc.' in f.read()

def recursive_overwrite(src, dest, ignore=None):
	if os.path.isdir(src):
		if not os.path.isdir(dest):
			os.makedirs(dest)
		files = os.listdir(src)
		ignored = ignore(src, files) if ignore else set()
		for f in files:
			if f not in ignored:
				recursive_overwrite(os.path.join(src, f), os.path.join(dest, f), ignore)
	else:
		shutil.copyfile(src, dest)

def find_replace_in_file(file_path, old_text, new_text):
	with open(file_path, 'r') as f:
		file_data = f.read()
	file_data = file_data.replace(old_text, new_text)
	with open(file_path, 'w') as f:
		f.write(file_data)

def remove_path(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	elif os.path.exists(path):
		os.remove(path)

# ── 1. Copy from SDK ──────────────────────────────────────────────────────

# Maps target dirs (under VENDOR_DIR) to source dirs (under QNN_SDK_ROOT)
target_src_dirs = {
	'Log':            os.path.join(SAMPLEAPP_SRC_PREFIX, 'Log'),
	'PAL':            os.path.join(SAMPLEAPP_SRC_PREFIX, 'PAL'),
	'QNN':            'include/QNN',
	'Utils':          os.path.join(SAMPLEAPP_SRC_PREFIX, 'Utils'),
	'WrapperUtils':   os.path.join(SAMPLEAPP_SRC_PREFIX, 'WrapperUtils'),
	'qnn-api':        os.path.join(GENIE_SRC_PREFIX, 'qualla/engines/qnn-api'),
}

# Additional individual files to copy: (src_rel_path, target_rel_path)
extra_files = [
	(os.path.join(SAMPLEAPP_SRC_PREFIX, 'SampleApp.hpp'),
	 os.path.join(VENDOR_DIR, 'QNN.hpp')),
	(os.path.join(SAMPLEAPP_SRC_PREFIX, 'QnnTypeMacros.hpp'),
	 os.path.join(VENDOR_DIR, 'QnnTypeMacros.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'resource-manager/include/ResourceManager.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/ResourceManager.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'resource-manager/src/ResourceManager.cpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/ResourceManager.cpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'qualla/include/qualla/detail/Log.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/Log.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'qualla/include/qualla/detail/dlOpenWrapper.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/dlOpenWrapper.hpp')),
]

if __name__ == '__main__':
	assert os.environ.get('QNN_SDK_ROOT'), 'Set QNN_SDK_ROOT env variable'
	qnn_root = os.environ.get('QNN_SDK_ROOT')

	# Copy directories
	for target_name, src_rel in target_src_dirs.items():
		src_path = os.path.join(qnn_root, src_rel)
		target_path = os.path.join(VENDOR_DIR, target_name)
		if not os.path.exists(src_path):
			print(f'WARNING: {src_rel} does not exist in SDK, skipping')
			continue
		# Remove old Qualcomm files from target
		for root, dirs, files in os.walk(target_path):
			for f in files:
				cur = os.path.join(root, f)
				if is_from_qnn(cur):
					os.remove(cur)
		recursive_overwrite(src_path, target_path)

	# Copy extra files
	for src_rel, target_rel in extra_files:
		src_path = os.path.join(qnn_root, src_rel)
		if os.path.exists(src_path):
			os.makedirs(os.path.dirname(target_rel), exist_ok=True)
			shutil.copyfile(src_path, target_rel)

	# ── 2. Remove unused files/directories ────────────────────────────────

	# Unused QNN backend subdirs (we only use HTP and System)
	for d in ['CPU', 'DSP', 'GPU', 'GPU.unused', 'HTA', 'Saver', 'IR',
	          'LPAI', 'TFLiteDelegate', 'GenAiTransformer',
	          'LoraAdapterBinUpdater', 'HTPQEMU']:
		remove_path(os.path.join(VENDOR_DIR, 'QNN', d))

	# HTP/core is internal headers not needed for our build
	remove_path(os.path.join(VENDOR_DIR, 'QNN', 'HTP', 'core'))

	# PAL windows (not needed for Android)
	remove_path(os.path.join(VENDOR_DIR, 'PAL/src/windows'))

	# Unused files from qnn-api with unresolvable dependencies
	for f in ['ClientBuffer.hpp', 'ClientBuffer.cpp',
	          'DmaBufAllocator.hpp', 'DmaBufAllocator.cpp', 'IBufferAlloc.hpp',
	          'IOTensor.hpp', 'IOTensor.cpp', 'qnn-utils.hpp', 'qnn-utils.cpp',
	          'QnnApi.hpp', 'QnnApi.cpp', 'QnnApiUtils.hpp', 'QnnApiUtils.cpp',
	          'RpcMem.hpp', 'RpcMem.cpp', 'QnnWrapperUtils.hpp',
	          'BufferUtils.hpp', 'BufferUtils.cpp',
	          'QnnTypeUtils.hpp', 'QnnTypeUtils.cpp']:
		remove_path(os.path.join(VENDOR_DIR, 'qnn-api', f))

	# Unused qnn-api subdirs with unresolvable dependencies
	for d in ['buffer', 'config', 'PAL']:
		remove_path(os.path.join(VENDOR_DIR, 'qnn-api', d))

	# Orphaned qualla/ at vendor root (duplicate of qnn-api/qualla/)
	remove_path(os.path.join(VENDOR_DIR, 'qualla'))

	# ── 3. Apply compatibility patches ─────────────────────────────────────

	# SDK uses SampleApp.hpp, our build uses QNN.hpp
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/DynamicLoadUtil.hpp'),
		'SampleApp.hpp', 'QNN.hpp')
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/QnnSampleAppUtils.hpp'),
		'SampleApp.hpp', 'QNN.hpp')

	# IOTensor.hpp: make members public for our usage
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/IOTensor.hpp'),
		'private', 'public')

	# dlOpenWrapper.hpp: fix const-correctness
	dl_open_wrapper = os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/dlOpenWrapper.hpp')
	find_replace_in_file(dl_open_wrapper, 'static const int s_anchor', 'static int s_anchor')
	find_replace_in_file(dl_open_wrapper, 'reinterpret_cast<const void*>', 'reinterpret_cast<void*>')

	print('QNN vendor files updated successfully.')
