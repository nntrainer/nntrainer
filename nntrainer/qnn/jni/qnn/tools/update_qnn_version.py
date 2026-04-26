import os
import shutil

def is_from_qnn(file_path):
	# Only check cpp/hpp/h
	if not file_path.endswith('.cpp') and not file_path.endswith('.hpp') and not file_path.endswith('.h') :
		return False

	# check if 'Qualcomm Technologies, Inc.' is in content
	with open(file_path, 'r') as actual_file:
		content = actual_file.read()
		return 'Qualcomm Technologies, Inc.' in content

def recursive_overwrite(src, dest, ignore=None):
	if os.path.isdir(src):
		if not os.path.isdir(dest):
			os.makedirs(dest)
		files = os.listdir(src)
		if ignore is not None:
			ignored = ignore(src, files)
		else:
			ignored = set()
		for f in files:
			if f not in ignored:
				recursive_overwrite(os.path.join(src, f), 
									os.path.join(dest, f), 
									ignore)
	else:
		shutil.copyfile(src, dest)

def find_replace_in_file(file_path, old_text, new_text):
	with open(file_path, 'r') as read_file:
		file_data = read_file.read()
	file_data = file_data.replace(old_text, new_text)
	with open(file_path, 'w') as write_file:
		write_file.write(file_data)

# Usage: python3 update_qnn_version.py
if __name__ == '__main__':
	target_directories = [
		'npu/jni/qnn/Log',
		'npu/jni/qnn/PAL',
		'npu/jni/qnn/QNN',
		'npu/jni/qnn/Utils',
		'npu/jni/qnn/WrapperUtils',
		'npu/jni/qnn-api'
	]

	sampleapp_src_prefix = 'examples/QNN/SampleApp/SampleApp/src'

	src_directories = [
		os.path.join(sampleapp_src_prefix, 'Log'),
		os.path.join(sampleapp_src_prefix, 'PAL'),
		'include/QNN',
		os.path.join(sampleapp_src_prefix, 'Utils'),
		os.path.join(sampleapp_src_prefix, 'WrapperUtils'),
		'examples/Genie/Genie/src/qualla/engines/qnn-api',
	]

	assert os.environ.get('QNN_SDK_ROOT'), 'this screipt will replace qnn files in npu/jni with QNN_SDK_ROOT, set QNN_SDK_ROOT env variable'
	qnn_root = os.environ.get('QNN_SDK_ROOT')

	for target_dir, source_dir in zip(target_directories, src_directories):
		target_files = list()
		for root, dirs, files in os.walk(target_dir):
			for file in files:
				cur_path = os.path.join(root, file)
				if not is_from_qnn(cur_path):
					continue
				os.remove(cur_path)


		assert os.path.exists(os.path.join(qnn_root, source_dir)), f'{source_dir} does not exist'
		recursive_overwrite(os.path.join(qnn_root, source_dir), target_dir)

	os.remove('npu/jni/qnn/QnnTypeMacros.hpp')
	shutil.copyfile(os.path.join(qnn_root, sampleapp_src_prefix, 'QnnTypeMacros.hpp'), 'npu/jni/qnn/QnnTypeMacros.hpp')

	find_replace_in_file('npu/jni/qnn/Utils/DynamicLoadUtil.hpp', 'SampleApp.hpp', 'QNN.hpp')
	find_replace_in_file('npu/jni/qnn/Utils/QnnSampleAppUtils.hpp', 'SampleApp.hpp', 'QNN.hpp')
	find_replace_in_file('npu/jni/qnn/Utils/IOTensor.hpp', 'private', 'public')
	#find_replace_in_file('npu/jni/qnn-api/QnnTypeDef.hpp', 'Log.hpp', 'Log/Logger.hpp')
	find_replace_in_file('npu/jni/qnn-api/BackendExtensions.hpp', 'Log.hpp', 'Log/Logger.hpp')
	find_replace_in_file('npu/jni/qnn-api/BackendExtensions.cpp', 'Log.hpp', 'Log/Logger.hpp')

	os.remove('npu/jni/qnn-api/QnnTypeMacros.hpp')
	os.remove('npu/jni/qnn-api/Log.hpp')
	os.remove('npu/jni/qnn-api/ClientBuffer.hpp')
	os.remove('npu/jni/qnn-api/ClientBuffer.cpp')
	os.remove('npu/jni/qnn-api/DmaBufAllocator.hpp')
	os.remove('npu/jni/qnn-api/DmaBufAllocator.cpp')
	os.remove('npu/jni/qnn-api/IBufferAlloc.hpp')
	os.remove('npu/jni/qnn-api/IOTensor.hpp')
	os.remove('npu/jni/qnn-api/IOTensor.cpp')
	os.remove('npu/jni/qnn-api/qnn-utils.hpp')
	os.remove('npu/jni/qnn-api/qnn-utils.cpp')
	os.remove('npu/jni/qnn-api/QnnApi.hpp')
	os.remove('npu/jni/qnn-api/QnnApi.cpp')
	os.remove('npu/jni/qnn-api/QnnApiUtils.hpp')
	os.remove('npu/jni/qnn-api/QnnApiUtils.cpp')
	os.remove('npu/jni/qnn-api/RpcMem.hpp')
	os.remove('npu/jni/qnn-api/RpcMem.cpp')
	os.remove('npu/jni/qnn-api/QnnWrapperUtils.hpp')

	shutil.rmtree('npu/jni/qnn/PAL/src/windows')
