# Execute gbs with --define "testcoverage 1" in case that you must get unittest coverage statistics
%define         use_cblas 1
%define         nnstreamer_filter 1
%define         nnstreamer_trainer 1
%define         nnstreamer_subplugin_path lib/nnstreamer
%define         use_gym 0
%define         use_ruy 0
%define         use_biqgemm 0
%define         support_ccapi 1
%define         support_nnstreamer_backbone 1
%define         support_tflite_backbone 1
%define         support_tflite_interpreter 1
%define         nntrainerapplicationdir %{_libdir}/nntrainer/bin
%define         gen_input $(pwd)/test/input_gen/genInput.py
%define         support_data_augmentation_opencv 1
%define         configure_subplugin_install_path -Dnnstreamer-subplugin-install-path=%{nnstreamer_subplugin_path}
%bcond_with tizen

# dependency resolution
%if 0%{tizen_version_major}%{tizen_version_minor} >= 65
# (rpm -qi nnstreamer --queryformat '%%{VERSION}' | sed -e 's/\.//g') >= 171

%define         capi_machine_learning_common	   capi-machine-learning-common
%define         capi_machine_learning_inference	 capi-machine-learning-inference

%define         capi_ml_common_pkg_name          capi-ml-common
%define         capi_ml_inference_pkg_name       capi-ml-inference

%else

%if 0%{tizen_version_major}%{tizen_version_minor} >= 60
# (rpm -qi nnstreamer --queryformat '%%{VERSION}' | sed -e 's/\.//g') >= 160
%define         capi_machine_learning_common     capi-ml-common
%define         capi_machine_learning_inference  capi-nnstreamer

%define         capi_ml_common_pkg_name          capi-nnstreamer
%define         capi_ml_inference_pkg_name       capi-nnstreamer
%else
%{error: nnst_version < 6.0 is not supported}
# shouldn't reach here. Below is for the reference.
# If you prepare ml-common-api.h, below can work though.
%define         capi_machine_learning_inference  capi-nnstreamer
%define         capi_machine_learning_common	   capi-nnstreamer

%define         capi_ml_common_pkg_name          capi-nnstreamer
%define         capi_ml_inference_pkg_name       capi-nnstreamer
%endif # 0%{tizen_version_major}%{tizen_version_minor} >= 60

%endif # 0%{tizen_version_major}%{tizen_version_minor} >= 65

### nntrainer fp16 implementation relies on NEON, which requires armv8.2-a
### armv7l Tizen: do not support fp16 neon.
### aarch64 Tizen: uses armv8.0a. no fp16 neon.
### x86/x64 Tizen: it has gcc9. x86 requires gcc >= 12 for fp16

## Float16 support
%if 0%{?enable_fp16}
%define fp16_support -Denable-fp16=true
%else
%define fp16_support -Denable-fp16=false
%endif # enable_fp16

%if (0%{?enable_fp16}) && (0%{?enable_transformer})
%define transformer_support -Denable-transformer=true
%else
%define transformer_support -Denable-transformer=false
%endif

## GPU flag
## To enable OpenCL, pass the flag to gbs build with: --define "_with_gpu 1"
%bcond_with gpu

%if %{with gpu}
%define opencl_support -Denable-opencl=true
%else
%define opencl_support -Denable-opencl=false
%endif # gpu

Name:		nntrainer
Summary:	Software framework for training neural networks
Version:	0.6.0
Release:	0
Packager:	Jijoong Moon <jijoong.moon@samsung.com>
License:	Apache-2.0
Source0:	nntrainer-%{version}.tar.gz
Source1001:	nntrainer.manifest
%if %{with tizen}
Source1002:     capi-machine-learning-training.manifest
%endif
Source2001:	trainset.tar.gz
Source2002:	valset.tar.gz
Source2003:	testset.tar.gz
Source2004:	label.dat
Source2005:	unittest_layers.tar.gz

BuildRequires:	meson >= 0.55.0
BuildRequires:	openblas-devel
BuildRequires:	iniparser-devel >= 4.1
BuildRequires:	gtest-devel
BuildRequires:	python3
BuildRequires:	python3-numpy
BuildRequires:	flatbuffers-devel
BuildRequires:  cmake
%if %{with gpu}
BuildRequires:  opencl-headers
BuildRequires:  opencl-icd-loader
%endif # gpu

%if 0%{?unit_test}
BuildRequires:	ssat >= 1.1.0
%endif

# OpenAI interface
%define enable_gym -Duse_gym=false
%if 0%{?use_gym}
BuildRequires:	gym-http-api-devel
%define enable_gym -Duse_gym=true
%endif

%if 0%{?testcoverage}
# to be compatible with gcc-9, lcov should have a higher version than 1.14.1
BuildRequires: lcov
# BuildRequires:	taos-ci-unittest-coverage-assessment
%endif

%if 0%{?support_data_augmentation_opencv}
BuildRequires: opencv-devel
%endif

%if %{with tizen}
BuildRequires:	pkgconfig(capi-system-info)
BuildRequires:	pkgconfig(capi-base-common)
BuildRequires:	pkgconfig(dlog)

%if 0%{?support_nnstreamer_backbone}
BuildRequires: nnstreamer-tensorflow2-lite

Requires:	nnstreamer-tensorflow2-lite
%endif # support_nnstreamer_backbone

%if 0%{?support_tflite_backbone}
BuildRequires: tensorflow2-lite-devel
%endif # support_tflite_backbone

%if 0%{?support_tflite_interpreter}
BuildRequires: tensorflow2-lite-devel
%endif # support_tflite_interpreter

%define enable_nnstreamer_tensor_filter -Denable-nnstreamer-tensor-filter=disabled
%define enable_nnstreamer_tensor_trainer -Denable-nnstreamer-tensor-trainer=disabled

%if  0%{?nnstreamer_filter}
Requires:	nnstreamer-nntrainer = %{version}-%{release}
BuildRequires:	nnstreamer-devel
%define enable_nnstreamer_tensor_filter -Denable-nnstreamer-tensor-filter=enabled

%if 0%{?unit_test}
%if 0%{tizen_version_major}%{tizen_version_minor} > 60
BuildRequires:	nnstreamer-test-devel
%endif
BuildRequires:	gst-plugins-good-extra
BuildRequires:	python
%endif # unit_test
%endif # nnstreamer_filter

%if  0%{?nnstreamer_trainer}
Requires:	nnstreamer-nntrainer = %{version}-%{release}
BuildRequires:	nnstreamer-devel
%define enable_nnstreamer_tensor_trainer -Denable-nnstreamer-tensor-trainer=enabled
%endif # nnstreamer_trainer
%endif # tizen

Requires:	nntrainer-core = %{version}-%{release}

%description
NNtrainer Meta package for tizen

%package core
Summary:	Software framework for training neural networks
Requires:	iniparser >= 4.1
Requires:	libopenblas_pthreads0

%description core
NNtrainer is Software Framework for Training Neural Network Models on Devices.

%package devel
Summary:	Development package for custom nntrainer developers
Requires:	nntrainer = %{version}-%{release}
Requires:	openblas-devel

%description devel
Development package for custom nntrainer developers.
This contains corresponding header files and .pc pkgconfig file.

%package devel-static
Summary:        Static library for nntrainer-devel package
Requires:       nntrainer-devel = %{version}-%{release}
%description devel-static
Static library package of nntrainer-devel

%package applications
Summary:	NNTrainer Examples
Requires:	nntrainer = %{version}-%{release}
%if %{with tizen}
Requires:	capi-machine-learning-training = %{version}-%{release}
%endif #tizen
Requires:	%{capi_machine_learning_inference}
Requires:	nnstreamer-tensorflow2-lite
%if 0%{tizen_version_major}%{tizen_version_minor} > 60
BuildRequires:	nnstreamer-test-devel
%endif
BuildRequires:	nnstreamer-tensorflow2-lite
BuildRequires:	tensorflow2-lite-devel
BuildRequires:	pkgconfig(jsoncpp)
BuildRequires:	pkgconfig(libcurl)
BuildRequires:	pkgconfig(dlog)
BuildRequires:	%{capi_machine_learning_inference}-devel
BuildRequires:	glib2-devel
BuildRequires:	gstreamer-devel

%description applications
NNTrainer Examples for test purpose.

# gcov
%if 0%{?gcov:1}
%package gcov
Summary:         NNTrainer gcov
Group:           Machine Learning/ML Framework

%description gcov
NNTrainer gcov object
%endif

%if 0%{?testcoverage}
%package unittest-coverage
Summary:	NNTrainer UnitTest Coverage Analysis Result

%description unittest-coverage
HTML pages of lcov results of NNTrainer generated during rpmbuild
%endif

%if %{with tizen}
%package -n capi-machine-learning-training
Summary:         Tizen Native API for NNTrainer
Group:           Machine Learning/ML Framework
Requires:        %{name} = %{version}-%{release}
Requires:        %{capi_machine_learning_common}
%description -n capi-machine-learning-training
Tizen Native API wrapper for NNTrainer.
You can train neural networks efficiently.

%post -n capi-machine-learning-training -p /sbin/ldconfig
%postun -n capi-machine-learning-training -p /sbin/ldconfig

%package -n capi-machine-learning-training-devel
Summary:         Tizen Native API Devel Kit for NNTrainer
Group:           Machine Learning/ML Framework
Requires:        capi-machine-learning-training = %{version}-%{release}
Requires:        %{capi_machine_learning_common}-devel
%description -n capi-machine-learning-training-devel
Developmental kit for Tizen Native NNTrainer API.

%package -n capi-machine-learning-training-devel-static
Summary:         Static library for Tizen Native API
Group:           Machine Learning/ML Framework
Requires:        capi-machine-learning-training-devel = %{version}-%{release}
%description -n capi-machine-learning-training-devel-static
Static library of capi-machine-learning-training-devel package.

%package tizen-internal-devel
Summary:	Tizen internal headers for Tizen Machine Learning Training API
Group:		Machine Learning/ML Framework
Requires:	capi-machine-learning-training-devel = %{version}-%{release}
%description tizen-internal-devel
Tizen internal headers for Tizen Machine Learning Training API.

%if 0%{?support_ccapi}
%package -n ccapi-machine-learning-training
Summary:         Tizen Native API for NNTrainer
Group:           Machine Learning/ML Framework
Requires:        %{name} = %{version}-%{release}
%description -n ccapi-machine-learning-training
Tizen Native API wrapper for NNTrainer.
You can train neural networks efficiently.

%post -n ccapi-machine-learning-training -p /sbin/ldconfig
%postun -n ccapi-machine-learning-training -p /sbin/ldconfig

%package -n ccapi-machine-learning-training-devel
Summary:         Tizen Native API Devel Kit for NNTrainer
Group:           Machine Learning/ML Framework
Requires:        ccapi-machine-learning-training = %{version}-%{release}
%description -n ccapi-machine-learning-training-devel
Developmental kit for Tizen Native NNTrainer API.

%package -n ccapi-machine-learning-training-devel-static
Summary:         Static library for Tizen c++ API
Group:           Machine Learning/ML Framework
Requires:        ccapi-machine-learning-training-devel = %{version}-%{release}
%description -n ccapi-machine-learning-training-devel-static
Static library of ccapi-machine-learning-training-devel package.
%endif

%if 0%{?nnstreamer_filter}
%package -n nnstreamer-nntrainer
Summary: NNStreamer NNTrainer support
Requires: %{name} = %{version}-%{release}
Requires:	nnstreamer
%description -n nnstreamer-nntrainer
NNSteamer tensor filter for nntrainer to support inference.

%package -n nnstreamer-nntrainer-devel-static
Summary: NNStreamer NNTrainer support
Requires: nntrainer-devel-static = %{version}-%{release}
Requires:	nnstreamer-nntrainer = %{version}-%{release}
%description -n nnstreamer-nntrainer-devel-static
NNSteamer tensor filter static package for nntrainer to support inference.
%endif #nnstreamer_filter

%if 0%{?nnstreamer_trainer}
%package -n nnstreamer-nntrainer-trainer
Summary: NNStreamer NNTrainer support
Requires: %{name} = %{version}-%{release}
Requires:	nnstreamer
%description -n nnstreamer-nntrainer-trainer
NNSteamer tensor trainer for nntrainer to support inference.

%package -n nnstreamer-nntrainer-trainer-devel-static
Summary: NNStreamer NNTrainer support
Requires: nntrainer-devel-static = %{version}-%{release}
Requires:	nnstreamer-nntrainer-trainer = %{version}-%{release}
%description -n nnstreamer-nntrainer-trainer-devel-static
NNSteamer tensor trainer static package for nntrainer to support inference.
%endif #nnstreamer_trainer

%if 0%{?use_ruy}
%package -n ruy
Summary: Ruy support in NNTrainer
%description -n ruy
%endif

%if %{with gpu}
%package -n clblast
Summary: CLBlast as an OpenCL backend for BLAS operations in NNTrainer
%description -n clblast
%endif # clblast

%endif #tizen

## Define build options
%define platform -Dplatform=tizen
%define enable_tizen_feature_check -Denable-tizen-feature-check=true -Dtizen-version-major=0%{?tizen_version_major} -Dtizen-version-minor=0%{?tizen_version_minor}
%define install_app -Dinstall-app=true
%define enable_ccapi -Denable-ccapi=false
%define enable_nnstreamer_backbone -Denable-nnstreamer-backbone=false
%define enable_tflite_backbone -Denable-tflite-backbone=false
%define enable_tflite_interpreter -Denable-tflite-interpreter=false
%define enable_profile -Denable-profile=false
%define capi_ml_pkg_dep_resolution -Dcapi-ml-inference-actual=%{?capi_ml_inference_pkg_name} -Dcapi-ml-common-actual=%{?capi_ml_common_pkg_name}
%define enable_reduce_tolerance -Dreduce-tolerance=true
%define enable_debug -Denable-debug=false

# enable full tolerance on the CI
%if 0%{?unit_test}
%define enable_reduce_tolerance -Dreduce-tolerance=false
%endif

# enable debug on the CI for build
%if 0%{?unit_test}
%define enable_debug -Denable-debug=true
%endif

%if %{with tizen}
%define platform -Dplatform=tizen

%if 0%{?support_ccapi}
%define enable_ccapi -Denable-ccapi=true
%endif # support_ccapi
%endif # tizen

# Using cblas for Matrix calculation
%if 0%{?use_cblas}
%define enable_cblas -Denable-blas=true
%endif


%if 0%{?use_ruy}
%define enable_ruy -Denable-ruy=true
%else
%define enable_ruy -Denable-ruy=false
%endif

%if 0%{?use_biqgemm}
%define enable_biqgemm -Denable-biqgemm=true
%else
%define enable_biqgemm -Denable-biqgemm=false
%endif

%if 0%{?support_nnstreamer_backbone}
%define enable_nnstreamer_backbone -Denable-nnstreamer-backbone=true
%endif

%if 0%{?support_tflite_backbone}
%define enable_tflite_backbone -Denable-tflite-backbone=true
%endif

%if 0%{?unit_test}
%define enable_profile -Denable-profile=true
%endif

%if 0%{?support_tflite_interpreter}
%define enable_tflite_interpreter -Denable-tflite-interpreter=true
%endif

%prep
%setup -q
cp %{SOURCE1001} .
cp %{SOURCE2001} .
cp %{SOURCE2002} .
cp %{SOURCE2003} .
cp %{SOURCE2004} .
cp %{SOURCE2005} .

%if %{with tizen}
cp %{SOURCE1002} .
%endif

%build
CXXFLAGS=`echo $CXXFLAGS | sed -e "s|-std=gnu++11||"`

%if 0%{?testcoverage} || 0%{?gcov:1}
export CFLAGS+=" -fprofile-arcs -ftest-coverage"
export CXXFLAGS+=" -fprofile-arcs -ftest-coverage"
%endif

%if 0%{?enable_fp16}
export CFLAGS+=" -march=armv8.2-a+fp16+dotprod+i8mm"
export CXXFLAGS+=" -march=armv8.2-a+fp16+dotprod+i8mm"
%endif

# Add backward competibility for tizen < 6
%if 0%{tizen_version_major} < 6
ln -sf %{_includedir}/nnstreamer/nnstreamer.h %{_includedir}/nnstreamer/ml-api-common.h
ln -sf %{_libdir}/pkgconfig/capi-nnstreamer.pc %{_libdir}/pkgconfig/capi-ml-common.pc
%endif

# Setup Ruy
%if 0%{?use_ruy}
tar -xf packaging/ruy.tar.gz -C subprojects
%endif

# Setup CLBlast
%if %{with gpu}
tar -xf packaging/clblast.tar.gz -C subprojects
%endif # clblast

mkdir -p build
meson --buildtype=plain --prefix=%{_prefix} --sysconfdir=%{_sysconfdir} \
      --libdir=%{_lib} --bindir=%{nntrainerapplicationdir} \
      --includedir=include %{install_app} %{platform} \
      %{enable_tizen_feature_check} %{enable_cblas} %{enable_ccapi} \
      %{enable_gym} %{enable_nnstreamer_tensor_filter} %{enable_nnstreamer_tensor_trainer} \
      %{enable_profile} %{enable_nnstreamer_backbone} %{enable_tflite_backbone} \
      %{enable_tflite_interpreter} %{capi_ml_pkg_dep_resolution} \
      %{enable_reduce_tolerance} %{configure_subplugin_install_path} %{enable_debug} \
      -Dml-api-support=enabled -Denable-nnstreamer-tensor-filter=enabled \
      -Denable-nnstreamer-tensor-trainer=enabled -Denable-capi=enabled \
      %{fp16_support} %{opencl_support} build --wrap-mode=nodownload \
      %{transformer_support}

ninja -C build %{?_smp_mflags}

%if 0%{?unit_test}

%if 0%{?testcoverage}
lcov -i -c -o unittest_base.info -d build -b build --ignore-errors mismatch,source,gcov
%endif

export NNSTREAMER_CONF=$(pwd)/test/nnstreamer/nnstreamer-test.ini
export NNSTREAMER_FILTERS=$(pwd)/build/nnstreamer/tensor_filter
export NNSTREAMER_TRAINERS=$(pwd)/build/nnstreamer/tensor_trainer
meson test -C build -t 2.0 --print-errorlogs

# unittest for nntrainer plugin for nnstreamer
# todo: migrate this to meson test soon
%if 0%{?nnstreamer_filter}
pushd test/nnstreamer
ssat
popd
%endif #nnstreamer_filter
%if 0%{?nnstreamer_trainer}
pushd test/nnstreamer
ssat
popd
%endif #nnstreamer_trainer
%endif #unit_test

%if 0%{?gcov:1}
mkdir -p gcov-obj
find . -name '*.gcno' -exec cp '{}' gcov-obj ';'
%endif

%install
DESTDIR=%{buildroot} ninja -C build %{?_smp_mflags} install

%if 0%{?gcov:1}
mkdir -p %{buildroot}%{_datadir}/gcov/obj/%{name}
install -m 0644 gcov-obj/* %{buildroot}%{_datadir}/gcov/obj/%{name}
%endif

%if 0%{?testcoverage}
##
# The included directories are:
#
# api: nnstreamer api
# gst: the nnstreamer elements
# nnstreamer_example: custom plugin examples
# common: common libraries for gst (elements)
# include: common library headers and headers for external code (packaged as "devel")
#
# Intentionally excluded directories are:
#
# tests: We are not going to show testcoverage of the test code itself or example applications

%if %{with tizen}
%define testtarget $(pwd)/api/capi
%else
%define testtarget
%endif

# 'lcov' generates the date format with UTC time zone by default. Let's replace UTC with KST.
# If you ccan get a root privilege, run ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
TZ='Asia/Seoul'; export TZ
$(pwd)/test/unittestcoverage.py module $(pwd)/nntrainer %testtarget

# Get commit info
VCS=`cat ${RPM_SOURCE_DIR}/nntrainer.spec | grep "^VCS:" | sed "s|VCS:\\W*\\(.*\\)|\\1|"`

# Create human readable unit test coverage report web page.
# Create null gcda files if gcov didn't create it because there is completely no unit test for them.
find . -name "*.gcno" -exec sh -c 'touch -a "${1%.gcno}.gcda"' _ {} \;
# Remove gcda for meaningless file (CMake's autogenerated)
find . -name "CMakeCCompilerId*.gcda" -delete
find . -name "CMakeCXXCompilerId*.gcda" -delete
#find . -path "/build/*.j

# Generate report
lcov -t 'NNTrainer Unit Test Coverage' -c -o unittest_nntrainer.info -d build/nntrainer -b build --ignore-errors mismatch,source,empty,gcov
lcov -c -o unittest_capi.info -d build/api/capi -b build --ignore-errors mismatch,source,empty,gcov
lcov -a unittest_base.info -a unittest_nntrainer.info -a unittest_capi.info -o unittest_total.info --ignore-errors mismatch,empty,gcov
lcov -r unittest_total.info "*/Applications/*" "*/nnstreamer/*" "*/meson*/*" "*/build/*" "*/test/*" "*/usr/*" "*/subprojects/*" -o unittest-filtered.info --ignore-errors unused,gcov
# nnstreamer layer is untestable here

# Visualize the report
genhtml -o result unittest-filtered.info -t "nntrainer %{version}-%{release} ${VCS}" --ignore-errors mismatch,source -p ${RPM_BUILD_DIR} --synthesize-missing

mkdir -p %{buildroot}%{_datadir}/nntrainer/unittest/
cp -r result %{buildroot}%{_datadir}/nntrainer/unittest/
%endif  # test coverage

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files

%files core
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/libnntrainer.so
%config %{_sysconfdir}/nntrainer.ini

%files devel
# node exporter and its dependencies
%{_includedir}/nntrainer/nntrainer_error.h
%{_includedir}/nntrainer/connection.h
%{_includedir}/nntrainer/common_properties.h
%{_includedir}/nntrainer/base_properties.h
%{_includedir}/nntrainer/node_exporter.h
%{_includedir}/nntrainer/nntr_threads.h
%{_includedir}/nntrainer/profiler.h
# tensor headers
%{_includedir}/nntrainer/memory_data.h
%{_includedir}/nntrainer/tensor.h
%{_includedir}/nntrainer/tensor_base.h
%{_includedir}/nntrainer/q4_k_tensor.h
%{_includedir}/nntrainer/q6_k_tensor.h
%{_includedir}/nntrainer/q4_0_tensor.h
%{_includedir}/nntrainer/q4_0_utils.h
%{_includedir}/nntrainer/int4_tensor.h
%{_includedir}/nntrainer/int4_utils.h
%{_includedir}/nntrainer/uint4_tensor.h
%{_includedir}/nntrainer/char_tensor.h
%{_includedir}/nntrainer/short_tensor.h
%{_includedir}/nntrainer/uint_tensor.h
%{_includedir}/nntrainer/float_tensor.h
%if 0%{?use_biqgemm}
%{_includedir}/nntrainer/bcq_tensor.h
%endif
%if 0%{?enable_fp16}
%{_includedir}/nntrainer/half_tensor.h
%endif
%{_includedir}/nntrainer/tensor_wrap_specs.h
%{_includedir}/nntrainer/cpu_backend.h
%{_includedir}/nntrainer/fallback_internal.h
%{_includedir}/nntrainer/fallback_kleidiai.h
%if 0%{?use_cblas}
%{_includedir}/nntrainer/cblas_interface.h
%endif
%{_includedir}/nntrainer/ggml_interface.h
%{_includedir}/nntrainer/nntr_ggml_impl.h
%{_includedir}/nntrainer/nntr_ggml_impl_common.h
%{_includedir}/nntrainer/nntr_ggml_impl_utils.h
%{_includedir}/nntrainer/bs_thread_pool.h
%{_includedir}/nntrainer/bs_thread_pool_manager.hpp
%ifarch %{ix86} x86_64
%{_includedir}/nntrainer/x86_compute_backend.h
%{_includedir}/nntrainer/avx2_impl.h
%endif
%ifarch aarch64
%{_includedir}/nntrainer/arm_compute_backend.h
%{_includedir}/nntrainer/neon_impl.h
%{_includedir}/nntrainer/neon_setting.h
%{_includedir}/nntrainer/neon_mathfun.h
%{_includedir}/nntrainer/neon_mathfun.hxx
%{_includedir}/nntrainer/matrix_transpose_neon.h
%if 0%{?enable_fp16}
%{_includedir}/nntrainer/hgemm.h
%{_includedir}/nntrainer/hgemm_common.h
%{_includedir}/nntrainer/hgemm_kernel.h
%{_includedir}/nntrainer/hgemm_util.h
%{_includedir}/nntrainer/hgemm_noTrans.h
%{_includedir}/nntrainer/hgemm_transA.h
%{_includedir}/nntrainer/hgemm_transAB.h
%{_includedir}/nntrainer/hgemm_transB.h
%{_includedir}/nntrainer/hgemm_pack.h
%{_includedir}/nntrainer/hgemm_padding.h
%{_includedir}/nntrainer/hgemm_padding_a.h
%{_includedir}/nntrainer/hgemm_padding_b.h
%{_includedir}/nntrainer/kai_common.h
%{_includedir}/nntrainer/mask_neon.h
%{_includedir}/nntrainer/matrix_transpose_kernels_neon.h
%{_includedir}/nntrainer/neon_kleidiai.h
%{_includedir}/nntrainer/kai_lhs_quant_pack_qai8dxp_f32.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h
%{_includedir}/nntrainer/kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h
%{_includedir}/nntrainer/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h
%{_includedir}/nntrainer/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h
%endif
%endif
%ifarch %arm
    %{_includedir}/nntrainer/arm_compute_backend.h
    %{_includedir}/nntrainer/armv7_neon.h
    %{_includedir}/nntrainer/neon_impl.h
    %{_includedir}/nntrainer/neon_setting.h
    %{_includedir}/nntrainer/neon_mathfun.h
    %{_includedir}/nntrainer/neon_mathfun.hxx
    %{_includedir}/nntrainer/matrix_transpose_neon.h
%endif
%ifnarch %{ix86} %arm x86_64 aarch64
%{_includedir}/nntrainer/fallback.h
%endif
%{_includedir}/nntrainer/var_grad.h
%{_includedir}/nntrainer/weight.h
%{_includedir}/nntrainer/quantizer.h
# @todo: update dataset headers
%{_includedir}/nntrainer/databuffer.h
%{_includedir}/nntrainer/databuffer_factory.h
# layer headers
%{_includedir}/nntrainer/layer_context.h
%{_includedir}/nntrainer/layer_devel.h
%{_includedir}/nntrainer/layer_impl.h
%{_includedir}/nntrainer/operation_layer.h
# custom layer kits
%{_includedir}/nntrainer/app_context.h
%{_includedir}/nntrainer/context.h
%{_includedir}/nntrainer/engine.h
# optimizer headers
%{_includedir}/nntrainer/optimizer_context.h
%{_includedir}/nntrainer/optimizer_devel.h
%{_includedir}/nntrainer/lr_scheduler.h
# for logging
%{_includedir}/nntrainer/nntrainer_log.h
%{_includedir}/nntrainer/nntrainer_logger.h
%{_libdir}/pkgconfig/nntrainer.pc
# update this to enable external applications
# @todo filter out headers that should be hidden, and classifiy in the appropriate place if not
%{_includedir}/nntrainer/util_func.h
%{_includedir}/nntrainer/noncopyable.h
%{_includedir}/nntrainer/nonmovable.h
%{_includedir}/nntrainer/singleton.h
%{_includedir}/nntrainer/fp16.h
%{_includedir}/nntrainer/util_simd.h
%{_includedir}/nntrainer/dynamic_library_loader.h
%{_includedir}/nntrainer/mman_windows.h
%{_includedir}/nntrainer/loss_layer.h
%{_includedir}/nntrainer/acti_func.h
# model headers
%{_includedir}/nntrainer/neuralnet.h
## neuralnet.h
%{_includedir}/nntrainer/compiler_fwd.h 
%{_includedir}/nntrainer/dynamic_training_optimization.h
%{_includedir}/nntrainer/layer_node.h
%{_includedir}/nntrainer/graph_node.h
%{_includedir}/nntrainer/model_common_properties.h
%{_includedir}/nntrainer/network_graph.h
%{_includedir}/nntrainer/graph_core.h
%{_includedir}/nntrainer/manager.h
%{_includedir}/nntrainer/basic_planner.h
%{_includedir}/nntrainer/memory_planner.h
%{_includedir}/nntrainer/tensor_pool.h
%{_includedir}/nntrainer/cache_loader.h
%{_includedir}/nntrainer/task.h
%{_includedir}/nntrainer/task_executor.h
%{_includedir}/nntrainer/cache_pool.h
%{_includedir}/nntrainer/cache_elem.h
%{_includedir}/nntrainer/memory_pool.h
%{_includedir}/nntrainer/swap_device.h
%{_includedir}/nntrainer/optimizer_wrapped.h
%{_includedir}/nntrainer/mem_allocator.h
%{_includedir}/nntrainer/tensor_layer.h
%{_includedir}/nntrainer/weight_layer.h
%if %{with gpu}
%{_includedir}/nntrainer/CL/cl.h
%{_includedir}/nntrainer/CL/cl_platform.h
%{_includedir}/nntrainer/CL/opencl.h
%{_includedir}/nntrainer/attention_kernel_interface.h
%{_includedir}/nntrainer/attention_kernel_strings.h
%{_includedir}/nntrainer/blas_kernel_interface.h
%{_includedir}/nntrainer/blas_kernel_strings.h
%{_includedir}/nntrainer/cl_buffer_manager.h
%{_includedir}/nntrainer/cl_context.h
%{_includedir}/nntrainer/cl_platform.h
%{_includedir}/nntrainer/opencl_command_queue_manager.h
%{_includedir}/nntrainer/opencl_context_manager.h
%{_includedir}/nntrainer/opencl_kernel.h
%{_includedir}/nntrainer/opencl_program.h
%endif # gpu
%files devel-static
%{_libdir}/libnntrainer*.a
%exclude %{_libdir}/libcapi*.a

%if %{with tizen}
%files -n capi-machine-learning-training
%manifest capi-machine-learning-training.manifest
%license LICENSE
%{_libdir}/libcapi-nntrainer.so

%files -n capi-machine-learning-training-devel
%{_includedir}/nntrainer/nntrainer.h
%{_includedir}/nntrainer/nntrainer-api-common.h
%{_libdir}/pkgconfig/capi-ml-training.pc

%files -n capi-machine-learning-training-devel-static
%{_libdir}/libcapi-nntrainer.a

%files tizen-internal-devel
%{_includedir}/nntrainer/nntrainer-tizen-internal.h

%if 0%{?support_ccapi}
%files -n ccapi-machine-learning-training
%manifest capi-machine-learning-training.manifest
%license LICENSE
%{_libdir}/libccapi-nntrainer.so

%files -n ccapi-machine-learning-training-devel
%{_includedir}/nntrainer/common.h
%{_includedir}/nntrainer/model.h
%{_includedir}/nntrainer/layer.h
%{_includedir}/nntrainer/tensor_api.h
%{_includedir}/nntrainer/optimizer.h
%{_includedir}/nntrainer/dataset.h
%{_includedir}/nntrainer/tensor_dim.h
%{_libdir}/pkgconfig/ccapi-ml-training.pc

%files -n ccapi-machine-learning-training-devel-static
%{_libdir}/libccapi-nntrainer.a
%endif # support_ccapi

%if 0%{?nnstreamer_filter}
%files -n nnstreamer-nntrainer
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_prefix}/%{nnstreamer_subplugin_path}/filters/libnnstreamer_filter_nntrainer.so

%files -n nnstreamer-nntrainer-devel-static
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/libnnstreamer_filter_nntrainer.a
%endif #nnstreamer_filter

%if 0%{?nnstreamer_trainer}
%files -n nnstreamer-nntrainer-trainer
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_prefix}/%{nnstreamer_subplugin_path}/trainers/libnnstreamer_trainer_nntrainer.so

%files -n nnstreamer-nntrainer-trainer-devel-static
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/libnnstreamer_trainer_nntrainer.a
%endif #nnstreamer_trainer

# Ruy
%if 0%{?use_ruy}
%files -n ruy
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/libruy*.a
%{_libdir}/libclog.a
%{_libdir}/libcpuinfo.a
%{_bindir}/cache_info
%{_bindir}/cpu_info
%{_bindir}/isa_info
%ifarch x86_64
%{_bindir}/cpuid_dump
%endif #x86_64
%endif

# CLBlast
%if %{with gpu}
%files -n clblast
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/libclblast.so
%{_bindir}/clblast_tuner_copy_fast
%{_bindir}/clblast_tuner_copy_pad
%{_bindir}/clblast_tuner_invert
%{_bindir}/clblast_tuner_routine_xgemm
%{_bindir}/clblast_tuner_routine_xtrsv
%{_bindir}/clblast_tuner_transpose_fast
%{_bindir}/clblast_tuner_transpose_pad
%{_bindir}/clblast_tuner_xaxpy
%{_bindir}/clblast_tuner_xconvgemm
%{_bindir}/clblast_tuner_xdot
%{_bindir}/clblast_tuner_xgemm
%{_bindir}/clblast_tuner_xgemm_direct
%{_bindir}/clblast_tuner_xgemv
%{_bindir}/clblast_tuner_xger
%endif

%endif #tizen

%files applications
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/nntrainer/bin/applications/*
%{_libdir}/nntrainer/layers/*layer.so

%if 0%{?gcov:1}
%files gcov
%{_datadir}/gcov/obj/*
%endif

%if 0%{?testcoverage}
%files unittest-coverage
%{_datadir}/nntrainer/unittest/*
%endif

%changelog
* Thu Jan 23 2025 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 0.6.0
* Tue Apr 04 2023 Jijoong Moon <jijoong.moon@samsung.com>
- Release of 0.5.0
* Mon Sep 26 2022 Jijoong Moon <jijoong.moon@samsung.com>
- Release of 0.4.0
* Fri Sep 24 2021 Jijoong Moon <jijoong.moon@samsung.com>
- Release of 0.3.0
* Wed Jun 02 2021 Jijoong Moon <jijoong.moon@samsung.com>
- Release of 0.2.0
* Wed Sep 23 2020 Jijoong Moon <jijoong.moon@samsung.com>
- Release of 0.1.1
* Mon Aug 10 2020 Jijoong Moon <jijoong.moon@samsung.com>
- Release of 0.1.0.rc1
* Wed Mar 18 2020 Jijoong Moon <jijoong.moon@samsung.com>
- packaged nntrainer
