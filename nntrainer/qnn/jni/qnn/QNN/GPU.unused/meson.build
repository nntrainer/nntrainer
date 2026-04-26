gpu_headers=[
'QnnGpuBackend.h',
'QnnGpuCommon.h',
'QnnGpuContext.h',
'QnnGpuGraph.h',
'QnnGpuOpPackage.h',
]

qnn_dir_include_dir = nntrainer_prefix / 'include' / 'nntrainer' / 'npu' / 'qnn' / 'QNN'

install_subdir ('GPU', install_dir : qnn_dir_include_dir )

install_headers(gpu_headers, install_dir : qnn_dir_include_dir / 'GPU')