qnn_dirs = [
  'HTA',  
  'HTP',
  'LPAI',
  'DSP',
  'HTPQEMU',
  'GPU',
  'CPU',
  'Saver',
  'System',
]

foreach e : qnn_dirs
  subdir(e)
endforeach

qnn_dir_headers=[
 'QnnBackend.h',
 'QnnDevice.h',
 'QnnInterface.h',
 'QnnOpDef.h',
 'QnnProperty.h',
 'QnnTensor.h',
 'QnnCommon.h',
 'QnnError.h',
 'QnnLog.h',
 'QnnOpPackage.h',
 'QnnSdkBuildId.h',
 'QnnTypes.h',
 'QnnContext.h',
 'QnnGraph.h',
 'QnnMem.h',
 'QnnProfile.h',
 'QnnSignal.h',
]

qnn_include_dir = nntrainer_prefix / 'include' / 'nntrainer' / 'npu' / 'qnn'

install_subdir ('QNN', install_dir : qnn_include_dir )

install_headers(qnn_dir_headers, install_dir : qnn_include_dir / 'QNN')