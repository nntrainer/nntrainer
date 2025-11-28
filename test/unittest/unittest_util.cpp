#include "unittest_util.h"
#include <cl_context.h>
#include <engine.h>

namespace nntrainer {

void *allocateSVM(size_t size_bytes) {
  auto *blas_cc = static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  void *ptr = blas_cc->context_inst_.createSVMRegion(size_bytes);
  if (!ptr) {
    throw std::runtime_error("Failed to allocate SVM for unit test.");
  }
  return ptr;
}

void freeSVM(void *ptr) {
  auto *blas_cc = static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  blas_cc->context_inst_.releaseSVMRegion(ptr);
}

} // namespace nntrainer
