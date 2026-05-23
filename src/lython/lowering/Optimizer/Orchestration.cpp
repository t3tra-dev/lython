#include "Optimizer/Utils.h"

namespace py::optimizer::pipeline {

void preLowering(mlir::ModuleOp module) {
  containerPre(module);
  callPre(module);
  classLayoutPre(module);
  scalarPre(module);
  ::py::optimizer::refcount::sinkClassDecrefs(module);
  ::py::optimizer::refcount::markFinalLocal(module);
  zeroCostProofPre(module);
  zeroCostRewritePre(module);
}

void postLowering(mlir::ModuleOp module) {
  scalarPost(module);
  refcountPost(module);
  ::py::optimizer::scalar::dce(module);
}

} // namespace py::optimizer::pipeline
