#include "Optimizer/Utils.h"

namespace py::optimizer::pipeline {

void preLowering(mlir::ModuleOp module) {
  ::py::optimizer::scalar::foldStaticBuiltinPrintRepr(module);
  containerPre(module);
  callPre(module);
  classLayoutPre(module);
  scalarPre(module);
  ::py::optimizer::refcount::sinkClassDecrefs(module);
  ::py::optimizer::refcount::markFinalLocal(module);
  zeroCostProofPre(module);
  zeroCostRewritePre(module);
}

void postValueLowering(mlir::ModuleOp module) {
  scalarPost(module);
  refcountPost(module);
  ::py::optimizer::scalar::dce(module);
}

void finalLLVMCleanup(mlir::ModuleOp module) {
  ::py::optimizer::scalar::cseConstants(module);
  ::py::optimizer::scalar::dce(module);
  ::py::optimizer::runtime::Func::eraseUnusedDecls(module);
}

} // namespace py::optimizer::pipeline
