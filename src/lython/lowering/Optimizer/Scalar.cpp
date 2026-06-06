#include "Optimizer/Utils.h"

namespace py::optimizer {

void pipeline::scalarPre(mlir::ModuleOp module) {
  scalar::removeUnusedNone(module);
  scalar::dropNoneDecrefs(module);
  scalar::fuseStrConcat3(module);
  scalar::foldIntConstants(module);
  scalar::hoistInts(module);
  scalar::dce(module);
}

void pipeline::scalarPost(mlir::ModuleOp module) {
  scalar::cseConstants(module);
}

} // namespace py::optimizer
