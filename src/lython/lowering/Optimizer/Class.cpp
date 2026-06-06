#include "Optimizer/Utils.h"

namespace py::optimizer::publication {

void prepare(mlir::ModuleOp module) {
  compute(module);
  insertBoundaries(module);
  compute(module);
}

} // namespace py::optimizer::publication

namespace py::optimizer {

void pipeline::classLayoutPre(mlir::ModuleOp module) {
  class_state::eliminatePublishes(module);
  consume::attrSetValues(module);
  class_state::markFirstStores(module);
}

void pipeline::callPre(mlir::ModuleOp module) {
  call::cleanupClassIncrefs(module);
  call::rewritePreferred(module);
}

} // namespace py::optimizer
