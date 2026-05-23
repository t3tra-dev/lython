#include "Optimizer/Utils.h"

namespace py::optimizer {

void pipeline::scalarPre(mlir::ModuleOp module) {
  scalar::removeUnusedNone(module);
  scalar::dropNoneDecrefs(module);
  scalar::hoistInts(module);
  scalar::dropSmallIntDecrefs(module);
}

void pipeline::scalarPost(mlir::ModuleOp module) {
  scalar::cseSingletons(module);
  scalar::elideBoolBoxing(module);
  scalar::elideLongArithRoundTrips(module);
  scalar::elideLongBoxing(module);
  scalar::cseSmallInts(module);
  scalar::cseConstants(module);
}

} // namespace py::optimizer
