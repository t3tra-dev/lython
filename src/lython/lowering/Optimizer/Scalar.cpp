#include "Optimizer/Utils.h"

using namespace mlir;

namespace py::optimizer {

void runScalarPreLoweringOptimizations(ModuleOp module) {
  removeUnusedNoneOps(module);
  removeNoneDecrefs(module);
  hoistIntConstants(module);
  removeSmallIntDecrefs(module);
}

void runScalarPostLoweringOptimizations(ModuleOp module) {
  cseStringCreation(module);
  cseSingletonGetters(module);
  eliminateBoolBoxingUnboxing(module);
  eliminateLongArithmeticRoundTrips(module);
  eliminateLongBoxingUnboxing(module);
  cseSmallIntFromI64(module);
  cseConstants(module);
}

} // namespace py::optimizer
