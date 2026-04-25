#include "Optimizer/Utils.h"

using namespace mlir;

namespace py {

void runPreLoweringOptimizations(ModuleOp module) {
  optimizer::runContainerPreLoweringOptimizations(module);
  optimizer::runCallPreLoweringOptimizations(module);
  optimizer::runClassLayoutPreLoweringOptimizations(module);
  optimizer::runScalarPreLoweringOptimizations(module);
  optimizer::sinkClassDecrefsPastBorrowedAttrUses(module);
  optimizer::markFinalLocalClassDecrefs(module);
}

void runPostLoweringOptimizations(ModuleOp module) {
  optimizer::runScalarPostLoweringOptimizations(module);
  optimizer::runRefcountPostLoweringOptimizations(module);
  optimizer::eliminateDeadCode(module);
}

} // namespace py
