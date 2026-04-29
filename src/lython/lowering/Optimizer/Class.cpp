#include "Optimizer/Utils.h"

using namespace mlir;

namespace py {

void runEarlyPublicationPreparation(ModuleOp module) {
  optimizer::computeLocalPublicationSummaries(module);
  optimizer::insertPublishesAtPublicationBoundaries(module);
  optimizer::computeLocalPublicationSummaries(module);
}

} // namespace py

namespace py::optimizer {

void runClassLayoutPreLoweringOptimizations(ModuleOp module) {
  eliminateRedundantClassPublishes(module);
  markKnownLocalStaticClassAccesses(module);
  markConsumedAttrSetValues(module);
  markZeroInitializedSelfFirstStores(module);
}

void runCallPreLoweringOptimizations(ModuleOp module) {
  applyStaticMakeFunctionDefaults(module);
  cleanupRedundantClassIncrefsAfterDirectCalls(module);
  rewriteDirectFuncCallsToPreferredHelpers(module);
}

} // namespace py::optimizer
