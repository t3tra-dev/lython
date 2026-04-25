#include "Common/RuntimeSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace py {
namespace {

struct PublicationPreparationPass
    : public PassWrapper<PublicationPreparationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PublicationPreparationPass)

  StringRef getArgument() const override { return "py-prepare-publication"; }

  StringRef getDescription() const override {
    return "Insert explicit py.publish boundaries before refcount insertion";
  }

  void runOnOperation() override {
    runEarlyPublicationPreparation(getOperation());
  }
};

struct PyOptimizationPass
    : public PassWrapper<PyOptimizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PyOptimizationPass)

  StringRef getArgument() const override { return "py-optimize"; }

  StringRef getDescription() const override {
    return "Apply Py dialect-specific optimizations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    runPreLoweringOptimizations(module);
    runPostLoweringOptimizations(module);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPyOptimizationPass() {
  return std::make_unique<PyOptimizationPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createPublicationPreparationPass() {
  return std::make_unique<PublicationPreparationPass>();
}

} // namespace py
