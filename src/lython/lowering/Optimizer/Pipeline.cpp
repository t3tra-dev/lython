#include "Common/RuntimeSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace py {
namespace {

struct PublicationPreparationPass
    : public mlir::PassWrapper<PublicationPreparationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PublicationPreparationPass)

  llvm::StringRef getArgument() const override {
    return "py-prepare-publication";
  }

  llvm::StringRef getDescription() const override {
    return "Insert explicit py.publish boundaries before refcount insertion";
  }

  void runOnOperation() override {
    optimizer::publication::prepare(getOperation());
  }
};

struct PyOptimizationPass
    : public mlir::PassWrapper<PyOptimizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PyOptimizationPass)

  llvm::StringRef getArgument() const override { return "py-optimize"; }

  llvm::StringRef getDescription() const override {
    return "Apply Py dialect-specific optimizations";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    optimizer::pipeline::preLowering(module);
    optimizer::pipeline::postLowering(module);
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPyOptimizationPass() {
  return std::make_unique<PyOptimizationPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass() {
  return std::make_unique<PublicationPreparationPass>();
}

} // namespace py
