#include "Common/RuntimeSupport.h"
#include "ThreadSafety/Verifier.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace py {
namespace {

struct LLVMThreadSafetyVerifierPass
    : public mlir::PassWrapper<LLVMThreadSafetyVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMThreadSafetyVerifierPass)

  llvm::StringRef getArgument() const override {
    return "py-llvm-thread-safety-verify";
  }

  llvm::StringRef getDescription() const override {
    return "Verify Lython no-GIL atomic ordering and shared retain contracts";
  }

  void runOnOperation() override {
    if (mlir::failed(threadsafe::verifier::module::verify(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafetyVerifierPass() {
  return std::make_unique<LLVMThreadSafetyVerifierPass>();
}

} // namespace py
