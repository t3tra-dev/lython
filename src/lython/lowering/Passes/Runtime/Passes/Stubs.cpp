#include "Common/RuntimeSupport.h"
#include "Passes/Runtime/Cleanup/Transforms.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace py::runtime_lowering {
namespace {

class NoOpModulePass
    : public mlir::PassWrapper<NoOpModulePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  NoOpModulePass(llvm::StringRef argument, llvm::StringRef description)
      : argument(argument.str()), description(description.str()) {}

  NoOpModulePass(const NoOpModulePass &other)
      : mlir::PassWrapper<NoOpModulePass, mlir::OperationPass<mlir::ModuleOp>>(
            other),
        argument(other.argument), description(other.description) {}

  llvm::StringRef getArgument() const final { return argument; }
  llvm::StringRef getDescription() const final { return description; }
  void runOnOperation() final {}

private:
  std::string argument;
  std::string description;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
makeNoOpPass(llvm::StringRef argument, llvm::StringRef description) {
  return std::make_unique<NoOpModulePass>(argument, description);
}

} // namespace
} // namespace py::runtime_lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-publication-preparation",
      "prepare a resolved module for runtime lowering");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountPairElisionPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-refcount-elision", "elide proven redundant ownership pairs");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPyOptimizationPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-py-optimization", "optimize resolved Py dialect operations");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-ownership-verifier", "verify ownership after runtime lowering");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-llvm-call-ownership-verifier",
      "verify lowered LLVM call ownership metadata");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafetyVerifierPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-llvm-thread-safety-verifier",
      "verify lowered LLVM thread-safety metadata");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass() {
  return runtime_lowering::makeNoOpPass("lython-native-verification",
                                        "verify native function declarations");
}

namespace lowering::runtime::cleanup {

bool unreachableBlocks(mlir::ModuleOp) { return false; }
bool pyBridgeCasts(mlir::Operation *) { return false; }
bool pyMultiCasts(mlir::Operation *) { return false; }
bool voidPyReturns(mlir::Operation *) { return false; }
bool memrefDescriptorCasts(mlir::Operation *) { return false; }
bool memrefRuntimeCalls(mlir::Operation *) { return false; }
bool pointerRoundTrips(mlir::ModuleOp) { return false; }
bool llvmFuncReturns(mlir::Operation *) { return false; }
bool finalBoundary(mlir::ModuleOp) { return false; }

} // namespace lowering::runtime::cleanup

void collectLoweredSafetyContracts(mlir::ModuleOp, LoweredSafetyContracts &) {}

void collectLoweredSafetyContracts(mlir::ModuleOp, const PyLLVMTypeConverter &,
                                   LoweredSafetyContracts &) {}

mlir::LogicalResult
preserveLoweredSafetyContracts(mlir::ModuleOp, const LoweredSafetyContracts &) {
  return mlir::success();
}

mlir::LogicalResult verifyOwnership(mlir::ModuleOp) { return mlir::success(); }

mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp) {
  return mlir::success();
}

namespace optimizer::publication {
void prepare(mlir::ModuleOp) {}
} // namespace optimizer::publication

namespace optimizer::pipeline {
void preLowering(mlir::ModuleOp) {}
void postValueLowering(mlir::ModuleOp) {}
void finalLLVMCleanup(mlir::ModuleOp) {}
} // namespace optimizer::pipeline

} // namespace py
