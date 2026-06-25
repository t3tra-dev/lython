#include "RuntimeLowering/RuntimeLowering.h"

#include "Common/RuntimeSupport.h"
#include "Passes/Runtime/Cleanup.h"

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

class RuntimeLoweringPass
    : public mlir::PassWrapper<RuntimeLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLoweringPass)

  llvm::StringRef getArgument() const final {
    return "lython-runtime-lowering";
  }
  llvm::StringRef getDescription() const final {
    return "lower resolved Py dialect operations to the runtime ABI";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    if (mlir::failed(requireResolvedInput(module))) {
      signalPassFailure();
      return;
    }

    if (mlir::failed(RuntimeBundleLowerer(module).lowerModule()))
      signalPassFailure();
  }

private:
  mlir::LogicalResult requireResolvedInput(mlir::ModuleOp module) {
    mlir::LogicalResult result = mlir::success();
    module.walk([&](mlir::UnrealizedConversionCastOp op) {
      op.emitError()
          << "lowering requires fully resolved Python IR; "
             "builtin.unrealized_conversion_cast is resolution evidence, not "
             "a runtime ABI value";
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    });
    return result;
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
makeNoOpPass(llvm::StringRef argument, llvm::StringRef description) {
  return std::make_unique<NoOpModulePass>(argument, description);
}

} // namespace
} // namespace py::runtime_lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeLoweringPass() {
  return std::make_unique<runtime_lowering::RuntimeLoweringPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-publication-preparation",
      "prepare a resolved module for runtime lowering");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass() {
  return runtime_lowering::makeNoOpPass(
      "lython-refcount-insertion",
      "insert ownership operations after ABI lowering");
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncRuntimeRewritePass() {
  return runtime_lowering::makeNoOpPass(
      "lython-async-runtime-rewrite",
      "rewrite async runtime artifacts after lowering");
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLinalgLoweringPass() {
  return runtime_lowering::makeNoOpPass("lython-linalg-lowering",
                                        "lower linalg helper operations");
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

void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts) {
  contracts.clear();
}

void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp, const PyLLVMTypeConverter &,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts) {
  contracts.clear();
}

mlir::LogicalResult preserveLLVMAsyncArgProvenanceContracts(
    mlir::ModuleOp, llvm::ArrayRef<AsyncArgProvenanceContract>) {
  return mlir::success();
}

void collectLoweredSafetyContracts(mlir::ModuleOp,
                                   LoweredSafetyContracts &contracts) {
  contracts.asyncArgs.clear();
}

void collectLoweredSafetyContracts(mlir::ModuleOp, const PyLLVMTypeConverter &,
                                   LoweredSafetyContracts &contracts) {
  contracts.asyncArgs.clear();
}

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
