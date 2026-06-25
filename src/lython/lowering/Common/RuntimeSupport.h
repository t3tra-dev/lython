#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <string>

namespace llvm {
class Module;
}

namespace py {

class PyLLVMTypeConverter;

enum class AsyncArgProvenanceKind { RuntimeHandle, ExceptionCell };

struct AsyncArgProvenanceContract {
  std::string function;
  unsigned argumentIndex = 0;
  AsyncArgProvenanceKind kind = AsyncArgProvenanceKind::RuntimeHandle;
};

struct LoweredSafetyContracts {
  llvm::SmallVector<AsyncArgProvenanceContract, 8> asyncArgs;
};

struct PythonCallSiteRange {
  std::string caller;
  std::string callee;
  std::string filename;
  std::string functionName;
  std::int32_t line = 0;
  std::int32_t column = 0;
  std::int32_t endLine = 0;
  std::int32_t endColumn = 0;
};

void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts);
void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp module, const PyLLVMTypeConverter &typeConverter,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts);
mlir::LogicalResult preserveLLVMAsyncArgProvenanceContracts(
    mlir::ModuleOp module,
    llvm::ArrayRef<AsyncArgProvenanceContract> contracts);

void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   LoweredSafetyContracts &contracts);
void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   const PyLLVMTypeConverter &typeConverter,
                                   LoweredSafetyContracts &contracts);
mlir::LogicalResult
preserveLoweredSafetyContracts(mlir::ModuleOp module,
                               const LoweredSafetyContracts &contracts);

void collectPythonCallSiteRanges(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<PythonCallSiteRange> &callSites);
bool installPythonExceptionCleanupFrames(
    llvm::Module &module, llvm::ArrayRef<PythonCallSiteRange> callSites);

mlir::LogicalResult verifyOwnership(mlir::ModuleOp module);
mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp module);

namespace optimizer::publication {
void prepare(mlir::ModuleOp module);
}

namespace optimizer::pipeline {
void preLowering(mlir::ModuleOp module);
void postValueLowering(mlir::ModuleOp module);
void finalLLVMCleanup(mlir::ModuleOp module);
} // namespace optimizer::pipeline

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeLoweringPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountPairElisionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createPyOptimizationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafetyVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncRuntimeRewritePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLinalgLoweringPass();

} // namespace py
