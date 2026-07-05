#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <string>

namespace py {

class PyLLVMTypeConverter;

struct LoweredSafetyContract {
  std::string functionName;
  std::string operationName;
  std::string role;
  std::string ordering;
  std::string retainPremise;
  std::string retainPremiseSource;
  std::int64_t ordinal = 0;
  std::int64_t slot = -1;
};

struct LoweredSafetyContracts {
  llvm::SmallVector<LoweredSafetyContract, 32> contracts;
};

void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   LoweredSafetyContracts &contracts);
void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   const PyLLVMTypeConverter &typeConverter,
                                   LoweredSafetyContracts &contracts);
mlir::LogicalResult
preserveLoweredSafetyContracts(mlir::ModuleOp module,
                               const LoweredSafetyContracts &contracts);

mlir::LogicalResult verifyOwnership(mlir::ModuleOp module);
mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp module);
mlir::LogicalResult verifyAlgorithmMEvidence(mlir::ModuleOp module);
mlir::LogicalResult verifyRuntimeManifestCompleteness(mlir::ModuleOp module);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafeVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAlgorithmMEvidenceVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeManifestCompletenessVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass();

} // namespace py
