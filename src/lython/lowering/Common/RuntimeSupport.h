#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <string>

namespace llvm {
class Module;
}

namespace py {

enum class TensorLoweringArchitecture {
  Generic,
  ArmSME,
  X86SSE42,
  X86AVX2FMA,
};

struct TensorLoweringTarget {
  TensorLoweringArchitecture architecture = TensorLoweringArchitecture::Generic;

  bool usesArmSME() const {
    return architecture == TensorLoweringArchitecture::ArmSME;
  }

  bool usesX86() const {
    return architecture == TensorLoweringArchitecture::X86SSE42 ||
           architecture == TensorLoweringArchitecture::X86AVX2FMA;
  }

  bool usesX86SSE42() const {
    return architecture == TensorLoweringArchitecture::X86SSE42;
  }

  bool usesX86AVX2FMA() const {
    return architecture == TensorLoweringArchitecture::X86AVX2FMA;
  }
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

void collectPythonCallSiteRanges(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<PythonCallSiteRange> &callSites);
bool installPythonExceptionCleanupFrames(
    llvm::Module &module, llvm::ArrayRef<PythonCallSiteRange> callSites);

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
createAsyncThunkLoweringPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLinalgLoweringPass(TensorLoweringTarget target = {});

} // namespace py
