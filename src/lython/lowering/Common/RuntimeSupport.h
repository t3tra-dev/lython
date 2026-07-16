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
  AppleAMX,
  X86SSE42,
  X86AVX2FMA,
};

struct TensorLoweringTarget {
  TensorLoweringArchitecture architecture = TensorLoweringArchitecture::Generic;

  // FEAT_SME_F64F64 is optional on top of SME: without it the ZA double-word
  // tiles the f64 outer product needs do not exist, and codegen would fail to
  // select FMOPA.D rather than degrade.
  bool armSMEF64F64 = false;

  // FEAT_SME2 adds the multi-vector loads (LD1W {z0-z3}, pn/z) and the
  // predicate-as-counter registers the wide register-block kernels need.
  bool armSME2 = false;

  bool usesArmSME() const {
    return architecture == TensorLoweringArchitecture::ArmSME;
  }

  // Apple's matrix coprocessor. Unlike every other architecture here it is not
  // settled at compile time: the instructions are undocumented reserved A64
  // words with no capability bit, so a binary that emits them must ask the
  // running machine (see LyMatrixBackend) and keep a portable path for when
  // the answer is no.
  bool usesAppleAMX() const {
    return architecture == TensorLoweringArchitecture::AppleAMX;
  }

  bool usesArmSMEF64() const { return usesArmSME() && armSMEF64F64; }

  bool usesArmSME2() const { return usesArmSME() && armSME2; }

  bool usesX86() const {
    return architecture == TensorLoweringArchitecture::X86SSE42 ||
           architecture == TensorLoweringArchitecture::X86AVX2FMA;
  }

  // The two facts the generic kernel's register tile has to be sized against.
  // Exposed as capacities rather than asked for by ISA name: what that tile
  // needs to know is how many vectors it can keep live and how wide they are,
  // and a schedule written against those transfers to the next target without
  // reading it here.
  //
  // Arm's NEON file is 32 registers; x86's SSE/AVX are 16 (AVX-512 would be 32,
  // and is not a target here).
  unsigned vectorRegisterCount() const { return usesX86() ? 16 : 32; }

  unsigned vectorRegisterBits() const {
    return architecture == TensorLoweringArchitecture::X86AVX2FMA ? 256 : 128;
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
void installArmStreamingCompatibleMemoryRoutines(llvm::Module &module);

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
