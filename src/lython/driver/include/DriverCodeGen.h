#pragma once

#include "Common/LoweringPipeline.h"
#include "Common/RuntimeSupport.h"
#include "SanitizerSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace lython::driver {

struct DriverOptions;

enum class LLVMSafetyEffectKind {
  AtomicRMW,
  AtomicCmpXchg,
  AtomicLoad,
  AtomicStore,
};

struct LLVMSafetyContract {
  int64_t id = -1;
  std::string functionName;
  LLVMSafetyEffectKind kind;
  std::optional<llvm::AtomicRMWInst::BinOp> rmwBinOp;
  std::optional<int64_t> integerOperand;
  std::optional<llvm::AtomicOrdering> ordering;
};

struct LLVMSafetyProfile {
  llvm::SmallVector<LLVMSafetyContract, 64> contracts;
};

enum class LLVMSafetyContractCoverage {
  RequireEveryContract,
  AllowOptimizerElision,
};

// --- Target configuration -------------------------------------------------

std::string configuredTargetSysrootOverride(const DriverOptions &options);

llvm::Triple codeGenTripleForTarget(py::TensorLoweringTarget target,
                                    const DriverOptions &options);

std::string codeGenCPUNameForTarget(py::TensorLoweringTarget target,
                                    const llvm::Triple &triple,
                                    const DriverOptions &options);

std::string codeGenFeaturesForTarget(py::TensorLoweringTarget target,
                                     const llvm::Triple &triple,
                                     const DriverOptions &options);

llvm::ExceptionHandling
exceptionModelForTargetTriple(const llvm::Triple &triple);

std::unique_ptr<llvm::TargetMachine>
createCodeGenTargetMachine(py::TensorLoweringTarget target,
                           const DriverOptions &options,
                           std::string *normalizedTriple,
                           llvm::raw_ostream &diag);

mlir::LogicalResult
configureLLVMModuleCodeGenTarget(llvm::Module &llvmModule,
                                 py::TensorLoweringTarget tensorTarget,
                                 const DriverOptions &options,
                                 llvm::raw_ostream &diag);

// --- Safety contract collection / verification ------------------------------

void registerPySafetyLLVMIRTranslation(mlir::DialectRegistry &registry);

void collectLLVMSafetyContracts(mlir::ModuleOp module,
                                LLVMSafetyProfile &profile);

void collectLinkedLLVMSafetyContracts(llvm::Module &llvmModule,
                                      LLVMSafetyProfile &profile);

mlir::LogicalResult
verifyLLVMIRSafetyMetadataAttached(llvm::Module &llvmModule,
                                   const LLVMSafetyProfile &profile,
                                   llvm::raw_ostream &diag);

mlir::LogicalResult verifyPostCoroLLVMThreadSafe(
    llvm::Module &llvmModule, const LLVMSafetyProfile &profile,
    llvm::raw_ostream &diag,
    LLVMSafetyContractCoverage coverage =
        LLVMSafetyContractCoverage::RequireEveryContract);

mlir::LogicalResult
verifyOptimizedLLVMThreadSafe(llvm::Module &llvmModule,
                              const LLVMSafetyProfile &profile,
                              llvm::raw_ostream &diag);

// --- LLVM IR finalization ---------------------------------------------------

void dumpLLVMForPass(const py::IRDumpConfig &config, llvm::StringRef passName,
                     llvm::Module &module);

mlir::LogicalResult writeLLVMIR(llvm::Module &llvmModule,
                                llvm::StringRef outputPath,
                                llvm::raw_ostream &diag);

mlir::LogicalResult installAOTEntryPoint(llvm::Module &llvmModule,
                                         llvm::raw_ostream &diag);

void runLLVMCoroLowering(
    llvm::Module &llvmModule, const SanitizerConfig &sanitizers,
    llvm::TargetMachine *targetMachine = nullptr,
    llvm::OptimizationLevel optimizationLevel = llvm::OptimizationLevel::O2);

void rewriteExceptionPersonalityForTarget(llvm::Module &llvmModule);

void attachPythonDebugInfo(mlir::ModuleOp module);

std::string pythonTracebackPath(llvm::StringRef inputPath, bool releaseMode);

} // namespace lython::driver
