#pragma once

#include "DriverCodeGen.h"
#include "SanitizerSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

namespace lython::driver {

// Structured form of the lyc target/mode configuration. Every driver entry
// point receives it explicitly so library callers (tests, fuzzers) can
// compile without process-global state.
struct DriverOptions {
  std::string targetTriple;
  std::string targetCPU;
  std::string targetFPU;
  std::string targetFloatABI;
  std::string targetSysroot;
  SanitizerConfig sanitizers;
  bool releaseMode = false;
  bool auditRuntimeManifest = false;
};

// Registers every dialect, external model, translation interface, and pass
// registry entry the compilation pipeline depends on. Idempotent.
void registerLythonDialects(mlir::DialectRegistry &registry);

mlir::OwningOpRef<mlir::ModuleOp>
parseModuleFromBuffer(llvm::StringRef buffer, mlir::MLIRContext &context,
                      llvm::raw_ostream &diag);

// Frontend: parse -> local/embedded import resolution -> py dialect emission.
// importBaseDir is the search base for local source imports; a nonexistent
// directory restricts resolution to the embedded stdlib.
mlir::LogicalResult
emitMLIRFromSource(llvm::StringRef source, llvm::StringRef sourcePath,
                   llvm::StringRef importBaseDir, const DriverOptions &options,
                   mlir::MLIRContext &context,
                   mlir::OwningOpRef<mlir::ModuleOp> &module,
                   llvm::raw_ostream &diag);

mlir::LogicalResult emitMLIRFromFile(llvm::StringRef pythonFile,
                                     const DriverOptions &options,
                                     mlir::MLIRContext &context,
                                     mlir::OwningOpRef<mlir::ModuleOp> &module,
                                     llvm::raw_ostream &diag);

py::TensorLoweringTarget
detectTensorLoweringTarget(const DriverOptions &options);

mlir::LogicalResult stampTargetPlatformFacts(mlir::ModuleOp module,
                                             py::TensorLoweringTarget target,
                                             const DriverOptions &options,
                                             llvm::raw_ostream &diag);

struct VerifiedLLVMModule {
  std::unique_ptr<llvm::LLVMContext> llvmContext;
  std::unique_ptr<llvm::Module> llvmModule;
  LLVMSafetyProfile safetyProfile;
};

// MLIR (LLVM dialect) -> LLVM IR translation plus the non-release safety
// metadata verification that gates both the JIT and AOT paths.
mlir::LogicalResult
translateToVerifiedLLVMIR(mlir::ModuleOp module, const DriverOptions &options,
                          const py::IRDumpConfig &irDump,
                          VerifiedLLVMModule &out, llvm::raw_ostream &diag);

// One-call pipeline for library callers: parse -> import resolution -> emit ->
// target facts -> lowering pipeline (verifiers per options.releaseMode) ->
// verified LLVM IR. Stops before JIT execution / object emission.
mlir::LogicalResult
compilePythonSourceToLLVMIR(llvm::StringRef source, llvm::StringRef sourcePath,
                            llvm::StringRef importBaseDir,
                            const DriverOptions &options,
                            mlir::MLIRContext &context,
                            VerifiedLLVMModule &out, llvm::raw_ostream &diag);

} // namespace lython::driver
