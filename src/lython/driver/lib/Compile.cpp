#include "Driver.h"
#include "DriverCodeGen.h"

#include "Common/Instrumentation.h"
#include "Common/LoweringPipeline.h"
#include "Common/RuntimeSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

using namespace mlir;

namespace lython::driver {
using py::PerfScope;

LogicalResult translateToVerifiedLLVMIR(ModuleOp module,
                                        const DriverOptions &options,
                                        const py::IRDumpConfig &irDump,
                                        VerifiedLLVMModule &out,
                                        llvm::raw_ostream &diag) {
  LLVMSafetyProfile safetyProfile;
  if (!options.releaseMode)
    collectLLVMSafetyContracts(module, safetyProfile);
  llvm::SmallVector<py::PythonCallSiteRange, 16> pythonCallSites;
  py::collectPythonCallSiteRanges(module, pythonCallSites);
  attachPythonDebugInfo(module);

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    diag << "Failed to translate to LLVM IR\n";
    return failure();
  }
  py::installPythonExceptionCleanupFrames(*llvmModule, pythonCallSites);
  dumpLLVMForPass(irDump, "llvm-translation", *llvmModule);
  if (!options.releaseMode &&
      failed(verifyLLVMIRSafetyMetadataAttached(*llvmModule, safetyProfile,
                                                diag)))
    return failure();
  if (!options.releaseMode &&
      failed(verifyPostCoroLLVMThreadSafe(*llvmModule, safetyProfile, diag)))
    return failure();

  out.llvmContext = std::move(llvmContext);
  out.llvmModule = std::move(llvmModule);
  out.safetyProfile = std::move(safetyProfile);
  return success();
}

LogicalResult compilePythonSourceToLLVMIR(llvm::StringRef source,
                                          llvm::StringRef sourcePath,
                                          llvm::StringRef importBaseDir,
                                          const DriverOptions &options,
                                          MLIRContext &context,
                                          VerifiedLLVMModule &out,
                                          llvm::raw_ostream &diag) {
  OwningOpRef<ModuleOp> module;
  if (failed(emitMLIRFromSource(source, sourcePath, importBaseDir, options,
                                context, module, diag)))
    return failure();

  py::TensorLoweringTarget tensorTarget = detectTensorLoweringTarget(options);
  if (failed(stampTargetPlatformFacts(*module, tensorTarget, options, diag)))
    return failure();

  // Library callers get deterministic behavior: IR dumping stays off instead
  // of reading LYTHON_IR_DUMP from the environment.
  py::IRDumpConfig irDump;
  {
    PerfScope perf("lowering");
    py::LoweringPipelineOptions loweringOptions;
    loweringOptions.auditRuntimeManifest = options.auditRuntimeManifest;
    loweringOptions.enableVerifiers = !options.releaseMode;
    if (failed(py::runLoweringPipeline(*module, tensorTarget, irDump,
                                       loweringOptions))) {
      diag << "Failed to run lowering pipeline\n";
      return failure();
    }
  }

  return translateToVerifiedLLVMIR(*module, options, irDump, out, diag);
}

} // namespace lython::driver
