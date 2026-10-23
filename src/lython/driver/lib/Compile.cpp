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
  llvm::SmallVector<py::PythonCallSiteRange, 16> pythonCallSites;
  llvm::DenseMap<std::int64_t, unsigned> tryInlineDepths;
  llvm::SmallVector<std::string, 4> ctypesSymbols;
  {
    PerfScope perf("translate-to-llvm.collect");
    if (!options.releaseMode)
      collectLLVMSafetyContracts(module, safetyProfile);
    py::installFrameRecords(module);
    py::collectPythonCallSiteRanges(module, pythonCallSites);
    py::collectPythonTryInlineDepths(module, tryInlineDepths);
    py::collectCtypesForeignSymbols(module, ctypesSymbols);
  }
  {
    PerfScope perf("translate-to-llvm.debug-info");
    attachPythonDebugInfo(module);
  }
  // Translation folds constant struct GEPs into byte offsets with the
  // module's layout, so the target's has to be there first (see the same
  // step in RuntimeLibrary.cpp).
  stampTargetDataLayout(module, options);

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvmModule;
  {
    PerfScope perf("translate-to-llvm.translate");
    // ⛔ Not verified here as well when the module is verified below, after
    // the exception frames are installed: the same check twice.
    llvmModule = mlir::translateModuleToLLVMIR(
        module, *llvmContext, "LLVMDialectModule",
        /*disableVerification=*/!options.releaseMode);
  }
  if (!llvmModule) {
    diag << "Failed to translate to LLVM IR\n";
    return failure();
  }
  {
    PerfScope perf("translate-to-llvm.exception-frames");
    py::installJsHostEntryPoints(
        *llvmModule,
        codeGenTripleForTarget(py::TensorLoweringTarget{}, options));
    py::internalizePythonFunctions(*llvmModule);
    dumpLLVMForPass(irDump, "pre-eh-llvm", *llvmModule);
    py::markCLibraryDeclarationsNonUnwinding(*llvmModule, ctypesSymbols);
    py::installPythonExceptionCleanupFrames(
        *llvmModule,
        codeGenTripleForTarget(py::TensorLoweringTarget{}, options),
        pythonCallSites, tryInlineDepths);
    py::installArmStreamingCompatibleMemoryRoutines(*llvmModule);
  }
  dumpLLVMForPass(irDump, "llvm-translation", *llvmModule);
  if (!options.releaseMode) {
    PerfScope perf("translate-to-llvm.verify");
    if (failed(verifyPostCoroLLVMThreadSafe(*llvmModule, safetyProfile, diag)))
      return failure();
  }

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
