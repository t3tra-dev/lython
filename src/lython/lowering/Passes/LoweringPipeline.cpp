#include "Common/LoweringPipeline.h"

#include "Common/Instrumentation.h"
#include "Common/RuntimeLibrary.h"
#include "Common/RuntimeSupport.h"
#include "Passes/Runtime/Cleanup/Transforms.h"
#include "Passes/Runtime/Arch/Arm/PrimitiveTensorArmSME.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace py {
namespace {

std::string trimEnvToken(llvm::StringRef token) {
  token = token.trim();
  return token.str();
}

void dumpMLIRForPass(const IRDumpConfig &config, llvm::StringRef passName,
                     ModuleOp module) {
  if (!config.shouldDump(passName))
    return;
  llvm::errs() << "\n=== [LYTHON_IR_DUMP:" << passName << " MLIR] ===\n";
  module->print(llvm::errs());
  llvm::errs() << "\n";
}

template <typename Populate>
LogicalResult runLoweringPhase(llvm::StringRef name, ModuleOp module,
                               Populate populate) {
  std::string phase = (llvm::Twine("lowering.") + name).str();
  PerfScope perf(phase);
  PassManager pm(module.getContext());
  populate(pm);
  return pm.run(module);
}

LogicalResult requireNoAsyncDialectOps(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() != "async")
      return WalkResult::advance();
    op->emitError()
        << "unlowered async dialect operation remains after Lython async "
           "runtime lowering; MLIR bundled async runtime is not part of the "
           "runtime model";
    result = failure();
    return WalkResult::interrupt();
  });
  return result;
}

} // namespace

IRDumpConfig IRDumpConfig::fromEnv() {
  IRDumpConfig config;
  auto value = llvm::sys::Process::GetEnv("LYTHON_IR_DUMP");
  if (!value || value->empty())
    return config;
  llvm::SmallVector<llvm::StringRef, 16> tokens;
  llvm::StringRef(*value).split(tokens, ",", /*MaxSplit=*/-1,
                                /*KeepEmpty=*/false);
  for (llvm::StringRef token : tokens) {
    std::string name = trimEnvToken(token);
    if (name.empty())
      continue;
    if (name == "all" || name == "*") {
      config.all = true;
      continue;
    }
    config.passes.insert(std::move(name));
  }
  return config;
}

bool IRDumpConfig::shouldDump(llvm::StringRef passName) const {
  return all || passes.count(passName.str()) != 0;
}

LogicalResult runLoweringPipeline(ModuleOp module,
                                  TensorLoweringTarget tensorTarget,
                                  const IRDumpConfig &irDump) {
  dumpMLIRForPass(irDump, "frontend", module);

  // Phase 1: native target and ABI facts.
  if (failed(
          runLoweringPhase("native-verification", module, [&](PassManager &pm) {
            pm.addPass(createNativeVerificationPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "native-verification", module);

  // Phase 2: publish high-level callable/runtime metadata before rewrites.
  if (failed(runLoweringPhase("publication-preparation", module,
                              [&](PassManager &pm) {
                                pm.addPass(createPublicationPreparationPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "publication-preparation", module);

  // Phase 3: high-level Py semantic optimizations before runtime import.
  if (failed(runLoweringPhase("py-optimization", module, [&](PassManager &pm) {
        pm.addPass(createPyOptimizationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "py-optimization", module);

  // Phase 4: quantitative ownership verification over high-level Py IR.
  if (failed(
          runLoweringPhase("ownership-verifier", module, [&](PassManager &pm) {
            pm.addPass(createOwnershipVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "ownership-verifier", module);

  // Phase 5: generic cleanup while Python-level structure is still visible.
  if (failed(runLoweringPhase("canonicalize", module, [&](PassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "canonicalize", module);

  // Phase 6: numeric kernel lowering before Python object lowering.
  if (failed(runLoweringPhase("linalg-lowering", module, [&](PassManager &pm) {
        pm.addPass(createLinalgLoweringPass(tensorTarget));
      })))
    return failure();
  dumpMLIRForPass(irDump, "linalg-lowering", module);

  // Phase 7: import runtime object definitions written in MLIR.
  {
    PerfScope perf("lowering.runtime-objects");
    if (failed(runtime_library::embedObjectModules(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "runtime-objects", module);

  // Phase 8: lower Py dialect values into runtime bundles and calls.
  if (failed(runLoweringPhase("runtime-lowering", module, [&](PassManager &pm) {
        pm.addPass(createRuntimeLoweringPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "runtime-lowering", module);

  // Phase 9: insert and simplify ownership operations once calls are concrete.
  if (failed(
          runLoweringPhase("refcount-insertion", module, [&](PassManager &pm) {
            pm.addPass(createRefCountInsertionPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "refcount-insertion", module);

  if (failed(runLoweringPhase("refcount-elision", module, [&](PassManager &pm) {
        pm.addPass(createRefCountPairElisionPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "refcount-elision", module);

  // Phase 10: lower Lython-owned async thunks before symbol cleanup.
  if (failed(runLoweringPhase("async-thunk-lowering", module,
                              [&](PassManager &pm) {
                                pm.addPass(createAsyncThunkLoweringPass());
                                pm.addPass(mlir::createCanonicalizerPass());
                                pm.addPass(mlir::createCSEPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "async-thunk-lowering", module);
  if (failed(requireNoAsyncDialectOps(module)))
    return failure();

  // Phase 11: remove artifacts from runtime embedding and lowering.
  if (failed(runLoweringPhase("post-lowering-cleanup", module,
                              [&](PassManager &pm) {
                                pm.addPass(mlir::createCanonicalizerPass());
                                pm.addPass(mlir::createCSEPass());
                                pm.addPass(mlir::createSymbolDCEPass());
                              })))
    return failure();
  {
    PerfScope perf("lowering.pointer-roundtrip-cleanup");
    while (lowering::runtime::cleanup::pointerRoundTrips(module))
      ;
  }
  dumpMLIRForPass(irDump, "post-lowering-cleanup", module);

  // Phase 12: validate ownership and no-GIL contracts before final lowering.
  if (failed(
          runLoweringPhase("llvm-call-verifier", module, [&](PassManager &pm) {
            pm.addPass(createLLVMCallOwnershipVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "llvm-call-verifier", module);

  if (failed(runLoweringPhase(
          "thread-safety-verifier", module, [&](PassManager &pm) {
            pm.addPass(createLLVMThreadSafetyVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "thread-safety-verifier", module);

  // Phase 13: final lowering to LLVM dialect.
  {
    LoweredSafetyContracts finalSafetyContracts;
    {
      PerfScope perf("lowering.collect-final-safety-contracts");
      collectLoweredSafetyContracts(module, finalSafetyContracts);
    }
    if (failed(
            runLoweringPhase("convert-to-llvm", module, [&](PassManager &pm) {
              mlir::ConvertVectorToLLVMPassOptions vectorOptions;
              vectorOptions.reassociateFPReductions = true;
              vectorOptions.x86Vector = tensorTarget.usesX86();
              mlir::VectorTransferToSCFOptions transferOptions;
              transferOptions.setTargetRank(1);
              pm.addPass(mlir::createLowerAffinePass());
              pm.addPass(mlir::memref::createExpandStridedMetadataPass());
              pm.addPass(mlir::createLowerAffinePass());
              if (tensorTarget.usesArmSME())
                runtime_lowering::arch::arm::
                    addSMEPreControlFlowLLVMPrepPipeline(pm);
              pm.addNestedPass<mlir::func::FuncOp>(
                  mlir::vector::createLowerVectorMultiReductionPass(
                      mlir::vector::VectorMultiReductionLowering::
                          InnerReduction));
              pm.addPass(mlir::createConvertVectorToSCFPass(transferOptions));
              pm.addPass(mlir::createLowerAffinePass());
              pm.addPass(mlir::createCanonicalizerPass());
              pm.addPass(mlir::createConvertVectorToLLVMPass(vectorOptions));
              pm.addPass(mlir::createConvertSCFToCFPass());
              if (tensorTarget.usesArmSME())
                runtime_lowering::arch::arm::
                    addSMEPostControlFlowLLVMPrepPipeline(pm);
              pm.addPass(mlir::createArithToLLVMConversionPass());
              pm.addPass(mlir::createConvertControlFlowToLLVMPass());
              pm.addPass(mlir::createConvertToLLVMPass());
              pm.addPass(mlir::createReconcileUnrealizedCastsPass());
              pm.addNestedPass<mlir::func::FuncOp>(
                  mlir::createReconcileUnrealizedCastsPass());
              pm.addPass(mlir::createCanonicalizerPass());
            })))
      return failure();
    {
      PerfScope perf("lowering.preserve-final-safety-contracts");
      if (failed(preserveLoweredSafetyContracts(module, finalSafetyContracts)))
        return failure();
    }
    {
      PerfScope perf("lowering.final-llvm-cleanup");
      optimizer::pipeline::finalLLVMCleanup(module);
    }
  }
  dumpMLIRForPass(irDump, "convert-to-llvm", module);

  // Phase 14: re-check contracts after final conversion rewrites.
  if (failed(runLoweringPhase(
          "final-ownership-verifier", module, [&](PassManager &pm) {
            pm.addPass(createLLVMCallOwnershipVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "final-ownership-verifier", module);

  if (failed(runLoweringPhase(
          "final-thread-safety-verifier", module, [&](PassManager &pm) {
            pm.addPass(createLLVMThreadSafetyVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "final-thread-safety-verifier", module);

  return success();
}

} // namespace py
