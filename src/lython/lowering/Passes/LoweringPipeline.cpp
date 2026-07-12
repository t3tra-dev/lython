// Layer 3 of the transformation stack:
// the global lowering stages and their order. Per-op lowerings live in
// RuntimeBundleLowerer (layers 1-2, Passes/Runtime/); target-selected
// schedules ship as transform-dialect strategies in the module manifests
// (layer 4, applied at phase 8b).
#include "Common/LoweringPipeline.h"

#include "Common/Instrumentation.h"
#include "Common/RuntimeLibrary.h"
#include "Common/RuntimeSupport.h"
#include "Passes/Runtime/Arch/Arm/PrimitiveTensorArmSME.h"
#include "Passes/Runtime/Cleanup/Transforms.h"
#include "Passes/Runtime/Ctypes/CallbackThunks.h"
#include "runtime/Verification.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
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
                               bool enableVerifier, Populate populate) {
  std::string phase = (llvm::Twine("lowering.") + name).str();
  PerfScope perf(phase);
  PassManager pm(module.getContext());
  pm.enableVerifier(enableVerifier);
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
                                  const IRDumpConfig &irDump,
                                  LoweringPipelineOptions options) {
  dumpMLIRForPass(irDump, "frontend", module);
  auto runPhase = [&](llvm::StringRef name, auto populate) {
    return runLoweringPhase(name, module, options.enableVerifiers, populate);
  };
  auto runVerifierPhase = [&](llvm::StringRef name, auto populate) {
    if (!options.enableVerifiers)
      return success();
    return runPhase(name, populate);
  };

  // Phase 1: native target and ABI facts.
  if (failed(runVerifierPhase("native-verification", [&](PassManager &pm) {
        pm.addPass(createNativeVerificationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "native-verification", module);

  // Phase 2: publish high-level callable/runtime metadata before rewrites.
  if (failed(runPhase("publication-preparation", [&](PassManager &pm) {
        pm.addPass(createPublicationPreparationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "publication-preparation", module);

  // Phase 3: high-level Py semantic optimizations before runtime import.
  if (failed(runPhase("py-optimization", [&](PassManager &pm) {
        pm.addPass(createPyOptimizationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "py-optimization", module);

  // Phase 4: semantic evidence verification before lowering consumes Py ops.
  if (failed(runVerifierPhase(
          "algorithmm-evidence-verifier", [&](PassManager &pm) {
            pm.addPass(createAlgorithmMEvidenceVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "algorithmm-evidence-verifier", module);

  // Phase 5: quantitative ownership verification over high-level Py IR.
  if (failed(runVerifierPhase("ownership-verifier", [&](PassManager &pm) {
        pm.addPass(createOwnershipVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "ownership-verifier", module);

  // Phase 6: generic cleanup while Python-level structure is still visible.
  if (failed(runPhase("canonicalize", [&](PassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "canonicalize", module);

  // Phase 7: numeric kernel lowering before Python object lowering.
  if (failed(runPhase("linalg-lowering", [&](PassManager &pm) {
        pm.addPass(createLinalgLoweringPass(tensorTarget));
      })))
    return failure();
  dumpMLIRForPass(irDump, "linalg-lowering", module);

  // Phase 8: import runtime object definitions written in MLIR.
  {
    PerfScope perf("lowering.runtime-objects");
    if (failed(runtime_library::embedObjectModules(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "runtime-objects", module);

  // Phase 8b: interpret manifest-declared lowering strategies (transform
  // dialect) against the user module.
  {
    PerfScope perf("lowering.lowering-strategies");
    if (failed(runtime_library::applyEmbeddedLoweringStrategies(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "lowering-strategies", module);

  if (options.auditRuntimeManifest && options.enableVerifiers) {
    if (failed(runVerifierPhase(
            "runtime-manifest-completeness", [&](PassManager &pm) {
              pm.addPass(createRuntimeManifestCompletenessVerifierPass());
            })))
      return failure();
    dumpMLIRForPass(irDump, "runtime-manifest-completeness", module);
  }

  // Phase 9: lower Py dialect values into runtime bundles and calls.
  if (failed(runPhase("runtime-lowering", [&](PassManager &pm) {
        pm.addPass(createRuntimeLoweringPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "runtime-lowering", module);

  if (failed(runVerifierPhase("runtime-native-verifier", [&](PassManager &pm) {
        pm.addPass(createNativeVerificationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "runtime-native-verifier", module);

  // Phase 10: insert and simplify ownership operations once calls are concrete.
  if (failed(runPhase("refcount-insertion", [&](PassManager &pm) {
        pm.addPass(createRefCountInsertionPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "refcount-insertion", module);

  if (failed(runPhase("refcount-elision", [&](PassManager &pm) {
        pm.addPass(createRefCountPairElisionPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "refcount-elision", module);

  if (failed(runVerifierPhase(
          "pre-cleanup-llvm-call-verifier", [&](PassManager &pm) {
            pm.addPass(createLLVMCallOwnershipVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "pre-cleanup-llvm-call-verifier", module);

  // Phase 11: lower Lython-owned async thunks before symbol cleanup.
  if (failed(runPhase("async-thunk-lowering", [&](PassManager &pm) {
        pm.addPass(createAsyncThunkLoweringPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "async-thunk-lowering", module);
  if (failed(requireNoAsyncDialectOps(module)))
    return failure();

  // Phase 12: remove artifacts from runtime embedding and lowering. Symbol
  // DCE runs AFTER the ownership verifiers (phase 13): manifest contract
  // witnesses (e.g. ly.runtime.shape declarations) must outlive verification.
  if (failed(runPhase("post-lowering-cleanup", [&](PassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      })))
    return failure();
  {
    PerfScope perf("lowering.pointer-roundtrip-cleanup");
    while (lowering::runtime::cleanup::pointerRoundTrips(module))
      ;
  }
  dumpMLIRForPass(irDump, "post-lowering-cleanup", module);

  // Phase 13: validate ownership and no-GIL contracts before final lowering.
  if (failed(runVerifierPhase("llvm-call-verifier", [&](PassManager &pm) {
        pm.addPass(createLLVMCallOwnershipVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "llvm-call-verifier", module);

  if (failed(runVerifierPhase("thread-safety-verifier", [&](PassManager &pm) {
        pm.addPass(createLLVMThreadSafeVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "thread-safety-verifier", module);

  // Contract witnesses are no longer needed once verification is done.
  if (failed(runPhase("post-verifier-symbol-dce", [&](PassManager &pm) {
        pm.addPass(mlir::createSymbolDCEPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "post-verifier-symbol-dce", module);

  // Phase 14: final lowering to LLVM dialect.
  {
    LoweredSafetyContracts finalSafetyContracts;
    if (options.enableVerifiers) {
      PerfScope perf("lowering.collect-final-safety-contracts");
      collectLoweredSafetyContracts(module, finalSafetyContracts);
    }
    if (failed(runPhase("convert-to-llvm", [&](PassManager &pm) {
          mlir::ConvertVectorToLLVMPassOptions vectorOptions;
          vectorOptions.reassociateFPReductions = true;
          vectorOptions.x86Vector = tensorTarget.usesX86();
          mlir::VectorTransferToSCFOptions transferOptions;
          transferOptions.setTargetRank(1);
          pm.addPass(mlir::createLowerAffinePass());
          pm.addPass(mlir::memref::createExpandStridedMetadataPass());
          pm.addPass(mlir::createLowerAffinePass());
          if (tensorTarget.usesArmSME())
            lowering::arch::arm::addSMEPreControlFlowLLVMPrepPipeline(pm);
          pm.addNestedPass<mlir::func::FuncOp>(
              mlir::vector::createLowerVectorMultiReductionPass(
                  mlir::vector::VectorMultiReductionLowering::InnerReduction));
          pm.addPass(mlir::createConvertVectorToSCFPass(transferOptions));
          pm.addPass(mlir::createLowerAffinePass());
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createConvertVectorToLLVMPass(vectorOptions));
          pm.addPass(mlir::createSCFToControlFlowPass());
          if (tensorTarget.usesArmSME())
            lowering::arch::arm::addSMEPostControlFlowLLVMPrepPipeline(pm);
          pm.addPass(mlir::createArithToLLVMConversionPass());
          pm.addPass(mlir::createUBToLLVMConversionPass());
          pm.addPass(mlir::createConvertControlFlowToLLVMPass());
          pm.addPass(mlir::createConvertToLLVMPass());
          pm.addPass(mlir::createReconcileUnrealizedCastsPass());
          pm.addNestedPass<mlir::func::FuncOp>(
              mlir::createReconcileUnrealizedCastsPass());
          pm.addPass(mlir::createCanonicalizerPass());
        })))
      return failure();
    if (options.enableVerifiers) {
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

  // Phase 13c: materialize ctypes callback thunks -- function addresses only
  // exist at the LLVM layer (see Ctypes/CallbackThunks.h).
  {
    PerfScope perf("lowering.callback-thunks");
    if (failed(lowering::ctypes::materializeCallbackThunks(module)))
      return failure();
    if (failed(lowering::ctypes::materializeSymbolAddresses(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "callback-thunks", module);

  // Phase 14: re-check contracts after final conversion rewrites.
  if (failed(runVerifierPhase("final-native-verifier", [&](PassManager &pm) {
        pm.addPass(createNativeVerificationPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "final-native-verifier", module);

  if (failed(runVerifierPhase("final-ownership-verifier", [&](PassManager &pm) {
        pm.addPass(createLLVMCallOwnershipVerifierPass());
      })))
    return failure();
  dumpMLIRForPass(irDump, "final-ownership-verifier", module);

  if (failed(runVerifierPhase("final-thread-safety-verifier",
                              [&](PassManager &pm) {
                                pm.addPass(createLLVMThreadSafeVerifierPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "final-thread-safety-verifier", module);

  return success();
}

} // namespace py
