#pragma once

#include "Common/RuntimeSupport.h"

#include "llvm/ADT/StringRef.h"

#include "mlir/Pass/PassManager.h"

#include <set>
#include <string>

namespace py {

struct IRDumpConfig {
  bool all = false;
  std::set<std::string> passes;

  static IRDumpConfig fromEnv();
  bool shouldDump(llvm::StringRef passName) const;
};

struct LoweringPipelineOptions {
  bool auditRuntimeManifest = false;
  bool enableVerifiers = true;
};

// Where a pass that rewrites inside one function at a time goes: nested under
// the module's isolated children when the context has threads to run them on
// at once, and the module-wide pass manager itself when it does not.
mlir::OpPassManager &eachIsolatedChild(mlir::OpPassManager &pm,
                                       mlir::MLIRContext *context);

mlir::LogicalResult runLoweringPipeline(mlir::ModuleOp module,
                                        TensorLoweringTarget tensorTarget,
                                        const IRDumpConfig &irDump,
                                        LoweringPipelineOptions options = {});

} // namespace py
