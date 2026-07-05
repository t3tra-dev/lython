#pragma once

#include "Common/RuntimeSupport.h"

#include "llvm/ADT/StringRef.h"

#include <set>
#include <string>

namespace py {

struct IRDumpConfig {
  bool all = false;
  std::set<std::string> passes;

  static IRDumpConfig fromEnv();
  bool shouldDump(llvm::StringRef passName) const;
};

mlir::LogicalResult runLoweringPipeline(mlir::ModuleOp module,
                                        TensorLoweringTarget tensorTarget,
                                        const IRDumpConfig &irDump,
                                        bool auditRuntimeManifest = false);

} // namespace py
