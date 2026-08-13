#pragma once

#include "Ast.h"
#include "Diagnostics.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <vector>

namespace lython::emitter {

struct EmitResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  parser::Diagnostics diagnostics;

  bool ok() const { return diagnostics.empty() && module; }
};

struct EmitOptions {
  struct SourceModule {
    std::string moduleName;
    std::string packageName;
    std::string sourceName;
    const parser::Node *moduleNode = nullptr;
    bool isStub = false;
  };

  bool sanitizeUndefined = false;
  // Set for `runtime/lib/*.py` compiled by RuntimePyLowering. Those modules
  // are linked into every program and reachable from a signal handler, so
  // their `: int` module globals keep the unboxed machine-word cell; a user
  // module's `: int` global is a Python integer and is boxed (see the
  // `ly.global.boxed` mark on py.global.get/set).
  bool runtimeInternal = false;
  std::string mainPackageName;
  std::string targetTriple;
  std::vector<SourceModule> sourceModules;
};

EmitResult emitModule(const parser::Node &module, mlir::MLIRContext &context,
                      std::string moduleName = "__main__",
                      std::string sourceName = {}, EmitOptions options = {});

} // namespace lython::emitter
