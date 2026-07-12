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

struct TextEmitResult {
  std::string mlir;
  parser::Diagnostics diagnostics;

  bool ok() const { return diagnostics.empty(); }
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
  std::string mainPackageName;
  std::string targetTriple;
  std::vector<SourceModule> sourceModules;
};

EmitResult emitModule(const parser::Node &module, mlir::MLIRContext &context,
                      std::string moduleName = "__main__",
                      std::string sourceName = {}, EmitOptions options = {});
TextEmitResult emitModuleText(const parser::Node &module,
                              std::string moduleName = "__main__",
                              std::string sourceName = {});

} // namespace lython::emitter
