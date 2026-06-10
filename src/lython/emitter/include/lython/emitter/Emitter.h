#pragma once

#include "lython/parser/Ast.h"
#include "lython/parser/Diagnostics.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <string>

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

EmitResult emitModule(const parser::Node &module, mlir::MLIRContext &context,
                      std::string moduleName = "__main__");
TextEmitResult emitModuleText(const parser::Node &module,
                              std::string moduleName = "__main__");

} // namespace lython::emitter
