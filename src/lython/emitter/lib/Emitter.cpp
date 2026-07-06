#include "Emitter.h"

#include "EmitterCore.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace lython::emitter {

EmitResult emitModule(const parser::Node &moduleNode,
                      mlir::MLIRContext &context, std::string moduleName,
                      std::string sourceName, EmitOptions options) {
  ModuleEmitter emitter(moduleNode, context, std::move(moduleName),
                        std::move(sourceName), options);
  return emitter.emit();
}

TextEmitResult emitModuleText(const parser::Node &moduleNode,
                              std::string moduleName, std::string sourceName) {
  mlir::MLIRContext context;
  EmitResult emitted = emitModule(moduleNode, context, std::move(moduleName),
                                  std::move(sourceName));
  TextEmitResult result;
  result.diagnostics = std::move(emitted.diagnostics);
  if (!emitted.module)
    return result;

  llvm::raw_string_ostream os(result.mlir);
  emitted.module->print(os);
  os << "\n";
  return result;
}

} // namespace lython::emitter
