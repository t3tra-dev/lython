#include "lython/emitter/Emitter.h"

#include "Builder.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace lython::emitter {

EmitResult emitModule(const parser::Node &module, mlir::MLIRContext &context,
                      std::string moduleName) {
  Builder builder(context, std::move(moduleName));
  return builder.emit(module);
}

TextEmitResult emitModuleText(const parser::Node &module,
                              std::string moduleName) {
  mlir::MLIRContext context;
  EmitResult emitted = emitModule(module, context, std::move(moduleName));
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
