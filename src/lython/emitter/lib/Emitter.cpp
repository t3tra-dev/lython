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

} // namespace lython::emitter
