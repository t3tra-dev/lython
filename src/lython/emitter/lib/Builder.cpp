#include "BuilderImpl.h"

#include <memory>
#include <utility>

namespace lython::emitter {

Builder::Builder(mlir::MLIRContext &context, std::string moduleName)
    : impl(std::make_unique<Impl>(context, std::move(moduleName))) {}

Builder::~Builder() = default;

EmitResult Builder::emit(const parser::Node &module) {
  return impl->emit(module);
}

} // namespace lython::emitter
