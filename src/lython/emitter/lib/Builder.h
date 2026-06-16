#pragma once

#include "Emitter.h"

#include <memory>

namespace lython::emitter {

class Builder {
public:
  Builder(mlir::MLIRContext &context, std::string moduleName);
  ~Builder();

  EmitResult emit(const parser::Node &module);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace lython::emitter
