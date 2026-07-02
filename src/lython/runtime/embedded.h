#pragma once

#include <cstddef>

namespace py::runtime_library::embedded {

enum class ModuleKind {
  MLIRBytecode,
  NativeMLIRBytecode,
};

// One runtime module compiled into the binary at build time. The registry
// carries high-level MLIR bytecode modules (object, long, unicode, exception,
// typing, ...) plus native support MLIR bytecode. Native support is lowered to
// LLVM IR inside lyc after the final target triple/data layout is known.
struct Module {
  const char *name;
  ModuleKind kind;
  const unsigned char *data;
  std::size_t size;
};

const Module *modules();
std::size_t moduleCount();

} // namespace py::runtime_library::embedded
