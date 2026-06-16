#pragma once

#include <cstddef>

namespace py::runtime_library::embedded {

// One runtime MLIR bytecode module compiled into the binary at build time.
// The registry carries the runtime object modules (object, long, unicode,
// exception, ...) and the typing manifest ("typing"); see
// src/lython/lowering/CMakeLists.txt.
struct Module {
  const char *name;
  const unsigned char *data;
  std::size_t size;
};

const Module *modules();
std::size_t moduleCount();

} // namespace py::runtime_library::embedded
