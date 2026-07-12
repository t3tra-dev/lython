#pragma once

#include <cstddef>

namespace py::runtime_library::embedded {

enum class ModuleKind {
  MLIRBytecode,
  NativeMLIRBytecode,
};

// One runtime module compiled into the binary at build time: module manifest
// bytecode (runtime/modules/*.mlir) plus pre-lowered runtime-internal lib
// bytecode. NativeMLIRBytecode entries are lowered to LLVM IR inside lyc only
// after the final target triple/data layout is known.
struct Module {
  const char *name;
  ModuleKind kind;
  const unsigned char *data;
  std::size_t size;
};

const Module *modules();
std::size_t moduleCount();

// Extension point for runtime modules that are PRE-LOWERED FROM PYTHON
// SOURCE at build time (runtime/lib/*.py). They cannot live in modules():
// the pre-lowering tool itself links this library and compiles them, so the
// base registry must not depend on their existence. lyc registers the
// generated set at startup; the tool never does.
void registerExtraModules(const Module *extra, std::size_t count);
const Module *extraModules();
std::size_t extraModuleCount();

// Defined by the generated embedded_lib_internal.cpp linked into lyc only;
// calls registerExtraModules with the pre-lowered runtime/lib artifacts.
void registerPyRuntimeEmbeddedModules();

// Python stdlib modules shipped as SOURCE (runtime/lib/*.py, CPython's Lib/
// counterpart). The import search resolves them after user files and before
// module manifests; they compile with the user program like any source
// module. Defined by the generated embedded_stdlib.cpp linked into lyc only.
struct StdlibSourceModule {
  const char *name;
  const unsigned char *source;
  std::size_t size;
};
const StdlibSourceModule *stdlibSourceModules();
std::size_t stdlibSourceModuleCount();

} // namespace py::runtime_library::embedded
