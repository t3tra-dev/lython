#pragma once

// Single source of truth for the builtin exception taxonomy. The numeric
// class ids must match the `ly.runtime.class_id` attributes the runtime
// manifests (src/lython/runtime/modules/*.mlir) assign to the corresponding
// `LyX_New` initializers; the name-based subclass relation used by the static
// checker (PyDialectTypes) and the id-based relation compiled into the native
// support module (exception_base_class_id / exception_class_name /
// `.tb_class.*`) are all derived from this one table so they cannot drift
// apart.

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace py::exceptions {

// Base id 0 terminates the chain (only BaseException points at it).
inline constexpr std::int64_t kRootClassId = 0;

struct BuiltinExceptionInfo {
  std::int64_t classId;
  llvm::StringLiteral name;
  std::int64_t baseClassId;
  // Manifest contract; not always under builtins (CancelledError is asyncio's,
  // UnsupportedOperation is _io's).
  llvm::StringLiteral contract;
};

inline constexpr BuiltinExceptionInfo kBuiltinExceptions[] = {
    {5, llvm::StringLiteral("BaseException"), kRootClassId,
     llvm::StringLiteral("builtins.BaseException")},
    {50, llvm::StringLiteral("Exception"), 5,
     llvm::StringLiteral("builtins.Exception")},
    {51, llvm::StringLiteral("RuntimeError"), 50,
     llvm::StringLiteral("builtins.RuntimeError")},
    {52, llvm::StringLiteral("TypeError"), 50,
     llvm::StringLiteral("builtins.TypeError")},
    {53, llvm::StringLiteral("ValueError"), 50,
     llvm::StringLiteral("builtins.ValueError")},
    {54, llvm::StringLiteral("KeyError"), 60,
     llvm::StringLiteral("builtins.KeyError")},
    {55, llvm::StringLiteral("IndexError"), 60,
     llvm::StringLiteral("builtins.IndexError")},
    {56, llvm::StringLiteral("AssertionError"), 50,
     llvm::StringLiteral("builtins.AssertionError")},
    {57, llvm::StringLiteral("StopIteration"), 50,
     llvm::StringLiteral("builtins.StopIteration")},
    {58, llvm::StringLiteral("StopAsyncIteration"), 50,
     llvm::StringLiteral("builtins.StopAsyncIteration")},
    {59, llvm::StringLiteral("ArithmeticError"), 50,
     llvm::StringLiteral("builtins.ArithmeticError")},
    {60, llvm::StringLiteral("LookupError"), 50,
     llvm::StringLiteral("builtins.LookupError")},
    {61, llvm::StringLiteral("ZeroDivisionError"), 59,
     llvm::StringLiteral("builtins.ZeroDivisionError")},
    {62, llvm::StringLiteral("CancelledError"), 5,
     llvm::StringLiteral("asyncio.CancelledError")},
    {64, llvm::StringLiteral("SystemExit"), 5,
     llvm::StringLiteral("builtins.SystemExit")},
    {68, llvm::StringLiteral("GeneratorExit"), 5,
     llvm::StringLiteral("builtins.GeneratorExit")},
    {66, llvm::StringLiteral("OSError"), 50,
     llvm::StringLiteral("builtins.OSError")},
    {67, llvm::StringLiteral("FileNotFoundError"), 66,
     llvm::StringLiteral("builtins.FileNotFoundError")},
    {69, llvm::StringLiteral("UnsupportedOperation"), 66,
     llvm::StringLiteral("_io.UnsupportedOperation")},
};

inline const BuiltinExceptionInfo *findByName(llvm::StringRef name) {
  for (const BuiltinExceptionInfo &entry : kBuiltinExceptions)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

inline const BuiltinExceptionInfo *findByClassId(std::int64_t classId) {
  for (const BuiltinExceptionInfo &entry : kBuiltinExceptions)
    if (entry.classId == classId)
      return &entry;
  return nullptr;
}

// Leaf name of the direct base class; empty for BaseException and for names
// outside the builtin taxonomy.
inline llvm::StringRef builtinExceptionBaseName(llvm::StringRef name) {
  const BuiltinExceptionInfo *entry = findByName(name);
  if (!entry || entry->baseClassId == kRootClassId)
    return {};
  const BuiltinExceptionInfo *base = findByClassId(entry->baseClassId);
  return base ? llvm::StringRef(base->name) : llvm::StringRef();
}

} // namespace py::exceptions
