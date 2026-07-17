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
    // CPython 3.14 completion (Wave 1). Ids 100+ leave room below for
    // non-exception runtime classes; user-defined exception classes start at
    // 2^32 (kSourceClassIdBase), so every builtin id must stay below that.
    {100, llvm::StringLiteral("KeyboardInterrupt"), 5,
     llvm::StringLiteral("builtins.KeyboardInterrupt")},
    {101, llvm::StringLiteral("BaseExceptionGroup"), 5,
     llvm::StringLiteral("builtins.BaseExceptionGroup")},
    // ExceptionGroup's second base (Exception) lives in
    // kBuiltinExceptionExtraEdges; the primary chain keeps
    // BaseExceptionGroup so except BaseExceptionGroup matches by walk.
    {102, llvm::StringLiteral("ExceptionGroup"), 101,
     llvm::StringLiteral("builtins.ExceptionGroup")},
    {103, llvm::StringLiteral("FloatingPointError"), 59,
     llvm::StringLiteral("builtins.FloatingPointError")},
    {104, llvm::StringLiteral("OverflowError"), 59,
     llvm::StringLiteral("builtins.OverflowError")},
    {105, llvm::StringLiteral("BufferError"), 50,
     llvm::StringLiteral("builtins.BufferError")},
    {106, llvm::StringLiteral("EOFError"), 50,
     llvm::StringLiteral("builtins.EOFError")},
    {107, llvm::StringLiteral("ImportError"), 50,
     llvm::StringLiteral("builtins.ImportError")},
    {108, llvm::StringLiteral("ModuleNotFoundError"), 107,
     llvm::StringLiteral("builtins.ModuleNotFoundError")},
    {109, llvm::StringLiteral("MemoryError"), 50,
     llvm::StringLiteral("builtins.MemoryError")},
    {110, llvm::StringLiteral("NameError"), 50,
     llvm::StringLiteral("builtins.NameError")},
    {111, llvm::StringLiteral("UnboundLocalError"), 110,
     llvm::StringLiteral("builtins.UnboundLocalError")},
    {112, llvm::StringLiteral("AttributeError"), 50,
     llvm::StringLiteral("builtins.AttributeError")},
    {113, llvm::StringLiteral("ReferenceError"), 50,
     llvm::StringLiteral("builtins.ReferenceError")},
    {114, llvm::StringLiteral("NotImplementedError"), 51,
     llvm::StringLiteral("builtins.NotImplementedError")},
    {115, llvm::StringLiteral("RecursionError"), 51,
     llvm::StringLiteral("builtins.RecursionError")},
    {116, llvm::StringLiteral("PythonFinalizationError"), 51,
     llvm::StringLiteral("builtins.PythonFinalizationError")},
    {117, llvm::StringLiteral("SyntaxError"), 50,
     llvm::StringLiteral("builtins.SyntaxError")},
    {118, llvm::StringLiteral("IndentationError"), 117,
     llvm::StringLiteral("builtins.IndentationError")},
    {119, llvm::StringLiteral("TabError"), 118,
     llvm::StringLiteral("builtins.TabError")},
    {120, llvm::StringLiteral("SystemError"), 50,
     llvm::StringLiteral("builtins.SystemError")},
    {121, llvm::StringLiteral("UnicodeError"), 53,
     llvm::StringLiteral("builtins.UnicodeError")},
    {122, llvm::StringLiteral("UnicodeDecodeError"), 121,
     llvm::StringLiteral("builtins.UnicodeDecodeError")},
    {123, llvm::StringLiteral("UnicodeEncodeError"), 121,
     llvm::StringLiteral("builtins.UnicodeEncodeError")},
    {124, llvm::StringLiteral("UnicodeTranslateError"), 121,
     llvm::StringLiteral("builtins.UnicodeTranslateError")},
    {125, llvm::StringLiteral("Warning"), 50,
     llvm::StringLiteral("builtins.Warning")},
    {126, llvm::StringLiteral("BytesWarning"), 125,
     llvm::StringLiteral("builtins.BytesWarning")},
    {127, llvm::StringLiteral("DeprecationWarning"), 125,
     llvm::StringLiteral("builtins.DeprecationWarning")},
    {128, llvm::StringLiteral("EncodingWarning"), 125,
     llvm::StringLiteral("builtins.EncodingWarning")},
    {129, llvm::StringLiteral("FutureWarning"), 125,
     llvm::StringLiteral("builtins.FutureWarning")},
    {130, llvm::StringLiteral("ImportWarning"), 125,
     llvm::StringLiteral("builtins.ImportWarning")},
    {131, llvm::StringLiteral("PendingDeprecationWarning"), 125,
     llvm::StringLiteral("builtins.PendingDeprecationWarning")},
    {132, llvm::StringLiteral("ResourceWarning"), 125,
     llvm::StringLiteral("builtins.ResourceWarning")},
    {133, llvm::StringLiteral("RuntimeWarning"), 125,
     llvm::StringLiteral("builtins.RuntimeWarning")},
    {134, llvm::StringLiteral("SyntaxWarning"), 125,
     llvm::StringLiteral("builtins.SyntaxWarning")},
    {135, llvm::StringLiteral("UnicodeWarning"), 125,
     llvm::StringLiteral("builtins.UnicodeWarning")},
    {136, llvm::StringLiteral("UserWarning"), 125,
     llvm::StringLiteral("builtins.UserWarning")},
    {137, llvm::StringLiteral("BlockingIOError"), 66,
     llvm::StringLiteral("builtins.BlockingIOError")},
    {138, llvm::StringLiteral("ChildProcessError"), 66,
     llvm::StringLiteral("builtins.ChildProcessError")},
    {139, llvm::StringLiteral("ConnectionError"), 66,
     llvm::StringLiteral("builtins.ConnectionError")},
    {140, llvm::StringLiteral("BrokenPipeError"), 139,
     llvm::StringLiteral("builtins.BrokenPipeError")},
    {141, llvm::StringLiteral("ConnectionAbortedError"), 139,
     llvm::StringLiteral("builtins.ConnectionAbortedError")},
    {142, llvm::StringLiteral("ConnectionRefusedError"), 139,
     llvm::StringLiteral("builtins.ConnectionRefusedError")},
    {143, llvm::StringLiteral("ConnectionResetError"), 139,
     llvm::StringLiteral("builtins.ConnectionResetError")},
    {144, llvm::StringLiteral("FileExistsError"), 66,
     llvm::StringLiteral("builtins.FileExistsError")},
    {145, llvm::StringLiteral("InterruptedError"), 66,
     llvm::StringLiteral("builtins.InterruptedError")},
    {146, llvm::StringLiteral("IsADirectoryError"), 66,
     llvm::StringLiteral("builtins.IsADirectoryError")},
    {147, llvm::StringLiteral("NotADirectoryError"), 66,
     llvm::StringLiteral("builtins.NotADirectoryError")},
    {148, llvm::StringLiteral("PermissionError"), 66,
     llvm::StringLiteral("builtins.PermissionError")},
    {149, llvm::StringLiteral("ProcessLookupError"), 66,
     llvm::StringLiteral("builtins.ProcessLookupError")},
    {150, llvm::StringLiteral("TimeoutError"), 66,
     llvm::StringLiteral("builtins.TimeoutError")},
};

// Secondary subclass edges for the multiple-inheritance members of the
// taxonomy (the id chain in BuiltinExceptionInfo is single-parent).
// ExceptionGroup is both a BaseExceptionGroup and an Exception; matchers
// (static name walk and the generated LyEH_ClassIdMatches) must accept
// these edges in addition to the primary chain.
struct BuiltinExceptionExtraEdge {
  std::int64_t classId;
  std::int64_t extraBaseClassId;
};

inline constexpr BuiltinExceptionExtraEdge kBuiltinExceptionExtraEdges[] = {
    {102, 50}, // ExceptionGroup -> Exception
};

// errno -> OSError-subclass mapping (CPython exceptions.c oserror_use_init
// dispatch table). Values are per-libc: the common POSIX subset shares
// numbers, the socket/async members diverge between the BSD family (Darwin)
// and Linux. Nothing reads errno at runtime yet (there is no per-target
// errno accessor shim); this table is the single source of truth for when
// that wiring lands.
struct OSErrorErrnoMapping {
  llvm::StringLiteral posixName;
  int darwinValue; // BSD family
  int linuxValue;
  std::int64_t classId;
};

inline constexpr OSErrorErrnoMapping kOSErrorErrnoMap[] = {
    {llvm::StringLiteral("EPERM"), 1, 1, 148},        // PermissionError
    {llvm::StringLiteral("ENOENT"), 2, 2, 67},        // FileNotFoundError
    {llvm::StringLiteral("ESRCH"), 3, 3, 149},        // ProcessLookupError
    {llvm::StringLiteral("EINTR"), 4, 4, 145},        // InterruptedError
    {llvm::StringLiteral("ECHILD"), 10, 10, 138},     // ChildProcessError
    {llvm::StringLiteral("EACCES"), 13, 13, 148},     // PermissionError
    {llvm::StringLiteral("EEXIST"), 17, 17, 144},     // FileExistsError
    {llvm::StringLiteral("ENOTDIR"), 20, 20, 147},    // NotADirectoryError
    {llvm::StringLiteral("EISDIR"), 21, 21, 146},     // IsADirectoryError
    {llvm::StringLiteral("EPIPE"), 32, 32, 140},      // BrokenPipeError
    {llvm::StringLiteral("EAGAIN"), 35, 11, 137},     // BlockingIOError
    {llvm::StringLiteral("EINPROGRESS"), 36, 115, 137},
    {llvm::StringLiteral("EALREADY"), 37, 114, 137},
    {llvm::StringLiteral("ECONNABORTED"), 53, 103, 141},
    {llvm::StringLiteral("ECONNRESET"), 54, 104, 143},
    {llvm::StringLiteral("ESHUTDOWN"), 58, 108, 140},
    {llvm::StringLiteral("ETIMEDOUT"), 60, 110, 150}, // TimeoutError
    {llvm::StringLiteral("ECONNREFUSED"), 61, 111, 142},
};

inline std::int64_t oserrorSubclassForErrno(int errnoValue, bool isLinux) {
  for (const OSErrorErrnoMapping &entry : kOSErrorErrnoMap)
    if ((isLinux ? entry.linuxValue : entry.darwinValue) == errnoValue)
      return entry.classId;
  return 66; // plain OSError
}

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
// outside the builtin taxonomy. Only the primary chain: multiple-inheritance
// members' extra edges are in kBuiltinExceptionExtraEdges, and subclass
// queries should go through isBuiltinExceptionSubclassName.
inline llvm::StringRef builtinExceptionBaseName(llvm::StringRef name) {
  const BuiltinExceptionInfo *entry = findByName(name);
  if (!entry || entry->baseClassId == kRootClassId)
    return {};
  const BuiltinExceptionInfo *base = findByClassId(entry->baseClassId);
  return base ? llvm::StringRef(base->name) : llvm::StringRef();
}

// Subclass relation over class ids, primary chain plus extra edges.
inline bool isBuiltinExceptionSubclassId(std::int64_t classId,
                                         std::int64_t superClassId) {
  std::int64_t current = classId;
  // The taxonomy is acyclic and shallow; the bound only guards table bugs.
  for (unsigned depth = 0; depth < 16 && current != kRootClassId; ++depth) {
    if (current == superClassId)
      return true;
    for (const BuiltinExceptionExtraEdge &edge : kBuiltinExceptionExtraEdges)
      if (edge.classId == current &&
          isBuiltinExceptionSubclassId(edge.extraBaseClassId, superClassId))
        return true;
    const BuiltinExceptionInfo *entry = findByClassId(current);
    if (!entry)
      return false;
    current = entry->baseClassId;
  }
  return false;
}

// Subclass relation over leaf names; false when either name is outside the
// builtin taxonomy.
inline bool isBuiltinExceptionSubclassName(llvm::StringRef name,
                                           llvm::StringRef superName) {
  const BuiltinExceptionInfo *sub = findByName(name);
  const BuiltinExceptionInfo *super = findByName(superName);
  if (!sub || !super)
    return false;
  return isBuiltinExceptionSubclassId(sub->classId, super->classId);
}

} // namespace py::exceptions
