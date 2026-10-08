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

#include "ClassIds.h"

#include <cstdint>

namespace py::exceptions {

// Base id 0 terminates the chain (only BaseException points at it).
inline constexpr std::int64_t kRootClassId = class_ids::kNoClass;

struct BuiltinExceptionInfo {
  std::int64_t classId;
  llvm::StringLiteral name;
  std::int64_t baseClassId;
  // Manifest contract; not always under builtins (UnsupportedOperation is
  // _io's).
  llvm::StringLiteral contract;
};

inline constexpr BuiltinExceptionInfo kBuiltinExceptions[] = {
    {class_ids::of("builtins.BaseException"), llvm::StringLiteral("BaseException"), kRootClassId,
     llvm::StringLiteral("builtins.BaseException")},
    {class_ids::of("builtins.Exception"), llvm::StringLiteral("Exception"), class_ids::of("builtins.BaseException"),
     llvm::StringLiteral("builtins.Exception")},
    {class_ids::of("builtins.RuntimeError"), llvm::StringLiteral("RuntimeError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.RuntimeError")},
    {class_ids::of("builtins.TypeError"), llvm::StringLiteral("TypeError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.TypeError")},
    {class_ids::of("builtins.ValueError"), llvm::StringLiteral("ValueError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.ValueError")},
    {class_ids::of("builtins.KeyError"), llvm::StringLiteral("KeyError"), class_ids::of("builtins.LookupError"),
     llvm::StringLiteral("builtins.KeyError")},
    {class_ids::of("builtins.IndexError"), llvm::StringLiteral("IndexError"), class_ids::of("builtins.LookupError"),
     llvm::StringLiteral("builtins.IndexError")},
    {class_ids::of("builtins.AssertionError"), llvm::StringLiteral("AssertionError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.AssertionError")},
    {class_ids::of("builtins.StopIteration"), llvm::StringLiteral("StopIteration"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.StopIteration")},
    {class_ids::of("builtins.StopAsyncIteration"), llvm::StringLiteral("StopAsyncIteration"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.StopAsyncIteration")},
    {class_ids::of("builtins.ArithmeticError"), llvm::StringLiteral("ArithmeticError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.ArithmeticError")},
    {class_ids::of("builtins.LookupError"), llvm::StringLiteral("LookupError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.LookupError")},
    {class_ids::of("builtins.ZeroDivisionError"), llvm::StringLiteral("ZeroDivisionError"), class_ids::of("builtins.ArithmeticError"),
     llvm::StringLiteral("builtins.ZeroDivisionError")},
    {class_ids::of("builtins.SystemExit"), llvm::StringLiteral("SystemExit"), class_ids::of("builtins.BaseException"),
     llvm::StringLiteral("builtins.SystemExit")},
    {class_ids::of("builtins.GeneratorExit"), llvm::StringLiteral("GeneratorExit"), class_ids::of("builtins.BaseException"),
     llvm::StringLiteral("builtins.GeneratorExit")},
    {class_ids::of("builtins.OSError"), llvm::StringLiteral("OSError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.OSError")},
    {class_ids::of("builtins.FileNotFoundError"), llvm::StringLiteral("FileNotFoundError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.FileNotFoundError")},
    {class_ids::of("_io.UnsupportedOperation"), llvm::StringLiteral("UnsupportedOperation"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("_io.UnsupportedOperation")},
    // CPython 3.14 completion (Wave 1). Numbers come from ClassIds.h by
    // name; user-defined classes start at 2^32 (kSourceClassIdBase), above
    // every one of them.
    {class_ids::of("builtins.KeyboardInterrupt"), llvm::StringLiteral("KeyboardInterrupt"), class_ids::of("builtins.BaseException"),
     llvm::StringLiteral("builtins.KeyboardInterrupt")},
    {class_ids::of("builtins.BaseExceptionGroup"), llvm::StringLiteral("BaseExceptionGroup"), class_ids::of("builtins.BaseException"),
     llvm::StringLiteral("builtins.BaseExceptionGroup")},
    // ExceptionGroup's second base (Exception) lives in
    // kBuiltinExceptionExtraEdges; the primary chain keeps
    // BaseExceptionGroup so except BaseExceptionGroup matches by walk.
    {class_ids::of("builtins.ExceptionGroup"), llvm::StringLiteral("ExceptionGroup"), class_ids::of("builtins.BaseExceptionGroup"),
     llvm::StringLiteral("builtins.ExceptionGroup")},
    {class_ids::of("builtins.FloatingPointError"), llvm::StringLiteral("FloatingPointError"), class_ids::of("builtins.ArithmeticError"),
     llvm::StringLiteral("builtins.FloatingPointError")},
    {class_ids::of("builtins.OverflowError"), llvm::StringLiteral("OverflowError"), class_ids::of("builtins.ArithmeticError"),
     llvm::StringLiteral("builtins.OverflowError")},
    {class_ids::of("builtins.BufferError"), llvm::StringLiteral("BufferError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.BufferError")},
    {class_ids::of("builtins.EOFError"), llvm::StringLiteral("EOFError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.EOFError")},
    {class_ids::of("builtins.ImportError"), llvm::StringLiteral("ImportError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.ImportError")},
    {class_ids::of("builtins.ModuleNotFoundError"), llvm::StringLiteral("ModuleNotFoundError"), class_ids::of("builtins.ImportError"),
     llvm::StringLiteral("builtins.ModuleNotFoundError")},
    {class_ids::of("builtins.MemoryError"), llvm::StringLiteral("MemoryError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.MemoryError")},
    {class_ids::of("builtins.NameError"), llvm::StringLiteral("NameError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.NameError")},
    {class_ids::of("builtins.UnboundLocalError"), llvm::StringLiteral("UnboundLocalError"), class_ids::of("builtins.NameError"),
     llvm::StringLiteral("builtins.UnboundLocalError")},
    {class_ids::of("builtins.AttributeError"), llvm::StringLiteral("AttributeError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.AttributeError")},
    {class_ids::of("builtins.ReferenceError"), llvm::StringLiteral("ReferenceError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.ReferenceError")},
    {class_ids::of("builtins.NotImplementedError"), llvm::StringLiteral("NotImplementedError"), class_ids::of("builtins.RuntimeError"),
     llvm::StringLiteral("builtins.NotImplementedError")},
    {class_ids::of("builtins.RecursionError"), llvm::StringLiteral("RecursionError"), class_ids::of("builtins.RuntimeError"),
     llvm::StringLiteral("builtins.RecursionError")},
    {class_ids::of("builtins.PythonFinalizationError"), llvm::StringLiteral("PythonFinalizationError"), class_ids::of("builtins.RuntimeError"),
     llvm::StringLiteral("builtins.PythonFinalizationError")},
    {class_ids::of("builtins.SyntaxError"), llvm::StringLiteral("SyntaxError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.SyntaxError")},
    {class_ids::of("builtins.IndentationError"), llvm::StringLiteral("IndentationError"), class_ids::of("builtins.SyntaxError"),
     llvm::StringLiteral("builtins.IndentationError")},
    {class_ids::of("builtins.TabError"), llvm::StringLiteral("TabError"), class_ids::of("builtins.IndentationError"),
     llvm::StringLiteral("builtins.TabError")},
    {class_ids::of("builtins.SystemError"), llvm::StringLiteral("SystemError"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.SystemError")},
    {class_ids::of("builtins.UnicodeError"), llvm::StringLiteral("UnicodeError"), class_ids::of("builtins.ValueError"),
     llvm::StringLiteral("builtins.UnicodeError")},
    {class_ids::of("builtins.UnicodeDecodeError"), llvm::StringLiteral("UnicodeDecodeError"), class_ids::of("builtins.UnicodeError"),
     llvm::StringLiteral("builtins.UnicodeDecodeError")},
    {class_ids::of("builtins.UnicodeEncodeError"), llvm::StringLiteral("UnicodeEncodeError"), class_ids::of("builtins.UnicodeError"),
     llvm::StringLiteral("builtins.UnicodeEncodeError")},
    {class_ids::of("builtins.UnicodeTranslateError"), llvm::StringLiteral("UnicodeTranslateError"), class_ids::of("builtins.UnicodeError"),
     llvm::StringLiteral("builtins.UnicodeTranslateError")},
    {class_ids::of("builtins.Warning"), llvm::StringLiteral("Warning"), class_ids::of("builtins.Exception"),
     llvm::StringLiteral("builtins.Warning")},
    {class_ids::of("builtins.BytesWarning"), llvm::StringLiteral("BytesWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.BytesWarning")},
    {class_ids::of("builtins.DeprecationWarning"), llvm::StringLiteral("DeprecationWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.DeprecationWarning")},
    {class_ids::of("builtins.EncodingWarning"), llvm::StringLiteral("EncodingWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.EncodingWarning")},
    {class_ids::of("builtins.FutureWarning"), llvm::StringLiteral("FutureWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.FutureWarning")},
    {class_ids::of("builtins.ImportWarning"), llvm::StringLiteral("ImportWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.ImportWarning")},
    {class_ids::of("builtins.PendingDeprecationWarning"), llvm::StringLiteral("PendingDeprecationWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.PendingDeprecationWarning")},
    {class_ids::of("builtins.ResourceWarning"), llvm::StringLiteral("ResourceWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.ResourceWarning")},
    {class_ids::of("builtins.RuntimeWarning"), llvm::StringLiteral("RuntimeWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.RuntimeWarning")},
    {class_ids::of("builtins.SyntaxWarning"), llvm::StringLiteral("SyntaxWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.SyntaxWarning")},
    {class_ids::of("builtins.UnicodeWarning"), llvm::StringLiteral("UnicodeWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.UnicodeWarning")},
    {class_ids::of("builtins.UserWarning"), llvm::StringLiteral("UserWarning"), class_ids::of("builtins.Warning"),
     llvm::StringLiteral("builtins.UserWarning")},
    {class_ids::of("builtins.BlockingIOError"), llvm::StringLiteral("BlockingIOError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.BlockingIOError")},
    {class_ids::of("builtins.ChildProcessError"), llvm::StringLiteral("ChildProcessError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.ChildProcessError")},
    {class_ids::of("builtins.ConnectionError"), llvm::StringLiteral("ConnectionError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.ConnectionError")},
    {class_ids::of("builtins.BrokenPipeError"), llvm::StringLiteral("BrokenPipeError"), class_ids::of("builtins.ConnectionError"),
     llvm::StringLiteral("builtins.BrokenPipeError")},
    {class_ids::of("builtins.ConnectionAbortedError"), llvm::StringLiteral("ConnectionAbortedError"), class_ids::of("builtins.ConnectionError"),
     llvm::StringLiteral("builtins.ConnectionAbortedError")},
    {class_ids::of("builtins.ConnectionRefusedError"), llvm::StringLiteral("ConnectionRefusedError"), class_ids::of("builtins.ConnectionError"),
     llvm::StringLiteral("builtins.ConnectionRefusedError")},
    {class_ids::of("builtins.ConnectionResetError"), llvm::StringLiteral("ConnectionResetError"), class_ids::of("builtins.ConnectionError"),
     llvm::StringLiteral("builtins.ConnectionResetError")},
    {class_ids::of("builtins.FileExistsError"), llvm::StringLiteral("FileExistsError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.FileExistsError")},
    {class_ids::of("builtins.InterruptedError"), llvm::StringLiteral("InterruptedError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.InterruptedError")},
    {class_ids::of("builtins.IsADirectoryError"), llvm::StringLiteral("IsADirectoryError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.IsADirectoryError")},
    {class_ids::of("builtins.NotADirectoryError"), llvm::StringLiteral("NotADirectoryError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.NotADirectoryError")},
    {class_ids::of("builtins.PermissionError"), llvm::StringLiteral("PermissionError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.PermissionError")},
    {class_ids::of("builtins.ProcessLookupError"), llvm::StringLiteral("ProcessLookupError"), class_ids::of("builtins.OSError"),
     llvm::StringLiteral("builtins.ProcessLookupError")},
    {class_ids::of("builtins.TimeoutError"), llvm::StringLiteral("TimeoutError"), class_ids::of("builtins.OSError"),
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
    {class_ids::of("builtins.ExceptionGroup"), class_ids::of("builtins.Exception")}, // ExceptionGroup -> Exception
};

// OSError itself: the fallback for an errno with no dedicated subclass.
inline constexpr std::int64_t kOSErrorClassId = class_ids::of("builtins.OSError");

// errno -> OSError-subclass mapping (CPython exceptions.c oserror_use_init
// dispatch table). Values are per-libc: the common POSIX subset shares
// numbers between the BSD family (Darwin) and Linux, and the socket/async
// members diverge. wasi-libc numbers them its own way, sharing nothing with
// either (ENOENT is 44). The runtime reads it through
// LyHost_OSErrorClassId, which the OS support cluster
// (lowering/Common/OsSupportBuilder.cpp) compiles into a select chain against
// the target's errno numbering.
enum class ErrnoNumbering { Linux, BSD, WASI };

struct OSErrorErrnoMapping {
  llvm::StringLiteral posixName;
  int darwinValue; // BSD family
  int linuxValue;
  int wasiValue;
  std::int64_t classId;

  constexpr int valueFor(ErrnoNumbering numbering) const {
    switch (numbering) {
    case ErrnoNumbering::BSD:
      return darwinValue;
    case ErrnoNumbering::WASI:
      return wasiValue;
    case ErrnoNumbering::Linux:
      break;
    }
    return linuxValue;
  }
};

inline constexpr OSErrorErrnoMapping kOSErrorErrnoMap[] = {
    {llvm::StringLiteral("EPERM"), 1, 1, 63, class_ids::of("builtins.PermissionError")},     // PermissionError
    {llvm::StringLiteral("ENOENT"), 2, 2, 44, class_ids::of("builtins.FileNotFoundError")},     // FileNotFoundError
    {llvm::StringLiteral("ESRCH"), 3, 3, 71, class_ids::of("builtins.ProcessLookupError")},     // ProcessLookupError
    {llvm::StringLiteral("EINTR"), 4, 4, 27, class_ids::of("builtins.InterruptedError")},     // InterruptedError
    {llvm::StringLiteral("ECHILD"), 10, 10, 12, class_ids::of("builtins.ChildProcessError")},  // ChildProcessError
    {llvm::StringLiteral("EACCES"), 13, 13, 2, class_ids::of("builtins.PermissionError")},   // PermissionError
    {llvm::StringLiteral("EEXIST"), 17, 17, 20, class_ids::of("builtins.FileExistsError")},  // FileExistsError
    {llvm::StringLiteral("ENOTDIR"), 20, 20, 54, class_ids::of("builtins.NotADirectoryError")}, // NotADirectoryError
    {llvm::StringLiteral("EISDIR"), 21, 21, 31, class_ids::of("builtins.IsADirectoryError")},  // IsADirectoryError
    {llvm::StringLiteral("EPIPE"), 32, 32, 64, class_ids::of("builtins.BrokenPipeError")},   // BrokenPipeError
    {llvm::StringLiteral("EAGAIN"), 35, 11, 6, class_ids::of("builtins.BlockingIOError")},   // BlockingIOError
    {llvm::StringLiteral("EINPROGRESS"), 36, 115, 26, class_ids::of("builtins.BlockingIOError")},
    {llvm::StringLiteral("EALREADY"), 37, 114, 7, class_ids::of("builtins.BlockingIOError")},
    {llvm::StringLiteral("ECONNABORTED"), 53, 103, 13, class_ids::of("builtins.ConnectionAbortedError")},
    {llvm::StringLiteral("ECONNRESET"), 54, 104, 15, class_ids::of("builtins.ConnectionResetError")},
    {llvm::StringLiteral("ESHUTDOWN"), 58, 108, 140, class_ids::of("builtins.BrokenPipeError")},
    {llvm::StringLiteral("ETIMEDOUT"), 60, 110, 73, class_ids::of("builtins.TimeoutError")}, // TimeoutError
    {llvm::StringLiteral("ECONNREFUSED"), 61, 111, 14, class_ids::of("builtins.ConnectionRefusedError")},
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
