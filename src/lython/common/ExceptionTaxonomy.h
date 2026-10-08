#pragma once

// Single source of truth for the builtin exception taxonomy: the static
// checker's subclass relation (PyDialectTypes) and every runtime exception
// type object's base (lowering/Common/TypeObjects.cpp) are derived from this
// one table, so they cannot drift apart. A class is its contract; its type
// object is named after it.

#include "llvm/ADT/StringRef.h"


#include <cstdint>

namespace py::exceptions {

struct BuiltinExceptionInfo {
  // Manifest contract; not always under builtins (UnsupportedOperation is
  // _io's). It is the class's identity: its type object is named after it.
  llvm::StringLiteral contract;
  llvm::StringLiteral name;
  // The primary base's contract; empty for the root (BaseException, whose
  // type object's base is object's).
  llvm::StringLiteral baseContract;
};

inline constexpr BuiltinExceptionInfo kBuiltinExceptions[] = {
    {llvm::StringLiteral("builtins.BaseException"), llvm::StringLiteral("BaseException"),
     llvm::StringLiteral("")},
    {llvm::StringLiteral("builtins.Exception"), llvm::StringLiteral("Exception"),
     llvm::StringLiteral("builtins.BaseException")},
    {llvm::StringLiteral("builtins.RuntimeError"), llvm::StringLiteral("RuntimeError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.TypeError"), llvm::StringLiteral("TypeError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.ValueError"), llvm::StringLiteral("ValueError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.KeyError"), llvm::StringLiteral("KeyError"),
     llvm::StringLiteral("builtins.LookupError")},
    {llvm::StringLiteral("builtins.IndexError"), llvm::StringLiteral("IndexError"),
     llvm::StringLiteral("builtins.LookupError")},
    {llvm::StringLiteral("builtins.AssertionError"), llvm::StringLiteral("AssertionError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.StopIteration"), llvm::StringLiteral("StopIteration"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.StopAsyncIteration"), llvm::StringLiteral("StopAsyncIteration"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.ArithmeticError"), llvm::StringLiteral("ArithmeticError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.LookupError"), llvm::StringLiteral("LookupError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.ZeroDivisionError"), llvm::StringLiteral("ZeroDivisionError"),
     llvm::StringLiteral("builtins.ArithmeticError")},
    {llvm::StringLiteral("builtins.SystemExit"), llvm::StringLiteral("SystemExit"),
     llvm::StringLiteral("builtins.BaseException")},
    {llvm::StringLiteral("builtins.GeneratorExit"), llvm::StringLiteral("GeneratorExit"),
     llvm::StringLiteral("builtins.BaseException")},
    {llvm::StringLiteral("builtins.OSError"), llvm::StringLiteral("OSError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.FileNotFoundError"), llvm::StringLiteral("FileNotFoundError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("_io.UnsupportedOperation"), llvm::StringLiteral("UnsupportedOperation"),
     llvm::StringLiteral("builtins.OSError")},
    // CPython 3.14 completion (Wave 1).
    {llvm::StringLiteral("builtins.KeyboardInterrupt"), llvm::StringLiteral("KeyboardInterrupt"),
     llvm::StringLiteral("builtins.BaseException")},
    {llvm::StringLiteral("builtins.BaseExceptionGroup"), llvm::StringLiteral("BaseExceptionGroup"),
     llvm::StringLiteral("builtins.BaseException")},
    // ExceptionGroup's second base (Exception) lives in
    // kBuiltinExceptionExtraEdges; the primary chain keeps
    // BaseExceptionGroup so except BaseExceptionGroup matches by walk.
    {llvm::StringLiteral("builtins.ExceptionGroup"), llvm::StringLiteral("ExceptionGroup"),
     llvm::StringLiteral("builtins.BaseExceptionGroup")},
    {llvm::StringLiteral("builtins.FloatingPointError"), llvm::StringLiteral("FloatingPointError"),
     llvm::StringLiteral("builtins.ArithmeticError")},
    {llvm::StringLiteral("builtins.OverflowError"), llvm::StringLiteral("OverflowError"),
     llvm::StringLiteral("builtins.ArithmeticError")},
    {llvm::StringLiteral("builtins.BufferError"), llvm::StringLiteral("BufferError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.EOFError"), llvm::StringLiteral("EOFError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.ImportError"), llvm::StringLiteral("ImportError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.ModuleNotFoundError"), llvm::StringLiteral("ModuleNotFoundError"),
     llvm::StringLiteral("builtins.ImportError")},
    {llvm::StringLiteral("builtins.MemoryError"), llvm::StringLiteral("MemoryError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.NameError"), llvm::StringLiteral("NameError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.UnboundLocalError"), llvm::StringLiteral("UnboundLocalError"),
     llvm::StringLiteral("builtins.NameError")},
    {llvm::StringLiteral("builtins.AttributeError"), llvm::StringLiteral("AttributeError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.ReferenceError"), llvm::StringLiteral("ReferenceError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.NotImplementedError"), llvm::StringLiteral("NotImplementedError"),
     llvm::StringLiteral("builtins.RuntimeError")},
    {llvm::StringLiteral("builtins.RecursionError"), llvm::StringLiteral("RecursionError"),
     llvm::StringLiteral("builtins.RuntimeError")},
    {llvm::StringLiteral("builtins.PythonFinalizationError"), llvm::StringLiteral("PythonFinalizationError"),
     llvm::StringLiteral("builtins.RuntimeError")},
    {llvm::StringLiteral("builtins.SyntaxError"), llvm::StringLiteral("SyntaxError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.IndentationError"), llvm::StringLiteral("IndentationError"),
     llvm::StringLiteral("builtins.SyntaxError")},
    {llvm::StringLiteral("builtins.TabError"), llvm::StringLiteral("TabError"),
     llvm::StringLiteral("builtins.IndentationError")},
    {llvm::StringLiteral("builtins.SystemError"), llvm::StringLiteral("SystemError"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.UnicodeError"), llvm::StringLiteral("UnicodeError"),
     llvm::StringLiteral("builtins.ValueError")},
    {llvm::StringLiteral("builtins.UnicodeDecodeError"), llvm::StringLiteral("UnicodeDecodeError"),
     llvm::StringLiteral("builtins.UnicodeError")},
    {llvm::StringLiteral("builtins.UnicodeEncodeError"), llvm::StringLiteral("UnicodeEncodeError"),
     llvm::StringLiteral("builtins.UnicodeError")},
    {llvm::StringLiteral("builtins.UnicodeTranslateError"), llvm::StringLiteral("UnicodeTranslateError"),
     llvm::StringLiteral("builtins.UnicodeError")},
    {llvm::StringLiteral("builtins.Warning"), llvm::StringLiteral("Warning"),
     llvm::StringLiteral("builtins.Exception")},
    {llvm::StringLiteral("builtins.BytesWarning"), llvm::StringLiteral("BytesWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.DeprecationWarning"), llvm::StringLiteral("DeprecationWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.EncodingWarning"), llvm::StringLiteral("EncodingWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.FutureWarning"), llvm::StringLiteral("FutureWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.ImportWarning"), llvm::StringLiteral("ImportWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.PendingDeprecationWarning"), llvm::StringLiteral("PendingDeprecationWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.ResourceWarning"), llvm::StringLiteral("ResourceWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.RuntimeWarning"), llvm::StringLiteral("RuntimeWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.SyntaxWarning"), llvm::StringLiteral("SyntaxWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.UnicodeWarning"), llvm::StringLiteral("UnicodeWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.UserWarning"), llvm::StringLiteral("UserWarning"),
     llvm::StringLiteral("builtins.Warning")},
    {llvm::StringLiteral("builtins.BlockingIOError"), llvm::StringLiteral("BlockingIOError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.ChildProcessError"), llvm::StringLiteral("ChildProcessError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.ConnectionError"), llvm::StringLiteral("ConnectionError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.BrokenPipeError"), llvm::StringLiteral("BrokenPipeError"),
     llvm::StringLiteral("builtins.ConnectionError")},
    {llvm::StringLiteral("builtins.ConnectionAbortedError"), llvm::StringLiteral("ConnectionAbortedError"),
     llvm::StringLiteral("builtins.ConnectionError")},
    {llvm::StringLiteral("builtins.ConnectionRefusedError"), llvm::StringLiteral("ConnectionRefusedError"),
     llvm::StringLiteral("builtins.ConnectionError")},
    {llvm::StringLiteral("builtins.ConnectionResetError"), llvm::StringLiteral("ConnectionResetError"),
     llvm::StringLiteral("builtins.ConnectionError")},
    {llvm::StringLiteral("builtins.FileExistsError"), llvm::StringLiteral("FileExistsError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.InterruptedError"), llvm::StringLiteral("InterruptedError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.IsADirectoryError"), llvm::StringLiteral("IsADirectoryError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.NotADirectoryError"), llvm::StringLiteral("NotADirectoryError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.PermissionError"), llvm::StringLiteral("PermissionError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.ProcessLookupError"), llvm::StringLiteral("ProcessLookupError"),
     llvm::StringLiteral("builtins.OSError")},
    {llvm::StringLiteral("builtins.TimeoutError"), llvm::StringLiteral("TimeoutError"),
     llvm::StringLiteral("builtins.OSError")},
};

// Secondary subclass edges for the multiple-inheritance members of the
// taxonomy (the id chain in BuiltinExceptionInfo is single-parent).
// ExceptionGroup is both a BaseExceptionGroup and an Exception; matchers
// (static name walk and the generated LyType_IsSubtype) must accept
// these edges in addition to the primary chain.
struct BuiltinExceptionExtraEdge {
  llvm::StringLiteral contract;
  llvm::StringLiteral extraBaseContract;
};

inline constexpr BuiltinExceptionExtraEdge kBuiltinExceptionExtraEdges[] = {
    {llvm::StringLiteral("builtins.ExceptionGroup"), llvm::StringLiteral("builtins.Exception")}, // ExceptionGroup -> Exception
};

// OSError itself: the fallback for an errno with no dedicated subclass.
inline constexpr llvm::StringLiteral kOSErrorContract{"builtins.OSError"};

// errno -> OSError-subclass mapping (CPython exceptions.c oserror_use_init
// dispatch table). Values are per-libc: the common POSIX subset shares
// numbers between the BSD family (Darwin) and Linux, and the socket/async
// members diverge. wasi-libc numbers them its own way, sharing nothing with
// either (ENOENT is 44). The runtime reads it through
// LyHost_OSErrorClass, which the OS support cluster
// (lowering/Common/OsSupportBuilder.cpp) compiles into a select chain against
// the target's errno numbering.
enum class ErrnoNumbering { Linux, BSD, WASI };

struct OSErrorErrnoMapping {
  llvm::StringLiteral posixName;
  int darwinValue; // BSD family
  int linuxValue;
  int wasiValue;
  llvm::StringLiteral contract;

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
    {llvm::StringLiteral("EPERM"), 1, 1, 63, llvm::StringLiteral("builtins.PermissionError")},     // PermissionError
    {llvm::StringLiteral("ENOENT"), 2, 2, 44, llvm::StringLiteral("builtins.FileNotFoundError")},     // FileNotFoundError
    {llvm::StringLiteral("ESRCH"), 3, 3, 71, llvm::StringLiteral("builtins.ProcessLookupError")},     // ProcessLookupError
    {llvm::StringLiteral("EINTR"), 4, 4, 27, llvm::StringLiteral("builtins.InterruptedError")},     // InterruptedError
    {llvm::StringLiteral("ECHILD"), 10, 10, 12, llvm::StringLiteral("builtins.ChildProcessError")},  // ChildProcessError
    {llvm::StringLiteral("EACCES"), 13, 13, 2, llvm::StringLiteral("builtins.PermissionError")},   // PermissionError
    {llvm::StringLiteral("EEXIST"), 17, 17, 20, llvm::StringLiteral("builtins.FileExistsError")},  // FileExistsError
    {llvm::StringLiteral("ENOTDIR"), 20, 20, 54, llvm::StringLiteral("builtins.NotADirectoryError")}, // NotADirectoryError
    {llvm::StringLiteral("EISDIR"), 21, 21, 31, llvm::StringLiteral("builtins.IsADirectoryError")},  // IsADirectoryError
    {llvm::StringLiteral("EPIPE"), 32, 32, 64, llvm::StringLiteral("builtins.BrokenPipeError")},   // BrokenPipeError
    {llvm::StringLiteral("EAGAIN"), 35, 11, 6, llvm::StringLiteral("builtins.BlockingIOError")},   // BlockingIOError
    {llvm::StringLiteral("EINPROGRESS"), 36, 115, 26, llvm::StringLiteral("builtins.BlockingIOError")},
    {llvm::StringLiteral("EALREADY"), 37, 114, 7, llvm::StringLiteral("builtins.BlockingIOError")},
    {llvm::StringLiteral("ECONNABORTED"), 53, 103, 13, llvm::StringLiteral("builtins.ConnectionAbortedError")},
    {llvm::StringLiteral("ECONNRESET"), 54, 104, 15, llvm::StringLiteral("builtins.ConnectionResetError")},
    {llvm::StringLiteral("ESHUTDOWN"), 58, 108, 140, llvm::StringLiteral("builtins.BrokenPipeError")},
    {llvm::StringLiteral("ETIMEDOUT"), 60, 110, 73, llvm::StringLiteral("builtins.TimeoutError")}, // TimeoutError
    {llvm::StringLiteral("ECONNREFUSED"), 61, 111, 14, llvm::StringLiteral("builtins.ConnectionRefusedError")},
};

inline const BuiltinExceptionInfo *findByName(llvm::StringRef name) {
  for (const BuiltinExceptionInfo &entry : kBuiltinExceptions)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

inline const BuiltinExceptionInfo *findByContract(llvm::StringRef contract) {
  for (const BuiltinExceptionInfo &entry : kBuiltinExceptions)
    if (entry.contract == contract)
      return &entry;
  return nullptr;
}

// Subclass relation over contracts, primary chain plus extra edges.
inline bool isBuiltinExceptionSubclassContract(llvm::StringRef contract,
                                               llvm::StringRef superContract) {
  llvm::StringRef current = contract;
  // The taxonomy is acyclic and shallow; the bound only guards table bugs.
  for (unsigned depth = 0; depth < 16 && !current.empty(); ++depth) {
    if (current == superContract)
      return true;
    for (const BuiltinExceptionExtraEdge &edge : kBuiltinExceptionExtraEdges)
      if (edge.contract == current &&
          isBuiltinExceptionSubclassContract(edge.extraBaseContract,
                                             superContract))
        return true;
    const BuiltinExceptionInfo *entry = findByContract(current);
    if (!entry)
      return false;
    current = entry->baseContract;
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
  return isBuiltinExceptionSubclassContract(sub->contract, super->contract);
}

} // namespace py::exceptions
