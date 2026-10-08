#pragma once
// The runtime classes, by the name that IS their identity. A class's number --
// word 1 of its header, what every class test compares, what a raise names --
// is its position in this list, so nothing spells a number of its own:
//
//   - the manifests write `ly.class_id_of = "builtins.ValueError"` on a
//     constant or a static header, and a constructor carries the
//     `ly.runtime.class_id` marker beside its `ly.runtime.contract`; the
//     runtime bytecode tool (RuntimeMlirBytecode.cpp) writes the numbers;
//   - C++ asks `py::class_ids::of("builtins.int")`, a constant expression.
//
// A class is added by adding its name. A name the manifests use that is not
// here is refused when the runtime is built. Program classes are numbered from
// 2^32 by the lowering (`runtimeClassIdForClass`), above all of these.
//
// ⛔ `builtins.object` is 0, the number that also means "no class" and that a
// None slot reads as. Separating the three is the None-object work; until then
// object keeps 0 so that what reads 0 as None keeps reading it.

#include <cstdint>
#include <string_view>

namespace py::class_ids {

// No class: the end of a base chain, a slot that names none.
inline constexpr std::int64_t kNoClass = 0;

inline constexpr std::string_view kRuntimeClasses[] = {
    "builtins.object",
    "builtins.int",
    "builtins.float",
    "builtins.range",
    "builtins.str",
    "builtins.BaseException",
    "builtins.function",
    "builtins.str_iterator",
    "contextlib.nullcontext",
    "types.CoroutineType",
    "builtins.list",
    "builtins.tuple",
    "builtins.dict",
    "builtins.complex",
    "types.CoroutineAwaitIterator",
    "lyrt.Counter",
    "builtins.range_iterator",
    "builtins.set",
    "builtins.bool",
    "builtins.frozenset",
    "builtins.bytes_iterator",
    "builtins.slice",
    "builtins.bytearray",
    "builtins.memoryview",
    "builtins.bytearray_iterator",
    "builtins.memory_iterator",
    "builtins.Exception",
    "builtins.RuntimeError",
    "builtins.TypeError",
    "builtins.ValueError",
    "builtins.KeyError",
    "builtins.IndexError",
    "builtins.AssertionError",
    "builtins.StopIteration",
    "builtins.StopAsyncIteration",
    "builtins.ArithmeticError",
    "builtins.LookupError",
    "builtins.ZeroDivisionError",
    "types.GeneratorType",
    "builtins.SystemExit",
    "_io.TextIOWrapper",
    "builtins.OSError",
    "builtins.FileNotFoundError",
    "builtins.GeneratorExit",
    "_io.UnsupportedOperation",
    "builtins.bytes",
    "_io.BytesIO",
    "_io.FileIO",
    "_io.StringIO",
    "builtins.KeyboardInterrupt",
    "builtins.BaseExceptionGroup",
    "builtins.ExceptionGroup",
    "builtins.FloatingPointError",
    "builtins.OverflowError",
    "builtins.BufferError",
    "builtins.EOFError",
    "builtins.ImportError",
    "builtins.ModuleNotFoundError",
    "builtins.MemoryError",
    "builtins.NameError",
    "builtins.UnboundLocalError",
    "builtins.AttributeError",
    "builtins.ReferenceError",
    "builtins.NotImplementedError",
    "builtins.RecursionError",
    "builtins.PythonFinalizationError",
    "builtins.SyntaxError",
    "builtins.IndentationError",
    "builtins.TabError",
    "builtins.SystemError",
    "builtins.UnicodeError",
    "builtins.UnicodeDecodeError",
    "builtins.UnicodeEncodeError",
    "builtins.UnicodeTranslateError",
    "builtins.Warning",
    "builtins.BytesWarning",
    "builtins.DeprecationWarning",
    "builtins.EncodingWarning",
    "builtins.FutureWarning",
    "builtins.ImportWarning",
    "builtins.PendingDeprecationWarning",
    "builtins.ResourceWarning",
    "builtins.RuntimeWarning",
    "builtins.SyntaxWarning",
    "builtins.UnicodeWarning",
    "builtins.UserWarning",
    "builtins.BlockingIOError",
    "builtins.ChildProcessError",
    "builtins.ConnectionError",
    "builtins.BrokenPipeError",
    "builtins.ConnectionAbortedError",
    "builtins.ConnectionRefusedError",
    "builtins.ConnectionResetError",
    "builtins.FileExistsError",
    "builtins.InterruptedError",
    "builtins.IsADirectoryError",
    "builtins.NotADirectoryError",
    "builtins.PermissionError",
    "builtins.ProcessLookupError",
    "builtins.TimeoutError",
    "_js.JsProxy",
};

// The number of a runtime class, or -1 for a name this table does not list
// (a constant expression wherever the name is one).
constexpr std::int64_t lookup(std::string_view contract) {
  std::int64_t position = 0;
  for (std::string_view entry : kRuntimeClasses) {
    if (entry == contract)
      return entry == "builtins.object" ? kNoClass : position + 1;
    ++position;
  }
  return -1;
}

// The same, for a name that must be listed: an unlisted one fails to compile
// where the call is a constant expression.
constexpr std::int64_t of(std::string_view contract) {
  std::int64_t id = lookup(contract);
  if (id < 0)
    throw "not a runtime class";
  return id;
}

// The name of a runtime class number, or empty.
constexpr std::string_view contractOf(std::int64_t id) {
  for (std::string_view entry : kRuntimeClasses)
    if (lookup(entry) == id)
      return entry;
  return {};
}

} // namespace py::class_ids
