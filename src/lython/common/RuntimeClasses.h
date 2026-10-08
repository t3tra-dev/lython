#pragma once
// The runtime classes, by the name that IS their identity. An object's header
// word 1 is the address of its class's type object (lowering/Common/
// TypeObjects.h), which is named after the class: `__ly_type.builtins.int`.
// Nothing numbers a class.
//
//   - the manifests write `ly.class_of = "builtins.ValueError"` on a
//     constant or a static header, and a constructor carries the
//     `ly.runtime.class` marker beside its `ly.runtime.contract`; the
//     lowering makes each the class's type-object address;
//   - C++ asks `type_objects::classWord(builder, loc, "builtins.int")`.
//
// A class is added by adding its name. A name the manifests use that is not
// here is refused when the runtime is built.

#include <string_view>

namespace py::runtime_classes {

inline constexpr std::string_view kRuntimeClasses[] = {
    "builtins.object",
    "types.NoneType",
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

// Whether a name is a runtime class.
constexpr bool isListed(std::string_view contract) {
  for (std::string_view entry : kRuntimeClasses)
    if (entry == contract)
      return true;
  return false;
}

} // namespace py::runtime_classes
