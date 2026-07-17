// Contract manifest for `builtins`.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// Signatures follow typeshed 1:1 where the static surface supports them;
// classes are grouped as typeshed declares them. Runtime implementations are
// co-located below (modules/<name>.mlir = CPython Modules/: one file carries
// the module's contracts, lowering strategies, and native implementations).
//
// Conventions (shared by all contract manifests):
//   - `!py.contract<"builtins.int">` is a nominal manifest contract.
//   - `!py.contract<"$T">` names a generic parameter from `ly.typing.params`.
//   - Method contracts are Callable terms in `method_contracts`; hand-written
//     manifests may use the typeshed-shaped `!py.protocol<"Callable", ...>`
//     spelling with `!py.callable<...>` for nested/variadic terms.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["types.NoneType", "builtins.object", "builtins.bool", "builtins.BaseException", "builtins.int", "builtins.str", "builtins.Exception", "builtins.RuntimeError", "builtins.TypeError", "builtins.ValueError", "builtins.ArithmeticError", "builtins.LookupError", "builtins.ZeroDivisionError", "builtins.KeyError", "builtins.IndexError", "builtins.AssertionError", "builtins.StopIteration", "builtins.StopAsyncIteration", "builtins.SystemExit", "builtins.GeneratorExit", "builtins.OSError", "builtins.FileNotFoundError", "asyncio.CancelledError", "builtins.float", "builtins.bytes", "builtins.list", "builtins.tuple", "builtins.dict", "builtins.set", "builtins.range", "builtins.range_iterator", "builtins.str_iterator", "builtins.KeyboardInterrupt", "builtins.BaseExceptionGroup", "builtins.ExceptionGroup", "builtins.FloatingPointError", "builtins.OverflowError", "builtins.BufferError", "builtins.EOFError", "builtins.ImportError", "builtins.ModuleNotFoundError", "builtins.MemoryError", "builtins.NameError", "builtins.UnboundLocalError", "builtins.AttributeError", "builtins.ReferenceError", "builtins.NotImplementedError", "builtins.RecursionError", "builtins.PythonFinalizationError", "builtins.SyntaxError", "builtins.IndentationError", "builtins.TabError", "builtins.SystemError", "builtins.UnicodeError", "builtins.UnicodeDecodeError", "builtins.UnicodeEncodeError", "builtins.UnicodeTranslateError", "builtins.Warning", "builtins.BytesWarning", "builtins.DeprecationWarning", "builtins.EncodingWarning", "builtins.FutureWarning", "builtins.ImportWarning", "builtins.PendingDeprecationWarning", "builtins.ResourceWarning", "builtins.RuntimeWarning", "builtins.SyntaxWarning", "builtins.UnicodeWarning", "builtins.UserWarning", "builtins.BlockingIOError", "builtins.ChildProcessError", "builtins.ConnectionError", "builtins.BrokenPipeError", "builtins.ConnectionAbortedError", "builtins.ConnectionRefusedError", "builtins.ConnectionResetError", "builtins.FileExistsError", "builtins.InterruptedError", "builtins.IsADirectoryError", "builtins.NotADirectoryError", "builtins.PermissionError", "builtins.ProcessLookupError", "builtins.TimeoutError"],
  // Manifest Callable contracts for builtin free functions. These are the
  // single trusted source for these signatures; the emitter's seedBuiltins
  // reads them here instead of constructing the contracts in C++.
  ly.typing.function_names = ["builtins.print", "builtins.len", "builtins.hash"],
  ly.typing.function_contracts = [
    !py.callable<[], vararg = !py.contract<"builtins.tuple", [!py.contract<"builtins.object">]>, returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.object">], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.object">], returns = [!py.contract<"builtins.int">]>
  ]
} {
  py.class @object attributes {
    ly.typing.abstract,
    method_names = ["__init__", "__new__", "__repr__", "__str__", "__bool__",
                    "__eq__", "__ne__", "__hash__", "__getattribute__",
                    "__setattr__", "__delattr__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.object">>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.str">] -> [!py.contract<"typing.Any">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.str">, !py.contract<"typing.Any">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.str">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "classmethod", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}

  py.class @type attributes {
    base_names = ["object"], ly.typing.params = ["T", "P"],
    method_names = ["__call__", "__new__", "__or__", "__ror__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"$T">>, !py.paramspec<"P">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.type">>, !py.contract<"builtins.object">] -> [!py.type<!py.contract<"builtins.object">>]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.object">>, !py.contract<"typing.Any">] -> [!py.contract<"types.UnionType">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.object">>, !py.contract<"typing.Any">] -> [!py.contract<"types.UnionType">]>
    ],
    method_kinds = ["instance", "classmethod", "instance", "instance"]
  } {}

  py.class @function attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__call__", "__get__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.function">, !py.paramspec<"P">] -> [!py.contract<"typing.Any">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.function">, !py.contract<"builtins.object">, !py.union<!py.type<!py.contract<"builtins.object">>, !py.literal<None>>] -> [!py.contract<"typing.Any">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @staticmethod attributes {
    base_names = ["object"], ly.typing.params = ["P", "R"],
    method_names = ["__call__", "__get__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.staticmethod">, !py.paramspec<"P">] -> [!py.contract<"$R">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.staticmethod">, !py.union<!py.contract<"builtins.object">, !py.literal<None>>, !py.type<!py.contract<"builtins.object">>] -> [!py.protocol<"Callable", [!py.paramspec<"P">] -> [!py.contract<"$R">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @classmethod attributes {
    base_names = ["object"], ly.typing.params = ["T", "P", "R"],
    method_names = ["__get__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.classmethod">, !py.union<!py.contract<"$T">, !py.literal<None>>, !py.type<!py.contract<"$T">>] -> [!py.protocol<"Callable", [!py.paramspec<"P">] -> [!py.contract<"$R">]>]>
    ],
    method_kinds = ["instance"]
  } {}

  py.class @int attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__new__", "__add__", "__sub__", "__mul__", "__floordiv__",
                    "__truediv__", "__mod__", "__and__", "__or__", "__xor__",
                    "__lshift__", "__rshift__", "__neg__", "__pos__",
                    "__invert__", "__round__", "__int__", "__float__",
                    "__bool__", "__index__", "__hash__", "__lt__", "__le__",
                    "__gt__", "__ge__", "__repr__", "__str__", "__eq__", "__ne__",
                    "__pow__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.int">>, !py.union<!py.contract<"typing.SupportsInt">, !py.contract<"typing.SupportsIndex">, !py.contract<"builtins.str">, !py.contract<"builtins.bytes">, !py.contract<"builtins.bytearray">>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  py.class @bool attributes {
    base_names = ["int"], ly.typing.final,
    method_names = ["__new__", "__repr__", "__str__", "__bool__", "__and__",
                    "__or__", "__xor__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bool">>, !py.contract<"builtins.object">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}

  py.class @float attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__new__", "__repr__", "__add__", "__sub__", "__mul__",
                    "__truediv__", "__floordiv__", "__mod__", "__float__",
                    "__bool__", "__round__", "__lt__", "__le__", "__gt__",
                    "__ge__", "__str__", "__eq__", "__ne__", "__pow__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.float">>, !py.contract<"typing.SupportsFloat">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.union<!py.contract<"typing.SupportsIndex">, !py.literal<None>>] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  py.class @complex attributes {base_names = ["object"], ly.typing.final} {}

  py.class @str attributes {
    base_names = ["Sequence", "Hashable"],
    ly.typing.base_args = [[!py.contract<"builtins.str">], []],
    method_names = ["__new__", "__len__", "__iter__", "__getitem__", "__add__",
                    "__contains__", "__eq__", "__lt__", "__le__", "__gt__",
                    "__ge__", "join", "startswith", "startswith", "startswith",
                    "endswith", "endswith", "endswith", "__repr__", "__str__",
                    "__ne__", "encode", "upper", "lower", "casefold", "title",
                    "capitalize", "swapcase", "isalpha", "isspace", "isdecimal",
                    "isdigit", "isnumeric", "isupper", "islower", "isprintable",
                    "istitle", "isalnum", "isidentifier", "isascii", "find",
                    "find", "find", "rfind", "rfind", "rfind", "index", "index",
                    "index", "rindex", "rindex", "rindex", "count", "count",
                    "count", "replace", "replace", "strip", "strip", "lstrip",
                    "lstrip", "rstrip", "rstrip", "removeprefix",
                    "removesuffix", "center", "center", "ljust", "ljust",
                    "rjust", "rjust", "zfill", "expandtabs", "expandtabs",
                    "__mul__", "split", "split", "split", "rsplit", "rsplit",
                    "rsplit", "splitlines", "splitlines", "partition",
                    "rpartition"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.str">>, !py.contract<"builtins.object">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str_iterator">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.protocol<"Iterable", [!py.contract<"builtins.str">]>] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.tuple", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.tuple", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">]>]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  py.class @str_iterator attributes {
    base_names = ["Iterator"],
    ly.typing.base_args = [[!py.contract<"builtins.str">]],
    method_names = ["__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.str_iterator">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str_iterator">] -> [!py.contract<"builtins.str">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @bytes attributes {
    base_names = ["Sequence", "Hashable"],
    ly.typing.base_args = [[!py.contract<"builtins.int">], []],
    ly.typing.final,
    method_names = ["__len__", "__getitem__", "__add__", "__eq__", "__ne__",
                    "__bool__", "__repr__", "__str__", "decode", "decode",
                    "decode", "split", "split", "split", "find", "find", "find",
                    "count", "count", "count", "startswith", "startswith",
                    "startswith", "endswith", "endswith", "endswith", "strip",
                    "strip", "replace", "replace", "hex", "fromhex", "join",
                    "__mul__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytes">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytes">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytes">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytes">>, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.protocol<"Iterable", [!py.contract<"builtins.bytes">]>] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytes">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "classmethod", "instance", "instance"]
  } {}
  py.class @bytearray attributes {base_names = ["MutableSequence"],
                                 ly.typing.base_args = [[!py.contract<"builtins.int">]],
                                 ly.typing.final} {}
  py.class @memoryview attributes {base_names = ["Sequence"],
                                  ly.typing.base_args = [[!py.contract<"builtins.int">]]} {}

  py.class @slice attributes {
    base_names = ["object"], ly.typing.final,
    ly.typing.params = ["StartT", "StopT", "StepT"],
    method_names = ["indices"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.tuple">]>
    ],
    method_kinds = ["instance"]
  } {}

  py.class @tuple attributes {
    base_names = ["Sequence"], ly.typing.params = ["T"],
    ly.runtime.contract = "builtins.tuple", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">]],
    method_names = ["__len__", "__contains__", "__getitem__", "__iter__",
                    "__add__", "__mul__", "count", "index", "__repr__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.tuple">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.tuple">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.Any">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.Any">, !py.contract<"typing.SupportsIndex">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.contract<"builtins.str">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance"]
  } {}

  py.class @list attributes {
    base_names = ["MutableSequence"], ly.typing.params = ["T"],
    ly.runtime.contract = "builtins.list", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    ly.runtime.required_primitives = ["ensure_capacity"],
    ly.typing.structural_mutators = ["append"],
    ly.typing.base_args = [[!py.contract<"$T">]],
    method_names = ["__init__", "__init__", "append", "extend", "pop",
                    "insert", "remove", "clear", "__len__", "__iter__",
                    "__getitem__", "__setitem__", "__delitem__",
                    "__contains__", "__repr__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.contract<"builtins.str">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}

  py.class @set attributes {
    base_names = ["MutableSet"], ly.typing.params = ["T"],
    ly.typing.base_args = [[!py.contract<"$T">]],
    ly.runtime.contract = "builtins.set", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    ly.typing.structural_mutators = ["add"],
    method_names = ["__init__", "add", "__len__", "__iter__",
                    "__contains__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.set">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  py.class @dict attributes {
    base_names = ["MutableMapping"], ly.typing.params = ["K", "V"],
    ly.runtime.contract = "builtins.dict", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    ly.runtime.required_primitives = ["ensure_capacity"],
    ly.typing.structural_mutators = ["__setitem__"],
    ly.typing.base_args = [[!py.contract<"$K">, !py.contract<"$V">]],
    method_names = ["__init__", "__init__", "__len__", "__iter__",
                    "__getitem__", "get", "get", "get", "__setitem__",
                    "__delitem__", "__contains__", "keys", "values", "items",
                    "__repr__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.protocol<"Iterable", [!py.contract<"builtins.tuple", [!py.contract<"$K">, !py.contract<"$V">]>]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.protocol<"Iterator", [!py.contract<"$K">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">] -> [!py.contract<"$V">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">] -> [!py.union<!py.contract<"$V">, !py.literal<None>>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">, !py.contract<"$V">] -> [!py.contract<"$V">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">, !py.typevar<"D">] -> [!py.union<!py.contract<"$V">, !py.typevar<"D">>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">, !py.contract<"$V">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.dict_keys", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.dict_values", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.dict_items", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.str">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}

  py.class @MappingView attributes {
    base_names = ["Sized"], ly.typing.abstract,
    method_names = ["__len__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.MappingView">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @KeysView attributes {
    base_names = ["MappingView", "AbstractSet"], ly.typing.params = ["K"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[], [!py.contract<"$K">]],
    method_names = ["__contains__", "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.KeysView">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"typing.KeysView">] -> [!py.protocol<"Iterator", [!py.contract<"$K">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @ValuesView attributes {
    base_names = ["MappingView", "Collection"], ly.typing.params = ["V"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[], [!py.contract<"$V">]],
    method_names = ["__contains__", "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.ValuesView">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"typing.ValuesView">] -> [!py.protocol<"Iterator", [!py.contract<"$V">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @ItemsView attributes {
    base_names = ["MappingView", "AbstractSet"], ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.base_args = [[], [!py.contract<"builtins.tuple">]],
    method_names = ["__contains__", "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.ItemsView">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"typing.ItemsView">] -> [!py.protocol<"Iterator", [!py.contract<"builtins.tuple", [!py.contract<"$K">, !py.contract<"$V">]>]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @dict_keys attributes {
    base_names = ["KeysView"], ly.typing.final,
    ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$K">]],
    method_names = ["__reversed__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.dict_keys", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.protocol<"Iterator", [!py.contract<"$K">]>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @dict_values attributes {
    base_names = ["ValuesView"], ly.typing.final,
    ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$V">]],
    method_names = ["__reversed__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.dict_values", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.protocol<"Iterator", [!py.contract<"$V">]>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @dict_items attributes {
    base_names = ["ItemsView"], ly.typing.final,
    ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$K">, !py.contract<"$V">]],
    method_names = ["__reversed__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.dict_items", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.protocol<"Iterator", [!py.contract<"builtins.tuple", [!py.contract<"$K">, !py.contract<"$V">]>]>]>
    ],
    method_kinds = ["instance"]
  } {}

  py.class @frozenset attributes {base_names = ["AbstractSet", "Hashable"],
                                 ly.typing.params = ["T"],
                                 ly.typing.base_args = [[!py.contract<"$T">], []]} {}
  py.class @range attributes {base_names = ["Sequence", "Hashable"],
                             ly.typing.base_args = [[!py.contract<"builtins.int">], []],
                             ly.typing.final,
    method_names = ["__new__", "__new__", "__new__", "__init__", "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.range">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.range">>, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.range">>, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">, !py.paramspec<"P">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">] -> [!py.contract<"builtins.range_iterator">]>
    ],
    method_kinds = ["classmethod", "classmethod", "classmethod", "instance",
                    "instance"]
  } {}

  py.class @range_iterator attributes {
    base_names = ["Iterator"],
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    method_names = ["__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.range_iterator">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range_iterator">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @BaseException attributes {
    base_names = ["object"],
    field_names = ["args", "__cause__", "__context__", "__suppress_context__",
                   "__traceback__"],
    field_contract_types = [
      !py.contract<"builtins.tuple">,
      !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>,
      !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>,
      !py.contract<"builtins.bool">,
      !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>
    ],
    method_names = ["__init__", "with_traceback", "__str__", "__repr__",
                    "add_note"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">, !py.paramspec<"P">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">, !py.contract<"builtins.str">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance"]
  } {}
  py.class @Exception attributes {base_names = ["BaseException"]} {}
  py.class @RuntimeError attributes {base_names = ["Exception"]} {}
  py.class @TypeError attributes {base_names = ["Exception"]} {}
  py.class @ValueError attributes {base_names = ["Exception"]} {}
  py.class @ArithmeticError attributes {base_names = ["Exception"]} {}
  py.class @LookupError attributes {base_names = ["Exception"]} {}
  py.class @ZeroDivisionError attributes {base_names = ["ArithmeticError"]} {}
  py.class @KeyError attributes {base_names = ["LookupError"]} {}
  py.class @IndexError attributes {base_names = ["LookupError"]} {}
  py.class @AssertionError attributes {base_names = ["Exception"]} {}
  py.class @StopIteration attributes {
    base_names = ["Exception"],
    field_names = ["value"],
    field_contract_types = [!py.contract<"typing.Any">]
  } {}
  py.class @StopAsyncIteration attributes {base_names = ["Exception"]} {}
  // SystemExit derives from BaseException directly (never caught by
  // `except Exception`); the top-level runner converts it to the process
  // exit status instead of printing a traceback.
  py.class @SystemExit attributes {base_names = ["BaseException"]} {}
  // GeneratorExit derives from BaseException directly (never caught by
  // `except Exception`); generator.close() injects it at the suspension
  // point so the body's finally blocks run.
  py.class @GeneratorExit attributes {base_names = ["BaseException"]} {}
  py.class @OSError attributes {base_names = ["Exception"]} {}
  py.class @FileNotFoundError attributes {base_names = ["OSError"]} {}
  py.class @KeyboardInterrupt attributes {base_names = ["BaseException"]} {}
  py.class @BaseExceptionGroup attributes {base_names = ["BaseException"]} {}
  py.class @ExceptionGroup attributes {base_names = ["BaseExceptionGroup"]} {}
  py.class @FloatingPointError attributes {base_names = ["ArithmeticError"]} {}
  py.class @OverflowError attributes {base_names = ["ArithmeticError"]} {}
  py.class @BufferError attributes {base_names = ["Exception"]} {}
  py.class @EOFError attributes {base_names = ["Exception"]} {}
  py.class @ImportError attributes {base_names = ["Exception"]} {}
  py.class @ModuleNotFoundError attributes {base_names = ["ImportError"]} {}
  py.class @MemoryError attributes {base_names = ["Exception"]} {}
  py.class @NameError attributes {base_names = ["Exception"]} {}
  py.class @UnboundLocalError attributes {base_names = ["NameError"]} {}
  py.class @AttributeError attributes {base_names = ["Exception"]} {}
  py.class @ReferenceError attributes {base_names = ["Exception"]} {}
  py.class @NotImplementedError attributes {base_names = ["RuntimeError"]} {}
  py.class @RecursionError attributes {base_names = ["RuntimeError"]} {}
  py.class @PythonFinalizationError attributes {base_names = ["RuntimeError"]} {}
  py.class @SyntaxError attributes {base_names = ["Exception"]} {}
  py.class @IndentationError attributes {base_names = ["SyntaxError"]} {}
  py.class @TabError attributes {base_names = ["IndentationError"]} {}
  py.class @SystemError attributes {base_names = ["Exception"]} {}
  py.class @UnicodeError attributes {base_names = ["ValueError"]} {}
  py.class @UnicodeDecodeError attributes {base_names = ["UnicodeError"]} {}
  py.class @UnicodeEncodeError attributes {base_names = ["UnicodeError"]} {}
  py.class @UnicodeTranslateError attributes {base_names = ["UnicodeError"]} {}
  py.class @Warning attributes {base_names = ["Exception"]} {}
  py.class @BytesWarning attributes {base_names = ["Warning"]} {}
  py.class @DeprecationWarning attributes {base_names = ["Warning"]} {}
  py.class @EncodingWarning attributes {base_names = ["Warning"]} {}
  py.class @FutureWarning attributes {base_names = ["Warning"]} {}
  py.class @ImportWarning attributes {base_names = ["Warning"]} {}
  py.class @PendingDeprecationWarning attributes {base_names = ["Warning"]} {}
  py.class @ResourceWarning attributes {base_names = ["Warning"]} {}
  py.class @RuntimeWarning attributes {base_names = ["Warning"]} {}
  py.class @SyntaxWarning attributes {base_names = ["Warning"]} {}
  py.class @UnicodeWarning attributes {base_names = ["Warning"]} {}
  py.class @UserWarning attributes {base_names = ["Warning"]} {}
  py.class @BlockingIOError attributes {base_names = ["OSError"]} {}
  py.class @ChildProcessError attributes {base_names = ["OSError"]} {}
  py.class @ConnectionError attributes {base_names = ["OSError"]} {}
  py.class @BrokenPipeError attributes {base_names = ["ConnectionError"]} {}
  py.class @ConnectionAbortedError attributes {base_names = ["ConnectionError"]} {}
  py.class @ConnectionRefusedError attributes {base_names = ["ConnectionError"]} {}
  py.class @ConnectionResetError attributes {base_names = ["ConnectionError"]} {}
  py.class @FileExistsError attributes {base_names = ["OSError"]} {}
  py.class @InterruptedError attributes {base_names = ["OSError"]} {}
  py.class @IsADirectoryError attributes {base_names = ["OSError"]} {}
  py.class @NotADirectoryError attributes {base_names = ["OSError"]} {}
  py.class @PermissionError attributes {base_names = ["OSError"]} {}
  py.class @ProcessLookupError attributes {base_names = ["OSError"]} {}
  py.class @TimeoutError attributes {base_names = ["OSError"]} {}

  // ===========================================================
  // Lowering strategies (transform dialect carrier) -- layer 4 of the
  // transformation stack: target-selected,
  // schedule-shaped transformations that ship WITH the module instead of in
  // C++. Per-op lowerings stay in RuntimeBundleLowerer (layers 1-2); stage
  // ordering stays in LoweringPipeline.cpp (layer 3).
  //
  // A modules/<name>.mlir may declare lowering strategies as a nested
  // strategy-library module marked `transform.with_named_sequence`. The
  // runtime import skips it (it never merges into the user module); the
  // lowering pipeline's strategy interpreter collects these libraries from
  // the embedded manifests and applies matched sequences to the user module
  // via transform::applyTransforms -- matchers select payload
  // ops (transform.foreach_match), actions rewrite them
  // (transform.apply_patterns / transform.include).
  // ===========================================================
  module @__lython_lowering_strategies attributes {transform.with_named_sequence} {
    // Post-import cleanup: the user module was canonicalized before the
    // runtime implementations were imported (pipeline phase 6 runs before
    // phase 8), so the freshly imported runtime function bodies reach the
    // runtime lowering uncanonicalized. Re-run canonicalization + CSE over
    // the whole module once the implementations are in.
    transform.named_sequence @__lython_strategy_post_import_cleanup(%root: !transform.any_op) {
      %canonicalized = transform.apply_registered_pass "canonicalize" to %root : (!transform.any_op) -> !transform.any_op
      %cleaned = transform.apply_registered_pass "cse" to %canonicalized : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
  }

  // ===========================================================
  // Runtime implementations (contract + impl co-located).
  // ===========================================================

  // ===== impls: object =====
  memref.global "private" constant @__ly_bool_repr_true : memref<4xi8> = dense<[84, 114, 117, 101]>
  memref.global "private" constant @__ly_bool_repr_false : memref<5xi8> = dense<[70, 97, 108, 115, 101]>

  // ABI shape declarations are manifest entries: they describe the physical
  // runtime bundle for contracts whose structural signatures live elsewhere.
  func.func private @LyNone_Shape() attributes {ly.runtime.contract = "types.NoneType", ly.runtime.shape}

  func.func private @LyObject_Shape() -> memref<16xi64> attributes {ly.runtime.contract = "builtins.object", ly.runtime.shape}

  func.func private @LyBool_Shape() -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.shape}


  func.func private @LyBuiltin_Len() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.runtime.builtin = "len", ly.runtime.builtin_lowering = "method", ly.runtime.builtin_method = "__len__", ly.runtime.contract = "builtins.object", ly.runtime.primitive = "builtin_len", ly.runtime.result_contract = "builtins.int"}

  func.func private @LyBuiltin_Hash() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.runtime.builtin = "hash", ly.runtime.builtin_lowering = "method", ly.runtime.builtin_method = "__hash__", ly.runtime.contract = "builtins.object", ly.runtime.primitive = "builtin_hash", ly.runtime.result_contract = "builtins.int"}

  func.func @LyObject_Init(%header: memref<16xi64> {ly.ownership.object_header}) attributes {ly.runtime.class_id = 0 : i64, ly.runtime.contract = "builtins.object", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func private @LyObject_ReleaseBoxedPayloadRaw(%box: memref<16xi64>)

  func.func @LyObject_DecRef(%box: memref<16xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.object", ly.runtime.deallocator} {
    %storage = memref.cast %box : memref<16xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyObject_ReleaseBoxedPayloadRaw(%box) : (memref<16xi64>) -> ()
    memref.dealloc %box : memref<16xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyObject_DefaultRepr(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}, %prefix: memref<?xi8>, %prefix_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "default_repr", ly.runtime.result_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sixteen = arith.constant 16 : i64
    %max_digits_index = arith.constant 16 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %ascii_zero = arith.constant 48 : i64
    %ascii_a_minus_ten = arith.constant 87 : i64
    %ascii_gt = arith.constant 62 : i8

    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64, strided<[1], offset: ?>> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    %counted:2 = scf.for %i = %lower to %max_digits_index step %step iter_args(%n = %ptr, %digits = %zero) -> (i64, i64) {
      %active = arith.cmpi ne, %n, %zero : i64
      %next_n_active = arith.divui %n, %sixteen : i64
      %next_n = arith.select %active, %next_n_active, %n : i1, i64
      %incremented = arith.addi %digits, %one : i64
      %next_digits = arith.select %active, %incremented, %digits : i1, i64
      scf.yield %next_n, %next_digits : i64, i64
    }
    %ptr_is_zero = arith.cmpi eq, %ptr, %zero : i64
    %hex_digits = arith.select %ptr_is_zero, %one, %counted#1 : i1, i64
    %body_len = arith.addi %hex_digits, %one : i64
    %total_len = arith.addi %prefix_len, %body_len : i64
    %total_len_index = arith.index_cast %total_len : i64 to index
    %prefix_len_index = arith.index_cast %prefix_len : i64 to index
    %hex_digits_index = arith.index_cast %hex_digits : i64 to index
    %buffer = memref.alloca(%total_len_index) : memref<?xi8>

    scf.for %i = %lower to %prefix_len_index step %step {
      %byte = memref.load %prefix[%i] : memref<?xi8>
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }

    scf.for %i = %lower to %hex_digits_index step %step iter_args(%n = %ptr) -> (i64) {
      %digit = arith.remui %n, %sixteen : i64
      %ten = arith.constant 10 : i64
      %is_decimal = arith.cmpi ult, %digit, %ten : i64
      %decimal_ch = arith.addi %digit, %ascii_zero : i64
      %alpha_ch = arith.addi %digit, %ascii_a_minus_ten : i64
      %ch_i64 = arith.select %is_decimal, %decimal_ch, %alpha_ch : i1, i64
      %ch = arith.trunci %ch_i64 : i64 to i8
      %one_index = arith.constant 1 : index
      %last_digit = arith.subi %hex_digits_index, %one_index : index
      %offset = arith.subi %last_digit, %i : index
      %dest = arith.addi %prefix_len_index, %offset : index
      memref.store %ch, %buffer[%dest] : memref<?xi8>
      %next = arith.divui %n, %sixteen : i64
      scf.yield %next : i64
    }

    %suffix_pos = arith.addi %prefix_len_index, %hex_digits_index : index
    memref.store %ascii_gt, %buffer[%suffix_pos] : memref<?xi8>
    %start = arith.constant 0 : index
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%buffer, %start, %total_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBool_Repr(%value: i1) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %result:2 = scf.if %value -> (memref<2xi64>, memref<?xi8>) {
      %true_static = memref.get_global @__ly_bool_repr_true : memref<4xi8>
      %true_bytes = memref.cast %true_static : memref<4xi8> to memref<?xi8>
      %true_len = arith.constant 4 : i64
      %header, %bytes = func.call @LyUnicode_FromBytes(%true_bytes, %c0, %true_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    } else {
      %false_static = memref.get_global @__ly_bool_repr_false : memref<5xi8>
      %false_bytes = memref.cast %false_static : memref<5xi8> to memref<?xi8>
      %false_len = arith.constant 5 : i64
      %header, %bytes = func.call @LyUnicode_FromBytes(%false_bytes, %c0, %false_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // Boxed bool: two immortal singletons (CPython's True/False semantics).
  // Layout follows the shared header contract: [refcount, class_id, value].
  // The refcount starts at the immortal marker; generic retains/releases may
  // drift it but it never reaches zero, and the deallocator is a no-op.
  memref.global "private" @__ly_bool_box_true : memref<3xi64> = dense<[9223372036854775807, 22, 1]>
  memref.global "private" @__ly_bool_box_false : memref<3xi64> = dense<[9223372036854775807, 22, 0]>

  // box: canonical i1 -> the boxed singleton (no allocation).
  func.func @LyBool_Box(%value: i1) -> memref<3xi64> attributes {ly.runtime.class_id = 22 : i64, ly.runtime.contract = "builtins.bool", ly.runtime.primitive = "box"} {
    %true_box = memref.get_global @__ly_bool_box_true : memref<3xi64>
    %false_box = memref.get_global @__ly_bool_box_false : memref<3xi64>
    %box = arith.select %value, %true_box, %false_box : memref<3xi64>
    func.return %box : memref<3xi64>
  }

  // unbox: boxed singleton -> canonical i1.
  func.func @LyBool_Unbox(%header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.primitive = "unbox"} {
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %word = memref.load %header[%c2] : memref<3xi64>
    %value = arith.cmpi ne, %word, %c0_i64 : i64
    func.return %value : i1
  }

  // Boxed-conforming __repr__ (erased-element dispatch through the repr hook).
  func.func @LyBool_BoxedRepr(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %value = func.call @LyBool_Unbox(%header) : (memref<3xi64>) -> i1
    %result_header, %result_bytes = func.call @LyBool_Repr(%value) : (i1) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // Immortal singletons never deallocate; the release hook still needs a
  // conforming deallocator so boxed slots release without a miss.
  func.func @LyBool_DecRef(%header: memref<3xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bool", ly.runtime.deallocator} {
    func.return
  }

  func.func @LyBool_Str(%value: i1) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %header, %bytes = func.call @LyBool_Repr(%value) : (i1) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"} {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    // Tagged fast path: a header whose aligned pointer has bit 0 set is an
    // inline tagged long. It owns no memory and must
    // not be dereferenced at all.
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64, strided<[1], offset: ?>> -> index
    %ptr_bits = arith.index_cast %ptr_index : index to i64
    %tag_one = arith.constant 1 : i64
    %tag_bit = arith.andi %ptr_bits, %tag_one : i64
    %is_tagged = arith.cmpi eq, %tag_bit, %tag_one : i64
    cf.cond_br %is_tagged, ^done, ^probe

  ^probe:
    // Immortal fast path: immortality is fixed at object creation and never
    // changes, so a pre-RMW acquire read is a stable witness. Skipping the
    // RMW keeps immortal headers write-free, which lets constant literals
    // live in read-only sections.
    %observed = memref.load %header[%slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "object.refcount.load"} : memref<2xi64, strided<[1], offset: ?>>
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^done, ^mutate

  ^mutate:
    %previous = memref.generic_atomic_rmw %header[%slot] : memref<2xi64, strided<[1], offset: ?>> {
    ^bb0(%current : i64):
      %body_zero = arith.constant 0 : i64
      %body_one = arith.constant 1 : i64
      %body_immortal = arith.constant 9223372036854775807 : i64
      %body_positive = arith.cmpi sgt, %current, %body_zero : i64
      %body_immortal_check = arith.cmpi eq, %current, %body_immortal : i64
      %body_incremented = arith.addi %current, %body_one : i64
      %body_positive_next = arith.select %body_positive, %body_incremented, %current : i1, i64
      %body_next = arith.select %body_immortal_check, %current, %body_positive_next : i1, i64
      memref.atomic_yield %body_next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.retain_premise = "entry-borrowed", ly.atomic.role = "object.refcount.retain"}
    %is_immortal = arith.cmpi eq, %previous, %immortal : i64
    cf.cond_br %is_immortal, ^done, ^check_positive

  ^check_positive:
    %positive = arith.cmpi sgt, %previous, %zero : i64
    cf.assert %positive, "Ly_IncRef observed non-positive refcount"
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"} {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    // Immortal fast path: see Ly_IncRef. Immortal storage is never written,
    // so constant literal objects stay in read-only storage.
    %observed = memref.load %storage[%slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "object.refcount.load"} : memref<?xi64>
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^done, ^mutate

  ^mutate:
    %previous = memref.generic_atomic_rmw %storage[%slot] : memref<?xi64> {
    ^bb0(%current : i64):
      %body_zero = arith.constant 0 : i64
      %body_one = arith.constant 1 : i64
      %body_immortal = arith.constant 9223372036854775807 : i64
      %body_positive = arith.cmpi sgt, %current, %body_zero : i64
      %body_immortal_check = arith.cmpi eq, %current, %body_immortal : i64
      %body_decremented = arith.subi %current, %body_one : i64
      %body_positive_next = arith.select %body_positive, %body_decremented, %current : i1, i64
      %body_next = arith.select %body_immortal_check, %current, %body_positive_next : i1, i64
      memref.atomic_yield %body_next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "object.refcount.release"}
    %is_immortal = arith.cmpi eq, %previous, %immortal : i64
    cf.cond_br %is_immortal, ^done, ^check_positive

  ^check_positive:
    %positive = arith.cmpi sgt, %previous, %zero : i64
    cf.assert %positive, "Ly_DecRef observed non-positive refcount"
    %one = arith.constant 1 : i64
    %became_zero = arith.cmpi eq, %previous, %one : i64
    func.return %became_zero : i1

  ^done:
    %false = arith.constant false
    func.return %false : i1
  }

  func.func @LyBaseException_DecRef(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<3xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyUnicode_DecRef(%message_header) : (memref<2xi64>) -> ()
    memref.dealloc %header : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    // int is one allocation; the header view carries it. meta/digits are
    // interior views of the same block and must not be freed.
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    // str is one allocation; the header view carries it.
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: exception =====
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyEH_BorrowCurrentException() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "borrow_current"}

  func.func private @LyException_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.Exception", ly.runtime.shape}
  func.func private @LyRuntimeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.RuntimeError", ly.runtime.shape}
  func.func private @LyTypeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.TypeError", ly.runtime.shape}
  func.func private @LyValueError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ValueError", ly.runtime.shape}
  func.func private @LyArithmeticError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.shape}
  func.func private @LyLookupError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.LookupError", ly.runtime.shape}
  func.func private @LyZeroDivisionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.shape}
  func.func private @LyKeyError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.KeyError", ly.runtime.shape}
  func.func private @LyIndexError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.IndexError", ly.runtime.shape}
  func.func private @LyAssertionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.AssertionError", ly.runtime.shape}
  func.func private @LyKeyboardInterrupt_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.shape}
  func.func private @LyBaseExceptionGroup_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.shape}
  func.func private @LyExceptionGroup_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.shape}
  func.func private @LyFloatingPointError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.shape}
  func.func private @LyOverflowError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.OverflowError", ly.runtime.shape}
  func.func private @LyBufferError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BufferError", ly.runtime.shape}
  func.func private @LyEOFError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.EOFError", ly.runtime.shape}
  func.func private @LyImportError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ImportError", ly.runtime.shape}
  func.func private @LyModuleNotFoundError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.shape}
  func.func private @LyMemoryError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.MemoryError", ly.runtime.shape}
  func.func private @LyNameError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.NameError", ly.runtime.shape}
  func.func private @LyUnboundLocalError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.shape}
  func.func private @LyAttributeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.AttributeError", ly.runtime.shape}
  func.func private @LyReferenceError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ReferenceError", ly.runtime.shape}
  func.func private @LyNotImplementedError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.shape}
  func.func private @LyRecursionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.RecursionError", ly.runtime.shape}
  func.func private @LyPythonFinalizationError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.shape}
  func.func private @LySyntaxError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SyntaxError", ly.runtime.shape}
  func.func private @LyIndentationError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.IndentationError", ly.runtime.shape}
  func.func private @LyTabError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.TabError", ly.runtime.shape}
  func.func private @LySystemError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SystemError", ly.runtime.shape}
  func.func private @LyUnicodeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeError", ly.runtime.shape}
  func.func private @LyUnicodeDecodeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.shape}
  func.func private @LyUnicodeEncodeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.shape}
  func.func private @LyUnicodeTranslateError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.shape}
  func.func private @LyWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.Warning", ly.runtime.shape}
  func.func private @LyBytesWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BytesWarning", ly.runtime.shape}
  func.func private @LyDeprecationWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.shape}
  func.func private @LyEncodingWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.shape}
  func.func private @LyFutureWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.FutureWarning", ly.runtime.shape}
  func.func private @LyImportWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ImportWarning", ly.runtime.shape}
  func.func private @LyPendingDeprecationWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.shape}
  func.func private @LyResourceWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.shape}
  func.func private @LyRuntimeWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.shape}
  func.func private @LySyntaxWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.shape}
  func.func private @LyUnicodeWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.shape}
  func.func private @LyUserWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UserWarning", ly.runtime.shape}
  func.func private @LyBlockingIOError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.shape}
  func.func private @LyChildProcessError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.shape}
  func.func private @LyConnectionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ConnectionError", ly.runtime.shape}
  func.func private @LyBrokenPipeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.shape}
  func.func private @LyConnectionAbortedError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.shape}
  func.func private @LyConnectionRefusedError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.shape}
  func.func private @LyConnectionResetError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.shape}
  func.func private @LyFileExistsError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.FileExistsError", ly.runtime.shape}
  func.func private @LyInterruptedError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.InterruptedError", ly.runtime.shape}
  func.func private @LyIsADirectoryError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.shape}
  func.func private @LyNotADirectoryError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.shape}
  func.func private @LyPermissionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.PermissionError", ly.runtime.shape}
  func.func private @LyProcessLookupError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.shape}
  func.func private @LyTimeoutError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.TimeoutError", ly.runtime.shape}
  func.func private @LyStopIteration_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.StopIteration", ly.runtime.shape}
  func.func private @LyStopAsyncIteration_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.shape}
  func.func private @LySystemExit_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SystemExit", ly.runtime.shape}
  func.func private @LyGeneratorExit_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.shape}
  func.func private @LyOSError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.OSError", ly.runtime.shape}
  func.func private @LyFileNotFoundError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.shape}
  func.func private @LyCancelledError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "asyncio.CancelledError", ly.runtime.shape}

  func.func @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    func.call @LyUnicode_DecRef(%old_message_header) : (memref<2xi64>) -> ()
    func.return %header, %message_header, %message_bytes : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.SystemExit", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyGeneratorExit_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.OSError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyCancelledError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "asyncio.CancelledError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %layout_exception = arith.constant 5 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %class_slot = arith.constant 2 : index
    %zero_index = arith.constant 0 : index
    %zero_len = arith.constant 0 : i64

    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<3xi64>
    %empty_bytes = memref.alloca(%zero_index) : memref<?xi8>
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%empty_bytes, %zero_index, %zero_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)

    memref.store %one, %header[%refcount_slot] : memref<3xi64>
    memref.store %layout_exception, %header[%layout_slot] : memref<3xi64>
    memref.store %class_id, %header[%class_slot] : memref<3xi64>

    func.return %header, %message_header, %message_bytes : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 50 : i64, ly.runtime.contract = "builtins.Exception", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 51 : i64, ly.runtime.contract = "builtins.RuntimeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 52 : i64, ly.runtime.contract = "builtins.TypeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 53 : i64, ly.runtime.contract = "builtins.ValueError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 59 : i64, ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 60 : i64, ly.runtime.contract = "builtins.LookupError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 61 : i64, ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 54 : i64, ly.runtime.contract = "builtins.KeyError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 55 : i64, ly.runtime.contract = "builtins.IndexError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 56 : i64, ly.runtime.contract = "builtins.AssertionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 57 : i64, ly.runtime.contract = "builtins.StopIteration", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 58 : i64, ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 64 : i64, ly.runtime.contract = "builtins.SystemExit", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyGeneratorExit_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 68 : i64, ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 66 : i64, ly.runtime.contract = "builtins.OSError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 67 : i64, ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyCancelledError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 62 : i64, ly.runtime.contract = "asyncio.CancelledError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_BaseException : memref<13xi8> = dense<[66, 97, 115, 101, 69, 120, 99, 101, 112, 116, 105, 111, 110]>
  memref.global "private" constant @__ly_excname_Exception : memref<9xi8> = dense<[69, 120, 99, 101, 112, 116, 105, 111, 110]>
  memref.global "private" constant @__ly_excname_RuntimeError : memref<12xi8> = dense<[82, 117, 110, 116, 105, 109, 101, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_TypeError : memref<9xi8> = dense<[84, 121, 112, 101, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_ValueError : memref<10xi8> = dense<[86, 97, 108, 117, 101, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_ArithmeticError : memref<15xi8> = dense<[65, 114, 105, 116, 104, 109, 101, 116, 105, 99, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_LookupError : memref<11xi8> = dense<[76, 111, 111, 107, 117, 112, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_ZeroDivisionError : memref<17xi8> = dense<[90, 101, 114, 111, 68, 105, 118, 105, 115, 105, 111, 110, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_KeyError : memref<8xi8> = dense<[75, 101, 121, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_IndexError : memref<10xi8> = dense<[73, 110, 100, 101, 120, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_AssertionError : memref<14xi8> = dense<[65, 115, 115, 101, 114, 116, 105, 111, 110, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_StopIteration : memref<13xi8> = dense<[83, 116, 111, 112, 73, 116, 101, 114, 97, 116, 105, 111, 110]>
  memref.global "private" constant @__ly_excname_StopAsyncIteration : memref<18xi8> = dense<[83, 116, 111, 112, 65, 115, 121, 110, 99, 73, 116, 101, 114, 97, 116, 105, 111, 110]>
  memref.global "private" constant @__ly_excname_SystemExit : memref<10xi8> = dense<[83, 121, 115, 116, 101, 109, 69, 120, 105, 116]>
  memref.global "private" constant @__ly_excname_GeneratorExit : memref<13xi8> = dense<[71, 101, 110, 101, 114, 97, 116, 111, 114, 69, 120, 105, 116]>
  memref.global "private" constant @__ly_excname_OSError : memref<7xi8> = dense<[79, 83, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_FileNotFoundError : memref<17xi8> = dense<[70, 105, 108, 101, 78, 111, 116, 70, 111, 117, 110, 100, 69, 114, 114, 111, 114]>
  memref.global "private" constant @__ly_excname_CancelledError : memref<14xi8> = dense<[67, 97, 110, 99, 101, 108, 108, 101, 100, 69, 114, 114, 111, 114]>

  // CPython repr(e): ClassName(<repr of the message>) -- ClassName() when
  // the message is empty. A no-argument construction and an explicit empty
  // string are indistinguishable in the 3-word payload, so the empty case
  // renders as ClassName() (the CPython-visible difference is only
  // ClassName('')).
  memref.global "private" constant @__ly_exc_lparen : memref<1xi8> = dense<40>
  memref.global "private" constant @__ly_exc_rparen : memref<1xi8> = dense<41>
  func.func private @__ly_exception_repr(%message_header: memref<2xi64>, %message_bytes: memref<?xi8>, %name: memref<?xi8>, %name_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %name_h, %name_b = func.call @LyUnicode_FromBytes(%name, %c0, %name_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %lparen_ref = memref.get_global @__ly_exc_lparen : memref<1xi8>
    %lparen_dyn = memref.cast %lparen_ref : memref<1xi8> to memref<?xi8>
    %lp_h, %lp_b = func.call @LyUnicode_FromBytes(%lparen_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %rparen_ref = memref.get_global @__ly_exc_rparen : memref<1xi8>
    %rparen_dyn = memref.cast %rparen_ref : memref<1xi8> to memref<?xi8>
    %rp_h, %rp_b = func.call @LyUnicode_FromBytes(%rparen_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %head_h, %head_b = func.call @LyUnicode_Concat(%name_h, %name_b, %lp_h, %lp_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%name_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%lp_h) : (memref<2xi64>) -> ()
    %msg_len = func.call @__ly_unicode_count(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %has_message = arith.cmpi sgt, %msg_len, %zero : i64
    // Both branches yield a fresh owned str so the releases stay outside the
    // regions (region-crossing ownership of the head would trip the
    // affine-ownership verifier).
    %quoted:2 = scf.if %has_message -> (memref<2xi64>, memref<?xi8>) {
      %repr_h, %repr_b = func.call @LyUnicode_Repr(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %repr_h, %repr_b : memref<2xi64>, memref<?xi8>
    } else {
      %one_w = arith.constant 1 : i64
      %empty_h, %empty_b = func.call @__ly_unicode_alloc(%zero, %one_w) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %empty_h, %empty_b : memref<2xi64>, memref<?xi8>
    }
    %body_h, %body_b = func.call @LyUnicode_Concat(%head_h, %head_b, %quoted#0, %quoted#1) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%head_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%quoted#0) : (memref<2xi64>) -> ()
    %out_h, %out_b = func.call @LyUnicode_Concat(%body_h, %body_b, %rp_h, %rp_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%body_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%rp_h) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  // BaseException.args: () for an empty message, (message,) otherwise (the
  // payload stores at most one argument).
  func.func @LyBaseException_Args(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "args", ly.runtime.result_contract = "builtins.tuple", ly.runtime.element_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %msg_len = func.call @__ly_unicode_count(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %has_message = arith.cmpi sgt, %msg_len, %zero : i64
    %count = arith.select %has_message, %one, %zero : i64
    %tuple:3 = func.call @LyTuple_FromLength(%count) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    scf.if %has_message {
      %retained:2 = func.call @__ly_unicode_retain_self(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @__ly_unicode_store_item(%tuple#2, %zero, %retained#0, %retained#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
    }
    func.return %tuple#0, %tuple#1, %tuple#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  memref.global "private" constant @__ly_excname_KeyboardInterrupt : memref<17xi8> = dense<[75, 101, 121, 98, 111, 97, 114, 100, 73, 110, 116, 101, 114, 114, 117, 112, 116]>
  func.func @LyKeyboardInterrupt_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 100 : i64, ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyKeyboardInterrupt_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyKeyboardInterrupt_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyboardInterrupt_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_KeyboardInterrupt : memref<17xi8>
    %name_dyn = memref.cast %name_ref : memref<17xi8> to memref<?xi8>
    %name_len = arith.constant 17 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_BaseExceptionGroup : memref<18xi8> = dense<[66, 97, 115, 101, 69, 120, 99, 101, 112, 116, 105, 111, 110, 71, 114, 111, 117, 112]>
  func.func @LyBaseExceptionGroup_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 101 : i64, ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBaseExceptionGroup_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBaseExceptionGroup_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBaseExceptionGroup_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_BaseExceptionGroup : memref<18xi8>
    %name_dyn = memref.cast %name_ref : memref<18xi8> to memref<?xi8>
    %name_len = arith.constant 18 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ExceptionGroup : memref<14xi8> = dense<[69, 120, 99, 101, 112, 116, 105, 111, 110, 71, 114, 111, 117, 112]>
  func.func @LyExceptionGroup_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 102 : i64, ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyExceptionGroup_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyExceptionGroup_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyExceptionGroup_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ExceptionGroup : memref<14xi8>
    %name_dyn = memref.cast %name_ref : memref<14xi8> to memref<?xi8>
    %name_len = arith.constant 14 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_FloatingPointError : memref<18xi8> = dense<[70, 108, 111, 97, 116, 105, 110, 103, 80, 111, 105, 110, 116, 69, 114, 114, 111, 114]>
  func.func @LyFloatingPointError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 103 : i64, ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyFloatingPointError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyFloatingPointError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFloatingPointError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_FloatingPointError : memref<18xi8>
    %name_dyn = memref.cast %name_ref : memref<18xi8> to memref<?xi8>
    %name_len = arith.constant 18 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_OverflowError : memref<13xi8> = dense<[79, 118, 101, 114, 102, 108, 111, 119, 69, 114, 114, 111, 114]>
  func.func @LyOverflowError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 104 : i64, ly.runtime.contract = "builtins.OverflowError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyOverflowError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.OverflowError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyOverflowError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OverflowError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOverflowError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OverflowError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_OverflowError : memref<13xi8>
    %name_dyn = memref.cast %name_ref : memref<13xi8> to memref<?xi8>
    %name_len = arith.constant 13 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_BufferError : memref<11xi8> = dense<[66, 117, 102, 102, 101, 114, 69, 114, 114, 111, 114]>
  func.func @LyBufferError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 105 : i64, ly.runtime.contract = "builtins.BufferError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBufferError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BufferError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBufferError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BufferError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBufferError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BufferError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_BufferError : memref<11xi8>
    %name_dyn = memref.cast %name_ref : memref<11xi8> to memref<?xi8>
    %name_len = arith.constant 11 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_EOFError : memref<8xi8> = dense<[69, 79, 70, 69, 114, 114, 111, 114]>
  func.func @LyEOFError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 106 : i64, ly.runtime.contract = "builtins.EOFError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyEOFError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.EOFError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyEOFError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.EOFError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyEOFError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.EOFError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_EOFError : memref<8xi8>
    %name_dyn = memref.cast %name_ref : memref<8xi8> to memref<?xi8>
    %name_len = arith.constant 8 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ImportError : memref<11xi8> = dense<[73, 109, 112, 111, 114, 116, 69, 114, 114, 111, 114]>
  func.func @LyImportError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 107 : i64, ly.runtime.contract = "builtins.ImportError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyImportError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ImportError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyImportError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ImportError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyImportError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ImportError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ImportError : memref<11xi8>
    %name_dyn = memref.cast %name_ref : memref<11xi8> to memref<?xi8>
    %name_len = arith.constant 11 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ModuleNotFoundError : memref<19xi8> = dense<[77, 111, 100, 117, 108, 101, 78, 111, 116, 70, 111, 117, 110, 100, 69, 114, 114, 111, 114]>
  func.func @LyModuleNotFoundError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 108 : i64, ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyModuleNotFoundError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyModuleNotFoundError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyModuleNotFoundError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ModuleNotFoundError : memref<19xi8>
    %name_dyn = memref.cast %name_ref : memref<19xi8> to memref<?xi8>
    %name_len = arith.constant 19 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_MemoryError : memref<11xi8> = dense<[77, 101, 109, 111, 114, 121, 69, 114, 114, 111, 114]>
  func.func @LyMemoryError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 109 : i64, ly.runtime.contract = "builtins.MemoryError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyMemoryError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.MemoryError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyMemoryError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.MemoryError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyMemoryError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.MemoryError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_MemoryError : memref<11xi8>
    %name_dyn = memref.cast %name_ref : memref<11xi8> to memref<?xi8>
    %name_len = arith.constant 11 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_NameError : memref<9xi8> = dense<[78, 97, 109, 101, 69, 114, 114, 111, 114]>
  func.func @LyNameError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 110 : i64, ly.runtime.contract = "builtins.NameError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyNameError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.NameError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyNameError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NameError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNameError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NameError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_NameError : memref<9xi8>
    %name_dyn = memref.cast %name_ref : memref<9xi8> to memref<?xi8>
    %name_len = arith.constant 9 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_UnboundLocalError : memref<17xi8> = dense<[85, 110, 98, 111, 117, 110, 100, 76, 111, 99, 97, 108, 69, 114, 114, 111, 114]>
  func.func @LyUnboundLocalError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 111 : i64, ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnboundLocalError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnboundLocalError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnboundLocalError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_UnboundLocalError : memref<17xi8>
    %name_dyn = memref.cast %name_ref : memref<17xi8> to memref<?xi8>
    %name_len = arith.constant 17 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_AttributeError : memref<14xi8> = dense<[65, 116, 116, 114, 105, 98, 117, 116, 101, 69, 114, 114, 111, 114]>
  func.func @LyAttributeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 112 : i64, ly.runtime.contract = "builtins.AttributeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyAttributeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.AttributeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyAttributeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AttributeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAttributeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AttributeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_AttributeError : memref<14xi8>
    %name_dyn = memref.cast %name_ref : memref<14xi8> to memref<?xi8>
    %name_len = arith.constant 14 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ReferenceError : memref<14xi8> = dense<[82, 101, 102, 101, 114, 101, 110, 99, 101, 69, 114, 114, 111, 114]>
  func.func @LyReferenceError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 113 : i64, ly.runtime.contract = "builtins.ReferenceError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyReferenceError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ReferenceError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyReferenceError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ReferenceError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyReferenceError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ReferenceError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ReferenceError : memref<14xi8>
    %name_dyn = memref.cast %name_ref : memref<14xi8> to memref<?xi8>
    %name_len = arith.constant 14 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_NotImplementedError : memref<19xi8> = dense<[78, 111, 116, 73, 109, 112, 108, 101, 109, 101, 110, 116, 101, 100, 69, 114, 114, 111, 114]>
  func.func @LyNotImplementedError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 114 : i64, ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyNotImplementedError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyNotImplementedError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNotImplementedError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_NotImplementedError : memref<19xi8>
    %name_dyn = memref.cast %name_ref : memref<19xi8> to memref<?xi8>
    %name_len = arith.constant 19 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_RecursionError : memref<14xi8> = dense<[82, 101, 99, 117, 114, 115, 105, 111, 110, 69, 114, 114, 111, 114]>
  func.func @LyRecursionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 115 : i64, ly.runtime.contract = "builtins.RecursionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyRecursionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.RecursionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyRecursionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RecursionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRecursionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RecursionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_RecursionError : memref<14xi8>
    %name_dyn = memref.cast %name_ref : memref<14xi8> to memref<?xi8>
    %name_len = arith.constant 14 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_PythonFinalizationError : memref<23xi8> = dense<[80, 121, 116, 104, 111, 110, 70, 105, 110, 97, 108, 105, 122, 97, 116, 105, 111, 110, 69, 114, 114, 111, 114]>
  func.func @LyPythonFinalizationError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 116 : i64, ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyPythonFinalizationError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyPythonFinalizationError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPythonFinalizationError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_PythonFinalizationError : memref<23xi8>
    %name_dyn = memref.cast %name_ref : memref<23xi8> to memref<?xi8>
    %name_len = arith.constant 23 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_SyntaxError : memref<11xi8> = dense<[83, 121, 110, 116, 97, 120, 69, 114, 114, 111, 114]>
  func.func @LySyntaxError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 117 : i64, ly.runtime.contract = "builtins.SyntaxError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LySyntaxError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.SyntaxError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LySyntaxError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SyntaxError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySyntaxError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SyntaxError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_SyntaxError : memref<11xi8>
    %name_dyn = memref.cast %name_ref : memref<11xi8> to memref<?xi8>
    %name_len = arith.constant 11 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_IndentationError : memref<16xi8> = dense<[73, 110, 100, 101, 110, 116, 97, 116, 105, 111, 110, 69, 114, 114, 111, 114]>
  func.func @LyIndentationError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 118 : i64, ly.runtime.contract = "builtins.IndentationError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyIndentationError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.IndentationError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyIndentationError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndentationError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndentationError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndentationError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_IndentationError : memref<16xi8>
    %name_dyn = memref.cast %name_ref : memref<16xi8> to memref<?xi8>
    %name_len = arith.constant 16 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_TabError : memref<8xi8> = dense<[84, 97, 98, 69, 114, 114, 111, 114]>
  func.func @LyTabError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 119 : i64, ly.runtime.contract = "builtins.TabError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyTabError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.TabError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyTabError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TabError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTabError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TabError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_TabError : memref<8xi8>
    %name_dyn = memref.cast %name_ref : memref<8xi8> to memref<?xi8>
    %name_len = arith.constant 8 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_SystemError : memref<11xi8> = dense<[83, 121, 115, 116, 101, 109, 69, 114, 114, 111, 114]>
  func.func @LySystemError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 120 : i64, ly.runtime.contract = "builtins.SystemError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LySystemError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.SystemError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LySystemError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_SystemError : memref<11xi8>
    %name_dyn = memref.cast %name_ref : memref<11xi8> to memref<?xi8>
    %name_len = arith.constant 11 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_UnicodeError : memref<12xi8> = dense<[85, 110, 105, 99, 111, 100, 101, 69, 114, 114, 111, 114]>
  func.func @LyUnicodeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 121 : i64, ly.runtime.contract = "builtins.UnicodeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_UnicodeError : memref<12xi8>
    %name_dyn = memref.cast %name_ref : memref<12xi8> to memref<?xi8>
    %name_len = arith.constant 12 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_UnicodeDecodeError : memref<18xi8> = dense<[85, 110, 105, 99, 111, 100, 101, 68, 101, 99, 111, 100, 101, 69, 114, 114, 111, 114]>
  func.func @LyUnicodeDecodeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 122 : i64, ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeDecodeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeDecodeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeDecodeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_UnicodeDecodeError : memref<18xi8>
    %name_dyn = memref.cast %name_ref : memref<18xi8> to memref<?xi8>
    %name_len = arith.constant 18 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_UnicodeEncodeError : memref<18xi8> = dense<[85, 110, 105, 99, 111, 100, 101, 69, 110, 99, 111, 100, 101, 69, 114, 114, 111, 114]>
  func.func @LyUnicodeEncodeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 123 : i64, ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeEncodeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeEncodeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeEncodeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_UnicodeEncodeError : memref<18xi8>
    %name_dyn = memref.cast %name_ref : memref<18xi8> to memref<?xi8>
    %name_len = arith.constant 18 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_UnicodeTranslateError : memref<21xi8> = dense<[85, 110, 105, 99, 111, 100, 101, 84, 114, 97, 110, 115, 108, 97, 116, 101, 69, 114, 114, 111, 114]>
  func.func @LyUnicodeTranslateError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 124 : i64, ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeTranslateError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeTranslateError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeTranslateError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_UnicodeTranslateError : memref<21xi8>
    %name_dyn = memref.cast %name_ref : memref<21xi8> to memref<?xi8>
    %name_len = arith.constant 21 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_Warning : memref<7xi8> = dense<[87, 97, 114, 110, 105, 110, 103]>
  func.func @LyWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 125 : i64, ly.runtime.contract = "builtins.Warning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.Warning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Warning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Warning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_Warning : memref<7xi8>
    %name_dyn = memref.cast %name_ref : memref<7xi8> to memref<?xi8>
    %name_len = arith.constant 7 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_BytesWarning : memref<12xi8> = dense<[66, 121, 116, 101, 115, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyBytesWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 126 : i64, ly.runtime.contract = "builtins.BytesWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBytesWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BytesWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBytesWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BytesWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytesWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BytesWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_BytesWarning : memref<12xi8>
    %name_dyn = memref.cast %name_ref : memref<12xi8> to memref<?xi8>
    %name_len = arith.constant 12 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_DeprecationWarning : memref<18xi8> = dense<[68, 101, 112, 114, 101, 99, 97, 116, 105, 111, 110, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyDeprecationWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 127 : i64, ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyDeprecationWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyDeprecationWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyDeprecationWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_DeprecationWarning : memref<18xi8>
    %name_dyn = memref.cast %name_ref : memref<18xi8> to memref<?xi8>
    %name_len = arith.constant 18 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_EncodingWarning : memref<15xi8> = dense<[69, 110, 99, 111, 100, 105, 110, 103, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyEncodingWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 128 : i64, ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyEncodingWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyEncodingWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyEncodingWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_EncodingWarning : memref<15xi8>
    %name_dyn = memref.cast %name_ref : memref<15xi8> to memref<?xi8>
    %name_len = arith.constant 15 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_FutureWarning : memref<13xi8> = dense<[70, 117, 116, 117, 114, 101, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyFutureWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 129 : i64, ly.runtime.contract = "builtins.FutureWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyFutureWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.FutureWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyFutureWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FutureWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFutureWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FutureWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_FutureWarning : memref<13xi8>
    %name_dyn = memref.cast %name_ref : memref<13xi8> to memref<?xi8>
    %name_len = arith.constant 13 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ImportWarning : memref<13xi8> = dense<[73, 109, 112, 111, 114, 116, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyImportWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 130 : i64, ly.runtime.contract = "builtins.ImportWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyImportWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ImportWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyImportWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ImportWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyImportWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ImportWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ImportWarning : memref<13xi8>
    %name_dyn = memref.cast %name_ref : memref<13xi8> to memref<?xi8>
    %name_len = arith.constant 13 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_PendingDeprecationWarning : memref<25xi8> = dense<[80, 101, 110, 100, 105, 110, 103, 68, 101, 112, 114, 101, 99, 97, 116, 105, 111, 110, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyPendingDeprecationWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 131 : i64, ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyPendingDeprecationWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyPendingDeprecationWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPendingDeprecationWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_PendingDeprecationWarning : memref<25xi8>
    %name_dyn = memref.cast %name_ref : memref<25xi8> to memref<?xi8>
    %name_len = arith.constant 25 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ResourceWarning : memref<15xi8> = dense<[82, 101, 115, 111, 117, 114, 99, 101, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyResourceWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 132 : i64, ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyResourceWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyResourceWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyResourceWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ResourceWarning : memref<15xi8>
    %name_dyn = memref.cast %name_ref : memref<15xi8> to memref<?xi8>
    %name_len = arith.constant 15 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_RuntimeWarning : memref<14xi8> = dense<[82, 117, 110, 116, 105, 109, 101, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyRuntimeWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 133 : i64, ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyRuntimeWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyRuntimeWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_RuntimeWarning : memref<14xi8>
    %name_dyn = memref.cast %name_ref : memref<14xi8> to memref<?xi8>
    %name_len = arith.constant 14 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_SyntaxWarning : memref<13xi8> = dense<[83, 121, 110, 116, 97, 120, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LySyntaxWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 134 : i64, ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LySyntaxWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LySyntaxWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySyntaxWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_SyntaxWarning : memref<13xi8>
    %name_dyn = memref.cast %name_ref : memref<13xi8> to memref<?xi8>
    %name_len = arith.constant 13 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_UnicodeWarning : memref<14xi8> = dense<[85, 110, 105, 99, 111, 100, 101, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyUnicodeWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 135 : i64, ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_UnicodeWarning : memref<14xi8>
    %name_dyn = memref.cast %name_ref : memref<14xi8> to memref<?xi8>
    %name_len = arith.constant 14 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_UserWarning : memref<11xi8> = dense<[85, 115, 101, 114, 87, 97, 114, 110, 105, 110, 103]>
  func.func @LyUserWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 136 : i64, ly.runtime.contract = "builtins.UserWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUserWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UserWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUserWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UserWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUserWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UserWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_UserWarning : memref<11xi8>
    %name_dyn = memref.cast %name_ref : memref<11xi8> to memref<?xi8>
    %name_len = arith.constant 11 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_BlockingIOError : memref<15xi8> = dense<[66, 108, 111, 99, 107, 105, 110, 103, 73, 79, 69, 114, 114, 111, 114]>
  func.func @LyBlockingIOError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 137 : i64, ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBlockingIOError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBlockingIOError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBlockingIOError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_BlockingIOError : memref<15xi8>
    %name_dyn = memref.cast %name_ref : memref<15xi8> to memref<?xi8>
    %name_len = arith.constant 15 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ChildProcessError : memref<17xi8> = dense<[67, 104, 105, 108, 100, 80, 114, 111, 99, 101, 115, 115, 69, 114, 114, 111, 114]>
  func.func @LyChildProcessError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 138 : i64, ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyChildProcessError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyChildProcessError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyChildProcessError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ChildProcessError : memref<17xi8>
    %name_dyn = memref.cast %name_ref : memref<17xi8> to memref<?xi8>
    %name_len = arith.constant 17 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ConnectionError : memref<15xi8> = dense<[67, 111, 110, 110, 101, 99, 116, 105, 111, 110, 69, 114, 114, 111, 114]>
  func.func @LyConnectionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 139 : i64, ly.runtime.contract = "builtins.ConnectionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyConnectionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ConnectionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyConnectionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ConnectionError : memref<15xi8>
    %name_dyn = memref.cast %name_ref : memref<15xi8> to memref<?xi8>
    %name_len = arith.constant 15 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_BrokenPipeError : memref<15xi8> = dense<[66, 114, 111, 107, 101, 110, 80, 105, 112, 101, 69, 114, 114, 111, 114]>
  func.func @LyBrokenPipeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 140 : i64, ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBrokenPipeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBrokenPipeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBrokenPipeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_BrokenPipeError : memref<15xi8>
    %name_dyn = memref.cast %name_ref : memref<15xi8> to memref<?xi8>
    %name_len = arith.constant 15 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ConnectionAbortedError : memref<22xi8> = dense<[67, 111, 110, 110, 101, 99, 116, 105, 111, 110, 65, 98, 111, 114, 116, 101, 100, 69, 114, 114, 111, 114]>
  func.func @LyConnectionAbortedError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 141 : i64, ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyConnectionAbortedError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyConnectionAbortedError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionAbortedError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ConnectionAbortedError : memref<22xi8>
    %name_dyn = memref.cast %name_ref : memref<22xi8> to memref<?xi8>
    %name_len = arith.constant 22 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ConnectionRefusedError : memref<22xi8> = dense<[67, 111, 110, 110, 101, 99, 116, 105, 111, 110, 82, 101, 102, 117, 115, 101, 100, 69, 114, 114, 111, 114]>
  func.func @LyConnectionRefusedError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 142 : i64, ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyConnectionRefusedError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyConnectionRefusedError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionRefusedError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ConnectionRefusedError : memref<22xi8>
    %name_dyn = memref.cast %name_ref : memref<22xi8> to memref<?xi8>
    %name_len = arith.constant 22 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ConnectionResetError : memref<20xi8> = dense<[67, 111, 110, 110, 101, 99, 116, 105, 111, 110, 82, 101, 115, 101, 116, 69, 114, 114, 111, 114]>
  func.func @LyConnectionResetError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 143 : i64, ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyConnectionResetError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyConnectionResetError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionResetError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ConnectionResetError : memref<20xi8>
    %name_dyn = memref.cast %name_ref : memref<20xi8> to memref<?xi8>
    %name_len = arith.constant 20 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_FileExistsError : memref<15xi8> = dense<[70, 105, 108, 101, 69, 120, 105, 115, 116, 115, 69, 114, 114, 111, 114]>
  func.func @LyFileExistsError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 144 : i64, ly.runtime.contract = "builtins.FileExistsError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyFileExistsError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.FileExistsError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyFileExistsError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileExistsError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileExistsError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileExistsError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_FileExistsError : memref<15xi8>
    %name_dyn = memref.cast %name_ref : memref<15xi8> to memref<?xi8>
    %name_len = arith.constant 15 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_InterruptedError : memref<16xi8> = dense<[73, 110, 116, 101, 114, 114, 117, 112, 116, 101, 100, 69, 114, 114, 111, 114]>
  func.func @LyInterruptedError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 145 : i64, ly.runtime.contract = "builtins.InterruptedError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyInterruptedError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.InterruptedError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyInterruptedError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.InterruptedError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyInterruptedError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.InterruptedError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_InterruptedError : memref<16xi8>
    %name_dyn = memref.cast %name_ref : memref<16xi8> to memref<?xi8>
    %name_len = arith.constant 16 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_IsADirectoryError : memref<17xi8> = dense<[73, 115, 65, 68, 105, 114, 101, 99, 116, 111, 114, 121, 69, 114, 114, 111, 114]>
  func.func @LyIsADirectoryError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 146 : i64, ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyIsADirectoryError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyIsADirectoryError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIsADirectoryError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_IsADirectoryError : memref<17xi8>
    %name_dyn = memref.cast %name_ref : memref<17xi8> to memref<?xi8>
    %name_len = arith.constant 17 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_NotADirectoryError : memref<18xi8> = dense<[78, 111, 116, 65, 68, 105, 114, 101, 99, 116, 111, 114, 121, 69, 114, 114, 111, 114]>
  func.func @LyNotADirectoryError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 147 : i64, ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyNotADirectoryError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyNotADirectoryError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNotADirectoryError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_NotADirectoryError : memref<18xi8>
    %name_dyn = memref.cast %name_ref : memref<18xi8> to memref<?xi8>
    %name_len = arith.constant 18 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_PermissionError : memref<15xi8> = dense<[80, 101, 114, 109, 105, 115, 115, 105, 111, 110, 69, 114, 114, 111, 114]>
  func.func @LyPermissionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 148 : i64, ly.runtime.contract = "builtins.PermissionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyPermissionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.PermissionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyPermissionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PermissionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPermissionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PermissionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_PermissionError : memref<15xi8>
    %name_dyn = memref.cast %name_ref : memref<15xi8> to memref<?xi8>
    %name_len = arith.constant 15 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_ProcessLookupError : memref<18xi8> = dense<[80, 114, 111, 99, 101, 115, 115, 76, 111, 111, 107, 117, 112, 69, 114, 114, 111, 114]>
  func.func @LyProcessLookupError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 149 : i64, ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyProcessLookupError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyProcessLookupError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyProcessLookupError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ProcessLookupError : memref<18xi8>
    %name_dyn = memref.cast %name_ref : memref<18xi8> to memref<?xi8>
    %name_len = arith.constant 18 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_excname_TimeoutError : memref<12xi8> = dense<[84, 105, 109, 101, 111, 117, 116, 69, 114, 114, 111, 114]>
  func.func @LyTimeoutError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 150 : i64, ly.runtime.contract = "builtins.TimeoutError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyTimeoutError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.TimeoutError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyTimeoutError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TimeoutError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTimeoutError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TimeoutError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_TimeoutError : memref<12xi8>
    %name_dyn = memref.cast %name_ref : memref<12xi8> to memref<?xi8>
    %name_len = arith.constant 12 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyboardInterrupt_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBaseExceptionGroup_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyExceptionGroup_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyFloatingPointError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyOverflowError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.OverflowError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBufferError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BufferError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyEOFError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.EOFError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyImportError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ImportError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyModuleNotFoundError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyMemoryError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.MemoryError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyNameError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.NameError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnboundLocalError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyAttributeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.AttributeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyReferenceError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ReferenceError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyNotImplementedError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyRecursionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.RecursionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyPythonFinalizationError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LySyntaxError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.SyntaxError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyIndentationError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.IndentationError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyTabError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.TabError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LySystemError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.SystemError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeDecodeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeEncodeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeTranslateError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.Warning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBytesWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BytesWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyDeprecationWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyEncodingWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyFutureWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.FutureWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyImportWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ImportWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyPendingDeprecationWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyResourceWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyRuntimeWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LySyntaxWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUserWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UserWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBlockingIOError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyChildProcessError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyConnectionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ConnectionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBrokenPipeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyConnectionAbortedError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyConnectionRefusedError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyConnectionResetError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyFileExistsError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.FileExistsError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyInterruptedError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.InterruptedError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyIsADirectoryError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyNotADirectoryError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyPermissionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.PermissionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyProcessLookupError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyTimeoutError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.TimeoutError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyException_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.Exception", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyRuntimeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyTypeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.TypeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyValueError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ValueError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyArithmeticError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyLookupError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.LookupError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyZeroDivisionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyKeyError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.KeyError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyIndexError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.IndexError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyAssertionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.AssertionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyStopIteration_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.StopIteration", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyStopAsyncIteration_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LySystemExit_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.SystemExit", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }
  func.func private @LyGeneratorExit_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyOSError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.OSError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyFileNotFoundError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyCancelledError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "asyncio.CancelledError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }


  // __str__/__repr__ return the exception message as an owned str.

  // Deviation from CPython: repr(err) also yields the bare message (not

  // "Cls('msg')"), so that print(err) -- which resolves __repr__ -- matches

  // CPython's str-based print output.

  func.func @LyBaseException_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBaseException_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_BaseException : memref<13xi8>
    %name_dyn = memref.cast %name_ref : memref<13xi8> to memref<?xi8>
    %name_len = arith.constant 13 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_Exception : memref<9xi8>
    %name_dyn = memref.cast %name_ref : memref<9xi8> to memref<?xi8>
    %name_len = arith.constant 9 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_RuntimeError : memref<12xi8>
    %name_dyn = memref.cast %name_ref : memref<12xi8> to memref<?xi8>
    %name_len = arith.constant 12 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_TypeError : memref<9xi8>
    %name_dyn = memref.cast %name_ref : memref<9xi8> to memref<?xi8>
    %name_len = arith.constant 9 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ValueError : memref<10xi8>
    %name_dyn = memref.cast %name_ref : memref<10xi8> to memref<?xi8>
    %name_len = arith.constant 10 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ArithmeticError : memref<15xi8>
    %name_dyn = memref.cast %name_ref : memref<15xi8> to memref<?xi8>
    %name_len = arith.constant 15 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_LookupError : memref<11xi8>
    %name_dyn = memref.cast %name_ref : memref<11xi8> to memref<?xi8>
    %name_len = arith.constant 11 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_ZeroDivisionError : memref<17xi8>
    %name_dyn = memref.cast %name_ref : memref<17xi8> to memref<?xi8>
    %name_len = arith.constant 17 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_KeyError : memref<8xi8>
    %name_dyn = memref.cast %name_ref : memref<8xi8> to memref<?xi8>
    %name_len = arith.constant 8 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_IndexError : memref<10xi8>
    %name_dyn = memref.cast %name_ref : memref<10xi8> to memref<?xi8>
    %name_len = arith.constant 10 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_AssertionError : memref<14xi8>
    %name_dyn = memref.cast %name_ref : memref<14xi8> to memref<?xi8>
    %name_len = arith.constant 14 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_StopIteration : memref<13xi8>
    %name_dyn = memref.cast %name_ref : memref<13xi8> to memref<?xi8>
    %name_len = arith.constant 13 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_StopAsyncIteration : memref<18xi8>
    %name_dyn = memref.cast %name_ref : memref<18xi8> to memref<?xi8>
    %name_len = arith.constant 18 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyCancelledError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "asyncio.CancelledError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyCancelledError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "asyncio.CancelledError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_CancelledError : memref<14xi8>
    %name_dyn = memref.cast %name_ref : memref<14xi8> to memref<?xi8>
    %name_len = arith.constant 14 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemExit", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemExit", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_SystemExit : memref<10xi8>
    %name_dyn = memref.cast %name_ref : memref<10xi8> to memref<?xi8>
    %name_len = arith.constant 10 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyGeneratorExit_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyGeneratorExit_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_GeneratorExit : memref<13xi8>
    %name_dyn = memref.cast %name_ref : memref<13xi8> to memref<?xi8>
    %name_len = arith.constant 13 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OSError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OSError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_OSError : memref<7xi8>
    %name_dyn = memref.cast %name_ref : memref<7xi8> to memref<?xi8>
    %name_len = arith.constant 7 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %name_ref = memref.get_global @__ly_excname_FileNotFoundError : memref<17xi8>
    %name_dyn = memref.cast %name_ref : memref<17xi8> to memref<?xi8>
    %name_len = arith.constant 17 : i64
    %result:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // ===== impls: bytes =====
  // Physical twin of str ([0,16) header + byte payload in one entity block)
  // with byte-oriented semantics: __len__/__getitem__ count BYTES (str
  // counts codepoints), __getitem__ returns int, repr spells b'...'.
  func.func private @LyBytes_Shape() -> (memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.shape}

  memref.global "private" constant @__ly_bytes_msg_index_out_of_range : memref<18xi8> = dense<[105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>

  func.func private @__ly_bytes_raise_index_error() {
    %class_id = arith.constant 55 : i64
    %length = arith.constant 18 : i64
    %message_static = memref.get_global @__ly_bytes_msg_index_out_of_range : memref<18xi8>
    %message = memref.cast %message_static : memref<18xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_bytes_alloc(%len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.primitive = "alloc"} {
    %byte_count = arith.index_cast %len : i64 to index
    %block_prefix = arith.constant 16 : index
    %block_bytes = arith.addi %byte_count, %block_prefix : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %bytes_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %bytes = memref.view %block[%bytes_offset][%byte_count] : memref<?xi8> to memref<?xi8>
    %one = arith.constant 1 : i64
    %layout_bytes = arith.constant 70 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_bytes, %header[%layout_slot] : memref<2xi64>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytes_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"} {
    %header, %result_bytes = func.call @__ly_bytes_alloc(%len) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %byte_count = arith.index_cast %len : i64 to index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %index = %lower to %byte_count step %step {
      %source_index = arith.addi %start, %index : index
      %byte = memref.load %bytes[%source_index] : memref<?xi8>
      memref.store %byte, %result_bytes[%index] : memref<?xi8>
    }
    func.return %header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytes_Len(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__len__"} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    func.return %length : i64
  }

  func.func @LyBytes_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__bool__"} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %non_empty = arith.cmpi ne, %dim, %zero : index
    func.return %non_empty : i1
  }

  // bytes[i] is the byte VALUE (an int), unlike str's one-element slice.
  func.func @LyBytes_GetItem(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %raw_index: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__getitem__", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    %is_negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %length : i64
    %index = arith.select %is_negative, %from_end, %raw_index : i1, i64
    %lower_ok = arith.cmpi sge, %index, %zero : i64
    %upper_ok = arith.cmpi slt, %index, %length : i64
    %valid = arith.andi %lower_ok, %upper_ok : i1
    %value = scf.if %valid -> (i64) {
      %at = arith.index_cast %index : i64 to index
      %byte = memref.load %bytes[%at] : memref<?xi8>
      %wide = arith.extui %byte : i8 to i64
      scf.yield %wide : i64
    } else {
      func.call @__ly_bytes_raise_index_error() : () -> ()
      scf.yield %zero : i64
    }
    %result:3 = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyBytes_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__eq__"} {
    %c0 = arith.constant 0 : index
    %lhs_ptr_index = memref.extract_aligned_pointer_as_index %lhs_bytes : memref<?xi8> -> index
    %lhs_ptr = arith.index_cast %lhs_ptr_index : index to i64
    %lhs_dim = memref.dim %lhs_bytes, %c0 : memref<?xi8>
    %lhs_len = arith.index_cast %lhs_dim : index to i64
    %rhs_ptr_index = memref.extract_aligned_pointer_as_index %rhs_bytes : memref<?xi8> -> index
    %rhs_ptr = arith.index_cast %rhs_ptr_index : index to i64
    %rhs_dim = memref.dim %rhs_bytes, %c0 : memref<?xi8>
    %rhs_len = arith.index_cast %rhs_dim : index to i64
    %equal = func.call @raw_bytes_equal(%lhs_ptr, %lhs_len, %rhs_ptr, %rhs_len) : (i64, i64, i64, i64) -> i1
    func.return %equal : i1
  }

  func.func @LyBytes_NeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__ne__"} {
    %equal = func.call @LyBytes_EqBool(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i1
    %true_bit = arith.constant true
    %not_equal = arith.xori %equal, %true_bit : i1
    func.return %not_equal : i1
  }

  func.func @LyBytes_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__add__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_dim = memref.dim %lhs_bytes, %c0 : memref<?xi8>
    %rhs_dim = memref.dim %rhs_bytes, %c0 : memref<?xi8>
    %total_index = arith.addi %lhs_dim, %rhs_dim : index
    %total = arith.index_cast %total_index : index to i64
    %header, %bytes = func.call @__ly_bytes_alloc(%total) : (i64) -> (memref<2xi64>, memref<?xi8>)
    scf.for %i = %c0 to %lhs_dim step %c1 {
      %byte = memref.load %lhs_bytes[%i] : memref<?xi8>
      memref.store %byte, %bytes[%i] : memref<?xi8>
    }
    scf.for %i = %c0 to %rhs_dim step %c1 {
      %byte = memref.load %rhs_bytes[%i] : memref<?xi8>
      %dest = arith.addi %lhs_dim, %i : index
      memref.store %byte, %bytes[%dest] : memref<?xi8>
    }
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // bytes_repr (Objects/bytesobject.c): b'...' with \t \n \r \' \\ kept as
  // two-character escapes, other non-printable bytes as \xHH. CPython picks
  // double quotes when the payload contains ' but not "; this port always
  // uses single quotes and escapes ' instead.
  func.func @LyBytes_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>

    %tab = arith.constant 9 : i8
    %newline = arith.constant 10 : i8
    %carriage = arith.constant 13 : i8
    %quote = arith.constant 39 : i8
    %backslash = arith.constant 92 : i8
    %space = arith.constant 32 : i8
    %tilde_plus_one = arith.constant 127 : i8

    // Pass 1: output length (b + quote + payload + quote).
    %payload_len = scf.for %i = %c0 to %dim step %c1 iter_args(%acc = %c0) -> (index) {
      %byte = memref.load %bytes[%i] : memref<?xi8>
      %is_tab = arith.cmpi eq, %byte, %tab : i8
      %is_nl = arith.cmpi eq, %byte, %newline : i8
      %is_cr = arith.cmpi eq, %byte, %carriage : i8
      %is_quote = arith.cmpi eq, %byte, %quote : i8
      %is_bs = arith.cmpi eq, %byte, %backslash : i8
      %pair_a = arith.ori %is_tab, %is_nl : i1
      %pair_b = arith.ori %pair_a, %is_cr : i1
      %pair = arith.ori %pair_b, %is_quote : i1
      %pair_full = arith.ori %pair, %is_bs : i1
      %ge_space = arith.cmpi uge, %byte, %space : i8
      %lt_del = arith.cmpi ult, %byte, %tilde_plus_one : i8
      %printable = arith.andi %ge_space, %lt_del : i1
      %escaped_two = arith.select %pair_full, %two, %four : index
      %width = arith.select %printable, %one, %escaped_two : index
      %width_final = arith.select %pair_full, %two, %width : index
      %next = arith.addi %acc, %width_final : index
      scf.yield %next : index
    }
    %prefix = arith.constant 3 : index
    %out_len_index = arith.addi %payload_len, %prefix : index
    %out = memref.alloc(%out_len_index) : memref<?xi8>

    // Pass 2: fill.
    %b_char = arith.constant 98 : i8
    memref.store %b_char, %out[%c0] : memref<?xi8>
    memref.store %quote, %out[%c1] : memref<?xi8>
    %payload_start = arith.constant 2 : index
    %end_pos = scf.for %i = %c0 to %dim step %c1 iter_args(%pos = %payload_start) -> (index) {
      %byte = memref.load %bytes[%i] : memref<?xi8>
      %is_tab = arith.cmpi eq, %byte, %tab : i8
      %is_nl = arith.cmpi eq, %byte, %newline : i8
      %is_cr = arith.cmpi eq, %byte, %carriage : i8
      %is_quote = arith.cmpi eq, %byte, %quote : i8
      %is_bs = arith.cmpi eq, %byte, %backslash : i8
      %pair_a = arith.ori %is_tab, %is_nl : i1
      %pair_b = arith.ori %pair_a, %is_cr : i1
      %pair_c = arith.ori %pair_b, %is_quote : i1
      %pair_full = arith.ori %pair_c, %is_bs : i1
      %ge_space = arith.cmpi uge, %byte, %space : i8
      %lt_del = arith.cmpi ult, %byte, %tilde_plus_one : i8
      %printable_raw = arith.andi %ge_space, %lt_del : i1
      %true_bit = arith.constant true
      %not_pair = arith.xori %pair_full, %true_bit : i1
      %printable = arith.andi %printable_raw, %not_pair : i1
      %next = scf.if %printable -> (index) {
        memref.store %byte, %out[%pos] : memref<?xi8>
        %advanced = arith.addi %pos, %one : index
        scf.yield %advanced : index
      } else {
        %escaped = scf.if %pair_full -> (index) {
          memref.store %backslash, %out[%pos] : memref<?xi8>
          %second_pos = arith.addi %pos, %one : index
          %t_char = arith.constant 116 : i8
          %n_char = arith.constant 110 : i8
          %r_char = arith.constant 114 : i8
          %escape_a = arith.select %is_tab, %t_char, %byte : i8
          %escape_b = arith.select %is_nl, %n_char, %escape_a : i8
          %escape_c = arith.select %is_cr, %r_char, %escape_b : i8
          memref.store %escape_c, %out[%second_pos] : memref<?xi8>
          %advanced = arith.addi %pos, %two : index
          scf.yield %advanced : index
        } else {
          // \xHH
          memref.store %backslash, %out[%pos] : memref<?xi8>
          %x_pos = arith.addi %pos, %one : index
          %x_char = arith.constant 120 : i8
          memref.store %x_char, %out[%x_pos] : memref<?xi8>
          %wide = arith.extui %byte : i8 to i64
          %sixteen = arith.constant 16 : i64
          %high = arith.divui %wide, %sixteen : i64
          %low = arith.remui %wide, %sixteen : i64
          %ten = arith.constant 10 : i64
          %digit_base = arith.constant 48 : i64
          %alpha_base = arith.constant 87 : i64
          %high_is_alpha = arith.cmpi uge, %high, %ten : i64
          %high_base = arith.select %high_is_alpha, %alpha_base, %digit_base : i64
          %high_char_wide = arith.addi %high, %high_base : i64
          %high_char = arith.trunci %high_char_wide : i64 to i8
          %low_is_alpha = arith.cmpi uge, %low, %ten : i64
          %low_base = arith.select %low_is_alpha, %alpha_base, %digit_base : i64
          %low_char_wide = arith.addi %low, %low_base : i64
          %low_char = arith.trunci %low_char_wide : i64 to i8
          %high_pos = arith.addi %pos, %two : index
          memref.store %high_char, %out[%high_pos] : memref<?xi8>
          %three = arith.constant 3 : index
          %low_pos = arith.addi %pos, %three : index
          memref.store %low_char, %out[%low_pos] : memref<?xi8>
          %advanced = arith.addi %pos, %four : index
          scf.yield %advanced : index
        }
        scf.yield %escaped : index
      }
      scf.yield %next : index
    }
    memref.store %quote, %out[%end_pos] : memref<?xi8>

    %out_len = arith.index_cast %out_len_index : index to i64
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%out, %c0, %out_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %out : memref<?xi8>
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytes_Str(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @LyBytes_Repr(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // decode() with the default encoding: LyUnicode_FromBytes decodes the
  // UTF-8 into the adaptive-width payload and raises on invalid input.
  func.func @LyBytes_Decode(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%bytes, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // str.encode() with the default encoding: re-encode the adaptive-width
  // code units to UTF-8.
  func.func @LyUnicode_Encode(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "encode", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %length = func.call @__ly_unicode_utf8_length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %length_index = arith.index_cast %length : i64 to index
    %buffer = memref.alloc(%length_index) : memref<?xi8>
    func.call @__ly_unicode_utf8_fill(%header, %bytes, %buffer) : (memref<2xi64>, memref<?xi8>, memref<?xi8>) -> ()
    %result_header, %result_bytes = func.call @LyBytes_FromBytes(%buffer, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // Same release interface as str (header only): the two contracts are
  // physical twins, and a wider input list here would win the
  // longest-inputTypes disambiguation and hijack str groups; the ownership
  // collector tells the twins apart by ly.runtime.result_contract instead.

  // ===== bytes methods (Wave 1) =====

  // "unknown encoding: "
  memref.global "private" constant @__ly_bytes_msg_unknown_encoding : memref<18xi8> = dense<[117, 110, 107, 110, 111, 119, 110, 32, 101, 110, 99, 111, 100, 105, 110, 103, 58, 32]>
  // "only 'strict' error handling is supported for bytes.decode"
  memref.global "private" constant @__ly_bytes_msg_bad_errors : memref<58xi8> = dense<[111, 110, 108, 121, 32, 39, 115, 116, 114, 105, 99, 116, 39, 32, 101, 114, 114, 111, 114, 32, 104, 97, 110, 100, 108, 105, 110, 103, 32, 105, 115, 32, 115, 117, 112, 112, 111, 114, 116, 101, 100, 32, 102, 111, 114, 32, 98, 121, 116, 101, 115, 46, 100, 101, 99, 111, 100, 101]>
  // "empty separator"
  // (reuses @__ly_unicode_msg_empty_separator)
  // "subsection not found"
  memref.global "private" constant @__ly_bytes_msg_subsection_not_found : memref<20xi8> = dense<[115, 117, 98, 115, 101, 99, 116, 105, 111, 110, 32, 110, 111, 116, 32, 102, 111, 117, 110, 100]>
  // "non-hexadecimal number found in fromhex() arg at position "
  memref.global "private" constant @__ly_bytes_msg_fromhex : memref<58xi8> = dense<[110, 111, 110, 45, 104, 101, 120, 97, 100, 101, 99, 105, 109, 97, 108, 32, 110, 117, 109, 98, 101, 114, 32, 102, 111, 117, 110, 100, 32, 105, 110, 32, 102, 114, 111, 109, 104, 101, 120, 40, 41, 32, 97, 114, 103, 32, 97, 116, 32, 112, 111, 115, 105, 116, 105, 111, 110, 32]>

  // ASCII-insensitive equality of a str operand against an ASCII literal:
  // encoding/error-handler names are latin-1 width by construction.
  func.func private @__ly_unicode_equals_ascii(%header: memref<2xi64>, %bytes: memref<?xi8>, %expected: memref<?xi8>, %expected_len: i64) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %true_bit = arith.constant true
    %false_bit = arith.constant false
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %narrow = arith.cmpi eq, %width, %one : i64
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %same_len = arith.cmpi eq, %len, %expected_len : i64
    %comparable = arith.andi %narrow, %same_len : i1
    %result = scf.if %comparable -> (i1) {
      %expected_index = arith.index_cast %expected_len : i64 to index
      %all = scf.for %i = %c0 to %expected_index step %c1 iter_args(%acc = %true_bit) -> (i1) {
        %a = memref.load %bytes[%i] : memref<?xi8>
        %b = memref.load %expected[%i] : memref<?xi8>
        %eq = arith.cmpi eq, %a, %b : i8
        %next = arith.andi %acc, %eq : i1
        scf.yield %next : i1
      }
      scf.yield %all : i1
    } else {
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  // "utf-8" / "utf8" / "UTF-8" / "UTF8" (the spellings CPython sees most;
  // full codec-name normalization is out of scope until a codec registry
  // exists).
  func.func private @__ly_bytes_encoding_is_utf8(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i1 {
    %five = arith.constant 5 : i64
    %four = arith.constant 4 : i64
    %e1_ref = memref.get_global @__ly_bytes_enc_utf8_dash : memref<5xi8>
    %e1 = memref.cast %e1_ref : memref<5xi8> to memref<?xi8>
    %m1 = func.call @__ly_unicode_equals_ascii(%header, %bytes, %e1, %five) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %e2_ref = memref.get_global @__ly_bytes_enc_utf8 : memref<4xi8>
    %e2 = memref.cast %e2_ref : memref<4xi8> to memref<?xi8>
    %m2 = func.call @__ly_unicode_equals_ascii(%header, %bytes, %e2, %four) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %e3_ref = memref.get_global @__ly_bytes_enc_utf8_upper_dash : memref<5xi8>
    %e3 = memref.cast %e3_ref : memref<5xi8> to memref<?xi8>
    %m3 = func.call @__ly_unicode_equals_ascii(%header, %bytes, %e3, %five) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %e4_ref = memref.get_global @__ly_bytes_enc_utf8_upper : memref<4xi8>
    %e4 = memref.cast %e4_ref : memref<4xi8> to memref<?xi8>
    %m4 = func.call @__ly_unicode_equals_ascii(%header, %bytes, %e4, %four) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %a = arith.ori %m1, %m2 : i1
    %b = arith.ori %m3, %m4 : i1
    %result = arith.ori %a, %b : i1
    func.return %result : i1
  }
  memref.global "private" constant @__ly_bytes_enc_utf8_dash : memref<5xi8> = dense<[117, 116, 102, 45, 56]>
  memref.global "private" constant @__ly_bytes_enc_utf8 : memref<4xi8> = dense<[117, 116, 102, 56]>
  memref.global "private" constant @__ly_bytes_enc_utf8_upper_dash : memref<5xi8> = dense<[85, 84, 70, 45, 56]>
  memref.global "private" constant @__ly_bytes_enc_utf8_upper : memref<4xi8> = dense<[85, 84, 70, 56]>
  memref.global "private" constant @__ly_bytes_err_strict : memref<6xi8> = dense<[115, 116, 114, 105, 99, 116]>

  // bytes.decode(encoding): utf-8 only until a codec registry exists;
  // anything else raises LookupError like CPython's codec lookup.
  func.func @LyBytes_DecodeEncoding(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %enc_header: memref<2xi64> {ly.ownership.object_header}, %enc_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %is_utf8 = func.call @__ly_bytes_encoding_is_utf8(%enc_header, %enc_bytes) : (memref<2xi64>, memref<?xi8>) -> i1
    %true_bit = arith.constant true
    %bad = arith.xori %is_utf8, %true_bit : i1
    scf.if %bad {
      %class_id = arith.constant 60 : i64
      %length = arith.constant 18 : i64
      %start = arith.constant 0 : index
      %static = memref.get_global @__ly_bytes_msg_unknown_encoding : memref<18xi8>
      %message = memref.cast %static : memref<18xi8> to memref<?xi8>
      %prefix_h, %prefix_b = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %msg_h, %msg_b = func.call @LyUnicode_Concat(%prefix_h, %prefix_b, %enc_header, %enc_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%prefix_h) : (memref<2xi64>) -> ()
      %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
      %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %msg_h, %msg_b) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
      func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    }
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %result:2 = func.call @LyUnicode_FromBytes(%bytes, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // bytes.decode(encoding, errors): only 'strict' exists here, and it is
  // checked eagerly (CPython defers to first malformed input; deferring
  // would silently accept typos on clean input).
  func.func @LyBytes_DecodeEncodingErrors(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %enc_header: memref<2xi64> {ly.ownership.object_header}, %enc_bytes: memref<?xi8>, %err_header: memref<2xi64> {ly.ownership.object_header}, %err_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %six = arith.constant 6 : i64
    %strict_ref = memref.get_global @__ly_bytes_err_strict : memref<6xi8>
    %strict = memref.cast %strict_ref : memref<6xi8> to memref<?xi8>
    %is_strict = func.call @__ly_unicode_equals_ascii(%err_header, %err_bytes, %strict, %six) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %true_bit = arith.constant true
    %bad = arith.xori %is_strict, %true_bit : i1
    scf.if %bad {
      %class_id = arith.constant 60 : i64
      %length = arith.constant 58 : i64
      %static = memref.get_global @__ly_bytes_msg_bad_errors : memref<58xi8>
      %message = memref.cast %static : memref<58xi8> to memref<?xi8>
      func.call @__ly_unicode_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %result:2 = func.call @LyBytes_DecodeEncoding(%header, %bytes, %enc_header, %enc_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_bytes_match_at(%s: memref<?xi8>, %si: index, %t: memref<?xi8>, %n: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %all = scf.for %j = %c0 to %n step %c1 iter_args(%acc = %true_bit) -> (i1) {
      %sj = arith.addi %si, %j : index
      %a = memref.load %s[%sj] : memref<?xi8>
      %b = memref.load %t[%j] : memref<?xi8>
      %eq = arith.cmpi eq, %a, %b : i8
      %next = arith.andi %acc, %eq : i1
      scf.yield %next : i1
    }
    func.return %all : i1
  }

  func.func private @__ly_bytes_find_core(%s: memref<?xi8>, %t: memref<?xi8>, %start: i64, %end: i64, %n: i64, %reverse: i1) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %limit = arith.subi %end, %n : i64
    %viable = arith.cmpi sle, %start, %limit : i64
    %found = scf.if %viable -> (i64) {
      %n_index = arith.index_cast %n : i64 to index
      %span = arith.subi %limit, %start : i64
      %positions_i64 = arith.addi %span, %one : i64
      %positions = arith.index_cast %positions_i64 : i64 to index
      %scan = scf.for %k = %c0 to %positions step %c1 iter_args(%acc = %minus_one) -> (i64) {
        %k_i64 = arith.index_cast %k : index to i64
        %fwd = arith.addi %start, %k_i64 : i64
        %rev = arith.subi %limit, %k_i64 : i64
        %pos = arith.select %reverse, %rev, %fwd : i64
        %pos_index = arith.index_cast %pos : i64 to index
        %eq = func.call @__ly_bytes_match_at(%s, %pos_index, %t, %n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
        %not_yet = arith.cmpi eq, %acc, %minus_one : i64
        %take = arith.andi %eq, %not_yet : i1
        %next = arith.select %take, %pos, %acc : i64
        scf.yield %next : i64
      }
      scf.yield %scan : i64
    } else {
      scf.yield %minus_one : i64
    }
    func.return %found : i64
  }

  func.func private @__ly_bytes_len_of(%bytes: memref<?xi8>) -> i64 {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    func.return %len : i64
  }

  func.func @LyBytes_Find(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "find", ly.runtime.result_contract = "builtins.int"} {
    %false_bit = arith.constant false
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%sub_bytes) : (memref<?xi8>) -> i64
    %found = func.call @__ly_bytes_find_core(%bytes, %sub_bytes, %start, %end, %n, %false_bit) : (memref<?xi8>, memref<?xi8>, i64, i64, i64, i1) -> i64
    %result:3 = func.call @LyLong_FromI64(%found) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyBytes_CountSub(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "count", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%sub_bytes) : (memref<?xi8>) -> i64
    %is_empty = arith.cmpi eq, %n, %zero : i64
    %total = scf.if %is_empty -> (i64) {
      %span = arith.subi %end, %start : i64
      %viable = arith.cmpi sge, %span, %zero : i64
      %hits = arith.addi %span, %one : i64
      %count = arith.select %viable, %hits, %zero : i64
      scf.yield %count : i64
    } else {
      %n_index = arith.index_cast %n : i64 to index
      %scan:2 = scf.while (%pos = %start, %count = %zero) : (i64, i64) -> (i64, i64) {
        %tail = arith.addi %pos, %n : i64
        %more = arith.cmpi sle, %tail, %end : i64
        scf.condition(%more) %pos, %count : i64, i64
      } do {
      ^bb0(%pos: i64, %count: i64):
        %pos_index = arith.index_cast %pos : i64 to index
        %eq = func.call @__ly_bytes_match_at(%bytes, %pos_index, %sub_bytes, %n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
        %skip = arith.select %eq, %n, %one : i64
        %bump = arith.select %eq, %one, %zero : i64
        %next_pos = arith.addi %pos, %skip : i64
        %next_count = arith.addi %count, %bump : i64
        scf.yield %next_pos, %next_count : i64, i64
      }
      scf.yield %scan#1 : i64
    }
    %result:3 = func.call @LyLong_FromI64(%total) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyBytes_StartsWith(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %prefix_header: memref<2xi64> {ly.ownership.object_header}, %prefix_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "startswith"} {
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%prefix_bytes) : (memref<?xi8>) -> i64
    %tail = arith.addi %start, %n : i64
    %fits = arith.cmpi sle, %tail, %end : i64
    %result = scf.if %fits -> (i1) {
      %si = arith.index_cast %start : i64 to index
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_bytes_match_at(%bytes, %si, %prefix_bytes, %n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  func.func @LyBytes_EndsWith(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %suffix_header: memref<2xi64> {ly.ownership.object_header}, %suffix_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "endswith"} {
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%suffix_bytes) : (memref<?xi8>) -> i64
    %pos = arith.subi %end, %n : i64
    %fits = arith.cmpi sge, %pos, %start : i64
    %result = scf.if %fits -> (i1) {
      %si = arith.index_cast %pos : i64 to index
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_bytes_match_at(%bytes, %si, %suffix_bytes, %n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  func.func private @__ly_bytes_slice(%bytes: memref<?xi8>, %start: index, %end: index) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %span = arith.subi %end, %start : index
    %len = arith.index_cast %span : index to i64
    %result:2 = func.call @LyBytes_FromBytes(%bytes, %start, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_bytes_byte_is_ascii_space(%b: i64) -> i1 {
    %space = arith.constant 32 : i64
    %tab = arith.constant 9 : i64
    %cr = arith.constant 13 : i64
    %is_space = arith.cmpi eq, %b, %space : i64
    %ge_tab = arith.cmpi sge, %b, %tab : i64
    %le_cr = arith.cmpi sle, %b, %cr : i64
    %ctl = arith.andi %ge_tab, %le_cr : i1
    %result = arith.ori %is_space, %ctl : i1
    func.return %result : i1
  }

  func.func private @__ly_bytes_byte_in(%b: i64, %chars: memref<?xi8>, %n: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false_bit = arith.constant false
    %found = scf.for %j = %c0 to %n step %c1 iter_args(%acc = %false_bit) -> (i1) {
      %cj_i8 = memref.load %chars[%j] : memref<?xi8>
      %cj = arith.extui %cj_i8 : i8 to i64
      %eq = arith.cmpi eq, %b, %cj : i64
      %next = arith.ori %acc, %eq : i1
      scf.yield %next : i1
    }
    func.return %found : i1
  }

  func.func private @__ly_bytes_strip_core(%bytes: memref<?xi8>, %mode: i64, %use_chars: i1, %chars: memref<?xi8>, %chars_n: index) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %true_bit = arith.constant true
    %dim = memref.dim %bytes, %c0 : memref<?xi8>

    %left_mask = arith.andi %mode, %one : i64
    %strip_left = arith.cmpi ne, %left_mask, %zero : i64
    %begin = scf.if %strip_left -> (index) {
      %scan:2 = scf.while (%i = %c0, %go = %true_bit) : (index, i1) -> (index, i1) {
        %more = arith.cmpi ult, %i, %dim : index
        %continue = arith.andi %more, %go : i1
        scf.condition(%continue) %i, %go : index, i1
      } do {
      ^bb0(%i: index, %go: i1):
        %b_i8 = memref.load %bytes[%i] : memref<?xi8>
        %b = arith.extui %b_i8 : i8 to i64
        %stripped = scf.if %use_chars -> (i1) {
          %in = func.call @__ly_bytes_byte_in(%b, %chars, %chars_n) : (i64, memref<?xi8>, index) -> i1
          scf.yield %in : i1
        } else {
          %sp = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
          scf.yield %sp : i1
        }
        %next = arith.addi %i, %c1 : index
        %keep = arith.select %stripped, %next, %i : index
        scf.yield %keep, %stripped : index, i1
      }
      scf.yield %scan#0 : index
    } else {
      scf.yield %c0 : index
    }

    %right_mask = arith.andi %mode, %two : i64
    %strip_right = arith.cmpi ne, %right_mask, %zero : i64
    %finish = scf.if %strip_right -> (index) {
      %scan:2 = scf.while (%i = %dim, %go = %true_bit) : (index, i1) -> (index, i1) {
        %more = arith.cmpi ugt, %i, %begin : index
        %continue = arith.andi %more, %go : i1
        scf.condition(%continue) %i, %go : index, i1
      } do {
      ^bb0(%i: index, %go: i1):
        %prev = arith.subi %i, %c1 : index
        %b_i8 = memref.load %bytes[%prev] : memref<?xi8>
        %b = arith.extui %b_i8 : i8 to i64
        %stripped = scf.if %use_chars -> (i1) {
          %in = func.call @__ly_bytes_byte_in(%b, %chars, %chars_n) : (i64, memref<?xi8>, index) -> i1
          scf.yield %in : i1
        } else {
          %sp = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
          scf.yield %sp : i1
        }
        %keep = arith.select %stripped, %prev, %i : index
        scf.yield %keep, %stripped : index, i1
      }
      scf.yield %scan#0 : index
    } else {
      scf.yield %dim : index
    }

    %result:2 = func.call @__ly_bytes_slice(%bytes, %begin, %finish) : (memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytes_Strip(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.bytes"} {
    %mode = arith.constant 3 : i64
    %false_bit = arith.constant false
    %c0 = arith.constant 0 : index
    %result:2 = func.call @__ly_bytes_strip_core(%bytes, %mode, %false_bit, %bytes, %c0) : (memref<?xi8>, i64, i1, memref<?xi8>, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytes_StripChars(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %chars_header: memref<2xi64> {ly.ownership.object_header}, %chars_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.bytes"} {
    %mode = arith.constant 3 : i64
    %true_bit = arith.constant true
    %c0 = arith.constant 0 : index
    %dim = memref.dim %chars_bytes, %c0 : memref<?xi8>
    %result:2 = func.call @__ly_bytes_strip_core(%bytes, %mode, %true_bit, %chars_bytes, %dim) : (memref<?xi8>, i64, i1, memref<?xi8>, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytes_Replace(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %old_header: memref<2xi64> {ly.ownership.object_header}, %old_bytes: memref<?xi8>, %new_header: memref<2xi64> {ly.ownership.object_header}, %new_bytes: memref<?xi8>, %limit: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "replace", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %true_bit = arith.constant true
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %old_n = func.call @__ly_bytes_len_of(%old_bytes) : (memref<?xi8>) -> i64
    %new_n = func.call @__ly_bytes_len_of(%new_bytes) : (memref<?xi8>) -> i64
    %old_n_index = arith.index_cast %old_n : i64 to index
    %new_n_index = arith.index_cast %new_n : i64 to index
    %old_empty = arith.cmpi eq, %old_n, %zero : i64
    %bound = scf.if %old_empty -> (i64) {
      %plus = arith.addi %len, %one : i64
      scf.yield %plus : i64
    } else {
      scf.yield %len : i64
    }

    %measure:3 = scf.while (%i = %zero, %rem = %limit, %total = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
      %more = arith.cmpi slt, %i, %bound : i64
      scf.condition(%more) %i, %rem, %total : i64, i64, i64
    } do {
    ^bb0(%i: i64, %rem: i64, %total: i64):
      %has_budget = arith.cmpi ne, %rem, %zero : i64
      %tail = arith.addi %i, %old_n : i64
      %in_range = arith.cmpi sle, %tail, %len : i64
      %viable = arith.andi %has_budget, %in_range : i1
      %matched = scf.if %viable -> (i1) {
        %i_index = arith.index_cast %i : i64 to index
        %eq = func.call @__ly_bytes_match_at(%bytes, %i_index, %old_bytes, %old_n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
        scf.yield %eq : i1
      } else {
        %false_bit = arith.constant false
        scf.yield %false_bit : i1
      }
      %old_nonempty = arith.cmpi sgt, %old_n, %zero : i64
      %skip_char = arith.andi %matched, %old_nonempty : i1
      %in_str = arith.cmpi slt, %i, %len : i64
      %not_skip = arith.xori %skip_char, %true_bit : i1
      %emit_char = arith.andi %in_str, %not_skip : i1
      %new_contrib = arith.select %matched, %new_n, %zero : i64
      %char_contrib = arith.select %emit_char, %one, %zero : i64
      %next_total_a = arith.addi %total, %new_contrib : i64
      %next_total = arith.addi %next_total_a, %char_contrib : i64
      %stride = arith.select %skip_char, %old_n, %one : i64
      %next_i = arith.addi %i, %stride : i64
      %dec = arith.select %matched, %one, %zero : i64
      %next_rem = arith.subi %rem, %dec : i64
      scf.yield %next_i, %next_rem, %next_total : i64, i64, i64
    }

    %out_header, %out_bytes = func.call @__ly_bytes_alloc(%measure#2) : (i64) -> (memref<2xi64>, memref<?xi8>)

    %write:3 = scf.while (%i = %zero, %rem = %limit, %pos = %c0) : (i64, i64, index) -> (i64, i64, index) {
      %more = arith.cmpi slt, %i, %bound : i64
      scf.condition(%more) %i, %rem, %pos : i64, i64, index
    } do {
    ^bb0(%i: i64, %rem: i64, %pos: index):
      %has_budget = arith.cmpi ne, %rem, %zero : i64
      %tail = arith.addi %i, %old_n : i64
      %in_range = arith.cmpi sle, %tail, %len : i64
      %viable = arith.andi %has_budget, %in_range : i1
      %matched = scf.if %viable -> (i1) {
        %i_index = arith.index_cast %i : i64 to index
        %eq = func.call @__ly_bytes_match_at(%bytes, %i_index, %old_bytes, %old_n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
        scf.yield %eq : i1
      } else {
        %false_bit = arith.constant false
        scf.yield %false_bit : i1
      }
      %after_new = scf.if %matched -> (index) {
        scf.for %j = %c0 to %new_n_index step %c1 {
          %b = memref.load %new_bytes[%j] : memref<?xi8>
          %dst = arith.addi %pos, %j : index
          memref.store %b, %out_bytes[%dst] : memref<?xi8>
        }
        %advanced = arith.addi %pos, %new_n_index : index
        scf.yield %advanced : index
      } else {
        scf.yield %pos : index
      }
      %old_nonempty = arith.cmpi sgt, %old_n, %zero : i64
      %skip_char = arith.andi %matched, %old_nonempty : i1
      %in_str = arith.cmpi slt, %i, %len : i64
      %not_skip = arith.xori %skip_char, %true_bit : i1
      %emit_char = arith.andi %in_str, %not_skip : i1
      %after_char = scf.if %emit_char -> (index) {
        %i_index = arith.index_cast %i : i64 to index
        %b = memref.load %bytes[%i_index] : memref<?xi8>
        memref.store %b, %out_bytes[%after_new] : memref<?xi8>
        %advanced = arith.addi %after_new, %c1 : index
        scf.yield %advanced : index
      } else {
        scf.yield %after_new : index
      }
      %stride = arith.select %skip_char, %old_n, %one : i64
      %next_i = arith.addi %i, %stride : i64
      %dec = arith.select %matched, %one, %zero : i64
      %next_rem = arith.subi %rem, %dec : i64
      scf.yield %next_i, %next_rem, %after_char : i64, i64, index
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // Box packing for bytes container elements (class id 70), the bytes
  // sibling of __ly_unicode_store_item.
  func.func private @__ly_bytes_store_item(%items: memref<?xi64>, %slot: i64, %eh: memref<2xi64> {ly.ownership.object_header}, %eb: memref<?xi8>) attributes {ly.ownership.transfer_args = [2]} {
    %c0 = arith.constant 0 : index
    %handle_words = arith.constant 16 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %bytes_class = arith.constant 70 : i64
    %base_i64 = arith.muli %slot, %handle_words : i64
    %base = arith.index_cast %base_i64 : i64 to index
    %hdr_ptr_index = memref.extract_aligned_pointer_as_index %eh : memref<2xi64> -> index
    %hdr_ptr = arith.index_cast %hdr_ptr_index : index to i64
    %bytes_ptr_index = memref.extract_aligned_pointer_as_index %eb : memref<?xi8> -> index
    %bytes_ptr = arith.index_cast %bytes_ptr_index : index to i64
    %dim = memref.dim %eb, %c0 : memref<?xi8>
    %byte_len = arith.index_cast %dim : index to i64
    %w1 = arith.constant 1 : index
    %w2 = arith.constant 2 : index
    %w3 = arith.constant 3 : index
    %w4 = arith.constant 4 : index
    %w5 = arith.constant 5 : index
    %w9 = arith.constant 9 : index
    %w10 = arith.constant 10 : index
    %w14 = arith.constant 14 : index
    %one_idx = arith.constant 1 : index
    %sixteen_idx = arith.constant 16 : index
    scf.for %w = %c0 to %sixteen_idx step %one_idx {
      %slot_index = arith.addi %base, %w : index
      memref.store %zero, %items[%slot_index] : memref<?xi64>
    }
    %s0 = arith.addi %base, %c0 : index
    %s1 = arith.addi %base, %w1 : index
    %s2 = arith.addi %base, %w2 : index
    %s3 = arith.addi %base, %w3 : index
    %s4 = arith.addi %base, %w4 : index
    %s5 = arith.addi %base, %w5 : index
    %s9 = arith.addi %base, %w9 : index
    %s10 = arith.addi %base, %w10 : index
    %s14 = arith.addi %base, %w14 : index
    memref.store %one, %items[%s0] : memref<?xi64>
    memref.store %bytes_class, %items[%s1] : memref<?xi64>
    memref.store %hdr_ptr, %items[%s2] : memref<?xi64>
    memref.store %two, %items[%s3] : memref<?xi64>
    memref.store %hdr_ptr, %items[%s4] : memref<?xi64>
    memref.store %bytes_ptr, %items[%s5] : memref<?xi64>
    memref.store %two, %items[%s9] : memref<?xi64>
    memref.store %byte_len, %items[%s10] : memref<?xi64>
    memref.store %one, %items[%s14] : memref<?xi64>
    func.return
  }

  // bytes.split(sep[, maxsplit]).
  func.func @LyBytes_Split(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.bytes"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %false_bit = arith.constant false
    %true_a = arith.constant true
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %sep_n = func.call @__ly_bytes_len_of(%sep_bytes) : (memref<?xi8>) -> i64
    %sep_empty = arith.cmpi eq, %sep_n, %zero : i64
    scf.if %sep_empty {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 15 : i64
      %static = memref.get_global @__ly_unicode_msg_empty_separator : memref<15xi8>
      %message = memref.cast %static : memref<15xi8> to memref<?xi8>
      func.call @__ly_unicode_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }

    %count:3 = scf.while (%pos = %zero, %used = %zero, %go = %true_a) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %pos, %used, %go : i64, i64, i1
    } do {
    ^bb0(%pos: i64, %used: i64, %go: i1):
      %budget_left = arith.cmpi ne, %used, %maxsplit : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_bytes_find_core(%bytes, %sep_bytes, %pos, %len, %sep_n, %false_bit) : (memref<?xi8>, memref<?xi8>, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      %next_pos_hit = arith.addi %hit, %sep_n : i64
      %next_pos = arith.select %matched, %next_pos_hit, %pos : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_used = arith.addi %used, %bump : i64
      scf.yield %next_pos, %next_used, %matched : i64, i64, i1
    }

    %segments = arith.addi %count#1, %one : i64
    %list:3 = func.call @LyList_FromLength(%segments) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)

    %true_b = arith.constant true
    %emit:3 = scf.while (%pos = %zero, %slot = %zero, %go = %true_b) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %pos, %slot, %go : i64, i64, i1
    } do {
    ^bb0(%pos: i64, %slot: i64, %go: i1):
      %remaining = arith.subi %segments, %one : i64
      %budget_left = arith.cmpi slt, %slot, %remaining : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_bytes_find_core(%bytes, %sep_bytes, %pos, %len, %sep_n, %false_bit) : (memref<?xi8>, memref<?xi8>, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      scf.if %matched {
        %pos_index = arith.index_cast %pos : i64 to index
        %hit_index = arith.index_cast %hit : i64 to index
        %piece:2 = func.call @__ly_bytes_slice(%bytes, %pos_index, %hit_index) : (memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
        func.call @__ly_bytes_store_item(%list#2, %slot, %piece#0, %piece#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      }
      %next_pos_hit = arith.addi %hit, %sep_n : i64
      %next_pos = arith.select %matched, %next_pos_hit, %pos : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_slot = arith.addi %slot, %bump : i64
      scf.yield %next_pos, %next_slot, %matched : i64, i64, i1
    }
    %tail_start = arith.index_cast %emit#0 : i64 to index
    %len_index = arith.index_cast %len : i64 to index
    %tail:2 = func.call @__ly_bytes_slice(%bytes, %tail_start, %len_index) : (memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_bytes_store_item(%list#2, %emit#1, %tail#0, %tail#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
    func.return %list#0, %list#1, %list#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // bytes.split() -- ASCII whitespace runs.
  func.func @LyBytes_SplitWS(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %false_a = arith.constant false
    %dim = memref.dim %bytes, %c0 : memref<?xi8>

    %count:2 = scf.for %i = %c0 to %dim step %c1 iter_args(%segs = %zero, %in_run = %false_a) -> (i64, i1) {
      %b_i8 = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %b_i8 : i8 to i64
      %is_space = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
      %true_w = arith.constant true
      %non_space = arith.xori %is_space, %true_w : i1
      %not_in_run = arith.xori %in_run, %true_w : i1
      %starts = arith.andi %non_space, %not_in_run : i1
      %bump = arith.select %starts, %one, %zero : i64
      %next_segs = arith.addi %segs, %bump : i64
      scf.yield %next_segs, %non_space : i64, i1
    }

    %list:3 = func.call @LyList_FromLength(%count#0) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)

    %false_b = arith.constant false
    scf.for %i = %c0 to %dim step %c1 iter_args(%slot = %zero, %run_start = %dim, %in_run = %false_b) -> (i64, index, i1) {
      %b_i8 = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %b_i8 : i8 to i64
      %is_space = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
      %true_w = arith.constant true
      %non_space = arith.xori %is_space, %true_w : i1
      %not_in_run = arith.xori %in_run, %true_w : i1
      %starts = arith.andi %non_space, %not_in_run : i1
      %ends = arith.andi %is_space, %in_run : i1
      %new_start = arith.select %starts, %i, %run_start : index
      scf.if %ends {
        %piece:2 = func.call @__ly_bytes_slice(%bytes, %run_start, %i) : (memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
        func.call @__ly_bytes_store_item(%list#2, %slot, %piece#0, %piece#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      }
      %bump = arith.select %ends, %one, %zero : i64
      %next_slot = arith.addi %slot, %bump : i64
      %last = arith.subi %dim, %c1 : index
      %is_last = arith.cmpi eq, %i, %last : index
      %closes = arith.andi %is_last, %non_space : i1
      scf.if %closes {
        %piece:2 = func.call @__ly_bytes_slice(%bytes, %new_start, %dim) : (memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
        func.call @__ly_bytes_store_item(%list#2, %next_slot, %piece#0, %piece#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      }
      scf.yield %next_slot, %new_start, %non_space : i64, index, i1
    }
    func.return %list#0, %list#1, %list#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // bytes.join over a runtime list/tuple of bytes.
  func.func @LyBytes_Join(%sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>, %seq_header: memref<2xi64> {ly.ownership.object_header}, %seq_meta: memref<2xi64>, %seq_items: memref<?xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "join", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sep_n = func.call @__ly_bytes_len_of(%sep_bytes) : (memref<?xi8>) -> i64
    %n = memref.load %seq_meta[%c0] : memref<2xi64>
    %n_index = arith.index_cast %n : i64 to index

    %measure = scf.for %k = %c0 to %n_index step %c1 iter_args(%total = %zero) -> (i64) {
      %hdr, %ptr, %blen = func.call @__ly_unicode_item_words(%seq_items, %k) : (memref<?xi64>, index) -> (i64, i64, i64)
      %next = arith.addi %total, %blen : i64
      scf.yield %next : i64
    }
    %has_seps = arith.cmpi sgt, %n, %one : i64
    %sep_uses_i64 = arith.subi %n, %one : i64
    %sep_uses = arith.select %has_seps, %sep_uses_i64, %zero : i64
    %sep_total = arith.muli %sep_uses, %sep_n : i64
    %total = arith.addi %measure, %sep_total : i64

    %out_header, %out_bytes = func.call @__ly_bytes_alloc(%total) : (i64) -> (memref<2xi64>, memref<?xi8>)

    scf.for %k = %c0 to %n_index step %c1 iter_args(%pos = %c0) -> (index) {
      %is_first = arith.cmpi eq, %k, %c0 : index
      %true_k = arith.constant true
      %needs_sep = arith.xori %is_first, %true_k : i1
      %after_sep = scf.if %needs_sep -> (index) {
        %sep_n_index = arith.index_cast %sep_n : i64 to index
        scf.for %j = %c0 to %sep_n_index step %c1 {
          %b = memref.load %sep_bytes[%j] : memref<?xi8>
          %dst = arith.addi %pos, %j : index
          memref.store %b, %out_bytes[%dst] : memref<?xi8>
        }
        %advanced = arith.addi %pos, %sep_n_index : index
        scf.yield %advanced : index
      } else {
        scf.yield %pos : index
      }
      %hdr, %ptr, %blen = func.call @__ly_unicode_item_words(%seq_items, %k) : (memref<?xi64>, index) -> (i64, i64, i64)
      %blen_index = arith.index_cast %blen : i64 to index
      scf.for %j = %c0 to %blen_index step %c1 {
        %j_i64 = arith.index_cast %j : index to i64
        %addr = arith.addi %ptr, %j_i64 : i64
        %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
        %b = llvm.load %llptr : !llvm.ptr -> i8
        %dst = arith.addi %after_sep, %j : index
        memref.store %b, %out_bytes[%dst] : memref<?xi8>
      }
      %next_pos = arith.addi %after_sep, %blen_index : index
      scf.yield %next_pos : index
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // bytes.hex(): lowercase pairs, no separator.
  func.func @LyBytes_Hex(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "hex", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %out_count = arith.muli %len, %two : i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%out_count, %one) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    scf.for %i = %c0 to %dim step %c1 {
      %b_i8 = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %b_i8 : i8 to i64
      %pos = arith.muli %i, %c2 : index
      func.call @__ly_unicode_put_hex(%out_bytes, %one, %pos, %b, %c2) : (memref<?xi8>, i64, index, i64, index) -> ()
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // bytes.fromhex(str): ASCII whitespace between pairs is ignored; anything
  // else (or an odd trailing digit) raises ValueError at its position.
  func.func @LyBytes_FromHex(%str_header: memref<2xi64> {ly.ownership.object_header}, %str_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "fromhex"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %sixteen = arith.constant 16 : i64
    %minus_one = arith.constant -1 : i64
    %width = func.call @__ly_unicode_width(%str_header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%str_header, %str_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index

    // Pass 1: count digit pairs; report the first bad position.
    %scan:3 = scf.for %i = %c0 to %count_index step %c1 iter_args(%digits = %zero, %bad = %minus_one, %pending = %minus_one) -> (i64, i64, i64) {
      %cp = func.call @__ly_unicode_get(%str_bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %value = func.call @__ly_bytes_hex_digit(%cp) : (i64) -> i64
      %is_digit = arith.cmpi sge, %value, %zero : i64
      %is_ws = func.call @__ly_bytes_byte_is_ascii_space(%cp) : (i64) -> i1
      %i_i64 = arith.index_cast %i : index to i64
      %no_bad_yet = arith.cmpi eq, %bad, %minus_one : i64
      %has_pending = arith.cmpi sge, %pending, %zero : i64
      // whitespace inside a pair (pending digit) is an error at this index;
      // a non-digit non-ws is an error at this index.
      %true_h = arith.constant true
      %not_digit = arith.xori %is_digit, %true_h : i1
      %not_ws = arith.xori %is_ws, %true_h : i1
      %bad_char = arith.andi %not_digit, %not_ws : i1
      %ws_inside = arith.andi %is_ws, %has_pending : i1
      %new_bad_here = arith.ori %bad_char, %ws_inside : i1
      %record = arith.andi %new_bad_here, %no_bad_yet : i1
      %next_bad = arith.select %record, %i_i64, %bad : i64
      %bump = arith.select %is_digit, %one, %zero : i64
      %next_digits = arith.addi %digits, %bump : i64
      %next_pending_digit = arith.select %has_pending, %minus_one, %i_i64 : i64
      %next_pending = arith.select %is_digit, %next_pending_digit, %pending : i64
      scf.yield %next_digits, %next_bad, %next_pending : i64, i64, i64
    }
    // A trailing unpaired digit is an error at the end of the string.
    %odd = arith.andi %scan#0, %one : i64
    %is_odd = arith.cmpi ne, %odd, %zero : i64
    %no_bad = arith.cmpi eq, %scan#1, %minus_one : i64
    %odd_bad = arith.andi %is_odd, %no_bad : i1
    %bad_final = arith.select %odd_bad, %count, %scan#1 : i64
    %has_bad = arith.cmpi sge, %bad_final, %zero : i64
    scf.if %has_bad {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 58 : i64
      %static = memref.get_global @__ly_bytes_msg_fromhex : memref<58xi8>
      %message = memref.cast %static : memref<58xi8> to memref<?xi8>
      %prefix_h, %prefix_b = func.call @LyUnicode_FromBytes(%message, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %pos_h, %pos_b = func.call @LyUnicode_FromI64(%bad_final) : (i64) -> (memref<2xi64>, memref<?xi8>)
      %msg_h, %msg_b = func.call @LyUnicode_Concat(%prefix_h, %prefix_b, %pos_h, %pos_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%prefix_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%pos_h) : (memref<2xi64>) -> ()
      %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
      %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %msg_h, %msg_b) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
      func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    }
    %pairs = arith.divsi %scan#0, %two : i64
    %out_header, %out_bytes = func.call @__ly_bytes_alloc(%pairs) : (i64) -> (memref<2xi64>, memref<?xi8>)
    // Pass 2: emit the pairs.
    scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %minus_one, %slot = %c0) -> (i64, index) {
      %cp = func.call @__ly_unicode_get(%str_bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %value = func.call @__ly_bytes_hex_digit(%cp) : (i64) -> i64
      %is_digit = arith.cmpi sge, %value, %zero : i64
      %has_acc = arith.cmpi sge, %acc, %zero : i64
      %completes = arith.andi %is_digit, %has_acc : i1
      %next_slot = scf.if %completes -> (index) {
        %hi = arith.muli %acc, %sixteen : i64
        %byte = arith.addi %hi, %value : i64
        %byte_i8 = arith.trunci %byte : i64 to i8
        memref.store %byte_i8, %out_bytes[%slot] : memref<?xi8>
        %advanced = arith.addi %slot, %c1 : index
        scf.yield %advanced : index
      } else {
        scf.yield %slot : index
      }
      %start_acc = arith.select %has_acc, %minus_one, %value : i64
      %next_acc = arith.select %is_digit, %start_acc, %acc : i64
      scf.yield %next_acc, %next_slot : i64, index
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_bytes_hex_digit(%cp: i64) -> i64 {
    %zero_ch = arith.constant 48 : i64
    %nine_ch = arith.constant 57 : i64
    %a_ch = arith.constant 97 : i64
    %f_ch = arith.constant 102 : i64
    %A_ch = arith.constant 65 : i64
    %F_ch = arith.constant 70 : i64
    %ten = arith.constant 10 : i64
    %minus_one = arith.constant -1 : i64
    %ge0 = arith.cmpi sge, %cp, %zero_ch : i64
    %le9 = arith.cmpi sle, %cp, %nine_ch : i64
    %dec = arith.andi %ge0, %le9 : i1
    %dec_val = arith.subi %cp, %zero_ch : i64
    %gea = arith.cmpi sge, %cp, %a_ch : i64
    %lef = arith.cmpi sle, %cp, %f_ch : i64
    %lower = arith.andi %gea, %lef : i1
    %lower_delta = arith.subi %cp, %a_ch : i64
    %lower_val = arith.addi %lower_delta, %ten : i64
    %geA = arith.cmpi sge, %cp, %A_ch : i64
    %leF = arith.cmpi sle, %cp, %F_ch : i64
    %upper = arith.andi %geA, %leF : i1
    %upper_delta = arith.subi %cp, %A_ch : i64
    %upper_val = arith.addi %upper_delta, %ten : i64
    %v1 = arith.select %upper, %upper_val, %minus_one : i64
    %v2 = arith.select %lower, %lower_val, %v1 : i64
    %value = arith.select %dec, %dec_val, %v2 : i64
    func.return %value : i64
  }

  // bytes * int.
  func.func @LyBytes_Mul(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %repeat: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__mul__", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %n = arith.maxsi %repeat, %zero : i64
    %total = arith.muli %len, %n : i64
    %out_header, %out_bytes = func.call @__ly_bytes_alloc(%total) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %n_index = arith.index_cast %n : i64 to index
    scf.for %k = %c0 to %n_index step %c1 {
      %base = arith.muli %k, %dim : index
      scf.for %i = %c0 to %dim step %c1 {
        %b = memref.load %bytes[%i] : memref<?xi8>
        %dst = arith.addi %base, %i : index
        memref.store %b, %out_bytes[%dst] : memref<?xi8>
      }
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytes_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: long =====
  func.func private @LyLong_Shape() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.runtime.contract = "builtins.int", ly.runtime.shape}

  // Views of int operands used by the arithmetic entry points.
  func.func private @__ly_long_operand_view(%meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<?xi32>) {
    func.return %meta, %digits : memref<2xi64>, memref<?xi32>
  }

  memref.global "private" constant @__ly_long_msg_division_by_zero : memref<16xi8> = dense<[100, 105, 118, 105, 115, 105, 111, 110, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_integer_division_or_modulo_by_zero : memref<34xi8> = dense<[105, 110, 116, 101, 103, 101, 114, 32, 100, 105, 118, 105, 115, 105, 111, 110, 32, 111, 114, 32, 109, 111, 100, 117, 108, 111, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_integer_modulo_by_zero : memref<22xi8> = dense<[105, 110, 116, 101, 103, 101, 114, 32, 109, 111, 100, 117, 108, 111, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_negative_shift_count : memref<20xi8> = dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 115, 104, 105, 102, 116, 32, 99, 111, 117, 110, 116]>
  // "int too large to convert to a native 64-bit integer"
  memref.global "private" constant @__ly_long_msg_int_too_large : memref<51xi8> = dense<[105, 110, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 116, 111, 32, 99, 111, 110, 118, 101, 114, 116, 32, 116, 111, 32, 97, 32, 110, 97, 116, 105, 118, 101, 32, 54, 52, 45, 98, 105, 116, 32, 105, 110, 116, 101, 103, 101, 114]>
  // "invalid literal for int() with base 10" (CPython appends the repr of the
  // input; message concatenation needs the str-track's formatting work)
  memref.global "private" constant @__ly_long_msg_invalid_int_literal : memref<38xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 108, 105, 116, 101, 114, 97, 108, 32, 102, 111, 114, 32, 105, 110, 116, 40, 41, 32, 119, 105, 116, 104, 32, 98, 97, 115, 101, 32, 49, 48]>
  // "int ** negative int is rejected: the static result type is int; use float(base) ** exponent"
  memref.global "private" constant @__ly_long_msg_pow_negative_exponent : memref<91xi8> = dense<[105, 110, 116, 32, 42, 42, 32, 110, 101, 103, 97, 116, 105, 118, 101, 32, 105, 110, 116, 32, 105, 115, 32, 114, 101, 106, 101, 99, 116, 101, 100, 58, 32, 116, 104, 101, 32, 115, 116, 97, 116, 105, 99, 32, 114, 101, 115, 117, 108, 116, 32, 116, 121, 112, 101, 32, 105, 115, 32, 105, 110, 116, 59, 32, 117, 115, 101, 32, 102, 108, 111, 97, 116, 40, 98, 97, 115, 101, 41, 32, 42, 42, 32, 101, 120, 112, 111, 110, 101, 110, 116]>
  // "shift count too large"
  memref.global "private" constant @__ly_long_msg_shift_count_too_large : memref<21xi8> = dense<[115, 104, 105, 102, 116, 32, 99, 111, 117, 110, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101]>
  // "int too large to convert to float"
  memref.global "private" constant @__ly_long_msg_int_too_large_float : memref<33xi8> = dense<[105, 110, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 116, 111, 32, 99, 111, 110, 118, 101, 114, 116, 32, 116, 111, 32, 102, 108, 111, 97, 116]>
  // "integer division result too large for a float"
  memref.global "private" constant @__ly_long_msg_div_result_too_large : memref<45xi8> = dense<[105, 110, 116, 101, 103, 101, 114, 32, 100, 105, 118, 105, 115, 105, 111, 110, 32, 114, 101, 115, 117, 108, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 102, 111, 114, 32, 97, 32, 102, 108, 111, 97, 116]>
  // "cannot convert float NaN to integer"
  memref.global "private" constant @__ly_long_msg_float_nan : memref<35xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 99, 111, 110, 118, 101, 114, 116, 32, 102, 108, 111, 97, 116, 32, 78, 97, 78, 32, 116, 111, 32, 105, 110, 116, 101, 103, 101, 114]>
  // "cannot convert float infinity to integer"
  memref.global "private" constant @__ly_long_msg_float_infinity : memref<40xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 99, 111, 110, 118, 101, 114, 116, 32, 102, 108, 111, 97, 116, 32, 105, 110, 102, 105, 110, 105, 116, 121, 32, 116, 111, 32, 105, 110, 116, 101, 103, 101, 114]>
  // "zero to a negative power"
  memref.global "private" constant @__ly_long_msg_zero_negative_power : memref<24xi8> = dense<[122, 101, 114, 111, 32, 116, 111, 32, 97, 32, 110, 101, 103, 97, 116, 105, 118, 101, 32, 112, 111, 119, 101, 114]>
  // "negative number cannot be raised to a fractional power" (CPython returns
  // a complex here; complex is not implemented, so reject loudly instead)
  memref.global "private" constant @__ly_long_msg_fractional_power_negative : memref<54xi8> = dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 110, 117, 109, 98, 101, 114, 32, 99, 97, 110, 110, 111, 116, 32, 98, 101, 32, 114, 97, 105, 115, 101, 100, 32, 116, 111, 32, 97, 32, 102, 114, 97, 99, 116, 105, 111, 110, 97, 108, 32, 112, 111, 119, 101, 114]>
  memref.global "private" constant @__ly_long_zero_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_zero_meta : memref<2xi64> = dense<[0, 0]>
  memref.global "private" constant @__ly_long_zero_digits : memref<1xi32> = dense<[0]>
  memref.global "private" constant @__ly_long_one_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_one_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" constant @__ly_long_one_digits : memref<1xi32> = dense<[1]>
  memref.global "private" constant @__ly_long_two_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_two_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" constant @__ly_long_two_digits : memref<1xi32> = dense<[2]>

  func.func private @__ly_long_raise_message(%class_id: i64, %message: memref<?xi8>, %length: i64) {
    %start = arith.constant 0 : index
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @__ly_long_raise_true_div_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 16 : i64
    %message_static = memref.get_global @__ly_long_msg_division_by_zero : memref<16xi8>
    %message = memref.cast %message_static : memref<16xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython 3.14 unified the /, //, and % zero-divisor message to
  // "division by zero" (gh-87999), so the historic per-operator texts are
  // not used.
  func.func private @__ly_long_raise_floor_div_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 16 : i64
    %message_static = memref.get_global @__ly_long_msg_division_by_zero : memref<16xi8>
    %message = memref.cast %message_static : memref<16xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_mod_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 16 : i64
    %message_static = memref.get_global @__ly_long_msg_division_by_zero : memref<16xi8>
    %message = memref.cast %message_static : memref<16xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_negative_shift() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 20 : i64
    %message_static = memref.get_global @__ly_long_msg_negative_shift_count : memref<20xi8>
    %message = memref.cast %message_static : memref<20xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_alloc_raw(%sign: i64, %capacity: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %layout_int = arith.constant 1 : i64
    %needs_one = arith.cmpi sle, %capacity, %zero : i64
    %alloc_count_i64 = arith.select %needs_one, %one, %capacity : i1, i64
    %alloc_count = arith.index_cast %alloc_count_i64 : i64 to index
    // One entity, one allocation: [0,16) header, [16,32) meta, [32,..) digits.
    // The header view carries the allocation; meta/digits are interior views
    // and must never be freed separately.
    %four_bytes = arith.constant 4 : i64
    %block_prefix = arith.constant 32 : i64
    %digit_bytes = arith.muli %alloc_count_i64, %four_bytes : i64
    %block_bytes_i64 = arith.addi %digit_bytes, %block_prefix : i64
    %block_bytes = arith.index_cast %block_bytes_i64 : i64 to index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %meta_offset = arith.constant 16 : index
    %digits_offset = arith.constant 32 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %meta = memref.view %block[%meta_offset][] : memref<?xi8> to memref<2xi64>
    %digits = memref.view %block[%digits_offset][%alloc_count] : memref<?xi8> to memref<?xi32>
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_int, %header[%layout_slot] : memref<2xi64>
    memref.store %sign, %meta[%sign_slot] : memref<2xi64>
    memref.store %capacity, %meta[%digit_count_slot] : memref<2xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_i32 = arith.constant 0 : i32
    scf.for %iv = %c0 to %alloc_count step %c1 {
      memref.store %zero_i32, %digits[%iv] : memref<?xi32>
    }
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_normalize(%meta: memref<2xi64>, %digits: memref<?xi32>, %capacity: i64) {
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %capacity_index = arith.index_cast %capacity : i64 to index
    %last = scf.for %iv = %c0 to %capacity_index step %c1 iter_args(%last_iter = %zero) -> (i64) {
      %digit_i32 = memref.load %digits[%iv] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %nonzero = arith.cmpi ne, %digit, %zero : i64
      %next_index = arith.addi %iv, %c1 : index
      %next_count = arith.index_cast %next_index : index to i64
      %next = arith.select %nonzero, %next_count, %last_iter : i1, i64
      scf.yield %next : i64
    }
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    %raw_sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %is_zero = arith.cmpi eq, %last, %zero : i64
    %sign = arith.select %is_zero, %zero, %raw_sign : i1, i64
    memref.store %sign, %meta[%sign_slot] : memref<2xi64>
    memref.store %last, %meta[%digit_count_slot] : memref<2xi64>
    func.return
  }

  func.func private @__ly_long_copy_with_sign(%sign: i64, %meta_in: memref<2xi64>, %digits_in: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta_in[%count_slot] : memref<2xi64>
    %header, %meta, %digits = func.call @__ly_long_alloc_raw(%sign, %count) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = arith.index_cast %count : i64 to index
    scf.for %iv = %c0 to %count_index step %c1 {
      %digit = memref.load %digits_in[%iv] : memref<?xi32>
      memref.store %digit, %digits[%iv] : memref<?xi32>
    }
    func.call @__ly_long_normalize(%meta, %digits, %count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_copy(%meta_in: memref<2xi64>, %digits_in: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %sign_slot = arith.constant 0 : index
    %sign = memref.load %meta_in[%sign_slot] : memref<2xi64>
    %header, %meta, %digits = func.call @__ly_long_copy_with_sign(%sign, %meta_in, %digits_in) : (i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_abs_compare(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %count_slot = arith.constant 1 : index
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %lhs_longer = arith.cmpi sgt, %lhs_count, %rhs_count : i64
    %rhs_longer = arith.cmpi slt, %lhs_count, %rhs_count : i64
    %size_cmp = arith.select %lhs_longer, %one, %zero : i1, i64
    %size_cmp2 = arith.select %rhs_longer, %neg_one, %size_cmp : i1, i64
    %same_size = arith.cmpi eq, %size_cmp2, %zero : i64
    %result = scf.if %same_size -> (i64) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %count_index = arith.index_cast %lhs_count : i64 to index
      %cmp = scf.for %iv = %c0 to %count_index step %c1 iter_args(%state = %zero) -> (i64) {
        %still_equal = arith.cmpi eq, %state, %zero : i64
        %iv_next = arith.addi %iv, %c1 : index
        %rev = arith.subi %count_index, %iv_next : index
        %lhs_digit_i32 = memref.load %lhs_digits[%rev] : memref<?xi32>
        %rhs_digit_i32 = memref.load %rhs_digits[%rev] : memref<?xi32>
        %lhs_digit = arith.extui %lhs_digit_i32 : i32 to i64
        %rhs_digit = arith.extui %rhs_digit_i32 : i32 to i64
        %gt = arith.cmpi ugt, %lhs_digit, %rhs_digit : i64
        %lt = arith.cmpi ult, %lhs_digit, %rhs_digit : i64
        %digit_cmp = arith.select %gt, %one, %zero : i1, i64
        %digit_cmp2 = arith.select %lt, %neg_one, %digit_cmp : i1, i64
        %next = arith.select %still_equal, %digit_cmp2, %state : i1, i64
        scf.yield %next : i64
      }
      scf.yield %cmp : i64
    } else {
      scf.yield %size_cmp2 : i64
    }
    func.return %result : i64
  }

  func.func private @__ly_long_add_abs(%sign: i64, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %mask = arith.constant 1073741823 : i64
    %shift30 = arith.constant 30 : i64
    %count_slot = arith.constant 1 : index
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %lhs_ge = arith.cmpi sge, %lhs_count, %rhs_count : i64
    %max_count = arith.select %lhs_ge, %lhs_count, %rhs_count : i1, i64
    %capacity = arith.addi %max_count, %one : i64
    %header, %meta, %digits = func.call @__ly_long_alloc_raw(%sign, %capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %max_index = arith.index_cast %max_count : i64 to index
    %carry = scf.for %iv = %c0 to %max_index step %c1 iter_args(%carry_iter = %zero) -> (i64) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %has_lhs = arith.cmpi slt, %iv_i64, %lhs_count : i64
      %lhs_digit = scf.if %has_lhs -> (i64) {
        %digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      %has_rhs = arith.cmpi slt, %iv_i64, %rhs_count : i64
      %rhs_digit = scf.if %has_rhs -> (i64) {
        %digit_i32 = memref.load %rhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      %partial = arith.addi %lhs_digit, %rhs_digit : i64
      %sum = arith.addi %partial, %carry_iter : i64
      %out_i64 = arith.andi %sum, %mask : i64
      %out = arith.trunci %out_i64 : i64 to i32
      memref.store %out, %digits[%iv] : memref<?xi32>
      %next_carry = arith.shrui %sum, %shift30 : i64
      scf.yield %next_carry : i64
    }
    %carry_slot = arith.index_cast %max_count : i64 to index
    %carry_i32 = arith.trunci %carry : i64 to i32
    memref.store %carry_i32, %digits[%carry_slot] : memref<?xi32>
    func.call @__ly_long_normalize(%meta, %digits, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_sub_abs(%sign: i64, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %base = arith.constant 1073741824 : i64
    %count_slot = arith.constant 1 : index
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %header, %meta, %digits = func.call @__ly_long_alloc_raw(%sign, %lhs_count) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_index = arith.index_cast %lhs_count : i64 to index
    %borrow = scf.for %iv = %c0 to %lhs_index step %c1 iter_args(%borrow_iter = %zero) -> (i64) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %lhs_digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
      %lhs_digit = arith.extui %lhs_digit_i32 : i32 to i64
      %has_rhs = arith.cmpi slt, %iv_i64, %rhs_count : i64
      %rhs_digit = scf.if %has_rhs -> (i64) {
        %digit_i32 = memref.load %rhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      %rhs_with_borrow = arith.addi %rhs_digit, %borrow_iter : i64
      %needs_borrow = arith.cmpi ult, %lhs_digit, %rhs_with_borrow : i64
      %raw_diff = arith.subi %lhs_digit, %rhs_with_borrow : i64
      %borrowed_diff = arith.addi %raw_diff, %base : i64
      %diff = arith.select %needs_borrow, %borrowed_diff, %raw_diff : i1, i64
      %out = arith.trunci %diff : i64 to i32
      memref.store %out, %digits[%iv] : memref<?xi32>
      %next_borrow = arith.select %needs_borrow, %one, %zero : i1, i64
      scf.yield %next_borrow : i64
    }
    func.call @__ly_long_normalize(%meta, %digits, %lhs_count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_add_signed_general(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_effective_sign: i64, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    %result:3 = scf.if %lhs_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %h, %m, %d = func.call @__ly_long_copy_with_sign(%rhs_effective_sign, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %rhs_zero = arith.cmpi eq, %rhs_effective_sign, %zero : i64
      %inner:3 = scf.if %rhs_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
        %h, %m, %d = func.call @__ly_long_copy(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
        scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
      } else {
        %same_sign = arith.cmpi eq, %lhs_sign, %rhs_effective_sign : i64
        %combined:3 = scf.if %same_sign -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
          %h, %m, %d = func.call @__ly_long_add_abs(%lhs_sign, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
          scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
        } else {
          %cmp = func.call @__ly_long_abs_compare(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
          %lhs_bigger = arith.cmpi sgt, %cmp, %zero : i64
          %abs_result:3 = scf.if %lhs_bigger -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
            %h, %m, %d = func.call @__ly_long_sub_abs(%lhs_sign, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
            scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
          } else {
            %rhs_bigger = arith.cmpi slt, %cmp, %zero : i64
            %rhs_result:3 = scf.if %rhs_bigger -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
              %h, %m, %d = func.call @__ly_long_sub_abs(%rhs_effective_sign, %rhs_meta, %rhs_digits, %lhs_meta, %lhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
              scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
            } else {
              %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
              scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
            }
            scf.yield %rhs_result#0, %rhs_result#1, %rhs_result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
          }
          scf.yield %abs_result#0, %abs_result#1, %abs_result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
        }
        scf.yield %combined#0, %combined#1, %combined#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
      }
      scf.yield %inner#0, %inner#1, %inner#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %is_cached_zero = arith.cmpi eq, %value, %zero : i64
    cf.cond_br %is_cached_zero, ^cached_zero, ^check_cached_one

  ^cached_zero:
    %zero_header = memref.get_global @__ly_long_zero_header : memref<2xi64>
    %zero_meta = memref.get_global @__ly_long_zero_meta : memref<2xi64>
    %zero_digits_static = memref.get_global @__ly_long_zero_digits : memref<1xi32>
    %zero_digits = memref.cast %zero_digits_static : memref<1xi32> to memref<?xi32>
    func.return %zero_header, %zero_meta, %zero_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_cached_one:
    %is_cached_one = arith.cmpi eq, %value, %one : i64
    cf.cond_br %is_cached_one, ^cached_one, ^check_cached_two

  ^cached_one:
    %one_header = memref.get_global @__ly_long_one_header : memref<2xi64>
    %one_meta = memref.get_global @__ly_long_one_meta : memref<2xi64>
    %one_digits_static = memref.get_global @__ly_long_one_digits : memref<1xi32>
    %one_digits = memref.cast %one_digits_static : memref<1xi32> to memref<?xi32>
    func.return %one_header, %one_meta, %one_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_cached_two:
    %is_cached_two = arith.cmpi eq, %value, %two : i64
    cf.cond_br %is_cached_two, ^cached_two, ^heap

  ^cached_two:
    %two_header = memref.get_global @__ly_long_two_header : memref<2xi64>
    %two_meta = memref.get_global @__ly_long_two_meta : memref<2xi64>
    %two_digits_static = memref.get_global @__ly_long_two_digits : memref<1xi32>
    %two_digits = memref.cast %two_digits_static : memref<1xi32> to memref<?xi32>
    func.return %two_header, %two_meta, %two_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^heap:
    %is_zero = arith.cmpi eq, %value, %zero : i64
    %is_negative = arith.cmpi slt, %value, %zero : i64
    %negative_one = arith.constant -1 : i64
    %positive_one = arith.constant 1 : i64
    %signed_sign = arith.select %is_negative, %negative_one, %positive_one : i1, i64
    %sign = arith.select %is_zero, %zero, %signed_sign : i1, i64
    %abs_value = scf.if %is_negative -> (i64) {
      %negated = arith.subi %zero, %value : i64
      scf.yield %negated : i64
    } else {
      scf.yield %value : i64
    }
    %mask = arith.constant 1073741823 : i64
    %shift30 = arith.constant 30 : i64
    %shift60 = arith.constant 60 : i64
    %digit0_i64 = arith.andi %abs_value, %mask : i64
    %digit1_shifted = arith.shrui %abs_value, %shift30 : i64
    %digit1_i64 = arith.andi %digit1_shifted, %mask : i64
    %digit2_i64 = arith.shrui %abs_value, %shift60 : i64
    %digit1_nonzero = arith.cmpi ne, %digit1_i64, %zero : i64
    %digit2_nonzero = arith.cmpi ne, %digit2_i64, %zero : i64
    %one_or_two = arith.select %digit1_nonzero, %two, %one : i1, i64
    %nonzero_digits = arith.select %digit2_nonzero, %three, %one_or_two : i1, i64
    %ndigits = arith.select %is_zero, %zero, %nonzero_digits : i1, i64
    %alloc_digits_i64 = arith.select %is_zero, %one, %ndigits : i1, i64
    %alloc_digits = arith.index_cast %alloc_digits_i64 : i64 to index
    // One entity, one allocation (see __ly_long_alloc_raw).
    %four_bytes = arith.constant 4 : i64
    %block_prefix = arith.constant 32 : i64
    %digit_bytes = arith.muli %alloc_digits_i64, %four_bytes : i64
    %block_bytes_i64 = arith.addi %digit_bytes, %block_prefix : i64
    %block_bytes = arith.index_cast %block_bytes_i64 : i64 to index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %meta_offset = arith.constant 16 : index
    %digits_offset = arith.constant 32 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %meta = memref.view %block[%meta_offset][] : memref<?xi8> to memref<2xi64>
    %digits = memref.view %block[%digits_offset][%alloc_digits] : memref<?xi8> to memref<?xi32>
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %one, %header[%layout_slot] : memref<2xi64>
    memref.store %sign, %meta[%sign_slot] : memref<2xi64>
    memref.store %ndigits, %meta[%digit_count_slot] : memref<2xi64>
    %digit0_slot = arith.constant 0 : index
    %digit0 = arith.trunci %digit0_i64 : i64 to i32
    memref.store %digit0, %digits[%digit0_slot] : memref<?xi32>
    %has_digit1 = arith.cmpi sge, %ndigits, %two : i64
    scf.if %has_digit1 {
      %digit1_slot = arith.constant 1 : index
      %digit1 = arith.trunci %digit1_i64 : i64 to i32
      memref.store %digit1, %digits[%digit1_slot] : memref<?xi32>
    }
    %has_digit2 = arith.cmpi sge, %ndigits, %three : i64
    scf.if %has_digit2 {
      %digit2_slot = arith.constant 2 : index
      %digit2 = arith.trunci %digit2_i64 : i64 to i32
      memref.store %digit2, %digits[%digit2_slot] : memref<?xi32>
    }
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Constructor for compile-time digit spans (big int literals): the lowering
  // splits the literal into 30-bit limbs at compile time and this copies them
  // into a fresh heap object.
  func.func @LyLong_FromDigits(%sign: i64, %digits_in: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "from_digits"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = memref.dim %digits_in, %c0 : memref<?xi32>
    %count = arith.index_cast %count_index : index to i64
    %header, %meta, %digits = func.call @__ly_long_alloc_raw(%sign, %count) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    scf.for %iv = %c0 to %count_index step %c1 {
      %digit = memref.load %digits_in[%iv] : memref<?xi32>
      memref.store %digit, %digits[%iv] : memref<?xi32>
    }
    func.call @__ly_long_normalize(%meta, %digits, %count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"} {
    // Reading a wider value through the i64 window would silently truncate;
    // raise instead (never silently mis-execute).
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %fits, ^ok, ^too_large

  ^too_large:
    func.call @__ly_long_raise_too_large() : () -> ()
    cf.br ^ok

  ^ok:
    %result = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    func.return %result : i64
  }

  // 2^e for e in [-1022, 1023], built from the IEEE 754 exponent field. Powers
  // in this range are normal, so the significand is all zeros.
  func.func private @__ly_long_pow2_f64(%e: i64) -> f64 {
    %bias = arith.constant 1023 : i64
    %mant_bits = arith.constant 52 : i64
    %biased = arith.addi %e, %bias : i64
    %bits = arith.shli %biased, %mant_bits : i64
    %result = arith.bitcast %bits : i64 to f64
    func.return %result : f64
  }

  // Correctly-rounded int -> f64 (CPython PyLong_AsDouble / _PyLong_Frexp):
  // the top 55 bits are extracted exactly, every lower bit is folded into a
  // sticky bit, and the 55-bit window is rounded to 53 bits half-to-even.
  // The second result reports overflow (|x| rounds to >= 2^1024); the caller
  // decides which OverflowError-equivalent to raise.
  func.func private @__ly_long_view_as_f64_checked(%meta: memref<2xi64>, %digits: memref<?xi32>) -> (f64, i1) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %five = arith.constant 5 : i64
    %thirty = arith.constant 30 : i64
    %c30_minus = arith.constant 30 : i64
    %c60 = arith.constant 60 : i64
    %false = arith.constant false
    %zero_f64 = arith.constant 0.0 : f64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %is_negative = arith.cmpi slt, %sign, %zero : i64
    %nbits = func.call @__ly_long_bit_length(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %is_zero = arith.cmpi eq, %nbits, %zero : i64
    %magnitude:2 = scf.if %is_zero -> (f64, i1) {
      scf.yield %zero_f64, %false : f64, i1
    } else {
      %c63 = arith.constant 63 : i64
      %small = arith.cmpi sle, %nbits, %c63 : i64
      %inner:2 = scf.if %small -> (f64, i1) {
        // Up to 63 bits: assemble the unsigned magnitude and let uitofp do the
        // (correct, half-to-even) rounding.
        %c0 = arith.constant 0 : index
        %digit0_i32 = memref.load %digits[%c0] : memref<?xi32>
        %digit0 = arith.extui %digit0_i32 : i32 to i64
        %has_digit1 = arith.cmpi uge, %count, %two : i64
        %digit1 = scf.if %has_digit1 -> (i64) {
          %idx = arith.constant 1 : index
          %d_i32 = memref.load %digits[%idx] : memref<?xi32>
          %d = arith.extui %d_i32 : i32 to i64
          scf.yield %d : i64
        } else {
          scf.yield %zero : i64
        }
        %has_digit2 = arith.cmpi uge, %count, %three : i64
        %digit2 = scf.if %has_digit2 -> (i64) {
          %idx = arith.constant 2 : index
          %d_i32 = memref.load %digits[%idx] : memref<?xi32>
          %d = arith.extui %d_i32 : i32 to i64
          scf.yield %d : i64
        } else {
          scf.yield %zero : i64
        }
        %d1_shifted = arith.shli %digit1, %thirty : i64
        %d2_shifted = arith.shli %digit2, %c60 : i64
        %with_d1 = arith.ori %digit0, %d1_shifted : i64
        %mag = arith.ori %with_d1, %d2_shifted : i64
        %mag_f = arith.uitofp %mag : i64 to f64
        scf.yield %mag_f, %false : f64, i1
      } else {
        // shift >= 9 here, so divui/remui below are safe.
        %c55 = arith.constant 55 : i64
        %shift = arith.subi %nbits, %c55 : i64
        %digit_pos = arith.divui %shift, %thirty : i64
        %bit_pos = arith.remui %shift, %thirty : i64
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        // Window limb j (digit_pos + 0/1/2), zero when past the top limb.
        %limb0_index = arith.index_cast %digit_pos : i64 to index
        %l0_i32 = memref.load %digits[%limb0_index] : memref<?xi32>
        %l0 = arith.extui %l0_i32 : i32 to i64
        %pos1 = arith.addi %digit_pos, %one : i64
        %has_l1 = arith.cmpi ult, %pos1, %count : i64
        %l1 = scf.if %has_l1 -> (i64) {
          %idx = arith.index_cast %pos1 : i64 to index
          %d_i32 = memref.load %digits[%idx] : memref<?xi32>
          %d = arith.extui %d_i32 : i32 to i64
          scf.yield %d : i64
        } else {
          scf.yield %zero : i64
        }
        %pos2 = arith.addi %digit_pos, %two : i64
        %has_l2 = arith.cmpi ult, %pos2, %count : i64
        %l2 = scf.if %has_l2 -> (i64) {
          %idx = arith.index_cast %pos2 : i64 to index
          %d_i32 = memref.load %digits[%idx] : memref<?xi32>
          %d = arith.extui %d_i32 : i32 to i64
          scf.yield %d : i64
        } else {
          scf.yield %zero : i64
        }
        // top55 = magnitude >> shift; a nonzero l2 implies bit_pos >= 6, so
        // the shift amounts stay below 64 and nothing is truncated.
        %l0_part = arith.shrui %l0, %bit_pos : i64
        %l1_amount = arith.subi %c30_minus, %bit_pos : i64
        %l1_part = arith.shli %l1, %l1_amount : i64
        %l2_amount = arith.subi %c60, %bit_pos : i64
        %l2_part = arith.shli %l2, %l2_amount : i64
        %top_a = arith.ori %l0_part, %l1_part : i64
        %top55 = arith.ori %top_a, %l2_part : i64
        // Sticky: any bit below the window.
        %low_mask_full = arith.shli %one, %bit_pos : i64
        %low_mask = arith.subi %low_mask_full, %one : i64
        %l0_low = arith.andi %l0, %low_mask : i64
        %sticky_low = arith.cmpi ne, %l0_low, %zero : i64
        %digit_pos_index = arith.index_cast %digit_pos : i64 to index
        %sticky_rest = scf.for %iv = %c0 to %digit_pos_index step %c1 iter_args(%seen = %false) -> (i1) {
          %d_i32 = memref.load %digits[%iv] : memref<?xi32>
          %zero_i32 = arith.constant 0 : i32
          %nonzero = arith.cmpi ne, %d_i32, %zero_i32 : i32
          %next = arith.ori %seen, %nonzero : i1
          scf.yield %next : i1
        }
        %sticky = arith.ori %sticky_low, %sticky_rest : i1
        %sticky_i64 = arith.extui %sticky : i1 to i64
        %low = arith.ori %top55, %sticky_i64 : i64
        // Round half-to-even over the 2 extra bits (guard mask = 2; 3*mask-1
        // = 5 covers the sticky bit and the bit above the guard).
        %guard = arith.andi %low, %two : i64
        %guard_set = arith.cmpi ne, %guard, %zero : i64
        %near = arith.andi %low, %five : i64
        %near_set = arith.cmpi ne, %near, %zero : i64
        %round_up = arith.andi %guard_set, %near_set : i1
        %bumped = arith.addi %low, %two : i64
        %rounded = arith.select %round_up, %bumped, %low : i1, i64
        %m53 = arith.shrui %rounded, %two : i64
        // Overflow when the rounded magnitude reaches 2^1024.
        %c1024 = arith.constant 1024 : i64
        %c2_53 = arith.constant 9007199254740992 : i64
        %too_wide = arith.cmpi sgt, %nbits, %c1024 : i64
        %at_limit = arith.cmpi eq, %nbits, %c1024 : i64
        %mant_carry = arith.cmpi eq, %m53, %c2_53 : i64
        %carry_overflow = arith.andi %at_limit, %mant_carry : i1
        %overflow = arith.ori %too_wide, %carry_overflow : i1
        %big:2 = scf.if %overflow -> (f64, i1) {
          %true = arith.constant true
          scf.yield %zero_f64, %true : f64, i1
        } else {
          %c53 = arith.constant 53 : i64
          %e2 = arith.subi %nbits, %c53 : i64
          %scale = func.call @__ly_long_pow2_f64(%e2) : (i64) -> f64
          %m53_f = arith.uitofp %m53 : i64 to f64
          %mag_f = arith.mulf %m53_f, %scale : f64
          scf.yield %mag_f, %false : f64, i1
        }
        scf.yield %big#0, %big#1 : f64, i1
      }
      scf.yield %inner#0, %inner#1 : f64, i1
    }
    %negated = arith.negf %magnitude#0 : f64
    %signed = arith.select %is_negative, %negated, %magnitude#0 : i1, f64
    func.return %signed, %magnitude#1 : f64, i1
  }

  // Conversion entry used by unbox.f64 / __float__: correctly rounded, and
  // magnitudes at or beyond 2^1024 raise like CPython's float(int).
  func.func private @__ly_long_view_as_f64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> f64 {
    %value, %overflow = func.call @__ly_long_view_as_f64_checked(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> (f64, i1)
    cf.cond_br %overflow, ^too_large, ^ok

  ^too_large:
    func.call @__ly_long_raise_too_large_for_float() : () -> ()
    cf.br ^ok

  ^ok:
    func.return %value : f64
  }

  func.func @LyLong_AsF64(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> f64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "unbox.f64"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %value = func.call @__ly_long_view_as_f64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> f64
    func.return %value : f64
  }

  // Runtime-level float(int) (`__float__` on int): correctly-rounded digit
  // conversion boxed as a float object.
  func.func @LyLong_Float(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__float__", ly.runtime.result_contract = "builtins.float"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %value = func.call @__ly_long_view_as_f64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> f64
    %h, %p = func.call @LyFloat_FromF64(%value) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %h, %p : memref<2xi64>, memref<1xf64>
  }

  func.func @LyLong_Init(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyLong_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__bool__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %count_slot = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %result = arith.cmpi ne, %count, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_Pos(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__pos__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %h, %m, %d = func.call @__ly_long_copy(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Neg(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__neg__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %sign_slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %negated = arith.subi %zero, %sign : i64
    %h, %m, %d = func.call @__ly_long_copy_with_sign(%negated, %meta, %digits) : (i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Invert(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__invert__"} {
    %neg_h, %neg_m, %neg_d = func.call @LyLong_Neg(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %one = arith.constant 1 : i64
    %one_h, %one_m, %one_d = func.call @LyLong_FromI64(%one) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %result_h, %result_m, %result_d = func.call @LyLong_Sub(%neg_h, %neg_m, %neg_d, %one_h, %one_m, %one_d) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%neg_h) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%one_h) : (memref<2xi64>) -> ()
    func.return %result_h, %result_m, %result_d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Reads a (meta, digits) view whose magnitude is known to fit i64. Used by
  // AsI64 and by the small-operand fast paths of the arithmetic entry points.
  func.func private @__ly_long_view_as_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 {
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %ndigits = memref.load %meta[%digit_count_slot] : memref<2xi64>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %shift30 = arith.constant 30 : i64
    %shift60 = arith.constant 60 : i64
    %digit0_index = arith.constant 0 : index
    %digit0_i32 = memref.load %digits[%digit0_index] : memref<?xi32>
    %digit0 = arith.extui %digit0_i32 : i32 to i64
    %has_digit1 = arith.cmpi uge, %ndigits, %two : i64
    %digit1 = scf.if %has_digit1 -> (i64) {
      %idx = arith.constant 1 : index
      %digit_i32 = memref.load %digits[%idx] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      scf.yield %digit : i64
    } else {
      scf.yield %zero : i64
    }
    %has_digit2 = arith.cmpi uge, %ndigits, %three : i64
    %digit2 = scf.if %has_digit2 -> (i64) {
      %idx = arith.constant 2 : index
      %digit_i32 = memref.load %digits[%idx] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      scf.yield %digit : i64
    } else {
      scf.yield %zero : i64
    }
    %digit1_shifted = arith.shli %digit1, %shift30 : i64
    %with_digit1 = arith.ori %digit0, %digit1_shifted : i64
    %digit2_shifted = arith.shli %digit2, %shift60 : i64
    %magnitude = arith.ori %with_digit1, %digit2_shifted : i64
    %negated = arith.subi %zero, %magnitude : i64
    %is_negative = arith.cmpi slt, %sign, %zero : i64
    %signed = arith.select %is_negative, %negated, %magnitude : i1, i64
    %is_zero = arith.cmpi eq, %ndigits, %zero : i64
    %result = arith.select %is_zero, %zero, %signed : i1, i64
    func.return %result : i64
  }

  func.func private @__ly_long_view_fits_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i1 {
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %eight = arith.constant 8 : i64
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    %ndigits = memref.load %meta[%digit_count_slot] : memref<2xi64>
    %fits_two_limbs = arith.cmpi sle, %ndigits, %two : i64
    %fits = scf.if %fits_two_limbs -> (i1) {
      %true = arith.constant true
      scf.yield %true : i1
    } else {
      %has_three_limbs = arith.cmpi eq, %ndigits, %three : i64
      %three_limb_fits = scf.if %has_three_limbs -> (i1) {
        %digit0_slot = arith.constant 0 : index
        %digit1_slot = arith.constant 1 : index
        %digit2_slot = arith.constant 2 : index
        %digit0_i32 = memref.load %digits[%digit0_slot] : memref<?xi32>
        %digit1_i32 = memref.load %digits[%digit1_slot] : memref<?xi32>
        %digit2_i32 = memref.load %digits[%digit2_slot] : memref<?xi32>
        %digit0 = arith.extui %digit0_i32 : i32 to i64
        %digit1 = arith.extui %digit1_i32 : i32 to i64
        %digit2 = arith.extui %digit2_i32 : i32 to i64
        %high_lt_limit = arith.cmpi ult, %digit2, %eight : i64
        %high_eq_limit = arith.cmpi eq, %digit2, %eight : i64
        %low0_zero = arith.cmpi eq, %digit0, %zero : i64
        %low1_zero = arith.cmpi eq, %digit1, %zero : i64
        %low_zero = arith.andi %low0_zero, %low1_zero : i1
        %sign = memref.load %meta[%sign_slot] : memref<2xi64>
        %negative = arith.cmpi slt, %sign, %zero : i64
        %min_i64 = arith.andi %high_eq_limit, %low_zero : i1
        %negative_min_i64 = arith.andi %min_i64, %negative : i1
        %result = arith.ori %high_lt_limit, %negative_min_i64 : i1
        scf.yield %result : i1
      } else {
        %false = arith.constant false
        scf.yield %false : i1
      }
      scf.yield %three_limb_fits : i1
    }
    func.return %fits : i1
  }

  func.func private @__ly_long_raise_too_large() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 51 : i64
    %message_static = memref.get_global @__ly_long_msg_int_too_large : memref<51xi8>
    %message = memref.cast %message_static : memref<51xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_invalid_literal() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 38 : i64
    %message_static = memref.get_global @__ly_long_msg_invalid_int_literal : memref<38xi8>
    %message = memref.cast %message_static : memref<38xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_float_nan() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 35 : i64
    %message_static = memref.get_global @__ly_long_msg_float_nan : memref<35xi8>
    %message = memref.cast %message_static : memref<35xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_float_infinity() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 40 : i64
    %message_static = memref.get_global @__ly_long_msg_float_infinity : memref<40xi8>
    %message = memref.cast %message_static : memref<40xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_too_large_for_float() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 33 : i64
    %message_static = memref.get_global @__ly_long_msg_int_too_large_float : memref<33xi8>
    %message = memref.cast %message_static : memref<33xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_div_result_too_large() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 45 : i64
    %message_static = memref.get_global @__ly_long_msg_div_result_too_large : memref<45xi8>
    %message = memref.cast %message_static : memref<45xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_zero_negative_power() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 24 : i64
    %message_static = memref.get_global @__ly_long_msg_zero_negative_power : memref<24xi8>
    %message = memref.cast %message_static : memref<24xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_fractional_power_negative() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 54 : i64
    %message_static = memref.get_global @__ly_long_msg_fractional_power_negative : memref<54xi8>
    %message = memref.cast %message_static : memref<54xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_pow_negative() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 91 : i64
    %message_static = memref.get_global @__ly_long_msg_pow_negative_exponent : memref<91xi8>
    %message = memref.cast %message_static : memref<91xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_shift_too_large() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 21 : i64
    %message_static = memref.get_global @__ly_long_msg_shift_count_too_large : memref<21xi8>
    %message = memref.cast %message_static : memref<21xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_bit_length(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %is_zero = arith.cmpi eq, %count, %zero : i64
    %result = scf.if %is_zero -> (i64) {
      scf.yield %zero : i64
    } else {
      %c1 = arith.constant 1 : index
      %count_index = arith.index_cast %count : i64 to index
      %top_index = arith.subi %count_index, %c1 : index
      %top_i32 = memref.load %digits[%top_index] : memref<?xi32>
      %top = arith.extui %top_i32 : i32 to i64
      %c0 = arith.constant 0 : index
      %c30 = arith.constant 30 : index
      %top_bits = scf.for %iv = %c0 to %c30 step %c1 iter_args(%bits = %zero) -> (i64) {
        %iv_i64 = arith.index_cast %iv : index to i64
        %shifted = arith.shrui %top, %iv_i64 : i64
        %nonzero = arith.cmpi ne, %shifted, %zero : i64
        %next_bits = arith.addi %iv_i64, %one : i64
        %next = arith.select %nonzero, %next_bits, %bits : i1, i64
        scf.yield %next : i64
      }
      %full = arith.subi %count, %one : i64
      %full_bits = arith.muli %full, %thirty : i64
      %total = arith.addi %full_bits, %top_bits : i64
      scf.yield %total : i64
    }
    func.return %result : i64
  }

  // |lhs| = q * |rhs| + r with 0 <= r < |rhs|. Requires rhs != 0. Both results
  // are freshly allocated (never the immortal small-int cache), so callers may
  // flip signs / increment magnitudes in place before publishing them.
  //
  // Multi-digit divisors use bit-by-bit shift-subtract long division rather
  // than Knuth's algorithm D: the quotient-digit estimation/correction loop is
  // hard to verify in handwritten scf/arith, and correctness is the current
  // gate; swap in D behind this same contract if division ever profiles hot.
  func.func private @__ly_long_divmod_abs(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0, 3]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %base = arith.constant 1073741824 : i64
    %mask = arith.constant 1073741823 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %cmp = func.call @__ly_long_abs_compare(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
    %lhs_smaller = arith.cmpi slt, %cmp, %zero : i64
    %result:6 = scf.if %lhs_smaller -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %qh, %qm, %qd = func.call @__ly_long_alloc_raw(%zero, %zero) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      %rh, %rm, %rd = func.call @__ly_long_copy_with_sign(%one, %lhs_meta, %lhs_digits) : (i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %qh, %qm, %qd, %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
      %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
      // One spare limb so the floor adjustment (q + 1 in magnitude) can never
      // carry out of the allocation.
      %q_capacity = arith.addi %lhs_count, %one : i64
      %single = arith.cmpi eq, %rhs_count, %one : i64
      %inner:6 = scf.if %single -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) {
        %divisor_i32 = memref.load %rhs_digits[%c0] : memref<?xi32>
        %divisor = arith.extui %divisor_i32 : i32 to i64
        %qh, %qm, %qd = func.call @__ly_long_alloc_raw(%one, %q_capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
        %count_index = arith.index_cast %lhs_count : i64 to index
        %rem_final = scf.for %iv = %c0 to %count_index step %c1 iter_args(%rem = %zero) -> (i64) {
          %iv_next = arith.addi %iv, %c1 : index
          %rev = arith.subi %count_index, %iv_next : index
          %digit_i32 = memref.load %lhs_digits[%rev] : memref<?xi32>
          %digit = arith.extui %digit_i32 : i32 to i64
          %scaled = arith.muli %rem, %base : i64
          %acc = arith.addi %scaled, %digit : i64
          %q_digit = arith.divui %acc, %divisor : i64
          %rem_next = arith.remui %acc, %divisor : i64
          %q_digit_i32 = arith.trunci %q_digit : i64 to i32
          memref.store %q_digit_i32, %qd[%rev] : memref<?xi32>
          scf.yield %rem_next : i64
        }
        func.call @__ly_long_normalize(%qm, %qd, %q_capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
        %rem_sign = arith.cmpi ne, %rem_final, %zero : i64
        %r_sign = arith.select %rem_sign, %one, %zero : i1, i64
        %rh, %rm, %rd = func.call @__ly_long_alloc_raw(%r_sign, %one) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
        %rem_i32 = arith.trunci %rem_final : i64 to i32
        memref.store %rem_i32, %rd[%c0] : memref<?xi32>
        func.call @__ly_long_normalize(%rm, %rd, %one) : (memref<2xi64>, memref<?xi32>, i64) -> ()
        scf.yield %qh, %qm, %qd, %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
      } else {
        %nbits = func.call @__ly_long_bit_length(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
        %qh, %qm, %qd = func.call @__ly_long_alloc_raw(%one, %q_capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
        %r_capacity = arith.addi %rhs_count, %one : i64
        %rh, %rm, %rd = func.call @__ly_long_alloc_raw(%one, %r_capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
        %r_limbs = arith.index_cast %r_capacity : i64 to index
        %nbits_index = arith.index_cast %nbits : i64 to index
        scf.for %k = %c0 to %nbits_index step %c1 {
          %k_i64 = arith.index_cast %k : index to i64
          %j_plus = arith.subi %nbits, %k_i64 : i64
          %j = arith.subi %j_plus, %one : i64
          %digit_pos = arith.divui %j, %thirty : i64
          %bit_pos = arith.remui %j, %thirty : i64
          %digit_index = arith.index_cast %digit_pos : i64 to index
          %src_i32 = memref.load %lhs_digits[%digit_index] : memref<?xi32>
          %src = arith.extui %src_i32 : i32 to i64
          %shifted_src = arith.shrui %src, %bit_pos : i64
          %bit = arith.andi %shifted_src, %one : i64
          // r = (r << 1) | bit; r stays < 2*|rhs| <= capacity of r_capacity limbs.
          %shift_carry = scf.for %i = %c0 to %r_limbs step %c1 iter_args(%carry = %bit) -> (i64) {
            %d_i32 = memref.load %rd[%i] : memref<?xi32>
            %d = arith.extui %d_i32 : i32 to i64
            %doubled = arith.shli %d, %one : i64
            %with_carry = arith.ori %doubled, %carry : i64
            %out = arith.andi %with_carry, %mask : i64
            %out_i32 = arith.trunci %out : i64 to i32
            memref.store %out_i32, %rd[%i] : memref<?xi32>
            %twenty_nine = arith.constant 29 : i64
            %next_carry = arith.shrui %d, %twenty_nine : i64
            scf.yield %next_carry : i64
          }
          // ge = (r >= |rhs|), scanning limbs from the top with rhs padded to
          // r_capacity limbs.
          %cmp_state = scf.for %i = %c0 to %r_limbs step %c1 iter_args(%state = %zero) -> (i64) {
            %i_next = arith.addi %i, %c1 : index
            %rev = arith.subi %r_limbs, %i_next : index
            %r_digit_i32 = memref.load %rd[%rev] : memref<?xi32>
            %r_digit = arith.extui %r_digit_i32 : i32 to i64
            %rev_i64 = arith.index_cast %rev : index to i64
            %has_rhs = arith.cmpi slt, %rev_i64, %rhs_count : i64
            %rhs_digit = scf.if %has_rhs -> (i64) {
              %d_i32 = memref.load %rhs_digits[%rev] : memref<?xi32>
              %d = arith.extui %d_i32 : i32 to i64
              scf.yield %d : i64
            } else {
              scf.yield %zero : i64
            }
            %still_equal = arith.cmpi eq, %state, %zero : i64
            %gt = arith.cmpi ugt, %r_digit, %rhs_digit : i64
            %lt = arith.cmpi ult, %r_digit, %rhs_digit : i64
            %neg_one = arith.constant -1 : i64
            %c = arith.select %gt, %one, %zero : i1, i64
            %c2 = arith.select %lt, %neg_one, %c : i1, i64
            %next = arith.select %still_equal, %c2, %state : i1, i64
            scf.yield %next : i64
          }
          %r_ge = arith.cmpi sge, %cmp_state, %zero : i64
          scf.if %r_ge {
            // r -= |rhs|; record the quotient bit.
            %borrow_out = scf.for %i = %c0 to %r_limbs step %c1 iter_args(%borrow = %zero) -> (i64) {
              %r_digit_i32 = memref.load %rd[%i] : memref<?xi32>
              %r_digit = arith.extui %r_digit_i32 : i32 to i64
              %i_i64 = arith.index_cast %i : index to i64
              %has_rhs = arith.cmpi slt, %i_i64, %rhs_count : i64
              %rhs_digit = scf.if %has_rhs -> (i64) {
                %d_i32 = memref.load %rhs_digits[%i] : memref<?xi32>
                %d = arith.extui %d_i32 : i32 to i64
                scf.yield %d : i64
              } else {
                scf.yield %zero : i64
              }
              %sub = arith.addi %rhs_digit, %borrow : i64
              %needs_borrow = arith.cmpi ult, %r_digit, %sub : i64
              %raw = arith.subi %r_digit, %sub : i64
              %borrowed = arith.addi %raw, %base : i64
              %val = arith.select %needs_borrow, %borrowed, %raw : i1, i64
              %val_i32 = arith.trunci %val : i64 to i32
              memref.store %val_i32, %rd[%i] : memref<?xi32>
              %next_borrow = arith.select %needs_borrow, %one, %zero : i1, i64
              scf.yield %next_borrow : i64
            }
            %q_slot = arith.index_cast %digit_pos : i64 to index
            %q_old_i32 = memref.load %qd[%q_slot] : memref<?xi32>
            %q_old = arith.extui %q_old_i32 : i32 to i64
            %q_bit = arith.shli %one, %bit_pos : i64
            %q_new = arith.ori %q_old, %q_bit : i64
            %q_new_i32 = arith.trunci %q_new : i64 to i32
            memref.store %q_new_i32, %qd[%q_slot] : memref<?xi32>
          }
        }
        func.call @__ly_long_normalize(%qm, %qd, %q_capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
        func.call @__ly_long_normalize(%rm, %rd, %r_capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
        scf.yield %qh, %qm, %qd, %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
      }
      scf.yield %inner#0, %inner#1, %inner#2, %inner#3, %inner#4, %inner#5 : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2, %result#3, %result#4, %result#5 : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Floor-divmod on signed views. Owns nothing on entry; returns fresh owned
  // (q, r) satisfying lhs = q * rhs + r with r sharing rhs's sign (CPython).
  // Requires rhs != 0 (callers raise the operator-specific ZeroDivisionError).
  func.func private @__ly_long_floor_divmod(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0, 3]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %q_abs:6 = func.call @__ly_long_divmod_abs(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %sign_product = arith.muli %lhs_sign, %rhs_sign : i64
    %opposite = arith.cmpi slt, %sign_product, %zero : i64
    %r_count = memref.load %q_abs#4[%count_slot] : memref<2xi64>
    %r_nonzero = arith.cmpi ne, %r_count, %zero : i64
    %adjust = arith.andi %opposite, %r_nonzero : i1
    // cf blocks (not scf.if): the ownership verifier tracks the conditional
    // consumption of q_abs#3 per path, but not through scf.if region yields.
    cf.cond_br %adjust, ^flip, ^keep

  ^flip:
    // q = -(|q| + 1). divmod_abs left one spare limb, and |rhs| >= 2 here
    // (a remainder forces it), so the in-place increment cannot carry out.
    %q_count = memref.load %q_abs#1[%count_slot] : memref<2xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mask = arith.constant 1073741823 : i64
    %count_index = arith.index_cast %q_count : i64 to index
    %carry_final = scf.for %iv = %c0 to %count_index step %c1 iter_args(%carry = %one) -> (i64) {
      %d_i32 = memref.load %q_abs#2[%iv] : memref<?xi32>
      %d = arith.extui %d_i32 : i32 to i64
      %sum = arith.addi %d, %carry : i64
      %out = arith.andi %sum, %mask : i64
      %out_i32 = arith.trunci %out : i64 to i32
      memref.store %out_i32, %q_abs#2[%iv] : memref<?xi32>
      %thirty = arith.constant 30 : i64
      %next = arith.shrui %sum, %thirty : i64
      scf.yield %next : i64
    }
    %has_carry = arith.cmpi ne, %carry_final, %zero : i64
    %new_count = scf.if %has_carry -> (i64) {
      %one_i32 = arith.constant 1 : i32
      memref.store %one_i32, %q_abs#2[%count_index] : memref<?xi32>
      %grown = arith.addi %q_count, %one : i64
      scf.yield %grown : i64
    } else {
      scf.yield %q_count : i64
    }
    memref.store %new_count, %q_abs#1[%count_slot] : memref<2xi64>
    memref.store %neg_one, %q_abs#1[%sign_slot] : memref<2xi64>
    // r = sign(rhs) * (|rhs| - |r|).
    %rh, %rm, %rd = func.call @__ly_long_sub_abs(%rhs_sign, %rhs_meta, %rhs_digits, %q_abs#4, %q_abs#5) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%q_abs#3) : (memref<2xi64>) -> ()
    func.return %q_abs#0, %q_abs#1, %q_abs#2, %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^keep:
    // Same signs (or exact division): q keeps sign_product's sign, r keeps
    // rhs's sign. Both are fresh divmod_abs allocations, so store in place;
    // a zero magnitude keeps sign 0 from normalize.
    %kq_count = memref.load %q_abs#1[%count_slot] : memref<2xi64>
    %q_zero = arith.cmpi eq, %kq_count, %zero : i64
    %q_negative = arith.cmpi slt, %sign_product, %zero : i64
    %q_signed = arith.select %q_negative, %neg_one, %one : i1, i64
    %q_sign = arith.select %q_zero, %zero, %q_signed : i1, i64
    memref.store %q_sign, %q_abs#1[%sign_slot] : memref<2xi64>
    %rhs_negative = arith.cmpi slt, %rhs_sign, %zero : i64
    %r_signed = arith.select %rhs_negative, %neg_one, %one : i1, i64
    %r_sign = arith.select %r_nonzero, %r_signed, %zero : i1, i64
    memref.store %r_sign, %q_abs#4[%sign_slot] : memref<2xi64>
    func.return %q_abs#0, %q_abs#1, %q_abs#2, %q_abs#3, %q_abs#4, %q_abs#5 : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Add(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__add__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_two_limb = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_two_limb = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_two_limb = arith.andi %lhs_two_limb, %rhs_two_limb : i1
    cf.cond_br %both_two_limb, ^small, ^maybe_i64

  ^maybe_i64:
    %lhs_i64 = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_i64 = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_i64 = arith.andi %lhs_i64, %rhs_i64 : i1
    cf.cond_br %both_i64, ^small, ^digits

  ^small:
    // Use primitive arithmetic only while the result also fits signed i64.
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_sum = arith.addi %small_a, %small_b : i64
    %small_zero = arith.constant 0 : i64
    %small_a_negative = arith.cmpi slt, %small_a, %small_zero : i64
    %small_b_negative = arith.cmpi slt, %small_b, %small_zero : i64
    %small_sum_negative = arith.cmpi slt, %small_sum, %small_zero : i64
    %same_input_sign = arith.cmpi eq, %small_a_negative, %small_b_negative : i1
    %result_changed_sign = arith.cmpi ne, %small_sum_negative, %small_a_negative : i1
    %overflow = arith.andi %same_input_sign, %result_changed_sign : i1
    cf.cond_br %overflow, ^digits, ^small_fit

  ^small_fit:
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_sum) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %h, %m, %d = func.call @__ly_long_add_signed_general(%lhs_meta, %lhs_digits, %rhs_sign, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Sub(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__sub__"} {
    %lhs_meta_view, %lhs_digits_view = func.call @__ly_long_operand_view(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta_view[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_two_limb = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_two_limb = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_two_limb = arith.andi %lhs_two_limb, %rhs_two_limb : i1
    cf.cond_br %both_two_limb, ^small, ^maybe_i64

  ^maybe_i64:
    %lhs_i64 = func.call @__ly_long_view_fits_i64(%lhs_meta_view, %lhs_digits_view) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_i64 = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_i64 = arith.andi %lhs_i64, %rhs_i64 : i1
    cf.cond_br %both_i64, ^small, ^digits

  ^small:
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta_view, %lhs_digits_view) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_diff = arith.subi %small_a, %small_b : i64
    %small_zero = arith.constant 0 : i64
    %small_a_negative = arith.cmpi slt, %small_a, %small_zero : i64
    %small_b_negative = arith.cmpi slt, %small_b, %small_zero : i64
    %small_diff_negative = arith.cmpi slt, %small_diff, %small_zero : i64
    %different_input_sign = arith.cmpi ne, %small_a_negative, %small_b_negative : i1
    %result_changed_sign = arith.cmpi ne, %small_diff_negative, %small_a_negative : i1
    %overflow = arith.andi %different_input_sign, %result_changed_sign : i1
    cf.cond_br %overflow, ^digits, ^small_fit

  ^small_fit:
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_diff) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %rhs_neg = arith.subi %zero, %rhs_sign : i64
    %h, %m, %d = func.call @__ly_long_add_signed_general(%lhs_meta_view, %lhs_digits_view, %rhs_neg, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Mul(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mul__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_two_limb = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_two_limb = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_two_limb = arith.andi %lhs_two_limb, %rhs_two_limb : i1
    cf.cond_br %both_two_limb, ^small, ^maybe_i64

  ^maybe_i64:
    %lhs_i64 = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_i64 = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_i64 = arith.andi %lhs_i64, %rhs_i64 : i1
    cf.cond_br %both_i64, ^small, ^digits

  ^small:
    // Use primitive multiplication only when the product remains in i64.
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_low, %small_high = arith.mulsi_extended %small_a, %small_b : i64
    %small_c63 = arith.constant 63 : i64
    %small_sign_ext = arith.shrsi %small_low, %small_c63 : i64
    %small_fits = arith.cmpi eq, %small_high, %small_sign_ext : i64
    cf.cond_br %small_fits, ^small_fit, ^digits

  ^small_fit:
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_low) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %mask = arith.constant 1073741823 : i64
    %shift30 = arith.constant 30 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
    %any_zero = arith.ori %lhs_zero, %rhs_zero : i1
    %result:3 = scf.if %any_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %sign = arith.muli %lhs_sign, %rhs_sign : i64
      %sum_count = arith.addi %lhs_count, %rhs_count : i64
      %capacity = arith.addi %sum_count, %one : i64
      %h, %m, %d = func.call @__ly_long_alloc_raw(%sign, %capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %lhs_index = arith.index_cast %lhs_count : i64 to index
      %rhs_index = arith.index_cast %rhs_count : i64 to index
      %capacity_index = arith.index_cast %capacity : i64 to index
      scf.for %i = %c0 to %lhs_index step %c1 {
        %lhs_digit_i32 = memref.load %lhs_digits[%i] : memref<?xi32>
        %lhs_digit = arith.extui %lhs_digit_i32 : i32 to i64
        %carry = scf.for %j = %c0 to %rhs_index step %c1 iter_args(%carry_iter = %zero) -> (i64) {
          %out_index = arith.addi %i, %j : index
          %rhs_digit_i32 = memref.load %rhs_digits[%j] : memref<?xi32>
          %rhs_digit = arith.extui %rhs_digit_i32 : i32 to i64
          %old_i32 = memref.load %d[%out_index] : memref<?xi32>
          %old = arith.extui %old_i32 : i32 to i64
          %product = arith.muli %lhs_digit, %rhs_digit : i64
          %with_old = arith.addi %product, %old : i64
          %sum = arith.addi %with_old, %carry_iter : i64
          %out_i64 = arith.andi %sum, %mask : i64
          %out = arith.trunci %out_i64 : i64 to i32
          memref.store %out, %d[%out_index] : memref<?xi32>
          %next_carry = arith.shrui %sum, %shift30 : i64
          scf.yield %next_carry : i64
        }
        %tail_index = arith.addi %i, %rhs_index : index
        %tail_old_i32 = memref.load %d[%tail_index] : memref<?xi32>
        %tail_old = arith.extui %tail_old_i32 : i32 to i64
        %tail_sum = arith.addi %tail_old, %carry : i64
        %tail_out_i64 = arith.andi %tail_sum, %mask : i64
        %tail_out = arith.trunci %tail_out_i64 : i64 to i32
        memref.store %tail_out, %d[%tail_index] : memref<?xi32>
        %tail_carry = arith.shrui %tail_sum, %shift30 : i64
        %prop_start = arith.addi %tail_index, %c1 : index
        %ignored = scf.for %k = %prop_start to %capacity_index step %c1 iter_args(%carry_prop = %tail_carry) -> (i64) {
          %has_carry = arith.cmpi ne, %carry_prop, %zero : i64
          %next_carry = scf.if %has_carry -> (i64) {
            %old_i32 = memref.load %d[%k] : memref<?xi32>
            %old = arith.extui %old_i32 : i32 to i64
            %sum = arith.addi %old, %carry_prop : i64
            %out_i64 = arith.andi %sum, %mask : i64
            %out = arith.trunci %out_i64 : i64 to i32
            memref.store %out, %d[%k] : memref<?xi32>
            %carry_next = arith.shrui %sum, %shift30 : i64
            scf.yield %carry_next : i64
          } else {
            scf.yield %zero : i64
          }
          scf.yield %next_carry : i64
        }
      }
      func.call @__ly_long_normalize(%m, %d, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // |a| << k as a fresh positive magnitude (k >= 0). Sequential carry form of
  // CPython's v_lshift over the 30-bit limbs.
  func.func private @__ly_long_abs_lshift_raw(%meta: memref<2xi64>, %digits: memref<?xi32>, %k: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %offset = arith.divui %k, %thirty : i64
    %d = arith.remui %k, %thirty : i64
    %inv_d = arith.subi %thirty, %d : i64
    %count_plus = arith.addi %count, %offset : i64
    %capacity = arith.addi %count_plus, %one : i64
    %h, %m, %out = func.call @__ly_long_alloc_raw(%one, %capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = arith.index_cast %count : i64 to index
    %offset_index = arith.index_cast %offset : i64 to index
    %carry = scf.for %iv = %c0 to %count_index step %c1 iter_args(%carry_iter = %zero) -> (i64) {
      %digit_i32 = memref.load %digits[%iv] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %shifted = arith.shli %digit, %d : i64
      %with_carry = arith.ori %shifted, %carry_iter : i64
      %low = arith.andi %with_carry, %mask : i64
      %low_i32 = arith.trunci %low : i64 to i32
      %slot = arith.addi %iv, %offset_index : index
      memref.store %low_i32, %out[%slot] : memref<?xi32>
      // d == 0 shifts a 30-bit digit right by 30, which is zero: no carry.
      %next_carry = arith.shrui %digit, %inv_d : i64
      scf.yield %next_carry : i64
    }
    %carry_slot = arith.index_cast %count_plus : i64 to index
    %carry_i32 = arith.trunci %carry : i64 to i32
    memref.store %carry_i32, %out[%carry_slot] : memref<?xi32>
    func.call @__ly_long_normalize(%m, %out, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %m, %out : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // |a| >> k as a fresh positive magnitude (0 <= k < bit_length(a)), plus a
  // sticky flag: whether any shifted-out bit was nonzero.
  func.func private @__ly_long_abs_rshift_raw(%meta: memref<2xi64>, %digits: memref<?xi32>, %k: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, i1) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %false = arith.constant false
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %offset = arith.divui %k, %thirty : i64
    %d = arith.remui %k, %thirty : i64
    %inv_d = arith.subi %thirty, %d : i64
    %out_count = arith.subi %count, %offset : i64
    %h, %m, %out = func.call @__ly_long_alloc_raw(%one, %out_count) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out_index = arith.index_cast %out_count : i64 to index
    %offset_index = arith.index_cast %offset : i64 to index
    scf.for %iv = %c0 to %out_index step %c1 {
      %src = arith.addi %iv, %offset_index : index
      %digit_i32 = memref.load %digits[%src] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %low = arith.shrui %digit, %d : i64
      %src_next = arith.addi %src, %c1 : index
      %src_next_i64 = arith.index_cast %src_next : index to i64
      %has_next = arith.cmpi slt, %src_next_i64, %count : i64
      %high = scf.if %has_next -> (i64) {
        %next_i32 = memref.load %digits[%src_next] : memref<?xi32>
        %next = arith.extui %next_i32 : i32 to i64
        // d == 0 shifts left by 30 and masks to zero, as required.
        %shifted = arith.shli %next, %inv_d : i64
        %masked = arith.andi %shifted, %mask : i64
        scf.yield %masked : i64
      } else {
        scf.yield %zero : i64
      }
      %combined = arith.ori %low, %high : i64
      %combined_i32 = arith.trunci %combined : i64 to i32
      memref.store %combined_i32, %out[%iv] : memref<?xi32>
    }
    %low_mask_full = arith.shli %one, %d : i64
    %low_mask = arith.subi %low_mask_full, %one : i64
    %boundary_i32 = memref.load %digits[%offset_index] : memref<?xi32>
    %boundary = arith.extui %boundary_i32 : i32 to i64
    %boundary_low = arith.andi %boundary, %low_mask : i64
    %sticky_low = arith.cmpi ne, %boundary_low, %zero : i64
    %sticky = scf.for %iv = %c0 to %offset_index step %c1 iter_args(%seen = %false) -> (i1) {
      %digit_i32 = memref.load %digits[%iv] : memref<?xi32>
      %zero_i32 = arith.constant 0 : i32
      %nonzero = arith.cmpi ne, %digit_i32, %zero_i32 : i32
      %next = arith.ori %seen, %nonzero : i1
      scf.yield %next : i1
    }
    %sticky_any = arith.ori %sticky_low, %sticky : i1
    func.call @__ly_long_normalize(%m, %out, %out_count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %m, %out, %sticky_any : memref<2xi64>, memref<2xi64>, memref<?xi32>, i1
  }

  // Correctly-rounded int / int (port of CPython long_true_divide): compute
  // x = floor(|a| * 2^-shift / |b|) with 55..57 significant bits plus an
  // inexactness flag, round x half-to-even at the float precision implied by
  // shift, and scale back by 2^shift. The chosen shift clamps at DBL_MIN_EXP
  // so subnormal results round once, exactly as a double would.
  func.func @LyLong_TrueDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__truediv__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sign_slot = arith.constant 0 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %lhs_negative = arith.cmpi slt, %lhs_sign, %zero : i64
    %rhs_negative = arith.cmpi slt, %rhs_sign, %zero : i64
    %negate = arith.cmpi ne, %lhs_negative, %rhs_negative : i1
    %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
    cf.cond_br %rhs_zero, ^zero_divisor, ^check_zero_lhs

  ^zero_divisor:
    func.call @__ly_long_raise_true_div_zero() : () -> ()
    %dummy = arith.constant 0.0 : f64
    %zh, %zp = func.call @LyFloat_FromF64(%dummy) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %zh, %zp : memref<2xi64>, memref<1xf64>

  ^check_zero_lhs:
    %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    cf.cond_br %lhs_zero, ^signed_zero, ^widths

  ^signed_zero:
    // 0 / b keeps the sign of the quotient (CPython returns -0.0 for 0/-b).
    %pos_zero = arith.constant 0.0 : f64
    %neg_zero = arith.negf %pos_zero : f64
    %signed_zero = arith.select %negate, %neg_zero, %pos_zero : i1, f64
    %szh, %szp = func.call @LyFloat_FromF64(%signed_zero) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %szh, %szp : memref<2xi64>, memref<1xf64>

  ^widths:
    %a_bits = func.call @__ly_long_bit_length(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %b_bits = func.call @__ly_long_bit_length(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %diff = arith.subi %a_bits, %b_bits : i64
    %max_exp = arith.constant 1024 : i64
    %overflow_early = arith.cmpi sgt, %diff, %max_exp : i64
    cf.cond_br %overflow_early, ^overflow, ^check_underflow

  ^overflow:
    func.call @__ly_long_raise_div_result_too_large() : () -> ()
    %odummy = arith.constant 0.0 : f64
    %oh, %op = func.call @LyFloat_FromF64(%odummy) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %oh, %op : memref<2xi64>, memref<1xf64>

  ^check_underflow:
    %underflow_limit = arith.constant -1075 : i64
    %underflows = arith.cmpi slt, %diff, %underflow_limit : i64
    cf.cond_br %underflows, ^signed_zero, ^divide

  ^divide:
    // shift = max(diff, DBL_MIN_EXP) - DBL_MANT_DIG - 2 (see the CPython
    // comment for why the DBL_MIN_EXP clamp avoids double rounding).
    %min_exp = arith.constant -1021 : i64
    %diff_ge_min = arith.cmpi sge, %diff, %min_exp : i64
    %clamped = arith.select %diff_ge_min, %diff, %min_exp : i1, i64
    %c55 = arith.constant 55 : i64
    %shift = arith.subi %clamped, %c55 : i64
    %shift_positive = arith.cmpi sgt, %shift, %zero : i64
    %x:4 = scf.if %shift_positive -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, i1) {
      %sh:4 = func.call @__ly_long_abs_rshift_raw(%lhs_meta, %lhs_digits, %shift) : (memref<2xi64>, memref<?xi32>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, i1)
      scf.yield %sh#0, %sh#1, %sh#2, %sh#3 : memref<2xi64>, memref<2xi64>, memref<?xi32>, i1
    } else {
      %neg_shift = arith.subi %zero, %shift : i64
      %sh:3 = func.call @__ly_long_abs_lshift_raw(%lhs_meta, %lhs_digits, %neg_shift) : (memref<2xi64>, memref<?xi32>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      %exact = arith.constant false
      scf.yield %sh#0, %sh#1, %sh#2, %exact : memref<2xi64>, memref<2xi64>, memref<?xi32>, i1
    }
    %qr:6 = func.call @__ly_long_divmod_abs(%x#1, %x#2, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%x#0) : (memref<2xi64>) -> ()
    %count_slot = arith.constant 1 : index
    %rem_count = memref.load %qr#4[%count_slot] : memref<2xi64>
    %rem_nonzero = arith.cmpi ne, %rem_count, %zero : i64
    func.call @LyLong_DecRef(%qr#3) : (memref<2xi64>) -> ()
    %inexact = arith.ori %x#3, %rem_nonzero : i1
    %x_bits = func.call @__ly_long_bit_length(%qr#1, %qr#2) : (memref<2xi64>, memref<?xi32>) -> i64
    // val = x as an integer; x has at most 57 bits, so it fits i64 exactly.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %thirty = arith.constant 30 : i64
    %q_count = memref.load %qr#1[%count_slot] : memref<2xi64>
    %q_count_index = arith.index_cast %q_count : i64 to index
    %val = scf.for %iv = %c0 to %q_count_index step %c1 iter_args(%acc = %zero) -> (i64) {
      %iv_next = arith.addi %iv, %c1 : index
      %rev = arith.subi %q_count_index, %iv_next : index
      %digit_i32 = memref.load %qr#2[%rev] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %scaled = arith.shli %acc, %thirty : i64
      %next = arith.ori %scaled, %digit : i64
      scf.yield %next : i64
    }
    func.call @LyLong_DecRef(%qr#0) : (memref<2xi64>) -> ()
    // extra_bits = max(x_bits, DBL_MIN_EXP - shift) - DBL_MANT_DIG (2 or 3).
    %c53 = arith.constant 53 : i64
    %min_minus_shift = arith.subi %min_exp, %shift : i64
    %xb_ge = arith.cmpi sge, %x_bits, %min_minus_shift : i64
    %effective = arith.select %xb_ge, %x_bits, %min_minus_shift : i1, i64
    %extra_bits = arith.subi %effective, %c53 : i64
    %extra_minus = arith.subi %extra_bits, %one : i64
    %mask = arith.shli %one, %extra_minus : i64
    // Round half-to-even: bump iff the guard bit is set and any of {sticky
    // bits below it, the inexact flag, the bit above it} is set.
    %inexact_i64 = arith.extui %inexact : i1 to i64
    %low = arith.ori %val, %inexact_i64 : i64
    %two_const = arith.constant 2 : i64
    %neg_one_const = arith.constant -1 : i64
    %three = arith.constant 3 : i64
    %three_mask = arith.muli %mask, %three : i64
    %near_mask = arith.subi %three_mask, %one : i64
    %guard = arith.andi %low, %mask : i64
    %guard_set = arith.cmpi ne, %guard, %zero : i64
    %near = arith.andi %low, %near_mask : i64
    %near_set = arith.cmpi ne, %near, %zero : i64
    %round_up = arith.andi %guard_set, %near_set : i1
    %bumped = arith.addi %low, %mask : i64
    %rounded = arith.select %round_up, %bumped, %low : i1, i64
    %clear_full = arith.muli %mask, %two_const : i64
    %clear_mask = arith.subi %clear_full, %one : i64
    %keep_mask = arith.xori %clear_mask, %neg_one_const : i64
    %final_val = arith.andi %rounded, %keep_mask : i64
    %dx = arith.uitofp %final_val : i64 to f64
    // Overflow check before scaling (CPython checks against ldexp(1, x_bits)).
    %shift_plus_bits = arith.addi %shift, %x_bits : i64
    %at_limit = arith.cmpi eq, %shift_plus_bits, %max_exp : i64
    %past_limit = arith.cmpi sgt, %shift_plus_bits, %max_exp : i64
    %pow_xbits = func.call @__ly_long_pow2_f64(%x_bits) : (i64) -> f64
    %dx_carries = arith.cmpf oeq, %dx, %pow_xbits : f64
    %limit_carry = arith.andi %at_limit, %dx_carries : i1
    %overflows = arith.ori %past_limit, %limit_carry : i1
    cf.cond_br %overflows, ^overflow, ^scale

  ^scale:
    // dx * 2^shift; shift can reach -1076, below the smallest normal power,
    // so split the scaling. Both steps are exact: the first keeps a normal
    // result, and the rounded dx * 2^shift is representable by construction.
    %deep = arith.constant -1021 : i64
    %shift_small = arith.cmpi slt, %shift, %deep : i64
    %result = scf.if %shift_small -> (f64) {
      %pre = arith.constant -900 : i64
      %pre_pow = func.call @__ly_long_pow2_f64(%pre) : (i64) -> f64
      %partial = arith.mulf %dx, %pre_pow : f64
      %rest = arith.subi %shift, %pre : i64
      %rest_pow = func.call @__ly_long_pow2_f64(%rest) : (i64) -> f64
      %scaled = arith.mulf %partial, %rest_pow : f64
      scf.yield %scaled : f64
    } else {
      %pow = func.call @__ly_long_pow2_f64(%shift) : (i64) -> f64
      %scaled = arith.mulf %dx, %pow : f64
      scf.yield %scaled : f64
    }
    %negated = arith.negf %result : f64
    %signed = arith.select %negate, %negated, %result : i1, f64
    %h, %p = func.call @LyFloat_FromF64(%signed) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %h, %p : memref<2xi64>, memref<1xf64>
  }

  func.func @LyLong_FloorDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__floordiv__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
    cf.cond_br %rhs_zero, ^zero_divisor, ^nonzero

  ^zero_divisor:
    func.call @__ly_long_raise_floor_div_zero() : () -> ()
    %zh, %zm, %zd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %zh, %zm, %zd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^nonzero:
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    // divsi INT64_MIN / -1 is poison; that single case promotes to digits.
    %min_i64 = arith.constant -9223372036854775808 : i64
    %neg_one = arith.constant -1 : i64
    %lhs_is_min = arith.cmpi eq, %lhs, %min_i64 : i64
    %rhs_is_neg_one = arith.cmpi eq, %rhs, %neg_one : i64
    %overflows = arith.andi %lhs_is_min, %rhs_is_neg_one : i1
    cf.cond_br %overflows, ^digits, ^small_fit

  ^small_fit:
    %one = arith.constant 1 : i64
    %trunc_q = arith.divsi %lhs, %rhs : i64
    %trunc_r = arith.remsi %lhs, %rhs : i64
    %has_remainder = arith.cmpi ne, %trunc_r, %zero : i64
    %lhs_negative = arith.cmpi slt, %lhs, %zero : i64
    %rhs_negative = arith.cmpi slt, %rhs, %zero : i64
    %different_sign = arith.cmpi ne, %lhs_negative, %rhs_negative : i1
    %adjust = arith.andi %has_remainder, %different_sign : i1
    %decremented = arith.subi %trunc_q, %one : i64
    %floor_q = arith.select %adjust, %decremented, %trunc_q : i1, i64
    %h, %m, %d = func.call @LyLong_FromI64(%floor_q) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %divmod:6 = func.call @__ly_long_floor_divmod(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%divmod#3) : (memref<2xi64>) -> ()
    func.return %divmod#0, %divmod#1, %divmod#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Mod(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mod__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
    cf.cond_br %rhs_zero, ^zero_divisor, ^nonzero

  ^zero_divisor:
    func.call @__ly_long_raise_mod_zero() : () -> ()
    %zh, %zm, %zd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %zh, %zm, %zd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^nonzero:
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    // remsi INT64_MIN % -1 is poison alongside its divsi; promote that case.
    %min_i64 = arith.constant -9223372036854775808 : i64
    %neg_one = arith.constant -1 : i64
    %lhs_is_min = arith.cmpi eq, %lhs, %min_i64 : i64
    %rhs_is_neg_one = arith.cmpi eq, %rhs, %neg_one : i64
    %overflows = arith.andi %lhs_is_min, %rhs_is_neg_one : i1
    cf.cond_br %overflows, ^digits, ^small_fit

  ^small_fit:
    %trunc_r = arith.remsi %lhs, %rhs : i64
    %has_remainder = arith.cmpi ne, %trunc_r, %zero : i64
    %remainder_negative = arith.cmpi slt, %trunc_r, %zero : i64
    %rhs_negative = arith.cmpi slt, %rhs, %zero : i64
    %different_sign = arith.cmpi ne, %remainder_negative, %rhs_negative : i1
    %adjust = arith.andi %has_remainder, %different_sign : i1
    %adjusted = arith.addi %trunc_r, %rhs : i64
    %mod = arith.select %adjust, %adjusted, %trunc_r : i1, i64
    %h, %m, %d = func.call @LyLong_FromI64(%mod) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %divmod:6 = func.call @__ly_long_floor_divmod(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%divmod#0) : (memref<2xi64>) -> ()
    func.return %divmod#3, %divmod#4, %divmod#5 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Infinite two's-complement bitwise op over digit forms (kind: 0 = and,
  // 1 = or, 2 = xor). Negative operands are materialized as two's-complement
  // limb streams over max(n, m) + 1 limbs (the extra limb is pure sign
  // extension, so the result's sign is just the op on the extension limbs),
  // then a negative result is converted back to sign-magnitude. Scratch
  // buffers are plain allocations, not objects, so no ownership is threaded
  // through the loops.
  func.func private @__ly_long_bitop_general(%kind: i64, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %two = arith.constant 2 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %lhs_ge = arith.cmpi sge, %lhs_count, %rhs_count : i64
    %max_count = arith.select %lhs_ge, %lhs_count, %rhs_count : i1, i64
    %limbs = arith.addi %max_count, %one : i64
    %limbs_index = arith.index_cast %limbs : i64 to index
    %scratch = memref.alloc(%limbs_index) : memref<?xi32>
    %lhs_negative = arith.cmpi slt, %lhs_sign, %zero : i64
    %rhs_negative = arith.cmpi slt, %rhs_sign, %zero : i64
    %lhs_carry_init = arith.select %lhs_negative, %one, %zero : i1, i64
    %rhs_carry_init = arith.select %rhs_negative, %one, %zero : i1, i64
    %is_and = arith.cmpi eq, %kind, %zero : i64
    %is_or = arith.cmpi eq, %kind, %one : i64
    %ignored:2 = scf.for %iv = %c0 to %limbs_index step %c1 iter_args(%lhs_carry = %lhs_carry_init, %rhs_carry = %rhs_carry_init) -> (i64, i64) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %has_lhs = arith.cmpi slt, %iv_i64, %lhs_count : i64
      %lhs_digit = scf.if %has_lhs -> (i64) {
        %digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      %has_rhs = arith.cmpi slt, %iv_i64, %rhs_count : i64
      %rhs_digit = scf.if %has_rhs -> (i64) {
        %digit_i32 = memref.load %rhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      // Two's complement limb: invert (mask ^ d) and add the incoming carry.
      %lhs_inverted = arith.xori %lhs_digit, %mask : i64
      %lhs_tc_raw = arith.addi %lhs_inverted, %lhs_carry : i64
      %lhs_tc = arith.andi %lhs_tc_raw, %mask : i64
      %lhs_carry_next_raw = arith.shrui %lhs_tc_raw, %thirty : i64
      %lhs_a = arith.select %lhs_negative, %lhs_tc, %lhs_digit : i1, i64
      %lhs_carry_next = arith.select %lhs_negative, %lhs_carry_next_raw, %zero : i1, i64
      %rhs_inverted = arith.xori %rhs_digit, %mask : i64
      %rhs_tc_raw = arith.addi %rhs_inverted, %rhs_carry : i64
      %rhs_tc = arith.andi %rhs_tc_raw, %mask : i64
      %rhs_carry_next_raw = arith.shrui %rhs_tc_raw, %thirty : i64
      %rhs_a = arith.select %rhs_negative, %rhs_tc, %rhs_digit : i1, i64
      %rhs_carry_next = arith.select %rhs_negative, %rhs_carry_next_raw, %zero : i1, i64
      %and_limb = arith.andi %lhs_a, %rhs_a : i64
      %or_limb = arith.ori %lhs_a, %rhs_a : i64
      %xor_limb = arith.xori %lhs_a, %rhs_a : i64
      %or_or_xor = arith.select %is_or, %or_limb, %xor_limb : i1, i64
      %result_limb = arith.select %is_and, %and_limb, %or_or_xor : i1, i64
      %result_i32 = arith.trunci %result_limb : i64 to i32
      memref.store %result_i32, %scratch[%iv] : memref<?xi32>
      scf.yield %lhs_carry_next, %rhs_carry_next : i64, i64
    }
    // The top limb is pure sign extension (0 or mask); mask means negative.
    %top_index = arith.subi %limbs_index, %c1 : index
    %top_i32 = memref.load %scratch[%top_index] : memref<?xi32>
    %top = arith.extui %top_i32 : i32 to i64
    %result_negative = arith.cmpi eq, %top, %mask : i64
    %result_sign = arith.select %result_negative, %neg_one, %one : i1, i64
    %h, %m, %d = func.call @__ly_long_alloc_raw(%result_sign, %limbs) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %back_carry_init = arith.select %result_negative, %one, %zero : i1, i64
    %ignored2 = scf.for %iv = %c0 to %limbs_index step %c1 iter_args(%carry = %back_carry_init) -> (i64) {
      %limb_i32 = memref.load %scratch[%iv] : memref<?xi32>
      %limb = arith.extui %limb_i32 : i32 to i64
      %inverted = arith.xori %limb, %mask : i64
      %tc_raw = arith.addi %inverted, %carry : i64
      %tc = arith.andi %tc_raw, %mask : i64
      %carry_next_raw = arith.shrui %tc_raw, %thirty : i64
      %magnitude = arith.select %result_negative, %tc, %limb : i1, i64
      %carry_next = arith.select %result_negative, %carry_next_raw, %zero : i1, i64
      %magnitude_i32 = arith.trunci %magnitude : i64 to i32
      memref.store %magnitude_i32, %d[%iv] : memref<?xi32>
      scf.yield %carry_next : i64
    }
    memref.dealloc %scratch : memref<?xi32>
    func.call @__ly_long_normalize(%m, %d, %limbs) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitAnd(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__and__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.andi %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %kind = arith.constant 0 : i64
    %gh, %gm, %gd = func.call @__ly_long_bitop_general(%kind, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %gh, %gm, %gd : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitOr(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__or__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.ori %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %kind = arith.constant 1 : i64
    %gh, %gm, %gd = func.call @__ly_long_bitop_general(%kind, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %gh, %gm, %gd : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitXor(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__xor__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.xori %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %kind = arith.constant 2 : i64
    %gh, %gm, %gd = func.call @__ly_long_bitop_general(%kind, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %gh, %gm, %gd : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_LShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__lshift__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %negative_count = arith.cmpi slt, %rhs_sign, %zero : i64
    cf.cond_br %negative_count, ^negative, ^check_width

  ^negative:
    func.call @__ly_long_raise_negative_shift() : () -> ()
    %gh, %gm, %gd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %gh, %gm, %gd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_width:
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %count_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %count_fits, ^shift, ^huge_count

  ^huge_count:
    // 0 << anything is 0; any other base would exceed memory.
    %lhs_is_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    cf.cond_br %lhs_is_zero, ^zero_base, ^too_large

  ^zero_base:
    %zh, %zm, %zd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %zh, %zm, %zd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^too_large:
    func.call @__ly_long_raise_shift_too_large() : () -> ()
    %th, %tm, %td = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %th, %tm, %td : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^shift:
    %n = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %sixty_two = arith.constant 62 : i64
    %small_count = arith.cmpi sle, %n, %sixty_two : i64
    %try_small = arith.andi %lhs_fits, %small_count : i1
    cf.cond_br %try_small, ^small, ^digits

  ^small:
    %value = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %shifted = arith.shli %value, %n : i64
    %round_trip = arith.shrsi %shifted, %n : i64
    %exact = arith.cmpi eq, %round_trip, %value : i64
    cf.cond_br %exact, ^small_fit, ^digits

  ^small_fit:
    %sh, %sm, %sd = func.call @LyLong_FromI64(%shifted) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %lhs_zero_mag = arith.cmpi eq, %lhs_sign, %zero : i64
    cf.cond_br %lhs_zero_mag, ^zero_base, ^digits_shift

  ^digits_shift:
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %dig_shift = arith.divui %n, %thirty : i64
    %bit_shift = arith.remui %n, %thirty : i64
    %grow = arith.addi %lhs_count, %dig_shift : i64
    %capacity = arith.addi %grow, %one : i64
    %h, %m, %d = func.call @__ly_long_alloc_raw(%lhs_sign, %capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_index = arith.index_cast %lhs_count : i64 to index
    %dig_shift_index = arith.index_cast %dig_shift : i64 to index
    %down = arith.subi %thirty, %bit_shift : i64
    %carry_out = scf.for %iv = %c0 to %lhs_index step %c1 iter_args(%carry = %zero) -> (i64) {
      %digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %up = arith.shli %digit, %bit_shift : i64
      %with_carry = arith.ori %up, %carry : i64
      %out = arith.andi %with_carry, %mask : i64
      %out_i32 = arith.trunci %out : i64 to i32
      %slot = arith.addi %iv, %dig_shift_index : index
      memref.store %out_i32, %d[%slot] : memref<?xi32>
      // down == 30 when bit_shift == 0; a 30-bit digit shifted right by 30
      // is 0, so the carry degenerates correctly.
      %next_carry = arith.shrui %digit, %down : i64
      scf.yield %next_carry : i64
    }
    %tail = arith.addi %lhs_index, %dig_shift_index : index
    %carry_i32 = arith.trunci %carry_out : i64 to i32
    memref.store %carry_i32, %d[%tail] : memref<?xi32>
    func.call @__ly_long_normalize(%m, %d, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_RShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__rshift__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %negative_count = arith.cmpi slt, %rhs_sign, %zero : i64
    cf.cond_br %negative_count, ^negative, ^check_width

  ^negative:
    func.call @__ly_long_raise_negative_shift() : () -> ()
    %gh, %gm, %gd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %gh, %gm, %gd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_width:
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %count_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %count_fits, ^shift, ^saturated

  ^saturated:
    // Shifting everything out: 0 for non-negative, -1 for negative (floor).
    %lhs_negative_s = arith.cmpi slt, %lhs_sign, %zero : i64
    %sat = arith.select %lhs_negative_s, %neg_one, %zero : i1, i64
    %vh, %vm, %vd = func.call @LyLong_FromI64(%sat) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %vh, %vm, %vd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^shift:
    %n = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %lhs_fits, ^small, ^digits

  ^small:
    %value = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    // shrsi is poison for counts >= 64; arithmetic shift saturates at 63.
    %sixty_three = arith.constant 63 : i64
    %clamp = arith.minsi %n, %sixty_three : i64
    %shifted = arith.shrsi %value, %clamp : i64
    %sh, %sm, %sd = func.call @LyLong_FromI64(%shifted) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %dig_shift = arith.divui %n, %thirty : i64
    %bit_shift = arith.remui %n, %thirty : i64
    %all_out = arith.cmpi sge, %dig_shift, %lhs_count : i64
    cf.cond_br %all_out, ^saturated, ^digits_shift

  ^digits_shift:
    %new_count = arith.subi %lhs_count, %dig_shift : i64
    // One spare limb: the floor adjustment for negative values increments
    // the magnitude in place.
    %capacity = arith.addi %new_count, %one : i64
    %h, %m, %d = func.call @__ly_long_alloc_raw(%lhs_sign, %capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %new_index = arith.index_cast %new_count : i64 to index
    %dig_shift_index = arith.index_cast %dig_shift : i64 to index
    %lhs_index = arith.index_cast %lhs_count : i64 to index
    %up = arith.subi %thirty, %bit_shift : i64
    scf.for %iv = %c0 to %new_index step %c1 {
      %src = arith.addi %iv, %dig_shift_index : index
      %digit_i32 = memref.load %lhs_digits[%src] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %low = arith.shrui %digit, %bit_shift : i64
      %src_next = arith.addi %src, %c1 : index
      %has_next = arith.cmpi slt, %src_next, %lhs_index : index
      %high = scf.if %has_next -> (i64) {
        %next_i32 = memref.load %lhs_digits[%src_next] : memref<?xi32>
        %next = arith.extui %next_i32 : i32 to i64
        %shifted_up = arith.shli %next, %up : i64
        scf.yield %shifted_up : i64
      } else {
        scf.yield %zero : i64
      }
      %combined = arith.ori %low, %high : i64
      %out = arith.andi %combined, %mask : i64
      %out_i32 = arith.trunci %out : i64 to i32
      memref.store %out_i32, %d[%iv] : memref<?xi32>
    }
    %lhs_negative = arith.cmpi slt, %lhs_sign, %zero : i64
    scf.if %lhs_negative {
      // floor(-x / 2^n) = -((x >> n) + 1) when any bit was shifted out.
      %dropped_digits = scf.for %iv = %c0 to %dig_shift_index step %c1 iter_args(%sticky = %zero) -> (i64) {
        %digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        %merged = arith.ori %sticky, %digit : i64
        scf.yield %merged : i64
      }
      %boundary_i32 = memref.load %lhs_digits[%dig_shift_index] : memref<?xi32>
      %boundary = arith.extui %boundary_i32 : i32 to i64
      %low_mask_full = arith.shli %one, %bit_shift : i64
      %low_mask = arith.subi %low_mask_full, %one : i64
      %boundary_dropped = arith.andi %boundary, %low_mask : i64
      %all_dropped = arith.ori %dropped_digits, %boundary_dropped : i64
      %has_dropped = arith.cmpi ne, %all_dropped, %zero : i64
      scf.if %has_dropped {
        %carry_final = scf.for %iv = %c0 to %new_index step %c1 iter_args(%carry = %one) -> (i64) {
          %digit_i32 = memref.load %d[%iv] : memref<?xi32>
          %digit = arith.extui %digit_i32 : i32 to i64
          %sum = arith.addi %digit, %carry : i64
          %out = arith.andi %sum, %mask : i64
          %out_i32 = arith.trunci %out : i64 to i32
          memref.store %out_i32, %d[%iv] : memref<?xi32>
          %next = arith.shrui %sum, %thirty : i64
          scf.yield %next : i64
        }
        %overflowed = arith.cmpi ne, %carry_final, %zero : i64
        scf.if %overflowed {
          %one_i32 = arith.constant 1 : i32
          memref.store %one_i32, %d[%new_index] : memref<?xi32>
        }
      }
    }
    func.call @__ly_long_normalize(%m, %d, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // round(int, ndigits): identity for ndigits >= 0; otherwise round to the
  // nearest multiple of 10^-ndigits with ties to even (CPython int.__round__,
  // exact at any width: 10^n via square-and-multiply, then one divmod).
  func.func @LyLong_Round(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>, %ndigits: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__round__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %nonneg = arith.cmpi sge, %ndigits, %zero : i64
    cf.cond_br %nonneg, ^identity, ^negative

  ^identity:
    %ih, %im, %id = func.call @__ly_long_copy(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %ih, %im, %id : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^negative:
    %places = arith.subi %zero, %ndigits : i64
    // -INT64_MIN wraps negative; such a scale dwarfs any representable int.
    %wrapped = arith.cmpi sle, %places, %zero : i64
    cf.cond_br %wrapped, ^zero_result, ^check_width

  ^zero_result:
    %zh, %zm, %zd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %zh, %zm, %zd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_width:
    // 10^places >= 2^(3*places) > 2*|x| makes the result 0 (a tie needs
    // 10^places == 2*|x| exactly, which the general path handles). Checking
    // places >= bit_length first keeps 3*places from overflowing and bounds
    // the 10^places allocation by the width of x itself.
    %bit_len = func.call @__ly_long_bit_length(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %input_zero = arith.cmpi eq, %bit_len, %zero : i64
    cf.cond_br %input_zero, ^zero_result, ^check_places

  ^check_places:
    %huge_places = arith.cmpi sge, %places, %bit_len : i64
    cf.cond_br %huge_places, ^zero_result, ^check_three

  ^check_three:
    %three = arith.constant 3 : i64
    %two = arith.constant 2 : i64
    %three_places = arith.muli %places, %three : i64
    %bit_len_plus = arith.addi %bit_len, %two : i64
    %dominates = arith.cmpi sge, %three_places, %bit_len_plus : i64
    cf.cond_br %dominates, ^zero_result, ^general

  ^general:
    %ten = arith.constant 10 : i64
    %ten_h, %ten_m, %ten_d = func.call @LyLong_FromI64(%ten) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %scale:3 = func.call @__ly_long_pow_rec(%ten_h, %ten_m, %ten_d, %places) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%ten_h) : (memref<2xi64>) -> ()
    %qr:6 = func.call @__ly_long_divmod_abs(%meta, %digits, %scale#1, %scale#2) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %two_r:3 = func.call @__ly_long_add_abs(%one, %qr#4, %qr#5, %qr#4, %qr#5) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%qr#3) : (memref<2xi64>) -> ()
    %cmp = func.call @__ly_long_abs_compare(%two_r#1, %two_r#2, %scale#1, %scale#2) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
    func.call @LyLong_DecRef(%two_r#0) : (memref<2xi64>) -> ()
    %above_half = arith.cmpi sgt, %cmp, %zero : i64
    %exactly_half = arith.cmpi eq, %cmp, %zero : i64
    %q_digit0_slot = arith.constant 0 : index
    %q_digit0_i32 = memref.load %qr#2[%q_digit0_slot] : memref<?xi32>
    %q_digit0 = arith.extui %q_digit0_i32 : i32 to i64
    %q_low_bit = arith.andi %q_digit0, %one : i64
    %q_odd = arith.cmpi ne, %q_low_bit, %zero : i64
    %tie_up = arith.andi %exactly_half, %q_odd : i1
    %round_up = arith.ori %above_half, %tie_up : i1
    cf.cond_br %round_up, ^bump, ^scale_back

  ^bump:
    %one_meta = memref.get_global @__ly_long_one_meta : memref<2xi64>
    %one_digits_static = memref.get_global @__ly_long_one_digits : memref<1xi32>
    %one_digits = memref.cast %one_digits_static : memref<1xi32> to memref<?xi32>
    %bumped:3 = func.call @__ly_long_add_abs(%one, %qr#1, %qr#2, %one_meta, %one_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%qr#0) : (memref<2xi64>) -> ()
    %br:3 = func.call @LyLong_Mul(%bumped#0, %bumped#1, %bumped#2, %scale#0, %scale#1, %scale#2) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%bumped#0) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%scale#0) : (memref<2xi64>) -> ()
    // Reapply the input's sign; the product of nonzero magnitudes is nonzero.
    %b_sign_slot = arith.constant 0 : index
    %b_input_sign = memref.load %meta[%b_sign_slot] : memref<2xi64>
    memref.store %b_input_sign, %br#1[%b_sign_slot] : memref<2xi64>
    func.return %br#0, %br#1, %br#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^scale_back:
    // A zero quotient short-circuits: the product would be the immortal zero
    // cache, whose meta must never be written.
    %q_count_slot = arith.constant 1 : index
    %q_count = memref.load %qr#1[%q_count_slot] : memref<2xi64>
    %q_zero = arith.cmpi eq, %q_count, %zero : i64
    cf.cond_br %q_zero, ^scale_back_zero, ^scale_back_mul

  ^scale_back_zero:
    func.call @LyLong_DecRef(%qr#0) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%scale#0) : (memref<2xi64>) -> ()
    %qz_h, %qz_m, %qz_d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %qz_h, %qz_m, %qz_d : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^scale_back_mul:
    %sr:3 = func.call @LyLong_Mul(%qr#0, %qr#1, %qr#2, %scale#0, %scale#1, %scale#2) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%qr#0) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%scale#0) : (memref<2xi64>) -> ()
    // The product of nonzero magnitudes is nonzero: reapply the input's sign.
    %s_sign_slot = arith.constant 0 : index
    %s_input_sign = memref.load %meta[%s_sign_slot] : memref<2xi64>
    memref.store %s_input_sign, %sr#1[%s_sign_slot] : memref<2xi64>
    func.return %sr#0, %sr#1, %sr#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Base-10 int(str) parse: optional surrounding ASCII whitespace, optional
  // sign, digits with single interior underscores. Arbitrary length via
  // in-place multiply-by-10-and-add over the digit limbs. Runtime-level
  // __int__ on str (not part of the typed manifest surface: CPython has no
  // str.__int__; only the emitter's int(x) rewrite targets it). Unicode
  // digits are not accepted yet (ASCII only until the UCD tables land).
  func.func @LyLong_FromStr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__int__", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %ten = arith.constant 10 : i64
    %mask = arith.constant 1073741823 : i64
    %thirty = arith.constant 30 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %len = memref.dim %bytes, %c0 : memref<?xi8>
    %len_i64 = arith.index_cast %len : index to i64
    // First and last non-whitespace positions (space, \t..\r).
    %space = arith.constant 32 : i8
    %tab = arith.constant 9 : i8
    %cr = arith.constant 13 : i8
    %first = scf.for %iv = %c0 to %len step %c1 iter_args(%found = %neg_one) -> (i64) {
      %c = memref.load %bytes[%iv] : memref<?xi8>
      %is_space = arith.cmpi eq, %c, %space : i8
      %ge_tab = arith.cmpi sge, %c, %tab : i8
      %le_cr = arith.cmpi sle, %c, %cr : i8
      %is_ctl = arith.andi %ge_tab, %le_cr : i1
      %is_ws = arith.ori %is_space, %is_ctl : i1
      %unset = arith.cmpi eq, %found, %neg_one : i64
      %iv_i64 = arith.index_cast %iv : index to i64
      // found records the first non-ws index once, then sticks.
      %candidate = arith.select %is_ws, %found, %iv_i64 : i1, i64
      %next = arith.select %unset, %candidate, %found : i1, i64
      scf.yield %next : i64
    }
    %last = scf.for %iv = %c0 to %len step %c1 iter_args(%found = %neg_one) -> (i64) {
      %iv_next = arith.addi %iv, %c1 : index
      %rev = arith.subi %len, %iv_next : index
      %c = memref.load %bytes[%rev] : memref<?xi8>
      %is_space = arith.cmpi eq, %c, %space : i8
      %ge_tab = arith.cmpi sge, %c, %tab : i8
      %le_cr = arith.cmpi sle, %c, %cr : i8
      %is_ctl = arith.andi %ge_tab, %le_cr : i1
      %is_ws = arith.ori %is_space, %is_ctl : i1
      %unset = arith.cmpi eq, %found, %neg_one : i64
      %rev_i64 = arith.index_cast %rev : index to i64
      %candidate = arith.select %is_ws, %found, %rev_i64 : i1, i64
      %next = arith.select %unset, %candidate, %found : i1, i64
      scf.yield %next : i64
    }
    %no_content = arith.cmpi eq, %first, %neg_one : i64
    cf.cond_br %no_content, ^invalid_early, ^signed

  ^invalid_early:
    func.call @__ly_long_raise_invalid_literal() : () -> ()
    %eh, %em, %ed = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %eh, %em, %ed : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^signed:
    %end = arith.addi %last, %one : i64
    %first_char_index = arith.index_cast %first : i64 to index
    %first_char = memref.load %bytes[%first_char_index] : memref<?xi8>
    %plus = arith.constant 43 : i8
    %minus = arith.constant 45 : i8
    %is_plus = arith.cmpi eq, %first_char, %plus : i8
    %is_minus = arith.cmpi eq, %first_char, %minus : i8
    %has_sign = arith.ori %is_plus, %is_minus : i1
    %after_sign = arith.addi %first, %one : i64
    %digits_start = arith.select %has_sign, %after_sign, %first : i1, i64
    %sign = arith.select %is_minus, %neg_one, %one : i1, i64
    %no_digits = arith.cmpi sge, %digits_start, %end : i64
    cf.cond_br %no_digits, ^invalid_early, ^parse

  ^parse:
    // Capacity: 4 bits per char comfortably over-approximates log2(10).
    %nchars = arith.subi %end, %digits_start : i64
    %four = arith.constant 4 : i64
    %bits = arith.muli %nchars, %four : i64
    %limbs = arith.divui %bits, %thirty : i64
    %two = arith.constant 2 : i64
    %capacity = arith.addi %limbs, %two : i64
    %h, %m, %d = func.call @__ly_long_alloc_raw(%sign, %capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %start_index = arith.index_cast %digits_start : i64 to index
    %end_index = arith.index_cast %end : i64 to index
    %ascii_zero = arith.constant 48 : i8
    %ascii_nine = arith.constant 57 : i8
    %underscore = arith.constant 95 : i8
    %parse:3 = scf.for %iv = %start_index to %end_index step %c1 iter_args(%count = %one, %invalid = %zero, %prev_us = %one) -> (i64, i64, i64) {
      // prev_us starts as 1 so a leading underscore is rejected.
      %c = memref.load %bytes[%iv] : memref<?xi8>
      %is_us = arith.cmpi eq, %c, %underscore : i8
      %ge_zero = arith.cmpi sge, %c, %ascii_zero : i8
      %le_nine = arith.cmpi sle, %c, %ascii_nine : i8
      %is_digit = arith.andi %ge_zero, %le_nine : i1
      %prev_was_us = arith.cmpi ne, %prev_us, %zero : i64
      %bad_us = arith.andi %is_us, %prev_was_us : i1
      %not_token = arith.ori %is_us, %is_digit : i1
      %true_bit = arith.constant true
      %bad_char = arith.xori %not_token, %true_bit : i1
      %new_invalid_flag = arith.ori %bad_us, %bad_char : i1
      %new_invalid_i64 = arith.extui %new_invalid_flag : i1 to i64
      %invalid_next = arith.ori %invalid, %new_invalid_i64 : i64
      %digit_val_i8 = arith.subi %c, %ascii_zero : i8
      %digit_val_raw = arith.extui %digit_val_i8 : i8 to i64
      %digit_val = arith.select %is_digit, %digit_val_raw, %zero : i1, i64
      %count_next = scf.if %is_digit -> (i64) {
        // x = x * 10 + digit over the active limbs; carry-in is the new digit.
        %count_index = arith.index_cast %count : i64 to index
        %carry_out = scf.for %i = %c0 to %count_index step %c1 iter_args(%carry = %digit_val) -> (i64) {
          %limb_i32 = memref.load %d[%i] : memref<?xi32>
          %limb = arith.extui %limb_i32 : i32 to i64
          %scaled = arith.muli %limb, %ten : i64
          %sum = arith.addi %scaled, %carry : i64
          %out = arith.andi %sum, %mask : i64
          %out_i32 = arith.trunci %out : i64 to i32
          memref.store %out_i32, %d[%i] : memref<?xi32>
          %next_carry = arith.shrui %sum, %thirty : i64
          scf.yield %next_carry : i64
        }
        %has_carry = arith.cmpi ne, %carry_out, %zero : i64
        %grown = scf.if %has_carry -> (i64) {
          %slot = arith.index_cast %count : i64 to index
          %carry_i32 = arith.trunci %carry_out : i64 to i32
          memref.store %carry_i32, %d[%slot] : memref<?xi32>
          %count_grown = arith.addi %count, %one : i64
          scf.yield %count_grown : i64
        } else {
          scf.yield %count : i64
        }
        scf.yield %grown : i64
      } else {
        scf.yield %count : i64
      }
      %prev_us_next = arith.extui %is_us : i1 to i64
      scf.yield %count_next, %invalid_next, %prev_us_next : i64, i64, i64
    }
    %trailing_us = arith.cmpi ne, %parse#2, %zero : i64
    %char_invalid = arith.cmpi ne, %parse#1, %zero : i64
    %any_invalid = arith.ori %trailing_us, %char_invalid : i1
    cf.cond_br %any_invalid, ^invalid_parsed, ^done

  ^invalid_parsed:
    func.call @LyLong_DecRef(%h) : (memref<2xi64>) -> ()
    func.call @__ly_long_raise_invalid_literal() : () -> ()
    %ph, %pm, %pd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %ph, %pm, %pd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^done:
    func.call @__ly_long_normalize(%m, %d, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Truncating float -> int conversion (runtime-level __int__ on float).
  // |x| < 2^63 goes through fptosi; larger finite magnitudes are exact:
  // every such double is the integer mantissa * 2^exponent, so the digits
  // are the 53 mantissa bits placed at bit offset `exponent` (CPython
  // PyLong_FromDouble). NaN and infinity raise with CPython's messages.
  func.func @LyFloat_Int(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__int__", ly.runtime.result_contract = "builtins.int"} {
    %c0 = arith.constant 0 : index
    %value = memref.load %payload[%c0] : memref<1xf64>
    %is_nan = arith.cmpf uno, %value, %value : f64
    cf.cond_br %is_nan, ^nan, ^check_inf

  ^nan:
    func.call @__ly_long_raise_float_nan() : () -> ()
    %zero_nan = arith.constant 0 : i64
    %nh, %nm, %nd = func.call @LyLong_FromI64(%zero_nan) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %nh, %nm, %nd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_inf:
    %bits = arith.bitcast %value : f64 to i64
    %exp_shift = arith.constant 52 : i64
    %exp_mask = arith.constant 2047 : i64
    %exp_raw_shifted = arith.shrui %bits, %exp_shift : i64
    %exp_raw = arith.andi %exp_raw_shifted, %exp_mask : i64
    %is_inf = arith.cmpi eq, %exp_raw, %exp_mask : i64
    cf.cond_br %is_inf, ^infinity, ^check_range

  ^infinity:
    func.call @__ly_long_raise_float_infinity() : () -> ()
    %zero_inf = arith.constant 0 : i64
    %ih, %im, %id = func.call @LyLong_FromI64(%zero_inf) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %ih, %im, %id : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_range:
    %lower = arith.constant -9.2233720368547758E+18 : f64
    %upper = arith.constant 9.2233720368547758E+18 : f64
    %ge_lower = arith.cmpf oge, %value, %lower : f64
    %lt_upper = arith.cmpf olt, %value, %upper : f64
    %in_range = arith.andi %ge_lower, %lt_upper : i1
    cf.cond_br %in_range, ^convert, ^big

  ^big:
    // Finite with |x| >= 2^63: normal, exponent field >= 1086. The value is
    // (mantissa | 2^52) * 2^(exp_raw - 1075) with a positive binary exponent,
    // so each 30-bit limb is a window of the shifted 53-bit mantissa.
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %thirty = arith.constant 30 : i64
    %mask30 = arith.constant 1073741823 : i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %implicit_bit = arith.constant 4503599627370496 : i64
    %mant_low = arith.andi %bits, %mant_mask : i64
    %mantissa = arith.ori %mant_low, %implicit_bit : i64
    %exp_bias = arith.constant 1075 : i64
    %e = arith.subi %exp_raw, %exp_bias : i64
    %sign_negative = arith.cmpi slt, %bits, %zero : i64
    %sign = arith.select %sign_negative, %neg_one, %one : i1, i64
    %c53 = arith.constant 53 : i64
    %c29 = arith.constant 29 : i64
    %total_bits = arith.addi %e, %c53 : i64
    %rounded_up = arith.addi %total_bits, %c29 : i64
    %ndigits = arith.divui %rounded_up, %thirty : i64
    %h, %m, %d = func.call @__ly_long_alloc_raw(%sign, %ndigits) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0i = arith.constant 0 : index
    %c1i = arith.constant 1 : index
    %ndigits_index = arith.index_cast %ndigits : i64 to index
    scf.for %j = %c0i to %ndigits_index step %c1i {
      %j_i64 = arith.index_cast %j : index to i64
      %j30 = arith.muli %j_i64, %thirty : i64
      %s = arith.subi %j30, %e : i64
      // s >= 0: window is mantissa >> s (zero once s >= 53).
      // s in (-30, 0): window is (mantissa << -s) & mask.
      // s <= -30: below the mantissa, zero.
      %s_nonneg = arith.cmpi sge, %s, %zero : i64
      %s_lt53 = arith.cmpi slt, %s, %c53 : i64
      %right_ok = arith.andi %s_nonneg, %s_lt53 : i1
      %s_clamped = arith.select %right_ok, %s, %zero : i1, i64
      %right_shifted = arith.shrui %mantissa, %s_clamped : i64
      %right_part = arith.select %right_ok, %right_shifted, %zero : i1, i64
      %ns = arith.subi %zero, %s : i64
      %s_negative = arith.cmpi slt, %s, %zero : i64
      %ns_lt30 = arith.cmpi slt, %ns, %thirty : i64
      %left_ok = arith.andi %s_negative, %ns_lt30 : i1
      %ns_clamped = arith.select %left_ok, %ns, %zero : i1, i64
      %left_shifted = arith.shli %mantissa, %ns_clamped : i64
      %left_part = arith.select %left_ok, %left_shifted, %zero : i1, i64
      %part = arith.select %s_nonneg, %right_part, %left_part : i1, i64
      %digit_i64 = arith.andi %part, %mask30 : i64
      %digit = arith.trunci %digit_i64 : i64 to i32
      memref.store %digit, %d[%j] : memref<?xi32>
    }
    func.call @__ly_long_normalize(%m, %d, %ndigits) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^convert:
    %truncated = arith.fptosi %value : f64 to i64
    %h2, %m2, %d2 = func.call @LyLong_FromI64(%truncated) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h2, %m2, %d2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Square-and-multiply as recursion (depth <= 63): each frame creates,
  // consumes, and returns owned values linearly, which the affine-ownership
  // verifier can follow; a loop would have to thread owned iter_args, which
  // it cannot.
  func.func private @__ly_long_pow_rec(%base_header: memref<2xi64>, %base_meta: memref<2xi64>, %base_digits: memref<?xi32>, %exp: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %is_zero = arith.cmpi eq, %exp, %zero : i64
    cf.cond_br %is_zero, ^base_case, ^recurse

  ^base_case:
    %oh, %om, %od = func.call @LyLong_FromI64(%one) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %oh, %om, %od : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^recurse:
    %half_exp = arith.shrui %exp, %one : i64
    %half:3 = func.call @__ly_long_pow_rec(%base_header, %base_meta, %base_digits, %half_exp) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %square:3 = func.call @LyLong_Mul(%half#0, %half#1, %half#2, %half#0, %half#1, %half#2) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%half#0) : (memref<2xi64>) -> ()
    %low_bit = arith.andi %exp, %one : i64
    %is_odd = arith.cmpi ne, %low_bit, %zero : i64
    cf.cond_br %is_odd, ^odd, ^even

  ^odd:
    %with_base:3 = func.call @LyLong_Mul(%square#0, %square#1, %square#2, %base_header, %base_meta, %base_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%square#0) : (memref<2xi64>) -> ()
    func.return %with_base#0, %with_base#1, %with_base#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^even:
    func.return %square#0, %square#1, %square#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Pow(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__pow__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %negative_exp = arith.cmpi slt, %rhs_sign, %zero : i64
    cf.cond_br %negative_exp, ^negative, ^check_width

  ^negative:
    // CPython returns a float here; the static result type is int, so reject
    // loudly instead of changing the result type (deviation, see message).
    func.call @__ly_long_raise_pow_negative() : () -> ()
    %gh, %gm, %gd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %gh, %gm, %gd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_width:
    %exp_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %exp_fits, ^pow, ^huge_exp

  ^huge_exp:
    // Bases with |base| > 1 would not fit in memory; 0/1/-1 with a 2^63+
    // exponent are legal but degenerate. -1 keeps the parity of the lowest
    // exponent limb.
    %count_slot = arith.constant 1 : index
    %lhs_sign_h = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %lhs_count_h = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %is_zero_base = arith.cmpi eq, %lhs_count_h, %zero : i64
    %is_one_limb = arith.cmpi eq, %lhs_count_h, %one : i64
    %c0g = arith.constant 0 : index
    %digit0_g = memref.load %lhs_digits[%c0g] : memref<?xi32>
    %digit0_g_i64 = arith.extui %digit0_g : i32 to i64
    %magnitude_le_one = arith.cmpi ule, %digit0_g_i64, %one : i64
    %is_unit_base = arith.andi %is_one_limb, %magnitude_le_one : i1
    %is_small_base = arith.ori %is_zero_base, %is_unit_base : i1
    cf.cond_br %is_small_base, ^huge_exp_small_base, ^huge_exp_reject

  ^huge_exp_reject:
    func.call @__ly_long_raise_too_large() : () -> ()
    %rh, %rm, %rd = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^huge_exp_small_base:
    %c0h = arith.constant 0 : index
    %digit0_h = memref.load %rhs_digits[%c0h] : memref<?xi32>
    %digit0_h_i64 = arith.extui %digit0_h : i32 to i64
    %exp_odd_h = arith.andi %digit0_h_i64, %one : i64
    %exp_is_odd = arith.cmpi ne, %exp_odd_h, %zero : i64
    %neg_base = arith.cmpi slt, %lhs_sign_h, %zero : i64
    %flip = arith.andi %neg_base, %exp_is_odd : i1
    %neg_one_h = arith.constant -1 : i64
    %abs_result = arith.cmpi eq, %lhs_sign_h, %zero : i64
    %zero_or_sign = arith.select %flip, %neg_one_h, %one : i1, i64
    %small_value = arith.select %abs_result, %zero, %zero_or_sign : i1, i64
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^pow:
    %exp = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %h, %m, %d = func.call @__ly_long_pow_rec(%lhs_header, %lhs_meta, %lhs_digits, %exp) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // CPython-compatible int hash: reduction of the magnitude modulo the
  // Mersenne prime 2^61 - 1 (digit-wise 30-bit rotation), sign applied,
  // -1 remapped to -2. Keeping the modulus scheme means float can later
  // satisfy hash(1) == hash(1.0). Int hashing is not randomized in CPython
  // either (SipHash applies to str/bytes).
  func.func @LyLong_Hash(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__hash__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %thirty = arith.constant 30 : i64
    %thirty_one = arith.constant 31 : i64
    %modulus = arith.constant 2305843009213693951 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = arith.index_cast %count : i64 to index
    %reduced = scf.for %iv = %c0 to %count_index step %c1 iter_args(%x = %zero) -> (i64) {
      %iv_next = arith.addi %iv, %c1 : index
      %rev = arith.subi %count_index, %iv_next : index
      %digit_i32 = memref.load %digits[%rev] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      // 61-bit rotate left by 30: keep low bits shifted up (mod trick), pull
      // the high 31 bits down.
      %up = arith.shli %x, %thirty : i64
      %up_masked = arith.andi %up, %modulus : i64
      %down = arith.shrui %x, %thirty_one : i64
      %rotated = arith.ori %up_masked, %down : i64
      %with_digit = arith.addi %rotated, %digit : i64
      %needs_reduce = arith.cmpi uge, %with_digit, %modulus : i64
      %reduced_once = arith.subi %with_digit, %modulus : i64
      %next = arith.select %needs_reduce, %reduced_once, %with_digit : i1, i64
      scf.yield %next : i64
    }
    %negated = arith.subi %zero, %reduced : i64
    %is_negative = arith.cmpi slt, %sign, %zero : i64
    %signed = arith.select %is_negative, %negated, %reduced : i1, i64
    %neg_one = arith.constant -1 : i64
    %neg_two = arith.constant -2 : i64
    %is_neg_one = arith.cmpi eq, %signed, %neg_one : i64
    %result = arith.select %is_neg_one, %neg_two, %signed : i1, i64
    func.return %result : i64
  }

  func.func @LyLong_Compare(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__richcompare__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_two_limb = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_two_limb = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_two_limb = arith.andi %lhs_two_limb, %rhs_two_limb : i1
    cf.cond_br %both_two_limb, ^small, ^maybe_i64

  ^maybe_i64:
    %lhs_i64 = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_i64 = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_i64 = arith.andi %lhs_i64, %rhs_i64 : i1
    cf.cond_br %both_i64, ^small, ^digits

  ^small:
    // Signed i64 is sufficient for this comparison.
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_zero = arith.constant 0 : i64
    %small_one = arith.constant 1 : i64
    %small_neg_one = arith.constant -1 : i64
    %small_gt = arith.cmpi sgt, %small_a, %small_b : i64
    %small_lt = arith.cmpi slt, %small_a, %small_b : i64
    %small_pos = arith.select %small_gt, %small_one, %small_zero : i1, i64
    %small_cmp = arith.select %small_lt, %small_neg_one, %small_pos : i1, i64
    func.return %small_cmp : i64

  ^digits:
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %sign_slot = arith.constant 0 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %sign_gt = arith.cmpi sgt, %lhs_sign, %rhs_sign : i64
    %sign_lt = arith.cmpi slt, %lhs_sign, %rhs_sign : i64
    %sign_cmp = arith.select %sign_gt, %one, %zero : i1, i64
    %sign_cmp2 = arith.select %sign_lt, %neg_one, %sign_cmp : i1, i64
    %same_sign = arith.cmpi eq, %sign_cmp2, %zero : i64
    %result = scf.if %same_sign -> (i64) {
      %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
      %same_sign_cmp = scf.if %lhs_zero -> (i64) {
        scf.yield %zero : i64
      } else {
        %abs_cmp = func.call @__ly_long_abs_compare(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
        %is_negative = arith.cmpi slt, %lhs_sign, %zero : i64
        %neg_abs = arith.muli %abs_cmp, %neg_one : i64
        %signed_cmp = arith.select %is_negative, %neg_abs, %abs_cmp : i1, i64
        scf.yield %signed_cmp : i64
      }
      scf.yield %same_sign_cmp : i64
    } else {
      scf.yield %sign_cmp2 : i64
    }
    func.return %result : i64
  }

  func.func @LyLong_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__eq__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi eq, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_NeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__ne__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi ne, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_LtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__lt__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi slt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_LeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__le__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sle, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_GtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__gt__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sgt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_GeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__ge__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sge, %cmp, %zero : i64
    func.return %result : i1
  }


  // str(int) == repr(int) in CPython; delegate.
  func.func @LyLong_Str(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %str_header, %str_bytes = func.call @LyLong_Repr(%header, %meta_raw, %digits_raw) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi8>)
    func.return %str_header, %str_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLong_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__repr__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %ten = arith.constant 10 : i64
    %base = arith.constant 1073741824 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_minus = arith.constant 45 : i8
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %is_zero = arith.cmpi eq, %count, %zero : i64
    %result:2 = scf.if %is_zero -> (memref<2xi64>, memref<?xi8>) {
      %h, %b = func.call @LyUnicode_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    } else {
      %count_index = arith.index_cast %count : i64 to index
      %tmp = memref.alloc(%count_index) : memref<?xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %iv = %c0 to %count_index step %c1 {
        %digit = memref.load %digits[%iv] : memref<?xi32>
        memref.store %digit, %tmp[%iv] : memref<?xi32>
      }
      %negative = arith.cmpi slt, %sign, %zero : i64
      %sign_extra = arith.select %negative, %one, %zero : i1, i64
      %decimal_capacity_base = arith.muli %count, %ten : i64
      %decimal_capacity = arith.addi %decimal_capacity_base, %sign_extra : i64
      %decimal_capacity_index = arith.index_cast %decimal_capacity : i64 to index
      %buffer = memref.alloc(%decimal_capacity_index) : memref<?xi8>
      %conversion:2 = scf.for %step_iv = %c0 to %decimal_capacity_index step %c1 iter_args(%active_count = %count, %pos = %decimal_capacity_index) -> (i64, index) {
        %active = arith.cmpi ne, %active_count, %zero : i64
        %next:2 = scf.if %active -> (i64, index) {
          %active_index = arith.index_cast %active_count : i64 to index
          %division:2 = scf.for %scan = %c0 to %active_index step %c1 iter_args(%rem_iter = %zero, %last_iter = %zero) -> (i64, i64) {
            %scan_next = arith.addi %scan, %c1 : index
            %rev = arith.subi %active_index, %scan_next : index
            %digit_i32 = memref.load %tmp[%rev] : memref<?xi32>
            %digit = arith.extui %digit_i32 : i32 to i64
            %scaled = arith.muli %rem_iter, %base : i64
            %accum = arith.addi %scaled, %digit : i64
            %quotient = arith.divui %accum, %ten : i64
            %remainder = arith.remui %accum, %ten : i64
            %quotient_i32 = arith.trunci %quotient : i64 to i32
            memref.store %quotient_i32, %tmp[%rev] : memref<?xi32>
            %nonzero = arith.cmpi ne, %quotient, %zero : i64
            %no_last = arith.cmpi eq, %last_iter, %zero : i64
            %take_last = arith.andi %nonzero, %no_last : i1
            %rev_next = arith.addi %rev, %c1 : index
            %rev_count = arith.index_cast %rev_next : index to i64
            %last = arith.select %take_last, %rev_count, %last_iter : i1, i64
            scf.yield %remainder, %last : i64, i64
          }
          %ascii_digit_i64 = arith.addi %division#0, %ascii_zero : i64
          %ascii_digit = arith.trunci %ascii_digit_i64 : i64 to i8
          %next_pos = arith.subi %pos, %c1 : index
          memref.store %ascii_digit, %buffer[%next_pos] : memref<?xi8>
          scf.yield %division#1, %next_pos : i64, index
        } else {
          scf.yield %active_count, %pos : i64, index
        }
        scf.yield %next#0, %next#1 : i64, index
      }
      %start = scf.if %negative -> (index) {
        %minus_pos = arith.subi %conversion#1, %c1 : index
        memref.store %ascii_minus, %buffer[%minus_pos] : memref<?xi8>
        scf.yield %minus_pos : index
      } else {
        scf.yield %conversion#1 : index
      }
      %length_index = arith.subi %decimal_capacity_index, %start : index
      %length = arith.index_cast %length_index : index to i64
      %h, %b = func.call @LyUnicode_FromBytes(%buffer, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %buffer : memref<?xi8>
      memref.dealloc %tmp : memref<?xi32>
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // ===== impls: unicode =====
  // Raw fd write boundary (built by RuntimeSupportBuilder): extracting the
  // payload pointer from the descriptor is the irreducibly-llvm part, so it
  // is the ONLY piece of the print path that stays out of this manifest.
  func.func private @LyHost_WriteBytes(i32, memref<?xi8>, i64)


  // Retain a borrowed str and hand the same object back as an owned result.
  // Evidence-selected container elements are retained through this primitive so
  // they survive their container's release (checked retain premise).
  func.func private @LyUnicode_Shape() -> (memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.str", ly.runtime.shape}

  memref.global "private" constant @__ly_unicode_msg_string_index_out_of_range : memref<25xi8> = dense<[115, 116, 114, 105, 110, 103, 32, 105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>

  func.func private @__ly_unicode_raise_index_error() {
    %class_id = arith.constant 55 : i64
    %length = arith.constant 25 : i64
    %start = arith.constant 0 : index
    %message_static = memref.get_global @__ly_unicode_msg_string_index_out_of_range : memref<25xi8>
    %message = memref.cast %message_static : memref<25xi8> to memref<?xi8>
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // "invalid utf-8 sequence"
  memref.global "private" constant @__ly_unicode_msg_invalid_utf8 : memref<22xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 117, 116, 102, 45, 56, 32, 115, 101, 113, 117, 101, 110, 99, 101]>

  func.func private @__ly_unicode_raise_decode_error() {
    %class_id = arith.constant 122 : i64
    %length = arith.constant 22 : i64
    %start = arith.constant 0 : index
    %message_static = memref.get_global @__ly_unicode_msg_invalid_utf8 : memref<22xi8>
    %message = memref.cast %message_static : memref<22xi8> to memref<?xi8>
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // PEP 393-style adaptive-width representation. One entity, one allocation:
  //   [0,16)               header [refcount, class id = 4]
  //   [16,24)              character width word: 1 (latin-1) / 2 (UCS-2) / 4 (UCS-4)
  //   [24, 24+len*width)   little-endian code units
  // The width word lives OUTSIDE the two-word header (rather than in spare
  // bits of header[1]) because header[1] is the class id that boxed payload
  // handles, repr dispatch and the release hook compare by equality.
  // Canonical-form invariant: every constructor picks the smallest width that
  // fits the widest code point, so equal strings always have identical width
  // and identical code-unit bytes (equality can stay bytewise).
  func.func private @__ly_unicode_alloc(%count: i64, %width: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "alloc"} {
    %data_bytes = arith.muli %count, %width : i64
    %byte_count = arith.index_cast %data_bytes : i64 to index
    %block_prefix = arith.constant 24 : index
    %block_bytes = arith.addi %byte_count, %block_prefix : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %width_offset = arith.constant 16 : index
    %bytes_offset = arith.constant 24 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %width_view = memref.view %block[%width_offset][] : memref<?xi8> to memref<1xi64>
    %bytes = memref.view %block[%bytes_offset][%byte_count] : memref<?xi8> to memref<?xi8>
    %one = arith.constant 1 : i64
    %layout_str = arith.constant 4 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %width_slot = arith.constant 0 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_str, %header[%layout_slot] : memref<2xi64>
    memref.store %width, %width_view[%width_slot] : memref<1xi64>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Character width of an existing str. Read through the header pointer: the
  // width word sits between the two public views, reachable from neither, and
  // every str header view is anchored at the block base by construction.
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64 {
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    %sixteen = arith.constant 16 : i64
    %addr = arith.addi %ptr, %sixteen : i64
    %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    %width = llvm.load %llptr : !llvm.ptr -> i64
    func.return %width : i64
  }

  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64 {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %data_bytes = arith.index_cast %dim : index to i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = arith.divsi %data_bytes, %width : i64
    func.return %count : i64
  }

  func.func private @__ly_unicode_width_for(%cp: i64) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %latin1_limit = arith.constant 256 : i64
    %ucs2_limit = arith.constant 65536 : i64
    %fits1 = arith.cmpi ult, %cp, %latin1_limit : i64
    %fits2 = arith.cmpi ult, %cp, %ucs2_limit : i64
    %wide = arith.select %fits2, %two, %four : i64
    %width = arith.select %fits1, %one, %wide : i64
    func.return %width : i64
  }

  // Code point at code-point index %i of a %width-wide code-unit buffer.
  func.func private @__ly_unicode_get(%bytes: memref<?xi8>, %width: i64, %i: index) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %eight = arith.constant 8 : i64
    %sixteen = arith.constant 16 : i64
    %twenty_four = arith.constant 24 : i64
    %width_index = arith.index_cast %width : i64 to index
    %off = arith.muli %i, %width_index : index
    %is1 = arith.cmpi eq, %width, %one : i64
    %cp = scf.if %is1 -> (i64) {
      %b0 = memref.load %bytes[%off] : memref<?xi8>
      %v = arith.extui %b0 : i8 to i64
      scf.yield %v : i64
    } else {
      %off1 = arith.addi %off, %c1 : index
      %b0 = memref.load %bytes[%off] : memref<?xi8>
      %b1 = memref.load %bytes[%off1] : memref<?xi8>
      %v0 = arith.extui %b0 : i8 to i64
      %v1 = arith.extui %b1 : i8 to i64
      %v1s = arith.shli %v1, %eight : i64
      %lo = arith.ori %v0, %v1s : i64
      %is2 = arith.cmpi eq, %width, %two : i64
      %inner = scf.if %is2 -> (i64) {
        scf.yield %lo : i64
      } else {
        %off2 = arith.addi %off, %c2 : index
        %off3 = arith.addi %off, %c3 : index
        %b2 = memref.load %bytes[%off2] : memref<?xi8>
        %b3 = memref.load %bytes[%off3] : memref<?xi8>
        %v2 = arith.extui %b2 : i8 to i64
        %v3 = arith.extui %b3 : i8 to i64
        %v2s = arith.shli %v2, %sixteen : i64
        %v3s = arith.shli %v3, %twenty_four : i64
        %hi = arith.ori %v2s, %v3s : i64
        %full = arith.ori %lo, %hi : i64
        scf.yield %full : i64
      }
      scf.yield %inner : i64
    }
    func.return %cp : i64
  }

  func.func private @__ly_unicode_put(%bytes: memref<?xi8>, %width: i64, %i: index, %cp: i64) {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %eight = arith.constant 8 : i64
    %sixteen = arith.constant 16 : i64
    %twenty_four = arith.constant 24 : i64
    %width_index = arith.index_cast %width : i64 to index
    %off = arith.muli %i, %width_index : index
    %b0 = arith.trunci %cp : i64 to i8
    memref.store %b0, %bytes[%off] : memref<?xi8>
    %is_wide = arith.cmpi ne, %width, %one : i64
    scf.if %is_wide {
      %off1 = arith.addi %off, %c1 : index
      %s1 = arith.shrui %cp, %eight : i64
      %b1 = arith.trunci %s1 : i64 to i8
      memref.store %b1, %bytes[%off1] : memref<?xi8>
      %is_widest = arith.cmpi ne, %width, %two : i64
      scf.if %is_widest {
        %off2 = arith.addi %off, %c2 : index
        %off3 = arith.addi %off, %c3 : index
        %s2 = arith.shrui %cp, %sixteen : i64
        %s3 = arith.shrui %cp, %twenty_four : i64
        %b2 = arith.trunci %s2 : i64 to i8
        %b3 = arith.trunci %s3 : i64 to i8
        memref.store %b2, %bytes[%off2] : memref<?xi8>
        memref.store %b3, %bytes[%off3] : memref<?xi8>
      }
    }
    func.return
  }

  // Decode one UTF-8 sequence at byte offset %i (relative to %start) of a
  // %len-byte input. Returns (code point, next offset, ok). Strict: truncated
  // sequences, stray continuation bytes, overlong forms, surrogates and
  // values above U+10FFFF are rejected (ok = false).
  func.func private @__ly_utf8_step(%bytes: memref<?xi8>, %start: index, %len: i64, %i: i64) -> (i64, i64, i1) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %false_bit = arith.constant false
    %i_index = arith.index_cast %i : i64 to index
    %pos0 = arith.addi %start, %i_index : index
    %b0_i8 = memref.load %bytes[%pos0] : memref<?xi8>
    %b0 = arith.extui %b0_i8 : i8 to i64
    %ascii_limit = arith.constant 128 : i64
    %is_ascii = arith.cmpi ult, %b0, %ascii_limit : i64
    %result:3 = scf.if %is_ascii -> (i64, i64, i1) {
      %next = arith.addi %i, %one : i64
      scf.yield %b0, %next, %true_bit : i64, i64, i1
    } else {
      %lead2_lo = arith.constant 194 : i64
      %lead2_hi = arith.constant 223 : i64
      %lead3_lo = arith.constant 224 : i64
      %lead3_hi = arith.constant 239 : i64
      %lead4_lo = arith.constant 240 : i64
      %lead4_hi = arith.constant 244 : i64
      %ge2 = arith.cmpi uge, %b0, %lead2_lo : i64
      %le2 = arith.cmpi ule, %b0, %lead2_hi : i64
      %is2 = arith.andi %ge2, %le2 : i1
      %ge3 = arith.cmpi uge, %b0, %lead3_lo : i64
      %le3 = arith.cmpi ule, %b0, %lead3_hi : i64
      %is3 = arith.andi %ge3, %le3 : i1
      %ge4 = arith.cmpi uge, %b0, %lead4_lo : i64
      %le4 = arith.cmpi ule, %b0, %lead4_hi : i64
      %is4 = arith.andi %ge4, %le4 : i1
      %n34 = arith.select %is4, %four, %zero : i64
      %n3 = arith.select %is3, %three, %n34 : i64
      %n = arith.select %is2, %two, %n3 : i64
      %lead_ok = arith.cmpi ne, %n, %zero : i64
      %end = arith.addi %i, %n : i64
      %enough = arith.cmpi sle, %end, %len : i64
      %head_ok = arith.andi %lead_ok, %enough : i1
      %decoded:3 = scf.if %head_ok -> (i64, i64, i1) {
        %mask2 = arith.constant 31 : i64
        %mask3 = arith.constant 15 : i64
        %mask4 = arith.constant 7 : i64
        %init4 = arith.andi %b0, %mask4 : i64
        %init3_raw = arith.andi %b0, %mask3 : i64
        %init2_raw = arith.andi %b0, %mask2 : i64
        %init34 = arith.select %is3, %init3_raw, %init4 : i64
        %init = arith.select %is2, %init2_raw, %init34 : i64
        %n_index = arith.index_cast %n : i64 to index
        %tail:2 = scf.for %j = %c1 to %n_index step %c1 iter_args(%acc = %init, %ok = %true_bit) -> (i64, i1) {
          %pos = arith.addi %pos0, %j : index
          %cj_i8 = memref.load %bytes[%pos] : memref<?xi8>
          %cj = arith.extui %cj_i8 : i8 to i64
          %cont_mask = arith.constant 192 : i64
          %cont_tag = arith.constant 128 : i64
          %tag = arith.andi %cj, %cont_mask : i64
          %is_cont = arith.cmpi eq, %tag, %cont_tag : i64
          %payload_mask = arith.constant 63 : i64
          %payload = arith.andi %cj, %payload_mask : i64
          %shifted = arith.shli %acc, %six : i64
          %next_acc = arith.ori %shifted, %payload : i64
          %next_ok = arith.andi %ok, %is_cont : i1
          scf.yield %next_acc, %next_ok : i64, i1
        }
        // Range checks per sequence length. 2-byte overlongs are already
        // impossible (lead >= 0xC2).
        %min3 = arith.constant 2048 : i64
        %surrogate_lo = arith.constant 55296 : i64
        %surrogate_hi = arith.constant 57343 : i64
        %min4 = arith.constant 65536 : i64
        %max_cp = arith.constant 1114111 : i64
        %ge_min3 = arith.cmpi uge, %tail#0, %min3 : i64
        %ge_slo = arith.cmpi uge, %tail#0, %surrogate_lo : i64
        %le_shi = arith.cmpi ule, %tail#0, %surrogate_hi : i64
        %is_surrogate = arith.andi %ge_slo, %le_shi : i1
        %not_surrogate = arith.xori %is_surrogate, %true_bit : i1
        %ok3 = arith.andi %ge_min3, %not_surrogate : i1
        %ge_min4 = arith.cmpi uge, %tail#0, %min4 : i64
        %le_max = arith.cmpi ule, %tail#0, %max_cp : i64
        %ok4 = arith.andi %ge_min4, %le_max : i1
        %range34 = arith.select %is4, %ok4, %true_bit : i1
        %range_ok = arith.select %is3, %ok3, %range34 : i1
        %all_ok = arith.andi %tail#1, %range_ok : i1
        scf.yield %tail#0, %end, %all_ok : i64, i64, i1
      } else {
        %stop = arith.addi %i, %one : i64
        scf.yield %zero, %stop, %false_bit : i64, i64, i1
      }
      scf.yield %decoded#0, %decoded#1, %decoded#2 : i64, i64, i1
    }
    func.return %result#0, %result#1, %result#2 : i64, i64, i1
  }

  // str construction from UTF-8 bytes (literals, bytes.decode, host text).
  // Pass 1 validates and finds the widest code point; pass 2 decodes into the
  // smallest fitting width. Invalid UTF-8 raises (CPython raises
  // UnicodeDecodeError; the byte previously landed in the payload unchanged).
  func.func @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %scan:3 = scf.while (%i = %zero, %count = %zero, %maxcp = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
      %more = arith.cmpi slt, %i, %len : i64
      scf.condition(%more) %i, %count, %maxcp : i64, i64, i64
    } do {
    ^bb0(%i: i64, %count: i64, %maxcp: i64):
      %cp, %next, %ok = func.call @__ly_utf8_step(%bytes, %start, %len, %i) : (memref<?xi8>, index, i64, i64) -> (i64, i64, i1)
      %bad = arith.xori %ok, %true_bit : i1
      scf.if %bad {
        func.call @__ly_unicode_raise_decode_error() : () -> ()
      }
      %bigger = arith.cmpi ugt, %cp, %maxcp : i64
      %new_max = arith.select %bigger, %cp, %maxcp : i64
      %new_count = arith.addi %count, %one : i64
      scf.yield %next, %new_count, %new_max : i64, i64, i64
    }
    %width = func.call @__ly_unicode_width_for(%scan#2) : (i64) -> i64
    %header, %out = func.call @__ly_unicode_alloc(%scan#1, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %count_index = arith.index_cast %scan#1 : i64 to index
    %fill:2 = scf.while (%i = %zero, %k = %c0) : (i64, index) -> (i64, index) {
      %more = arith.cmpi slt, %k, %count_index : index
      scf.condition(%more) %i, %k : i64, index
    } do {
    ^bb0(%i: i64, %k: index):
      %cp, %next, %ok = func.call @__ly_utf8_step(%bytes, %start, %len, %i) : (memref<?xi8>, index, i64, i64) -> (i64, i64, i1)
      func.call @__ly_unicode_put(%out, %width, %k, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      %next_k = arith.addi %k, %c1 : index
      scf.yield %next, %next_k : i64, index
    }
    func.return %header, %out : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_CodepointLength(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__len__"} {
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.return %count : i64
  }

  func.func @LyUnicode_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__bool__"} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %zero = arith.constant 0 : index
    %result = arith.cmpi ne, %dim, %zero : index
    func.return %result : i1
  }

  // Canonical form makes equality bytewise: equal strings share width, and
  // the width check also rejects the cross-width byte collisions (for
  // example latin-1 "\x00\x01" vs UCS-2 U+0100).
  func.func @LyUnicode_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__eq__"} {
    %c0 = arith.constant 0 : index
    %lhs_dim = memref.dim %lhs_bytes, %c0 : memref<?xi8>
    %rhs_dim = memref.dim %rhs_bytes, %c0 : memref<?xi8>
    %lhs_width = func.call @__ly_unicode_width(%lhs_header) : (memref<2xi64>) -> i64
    %rhs_width = func.call @__ly_unicode_width(%rhs_header) : (memref<2xi64>) -> i64
    %same_len = arith.cmpi eq, %lhs_dim, %rhs_dim : index
    %same_width = arith.cmpi eq, %lhs_width, %rhs_width : i64
    %comparable = arith.andi %same_len, %same_width : i1
    %result = scf.if %comparable -> (i1) {
      %step = arith.constant 1 : index
      %true = arith.constant true
      %all_equal = scf.for %index = %c0 to %lhs_dim step %step iter_args(%current = %true) -> (i1) {
        %lhs_byte = memref.load %lhs_bytes[%index] : memref<?xi8>
        %rhs_byte = memref.load %rhs_bytes[%index] : memref<?xi8>
        %byte_equal = arith.cmpi eq, %lhs_byte, %rhs_byte : i8
        %next = arith.andi %current, %byte_equal : i1
        scf.yield %next : i1
      }
      scf.yield %all_equal : i1
    } else {
      %false = arith.constant false
      scf.yield %false : i1
    }
    func.return %result : i1
  }

  func.func @LyUnicode_NeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__ne__"} {
    %eq = func.call @LyUnicode_EqBool(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i1
    %true = arith.constant true
    %false = arith.constant false
    %result = arith.select %eq, %false, %true : i1
    func.return %result : i1
  }

  // Lexicographic code-point comparison (-1/0/1): the first differing code
  // point decides (matching CPython's code-point ordering), else the shorter
  // string orders first.
  func.func private @LyUnicode_Compare(%lhs_header: memref<2xi64>, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64>, %rhs_bytes: memref<?xi8>) -> i64 {
    %lhs_width = func.call @__ly_unicode_width(%lhs_header) : (memref<2xi64>) -> i64
    %rhs_width = func.call @__ly_unicode_width(%rhs_header) : (memref<2xi64>) -> i64
    %lhs_len = func.call @__ly_unicode_count(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @__ly_unicode_count(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %min_len = arith.minsi %lhs_len, %rhs_len : i64
    %zero = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i64
    %plus_one = arith.constant 1 : i64
    %lower = arith.constant 0 : index
    %upper = arith.index_cast %min_len : i64 to index
    %step = arith.constant 1 : index
    %cp_cmp = scf.for %index = %lower to %upper step %step iter_args(%acc = %zero) -> (i64) {
      %lhs_cp = func.call @__ly_unicode_get(%lhs_bytes, %lhs_width, %index) : (memref<?xi8>, i64, index) -> i64
      %rhs_cp = func.call @__ly_unicode_get(%rhs_bytes, %rhs_width, %index) : (memref<?xi8>, i64, index) -> i64
      %lt = arith.cmpi ult, %lhs_cp, %rhs_cp : i64
      %gt = arith.cmpi ugt, %lhs_cp, %rhs_cp : i64
      %gt_val = arith.select %gt, %plus_one, %zero : i64
      %this = arith.select %lt, %minus_one, %gt_val : i64
      %decided = arith.cmpi ne, %acc, %zero : i64
      %next = arith.select %decided, %acc, %this : i64
      scf.yield %next : i64
    }
    %len_lt = arith.cmpi slt, %lhs_len, %rhs_len : i64
    %len_gt = arith.cmpi sgt, %lhs_len, %rhs_len : i64
    %len_gt_val = arith.select %len_gt, %plus_one, %zero : i64
    %len_cmp = arith.select %len_lt, %minus_one, %len_gt_val : i64
    %prefix_equal = arith.cmpi eq, %cp_cmp, %zero : i64
    %result = arith.select %prefix_equal, %len_cmp, %cp_cmp : i64
    func.return %result : i64
  }

  func.func @LyUnicode_LtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__lt__"} {
    %cmp = func.call @LyUnicode_Compare(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi slt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyUnicode_LeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__le__"} {
    %cmp = func.call @LyUnicode_Compare(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sle, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyUnicode_GtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__gt__"} {
    %cmp = func.call @LyUnicode_Compare(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sgt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyUnicode_GeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__ge__"} {
    %cmp = func.call @LyUnicode_Compare(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sge, %cmp, %zero : i64
    func.return %result : i1
  }

  // O(1) code-point indexing: fixed-width code units make the scan-free load
  // possible; the result re-canonicalizes to the smallest width for that one
  // code point.
  func.func @LyUnicode_GetItem(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %raw_index: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__getitem__"} {
    %codepoints = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %is_negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %codepoints : i64
    %index = arith.select %is_negative, %from_end, %raw_index : i64
    %lower_ok = arith.cmpi sge, %index, %zero : i64
    %upper_ok = arith.cmpi slt, %index, %codepoints : i64
    %valid = arith.andi %lower_ok, %upper_ok : i1
    %result:2 = scf.if %valid -> (memref<2xi64>, memref<?xi8>) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %index_index = arith.index_cast %index : i64 to index
      %cp = func.call @__ly_unicode_get(%bytes, %width, %index_index) : (memref<?xi8>, i64, index) -> i64
      %new_width = func.call @__ly_unicode_width_for(%cp) : (i64) -> i64
      %result_header, %result_bytes = func.call @__ly_unicode_alloc(%one, %new_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      func.call @__ly_unicode_put(%result_bytes, %new_width, %c0, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      scf.yield %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
    } else {
      func.call @__ly_unicode_raise_index_error() : () -> ()
      %empty_width = arith.constant 1 : i64
      %result_header, %result_bytes = func.call @__ly_unicode_alloc(%zero, %empty_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Copy(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "copy"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_header, %new_bytes = func.call @__ly_unicode_alloc(%count, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    scf.for %i = %c0 to %dim step %c1 {
      %byte = memref.load %bytes[%i] : memref<?xi8>
      memref.store %byte, %new_bytes[%i] : memref<?xi8>
    }
    func.return %new_header, %new_bytes : memref<2xi64>, memref<?xi8>
  }

  // str(s) returns s itself (retained) -- CPython identity semantics.
  func.func @LyUnicode_Str(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Printability query implemented beside the UCD tables
  // (runtime/modules/unicodedata.mlir).
  func.func private @__ly_ucd_is_printable(%cp: i64) -> i1

  // Repr expansion width in characters for a code point that is neither the
  // chosen quote nor a backslash: 1 = pass through (UCD-printable, matching
  // CPython's unicode_repr), 2 = \t \n \r, 4/6/10 = \xNN / \uNNNN /
  // \U00NNNNNN for non-printable code points by magnitude.
  func.func private @__ly_unicode_repr_class(%cp: i64) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %ten = arith.constant 10 : i64
    %tab = arith.constant 9 : i64
    %nl = arith.constant 10 : i64
    %cr = arith.constant 13 : i64
    %latin1_limit = arith.constant 256 : i64
    %ucs2_limit = arith.constant 65536 : i64
    %is_tab = arith.cmpi eq, %cp, %tab : i64
    %is_nl = arith.cmpi eq, %cp, %nl : i64
    %is_cr = arith.cmpi eq, %cp, %cr : i64
    %tn = arith.ori %is_tab, %is_nl : i1
    %is_tnr = arith.ori %tn, %is_cr : i1
    %printable = func.call @__ly_ucd_is_printable(%cp) : (i64) -> i1
    %fits1 = arith.cmpi ult, %cp, %latin1_limit : i64
    %fits2 = arith.cmpi ult, %cp, %ucs2_limit : i64
    %six_or_ten = arith.select %fits2, %six, %ten : i64
    %escape = arith.select %fits1, %four, %six_or_ten : i64
    %escape_or_pass = arith.select %printable, %one, %escape : i64
    %class = arith.select %is_tnr, %two, %escape_or_pass : i64
    func.return %class : i64
  }

  // %digits lowercase hex digits of %value, most significant first, written
  // at code-point positions [%pos, %pos+%digits).
  func.func private @__ly_unicode_put_hex(%bytes: memref<?xi8>, %width: i64, %pos: index, %value: i64, %digits: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %four = arith.constant 4 : i64
    %fifteen = arith.constant 15 : i64
    %ten = arith.constant 10 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_a_minus_ten = arith.constant 87 : i64
    scf.for %d = %c0 to %digits step %c1 {
      %last = arith.subi %digits, %c1 : index
      %rev = arith.subi %last, %d : index
      %rev_i64 = arith.index_cast %rev : index to i64
      %shift = arith.muli %rev_i64, %four : i64
      %shifted = arith.shrui %value, %shift : i64
      %nibble = arith.andi %shifted, %fifteen : i64
      %is_decimal = arith.cmpi ult, %nibble, %ten : i64
      %base = arith.select %is_decimal, %ascii_zero, %ascii_a_minus_ten : i64
      %ch = arith.addi %nibble, %base : i64
      %dst = arith.addi %pos, %d : index
      func.call @__ly_unicode_put(%bytes, %width, %dst, %ch) : (memref<?xi8>, i64, index, i64) -> ()
    }
    func.return
  }

  // CPython `str.__repr__`: wrap in quotes and escape. The quote is `'` unless
  // the string contains `'` and no `"` (then `"`), matching unicode_repr.
  // Escapes: \\ , the chosen quote, \t \n \r, and \xNN per
  // __ly_unicode_repr_class. Code-point oriented; the output re-canonicalizes
  // to the smallest width that fits what is actually emitted.
  func.func @LyUnicode_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %zero64 = arith.constant 0 : i64
    %one64 = arith.constant 1 : i64
    %two64 = arith.constant 2 : i64
    %four64 = arith.constant 4 : i64
    %six64 = arith.constant 6 : i64
    %ten64 = arith.constant 10 : i64
    %sq = arith.constant 39 : i64
    %dq = arith.constant 34 : i64
    %bs = arith.constant 92 : i64
    %tab = arith.constant 9 : i64
    %nl = arith.constant 10 : i64
    %t_char = arith.constant 116 : i64
    %n_char = arith.constant 110 : i64
    %r_char = arith.constant 114 : i64
    %x_char = arith.constant 120 : i64
    %u_char = arith.constant 117 : i64
    %cap_u_char = arith.constant 85 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index

    // Pass 1: count `'`/`"`, the escaped body length (quotes counted as 1)
    // and the widest pass-through code point.
    %scan:4 = scf.for %i = %c0 to %count_index step %c1
        iter_args(%csq = %zero64, %cdq = %zero64, %common = %zero64, %maxcp = %zero64)
        -> (i64, i64, i64, i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_sq = arith.cmpi eq, %cp, %sq : i64
      %is_dq = arith.cmpi eq, %cp, %dq : i64
      %is_bs = arith.cmpi eq, %cp, %bs : i64
      %is_quote = arith.ori %is_sq, %is_dq : i1
      %class = func.call @__ly_unicode_repr_class(%cp) : (i64) -> i64
      %quote_or_class = arith.select %is_quote, %one64, %class : i64
      %contrib = arith.select %is_bs, %two64, %quote_or_class : i64
      %passes = arith.cmpi eq, %contrib, %one64 : i64
      %candidate = arith.select %passes, %cp, %zero64 : i64
      %bigger = arith.cmpi ugt, %candidate, %maxcp : i64
      %new_max = arith.select %bigger, %candidate, %maxcp : i64
      %add_sq = arith.select %is_sq, %one64, %zero64 : i64
      %add_dq = arith.select %is_dq, %one64, %zero64 : i64
      %csq2 = arith.addi %csq, %add_sq : i64
      %cdq2 = arith.addi %cdq, %add_dq : i64
      %common2 = arith.addi %common, %contrib : i64
      scf.yield %csq2, %cdq2, %common2, %new_max : i64, i64, i64, i64
    }

    %sq_present = arith.cmpi ne, %scan#0, %zero64 : i64
    %dq_absent = arith.cmpi eq, %scan#1, %zero64 : i64
    %use_double = arith.andi %sq_present, %dq_absent : i1
    %quote = arith.select %use_double, %dq, %sq : i64
    %count_q = arith.select %use_double, %scan#1, %scan#0 : i64
    %body = arith.addi %scan#2, %count_q : i64
    %total = arith.addi %body, %two64 : i64
    %out_width = func.call @__ly_unicode_width_for(%scan#3) : (i64) -> i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%total, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_unicode_put(%out_bytes, %out_width, %c0, %quote) : (memref<?xi8>, i64, index, i64) -> ()

    // Pass 2: fill the escaped body starting at position 1.
    scf.for %i = %c0 to %count_index step %c1 iter_args(%pos = %c1) -> (index) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_bs = arith.cmpi eq, %cp, %bs : i64
      %is_quote = arith.cmpi eq, %cp, %quote : i64
      %two_char = arith.ori %is_bs, %is_quote : i1
      %next_pos = scf.if %two_char -> (index) {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %bs) : (memref<?xi8>, i64, index, i64) -> ()
        %pos1 = arith.addi %pos, %c1 : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %pos1, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        %advanced = arith.addi %pos, %c2 : index
        scf.yield %advanced : index
      } else {
        %class = func.call @__ly_unicode_repr_class(%cp) : (i64) -> i64
        %is_pass = arith.cmpi eq, %class, %one64 : i64
        %inner = scf.if %is_pass -> (index) {
          func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %cp) : (memref<?xi8>, i64, index, i64) -> ()
          %advanced = arith.addi %pos, %c1 : index
          scf.yield %advanced : index
        } else {
          func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %bs) : (memref<?xi8>, i64, index, i64) -> ()
          %pos1 = arith.addi %pos, %c1 : index
          %is_two = arith.cmpi eq, %class, %two64 : i64
          %escaped = scf.if %is_two -> (index) {
            %is_tab = arith.cmpi eq, %cp, %tab : i64
            %is_nl = arith.cmpi eq, %cp, %nl : i64
            %nr_char = arith.select %is_nl, %n_char, %r_char : i64
            %second = arith.select %is_tab, %t_char, %nr_char : i64
            func.call @__ly_unicode_put(%out_bytes, %out_width, %pos1, %second) : (memref<?xi8>, i64, index, i64) -> ()
            %advanced = arith.addi %pos, %c2 : index
            scf.yield %advanced : index
          } else {
            %is_four = arith.cmpi eq, %class, %four64 : i64
            %is_six = arith.cmpi eq, %class, %six64 : i64
            %ubig_marker = arith.select %is_six, %u_char, %cap_u_char : i64
            %marker = arith.select %is_four, %x_char, %ubig_marker : i64
            func.call @__ly_unicode_put(%out_bytes, %out_width, %pos1, %marker) : (memref<?xi8>, i64, index, i64) -> ()
            %digits8 = arith.select %is_six, %c4, %c8 : index
            %digits = arith.select %is_four, %c2, %digits8 : index
            %pos2 = arith.addi %pos, %c2 : index
            func.call @__ly_unicode_put_hex(%out_bytes, %out_width, %pos2, %cp, %digits) : (memref<?xi8>, i64, index, i64, index) -> ()
            %class_index = arith.index_cast %class : i64 to index
            %advanced = arith.addi %pos, %class_index : index
            scf.yield %advanced : index
          }
          scf.yield %escaped : index
        }
        scf.yield %inner : index
      }
      scf.yield %next_pos : index
    }

    %total_idx = arith.index_cast %total : i64 to index
    %last = arith.subi %total_idx, %c1 : index
    func.call @__ly_unicode_put(%out_bytes, %out_width, %last, %quote) : (memref<?xi8>, i64, index, i64) -> ()
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_width = func.call @__ly_unicode_width(%lhs_header) : (memref<2xi64>) -> i64
    %rhs_width = func.call @__ly_unicode_width(%rhs_header) : (memref<2xi64>) -> i64
    %lhs_len = func.call @__ly_unicode_count(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @__ly_unicode_count(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    // Operands are canonical (smallest fitting width), so the wider operand
    // width IS the smallest width that fits the concatenation.
    %width = arith.maxsi %lhs_width, %rhs_width : i64
    %total_len = arith.addi %lhs_len, %rhs_len : i64
    %header, %bytes = func.call @__ly_unicode_alloc(%total_len, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %lhs_upper = arith.index_cast %lhs_len : i64 to index
    %rhs_upper = arith.index_cast %rhs_len : i64 to index
    scf.for %index = %c0 to %lhs_upper step %c1 {
      %cp = func.call @__ly_unicode_get(%lhs_bytes, %lhs_width, %index) : (memref<?xi8>, i64, index) -> i64
      func.call @__ly_unicode_put(%bytes, %width, %index, %cp) : (memref<?xi8>, i64, index, i64) -> ()
    }
    scf.for %index = %c0 to %rhs_upper step %c1 {
      %cp = func.call @__ly_unicode_get(%rhs_bytes, %rhs_width, %index) : (memref<?xi8>, i64, index) -> i64
      %dest = arith.addi %lhs_upper, %index : index
      func.call @__ly_unicode_put(%bytes, %width, %dest, %cp) : (memref<?xi8>, i64, index, i64) -> ()
    }
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }


  // ===== str search / comparison methods (Wave 1) =====

  // 7-field UCD ctype accessor (defined in the generated _ucd.mlir).
  func.func private @__ly_ucd_ctype(%cp: i64) -> (i64, i64, i64, i64, i64, i64, i64)

  func.func private @__ly_unicode_cp_is_space(%cp: i64) -> i1 {
    %zero = arith.constant 0 : i64
    %space_bit = arith.constant 16 : i64
    %u, %l, %f, %t, %dec, %dig, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64, i64)
    %masked = arith.andi %flags, %space_bit : i64
    %is_space = arith.cmpi ne, %masked, %zero : i64
    func.return %is_space : i1
  }

  // "substring not found"
  memref.global "private" constant @__ly_unicode_msg_substring_not_found : memref<19xi8> = dense<[115, 117, 98, 115, 116, 114, 105, 110, 103, 32, 110, 111, 116, 32, 102, 111, 117, 110, 100]>
  // "The fill character must be exactly one character long"
  memref.global "private" constant @__ly_unicode_msg_bad_fill : memref<53xi8> = dense<[84, 104, 101, 32, 102, 105, 108, 108, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 32, 109, 117, 115, 116, 32, 98, 101, 32, 101, 120, 97, 99, 116, 108, 121, 32, 111, 110, 101, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 32, 108, 111, 110, 103]>

  func.func private @__ly_unicode_raise(%class_id: i64, %message: memref<?xi8>, %length: i64) {
    %start = arith.constant 0 : index
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // CPython ADJUST_INDICES: negative indices count from the end (clamped to
  // 0), the end clamps to the length, and the start deliberately does NOT
  // clamp downward to the length -- find("", past-the-end) must miss.
  func.func private @__ly_unicode_adjust_range(%len: i64, %start_raw: i64, %end_raw: i64) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %start_neg = arith.cmpi slt, %start_raw, %zero : i64
    %start_shift = arith.addi %start_raw, %len : i64
    %start_from_end = arith.maxsi %start_shift, %zero : i64
    %start = arith.select %start_neg, %start_from_end, %start_raw : i64
    %end_over = arith.cmpi sgt, %end_raw, %len : i64
    %end_neg = arith.cmpi slt, %end_raw, %zero : i64
    %end_shift = arith.addi %end_raw, %len : i64
    %end_from_end = arith.maxsi %end_shift, %zero : i64
    %end_in = arith.select %end_neg, %end_from_end, %end_raw : i64
    %end = arith.select %end_over, %len, %end_in : i64
    func.return %start, %end : i64, i64
  }

  // s[si .. si+n) == t[ti .. ti+n), by code point.
  func.func private @__ly_unicode_match_at(%s_bytes: memref<?xi8>, %s_width: i64, %si: index, %t_bytes: memref<?xi8>, %t_width: i64, %ti: index, %n: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %all = scf.for %j = %c0 to %n step %c1 iter_args(%acc = %true_bit) -> (i1) {
      %sj = arith.addi %si, %j : index
      %tj = arith.addi %ti, %j : index
      %scp = func.call @__ly_unicode_get(%s_bytes, %s_width, %sj) : (memref<?xi8>, i64, index) -> i64
      %tcp = func.call @__ly_unicode_get(%t_bytes, %t_width, %tj) : (memref<?xi8>, i64, index) -> i64
      %eq = arith.cmpi eq, %scp, %tcp : i64
      %next = arith.andi %acc, %eq : i1
      scf.yield %next : i1
    }
    func.return %all : i1
  }

  // First (or, %reverse, last) index in [start, end-n] where the needle
  // matches; -1 when the window is empty or nothing matches. Indices are
  // pre-adjusted code-point positions.
  func.func private @__ly_unicode_find_core(%s_bytes: memref<?xi8>, %s_width: i64, %t_bytes: memref<?xi8>, %t_width: i64, %start: i64, %end: i64, %n: i64, %reverse: i1) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %limit = arith.subi %end, %n : i64
    %viable = arith.cmpi sle, %start, %limit : i64
    %found = scf.if %viable -> (i64) {
      %n_index = arith.index_cast %n : i64 to index
      %span = arith.subi %limit, %start : i64
      %positions_i64 = arith.addi %span, %one : i64
      %positions = arith.index_cast %positions_i64 : i64 to index
      %scan = scf.for %k = %c0 to %positions step %c1 iter_args(%acc = %minus_one) -> (i64) {
        %k_i64 = arith.index_cast %k : index to i64
        %fwd = arith.addi %start, %k_i64 : i64
        %rev = arith.subi %limit, %k_i64 : i64
        %pos = arith.select %reverse, %rev, %fwd : i64
        %pos_index = arith.index_cast %pos : i64 to index
        %ti = arith.constant 0 : index
        %eq = func.call @__ly_unicode_match_at(%s_bytes, %s_width, %pos_index, %t_bytes, %t_width, %ti, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        %not_yet = arith.cmpi eq, %acc, %minus_one : i64
        %take = arith.andi %eq, %not_yet : i1
        %next = arith.select %take, %pos, %acc : i64
        scf.yield %next : i64
      }
      scf.yield %scan : i64
    } else {
      scf.yield %minus_one : i64
    }
    func.return %found : i64
  }

  func.func @LyUnicode_StartsWith(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %prefix_header: memref<2xi64> {ly.ownership.object_header}, %prefix_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "startswith"} {
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_unicode_count(%prefix_header, %prefix_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %tail = arith.addi %start, %n : i64
    %fits = arith.cmpi sle, %tail, %end : i64
    %result = scf.if %fits -> (i1) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %prefix_width = func.call @__ly_unicode_width(%prefix_header) : (memref<2xi64>) -> i64
      %si = arith.index_cast %start : i64 to index
      %ti = arith.constant 0 : index
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_unicode_match_at(%bytes, %width, %si, %prefix_bytes, %prefix_width, %ti, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  func.func @LyUnicode_EndsWith(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %suffix_header: memref<2xi64> {ly.ownership.object_header}, %suffix_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "endswith"} {
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_unicode_count(%suffix_header, %suffix_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %pos = arith.subi %end, %n : i64
    %fits = arith.cmpi sge, %pos, %start : i64
    %result = scf.if %fits -> (i1) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %suffix_width = func.call @__ly_unicode_width(%suffix_header) : (memref<2xi64>) -> i64
      %si = arith.index_cast %pos : i64 to index
      %ti = arith.constant 0 : index
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_unicode_match_at(%bytes, %width, %si, %suffix_bytes, %suffix_width, %ti, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  // Shared find/rfind entry: adjusted range, then the linear core scan.
  func.func private @__ly_unicode_find_method(%header: memref<2xi64>, %bytes: memref<?xi8>, %sub_header: memref<2xi64>, %sub_bytes: memref<?xi8>, %start_raw: i64, %end_raw: i64, %reverse: i1) -> i64 {
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_unicode_count(%sub_header, %sub_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %sub_width = func.call @__ly_unicode_width(%sub_header) : (memref<2xi64>) -> i64
    %found = func.call @__ly_unicode_find_core(%bytes, %width, %sub_bytes, %sub_width, %start, %end, %n, %reverse) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
    func.return %found : i64
  }

  func.func @LyUnicode_Find(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "find", ly.runtime.result_contract = "builtins.int"} {
    %false_bit = arith.constant false
    %found = func.call @__ly_unicode_find_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %false_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %result:3 = func.call @LyLong_FromI64(%found) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyUnicode_RFind(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rfind", ly.runtime.result_contract = "builtins.int"} {
    %true_bit = arith.constant true
    %found = func.call @__ly_unicode_find_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %true_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %result:3 = func.call @LyLong_FromI64(%found) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_unicode_index_method(%header: memref<2xi64>, %bytes: memref<?xi8>, %sub_header: memref<2xi64>, %sub_bytes: memref<?xi8>, %start_raw: i64, %end_raw: i64, %reverse: i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %found = func.call @__ly_unicode_find_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %reverse) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %missing = arith.cmpi slt, %found, %zero : i64
    scf.if %missing {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 19 : i64
      %static = memref.get_global @__ly_unicode_msg_substring_not_found : memref<19xi8>
      %message = memref.cast %static : memref<19xi8> to memref<?xi8>
      func.call @__ly_unicode_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %result:3 = func.call @LyLong_FromI64(%found) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyUnicode_Index(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "index", ly.runtime.result_contract = "builtins.int"} {
    %false_bit = arith.constant false
    %result:3 = func.call @__ly_unicode_index_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %false_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyUnicode_RIndex(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rindex", ly.runtime.result_contract = "builtins.int"} {
    %true_bit = arith.constant true
    %result:3 = func.call @__ly_unicode_index_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %true_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Non-overlapping occurrence count in [start, end).
  func.func @LyUnicode_CountSub(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "count", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_unicode_count(%sub_header, %sub_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %is_empty = arith.cmpi eq, %n, %zero : i64
    %total = scf.if %is_empty -> (i64) {
      %span = arith.subi %end, %start : i64
      %viable = arith.cmpi sge, %span, %zero : i64
      %hits = arith.addi %span, %one : i64
      %count = arith.select %viable, %hits, %zero : i64
      scf.yield %count : i64
    } else {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %sub_width = func.call @__ly_unicode_width(%sub_header) : (memref<2xi64>) -> i64
      %n_index = arith.index_cast %n : i64 to index
      %scan:2 = scf.while (%pos = %start, %count = %zero) : (i64, i64) -> (i64, i64) {
        %tail = arith.addi %pos, %n : i64
        %more = arith.cmpi sle, %tail, %end : i64
        scf.condition(%more) %pos, %count : i64, i64
      } do {
      ^bb0(%pos: i64, %count: i64):
        %pos_index = arith.index_cast %pos : i64 to index
        %ti = arith.constant 0 : index
        %eq = func.call @__ly_unicode_match_at(%bytes, %width, %pos_index, %sub_bytes, %sub_width, %ti, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        %skip = arith.select %eq, %n, %one : i64
        %bump = arith.select %eq, %one, %zero : i64
        %next_pos = arith.addi %pos, %skip : i64
        %next_count = arith.addi %count, %bump : i64
        scf.yield %next_pos, %next_count : i64, i64
      }
      scf.yield %scan#1 : i64
    }
    %result:3 = func.call @LyLong_FromI64(%total) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }


  // ===== str slicing / transform methods (Wave 1) =====

  // Copy of the code points [start, end). Every producer must re-scan for
  // the widest code point: a canonical substring of a wide string can be
  // narrower than its source (equality stays bytewise only if slices
  // re-canonicalize).
  func.func private @__ly_unicode_slice(%header: memref<2xi64>, %bytes: memref<?xi8>, %start: index, %end: index) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %maxcp = scf.for %i = %start to %end step %c1 iter_args(%acc = %zero) -> (i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %bigger = arith.cmpi ugt, %cp, %acc : i64
      %next = arith.select %bigger, %cp, %acc : i64
      scf.yield %next : i64
    }
    %span = arith.subi %end, %start : index
    %count = arith.index_cast %span : index to i64
    %out_width = func.call @__ly_unicode_width_for(%maxcp) : (i64) -> i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%count, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    scf.for %i = %start to %end step %c1 {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %dst = arith.subi %i, %start : index
      func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // Retained self, the CPython identity-return fast path (strip() with
  // nothing to strip etc. still copies; only methods documented to return
  // the receiver unchanged use this).
  func.func private @__ly_unicode_retain_self(%header: memref<2xi64>, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Membership of %cp in the chars string (linear: strip char sets are tiny).
  func.func private @__ly_unicode_cp_in_str(%cp: i64, %t_bytes: memref<?xi8>, %t_width: i64, %t_count: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false_bit = arith.constant false
    %found = scf.for %j = %c0 to %t_count step %c1 iter_args(%acc = %false_bit) -> (i1) {
      %tcp = func.call @__ly_unicode_get(%t_bytes, %t_width, %j) : (memref<?xi8>, i64, index) -> i64
      %eq = arith.cmpi eq, %cp, %tcp : i64
      %next = arith.ori %acc, %eq : i1
      scf.yield %next : i1
    }
    func.return %found : i1
  }

  // Shared strip walk. %mode bit 1 = strip left, bit 2 = strip right.
  // %use_chars false = Unicode whitespace (str.strip()); true = membership
  // in the chars operand.
  func.func private @__ly_unicode_strip_core(%header: memref<2xi64>, %bytes: memref<?xi8>, %mode: i64, %use_chars: i1, %ch_bytes: memref<?xi8>, %ch_width: i64, %ch_count: index) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %true_bit = arith.constant true
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index

    %left_mask = arith.andi %mode, %one : i64
    %strip_left = arith.cmpi ne, %left_mask, %zero : i64
    %begin = scf.if %strip_left -> (index) {
      %scan:2 = scf.while (%i = %c0, %go = %true_bit) : (index, i1) -> (index, i1) {
        %more = arith.cmpi ult, %i, %count_index : index
        %continue = arith.andi %more, %go : i1
        scf.condition(%continue) %i, %go : index, i1
      } do {
      ^bb0(%i: index, %go: i1):
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %stripped = scf.if %use_chars -> (i1) {
          %in = func.call @__ly_unicode_cp_in_str(%cp, %ch_bytes, %ch_width, %ch_count) : (i64, memref<?xi8>, i64, index) -> i1
          scf.yield %in : i1
        } else {
          %sp = func.call @__ly_unicode_cp_is_space(%cp) : (i64) -> i1
          scf.yield %sp : i1
        }
        %next = arith.addi %i, %c1 : index
        %keep = arith.select %stripped, %next, %i : index
        scf.yield %keep, %stripped : index, i1
      }
      scf.yield %scan#0 : index
    } else {
      scf.yield %c0 : index
    }

    %right_mask = arith.andi %mode, %two : i64
    %strip_right = arith.cmpi ne, %right_mask, %zero : i64
    %finish = scf.if %strip_right -> (index) {
      %scan:2 = scf.while (%i = %count_index, %go = %true_bit) : (index, i1) -> (index, i1) {
        %more = arith.cmpi ugt, %i, %begin : index
        %continue = arith.andi %more, %go : i1
        scf.condition(%continue) %i, %go : index, i1
      } do {
      ^bb0(%i: index, %go: i1):
        %prev = arith.subi %i, %c1 : index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %prev) : (memref<?xi8>, i64, index) -> i64
        %stripped = scf.if %use_chars -> (i1) {
          %in = func.call @__ly_unicode_cp_in_str(%cp, %ch_bytes, %ch_width, %ch_count) : (i64, memref<?xi8>, i64, index) -> i1
          scf.yield %in : i1
        } else {
          %sp = func.call @__ly_unicode_cp_is_space(%cp) : (i64) -> i1
          scf.yield %sp : i1
        }
        %keep = arith.select %stripped, %prev, %i : index
        scf.yield %keep, %stripped : index, i1
      }
      scf.yield %scan#0 : index
    } else {
      scf.yield %count_index : index
    }

    %out:2 = func.call @__ly_unicode_slice(%header, %bytes, %begin, %finish) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %out#0, %out#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_unicode_strip_ws(%header: memref<2xi64>, %bytes: memref<?xi8>, %mode: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %false_bit = arith.constant false
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %result:2 = func.call @__ly_unicode_strip_core(%header, %bytes, %mode, %false_bit, %bytes, %one, %c0) : (memref<2xi64>, memref<?xi8>, i64, i1, memref<?xi8>, i64, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_unicode_strip_chars(%header: memref<2xi64>, %bytes: memref<?xi8>, %chars_header: memref<2xi64>, %chars_bytes: memref<?xi8>, %mode: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %true_bit = arith.constant true
    %ch_width = func.call @__ly_unicode_width(%chars_header) : (memref<2xi64>) -> i64
    %ch_count = func.call @__ly_unicode_count(%chars_header, %chars_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %ch_count_index = arith.index_cast %ch_count : i64 to index
    %result:2 = func.call @__ly_unicode_strip_core(%header, %bytes, %mode, %true_bit, %chars_bytes, %ch_width, %ch_count_index) : (memref<2xi64>, memref<?xi8>, i64, i1, memref<?xi8>, i64, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Strip(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 3 : i64
    %result:2 = func.call @__ly_unicode_strip_ws(%header, %bytes, %mode) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_LStrip(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 1 : i64
    %result:2 = func.call @__ly_unicode_strip_ws(%header, %bytes, %mode) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RStrip(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 2 : i64
    %result:2 = func.call @__ly_unicode_strip_ws(%header, %bytes, %mode) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_StripChars(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %chars_header: memref<2xi64> {ly.ownership.object_header}, %chars_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 3 : i64
    %result:2 = func.call @__ly_unicode_strip_chars(%header, %bytes, %chars_header, %chars_bytes, %mode) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_LStripChars(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %chars_header: memref<2xi64> {ly.ownership.object_header}, %chars_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 1 : i64
    %result:2 = func.call @__ly_unicode_strip_chars(%header, %bytes, %chars_header, %chars_bytes, %mode) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RStripChars(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %chars_header: memref<2xi64> {ly.ownership.object_header}, %chars_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 2 : i64
    %result:2 = func.call @__ly_unicode_strip_chars(%header, %bytes, %chars_header, %chars_bytes, %mode) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RemovePrefix(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %prefix_header: memref<2xi64> {ly.ownership.object_header}, %prefix_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "removeprefix", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %n = func.call @__ly_unicode_count(%prefix_header, %prefix_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %fits = arith.cmpi sle, %n, %len : i64
    %matched = scf.if %fits -> (i1) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %pwidth = func.call @__ly_unicode_width(%prefix_header) : (memref<2xi64>) -> i64
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_unicode_match_at(%bytes, %width, %c0, %prefix_bytes, %pwidth, %c0, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    %result:2 = scf.if %matched -> (memref<2xi64>, memref<?xi8>) {
      %n_index = arith.index_cast %n : i64 to index
      %len_index = arith.index_cast %len : i64 to index
      %sliced:2 = func.call @__ly_unicode_slice(%header, %bytes, %n_index, %len_index) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %sliced#0, %sliced#1 : memref<2xi64>, memref<?xi8>
    } else {
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RemoveSuffix(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %suffix_header: memref<2xi64> {ly.ownership.object_header}, %suffix_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "removesuffix", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %n = func.call @__ly_unicode_count(%suffix_header, %suffix_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %pos = arith.subi %len, %n : i64
    %nonempty = arith.cmpi sgt, %n, %zero : i64
    %fits = arith.cmpi sge, %pos, %zero : i64
    %check = arith.andi %nonempty, %fits : i1
    %matched = scf.if %check -> (i1) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %swidth = func.call @__ly_unicode_width(%suffix_header) : (memref<2xi64>) -> i64
      %n_index = arith.index_cast %n : i64 to index
      %pos_index = arith.index_cast %pos : i64 to index
      %eq = func.call @__ly_unicode_match_at(%bytes, %width, %pos_index, %suffix_bytes, %swidth, %c0, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    %result:2 = scf.if %matched -> (memref<2xi64>, memref<?xi8>) {
      %pos_index = arith.index_cast %pos : i64 to index
      %sliced:2 = func.call @__ly_unicode_slice(%header, %bytes, %c0, %pos_index) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %sliced#0, %sliced#1 : memref<2xi64>, memref<?xi8>
    } else {
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // str.replace via the unified walk (i <= len when old is empty, so the
  // trailing insertion happens; a match consumes old and suppresses the
  // char emit). Pass 1 measures (count, widest); pass 2 writes.
  func.func @LyUnicode_Replace(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %old_header: memref<2xi64> {ly.ownership.object_header}, %old_bytes: memref<?xi8>, %new_header: memref<2xi64> {ly.ownership.object_header}, %new_bytes: memref<?xi8>, %limit: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "replace", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %old_width = func.call @__ly_unicode_width(%old_header) : (memref<2xi64>) -> i64
    %new_width = func.call @__ly_unicode_width(%new_header) : (memref<2xi64>) -> i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %old_n = func.call @__ly_unicode_count(%old_header, %old_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_n = func.call @__ly_unicode_count(%new_header, %new_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %old_n_index = arith.index_cast %old_n : i64 to index
    %new_n_index = arith.index_cast %new_n : i64 to index
    %new_max = scf.for %j = %c0 to %new_n_index step %c1 iter_args(%acc = %zero) -> (i64) {
      %cp = func.call @__ly_unicode_get(%new_bytes, %new_width, %j) : (memref<?xi8>, i64, index) -> i64
      %bigger = arith.cmpi ugt, %cp, %acc : i64
      %next = arith.select %bigger, %cp, %acc : i64
      scf.yield %next : i64
    }
    %old_empty = arith.cmpi eq, %old_n, %zero : i64
    %bound = scf.if %old_empty -> (i64) {
      %plus = arith.addi %len, %one : i64
      scf.yield %plus : i64
    } else {
      scf.yield %len : i64
    }

    %measure:4 = scf.while (%i = %zero, %rem = %limit, %total = %zero, %maxcp = %zero) : (i64, i64, i64, i64) -> (i64, i64, i64, i64) {
      %more = arith.cmpi slt, %i, %bound : i64
      scf.condition(%more) %i, %rem, %total, %maxcp : i64, i64, i64, i64
    } do {
    ^bb0(%i: i64, %rem: i64, %total: i64, %maxcp: i64):
      %has_budget = arith.cmpi ne, %rem, %zero : i64
      %tail = arith.addi %i, %old_n : i64
      %in_range = arith.cmpi sle, %tail, %len : i64
      %viable = arith.andi %has_budget, %in_range : i1
      %matched = scf.if %viable -> (i1) {
        %i_index = arith.index_cast %i : i64 to index
        %eq = func.call @__ly_unicode_match_at(%bytes, %width, %i_index, %old_bytes, %old_width, %c0, %old_n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        scf.yield %eq : i1
      } else {
        %false_bit = arith.constant false
        scf.yield %false_bit : i1
      }
      %old_nonempty = arith.cmpi sgt, %old_n, %zero : i64
      %skip_char = arith.andi %matched, %old_nonempty : i1
      %in_str = arith.cmpi slt, %i, %len : i64
      %true_a = arith.constant true
      %not_skip = arith.xori %skip_char, %true_a : i1
      %emit_char = arith.andi %in_str, %not_skip : i1
      %new_contrib = arith.select %matched, %new_n, %zero : i64
      %char_contrib = arith.select %emit_char, %one, %zero : i64
      %next_total_a = arith.addi %total, %new_contrib : i64
      %next_total = arith.addi %next_total_a, %char_contrib : i64
      %match_max = arith.select %matched, %new_max, %zero : i64
      %cp = scf.if %emit_char -> (i64) {
        %i_index = arith.index_cast %i : i64 to index
        %value = func.call @__ly_unicode_get(%bytes, %width, %i_index) : (memref<?xi8>, i64, index) -> i64
        scf.yield %value : i64
      } else {
        scf.yield %zero : i64
      }
      %m1 = arith.maxui %maxcp, %match_max : i64
      %next_max = arith.maxui %m1, %cp : i64
      %stride = arith.select %skip_char, %old_n, %one : i64
      %next_i = arith.addi %i, %stride : i64
      %dec = arith.select %matched, %one, %zero : i64
      %next_rem = arith.subi %rem, %dec : i64
      scf.yield %next_i, %next_rem, %next_total, %next_max : i64, i64, i64, i64
    }

    %out_width = func.call @__ly_unicode_width_for(%measure#3) : (i64) -> i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%measure#2, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)

    %write:3 = scf.while (%i = %zero, %rem = %limit, %pos = %c0) : (i64, i64, index) -> (i64, i64, index) {
      %more = arith.cmpi slt, %i, %bound : i64
      scf.condition(%more) %i, %rem, %pos : i64, i64, index
    } do {
    ^bb0(%i: i64, %rem: i64, %pos: index):
      %has_budget = arith.cmpi ne, %rem, %zero : i64
      %tail = arith.addi %i, %old_n : i64
      %in_range = arith.cmpi sle, %tail, %len : i64
      %viable = arith.andi %has_budget, %in_range : i1
      %matched = scf.if %viable -> (i1) {
        %i_index = arith.index_cast %i : i64 to index
        %eq = func.call @__ly_unicode_match_at(%bytes, %width, %i_index, %old_bytes, %old_width, %c0, %old_n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        scf.yield %eq : i1
      } else {
        %false_bit = arith.constant false
        scf.yield %false_bit : i1
      }
      %after_new = scf.if %matched -> (index) {
        scf.for %j = %c0 to %new_n_index step %c1 {
          %cp = func.call @__ly_unicode_get(%new_bytes, %new_width, %j) : (memref<?xi8>, i64, index) -> i64
          %dst = arith.addi %pos, %j : index
          func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        }
        %advanced = arith.addi %pos, %new_n_index : index
        scf.yield %advanced : index
      } else {
        scf.yield %pos : index
      }
      %old_nonempty = arith.cmpi sgt, %old_n, %zero : i64
      %skip_char = arith.andi %matched, %old_nonempty : i1
      %in_str = arith.cmpi slt, %i, %len : i64
      %true_b = arith.constant true
      %not_skip = arith.xori %skip_char, %true_b : i1
      %emit_char = arith.andi %in_str, %not_skip : i1
      %after_char = scf.if %emit_char -> (index) {
        %i_index = arith.index_cast %i : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i_index) : (memref<?xi8>, i64, index) -> i64
        func.call @__ly_unicode_put(%out_bytes, %out_width, %after_new, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        %advanced = arith.addi %after_new, %c1 : index
        scf.yield %advanced : index
      } else {
        scf.yield %after_new : index
      }
      %stride = arith.select %skip_char, %old_n, %one : i64
      %next_i = arith.addi %i, %stride : i64
      %dec = arith.select %matched, %one, %zero : i64
      %next_rem = arith.subi %rem, %dec : i64
      scf.yield %next_i, %next_rem, %after_char : i64, i64, index
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // center/ljust/rjust core: %place 0 = pad left (rjust), 1 = pad right
  // (ljust), 2 = center with CPython's left = marg/2 + (marg & width & 1).
  func.func private @__ly_unicode_pad(%header: memref<2xi64>, %bytes: memref<?xi8>, %target: i64, %fill_header: memref<2xi64>, %fill_bytes: memref<?xi8>, %place: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %fill_n = func.call @__ly_unicode_count(%fill_header, %fill_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %single = arith.cmpi eq, %fill_n, %one : i64
    %true_bit = arith.constant true
    %bad = arith.xori %single, %true_bit : i1
    scf.if %bad {
      %class_id = arith.constant 52 : i64
      %length = arith.constant 53 : i64
      %static = memref.get_global @__ly_unicode_msg_bad_fill : memref<53xi8>
      %message = memref.cast %static : memref<53xi8> to memref<?xi8>
      func.call @__ly_unicode_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %len_index_bound = arith.index_cast %len : i64 to index
    %no_pad = arith.cmpi sle, %target, %len : i64
    %result:2 = scf.if %no_pad -> (memref<2xi64>, memref<?xi8>) {
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    } else {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %fill_width = func.call @__ly_unicode_width(%fill_header) : (memref<2xi64>) -> i64
      %fill_cp = func.call @__ly_unicode_get(%fill_bytes, %fill_width, %c0) : (memref<?xi8>, i64, index) -> i64
      %self_max = scf.for %i = %c0 to %len_index_bound step %c1 iter_args(%acc = %zero) -> (i64) {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %bigger = arith.cmpi ugt, %cp, %acc : i64
        %next = arith.select %bigger, %cp, %acc : i64
        scf.yield %next : i64
      }
      %maxcp = arith.maxui %self_max, %fill_cp : i64
      %out_width = func.call @__ly_unicode_width_for(%maxcp) : (i64) -> i64
      %out_header, %out_bytes = func.call @__ly_unicode_alloc(%target, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %margin = arith.subi %target, %len : i64
      %half = arith.divsi %margin, %two : i64
      %mw = arith.andi %margin, %target : i64
      %mw1 = arith.andi %mw, %one : i64
      %center_left = arith.addi %half, %mw1 : i64
      %is_left = arith.cmpi eq, %place, %zero : i64
      %is_right = arith.cmpi eq, %place, %one : i64
      %center_or = arith.select %is_right, %zero, %center_left : i64
      %left = arith.select %is_left, %margin, %center_or : i64
      %left_index = arith.index_cast %left : i64 to index
      %target_index = arith.index_cast %target : i64 to index
      scf.for %i = %c0 to %target_index step %c1 {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %i, %fill_cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
      scf.for %i = %c0 to %len_index_bound step %c1 {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %dst = arith.addi %left_index, %i : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
      scf.yield %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RJust(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %target: i64, %fill_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = " "}, %fill_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rjust", ly.runtime.result_contract = "builtins.str"} {
    %place = arith.constant 0 : i64
    %result:2 = func.call @__ly_unicode_pad(%header, %bytes, %target, %fill_header, %fill_bytes, %place) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_LJust(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %target: i64, %fill_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = " "}, %fill_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "ljust", ly.runtime.result_contract = "builtins.str"} {
    %place = arith.constant 1 : i64
    %result:2 = func.call @__ly_unicode_pad(%header, %bytes, %target, %fill_header, %fill_bytes, %place) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Center(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %target: i64, %fill_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = " "}, %fill_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "center", ly.runtime.result_contract = "builtins.str"} {
    %place = arith.constant 2 : i64
    %result:2 = func.call @__ly_unicode_pad(%header, %bytes, %target, %fill_header, %fill_bytes, %place) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // str.zfill: zero-pad on the left, keeping a leading sign in front.
  func.func @LyUnicode_ZFill(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %target: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "zfill", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %ascii_zero = arith.constant 48 : i64
    %plus = arith.constant 43 : i64
    %minus = arith.constant 45 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %no_pad = arith.cmpi sle, %target, %len : i64
    %result:2 = scf.if %no_pad -> (memref<2xi64>, memref<?xi8>) {
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    } else {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %len_index = arith.index_cast %len : i64 to index
      %self_max = scf.for %i = %c0 to %len_index step %c1 iter_args(%acc = %zero) -> (i64) {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %bigger = arith.cmpi ugt, %cp, %acc : i64
        %next = arith.select %bigger, %cp, %acc : i64
        scf.yield %next : i64
      }
      %maxcp = arith.maxui %self_max, %ascii_zero : i64
      %out_width = func.call @__ly_unicode_width_for(%maxcp) : (i64) -> i64
      %out_header, %out_bytes = func.call @__ly_unicode_alloc(%target, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %has_any = arith.cmpi sgt, %len, %zero : i64
      %first = scf.if %has_any -> (i64) {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %c0) : (memref<?xi8>, i64, index) -> i64
        scf.yield %cp : i64
      } else {
        scf.yield %zero : i64
      }
      %is_plus = arith.cmpi eq, %first, %plus : i64
      %is_minus = arith.cmpi eq, %first, %minus : i64
      %signed = arith.ori %is_plus, %is_minus : i1
      %sign_len = arith.select %signed, %c1, %c0 : index
      scf.if %signed {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %c0, %first) : (memref<?xi8>, i64, index, i64) -> ()
      }
      %margin = arith.subi %target, %len : i64
      %margin_index = arith.index_cast %margin : i64 to index
      %zeros_end = arith.addi %sign_len, %margin_index : index
      scf.for %i = %sign_len to %zeros_end step %c1 {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %i, %ascii_zero) : (memref<?xi8>, i64, index, i64) -> ()
      }
      scf.for %i = %sign_len to %len_index step %c1 {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %dst = arith.addi %margin_index, %i : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
      scf.yield %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // str.expandtabs: a tab advances to the next multiple of tabsize within
  // the current line (\n and \r reset the column); tabsize <= 0 deletes
  // tabs, matching CPython.
  func.func @LyUnicode_ExpandTabs(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %tabsize: i64 {ly.runtime.default_i64 = 8 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "expandtabs", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %tab = arith.constant 9 : i64
    %nl = arith.constant 10 : i64
    %cr = arith.constant 13 : i64
    %space = arith.constant 32 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %positive = arith.cmpi sgt, %tabsize, %zero : i64

    %measure:3 = scf.for %i = %c0 to %len_index step %c1 iter_args(%total = %zero, %col = %zero, %maxcp = %zero) -> (i64, i64, i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_tab = arith.cmpi eq, %cp, %tab : i64
      %step:3 = scf.if %is_tab -> (i64, i64, i64) {
        %rem = scf.if %positive -> (i64) {
          %m = arith.remsi %col, %tabsize : i64
          %incr = arith.subi %tabsize, %m : i64
          scf.yield %incr : i64
        } else {
          scf.yield %zero : i64
        }
        %fill_max = arith.cmpi sgt, %rem, %zero : i64
        %contrib_max = arith.select %fill_max, %space, %zero : i64
        %next_col = arith.addi %col, %rem : i64
        scf.yield %rem, %next_col, %contrib_max : i64, i64, i64
      } else {
        %is_nl = arith.cmpi eq, %cp, %nl : i64
        %is_cr = arith.cmpi eq, %cp, %cr : i64
        %resets = arith.ori %is_nl, %is_cr : i1
        %bumped = arith.addi %col, %one : i64
        %next_col = arith.select %resets, %zero, %bumped : i64
        scf.yield %one, %next_col, %cp : i64, i64, i64
      }
      %next_total = arith.addi %total, %step#0 : i64
      %next_max = arith.maxui %maxcp, %step#2 : i64
      scf.yield %next_total, %step#1, %next_max : i64, i64, i64
    }

    %out_width = func.call @__ly_unicode_width_for(%measure#2) : (i64) -> i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%measure#0, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)

    scf.for %i = %c0 to %len_index step %c1 iter_args(%pos = %c0, %col = %zero) -> (index, i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_tab = arith.cmpi eq, %cp, %tab : i64
      %step:2 = scf.if %is_tab -> (index, i64) {
        %rem = scf.if %positive -> (i64) {
          %m = arith.remsi %col, %tabsize : i64
          %incr = arith.subi %tabsize, %m : i64
          scf.yield %incr : i64
        } else {
          scf.yield %zero : i64
        }
        %rem_index = arith.index_cast %rem : i64 to index
        scf.for %j = %c0 to %rem_index step %c1 {
          %dst = arith.addi %pos, %j : index
          func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %space) : (memref<?xi8>, i64, index, i64) -> ()
        }
        %next_pos = arith.addi %pos, %rem_index : index
        %next_col = arith.addi %col, %rem : i64
        scf.yield %next_pos, %next_col : index, i64
      } else {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        %next_pos = arith.addi %pos, %c1 : index
        %is_nl = arith.cmpi eq, %cp, %nl : i64
        %is_cr = arith.cmpi eq, %cp, %cr : i64
        %resets = arith.ori %is_nl, %is_cr : i1
        %bumped = arith.addi %col, %one : i64
        %next_col = arith.select %resets, %zero, %bumped : i64
        scf.yield %next_pos, %next_col : index, i64
      }
      scf.yield %step#0, %step#1 : index, i64
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // str * int (left str only; negative repeats give the empty string).
  func.func @LyUnicode_Mul(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %repeat: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__mul__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %n = arith.maxsi %repeat, %zero : i64
    %total = arith.muli %len, %n : i64
    %is_empty = arith.cmpi eq, %total, %zero : i64
    %out_width = arith.select %is_empty, %one, %width : i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%total, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %len_index = arith.index_cast %len : i64 to index
    %n_index = arith.index_cast %n : i64 to index
    scf.for %k = %c0 to %n_index step %c1 {
      %base = arith.muli %k, %len_index : index
      scf.for %i = %c0 to %len_index step %c1 {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %dst = arith.addi %base, %i : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }


  // ===== str splitting / joining (Wave 1) =====

  // "empty separator"
  memref.global "private" constant @__ly_unicode_msg_empty_separator : memref<15xi8> = dense<[101, 109, 112, 116, 121, 32, 115, 101, 112, 97, 114, 97, 116, 111, 114]>

  // Pack an owned str into payload box slot %slot. The layout mirrors
  // objectPayloadHandleWords (BoxLayout.h): [0] refcount word, [1] class id,
  // [2] entity root, [3] value count, [4,9) physical pointers, [9,14)
  // physical sizes, [14] owned flag. The reference transfers to the
  // container; the deallocator releases it through the boxed-release hook.
  func.func private @__ly_unicode_store_item(%items: memref<?xi64>, %slot: i64, %eh: memref<2xi64> {ly.ownership.object_header}, %eb: memref<?xi8>) attributes {ly.ownership.transfer_args = [2]} {
    %c0 = arith.constant 0 : index
    %handle_words = arith.constant 16 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %str_class = arith.constant 4 : i64
    %base_i64 = arith.muli %slot, %handle_words : i64
    %base = arith.index_cast %base_i64 : i64 to index
    %hdr_ptr_index = memref.extract_aligned_pointer_as_index %eh : memref<2xi64> -> index
    %hdr_ptr = arith.index_cast %hdr_ptr_index : index to i64
    %bytes_ptr_index = memref.extract_aligned_pointer_as_index %eb : memref<?xi8> -> index
    %bytes_ptr = arith.index_cast %bytes_ptr_index : index to i64
    %dim = memref.dim %eb, %c0 : memref<?xi8>
    %byte_len = arith.index_cast %dim : index to i64
    %w0 = arith.constant 0 : index
    %w1 = arith.constant 1 : index
    %w2 = arith.constant 2 : index
    %w3 = arith.constant 3 : index
    %w4 = arith.constant 4 : index
    %w5 = arith.constant 5 : index
    %w6 = arith.constant 6 : index
    %w7 = arith.constant 7 : index
    %w8 = arith.constant 8 : index
    %w9 = arith.constant 9 : index
    %w10 = arith.constant 10 : index
    %w11 = arith.constant 11 : index
    %w12 = arith.constant 12 : index
    %w13 = arith.constant 13 : index
    %w14 = arith.constant 14 : index
    %w15 = arith.constant 15 : index
    %s0 = arith.addi %base, %w0 : index
    %s1 = arith.addi %base, %w1 : index
    %s2 = arith.addi %base, %w2 : index
    %s3 = arith.addi %base, %w3 : index
    %s4 = arith.addi %base, %w4 : index
    %s5 = arith.addi %base, %w5 : index
    %s6 = arith.addi %base, %w6 : index
    %s7 = arith.addi %base, %w7 : index
    %s8 = arith.addi %base, %w8 : index
    %s9 = arith.addi %base, %w9 : index
    %s10 = arith.addi %base, %w10 : index
    %s11 = arith.addi %base, %w11 : index
    %s12 = arith.addi %base, %w12 : index
    %s13 = arith.addi %base, %w13 : index
    %s14 = arith.addi %base, %w14 : index
    %s15 = arith.addi %base, %w15 : index
    memref.store %one, %items[%s0] : memref<?xi64>
    memref.store %str_class, %items[%s1] : memref<?xi64>
    memref.store %hdr_ptr, %items[%s2] : memref<?xi64>
    memref.store %two, %items[%s3] : memref<?xi64>
    memref.store %hdr_ptr, %items[%s4] : memref<?xi64>
    memref.store %bytes_ptr, %items[%s5] : memref<?xi64>
    memref.store %zero, %items[%s6] : memref<?xi64>
    memref.store %zero, %items[%s7] : memref<?xi64>
    memref.store %zero, %items[%s8] : memref<?xi64>
    memref.store %two, %items[%s9] : memref<?xi64>
    memref.store %byte_len, %items[%s10] : memref<?xi64>
    memref.store %zero, %items[%s11] : memref<?xi64>
    memref.store %zero, %items[%s12] : memref<?xi64>
    memref.store %zero, %items[%s13] : memref<?xi64>
    memref.store %one, %items[%s14] : memref<?xi64>
    memref.store %zero, %items[%s15] : memref<?xi64>
    func.return
  }

  // Store a freshly sliced [start, end) segment into box slot %slot.
  func.func private @__ly_unicode_store_slice(%items: memref<?xi64>, %slot: i64, %header: memref<2xi64>, %bytes: memref<?xi8>, %start: index, %end: index) {
    %piece:2 = func.call @__ly_unicode_slice(%header, %bytes, %start, %end) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_unicode_store_item(%items, %slot, %piece#0, %piece#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // str.split(sep[, maxsplit]) -- explicit separator form. The empty
  // separator raises like CPython.
  func.func @LyUnicode_Split(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %false_bit = arith.constant false
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %sep_n = func.call @__ly_unicode_count(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %sep_width = func.call @__ly_unicode_width(%sep_header) : (memref<2xi64>) -> i64
    %sep_empty = arith.cmpi eq, %sep_n, %zero : i64
    scf.if %sep_empty {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 15 : i64
      %static = memref.get_global @__ly_unicode_msg_empty_separator : memref<15xi8>
      %message = memref.cast %static : memref<15xi8> to memref<?xi8>
      func.call @__ly_unicode_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }

    // Pass 1: number of separators actually used.
    %true_split1 = arith.constant true
    %count:3 = scf.while (%pos = %zero, %used = %zero, %go = %true_split1) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %pos, %used, %go : i64, i64, i1
    } do {
    ^bb0(%pos: i64, %used: i64, %go: i1):
      %budget_left = arith.cmpi ne, %used, %maxsplit : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %pos, %len, %sep_n, %false_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      %next_pos_hit = arith.addi %hit, %sep_n : i64
      %next_pos = arith.select %matched, %next_pos_hit, %pos : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_used = arith.addi %used, %bump : i64
      scf.yield %next_pos, %next_used, %matched : i64, i64, i1
    }

    %segments = arith.addi %count#1, %one : i64
    %list:3 = func.call @LyList_FromLength(%segments) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)

    // Pass 2: emit the segments.
    %true_split2 = arith.constant true
    %emit:3 = scf.while (%pos = %zero, %slot = %zero, %go = %true_split2) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %pos, %slot, %go : i64, i64, i1
    } do {
    ^bb0(%pos: i64, %slot: i64, %go: i1):
      %remaining = arith.subi %segments, %one : i64
      %budget_left = arith.cmpi slt, %slot, %remaining : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %pos, %len, %sep_n, %false_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      scf.if %matched {
        %pos_index = arith.index_cast %pos : i64 to index
        %hit_index = arith.index_cast %hit : i64 to index
        func.call @__ly_unicode_store_slice(%list#2, %slot, %header, %bytes, %pos_index, %hit_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      %next_pos_hit = arith.addi %hit, %sep_n : i64
      %next_pos = arith.select %matched, %next_pos_hit, %pos : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_slot = arith.addi %slot, %bump : i64
      scf.yield %next_pos, %next_slot, %matched : i64, i64, i1
    }
    %tail_start = arith.index_cast %emit#0 : i64 to index
    %len_index = arith.index_cast %len : i64 to index
    func.call @__ly_unicode_store_slice(%list#2, %emit#1, %header, %bytes, %tail_start, %len_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
    func.return %list#0, %list#1, %list#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // str.rsplit(sep[, maxsplit]): scans right-to-left (a left scan with a
  // skip count is wrong when candidate matches overlap: "aaa".rsplit("aa")
  // is ['a', ''], not ['', 'a']), filling slots from the back.
  func.func @LyUnicode_RSplit(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rsplit", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %true_bit = arith.constant true
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %sep_n = func.call @__ly_unicode_count(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %sep_width = func.call @__ly_unicode_width(%sep_header) : (memref<2xi64>) -> i64
    %sep_empty = arith.cmpi eq, %sep_n, %zero : i64
    scf.if %sep_empty {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 15 : i64
      %static = memref.get_global @__ly_unicode_msg_empty_separator : memref<15xi8>
      %message = memref.cast %static : memref<15xi8> to memref<?xi8>
      func.call @__ly_unicode_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }

    %true_rsplit1 = arith.constant true
    %count:3 = scf.while (%end = %len, %used = %zero, %go = %true_rsplit1) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %end, %used, %go : i64, i64, i1
    } do {
    ^bb0(%end: i64, %used: i64, %go: i1):
      %budget_left = arith.cmpi ne, %used, %maxsplit : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %zero, %end, %sep_n, %true_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      %next_end = arith.select %matched, %hit, %end : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_used = arith.addi %used, %bump : i64
      scf.yield %next_end, %next_used, %matched : i64, i64, i1
    }

    %segments = arith.addi %count#1, %one : i64
    %list:3 = func.call @LyList_FromLength(%segments) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)

    %true_rsplit2 = arith.constant true
    %emit:3 = scf.while (%end = %len, %slot = %count#1, %go = %true_rsplit2) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %end, %slot, %go : i64, i64, i1
    } do {
    ^bb0(%end: i64, %slot: i64, %go: i1):
      %budget_left = arith.cmpi sgt, %slot, %zero : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %zero, %end, %sep_n, %true_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      scf.if %matched {
        %seg_start = arith.addi %hit, %sep_n : i64
        %seg_start_index = arith.index_cast %seg_start : i64 to index
        %end_index = arith.index_cast %end : i64 to index
        func.call @__ly_unicode_store_slice(%list#2, %slot, %header, %bytes, %seg_start_index, %end_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      %next_end = arith.select %matched, %hit, %end : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_slot = arith.subi %slot, %bump : i64
      scf.yield %next_end, %next_slot, %matched : i64, i64, i1
    }
    %c0 = arith.constant 0 : index
    %head_end = arith.index_cast %emit#0 : i64 to index
    func.call @__ly_unicode_store_slice(%list#2, %zero, %header, %bytes, %c0, %head_end) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
    func.return %list#0, %list#1, %list#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // Whitespace split (str.split() / str.rsplit() with no separator): runs
  // of Unicode whitespace delimit; leading/trailing whitespace produces no
  // empty segments. Unlimited maxsplit makes the two directions agree, so
  // one forward implementation serves both names.
  func.func private @__ly_unicode_split_ws_core(%header: memref<2xi64>, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %len_index = arith.index_cast %len : i64 to index

    // Pass 1: segment count = number of non-space runs.
    %false_ws1 = arith.constant false
    %count:2 = scf.for %i = %c0 to %len_index step %c1 iter_args(%segs = %zero, %in_run = %false_ws1) -> (i64, i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_space = func.call @__ly_unicode_cp_is_space(%cp) : (i64) -> i1
      %true_w = arith.constant true
      %non_space = arith.xori %is_space, %true_w : i1
      %not_in_run = arith.xori %in_run, %true_w : i1
      %starts = arith.andi %non_space, %not_in_run : i1
      %bump = arith.select %starts, %one, %zero : i64
      %next_segs = arith.addi %segs, %bump : i64
      scf.yield %next_segs, %non_space : i64, i1
    }

    %list:3 = func.call @LyList_FromLength(%count#0) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)

    // Pass 2: emit each run.
    %false_ws2 = arith.constant false
    scf.for %i = %c0 to %len_index step %c1 iter_args(%slot = %zero, %run_start = %len_index, %in_run = %false_ws2) -> (i64, index, i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_space = func.call @__ly_unicode_cp_is_space(%cp) : (i64) -> i1
      %true_w = arith.constant true
      %non_space = arith.xori %is_space, %true_w : i1
      %not_in_run = arith.xori %in_run, %true_w : i1
      %starts = arith.andi %non_space, %not_in_run : i1
      %ends = arith.andi %is_space, %in_run : i1
      %new_start = arith.select %starts, %i, %run_start : index
      scf.if %ends {
        func.call @__ly_unicode_store_slice(%list#2, %slot, %header, %bytes, %run_start, %i) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      %bump = arith.select %ends, %one, %zero : i64
      %next_slot = arith.addi %slot, %bump : i64
      %last = arith.subi %len_index, %c1 : index
      %is_last = arith.cmpi eq, %i, %last : index
      %closes = arith.andi %is_last, %non_space : i1
      scf.if %closes {
        func.call @__ly_unicode_store_slice(%list#2, %next_slot, %header, %bytes, %new_start, %len_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      scf.yield %next_slot, %new_start, %non_space : i64, index, i1
    }
    func.return %list#0, %list#1, %list#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyUnicode_SplitWS(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %result:3 = func.call @__ly_unicode_split_ws_core(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyUnicode_RSplitWS(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rsplit", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %result:3 = func.call @__ly_unicode_split_ws_core(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // Unicode line boundaries per CPython str.splitlines: \n \v \f \r \x1c
  // \x1d \x1e \x85 \u2028 \u2029, with \r\n as one boundary.
  func.func private @__ly_unicode_cp_is_linebreak(%cp: i64) -> i1 {
    %nl = arith.constant 10 : i64
    %vt = arith.constant 11 : i64
    %ff = arith.constant 12 : i64
    %cr = arith.constant 13 : i64
    %fs = arith.constant 28 : i64
    %gs = arith.constant 29 : i64
    %rs = arith.constant 30 : i64
    %nel = arith.constant 133 : i64
    %ls = arith.constant 8232 : i64
    %ps = arith.constant 8233 : i64
    %in_c0 = arith.cmpi sge, %cp, %nl : i64
    %le_cr = arith.cmpi sle, %cp, %cr : i64
    %ctl = arith.andi %in_c0, %le_cr : i1
    %ge_fs = arith.cmpi sge, %cp, %fs : i64
    %le_rs = arith.cmpi sle, %cp, %rs : i64
    %seps = arith.andi %ge_fs, %le_rs : i1
    %is_nel = arith.cmpi eq, %cp, %nel : i64
    %is_ls = arith.cmpi eq, %cp, %ls : i64
    %is_ps = arith.cmpi eq, %cp, %ps : i64
    %a = arith.ori %ctl, %seps : i1
    %b = arith.ori %is_nel, %is_ls : i1
    %c = arith.ori %b, %is_ps : i1
    %result = arith.ori %a, %c : i1
    func.return %result : i1
  }

  func.func private @__ly_unicode_splitlines_core(%header: memref<2xi64>, %bytes: memref<?xi8>, %keepends: i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0]} {
    %c1i = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %cr = arith.constant 13 : i64
    %nl = arith.constant 10 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64

    // Pass 1: line count.
    %count:2 = scf.while (%i = %zero, %lines = %zero) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi slt, %i, %len : i64
      scf.condition(%more) %i, %lines : i64, i64
    } do {
    ^bb0(%i: i64, %lines: i64):
      // Scan to the end of this line's content.
      %true_lines1 = arith.constant true
      %scan:2 = scf.while (%j = %i, %go = %true_lines1) : (i64, i1) -> (i64, i1) {
        %in_bounds = arith.cmpi slt, %j, %len : i64
        %continue = arith.andi %in_bounds, %go : i1
        scf.condition(%continue) %j, %go : i64, i1
      } do {
      ^bb1(%j: i64, %go: i1):
        %j_index = arith.index_cast %j : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %j_index) : (memref<?xi8>, i64, index) -> i64
        %brk = func.call @__ly_unicode_cp_is_linebreak(%cp) : (i64) -> i1
        %true_x = arith.constant true
        %not_brk = arith.xori %brk, %true_x : i1
        %next = arith.addi %j, %one : i64
        %sel = arith.select %brk, %j, %next : i64
        scf.yield %sel, %not_brk : i64, i1
      }
      %ended = arith.cmpi slt, %scan#0, %len : i64
      %skip = scf.if %ended -> (i64) {
        %eol_index = arith.index_cast %scan#0 : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %eol_index) : (memref<?xi8>, i64, index) -> i64
        %is_cr = arith.cmpi eq, %cp, %cr : i64
        %next_pos = arith.addi %scan#0, %one : i64
        %has_next = arith.cmpi slt, %next_pos, %len : i64
        %pair = arith.andi %is_cr, %has_next : i1
        %crlf = scf.if %pair -> (i1) {
          %next_index = arith.index_cast %next_pos : i64 to index
          %cp2 = func.call @__ly_unicode_get(%bytes, %width, %next_index) : (memref<?xi8>, i64, index) -> i64
          %is_nl = arith.cmpi eq, %cp2, %nl : i64
          scf.yield %is_nl : i1
        } else {
          %false_x = arith.constant false
          scf.yield %false_x : i1
        }
        %stride = arith.select %crlf, %two, %one : i64
        scf.yield %stride : i64
      } else {
        scf.yield %zero : i64
      }
      %next_i = arith.addi %scan#0, %skip : i64
      %next_lines = arith.addi %lines, %one : i64
      scf.yield %next_i, %next_lines : i64, i64
    }

    %list:3 = func.call @LyList_FromLength(%count#1) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)

    // Pass 2: emit each line ([start, eol) or [start, eol+break)).
    %emitted:2 = scf.while (%i = %zero, %slot = %zero) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi slt, %i, %len : i64
      scf.condition(%more) %i, %slot : i64, i64
    } do {
    ^bb0(%i: i64, %slot: i64):
      %true_lines2 = arith.constant true
      %scan:2 = scf.while (%j = %i, %go = %true_lines2) : (i64, i1) -> (i64, i1) {
        %in_bounds = arith.cmpi slt, %j, %len : i64
        %continue = arith.andi %in_bounds, %go : i1
        scf.condition(%continue) %j, %go : i64, i1
      } do {
      ^bb1(%j: i64, %go: i1):
        %j_index = arith.index_cast %j : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %j_index) : (memref<?xi8>, i64, index) -> i64
        %brk = func.call @__ly_unicode_cp_is_linebreak(%cp) : (i64) -> i1
        %true_x = arith.constant true
        %not_brk = arith.xori %brk, %true_x : i1
        %next = arith.addi %j, %one : i64
        %sel = arith.select %brk, %j, %next : i64
        scf.yield %sel, %not_brk : i64, i1
      }
      %ended = arith.cmpi slt, %scan#0, %len : i64
      %skip = scf.if %ended -> (i64) {
        %eol_index = arith.index_cast %scan#0 : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %eol_index) : (memref<?xi8>, i64, index) -> i64
        %is_cr = arith.cmpi eq, %cp, %cr : i64
        %next_pos = arith.addi %scan#0, %one : i64
        %has_next = arith.cmpi slt, %next_pos, %len : i64
        %pair = arith.andi %is_cr, %has_next : i1
        %crlf = scf.if %pair -> (i1) {
          %next_index = arith.index_cast %next_pos : i64 to index
          %cp2 = func.call @__ly_unicode_get(%bytes, %width, %next_index) : (memref<?xi8>, i64, index) -> i64
          %is_nl = arith.cmpi eq, %cp2, %nl : i64
          scf.yield %is_nl : i1
        } else {
          %false_x = arith.constant false
          scf.yield %false_x : i1
        }
        %stride = arith.select %crlf, %two, %one : i64
        scf.yield %stride : i64
      } else {
        scf.yield %zero : i64
      }
      %with_break = arith.addi %scan#0, %skip : i64
      %seg_end = arith.select %keepends, %with_break, %scan#0 : i64
      %i_index = arith.index_cast %i : i64 to index
      %seg_end_index = arith.index_cast %seg_end : i64 to index
      func.call @__ly_unicode_store_slice(%list#2, %slot, %header, %bytes, %i_index, %seg_end_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      %next_i = arith.addi %scan#0, %skip : i64
      %next_slot = arith.addi %slot, %one : i64
      scf.yield %next_i, %next_slot : i64, i64
    }
    func.return %list#0, %list#1, %list#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyUnicode_SplitLines(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "splitlines", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %false_bit = arith.constant false
    %result:3 = func.call @__ly_unicode_splitlines_core(%header, %bytes, %false_bit) : (memref<2xi64>, memref<?xi8>, i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyUnicode_SplitLinesKeep(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %keepends: i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "splitlines", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %result:3 = func.call @__ly_unicode_splitlines_core(%header, %bytes, %keepends) : (memref<2xi64>, memref<?xi8>, i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // str.partition / str.rpartition: a 3-tuple of fresh strs (the separator
  // element is a retained reference to the operand, transferred to the
  // tuple).
  func.func private @__ly_unicode_partition_core(%header: memref<2xi64>, %bytes: memref<?xi8>, %sep_header: memref<2xi64>, %sep_bytes: memref<?xi8>, %reverse: i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %sep_n = func.call @__ly_unicode_count(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %sep_width = func.call @__ly_unicode_width(%sep_header) : (memref<2xi64>) -> i64
    %sep_empty = arith.cmpi eq, %sep_n, %zero : i64
    scf.if %sep_empty {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 15 : i64
      %static = memref.get_global @__ly_unicode_msg_empty_separator : memref<15xi8>
      %message = memref.cast %static : memref<15xi8> to memref<?xi8>
      func.call @__ly_unicode_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %tuple:3 = func.call @LyTuple_FromLength(%three) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    %hit = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %zero, %len, %sep_n, %reverse) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
    %matched = arith.cmpi sge, %hit, %zero : i64
    scf.if %matched {
      %hit_index = arith.index_cast %hit : i64 to index
      %after = arith.addi %hit, %sep_n : i64
      %after_index = arith.index_cast %after : i64 to index
      %len_index = arith.index_cast %len : i64 to index
      func.call @__ly_unicode_store_slice(%tuple#2, %zero, %header, %bytes, %c0, %hit_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      %sep_copy:2 = func.call @__ly_unicode_retain_self(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @__ly_unicode_store_item(%tuple#2, %one, %sep_copy#0, %sep_copy#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      func.call @__ly_unicode_store_slice(%tuple#2, %two, %header, %bytes, %after_index, %len_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
    } else {
      %len_index = arith.index_cast %len : i64 to index
      %whole:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      %empty1:2 = func.call @__ly_unicode_alloc(%zero, %one) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %empty2:2 = func.call @__ly_unicode_alloc(%zero, %one) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %whole_slot = arith.select %reverse, %two, %zero : i64
      %e1_slot = arith.select %reverse, %zero, %one : i64
      %e2_slot = arith.select %reverse, %one, %two : i64
      func.call @__ly_unicode_store_item(%tuple#2, %whole_slot, %whole#0, %whole#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      func.call @__ly_unicode_store_item(%tuple#2, %e1_slot, %empty1#0, %empty1#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      func.call @__ly_unicode_store_item(%tuple#2, %e2_slot, %empty2#0, %empty2#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
    }
    func.return %tuple#0, %tuple#1, %tuple#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyUnicode_Partition(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "partition", ly.runtime.result_contract = "builtins.tuple", ly.runtime.element_contract = "builtins.str"} {
    %false_bit = arith.constant false
    %result:3 = func.call @__ly_unicode_partition_core(%header, %bytes, %sep_header, %sep_bytes, %false_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyUnicode_RPartition(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rpartition", ly.runtime.result_contract = "builtins.tuple", ly.runtime.element_contract = "builtins.str"} {
    %true_bit = arith.constant true
    %result:3 = func.call @__ly_unicode_partition_core(%header, %bytes, %sep_header, %sep_bytes, %true_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i1) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // Code point %i of a boxed element's code-unit buffer, addressed by the
  // raw pointer words a payload box carries (native container elements have
  // no memref views to reconstruct in manifest code).
  func.func private @__ly_unicode_get_raw(%ptr: i64, %width: i64, %i: index) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %i_i64 = arith.index_cast %i : index to i64
    %off = arith.muli %i_i64, %width : i64
    %addr = arith.addi %ptr, %off : i64
    %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    %is1 = arith.cmpi eq, %width, %one : i64
    %cp = scf.if %is1 -> (i64) {
      %b = llvm.load %llptr : !llvm.ptr -> i8
      %v = arith.extui %b : i8 to i64
      scf.yield %v : i64
    } else {
      %is2 = arith.cmpi eq, %width, %two : i64
      %inner = scf.if %is2 -> (i64) {
        %h = llvm.load %llptr : !llvm.ptr -> i16
        %v = arith.extui %h : i16 to i64
        scf.yield %v : i64
      } else {
        %w = llvm.load %llptr : !llvm.ptr -> i32
        %v = arith.extui %w : i32 to i64
        scf.yield %v : i64
      }
      scf.yield %inner : i64
    }
    func.return %cp : i64
  }

  // (header ptr, code-unit ptr, byte length) words of element %slot.
  func.func private @__ly_unicode_item_words(%items: memref<?xi64>, %slot: index) -> (i64, i64, i64) {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index
    %c16 = arith.constant 16 : index
    %base = arith.muli %slot, %c16 : index
    %hdr_slot = arith.addi %base, %c2 : index
    %ptr_slot = arith.addi %base, %c5 : index
    %len_slot = arith.addi %base, %c10 : index
    %hdr = memref.load %items[%hdr_slot] : memref<?xi64>
    %ptr = memref.load %items[%ptr_slot] : memref<?xi64>
    %blen = memref.load %items[%len_slot] : memref<?xi64>
    func.return %hdr, %ptr, %blen : i64, i64, i64
  }

  // Character width of a boxed str element (the width word at header+16).
  func.func private @__ly_unicode_raw_width(%hdr_ptr: i64) -> i64 {
    %sixteen = arith.constant 16 : i64
    %addr = arith.addi %hdr_ptr, %sixteen : i64
    %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    %width = llvm.load %llptr : !llvm.ptr -> i64
    func.return %width : i64
  }

  // str.join over a runtime list/tuple of strs (identical physical layout).
  func.func @LyUnicode_Join(%sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>, %seq_header: memref<2xi64> {ly.ownership.object_header}, %seq_meta: memref<2xi64>, %seq_items: memref<?xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "join", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sep_n = func.call @__ly_unicode_count(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %sep_width = func.call @__ly_unicode_width(%sep_header) : (memref<2xi64>) -> i64
    %n = memref.load %seq_meta[%c0] : memref<2xi64>
    %n_index = arith.index_cast %n : i64 to index

    // Pass 1: total code points and widest element (operands are canonical,
    // so widths bound the widest code point exactly).
    %measure:2 = scf.for %k = %c0 to %n_index step %c1 iter_args(%total = %zero, %wmax = %one) -> (i64, i64) {
      %hdr, %ptr, %blen = func.call @__ly_unicode_item_words(%seq_items, %k) : (memref<?xi64>, index) -> (i64, i64, i64)
      %w = func.call @__ly_unicode_raw_width(%hdr) : (i64) -> i64
      %count = arith.divsi %blen, %w : i64
      %next_total = arith.addi %total, %count : i64
      %wider = arith.cmpi sgt, %w, %wmax : i64
      %next_wmax = arith.select %wider, %w, %wmax : i64
      scf.yield %next_total, %next_wmax : i64, i64
    }
    %has_seps = arith.cmpi sgt, %n, %one : i64
    %sep_uses_i64 = arith.subi %n, %one : i64
    %sep_uses = arith.select %has_seps, %sep_uses_i64, %zero : i64
    %sep_total = arith.muli %sep_uses, %sep_n : i64
    %total = arith.addi %measure#0, %sep_total : i64
    %true_j = arith.constant true
    %sep_matters = arith.andi %has_seps, %true_j : i1
    %wmax_with_sep = arith.maxsi %measure#1, %sep_width : i64
    %out_width = arith.select %sep_matters, %wmax_with_sep, %measure#1 : i64

    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%total, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)

    // Pass 2: elements with separators between them.
    scf.for %k = %c0 to %n_index step %c1 iter_args(%pos = %c0) -> (index) {
      %is_first = arith.cmpi eq, %k, %c0 : index
      %true_k = arith.constant true
      %needs_sep = arith.xori %is_first, %true_k : i1
      %after_sep = scf.if %needs_sep -> (index) {
        %sep_n_index = arith.index_cast %sep_n : i64 to index
        scf.for %j = %c0 to %sep_n_index step %c1 {
          %cp = func.call @__ly_unicode_get(%sep_bytes, %sep_width, %j) : (memref<?xi8>, i64, index) -> i64
          %dst = arith.addi %pos, %j : index
          func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        }
        %advanced = arith.addi %pos, %sep_n_index : index
        scf.yield %advanced : index
      } else {
        scf.yield %pos : index
      }
      %hdr, %ptr, %blen = func.call @__ly_unicode_item_words(%seq_items, %k) : (memref<?xi64>, index) -> (i64, i64, i64)
      %w = func.call @__ly_unicode_raw_width(%hdr) : (i64) -> i64
      %count = arith.divsi %blen, %w : i64
      %count_index = arith.index_cast %count : i64 to index
      scf.for %j = %c0 to %count_index step %c1 {
        %cp = func.call @__ly_unicode_get_raw(%ptr, %w, %j) : (i64, i64, index) -> i64
        %dst = arith.addi %after_sep, %j : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
      %next_pos = arith.addi %after_sep, %count_index : index
      scf.yield %next_pos : index
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // UTF-8 byte length of the encoded form (encode / print paths).
  func.func private @__ly_unicode_utf8_length(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %lim1 = arith.constant 128 : i64
    %lim2 = arith.constant 2048 : i64
    %lim3 = arith.constant 65536 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %total = scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %zero) -> (i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %fits1 = arith.cmpi ult, %cp, %lim1 : i64
      %fits2 = arith.cmpi ult, %cp, %lim2 : i64
      %fits3 = arith.cmpi ult, %cp, %lim3 : i64
      %three_or_four = arith.select %fits3, %three, %four : i64
      %two_plus = arith.select %fits2, %two, %three_or_four : i64
      %contrib = arith.select %fits1, %one, %two_plus : i64
      %next = arith.addi %acc, %contrib : i64
      scf.yield %next : i64
    }
    func.return %total : i64
  }

  // Encode into a caller-provided UTF-8 buffer of exactly
  // __ly_unicode_utf8_length bytes.
  func.func private @__ly_unicode_utf8_fill(%header: memref<2xi64>, %bytes: memref<?xi8>, %out: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %six = arith.constant 6 : i64
    %twelve = arith.constant 12 : i64
    %eighteen = arith.constant 18 : i64
    %lim1 = arith.constant 128 : i64
    %lim2 = arith.constant 2048 : i64
    %lim3 = arith.constant 65536 : i64
    %cont_tag = arith.constant 128 : i64
    %payload_mask = arith.constant 63 : i64
    %lead2_tag = arith.constant 192 : i64
    %lead3_tag = arith.constant 224 : i64
    %lead4_tag = arith.constant 240 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %count_index step %c1 iter_args(%pos = %c0) -> (index) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %fits1 = arith.cmpi ult, %cp, %lim1 : i64
      %next = scf.if %fits1 -> (index) {
        %b0 = arith.trunci %cp : i64 to i8
        memref.store %b0, %out[%pos] : memref<?xi8>
        %advanced = arith.addi %pos, %c1 : index
        scf.yield %advanced : index
      } else {
        %fits2 = arith.cmpi ult, %cp, %lim2 : i64
        %inner2 = scf.if %fits2 -> (index) {
          %hi = arith.shrui %cp, %six : i64
          %b0v = arith.ori %hi, %lead2_tag : i64
          %lo = arith.andi %cp, %payload_mask : i64
          %b1v = arith.ori %lo, %cont_tag : i64
          %b0 = arith.trunci %b0v : i64 to i8
          %b1 = arith.trunci %b1v : i64 to i8
          %pos1 = arith.addi %pos, %c1 : index
          memref.store %b0, %out[%pos] : memref<?xi8>
          memref.store %b1, %out[%pos1] : memref<?xi8>
          %advanced = arith.addi %pos, %c2 : index
          scf.yield %advanced : index
        } else {
          %fits3 = arith.cmpi ult, %cp, %lim3 : i64
          %inner3 = scf.if %fits3 -> (index) {
            %hi = arith.shrui %cp, %twelve : i64
            %b0v = arith.ori %hi, %lead3_tag : i64
            %mid_raw = arith.shrui %cp, %six : i64
            %mid = arith.andi %mid_raw, %payload_mask : i64
            %b1v = arith.ori %mid, %cont_tag : i64
            %lo = arith.andi %cp, %payload_mask : i64
            %b2v = arith.ori %lo, %cont_tag : i64
            %b0 = arith.trunci %b0v : i64 to i8
            %b1 = arith.trunci %b1v : i64 to i8
            %b2 = arith.trunci %b2v : i64 to i8
            %pos1 = arith.addi %pos, %c1 : index
            %pos2 = arith.addi %pos, %c2 : index
            memref.store %b0, %out[%pos] : memref<?xi8>
            memref.store %b1, %out[%pos1] : memref<?xi8>
            memref.store %b2, %out[%pos2] : memref<?xi8>
            %advanced = arith.addi %pos, %c3 : index
            scf.yield %advanced : index
          } else {
            %hi = arith.shrui %cp, %eighteen : i64
            %b0v = arith.ori %hi, %lead4_tag : i64
            %mid1_raw = arith.shrui %cp, %twelve : i64
            %mid1 = arith.andi %mid1_raw, %payload_mask : i64
            %b1v = arith.ori %mid1, %cont_tag : i64
            %mid2_raw = arith.shrui %cp, %six : i64
            %mid2 = arith.andi %mid2_raw, %payload_mask : i64
            %b2v = arith.ori %mid2, %cont_tag : i64
            %lo = arith.andi %cp, %payload_mask : i64
            %b3v = arith.ori %lo, %cont_tag : i64
            %b0 = arith.trunci %b0v : i64 to i8
            %b1 = arith.trunci %b1v : i64 to i8
            %b2 = arith.trunci %b2v : i64 to i8
            %b3 = arith.trunci %b3v : i64 to i8
            %pos1 = arith.addi %pos, %c1 : index
            %pos2 = arith.addi %pos, %c2 : index
            %pos3 = arith.addi %pos, %c3 : index
            memref.store %b0, %out[%pos] : memref<?xi8>
            memref.store %b1, %out[%pos1] : memref<?xi8>
            memref.store %b2, %out[%pos2] : memref<?xi8>
            memref.store %b3, %out[%pos3] : memref<?xi8>
            %advanced = arith.addi %pos, %c4 : index
            scf.yield %advanced : index
          }
          scf.yield %inner3 : index
        }
        scf.yield %inner2 : index
      }
      scf.yield %next : index
    }
    func.return
  }

  memref.global "private" constant @__ly_print_end : memref<1xi8> = dense<[10]>

  // print(object) after the emitter's sep-join desugar: CPython
  // builtin_print_impl's tail with objects_length == 1, file = sys.stdout
  // (fd 1) and the default end = "\n" (Python/bltinmodule.c). The str
  // rendering of each argument stays ahead of this sink (method_sink
  // dispatch / emitter desugar) because it needs per-value evidence.
  func.func @LyUnicode_PrintLine(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.builtin = "print", ly.runtime.builtin_lowering = "method_sink", ly.runtime.builtin_method = "__repr__", ly.runtime.builtin_sink_contract = "builtins.str", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print_line", ly.runtime.result_contract = "types.NoneType"} {
    %stdout = arith.constant 1 : i32
    func.call @LyUnicode_Print(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    %end_static = memref.get_global @__ly_print_end : memref<1xi8>
    %end = memref.cast %end_static : memref<1xi8> to memref<?xi8>
    %end_length = arith.constant 1 : i64
    func.call @LyHost_WriteBytes(%stdout, %end, %end_length) : (i32, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func @LyUnicode_Print(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print"} {
    %stdout = arith.constant 1 : i32
    %length = func.call @__ly_unicode_utf8_length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %length_index = arith.index_cast %length : i64 to index
    %buffer = memref.alloc(%length_index) : memref<?xi8>
    func.call @__ly_unicode_utf8_fill(%header, %bytes, %buffer) : (memref<2xi64>, memref<?xi8>, memref<?xi8>) -> ()
    func.call @LyHost_WriteBytes(%stdout, %buffer, %length) : (i32, memref<?xi8>, i64) -> ()
    memref.dealloc %buffer : memref<?xi8>
    func.return
  }

  func.func @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %buffer = memref.alloca() : memref<21xi8>
    %zero = arith.constant 0 : i64
    %ten = arith.constant 10 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_minus = arith.constant 45 : i8
    %end = arith.constant 20 : index
    %last = arith.constant 19 : index

    %is_negative = arith.cmpi slt, %value, %zero : i64
    %negated = arith.subi %zero, %value : i64
    %abs_value = arith.select %is_negative, %negated, %value : i64
    %is_zero = arith.cmpi eq, %abs_value, %zero : i64
    cf.cond_br %is_zero, ^format_zero, ^format_digits

  ^format_zero:
    %zero_ch_i64 = arith.constant 48 : i64
    %zero_ch = arith.trunci %zero_ch_i64 : i64 to i8
    memref.store %zero_ch, %buffer[%last] : memref<21xi8>
    cf.br ^finish(%last : index)

  ^format_digits:
    %lower = arith.constant 0 : index
    %upper = arith.constant 20 : index
    %step = arith.constant 1 : index
    %result:2 = scf.for %i = %lower to %upper step %step iter_args(%n = %abs_value, %pos = %last) -> (i64, index) {
      %active = arith.cmpi ne, %n, %zero : i64
      %next:2 = scf.if %active -> (i64, index) {
        %digit = arith.remui %n, %ten : i64
        %digit_ch_i64 = arith.addi %digit, %ascii_zero : i64
        %digit_ch = arith.trunci %digit_ch_i64 : i64 to i8
        memref.store %digit_ch, %buffer[%pos] : memref<21xi8>
        %quotient = arith.divui %n, %ten : i64
        %one_index = arith.constant 1 : index
        %next_pos = arith.subi %pos, %one_index : index
        scf.yield %quotient, %next_pos : i64, index
      } else {
        scf.yield %n, %pos : i64, index
      }
      scf.yield %next#0, %next#1 : i64, index
    }
    %one_finish = arith.constant 1 : index
    %first_digit = arith.addi %result#1, %one_finish : index
    %start = scf.if %is_negative -> (index) {
      %minus_pos = arith.subi %first_digit, %one_finish : index
      memref.store %ascii_minus, %buffer[%minus_pos] : memref<21xi8>
      scf.yield %minus_pos : index
    } else {
      scf.yield %first_digit : index
    }
    cf.br ^finish(%start : index)

  ^finish(%start_index: index):
    %length_index = arith.subi %end, %start_index : index
    %length = arith.index_cast %length_index : index to i64
    %buffer_view = memref.cast %buffer : memref<21xi8> to memref<?xi8>
    %header, %bytes = func.call @LyUnicode_FromBytes(%buffer_view, %start_index, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_FromF64(%value: f64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %buffer = memref.alloca() : memref<48xi8>
    %zero_f = arith.constant 0.0 : f64
    %half_f = arith.constant 0.5 : f64
    %scale_f = arith.constant 1000000.0 : f64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %six = arith.constant 6 : i64
    %ten = arith.constant 10 : i64
    %million = arith.constant 1000000 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_minus = arith.constant 45 : i8
    %ascii_dot = arith.constant 46 : i8
    %start_loop = arith.constant 0 : index
    %frac_loop_end = arith.constant 6 : index
    %int_loop_end = arith.constant 20 : index
    %step = arith.constant 1 : index
    %end = arith.constant 48 : index
    %last = arith.constant 47 : index
    %one_index = arith.constant 1 : index

    %is_negative = arith.cmpf olt, %value, %zero_f : f64
    %negated = arith.subf %zero_f, %value : f64
    %abs_value = arith.select %is_negative, %negated, %value : i1, f64
    %raw_int = arith.fptosi %abs_value : f64 to i64
    %raw_int_float = arith.sitofp %raw_int : i64 to f64
    %fraction = arith.subf %abs_value, %raw_int_float : f64
    %scaled_fraction = arith.mulf %fraction, %scale_f : f64
    %rounded_fraction = arith.addf %scaled_fraction, %half_f : f64
    %raw_scaled = arith.fptosi %rounded_fraction : f64 to i64
    %carry_fraction = arith.cmpi eq, %raw_scaled, %million : i64
    %int_carry = arith.select %carry_fraction, %one, %zero : i1, i64
    %int_part = arith.addi %raw_int, %int_carry : i64
    %frac_part = arith.select %carry_fraction, %zero, %raw_scaled : i1, i64

    %trimmed:2 = scf.for %i = %start_loop to %frac_loop_end step %step iter_args(%n = %frac_part, %digits = %six) -> (i64, i64) {
      %more_than_one = arith.cmpi sgt, %digits, %one : i64
      %remainder = arith.remui %n, %ten : i64
      %trailing_zero = arith.cmpi eq, %remainder, %zero : i64
      %can_trim = arith.andi %more_than_one, %trailing_zero : i1
      %quotient = arith.divui %n, %ten : i64
      %next_n = arith.select %can_trim, %quotient, %n : i1, i64
      %decremented_digits = arith.subi %digits, %one : i64
      %next_digits = arith.select %can_trim, %decremented_digits, %digits : i1, i64
      scf.yield %next_n, %next_digits : i64, i64
    }

    %frac_digits_index = arith.index_cast %trimmed#1 : i64 to index
    %after_frac:2 = scf.for %i = %start_loop to %frac_loop_end step %step iter_args(%n = %trimmed#0, %pos = %last) -> (i64, index) {
      %active = arith.cmpi ult, %i, %frac_digits_index : index
      %next:2 = scf.if %active -> (i64, index) {
        %digit = arith.remui %n, %ten : i64
        %digit_ch_i64 = arith.addi %digit, %ascii_zero : i64
        %digit_ch = arith.trunci %digit_ch_i64 : i64 to i8
        memref.store %digit_ch, %buffer[%pos] : memref<48xi8>
        %quotient = arith.divui %n, %ten : i64
        %next_pos = arith.subi %pos, %one_index : index
        scf.yield %quotient, %next_pos : i64, index
      } else {
        scf.yield %n, %pos : i64, index
      }
      scf.yield %next#0, %next#1 : i64, index
    }

    memref.store %ascii_dot, %buffer[%after_frac#1] : memref<48xi8>
    %int_pos = arith.subi %after_frac#1, %one_index : index
    %int_is_zero = arith.cmpi eq, %int_part, %zero : i64
    %after_int:2 = scf.if %int_is_zero -> (i64, index) {
      %zero_ch_i64 = arith.constant 48 : i64
      %zero_ch = arith.trunci %zero_ch_i64 : i64 to i8
      memref.store %zero_ch, %buffer[%int_pos] : memref<48xi8>
      %next_pos = arith.subi %int_pos, %one_index : index
      scf.yield %zero, %next_pos : i64, index
    } else {
      %result:2 = scf.for %i = %start_loop to %int_loop_end step %step iter_args(%n = %int_part, %pos = %int_pos) -> (i64, index) {
        %active = arith.cmpi ne, %n, %zero : i64
        %next:2 = scf.if %active -> (i64, index) {
          %digit = arith.remui %n, %ten : i64
          %digit_ch_i64 = arith.addi %digit, %ascii_zero : i64
          %digit_ch = arith.trunci %digit_ch_i64 : i64 to i8
          memref.store %digit_ch, %buffer[%pos] : memref<48xi8>
          %quotient = arith.divui %n, %ten : i64
          %next_pos = arith.subi %pos, %one_index : index
          scf.yield %quotient, %next_pos : i64, index
        } else {
          scf.yield %n, %pos : i64, index
        }
        scf.yield %next#0, %next#1 : i64, index
      }
      scf.yield %result#0, %result#1 : i64, index
    }

    %first_digit = arith.addi %after_int#1, %one_index : index
    %start_index = scf.if %is_negative -> (index) {
      memref.store %ascii_minus, %buffer[%after_int#1] : memref<48xi8>
      scf.yield %after_int#1 : index
    } else {
      scf.yield %first_digit : index
    }
    %length_index = arith.subi %end, %start_index : index
    %length = arith.index_cast %length_index : index to i64
    %buffer_view = memref.cast %buffer : memref<48xi8> to memref<?xi8>
    %header, %bytes = func.call @LyUnicode_FromBytes(%buffer_view, %start_index, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // ===== impls: float =====
  func.func private @LyFloat_Shape() -> (memref<2xi64>, memref<1xf64>) attributes {ly.runtime.contract = "builtins.float", ly.runtime.shape}

  func.func @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"} {
    // One entity, one allocation: [0,16) header, [16,24) payload view.
    %block_bytes = arith.constant 24 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %part_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %payload = memref.view %block[%part_offset][] : memref<?xi8> to memref<1xf64>
    %one = arith.constant 1 : i64
    %layout_float = arith.constant 2 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %value_slot = arith.constant 0 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_float, %header[%layout_slot] : memref<2xi64>
    memref.store %value, %payload[%value_slot] : memref<1xf64>
    func.return %header, %payload : memref<2xi64>, memref<1xf64>
  }

  func.func @LyFloat_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.float", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyFloat_Init(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyFloat_AsF64(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"} {
    %value_slot = arith.constant 0 : index
    %value = memref.load %payload[%value_slot] : memref<1xf64>
    func.return %value : f64
  }

  func.func @LyFloat_Add(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__add__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %value = arith.addf %lhs, %rhs : f64
    %out_header, %out_payload = func.call @LyFloat_FromF64(%value) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %out_header, %out_payload : memref<2xi64>, memref<1xf64>
  }

  func.func @LyFloat_Sub(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__sub__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %value = arith.subf %lhs, %rhs : f64
    %out_header, %out_payload = func.call @LyFloat_FromF64(%value) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %out_header, %out_payload : memref<2xi64>, memref<1xf64>
  }

  func.func @LyFloat_Mul(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__mul__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %value = arith.mulf %lhs, %rhs : f64
    %out_header, %out_payload = func.call @LyFloat_FromF64(%value) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %out_header, %out_payload : memref<2xi64>, memref<1xf64>
  }

  func.func @LyFloat_TrueDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__truediv__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %value = arith.divf %lhs, %rhs : f64
    %out_header, %out_payload = func.call @LyFloat_FromF64(%value) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %out_header, %out_payload : memref<2xi64>, memref<1xf64>
  }

  // float ** float via libm pow (CPython float_pow also defers to C pow).
  // 0.0 ** negative raises ZeroDivisionError like CPython; a negative base
  // with a non-integral exponent yields a complex in CPython, which is not
  // implemented, so it raises instead (deviation until complex lands, R6).
  func.func @LyFloat_Pow(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__pow__"} {
    %base = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %exponent = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %zero = arith.constant 0.0 : f64
    %base_zero = arith.cmpf oeq, %base, %zero : f64
    %exp_negative = arith.cmpf olt, %exponent, %zero : f64
    %zero_negative = arith.andi %base_zero, %exp_negative : i1
    cf.cond_br %zero_negative, ^zero_to_negative, ^check_fractional

  ^zero_to_negative:
    func.call @__ly_long_raise_zero_negative_power() : () -> ()
    %dummy0 = arith.constant 0.0 : f64
    %zh, %zp = func.call @LyFloat_FromF64(%dummy0) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %zh, %zp : memref<2xi64>, memref<1xf64>

  ^check_fractional:
    %base_negative = arith.cmpf olt, %base, %zero : f64
    %exp_trunc = math.trunc %exponent : f64
    %exp_fractional = arith.cmpf one, %exp_trunc, %exponent : f64
    %complex_result = arith.andi %base_negative, %exp_fractional : i1
    cf.cond_br %complex_result, ^fractional_negative, ^pow

  ^fractional_negative:
    func.call @__ly_long_raise_fractional_power_negative() : () -> ()
    %dummy1 = arith.constant 0.0 : f64
    %fh, %fp = func.call @LyFloat_FromF64(%dummy1) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %fh, %fp : memref<2xi64>, memref<1xf64>

  ^pow:
    %value = math.powf %base, %exponent : f64
    %out_header, %out_payload = func.call @LyFloat_FromF64(%value) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %out_header, %out_payload : memref<2xi64>, memref<1xf64>
  }

  // Comparisons follow IEEE ordered semantics (NaN compares false), matching
  // CPython's float comparisons.
  func.func @LyFloat_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__eq__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %result = arith.cmpf oeq, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_NeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__ne__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %result = arith.cmpf une, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_LtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__lt__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %result = arith.cmpf olt, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_LeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__le__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %result = arith.cmpf ole, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_GtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__gt__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %result = arith.cmpf ogt, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_GeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_payload: memref<1xf64>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_payload: memref<1xf64>) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__ge__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header, %lhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header, %rhs_payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %result = arith.cmpf oge, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__bool__"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %zero = arith.constant 0.0 : f64
    %truth = arith.cmpf une, %value, %zero : f64
    func.return %truth : i1
  }

  // str(float) == repr(float) in CPython; delegate.
  func.func @LyFloat_Str(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %str_header, %str_bytes = func.call @LyFloat_Repr(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %str_header, %str_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFloat_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__repr__"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %str_header, %str_bytes = func.call @LyUnicode_FromF64(%value) : (f64) -> (memref<2xi64>, memref<?xi8>)
    func.return %str_header, %str_bytes : memref<2xi64>, memref<?xi8>
  }

  // ===== impls: collection =====
  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)

  func.func private @LyList_Shape() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.runtime.contract = "builtins.list", ly.runtime.shape}

  func.func private @LyTuple_Shape() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.shape}

  func.func private @LyDict_Shape() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.shape}

  func.func private @__ly_sequence_alloc(%class_id: i64, %length: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %minimum_capacity = arith.constant 64 : i64
    %handle_words = arith.constant 16 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 0 : index
    %capacity_slot = arith.constant 1 : index

    // One entity block: [0,16) header, [16,32) meta. The items array is
    // interior growable state: allocated here, reallocated only by
    // ensure_capacity, freed only by the deallocator.
    %block_bytes = arith.constant 32 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %meta_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %meta = memref.view %block[%meta_offset][] : memref<?xi8> to memref<2xi64>
    %needs_min_capacity = arith.cmpi slt, %length, %minimum_capacity : i64
    %capacity = arith.select %needs_min_capacity, %minimum_capacity, %length : i1, i64
    %payload_words = arith.muli %capacity, %handle_words : i64
    %payload_words_index = arith.index_cast %payload_words : i64 to index
    %items = memref.alloc(%payload_words_index) : memref<?xi64>

    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %class_id, %header[%layout_slot] : memref<2xi64>
    memref.store %length, %meta[%length_slot] : memref<2xi64>
    memref.store %capacity, %meta[%capacity_slot] : memref<2xi64>
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func private @__ly_dict_alloc(%length: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %minimum_capacity = arith.constant 64 : i64
    %handle_words = arith.constant 16 : i64
    %class_id = arith.constant 12 : i64
    %zero = arith.constant 0 : i64
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 0 : index
    %capacity_slot = arith.constant 1 : index

    // One entity block: [0,16) header, [16,32) meta; keys/values/present are
    // interior arrays managed only by dict runtime functions.
    %block_bytes = arith.constant 32 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %meta_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %meta = memref.view %block[%meta_offset][] : memref<?xi8> to memref<2xi64>
    %needs_min_capacity = arith.cmpi slt, %length, %minimum_capacity : i64
    %capacity = arith.select %needs_min_capacity, %minimum_capacity, %length : i1, i64
    %capacity_index = arith.index_cast %capacity : i64 to index
    %payload_words = arith.muli %capacity, %handle_words : i64
    %payload_words_index = arith.index_cast %payload_words : i64 to index
    %keys = memref.alloc(%payload_words_index) : memref<?xi64>
    %values = memref.alloc(%payload_words_index) : memref<?xi64>
    %present = memref.alloc(%capacity_index) : memref<?xi64>

    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %class_id, %header[%layout_slot] : memref<2xi64>
    memref.store %length, %meta[%length_slot] : memref<2xi64>
    memref.store %capacity, %meta[%capacity_slot] : memref<2xi64>
    scf.for %i = %lower to %capacity_index step %step {
      memref.store %zero, %present[%i] : memref<?xi64>
    }
    func.return %header, %meta, %keys, %values, %present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
  }

  func.func @LyList_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 10 : i64, ly.runtime.contract = "builtins.list", ly.runtime.initializer = "__new__"} {
    %class_id = arith.constant 10 : i64
    %header, %meta, %items = func.call @__ly_sequence_alloc(%class_id, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyTuple_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 11 : i64, ly.runtime.contract = "builtins.tuple", ly.runtime.initializer = "__new__"} {
    %class_id = arith.constant 11 : i64
    %header, %meta, %items = func.call @__ly_sequence_alloc(%class_id, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyDict_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 12 : i64, ly.runtime.contract = "builtins.dict", ly.runtime.initializer = "__new__"} {
    %header, %meta, %keys, %values, %present = func.call @__ly_dict_alloc(%length) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>)
    func.return %header, %meta, %keys, %values, %present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
  }

  func.func @LyList_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) -> i64 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    func.return %length : i64
  }

  func.func @LyTuple_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) -> i64 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    func.return %length : i64
  }

  func.func @LyDict_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    func.return %length : i64
  }

  func.func private @__ly_sequence_ensure_capacity(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>, %required: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) {
    %minimum_capacity = arith.constant 64 : i64
    %handle_words = arith.constant 16 : i64
    %two = arith.constant 2 : i64
    %capacity_slot = arith.constant 1 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index

    %capacity = memref.load %meta[%capacity_slot] : memref<2xi64>
    %needs_grow = arith.cmpi slt, %capacity, %required : i64
    %out_header, %out_meta, %out_items = scf.if %needs_grow -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) {
      %doubled = arith.muli %capacity, %two : i64
      %below_min = arith.cmpi slt, %doubled, %minimum_capacity : i64
      %base_capacity = arith.select %below_min, %minimum_capacity, %doubled : i1, i64
      %below_required = arith.cmpi slt, %base_capacity, %required : i64
      %new_capacity = arith.select %below_required, %required, %base_capacity : i1, i64
      %old_words = arith.muli %capacity, %handle_words : i64
      %new_words = arith.muli %new_capacity, %handle_words : i64
      %old_words_index = arith.index_cast %old_words : i64 to index
      %new_words_index = arith.index_cast %new_words : i64 to index
      %new_items = memref.alloc(%new_words_index) : memref<?xi64>
      scf.for %i = %lower to %old_words_index step %step {
        %word = memref.load %items[%i] : memref<?xi64>
        memref.store %word, %new_items[%i] : memref<?xi64>
      }
      memref.dealloc %items : memref<?xi64>
      memref.store %new_capacity, %meta[%capacity_slot] : memref<2xi64>
      scf.yield %header, %meta, %new_items : memref<2xi64>, memref<2xi64>, memref<?xi64>
    } else {
      scf.yield %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
    }
    func.return %out_header, %out_meta, %out_items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyList_EnsureCapacity(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>, %required: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.transfer_args = [0], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.primitive = "ensure_capacity"} {
    %out_header, %out_meta, %out_items = func.call @__ly_sequence_ensure_capacity(%header, %meta, %items, %required) : (memref<2xi64>, memref<2xi64>, memref<?xi64>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %out_header, %out_meta, %out_items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyDict_EnsureCapacity(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>, %required: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.ownership.transfer_args = [0], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "ensure_capacity"} {
    %minimum_capacity = arith.constant 64 : i64
    %handle_words = arith.constant 16 : i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %capacity_slot = arith.constant 1 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index

    %capacity = memref.load %meta[%capacity_slot] : memref<2xi64>
    %needs_grow = arith.cmpi slt, %capacity, %required : i64
    %out_header, %out_meta, %out_keys, %out_values, %out_present = scf.if %needs_grow -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) {
      %doubled = arith.muli %capacity, %two : i64
      %below_min = arith.cmpi slt, %doubled, %minimum_capacity : i64
      %base_capacity = arith.select %below_min, %minimum_capacity, %doubled : i1, i64
      %below_required = arith.cmpi slt, %base_capacity, %required : i64
      %new_capacity = arith.select %below_required, %required, %base_capacity : i1, i64
      %old_words = arith.muli %capacity, %handle_words : i64
      %new_words = arith.muli %new_capacity, %handle_words : i64
      %old_words_index = arith.index_cast %old_words : i64 to index
      %new_words_index = arith.index_cast %new_words : i64 to index
      %old_capacity_index = arith.index_cast %capacity : i64 to index
      %new_capacity_index = arith.index_cast %new_capacity : i64 to index
      %new_keys = memref.alloc(%new_words_index) : memref<?xi64>
      %new_values = memref.alloc(%new_words_index) : memref<?xi64>
      %new_present = memref.alloc(%new_capacity_index) : memref<?xi64>
      scf.for %i = %lower to %new_capacity_index step %step {
        memref.store %zero, %new_present[%i] : memref<?xi64>
      }
      scf.for %i = %lower to %old_words_index step %step {
        %key_word = memref.load %keys[%i] : memref<?xi64>
        %value_word = memref.load %values[%i] : memref<?xi64>
        memref.store %key_word, %new_keys[%i] : memref<?xi64>
        memref.store %value_word, %new_values[%i] : memref<?xi64>
      }
      scf.for %i = %lower to %old_capacity_index step %step {
        %present_word = memref.load %present[%i] : memref<?xi64>
        memref.store %present_word, %new_present[%i] : memref<?xi64>
      }
      memref.dealloc %present : memref<?xi64>
      memref.dealloc %values : memref<?xi64>
      memref.dealloc %keys : memref<?xi64>
      memref.store %new_capacity, %meta[%capacity_slot] : memref<2xi64>
      scf.yield %header, %meta, %new_keys, %new_values, %new_present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
    } else {
      scf.yield %header, %meta, %keys, %values, %present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
    }
    func.return %out_header, %out_meta, %out_keys, %out_values, %out_present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
  }

  func.func private @raw_bytes_equal(%p1: i64, %n1: i64, %p2: i64, %n2: i64) -> i1

  // Probe for a str key among the occupied slots (0..len-1; no deletion
  // surface). Returns the slot index or -1.
  // The probe key arrives as raw (pointer, byte length) words from the
  // C++ lowering, so unlike the set matcher this cannot also compare the
  // adaptive-width character width: two strings of different widths with
  // identical code-unit bytes (latin-1 "\x01\x01" vs UCS-2 U+0101)
  // collide here. Fixing this needs the find_slot ABI to carry the probe
  // header (planned with the Wave 1 hash-based dict rework).
  func.func @LyDict_FindSlot(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>, %key_ptr: i64, %key_len: i64) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "find_slot"} {
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %handle_words = arith.constant 16 : i64
    %length_slot = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %len = memref.load %meta[%length_slot] : memref<2xi64>
    %len_index = arith.index_cast %len : i64 to index
    %found = scf.for %i = %c0 to %len_index step %c1 iter_args(%acc = %minus_one) -> (i64) {
      %ii = arith.index_cast %i : index to i64
      %base = arith.muli %ii, %handle_words : i64
      %c5_i64 = arith.constant 5 : i64
      %c10_i64 = arith.constant 10 : i64
      %kp_index_i64 = arith.addi %base, %c5_i64 : i64
      %kl_index_i64 = arith.addi %base, %c10_i64 : i64
      %kp_index = arith.index_cast %kp_index_i64 : i64 to index
      %kl_index = arith.index_cast %kl_index_i64 : i64 to index
      %kp = memref.load %keys[%kp_index] : memref<?xi64>
      %kl = memref.load %keys[%kl_index] : memref<?xi64>
      %eq = func.call @raw_bytes_equal(%kp, %kl, %key_ptr, %key_len) : (i64, i64, i64, i64) -> i1
      %not_yet = arith.cmpi eq, %acc, %minus_one : i64
      %take = arith.andi %eq, %not_yet : i1
      %next = arith.select %take, %ii, %acc : i1, i64
      scf.yield %next : i64
    }
    func.return %found : i64
  }

  // `key in d` for runtime dicts (str keys): a present slot with equal bytes.
  func.func @LyDict_Contains(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>, %key_header: memref<2xi64> {ly.ownership.object_header}, %key_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "__contains__", ly.runtime.result_contract = "builtins.bool"} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %ptr_index = memref.extract_aligned_pointer_as_index %key_bytes : memref<?xi8> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    %dim = memref.dim %key_bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %slot = func.call @LyDict_FindSlot(%header, %meta, %keys, %values, %present, %ptr, %len) : (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>, i64, i64) -> i64
    %found = arith.cmpi sge, %slot, %c0_i64 : i64
    func.return %found : i1
  }

  // Runtime dict insert/replace with a boxed str key and boxed value. The
  // caller retained both boxes; on key replacement the duplicate key box is
  // consumed here. Slots 0..len-1 are always present (no deletion surface).
  func.func @LyDict_SetItemBox(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>, %key_box: memref<16xi64>, %value_box: memref<16xi64>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.ownership.transfer_args = [0], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "setitem_box"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %handle_words = arith.constant 16 : i64
    %length_slot = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index

    %len = memref.load %meta[%length_slot] : memref<2xi64>
    %new_ptr = memref.load %key_box[%c5] : memref<16xi64>
    %new_len = memref.load %key_box[%c10] : memref<16xi64>

    %found = func.call @LyDict_FindSlot(%header, %meta, %keys, %values, %present, %new_ptr, %new_len) : (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>, i64, i64) -> i64

    %missing = arith.cmpi eq, %found, %minus_one : i64
    %out:5 = scf.if %missing -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) {
      // Insert at slot len (all lower slots occupied).
      %required = arith.addi %len, %one : i64
      %grown:5 = func.call @LyDict_EnsureCapacity(%header, %meta, %keys, %values, %present, %required) : (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>)
      %slot_base_i64 = arith.muli %len, %handle_words : i64
      %c16 = arith.constant 16 : index
      %slot_base = arith.index_cast %slot_base_i64 : i64 to index
      scf.for %w = %c0 to %c16 step %c1 {
        %kw = memref.load %key_box[%w] : memref<16xi64>
        %vw = memref.load %value_box[%w] : memref<16xi64>
        %dst = arith.addi %slot_base, %w : index
        memref.store %kw, %grown#2[%dst] : memref<?xi64>
        memref.store %vw, %grown#3[%dst] : memref<?xi64>
      }
      %len_slot_index = arith.index_cast %len : i64 to index
      memref.store %one, %grown#4[%len_slot_index] : memref<?xi64>
      memref.store %required, %grown#1[%length_slot] : memref<2xi64>
      scf.yield %grown#0, %grown#1, %grown#2, %grown#3, %grown#4 : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
    } else {
      // Replace: release the old value and the duplicate new key box.
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %found) : (memref<?xi64>, i64) -> ()
      func.call @LyObject_ReleaseBoxedPayloadRaw(%key_box) : (memref<16xi64>) -> ()
      %slot_base_i64 = arith.muli %found, %handle_words : i64
      %c16 = arith.constant 16 : index
      %slot_base = arith.index_cast %slot_base_i64 : i64 to index
      scf.for %w = %c0 to %c16 step %c1 {
        %vw = memref.load %value_box[%w] : memref<16xi64>
        %dst = arith.addi %slot_base, %w : index
        memref.store %vw, %values[%dst] : memref<?xi64>
      }
      scf.yield %header, %meta, %keys, %values, %present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
    }
    func.return %out#0, %out#1, %out#2, %out#3, %out#4 : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
  }

  func.func @LyList_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.list", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %length_slot = arith.constant 0 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    %length_index = arith.index_cast %length : i64 to index
    scf.for %i = %lower to %length_index step %step {
      %logical_index = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %logical_index) : (memref<?xi64>, i64) -> ()
    }
    memref.dealloc %items : memref<?xi64>
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // Uniform per-element repr dispatch, generated per program by the lowering
  // (class id -> the manifest __repr__); resolved at link. Returns an owned str
  // (header, bytes) plus a handled flag.
  func.func private @__ly_repr_boxed_by_contract(%box: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>, i1) attributes {ly.ownership.owned_results = [0]}

  memref.global "private" constant @__ly_repr_lbracket : memref<1xi8> = dense<91>
  memref.global "private" constant @__ly_repr_rbracket : memref<1xi8> = dense<93>
  memref.global "private" constant @__ly_repr_comma : memref<2xi8> = dense<[44, 32]>
  memref.global "private" constant @__ly_repr_lparen : memref<1xi8> = dense<40>
  memref.global "private" constant @__ly_repr_rparen : memref<1xi8> = dense<41>
  memref.global "private" constant @__ly_repr_lbrace : memref<1xi8> = dense<123>
  memref.global "private" constant @__ly_repr_rbrace : memref<1xi8> = dense<125>
  memref.global "private" constant @__ly_repr_colon : memref<2xi8> = dense<[58, 32]>

  // list.__repr__: `[e0, e1, ...]`, each element repr'd through the uniform
  // boxed-method hook. Intermediate strs are released explicitly (Concat
  // borrows its operands).
  func.func @LyList_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %len = memref.load %meta[%c0] : memref<2xi64>
    %len_idx = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr

    %open_ref = memref.get_global @__ly_repr_lbracket : memref<1xi8>
    %open_dyn = memref.cast %open_ref : memref<1xi8> to memref<?xi8>
    %r0_h, %r0_b = func.call @LyUnicode_FromBytes(%open_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)

    %loop:2 = scf.for %i = %c0 to %len_idx step %c1 iter_args(%rh = %r0_h, %rb = %r0_b) -> (memref<2xi64>, memref<?xi8>) {
      %i_i64 = arith.index_cast %i : index to i64
      %is_pos = arith.cmpi sgt, %i_i64, %c0_i64 : i64
      %sep:2 = scf.if %is_pos -> (memref<2xi64>, memref<?xi8>) {
        %sep_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
        %sep_dyn = memref.cast %sep_ref : memref<2xi8> to memref<?xi8>
        %sh, %sb = func.call @LyUnicode_FromBytes(%sep_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %ch, %cb = func.call @LyUnicode_Concat(%rh, %rb, %sh, %sb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%rh) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%sh) : (memref<2xi64>) -> ()
        scf.yield %ch, %cb : memref<2xi64>, memref<?xi8>
      } else {
        scf.yield %rh, %rb : memref<2xi64>, memref<?xi8>
      }
      %off = arith.muli %i_i64, %c16_i64 : i64
      %box_ptr = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %class_gep = llvm.getelementptr %box_ptr[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %class_id = llvm.load %class_gep : !llvm.ptr -> i64
      %erh, %erb, %ok = func.call @__ly_repr_boxed_by_contract(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
      cf.assert %ok, "repr: boxed element has no conforming __repr__"
      %nh, %nb = func.call @LyUnicode_Concat(%sep#0, %sep#1, %erh, %erb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%sep#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%erh) : (memref<2xi64>) -> ()
      scf.yield %nh, %nb : memref<2xi64>, memref<?xi8>
    }

    %close_ref = memref.get_global @__ly_repr_rbracket : memref<1xi8>
    %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %clh, %clb = func.call @LyUnicode_FromBytes(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%loop#0, %loop#1, %clh, %clb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%clh) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  // tuple.__repr__: `(e0, e1, ...)`; a single element gets a trailing comma
  // (`(1,)`), matching CPython. Same uniform element dispatch as LyList_Repr.
  func.func @LyTuple_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %len = memref.load %meta[%c0] : memref<2xi64>
    %len_idx = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr

    %open_ref = memref.get_global @__ly_repr_lparen : memref<1xi8>
    %open_dyn = memref.cast %open_ref : memref<1xi8> to memref<?xi8>
    %r0_h, %r0_b = func.call @LyUnicode_FromBytes(%open_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)

    %loop:2 = scf.for %i = %c0 to %len_idx step %c1 iter_args(%rh = %r0_h, %rb = %r0_b) -> (memref<2xi64>, memref<?xi8>) {
      %i_i64 = arith.index_cast %i : index to i64
      %is_pos = arith.cmpi sgt, %i_i64, %c0_i64 : i64
      %sep:2 = scf.if %is_pos -> (memref<2xi64>, memref<?xi8>) {
        %sep_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
        %sep_dyn = memref.cast %sep_ref : memref<2xi8> to memref<?xi8>
        %sh, %sb = func.call @LyUnicode_FromBytes(%sep_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %ch, %cb = func.call @LyUnicode_Concat(%rh, %rb, %sh, %sb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%rh) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%sh) : (memref<2xi64>) -> ()
        scf.yield %ch, %cb : memref<2xi64>, memref<?xi8>
      } else {
        scf.yield %rh, %rb : memref<2xi64>, memref<?xi8>
      }
      %off = arith.muli %i_i64, %c16_i64 : i64
      %box_ptr = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %class_gep = llvm.getelementptr %box_ptr[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %class_id = llvm.load %class_gep : !llvm.ptr -> i64
      %erh, %erb, %ok = func.call @__ly_repr_boxed_by_contract(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
      cf.assert %ok, "repr: boxed element has no conforming __repr__"
      %nh, %nb = func.call @LyUnicode_Concat(%sep#0, %sep#1, %erh, %erb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%sep#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%erh) : (memref<2xi64>) -> ()
      scf.yield %nh, %nb : memref<2xi64>, memref<?xi8>
    }

    // Single-element tuple: trailing comma.
    %is_single = arith.cmpi eq, %len, %c1_i64 : i64
    %comma:2 = scf.if %is_single -> (memref<2xi64>, memref<?xi8>) {
      %comma_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
      %comma_dyn = memref.cast %comma_ref : memref<2xi8> to memref<?xi8>
      %tc_h, %tc_b = func.call @LyUnicode_FromBytes(%comma_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %wh, %wb = func.call @LyUnicode_Concat(%loop#0, %loop#1, %tc_h, %tc_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%tc_h) : (memref<2xi64>) -> ()
      scf.yield %wh, %wb : memref<2xi64>, memref<?xi8>
    } else {
      scf.yield %loop#0, %loop#1 : memref<2xi64>, memref<?xi8>
    }

    %close_ref = memref.get_global @__ly_repr_rparen : memref<1xi8>
    %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %clh, %clb = func.call @LyUnicode_FromBytes(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%comma#0, %comma#1, %clh, %clb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%comma#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%clh) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTuple_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %length_slot = arith.constant 0 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    %length_index = arith.index_cast %length : i64 to index
    scf.for %i = %lower to %length_index step %step {
      %logical_index = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %logical_index) : (memref<?xi64>, i64) -> ()
    }
    memref.dealloc %items : memref<?xi64>
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // dict.__repr__: `{k0: v0, k1: v1}` over present slots (capacity order); key
  // and value each repr'd through the uniform boxed-method hook. Same manual
  // loop-ownership as the sequence reprs (Concat borrows).
  func.func @LyDict_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.dict", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_meta = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %capacity = memref.load %meta[%c1_meta] : memref<2xi64>
    %capacity_idx = arith.index_cast %capacity : i64 to index
    %keys_idx = memref.extract_aligned_pointer_as_index %keys : memref<?xi64> -> index
    %keys_i64 = arith.index_cast %keys_idx : index to i64
    %keys_ptr = llvm.inttoptr %keys_i64 : i64 to !llvm.ptr
    %values_idx = memref.extract_aligned_pointer_as_index %values : memref<?xi64> -> index
    %values_i64 = arith.index_cast %values_idx : index to i64
    %values_ptr = llvm.inttoptr %values_i64 : i64 to !llvm.ptr

    %open_ref = memref.get_global @__ly_repr_lbrace : memref<1xi8>
    %open_dyn = memref.cast %open_ref : memref<1xi8> to memref<?xi8>
    %r0_h, %r0_b = func.call @LyUnicode_FromBytes(%open_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)

    %loop:3 = scf.for %i = %c0 to %capacity_idx step %c1 iter_args(%rh = %r0_h, %rb = %r0_b, %emitted = %c0_i64) -> (memref<2xi64>, memref<?xi8>, i64) {
      %slot = memref.load %present[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %slot, %c0_i64 : i64
      %entry:3 = scf.if %is_present -> (memref<2xi64>, memref<?xi8>, i64) {
        %i_i64 = arith.index_cast %i : index to i64
        // separator ", " when this is not the first emitted entry
        %has_prev = arith.cmpi sgt, %emitted, %c0_i64 : i64
        %sep:2 = scf.if %has_prev -> (memref<2xi64>, memref<?xi8>) {
          %sep_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
          %sep_dyn = memref.cast %sep_ref : memref<2xi8> to memref<?xi8>
          %sh, %sb = func.call @LyUnicode_FromBytes(%sep_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
          %ch, %cb = func.call @LyUnicode_Concat(%rh, %rb, %sh, %sb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          func.call @LyUnicode_DecRef(%rh) : (memref<2xi64>) -> ()
          func.call @LyUnicode_DecRef(%sh) : (memref<2xi64>) -> ()
          scf.yield %ch, %cb : memref<2xi64>, memref<?xi8>
        } else {
          scf.yield %rh, %rb : memref<2xi64>, memref<?xi8>
        }
        %off = arith.muli %i_i64, %c16_i64 : i64
        // key repr
        %kbox = llvm.getelementptr %keys_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %kclass_gep = llvm.getelementptr %kbox[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %kclass = llvm.load %kclass_gep : !llvm.ptr -> i64
        %krh, %krb, %kok = func.call @__ly_repr_boxed_by_contract(%kbox, %kclass) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
        cf.assert %kok, "repr: boxed dict key has no conforming __repr__"
        %k1h, %k1b = func.call @LyUnicode_Concat(%sep#0, %sep#1, %krh, %krb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%sep#0) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%krh) : (memref<2xi64>) -> ()
        // ": "
        %colon_ref = memref.get_global @__ly_repr_colon : memref<2xi8>
        %colon_dyn = memref.cast %colon_ref : memref<2xi8> to memref<?xi8>
        %coh, %cob = func.call @LyUnicode_FromBytes(%colon_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %k2h, %k2b = func.call @LyUnicode_Concat(%k1h, %k1b, %coh, %cob) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%k1h) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%coh) : (memref<2xi64>) -> ()
        // value repr
        %vbox = llvm.getelementptr %values_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %vclass_gep = llvm.getelementptr %vbox[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %vclass = llvm.load %vclass_gep : !llvm.ptr -> i64
        %vrh, %vrb, %vok = func.call @__ly_repr_boxed_by_contract(%vbox, %vclass) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
        cf.assert %vok, "repr: boxed dict value has no conforming __repr__"
        %k3h, %k3b = func.call @LyUnicode_Concat(%k2h, %k2b, %vrh, %vrb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%k2h) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%vrh) : (memref<2xi64>) -> ()
        %next_emitted = arith.addi %emitted, %c1_i64 : i64
        scf.yield %k3h, %k3b, %next_emitted : memref<2xi64>, memref<?xi8>, i64
      } else {
        scf.yield %rh, %rb, %emitted : memref<2xi64>, memref<?xi8>, i64
      }
      scf.yield %entry#0, %entry#1, %entry#2 : memref<2xi64>, memref<?xi8>, i64
    }

    %close_ref = memref.get_global @__ly_repr_rbrace : memref<1xi8>
    %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %clh, %clb = func.call @LyUnicode_FromBytes(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%loop#0, %loop#1, %clh, %clb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%clh) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyDict_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.dict", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %capacity_slot = arith.constant 1 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %capacity = memref.load %meta[%capacity_slot] : memref<2xi64>
    %capacity_index = arith.index_cast %capacity : i64 to index
    scf.for %i = %lower to %capacity_index step %step {
      %occupied = memref.load %present[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %occupied, %zero : i64
      scf.if %is_present {
        %logical_index = arith.index_cast %i : index to i64
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%keys, %logical_index) : (memref<?xi64>, i64) -> ()
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %logical_index) : (memref<?xi64>, i64) -> ()
      }
    }
    memref.dealloc %present : memref<?xi64>
    memref.dealloc %values : memref<?xi64>
    memref.dealloc %keys : memref<?xi64>
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: set =====
  func.func private @boxed_int_value(%meta_bits: i64, %digits_bits: i64) -> i64

  func.func private @LySet_Shape() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.runtime.contract = "builtins.set", ly.runtime.shape}

  func.func @LySet_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 21 : i64, ly.runtime.contract = "builtins.set", ly.runtime.initializer = "__new__"} {
    %class_id = arith.constant 21 : i64
    %header, %meta, %items = func.call @__ly_sequence_alloc(%class_id, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LySet_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) -> i64 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    func.return %length : i64
  }

  // Boxed equality between the slot at `slot` in `items` and `box`: class ids
  // must match; ints compare by decoded value, strs by bytes. Other element
  // classes conservatively compare unequal (the lowering restricts set
  // elements to int/str evidence).
  func.func private @__ly_set_slot_matches(%items: memref<?xi64>, %slot: i64, %box: memref<16xi64>) -> i1 {
    %handle_words = arith.constant 16 : i64
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c10 = arith.constant 10 : index
    %int_class = arith.constant 1 : i64
    %str_class = arith.constant 4 : i64
    %false = arith.constant false
    %base_i64 = arith.muli %slot, %handle_words : i64
    %base = arith.index_cast %base_i64 : i64 to index
    %class_index = arith.addi %base, %c1 : index
    %slot_class = memref.load %items[%class_index] : memref<?xi64>
    %box_class = memref.load %box[%c1] : memref<16xi64>
    %same_class = arith.cmpi eq, %slot_class, %box_class : i64
    %result = scf.if %same_class -> (i1) {
      %is_int = arith.cmpi eq, %slot_class, %int_class : i64
      %inner = scf.if %is_int -> (i1) {
        %slot_meta_index = arith.addi %base, %c5 : index
        %slot_digits_index = arith.addi %base, %c6 : index
        %slot_meta = memref.load %items[%slot_meta_index] : memref<?xi64>
        %slot_digits = memref.load %items[%slot_digits_index] : memref<?xi64>
        %box_meta = memref.load %box[%c5] : memref<16xi64>
        %box_digits = memref.load %box[%c6] : memref<16xi64>
        %slot_value = func.call @boxed_int_value(%slot_meta, %slot_digits) : (i64, i64) -> i64
        %box_value = func.call @boxed_int_value(%box_meta, %box_digits) : (i64, i64) -> i64
        %eq = arith.cmpi eq, %slot_value, %box_value : i64
        scf.yield %eq : i1
      } else {
        %is_str = arith.cmpi eq, %slot_class, %str_class : i64
        %str_eq = scf.if %is_str -> (i1) {
          %slot_ptr_index = arith.addi %base, %c5 : index
          %slot_len_index = arith.addi %base, %c10 : index
          %slot_ptr = memref.load %items[%slot_ptr_index] : memref<?xi64>
          %slot_len = memref.load %items[%slot_len_index] : memref<?xi64>
          %box_ptr = memref.load %box[%c5] : memref<16xi64>
          %box_len = memref.load %box[%c10] : memref<16xi64>
          // Adaptive-width strs must also match character width: distinct
          // strings of different widths can share the same code-unit bytes
          // (latin-1 "\x00\x01" vs UCS-2 U+0100). The width word sits at
          // header+16; both handles carry the header pointer at word 2.
          %c2_slot = arith.constant 2 : index
          %slot_header_index = arith.addi %base, %c2_slot : index
          %slot_header_ptr = memref.load %items[%slot_header_index] : memref<?xi64>
          %box_header_ptr = memref.load %box[%c2_slot] : memref<16xi64>
          %sixteen = arith.constant 16 : i64
          %slot_width_addr = arith.addi %slot_header_ptr, %sixteen : i64
          %box_width_addr = arith.addi %box_header_ptr, %sixteen : i64
          %slot_width_ptr = llvm.inttoptr %slot_width_addr : i64 to !llvm.ptr
          %box_width_ptr = llvm.inttoptr %box_width_addr : i64 to !llvm.ptr
          %slot_width = llvm.load %slot_width_ptr : !llvm.ptr -> i64
          %box_width = llvm.load %box_width_ptr : !llvm.ptr -> i64
          %width_eq = arith.cmpi eq, %slot_width, %box_width : i64
          %bytes_eq = func.call @raw_bytes_equal(%slot_ptr, %slot_len, %box_ptr, %box_len) : (i64, i64, i64, i64) -> i1
          %eq = arith.andi %width_eq, %bytes_eq : i1
          scf.yield %eq : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %str_eq : i1
      }
      scf.yield %inner : i1
    } else {
      scf.yield %false : i1
    }
    func.return %result : i1
  }

  // Probe for an equal element among slots 0..len-1. Returns the slot index
  // or -1.
  func.func private @__ly_set_find_slot(%meta: memref<2xi64>, %items: memref<?xi64>, %box: memref<16xi64>) -> i64 {
    %minus_one = arith.constant -1 : i64
    %length_slot = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %len = memref.load %meta[%length_slot] : memref<2xi64>
    %len_index = arith.index_cast %len : i64 to index
    %found = scf.for %i = %c0 to %len_index step %c1 iter_args(%acc = %minus_one) -> (i64) {
      %ii = arith.index_cast %i : index to i64
      %eq = func.call @__ly_set_slot_matches(%items, %ii, %box) : (memref<?xi64>, i64, memref<16xi64>) -> i1
      %not_yet = arith.cmpi eq, %acc, %minus_one : i64
      %take = arith.andi %eq, %not_yet : i1
      %next = arith.select %take, %ii, %acc : i1, i64
      scf.yield %next : i64
    }
    func.return %found : i64
  }

  // Runtime set insert with a boxed element. The caller retained the box; a
  // duplicate element consumes it here.
  func.func @LySet_AddBox(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>, %elem_box: memref<16xi64>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.transfer_args = [0], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.primitive = "add_box"} {
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %handle_words = arith.constant 16 : i64
    %length_slot = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %len = memref.load %meta[%length_slot] : memref<2xi64>
    %found = func.call @__ly_set_find_slot(%meta, %items, %elem_box) : (memref<2xi64>, memref<?xi64>, memref<16xi64>) -> i64
    %missing = arith.cmpi eq, %found, %minus_one : i64
    %out:3 = scf.if %missing -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) {
      %required = arith.addi %len, %one : i64
      %grown:3 = func.call @__ly_sequence_ensure_capacity(%header, %meta, %items, %required) : (memref<2xi64>, memref<2xi64>, memref<?xi64>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
      %slot_base_i64 = arith.muli %len, %handle_words : i64
      %c16 = arith.constant 16 : index
      %slot_base = arith.index_cast %slot_base_i64 : i64 to index
      scf.for %w = %c0 to %c16 step %c1 {
        %word = memref.load %elem_box[%w] : memref<16xi64>
        %dst = arith.addi %slot_base, %w : index
        memref.store %word, %grown#2[%dst] : memref<?xi64>
      }
      memref.store %required, %grown#1[%length_slot] : memref<2xi64>
      scf.yield %grown#0, %grown#1, %grown#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
    } else {
      // Duplicate: consume the caller's retained box.
      func.call @LyObject_ReleaseBoxedPayloadRaw(%elem_box) : (memref<16xi64>) -> ()
      scf.yield %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
    }
    func.return %out#0, %out#1, %out#2 : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // Membership probe with a BORROWED transient box (not consumed).
  func.func @LySet_ContainsBox(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>, %elem_box: memref<16xi64>) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.primitive = "contains_box"} {
    %minus_one = arith.constant -1 : i64
    %found = func.call @__ly_set_find_slot(%meta, %items, %elem_box) : (memref<2xi64>, memref<?xi64>, memref<16xi64>) -> i64
    %result = arith.cmpi ne, %found, %minus_one : i64
    func.return %result : i1
  }

  func.func @LySet_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.set", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %length_slot = arith.constant 0 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    %length_index = arith.index_cast %length : i64 to index
    scf.for %i = %lower to %length_index step %step {
      %logical_index = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %logical_index) : (memref<?xi64>, i64) -> ()
    }
    memref.dealloc %items : memref<?xi64>
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: range =====
  func.func private @LyRange_Shape() -> (memref<2xi64>, memref<3xi64>) attributes {ly.runtime.contract = "builtins.range", ly.runtime.shape}
  func.func private @LyRangeIterator_Shape() -> (memref<2xi64>, memref<3xi64>) attributes {ly.runtime.contract = "builtins.range_iterator", ly.runtime.shape}

  memref.global "private" constant @__ly_range_msg_zero_step : memref<30xi8> = dense<[114, 97, 110, 103, 101, 40, 41, 32, 97, 114, 103, 32, 51, 32, 109, 117, 115, 116, 32, 110, 111, 116, 32, 98, 101, 32, 122, 101, 114, 111]>

  func.func private @__ly_range_raise_zero_step() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 30 : i64
    %start = arith.constant 0 : index
    %message_static = memref.get_global @__ly_range_msg_zero_step : memref<30xi8>
    %message = memref.cast %message_static : memref<30xi8> to memref<?xi8>
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyRange_New(%first: i64, %second: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}, %step_in: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> (memref<2xi64>, memref<3xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 3 : i64, ly.runtime.contract = "builtins.range", ly.runtime.initializer = "__new__"} {
    %sentinel = arith.constant 9223372036854775807 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %second_missing = arith.cmpi eq, %second, %sentinel : i64
    %start = arith.select %second_missing, %zero, %first : i64
    %stop = arith.select %second_missing, %first, %second : i64
    %step_missing = arith.cmpi eq, %step_in, %sentinel : i64
    %step = arith.select %step_missing, %one, %step_in : i64
    %step_zero = arith.cmpi eq, %step, %zero : i64
    cf.cond_br %step_zero, ^raise, ^make

  ^raise:
    func.call @__ly_range_raise_zero_step() : () -> ()
    cf.br ^make

  ^make:
    // One entity, one allocation: [0,16) header, [16,40) state view.
    %block_bytes = arith.constant 40 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %part_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %state = memref.view %block[%part_offset][] : memref<?xi8> to memref<3xi64>
    %layout_range = arith.constant 3 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %start_slot = arith.constant 0 : index
    %stop_slot = arith.constant 1 : index
    %step_slot = arith.constant 2 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_range, %header[%layout_slot] : memref<2xi64>
    memref.store %start, %state[%start_slot] : memref<3xi64>
    memref.store %stop, %state[%stop_slot] : memref<3xi64>
    memref.store %step, %state[%step_slot] : memref<3xi64>
    func.return %header, %state : memref<2xi64>, memref<3xi64>
  }

  func.func @LyRange_Init(%header: memref<2xi64> {ly.ownership.object_header}, %state: memref<3xi64>) attributes {ly.runtime.contract = "builtins.range", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func private @__ly_range_iterator_alloc(%current: i64, %stop: i64, %step: i64) -> (memref<2xi64>, memref<3xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 20 : i64, ly.runtime.contract = "builtins.range_iterator", ly.runtime.primitive = "alloc"} {
    // One entity, one allocation: [0,16) header, [16,40) state view.
    %block_bytes = arith.constant 40 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %part_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %state = memref.view %block[%part_offset][] : memref<?xi8> to memref<3xi64>
    %one = arith.constant 1 : i64
    %layout_range_iterator = arith.constant 20 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %current_slot = arith.constant 0 : index
    %stop_slot = arith.constant 1 : index
    %step_slot = arith.constant 2 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_range_iterator, %header[%layout_slot] : memref<2xi64>
    memref.store %current, %state[%current_slot] : memref<3xi64>
    memref.store %stop, %state[%stop_slot] : memref<3xi64>
    memref.store %step, %state[%step_slot] : memref<3xi64>
    func.return %header, %state : memref<2xi64>, memref<3xi64>
  }

  func.func @LyRange_Iter(%header: memref<2xi64> {ly.ownership.object_header}, %state: memref<3xi64>) -> (memref<2xi64>, memref<3xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.range_iterator"} {
    %start_slot = arith.constant 0 : index
    %stop_slot = arith.constant 1 : index
    %step_slot = arith.constant 2 : index
    %start = memref.load %state[%start_slot] : memref<3xi64>
    %stop = memref.load %state[%stop_slot] : memref<3xi64>
    %step = memref.load %state[%step_slot] : memref<3xi64>
    %iter_header, %iter_state = func.call @__ly_range_iterator_alloc(%start, %stop, %step) : (i64, i64, i64) -> (memref<2xi64>, memref<3xi64>)
    func.return %iter_header, %iter_state : memref<2xi64>, memref<3xi64>
  }

  func.func @LyRangeIterator_Iter(%header: memref<2xi64> {ly.ownership.object_header}, %state: memref<3xi64>) -> (memref<2xi64>, memref<3xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range_iterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.range_iterator"} {
    %header_view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %header, %state : memref<2xi64>, memref<3xi64>
  }

  func.func @LyRangeIterator_Next(%header: memref<2xi64> {ly.ownership.object_header}, %state: memref<3xi64>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, i1, memref<2xi64>, memref<3xi64>) attributes {ly.ownership.owned_results = [0, 4], ly.runtime.contract = "builtins.range_iterator", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.int", ly.runtime.next_contract = "builtins.range_iterator", ly.runtime.valid_result_index = 3 : i64} {
    %current_slot = arith.constant 0 : index
    %stop_slot = arith.constant 1 : index
    %step_slot = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %current = memref.load %state[%current_slot] : memref<3xi64>
    %stop = memref.load %state[%stop_slot] : memref<3xi64>
    %step = memref.load %state[%step_slot] : memref<3xi64>
    %ascending = arith.cmpi sgt, %step, %zero : i64
    %below = arith.cmpi slt, %current, %stop : i64
    %above = arith.cmpi sgt, %current, %stop : i64
    %valid = arith.select %ascending, %below, %above : i1
    %advanced = arith.addi %current, %step : i64
    %next_current = arith.select %valid, %advanced, %current : i1, i64
    memref.store %next_current, %state[%current_slot] : memref<3xi64>
    %header_view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    %element_value = arith.select %valid, %current, %zero : i1, i64
    %element:3 = func.call @LyLong_FromI64(%element_value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %element#0, %element#1, %element#2, %valid, %header, %state : memref<2xi64>, memref<2xi64>, memref<?xi32>, i1, memref<2xi64>, memref<3xi64>
  }

  func.func @LyRange_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.range", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyRangeIterator_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.range_iterator", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: str_iterator =====
  func.func private @__ly_str_iterator_alloc(%position: i64, %length: i64) -> (memref<2xi64>, memref<2xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 7 : i64, ly.runtime.contract = "builtins.str_iterator", ly.runtime.primitive = "alloc"} {
    // One entity, one allocation: [0,16) header, [16,32) state view.
    %block_bytes = arith.constant 32 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %part_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %state = memref.view %block[%part_offset][] : memref<?xi8> to memref<2xi64>
    %one = arith.constant 1 : i64
    %layout_str_iterator = arith.constant 7 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %position_slot = arith.constant 0 : index
    %length_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_str_iterator, %header[%layout_slot] : memref<2xi64>
    memref.store %position, %state[%position_slot] : memref<2xi64>
    memref.store %length, %state[%length_slot] : memref<2xi64>
    func.return %header, %state : memref<2xi64>, memref<2xi64>
  }

  func.func @LyUnicode_Iter(%source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.str_iterator"} {
    %zero = arith.constant 0 : i64
    %length = func.call @LyUnicode_CodepointLength(%source_header, %source_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %iter_header, %state = func.call @__ly_str_iterator_alloc(%zero, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>)
    %source_header_view = memref.cast %source_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%source_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_Iter(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str_iterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.str_iterator"} {
    %iter_header_view = memref.cast %iter_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iter_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_Next(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>, i1, memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 3], ly.runtime.contract = "builtins.str_iterator", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.str", ly.runtime.next_contract = "builtins.str_iterator", ly.runtime.valid_result_index = 2 : i64} {
    %position_slot = arith.constant 0 : index
    %length_slot = arith.constant 1 : index
    %position = memref.load %state[%position_slot] : memref<2xi64>
    %length = memref.load %state[%length_slot] : memref<2xi64>
    %valid = arith.cmpi slt, %position, %length : i64
    %one = arith.constant 1 : i64
    %next_position_candidate = arith.addi %position, %one : i64
    %next_position = arith.select %valid, %next_position_candidate, %position : i1, i64
    memref.store %next_position, %state[%position_slot] : memref<2xi64>
    %iter_header_view = memref.cast %iter_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iter_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()

    %element:2 = scf.if %valid -> (memref<2xi64>, memref<?xi8>) {
      %item_header, %item_bytes = func.call @LyUnicode_GetItem(%source_header, %source_bytes, %position) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %item_header, %item_bytes : memref<2xi64>, memref<?xi8>
    } else {
      %start = arith.constant 0 : index
      %zero = arith.constant 0 : i64
      %empty_header, %empty_bytes = func.call @LyUnicode_FromBytes(%source_bytes, %start, %zero) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %empty_header, %empty_bytes : memref<2xi64>, memref<?xi8>
    }

    func.return %element#0, %element#1, %valid, %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<?xi8>, i1, memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_DecRef(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str_iterator", ly.runtime.deallocator} {
    %storage = memref.cast %iter_header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyUnicode_DecRef(%source_header) : (memref<2xi64>) -> ()
    memref.dealloc %iter_header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
