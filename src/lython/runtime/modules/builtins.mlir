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
  ly.runtime.contracts = ["types.NoneType", "builtins.object", "builtins.bool", "builtins.BaseException", "builtins.int", "builtins.str", "builtins.Exception", "builtins.RuntimeError", "builtins.TypeError", "builtins.ValueError", "builtins.ArithmeticError", "builtins.LookupError", "builtins.ZeroDivisionError", "builtins.KeyError", "builtins.IndexError", "builtins.AssertionError", "builtins.StopIteration", "builtins.StopAsyncIteration", "builtins.SystemExit", "builtins.OSError", "builtins.FileNotFoundError", "asyncio.CancelledError", "builtins.float", "builtins.bytes", "builtins.list", "builtins.tuple", "builtins.dict", "builtins.set", "builtins.range", "builtins.range_iterator", "builtins.str_iterator"],
  // Manifest Callable contracts for builtin free functions. These are the
  // single trusted source for these signatures; the emitter's seedBuiltins
  // reads them here instead of constructing the contracts in C++.
  ly.typing.function_names = ["builtins.print", "builtins.len"],
  ly.typing.function_contracts = [
    !py.callable<[], vararg = !py.contract<"builtins.tuple", [!py.contract<"builtins.object">]>, returns = [!py.literal<None>]>,
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
                    "__gt__", "__ge__", "__repr__", "__str__", "__eq__", "__ne__"],
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
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.union<!py.contract<"typing.SupportsIndex">, !py.literal<None>>] -> [!py.contract<"builtins.int">]>,
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
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance"]
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
                    "__ge__", "__str__", "__eq__", "__ne__"],
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
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance"]
  } {}

  py.class @complex attributes {base_names = ["object"], ly.typing.final} {}

  py.class @str attributes {
    base_names = ["Sequence", "Hashable"],
    ly.typing.base_args = [[!py.contract<"builtins.str">], []],
    method_names = ["__new__", "__len__", "__iter__", "__getitem__", "__add__",
                    "__contains__", "__eq__", "__lt__", "__le__", "__gt__",
                    "__ge__", "join", "startswith", "endswith", "__repr__", "__str__", "__ne__",
                    "encode"],
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
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bytes">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
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
                    "__bool__", "__repr__", "__str__", "decode"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance"]
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
  py.class @OSError attributes {base_names = ["Exception"]} {}
  py.class @FileNotFoundError attributes {base_names = ["OSError"]} {}

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
  func.func private @LyStopIteration_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.StopIteration", ly.runtime.shape}
  func.func private @LyStopAsyncIteration_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.shape}
  func.func private @LySystemExit_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SystemExit", ly.runtime.shape}
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
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyCancelledError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "asyncio.CancelledError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyCancelledError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "asyncio.CancelledError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemExit", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemExit", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OSError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OSError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
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

  // decode() with the default encoding (the str representation IS UTF-8
  // bytes). Invalid UTF-8 is not diagnosed yet (CPython raises
  // UnicodeDecodeError); the bytes land in the str payload unchanged.
  func.func @LyBytes_Decode(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%bytes, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // str.encode() with the default encoding: the payload is already UTF-8.
  func.func @LyUnicode_Encode(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "encode", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    %result_header, %result_bytes = func.call @LyBytes_FromBytes(%bytes, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // Same release interface as str (header only): the two contracts are
  // physical twins, and a wider input list here would win the
  // longest-inputTypes disambiguation and hijack str groups; the ownership
  // collector tells the twins apart by ly.runtime.result_contract instead.
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

  func.func private @__ly_long_raise_floor_div_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 34 : i64
    %message_static = memref.get_global @__ly_long_msg_integer_division_or_modulo_by_zero : memref<34xi8>
    %message = memref.cast %message_static : memref<34xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_mod_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 22 : i64
    %message_static = memref.get_global @__ly_long_msg_integer_modulo_by_zero : memref<22xi8>
    %message = memref.cast %message_static : memref<22xi8> to memref<?xi8>
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

  func.func @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"} {
    %result = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    func.return %result : i64
  }

  func.func @LyLong_AsF64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> f64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "unbox.f64"} {
    %value = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %float_value = arith.sitofp %value : i64 to f64
    func.return %float_value : f64
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

  func.func @LyLong_TrueDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__truediv__"} {
    %lhs = func.call @LyLong_AsF64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> f64
    %rhs = func.call @LyLong_AsF64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> f64
    %zero = arith.constant 0.0 : f64
    %rhs_nonzero = arith.cmpf one, %rhs, %zero : f64
    %result:2 = scf.if %rhs_nonzero -> (memref<2xi64>, memref<1xf64>) {
      %quotient = arith.divf %lhs, %rhs : f64
      %h, %p = func.call @LyFloat_FromF64(%quotient) : (f64) -> (memref<2xi64>, memref<1xf64>)
      scf.yield %h, %p : memref<2xi64>, memref<1xf64>
    } else {
      func.call @__ly_long_raise_true_div_zero() : () -> ()
      %dummy = arith.constant 0.0 : f64
      %h, %p = func.call @LyFloat_FromF64(%dummy) : (f64) -> (memref<2xi64>, memref<1xf64>)
      scf.yield %h, %p : memref<2xi64>, memref<1xf64>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<1xf64>
  }

  func.func @LyLong_FloorDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__floordiv__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %rhs_nonzero = arith.cmpi ne, %rhs, %zero : i64
    %result:3 = scf.if %rhs_nonzero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
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
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      func.call @__ly_long_raise_floor_div_zero() : () -> ()
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Mod(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mod__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %rhs_nonzero = arith.cmpi ne, %rhs, %zero : i64
    %result:3 = scf.if %rhs_nonzero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %trunc_r = arith.remsi %lhs, %rhs : i64
      %has_remainder = arith.cmpi ne, %trunc_r, %zero : i64
      %remainder_negative = arith.cmpi slt, %trunc_r, %zero : i64
      %rhs_negative = arith.cmpi slt, %rhs, %zero : i64
      %different_sign = arith.cmpi ne, %remainder_negative, %rhs_negative : i1
      %adjust = arith.andi %has_remainder, %different_sign : i1
      %adjusted = arith.addi %trunc_r, %rhs : i64
      %mod = arith.select %adjust, %adjusted, %trunc_r : i1, i64
      %h, %m, %d = func.call @LyLong_FromI64(%mod) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      func.call @__ly_long_raise_mod_zero() : () -> ()
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitAnd(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__and__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.andi %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitOr(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__or__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.ori %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitXor(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__xor__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.xori %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_LShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__lshift__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %nonnegative = arith.cmpi sge, %rhs, %zero : i64
    %result:3 = scf.if %nonnegative -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %value = arith.shli %lhs, %rhs : i64
      %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      func.call @__ly_long_raise_negative_shift() : () -> ()
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_RShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__rshift__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %nonnegative = arith.cmpi sge, %rhs, %zero : i64
    %result:3 = scf.if %nonnegative -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %value = arith.shrsi %lhs, %rhs : i64
      %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      func.call @__ly_long_raise_negative_shift() : () -> ()
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Round(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>, %ndigits: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__round__"} {
    %value = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %round_to_integer = arith.cmpi sge, %ndigits, %zero : i64
    %rounded = scf.if %round_to_integer -> (i64) {
      scf.yield %value : i64
    } else {
      %one = arith.constant 1 : i64
      %two = arith.constant 2 : i64
      %ten = arith.constant 10 : i64
      %eighteen = arith.constant 18 : i64
      %places = arith.subi %zero, %ndigits : i64
      %too_large = arith.cmpi sgt, %places, %eighteen : i64
      %limited_places = arith.select %too_large, %eighteen, %places : i1, i64
      %upper = arith.index_cast %limited_places : i64 to index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %scale = scf.for %i = %c0 to %upper step %c1 iter_args(%current = %one) -> (i64) {
        %next = arith.muli %current, %ten : i64
        scf.yield %next : i64
      }
      %negative = arith.cmpi slt, %value, %zero : i64
      %negated = arith.subi %zero, %value : i64
      %abs_value = arith.select %negative, %negated, %value : i1, i64
      %quotient = arith.divsi %abs_value, %scale : i64
      %remainder = arith.remsi %abs_value, %scale : i64
      %twice_remainder = arith.muli %remainder, %two : i64
      %greater_than_half = arith.cmpi sgt, %twice_remainder, %scale : i64
      %exactly_half = arith.cmpi eq, %twice_remainder, %scale : i64
      %low_bit = arith.andi %quotient, %one : i64
      %quotient_odd = arith.cmpi ne, %low_bit, %zero : i64
      %tie_rounds_up = arith.andi %exactly_half, %quotient_odd : i1
      %rounds_up = arith.ori %greater_than_half, %tie_rounds_up : i1
      %incremented = arith.addi %quotient, %one : i64
      %rounded_quotient = arith.select %rounds_up, %incremented, %quotient : i1, i64
      %rounded_abs = arith.muli %rounded_quotient, %scale : i64
      %rounded_neg = arith.subi %zero, %rounded_abs : i64
      %signed = arith.select %negative, %rounded_neg, %rounded_abs : i1, i64
      scf.yield %signed : i64
    }
    %h, %m, %d = func.call @LyLong_FromI64(%rounded) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
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

  func.func private @__ly_unicode_alloc(%len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "alloc"} {
    // One entity, one allocation: [0,16) header, [16,..) bytes. The header
    // view carries the allocation; bytes is an interior view.
    %byte_count = arith.index_cast %len : i64 to index
    %block_prefix = arith.constant 16 : index
    %block_bytes = arith.addi %byte_count, %block_prefix : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %bytes_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %bytes = memref.view %block[%bytes_offset][%byte_count] : memref<?xi8> to memref<?xi8>
    %one = arith.constant 1 : i64
    %layout_str = arith.constant 4 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_str, %header[%layout_slot] : memref<2xi64>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"} {
    %header, %result_bytes = func.call @__ly_unicode_alloc(%len) : (i64) -> (memref<2xi64>, memref<?xi8>)
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

  func.func @LyUnicode_Length(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "byte_length"} {
    %c0 = arith.constant 0 : index
    %length_index = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %length_index : index to i64
    func.return %length : i64
  }

  func.func @LyUnicode_CodepointLength(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__len__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %byte_count = memref.dim %bytes, %c0 : memref<?xi8>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %continuation_mask = arith.constant 192 : i64
    %continuation_tag = arith.constant 128 : i64
    %count = scf.for %index = %c0 to %byte_count step %c1 iter_args(%current = %zero) -> (i64) {
      %byte = memref.load %bytes[%index] : memref<?xi8>
      %byte_i64 = arith.extui %byte : i8 to i64
      %tag = arith.andi %byte_i64, %continuation_mask : i64
      %is_continuation = arith.cmpi eq, %tag, %continuation_tag : i64
      %next = scf.if %is_continuation -> (i64) {
        scf.yield %current : i64
      } else {
        %incremented = arith.addi %current, %one : i64
        scf.yield %incremented : i64
      }
      scf.yield %next : i64
    }
    func.return %count : i64
  }

  func.func @LyUnicode_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__bool__"} {
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi ne, %length, %zero : i64
    func.return %result : i1
  }

  func.func @LyUnicode_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__eq__"} {
    %lhs_len = func.call @LyUnicode_Length(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @LyUnicode_Length(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %same_len = arith.cmpi eq, %lhs_len, %rhs_len : i64
    %result = scf.if %same_len -> (i1) {
      %lower = arith.constant 0 : index
      %upper = arith.index_cast %lhs_len : i64 to index
      %step = arith.constant 1 : index
      %true = arith.constant true
      %all_equal = scf.for %index = %lower to %upper step %step iter_args(%current = %true) -> (i1) {
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

  // Lexicographic byte comparison (-1/0/1): first differing byte decides
  // (unsigned, matching CPython's bytewise UTF-8 ordering), else the shorter
  // string orders first.
  func.func private @LyUnicode_Compare(%lhs_header: memref<2xi64>, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64>, %rhs_bytes: memref<?xi8>) -> i64 {
    %lhs_len = func.call @LyUnicode_Length(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @LyUnicode_Length(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %min_len = arith.minsi %lhs_len, %rhs_len : i64
    %zero = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i64
    %plus_one = arith.constant 1 : i64
    %lower = arith.constant 0 : index
    %upper = arith.index_cast %min_len : i64 to index
    %step = arith.constant 1 : index
    %byte_cmp = scf.for %index = %lower to %upper step %step iter_args(%acc = %zero) -> (i64) {
      %lhs_byte = memref.load %lhs_bytes[%index] : memref<?xi8>
      %rhs_byte = memref.load %rhs_bytes[%index] : memref<?xi8>
      %lhs_u = arith.extui %lhs_byte : i8 to i64
      %rhs_u = arith.extui %rhs_byte : i8 to i64
      %lt = arith.cmpi ult, %lhs_u, %rhs_u : i64
      %gt = arith.cmpi ugt, %lhs_u, %rhs_u : i64
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
    %prefix_equal = arith.cmpi eq, %byte_cmp, %zero : i64
    %result = arith.select %prefix_equal, %len_cmp, %byte_cmp : i64
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

  func.func @LyUnicode_GetItem(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %raw_index: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__getitem__"} {
    %codepoints = func.call @LyUnicode_CodepointLength(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %is_negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %codepoints : i64
    %index = arith.select %is_negative, %from_end, %raw_index : i1, i64
    %lower_ok = arith.cmpi sge, %index, %zero : i64
    %upper_ok = arith.cmpi slt, %index, %codepoints : i64
    %valid = arith.andi %lower_ok, %upper_ok : i1
    %result:2 = scf.if %valid -> (memref<2xi64>, memref<?xi8>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %byte_count = memref.dim %bytes, %c0 : memref<?xi8>
      %false = arith.constant false
      %continuation_mask = arith.constant 192 : i64
      %continuation_tag = arith.constant 128 : i64
      %scan:5 = scf.for %i = %c0 to %byte_count step %c1 iter_args(%ordinal = %zero, %start = %c0, %end = %byte_count, %found = %false, %done = %false) -> (i64, index, index, i1, i1) {
        %byte = memref.load %bytes[%i] : memref<?xi8>
        %byte_i64 = arith.extui %byte : i8 to i64
        %tag = arith.andi %byte_i64, %continuation_mask : i64
        %is_start = arith.cmpi ne, %tag, %continuation_tag : i64
        %ordinal_matches = arith.cmpi eq, %ordinal, %index : i64
        %not_found = arith.cmpi eq, %found, %false : i1
        %not_done = arith.cmpi eq, %done, %false : i1
        %start_candidate = arith.andi %is_start, %ordinal_matches : i1
        %take_start_base = arith.andi %start_candidate, %not_found : i1
        %take_start = arith.andi %take_start_base, %not_done : i1
        %next_start = arith.select %take_start, %i, %start : i1, index
        %next_found = arith.ori %found, %take_start : i1
        %end_candidate = arith.andi %found, %is_start : i1
        %take_end = arith.andi %end_candidate, %not_done : i1
        %next_end = arith.select %take_end, %i, %end : i1, index
        %next_done = arith.ori %done, %take_end : i1
        %incremented = arith.addi %ordinal, %one : i64
        %next_ordinal = arith.select %is_start, %incremented, %ordinal : i1, i64
        scf.yield %next_ordinal, %next_start, %next_end, %next_found, %next_done : i64, index, index, i1, i1
      }
      %length_index = arith.subi %scan#2, %scan#1 : index
      %length = arith.index_cast %length_index : index to i64
      %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%bytes, %scan#1, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
    } else {
      func.call @__ly_unicode_raise_index_error() : () -> ()
      %c0 = arith.constant 0 : index
      %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%bytes, %c0, %zero) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Copy(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "copy"} {
    %start = arith.constant 0 : index
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_header, %new_bytes = func.call @LyUnicode_FromBytes(%bytes, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %new_header, %new_bytes : memref<2xi64>, memref<?xi8>
  }

  // str(s) returns s itself (retained) -- CPython identity semantics.
  func.func @LyUnicode_Str(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // CPython `str.__repr__`: wrap in quotes and escape. The quote is `'` unless
  // the string contains `'` and no `"` (then `"`), matching unicode_repr.
  // Escapes: \\ , the chosen quote, \t \n \r, and \xNN for other bytes < 0x20
  // or 0x7f. Printable ASCII and UTF-8 continuation bytes (>= 0x80) pass
  // through (non-ASCII printability tables are out of scope). Byte-oriented.
  func.func @LyUnicode_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %n = memref.dim %bytes, %c0 : memref<?xi8>
    %true_i1 = arith.constant true
    %bs = arith.constant 92 : i8
    %sq = arith.constant 39 : i8
    %dq = arith.constant 34 : i8
    %tab = arith.constant 9 : i8
    %nl = arith.constant 10 : i8
    %cr = arith.constant 13 : i8
    %sp = arith.constant 32 : i8
    %del = arith.constant 127 : i8
    %four_i8 = arith.constant 4 : i8
    %fifteen8 = arith.constant 15 : i8
    %ten8 = arith.constant 10 : i8
    %x_char = arith.constant 120 : i8
    %t_char = arith.constant 116 : i8
    %n_char = arith.constant 110 : i8
    %r_char = arith.constant 114 : i8
    %ascii0 = arith.constant 48 : i8
    %ascii_a = arith.constant 87 : i8
    %zero64 = arith.constant 0 : i64
    %one64 = arith.constant 1 : i64
    %two64 = arith.constant 2 : i64
    %four64 = arith.constant 4 : i64

    // Pass 1: count `'`/`"` and the escaped body length (quotes counted as 1).
    %scan:3 = scf.for %i = %c0 to %n step %c1
        iter_args(%csq = %zero64, %cdq = %zero64, %common = %zero64)
        -> (i64, i64, i64) {
      %b = memref.load %bytes[%i] : memref<?xi8>
      %is_sq = arith.cmpi eq, %b, %sq : i8
      %is_dq = arith.cmpi eq, %b, %dq : i8
      %is_bs = arith.cmpi eq, %b, %bs : i8
      %is_tab = arith.cmpi eq, %b, %tab : i8
      %is_nl = arith.cmpi eq, %b, %nl : i8
      %is_cr = arith.cmpi eq, %b, %cr : i8
      %nr = arith.ori %is_nl, %is_cr : i1
      %is_tnr = arith.ori %is_tab, %nr : i1
      %is_two = arith.ori %is_bs, %is_tnr : i1
      %lt_sp = arith.cmpi ult, %b, %sp : i8
      %not_tnr = arith.xori %is_tnr, %true_i1 : i1
      %ctrl_low = arith.andi %lt_sp, %not_tnr : i1
      %is_del = arith.cmpi eq, %b, %del : i8
      %is_four = arith.ori %ctrl_low, %is_del : i1
      %len_two_one = arith.select %is_two, %two64, %one64 : i64
      %contrib = arith.select %is_four, %four64, %len_two_one : i64
      %add_sq = arith.select %is_sq, %one64, %zero64 : i64
      %add_dq = arith.select %is_dq, %one64, %zero64 : i64
      %csq2 = arith.addi %csq, %add_sq : i64
      %cdq2 = arith.addi %cdq, %add_dq : i64
      %common2 = arith.addi %common, %contrib : i64
      scf.yield %csq2, %cdq2, %common2 : i64, i64, i64
    }

    %sq_present = arith.cmpi ne, %scan#0, %zero64 : i64
    %dq_absent = arith.cmpi eq, %scan#1, %zero64 : i64
    %use_double = arith.andi %sq_present, %dq_absent : i1
    %quote = arith.select %use_double, %dq, %sq : i8
    %count_q = arith.select %use_double, %scan#1, %scan#0 : i64
    %body = arith.addi %scan#2, %count_q : i64
    %total = arith.addi %body, %two64 : i64

    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%total) : (i64) -> (memref<2xi64>, memref<?xi8>)
    memref.store %quote, %out_bytes[%c0] : memref<?xi8>

    // Pass 2: fill escaped bytes starting at position 1.
    scf.for %i = %c0 to %n step %c1 iter_args(%pos = %c1) -> (index) {
      %b = memref.load %bytes[%i] : memref<?xi8>
      %is_q = arith.cmpi eq, %b, %quote : i8
      %is_bs = arith.cmpi eq, %b, %bs : i8
      %is_tab = arith.cmpi eq, %b, %tab : i8
      %is_nl = arith.cmpi eq, %b, %nl : i8
      %is_cr = arith.cmpi eq, %b, %cr : i8
      %nr = arith.ori %is_nl, %is_cr : i1
      %is_tnr = arith.ori %is_tab, %nr : i1
      %two_bsq = arith.ori %is_bs, %is_q : i1
      %is_two = arith.ori %two_bsq, %is_tnr : i1
      %lt_sp = arith.cmpi ult, %b, %sp : i8
      %not_tnr = arith.xori %is_tnr, %true_i1 : i1
      %ctrl_low = arith.andi %lt_sp, %not_tnr : i1
      %is_del = arith.cmpi eq, %b, %del : i8
      %is_four = arith.ori %ctrl_low, %is_del : i1
      %is_escaped = arith.ori %is_two, %is_four : i1

      // second byte for 2-byte escapes: \\ -> '\', quote -> quote, \t\n\r.
      %sec_bs = arith.select %is_bs, %bs, %quote : i8
      %sec_q = arith.select %is_q, %quote, %sec_bs : i8
      %sec_t = arith.select %is_tab, %t_char, %sec_q : i8
      %sec_n = arith.select %is_nl, %n_char, %sec_t : i8
      %second = arith.select %is_cr, %r_char, %sec_n : i8

      %byte0 = arith.select %is_escaped, %bs, %b : i8
      %byte1 = arith.select %is_four, %x_char, %second : i8
      memref.store %byte0, %out_bytes[%pos] : memref<?xi8>
      %pos1 = arith.addi %pos, %c1 : index
      scf.if %is_escaped {
        memref.store %byte1, %out_bytes[%pos1] : memref<?xi8>
      }
      // 4-byte \xNN: high and low nibble as lowercase hex.
      %hi = arith.shrui %b, %four_i8 : i8
      %lo = arith.andi %b, %fifteen8 : i8
      %hi_lt = arith.cmpi ult, %hi, %ten8 : i8
      %hi_off = arith.select %hi_lt, %ascii0, %ascii_a : i8
      %hi_char = arith.addi %hi, %hi_off : i8
      %lo_lt = arith.cmpi ult, %lo, %ten8 : i8
      %lo_off = arith.select %lo_lt, %ascii0, %ascii_a : i8
      %lo_char = arith.addi %lo, %lo_off : i8
      %pos2 = arith.addi %pos, %c2 : index
      %pos3 = arith.addi %pos, %c3 : index
      scf.if %is_four {
        memref.store %hi_char, %out_bytes[%pos2] : memref<?xi8>
        memref.store %lo_char, %out_bytes[%pos3] : memref<?xi8>
      }
      %step_two_one = arith.select %is_two, %c2, %c1 : index
      %step = arith.select %is_four, %c4, %step_two_one : index
      %next_pos = arith.addi %pos, %step : index
      scf.yield %next_pos : index
    }

    %total_idx = arith.index_cast %total : i64 to index
    %last = arith.subi %total_idx, %c1 : index
    memref.store %quote, %out_bytes[%last] : memref<?xi8>
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"} {
    %lhs_len = func.call @LyUnicode_Length(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @LyUnicode_Length(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %total_len = arith.addi %lhs_len, %rhs_len : i64
    %header, %bytes = func.call @__ly_unicode_alloc(%total_len) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %zero = arith.constant 0 : index
    %lhs_upper = arith.index_cast %lhs_len : i64 to index
    %rhs_upper = arith.index_cast %rhs_len : i64 to index
    %lhs_offset = arith.index_cast %lhs_len : i64 to index
    %step = arith.constant 1 : index
    scf.for %index = %zero to %lhs_upper step %step {
      %byte = memref.load %lhs_bytes[%index] : memref<?xi8>
      memref.store %byte, %bytes[%index] : memref<?xi8>
    }
    scf.for %index = %zero to %rhs_upper step %step {
      %byte = memref.load %rhs_bytes[%index] : memref<?xi8>
      %dest = arith.addi %lhs_offset, %index : index
      memref.store %byte, %bytes[%dest] : memref<?xi8>
    }
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_print_end : memref<1xi8> = dense<[10]>

  // print(object) after the emitter's sep-join desugar: CPython
  // builtin_print_impl's tail with objects_length == 1, file = sys.stdout
  // (fd 1) and the default end = "\n" (Python/bltinmodule.c). The str
  // rendering of each argument stays ahead of this sink (method_sink
  // dispatch / emitter desugar) because it needs per-value evidence.
  func.func @LyUnicode_PrintLine(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.builtin = "print", ly.runtime.builtin_lowering = "method_sink", ly.runtime.builtin_method = "__repr__", ly.runtime.builtin_sink_contract = "builtins.str", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print_line", ly.runtime.result_contract = "types.NoneType"} {
    %stdout = arith.constant 1 : i32
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.call @LyHost_WriteBytes(%stdout, %bytes, %length) : (i32, memref<?xi8>, i64) -> ()
    %end_static = memref.get_global @__ly_print_end : memref<1xi8>
    %end = memref.cast %end_static : memref<1xi8> to memref<?xi8>
    %end_length = arith.constant 1 : i64
    func.call @LyHost_WriteBytes(%stdout, %end, %end_length) : (i32, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func @LyUnicode_Print(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print"} {
    %stdout = arith.constant 1 : i32
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.call @LyHost_WriteBytes(%stdout, %bytes, %length) : (i32, memref<?xi8>, i64) -> ()
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
          %eq = func.call @raw_bytes_equal(%slot_ptr, %slot_len, %box_ptr, %box_len) : (i64, i64, i64, i64) -> i1
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
