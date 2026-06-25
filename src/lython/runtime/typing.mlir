// Contract manifest for Python-visible types.
//
// This file is the source of truth for typeshed-shaped contracts. PyDialect
// only owns the primitive contract algebra (`!py.contract`, `!py.protocol`,
// `!py.union`, `!py.literal`, `!py.type`, `!py.self`, `!py.typevar`,
// `!py.paramspec`) and a small set of language/control operations. Classes
// from builtins.pyi and types.pyi are declared here, including builtin runtime
// objects such as int, str, list, dict, function, BaseException, and
// TracebackType.
//
// Conventions:
//   - `!py.contract<"builtins.int">` is a nominal manifest contract, not a
//     dialect primitive type.
//   - `!py.contract<"builtins.list", [!py.contract<"builtins.str">]>` names a
//     parameterized nominal contract; protocol parameterization uses the same
//     term list inside `!py.protocol<...>`.
//   - `!py.contract<"$T">` names a generic type parameter declared by
//     `ly.typing.params`.
//   - `None`, `True`, and `False` are singleton values represented by
//     `!py.literal<None>`, `!py.literal<True>`, and `!py.literal<False>`.
//   - Function/method contracts are `!py.protocol<"Callable", ...>` entries in
//     a class's `method_contracts`; there is no method-declaration op.
//   - Class instantiation is represented by `py.new` for `__new__` followed by
//     `py.init` for the selected `__init__` contract. `py.init` preserves the
//     Python-level `Literal[None]` return; the expression value remains the
//     instance produced by `py.new`.
//   - Runtime implementation symbols live in runtime/objects/*.mlir and are
//     linked to these contracts through `ly.runtime.contract`.

module {
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

  py.class @NoneType attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__bool__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"types.NoneType">] -> [!py.literal<False>]>
    ],
    method_kinds = ["instance"]
  } {}

  py.class @NotImplementedType attributes {base_names = ["object"],
                                          ly.typing.final} {}
  py.class @EllipsisType attributes {base_names = ["object"],
                                    ly.typing.final} {}
  py.class @UnionType attributes {base_names = ["object"], ly.typing.final} {}

  py.class @Protocol attributes {base_names = ["object"], ly.typing.abstract,
                                ly.typing.protocol} {}
  py.class @Callable attributes {base_names = ["Protocol"],
                                ly.typing.abstract, ly.typing.protocol} {}

  py.class @function attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__call__", "__get__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.function">, !py.paramspec<"P">] -> [!py.typevar<"R">]>,
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
                    "__gt__", "__ge__"],
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
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  py.class @bool attributes {
    base_names = ["int"], ly.typing.final,
    method_names = ["__new__", "__bool__", "__and__", "__or__", "__xor__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bool">>, !py.contract<"builtins.object">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance"]
  } {}

  py.class @float attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__new__", "__repr__", "__add__", "__sub__", "__mul__",
                    "__truediv__", "__floordiv__", "__mod__", "__float__",
                    "__bool__", "__round__", "__lt__", "__le__", "__gt__",
                    "__ge__"],
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
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}

  py.class @complex attributes {base_names = ["object"], ly.typing.final} {}

  py.class @str attributes {
    base_names = ["Sequence", "Hashable"],
    ly.typing.base_args = [[!py.contract<"builtins.str">], []],
    method_names = ["__new__", "__len__", "__iter__", "__getitem__", "__add__",
                    "__contains__", "__eq__", "__lt__", "__le__", "__gt__",
                    "__ge__", "join", "startswith", "endswith"],
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
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance"]
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

  py.class @bytes attributes {base_names = ["Sequence", "Hashable"],
                             ly.typing.base_args = [[!py.contract<"builtins.int">], []],
                             ly.typing.final} {}
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
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">]],
    method_names = ["__len__", "__contains__", "__getitem__", "__iter__",
                    "__add__", "__mul__", "count", "index"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.tuple">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.tuple">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.Any">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.Any">, !py.contract<"typing.SupportsIndex">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance"]
  } {}

  py.class @list attributes {
    base_names = ["MutableSequence"], ly.typing.params = ["T"],
    ly.typing.base_args = [[!py.contract<"$T">]],
    method_names = ["__init__", "__init__", "append", "extend", "pop",
                    "insert", "remove", "clear", "__len__", "__iter__",
                    "__getitem__", "__setitem__", "__delitem__",
                    "__contains__"],
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
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance"]
  } {}

  py.class @dict attributes {
    base_names = ["MutableMapping"], ly.typing.params = ["K", "V"],
    ly.typing.base_args = [[!py.contract<"$K">, !py.contract<"$V">]],
    method_names = ["__init__", "__init__", "__len__", "__iter__",
                    "__getitem__", "get", "get", "get", "__setitem__",
                    "__delitem__", "__contains__", "keys", "values", "items"],
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
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.dict_items", [!py.contract<"$K">, !py.contract<"$V">]>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance"]
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

  py.class @set attributes {base_names = ["MutableSet"],
                           ly.typing.params = ["T"],
                           ly.typing.base_args = [[!py.contract<"$T">]]} {}
  py.class @frozenset attributes {base_names = ["AbstractSet", "Hashable"],
                                 ly.typing.params = ["T"],
                                 ly.typing.base_args = [[!py.contract<"$T">], []]} {}
  py.class @range attributes {base_names = ["Sequence", "Hashable"],
                             ly.typing.base_args = [[!py.contract<"builtins.int">], []],
                             ly.typing.final} {}

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
  py.class @KeyError attributes {base_names = ["Exception"]} {}
  py.class @IndexError attributes {base_names = ["Exception"]} {}
  py.class @AssertionError attributes {base_names = ["Exception"]} {}
  py.class @StopIteration attributes {
    base_names = ["Exception"],
    field_names = ["value"],
    field_contract_types = [!py.contract<"typing.Any">]
  } {}
  py.class @StopAsyncIteration attributes {base_names = ["Exception"]} {}

  py.class @TracebackType attributes {
    base_names = ["object"], ly.typing.final,
    field_names = ["tb_next", "tb_frame", "tb_lasti", "tb_lineno"],
    field_contract_types = [
      !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>,
      !py.contract<"types.FrameType">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">
    ]
  } {}
  py.class @FrameType attributes {base_names = ["object"], ly.typing.abstract} {}
  py.class @GenericAlias attributes {base_names = ["object"], ly.typing.abstract} {}
  py.class @TextIO attributes {base_names = ["object"], ly.typing.abstract} {}
  py.class @AbstractEventLoop attributes {base_names = ["object"],
                                         ly.typing.abstract} {}
  py.class @Context attributes {base_names = ["object"], ly.typing.abstract} {}

  py.class @Hashable attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    method_names = ["__hash__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.Hashable">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @Sized attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    method_names = ["__len__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.Sized">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @Container attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T"], ly.typing.param_variance = ["contravariant"],
    method_names = ["__contains__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"Container", [!py.contract<"$T">]>, !py.contract<"$T">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @Iterable attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
    method_names = ["__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @Iterator attributes {
    base_names = ["Iterable", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    method_names = ["__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"Iterator", [!py.contract<"$T">]>] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.protocol<"Iterator", [!py.contract<"$T">]>] -> [!py.contract<"$T">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @Collection attributes {
    base_names = ["Sized", "Iterable", "Container", "Protocol"],
    ly.typing.abstract, ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[], [!py.contract<"$T">], [!py.contract<"$T">], []]
  } {}
  py.class @Sequence attributes {
    base_names = ["Collection", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    method_names = ["__getitem__", "__iter__", "__contains__", "count", "index"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"Sequence", [!py.contract<"$T">]>, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.protocol<"Sequence", [!py.contract<"$T">]>] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.protocol<"Sequence", [!py.contract<"$T">]>, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.protocol<"Sequence", [!py.contract<"$T">]>, !py.contract<"builtins.object">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.protocol<"Sequence", [!py.contract<"$T">]>, !py.contract<"builtins.object">, !py.contract<"typing.SupportsIndex">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance"]
  } {}
  py.class @MutableSequence attributes {
    base_names = ["Sequence", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    method_names = ["__setitem__", "__delitem__", "insert", "append",
                    "remove", "clear"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"MutableSequence", [!py.contract<"$T">]>, !py.contract<"typing.SupportsIndex">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.protocol<"MutableSequence", [!py.contract<"$T">]>, !py.contract<"typing.SupportsIndex">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.protocol<"MutableSequence", [!py.contract<"$T">]>, !py.contract<"typing.SupportsIndex">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.protocol<"MutableSequence", [!py.contract<"$T">]>, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.protocol<"MutableSequence", [!py.contract<"$T">]>, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.protocol<"MutableSequence", [!py.contract<"$T">]>] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance"]
  } {}
  py.class @Mapping attributes {
    base_names = ["Collection", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["invariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$K">], []],
    method_names = ["__getitem__", "get", "get", "get", "items", "keys",
                    "values", "__contains__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"Mapping", [!py.contract<"$K">, !py.contract<"$V">]>, !py.contract<"$K">] -> [!py.contract<"$V">]>,
      !py.protocol<"Callable", [!py.protocol<"Mapping", [!py.contract<"$K">, !py.contract<"$V">]>, !py.contract<"$K">] -> [!py.union<!py.contract<"$V">, !py.literal<None>>]>,
      !py.protocol<"Callable", [!py.protocol<"Mapping", [!py.contract<"$K">, !py.contract<"$V">]>, !py.contract<"$K">, !py.contract<"$V">] -> [!py.contract<"$V">]>,
      !py.protocol<"Callable", [!py.protocol<"Mapping", [!py.contract<"$K">, !py.contract<"$V">]>, !py.contract<"$K">, !py.typevar<"D">] -> [!py.union<!py.contract<"$V">, !py.typevar<"D">>]>,
      !py.protocol<"Callable", [!py.protocol<"Mapping", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.contract<"typing.ItemsView", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.protocol<"Mapping", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.contract<"typing.KeysView", [!py.contract<"$K">]>]>,
      !py.protocol<"Callable", [!py.protocol<"Mapping", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.contract<"typing.ValuesView", [!py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.protocol<"Mapping", [!py.contract<"$K">, !py.contract<"$V">]>, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance"]
  } {}
  py.class @MutableMapping attributes {
    base_names = ["Mapping", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["K", "V"],
    ly.typing.base_args = [[!py.contract<"$K">, !py.contract<"$V">], []],
    method_names = ["__setitem__", "__delitem__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"MutableMapping", [!py.contract<"$K">, !py.contract<"$V">]>, !py.contract<"$K">, !py.contract<"$V">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.protocol<"MutableMapping", [!py.contract<"$K">, !py.contract<"$V">]>, !py.contract<"$K">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @AbstractSet attributes {base_names = ["Collection", "Protocol"],
                                   ly.typing.abstract, ly.typing.protocol,
                                   ly.typing.params = ["T"]} {}
  py.class @MutableSet attributes {base_names = ["AbstractSet", "Protocol"],
                                  ly.typing.abstract, ly.typing.protocol,
                                  ly.typing.params = ["T"]} {}

  py.class @SupportsInt attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    method_names = ["__int__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.SupportsInt">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @SupportsFloat attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    method_names = ["__float__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.SupportsFloat">] -> [!py.contract<"builtins.float">]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @SupportsIndex attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    method_names = ["__index__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @SupportsBytes attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    method_names = ["__bytes__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.SupportsBytes">] -> [!py.contract<"builtins.bytes">]>
    ],
    method_kinds = ["instance"]
  } {}

  py.class @Generator attributes {
    base_names = ["Iterator", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["Y", "S", "R"],
    ly.typing.param_variance = ["covariant", "contravariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$Y">], []],
    method_names = ["__next__", "send", "throw", "throw", "close", "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.Generator">] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"typing.Generator">, !py.contract<"$S">] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"typing.Generator">, !py.type<!py.contract<"builtins.BaseException">>, !py.union<!py.contract<"builtins.BaseException">, !py.contract<"builtins.object">>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"typing.Generator">, !py.contract<"builtins.BaseException">, !py.literal<None>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"typing.Generator">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"typing.Generator">] -> [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance"]
  } {}
  py.class @Awaitable attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
    method_names = ["__await__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.Awaitable">] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$T">]>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @Coroutine attributes {
    base_names = ["Awaitable", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["Y", "S", "R"],
    ly.typing.param_variance = ["covariant", "contravariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$R">], []],
    method_names = ["send", "throw", "throw", "close"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.Coroutine">, !py.contract<"$S">] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"typing.Coroutine">, !py.type<!py.contract<"builtins.BaseException">>, !py.union<!py.contract<"builtins.BaseException">, !py.contract<"builtins.object">>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"typing.Coroutine">, !py.contract<"builtins.BaseException">, !py.literal<None>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"typing.Coroutine">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance"]
  } {}
  py.class @AsyncIterable attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
    method_names = ["__aiter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.AsyncIterable">] -> [!py.protocol<"AsyncIterator", [!py.contract<"$T">]>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @AsyncIterator attributes {
    base_names = ["AsyncIterable", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    method_names = ["__aiter__", "__anext__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.AsyncIterator">] -> [!py.protocol<"AsyncIterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"typing.AsyncIterator">] -> [!py.protocol<"Awaitable", [!py.contract<"$T">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @AsyncGenerator attributes {
    base_names = ["AsyncIterator", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["Y", "S"],
    ly.typing.base_args = [[!py.contract<"$Y">], []],
    method_names = ["__anext__", "asend", "athrow", "athrow", "aclose"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.AsyncGenerator">] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$Y">]>]>,
      !py.protocol<"Callable", [!py.contract<"typing.AsyncGenerator">, !py.contract<"$S">] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$Y">]>]>,
      !py.protocol<"Callable", [!py.contract<"typing.AsyncGenerator">, !py.type<!py.contract<"builtins.BaseException">>, !py.union<!py.contract<"builtins.BaseException">, !py.contract<"builtins.object">>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$Y">]>]>,
      !py.protocol<"Callable", [!py.contract<"typing.AsyncGenerator">, !py.contract<"builtins.BaseException">, !py.literal<None>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$Y">]>]>,
      !py.protocol<"Callable", [!py.contract<"typing.AsyncGenerator">] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.literal<None>]>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  py.class @ContextManager attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T", "ExitT"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.param_defaults = [!py.union<!py.contract<"builtins.bool">, !py.literal<None>>],
    method_names = ["__enter__", "__exit__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.ContextManager">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"typing.ContextManager">, !py.union<!py.type<!py.contract<"builtins.BaseException">>, !py.literal<None>>, !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$ExitT">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @nullcontext attributes {
    base_names = ["ContextManager"],
    ly.typing.base_args = [[!py.contract<"builtins.object">],
                           [!py.contract<"builtins.bool">]],
    method_names = ["__new__", "__init__", "__enter__", "__exit__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"contextlib.nullcontext">>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"contextlib.nullcontext">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"contextlib.nullcontext">] -> [!py.contract<"contextlib.nullcontext">]>,
      !py.protocol<"Callable", [!py.contract<"contextlib.nullcontext">, !py.union<!py.type<!py.contract<"builtins.BaseException">>, !py.literal<None>>, !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance"]
  } {}
  py.class @AsyncContextManager attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T", "ExitT"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.param_defaults = [!py.union<!py.contract<"builtins.bool">, !py.literal<None>>],
    method_names = ["__aenter__", "__aexit__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.AsyncContextManager">] -> [!py.protocol<"Awaitable", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"typing.AsyncContextManager">, !py.union<!py.type<!py.contract<"builtins.BaseException">>, !py.literal<None>>, !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.protocol<"Awaitable", [!py.contract<"$ExitT">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @Future attributes {
    base_names = ["Awaitable", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.param_variance = ["invariant"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    method_names = ["__await__", "__iter__", "result", "exception", "done",
                    "cancelled", "cancel", "add_done_callback",
                    "remove_done_callback", "set_result", "set_exception",
                    "get_loop"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">] -> [!py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">, !py.union<!py.contract<"builtins.object">, !py.literal<None>>] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">, !py.protocol<"Callable", [!py.self] -> [!py.contract<"builtins.object">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">, !py.protocol<"Callable", [!py.self] -> [!py.contract<"builtins.object">]>] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">, !py.union<!py.type<!py.contract<"builtins.BaseException">>, !py.contract<"builtins.BaseException">>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future">] -> [!py.contract<"asyncio.AbstractEventLoop">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance"]
  } {}
  py.class @Task attributes {
    base_names = ["Future", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    method_names = ["get_coro", "get_name", "set_name", "get_context",
                    "get_stack", "print_stack", "cancelling", "uncancel"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task">] -> [!py.union<!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$T">]>, !py.literal<None>>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task">, !py.contract<"builtins.object">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task">] -> [!py.contract<"contextvars.Context">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task">, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.contract<"builtins.list">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task">, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"typing.TextIO">, !py.literal<None>>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance"]
  } {}
}
