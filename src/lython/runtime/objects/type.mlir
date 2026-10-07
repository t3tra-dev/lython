// `type` -- CPython's Objects/typeobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// The contract only: a class is resolved at compile time, and no `type`
// method has a runtime implementation here.

module attributes {
  ly.typing.manifest
} {
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
}
