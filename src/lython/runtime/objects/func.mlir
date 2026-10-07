// `function`, `staticmethod` and `classmethod` -- CPython's
// Objects/funcobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// The contracts only: functions are compiled, and calling one is the
// lowering's, not a runtime method's.

module attributes {
  ly.typing.manifest
} {
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
}
