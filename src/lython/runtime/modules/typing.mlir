// Contract manifest for `typing` / `_typeshed` support protocols.
//
// Signature sources (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/typing.pyi
//   https://github.com/python/typeshed/blob/main/stdlib/_typeshed/__init__.pyi
//
// Holds the @Protocol root, the Callable protocol, the Supports* structural
// protocols, and the context-manager protocols.

module attributes {
  ly.typing.manifest
} {
  py.class @Protocol attributes {base_names = ["object"], ly.typing.abstract,
                                ly.typing.protocol} {}
  py.class @Callable attributes {base_names = ["Protocol"],
                                ly.typing.abstract, ly.typing.protocol} {}

  py.class @TextIO attributes {base_names = ["object"], ly.typing.abstract} {}

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
  py.class @AsyncContextManager attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T", "ExitT"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.param_defaults = [!py.union<!py.contract<"builtins.bool">, !py.literal<None>>],
    method_names = ["__aenter__", "__aexit__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"AsyncContextManager", [!py.contract<"$T">, !py.contract<"$ExitT">]>] -> [!py.protocol<"Awaitable", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.protocol<"AsyncContextManager", [!py.contract<"$T">, !py.contract<"$ExitT">]>, !py.union<!py.type<!py.contract<"builtins.BaseException">>, !py.literal<None>>, !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.protocol<"Awaitable", [!py.contract<"$ExitT">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

}
