// Contract manifest for the `collections.abc` protocol tower.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/_collections_abc.pyi
//   (re-exported by stdlib/collections/abc.pyi)
//
// Protocol names are bare (`Iterable`, not `collections.abc.Iterable`): the
// dialect spells them `!py.protocol<"Iterable", [...]>`. The tower roots at
// @Protocol (declared in modules/typing.mlir) and is expanded through base
// substitution by the protocol table.

module attributes {
  ly.typing.manifest
} {
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
  py.class @AbstractSet attributes {
    base_names = ["Collection", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.base_args = [[!py.contract<"$T">], []]
  } {}
  py.class @MutableSet attributes {
    base_names = ["AbstractSet", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.base_args = [[!py.contract<"$T">], []]
  } {}

  py.class @Generator attributes {
    base_names = ["Iterator", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["Y", "S", "R"],
    ly.typing.param_variance = ["covariant", "contravariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$Y">], []],
    method_names = ["__next__", "send", "throw", "throw", "throw", "close",
                    "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.contract<"$S">] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.contract<"builtins.BaseException">] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.type<!py.contract<"builtins.BaseException">>, !py.union<!py.contract<"builtins.BaseException">, !py.contract<"builtins.object">>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.contract<"builtins.BaseException">, !py.literal<None>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>] -> [!py.protocol<"Generator", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}
  py.class @Awaitable attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
    method_names = ["__await__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"Awaitable", [!py.contract<"$T">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$T">]>]>
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
      !py.protocol<"Callable", [!py.protocol<"Coroutine", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.contract<"$S">] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.protocol<"Coroutine", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.type<!py.contract<"builtins.BaseException">>, !py.union<!py.contract<"builtins.BaseException">, !py.contract<"builtins.object">>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.protocol<"Coroutine", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.contract<"builtins.BaseException">, !py.literal<None>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.protocol<"Coroutine", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance"]
  } {}
  py.class @AsyncIterable attributes {
    base_names = ["Protocol"], ly.typing.abstract, ly.typing.protocol,
    ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
    method_names = ["__aiter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"AsyncIterable", [!py.contract<"$T">]>] -> [!py.protocol<"AsyncIterator", [!py.contract<"$T">]>]>
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
      !py.protocol<"Callable", [!py.protocol<"AsyncIterator", [!py.contract<"$T">]>] -> [!py.protocol<"AsyncIterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.protocol<"AsyncIterator", [!py.contract<"$T">]>] -> [!py.protocol<"Awaitable", [!py.contract<"$T">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @AsyncGenerator attributes {
    base_names = ["AsyncIterator", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["Y", "S"],
    ly.typing.base_args = [[!py.contract<"$Y">], []],
    method_names = ["__anext__", "asend", "athrow", "athrow", "aclose"],
    method_contracts = [
      !py.protocol<"Callable", [!py.protocol<"AsyncGenerator", [!py.contract<"$Y">, !py.contract<"$S">]>] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$Y">]>]>,
      !py.protocol<"Callable", [!py.protocol<"AsyncGenerator", [!py.contract<"$Y">, !py.contract<"$S">]>, !py.contract<"$S">] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$Y">]>]>,
      !py.protocol<"Callable", [!py.protocol<"AsyncGenerator", [!py.contract<"$Y">, !py.contract<"$S">]>, !py.type<!py.contract<"builtins.BaseException">>, !py.union<!py.contract<"builtins.BaseException">, !py.contract<"builtins.object">>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$Y">]>]>,
      !py.protocol<"Callable", [!py.protocol<"AsyncGenerator", [!py.contract<"$Y">, !py.contract<"$S">]>, !py.contract<"builtins.BaseException">, !py.literal<None>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$Y">]>]>,
      !py.protocol<"Callable", [!py.protocol<"AsyncGenerator", [!py.contract<"$Y">, !py.contract<"$S">]>] -> [!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.literal<None>]>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance"]
  } {}

}
