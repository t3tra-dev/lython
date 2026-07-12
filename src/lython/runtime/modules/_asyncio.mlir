// Contract manifest for CPython `_asyncio` runtime classes.

module attributes {ly.typing.module = "_asyncio"} {
  py.class @Future attributes {
    base_names = ["Awaitable", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.param_variance = ["invariant"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    method_names = ["__init__", "__await__", "__iter__", "result", "exception",
                    "done", "cancelled", "cancel", "cancel",
                    "add_done_callback", "remove_done_callback", "set_result",
                    "set_exception", "get_loop"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>, !py.union<!py.contract<"builtins.object">, !py.literal<None>>] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>, !py.protocol<"Callable", [!py.self] -> [!py.contract<"builtins.object">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>, !py.protocol<"Callable", [!py.self] -> [!py.contract<"builtins.object">]>] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>, !py.union<!py.type<!py.contract<"builtins.BaseException">>, !py.contract<"builtins.BaseException">>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Future", [!py.contract<"$T">]>] -> [!py.contract<"asyncio.AbstractEventLoop">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance"]
  } {}

  py.class @FutureIter attributes {
    base_names = ["Generator"], ly.typing.final, ly.typing.params = ["T"],
    ly.typing.base_args = [[!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]],
    method_names = ["__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_asyncio.FutureIter", [!py.contract<"$T">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]>]>
    ],
    method_kinds = ["instance"]
  } {}

  py.class @Task attributes {
    base_names = ["Future", "Protocol"], ly.typing.abstract,
    ly.typing.protocol, ly.typing.params = ["T"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    method_names = ["__new__", "__init__", "__await__", "__iter__",
                    "get_coro", "get_name", "set_name", "get_context",
                    "get_stack", "print_stack", "cancelling", "uncancel"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"_asyncio.Task">>, !py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$T">]>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>, !py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>] -> [!py.union<!py.protocol<"Coroutine", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"$T">]>, !py.literal<None>>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>, !py.contract<"builtins.object">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>] -> [!py.contract<"contextvars.Context">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.contract<"builtins.list">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"typing.TextIO">, !py.literal<None>>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_asyncio.Task", [!py.contract<"$T">]>] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance"]
  } {}

  py.class @TaskIter attributes {
    base_names = ["Generator"], ly.typing.final, ly.typing.params = ["T"],
    ly.typing.base_args = [[!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]],
    method_names = ["__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_asyncio.TaskIter", [!py.contract<"$T">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$T">]>]>
    ],
    method_kinds = ["instance"]
  } {}
}
