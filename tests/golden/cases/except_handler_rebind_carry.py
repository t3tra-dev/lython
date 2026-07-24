# A local rebound inside an except handler is visible after the try. When the
# try body always raises it contributes no fall-through lane, and requiring
# one used to drop every carried lane -- so the handler's assignment was
# silently discarded and the post-try read answered the pre-try value.
def always_raises() -> int:
    out = 0
    try:
        raise ValueError("tag")
    except ValueError:
        out = 7
    return out


print(always_raises())


def uses_binding() -> int:
    out = 0
    try:
        raise ValueError("tag")
    except ValueError as e:
        out = len(str(e))
    return out


print(uses_binding())


def carries_str() -> str:
    out = "no"
    try:
        raise ValueError("tag")
    except ValueError as e:
        out = str(e)
    return out


print(carries_str())


def may_raise(n: int) -> int:
    out = 0
    try:
        if n > 0:
            raise ValueError("tag")
        out = 1
    except ValueError:
        out = 7
    return out


print(may_raise(1))
print(may_raise(0))


def two_handlers(kind: int) -> str:
    label = "none"
    try:
        if kind == 1:
            raise ValueError("v")
        raise KeyError("k")
    except ValueError:
        label = "value"
    except KeyError:
        label = "key"
    return label


print(two_handlers(1))
print(two_handlers(2))


def several_locals() -> str:
    a = 0
    b = "x"
    try:
        raise ValueError("boom")
    except ValueError as e:
        a = len(str(e))
        b = str(e) + "!"
    return str(a) + b


print(several_locals())


def at_module_level() -> None:
    pass


count = 0
name = "start"
try:
    raise RuntimeError("mod")
except RuntimeError as e:
    count = len(str(e))
    name = str(e)
print(count, name)
