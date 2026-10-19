# WHAT: `f(*xs)` with a list or a `tuple[T, ...]`, and `f(**d)` with a
# `dict[str, T]`, reach a function's, a method's and a constructor's *args and
# **kwargs -- a function forwarding its own, a function the call reaches
# before its body, explicit values before the starred one, keywords before and
# after the mapping in their order -- and **kwargs is a fresh dict the callee
# can write without writing the caller's. A key the call also gives raises
# CPython's TypeError, with its message.
#
# WHY THIS IS RUN: how many members a sequence has and which keys a mapping
# holds are run-time facts.
from collections.abc import Generator


def total(*args: int) -> int:
    s = 0
    for a in args:
        s += a
    return s


def show(**kwargs: int) -> str:
    return ",".join(k + "=" + str(v) for k, v in kwargs.items())


def relay(*args: int, **kwargs: int) -> str:
    return str(total(*args)) + "|" + show(**kwargs)


def reaches_later(xs: list[int]) -> int:
    return later(*xs)


def later(*args: int) -> int:
    return len(args)


def head(first: int, *rest: int) -> str:
    return str(first) + str(rest)


def named(a: int, **kw: int) -> str:
    return str(a) + str(kw)


def scribble(**kw: int) -> int:
    kw["written"] = 1
    return len(kw)


def letters(*args: str) -> Generator[str, None, None]:
    for a in args:
        yield a.upper()


xs = [1, 2, 3]
t = (4, 5)
print(total(*xs), total(*t), total(0, *xs), total(*[]))
d = {"b": 2, "a": 1}
print(show(**d), show(z=0, **d), show(**d, z=9))
print(relay(*xs, **d), relay())
print(reaches_later([7, 8]), head(1, *[2, 3]), head(1, *()))
print(list(letters(*["x", "y"])))
mine = {"k": 3}
print(scribble(**mine), mine)


class Fwd:
    def __init__(self, *parts: str, **opts: int) -> None:
        self.parts = parts
        self.opts = opts

    def total(self, *args: int) -> int:
        return total(*args)

    def relay(self, *args: int) -> int:
        return self.total(*args)

    def show(self, first: str, **kwargs: int) -> str:
        return first + ":" + show(**kwargs)


f = Fwd()
print(f.relay(1, 2, 3), f.total(*xs), f.total(1, *xs))
print(f.show("s", **d), f.show("t", w=0, **d))
made = Fwd(*["a", "b"], **{"k": 1})
print(made.parts, made.opts)

for attempt in range(4):
    try:
        if attempt == 0:
            show(b=0, **d)
        elif attempt == 1:
            named(1, **{"a": 2})
        elif attempt == 2:
            f.show("u", b=1, **d)
        else:
            show(**d, b=5)
    except TypeError as e:
        print("TypeError:", e)
