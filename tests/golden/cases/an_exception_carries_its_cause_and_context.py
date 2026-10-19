# WHAT: `__cause__`, `__context__` and `__suppress_context__` read the chain a
# raise records on the exception itself -- `raise X from Y`, `from None`, the
# exception a raise interrupts (inside a handler, a called function, a
# runtime raise), a program class as either end -- and keep it after the
# handler is gone. A re-raise leaves it alone, a raise that would close a
# cycle cuts it, an exception that left its handler is not chained to a later
# one, a generator chains what its caller handles and what it is suspended
# in, and except* hands each split group the original's chain.
#
# WHY THIS IS RUN: the chain is what the running program raised and handled.
from collections.abc import Generator
from contextlib import contextmanager


class AppError(Exception):
    pass


class Detail(AppError):
    def __init__(self, code: int) -> None:
        super().__init__("detail " + str(code))
        self.code = code


def lookup(d: dict[str, int], k: str) -> int:
    return d[k]


def show(e: BaseException) -> str:
    return repr(e.__cause__) + " " + repr(e.__context__) + " " + str(
        e.__suppress_context__)


try:
    try:
        raise KeyError("k")
    except KeyError as inner:
        raise ValueError("from") from inner
except ValueError as e:
    print("from:", show(e), type(e.__cause__).__name__)

try:
    try:
        raise KeyError("a")
    except KeyError:
        raise ValueError("implicit")
except ValueError as e:
    print("implicit:", show(e))

try:
    raise ValueError("alone")
except ValueError as e:
    print("alone:", show(e), e.__cause__ is None, e.__context__ is None)

try:
    try:
        raise KeyError("hidden")
    except KeyError:
        raise ValueError("shown") from None
except ValueError as e:
    print("from None:", show(e))

saved: list[BaseException] = []


def capture() -> None:
    try:
        try:
            lookup({}, "missing")
        except KeyError as k:
            raise AppError("wrapped") from k
    except AppError as e:
        saved.append(e)


capture()
print("stored:", show(saved[0]))

try:
    try:
        raise ValueError("first")
    except ValueError:
        lookup({"a": 1}, "b")
except KeyError as k:
    print("runtime raise:", show(k))

try:
    try:
        raise KeyError("x")
    except KeyError:
        try:
            raise ValueError("y")
        except ValueError as v:
            raise v
except ValueError as v2:
    print("re-raise:", show(v2))

try:
    try:
        raise KeyError("p")
    except KeyError:
        raise
except KeyError as p:
    print("bare raise:", show(p))

try:
    try:
        raise Detail(7)
    except Detail as d:
        raise AppError("app") from d
except AppError as a:
    c = a.__cause__
    if isinstance(c, Detail):
        print("program cause:", c.code, repr(a.__context__))

held: list[KeyError] = []
try:
    try:
        raise KeyError("A")
    except KeyError as a1:
        held.append(a1)
        raise ValueError("B")
except ValueError as b:
    try:
        raise held[0]
    except KeyError as a2:
        print("cycle cut:", repr(a2.__context__), repr(b.__context__))


def boom() -> None:
    raise RuntimeError("boom")


try:
    try:
        raise KeyError("h")
    except KeyError:
        boom()
except RuntimeError as r:
    print("called:", show(r))

try:
    try:
        raise KeyError("left")
    except KeyError:
        raise ValueError("escaped")
except ValueError:
    pass
try:
    raise IndexError("later")
except IndexError as i:
    print("after a handled escape:", show(i))


def body_raises() -> Generator[int, None, None]:
    yield 1
    raise KeyError("in body")


def suspended_in_handler() -> Generator[int, None, None]:
    try:
        raise IndexError("held")
    except IndexError:
        yield 2


try:
    try:
        raise ValueError("caller")
    except ValueError:
        g = body_raises()
        next(g)
        next(g)
except KeyError as k:
    print("generator body:", show(k))

h = suspended_in_handler()
next(h)
try:
    h.throw(RuntimeError("thrown"))
except RuntimeError as r:
    print("thrown in:", show(r))


@contextmanager
def converting() -> Generator[None, None, None]:
    try:
        yield
    except KeyError as e:
        raise LookupError("converted") from e


try:
    with converting():
        raise KeyError("inner")
except LookupError as le:
    print("contextmanager:", show(le))

try:
    try:
        raise KeyError("ctx")
    except KeyError:
        raise ExceptionGroup("eg", [ValueError("a"), TypeError("b")])
except* ValueError as eg:
    print("split V:", show(eg))
except* TypeError as eg2:
    print("split T:", show(eg2))
