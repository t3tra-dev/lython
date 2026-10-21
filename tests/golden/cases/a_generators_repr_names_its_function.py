# What: a generator's repr names its function's qualified name, by every route
# that renders it -- repr, str, an f-string, held as an Iterator or as object,
# inside a list, a tuple and a dict -- and its class is CPython's `generator`
# of `builtins`. The address after " at " is cut off.
# Why run: the name is rendered at run time from the generator object.
from collections.abc import Iterator


def gen() -> Iterator[int]:
    yield 1


class C:
    def m(self) -> Iterator[int]:
        yield 2


def outer() -> Iterator[int]:
    def inner() -> Iterator[int]:
        yield 3

    return inner()


def cut(s: str) -> str:
    while " at 0x" in s:
        start = s.index(" at 0x")
        s = s[:start] + s[s.index(">", start):]
    return s


g = gen()
it: Iterator[int] = C().m()
o: object = outer()
print(cut(repr(g)), cut(str(it)), cut(f"{o}"), cut(f"{g!r}"))
print(cut(repr([g, it])), cut(repr((o, 1))))
d: dict[str, object] = {"k": gen()}
print(cut(repr(d)))
print(type(g).__name__, type(g).__module__, type(g).__qualname__, type(g))
