# WHAT: a method that forwards its `*args` and `**kwargs` to a callable -- the
# callable form of assertRaises, written with `Callable[..., object]` and with
# a ParamSpec, addCleanup keeping a lambda that forwards them for later, one
# method forwarding to another, and a tuple starred into a method -- passes
# the arguments its call gave, a single one included, by position and by
# name (keyword-only too), while the body may still read the dict.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which exception a call
# raises and what the cleanups print are run-time answers. Each forwarding
# was refused: a one-element pack has no arity in its type, `**kwargs` has no
# names in its type, and `P.args` was a contract nothing is.
from typing import Callable


def parse(s: str, base: int = 10) -> int:
    return int(s, base)


def show(a: int, b: str = "b", *, c: float = 1.5) -> str:
    return "%d %s %.1f" % (a, b, c)


def close(name: str) -> None:
    print("close", name)


def add(a: int, b: int = 0) -> None:
    print("add", a + b)


class TestCase:
    def __init__(self) -> None:
        self._cleanups: list[Callable[[], object]] = []

    def assertRaises(self, expected: type[BaseException],
                     callable: Callable[..., object], *args: object,
                     **kwargs: object) -> str:
        try:
            callable(*args, **kwargs)
        except BaseException as e:
            if isinstance(e, expected):
                return "raised " + type(e).__name__
            raise
        return expected.__name__ + " not raised"

    def assertRaisesSpec[**P](self, expected: type[BaseException],
                              callable: Callable[P, object],
                              *args: P.args, **kwargs: P.kwargs) -> str:
        try:
            callable(*args, **kwargs)
        except BaseException as e:
            return "raised " + str(isinstance(e, expected))
        return "not raised"

    def addCleanup[**P](self, function: Callable[P, object], /,
                        *args: P.args, **kwargs: P.kwargs) -> None:
        self._cleanups.append(lambda: function(*args, **kwargs))

    def doCleanups(self) -> None:
        while self._cleanups:
            function = self._cleanups.pop()
            function()

    def call(self, f: Callable[..., str], *args: object,
             **kwargs: object) -> str:
        if "c" in kwargs:
            print("c given:", kwargs["c"], sorted(kwargs))
        return f(*args, **kwargs)

    def outer(self, f: Callable[..., str], *args: object,
              **kwargs: object) -> str:
        return self.call(f, *args, **kwargs)


t = TestCase()
print(t.assertRaises(ValueError, parse, "x"),
      t.assertRaises(ValueError, parse, "12", base=1),
      t.assertRaises(ValueError, parse, "12", base=8))
print(t.assertRaisesSpec(ValueError, parse, "ff"),
      t.assertRaisesSpec(ValueError, parse, "ff", 16))
print(t.call(show, 1), t.call(show, 2, "x"), t.call(show, 3, c=2.5),
      t.call(show, 4, b="y", c=0.5))
print(t.outer(show, 5), t.outer(show, 6, "z", c=3.5))
pair = (7, "w")
print(t.call(show, *pair))
t.addCleanup(close, "x")
t.addCleanup(add, 1, b=2)
t.addCleanup(add, 5)
t.doCleanups()
