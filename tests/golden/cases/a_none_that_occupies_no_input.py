# What: a value whose runtime shape is EMPTY standing in a logical input --
# a local bound to `None` that a nested callable captures, a parameter annotated
# `None`, and the same capture carried out on a returned closure. None of them
# occupies an ABI input, so the call's bound check read a trailing one as an
# overflow and the returned one named a result index past the end. Runtime
# values, because the question is which value the callee reads; the same capture
# of an `int | None` has always worked, a union having lanes.
from typing import Callable


def captured() -> str:
    v = None

    def show() -> str:
        return "n" if v is None else "s"

    return show()


def with_siblings() -> str:
    v = None
    n = 5
    tag = "t"

    def before() -> str:
        return ("n" if v is None else "s") + str(n)

    def after() -> str:
        return tag + ("n" if v is None else "s")

    return before() + after()


def by_lambda() -> str:
    v = None
    f = lambda: "n" if v is None else "s"
    return f()


def returned() -> Callable[[], str]:
    v = None

    def show() -> str:
        return "n" if v is None else "s"

    return show


class Holder:
    def run(self) -> str:
        v = None

        def show() -> str:
            return "n" if v is None else "s"

        return show()


def takes_none(v: None) -> str:
    return "n" if v is None else "s"


def widened() -> str:
    v: int | None = None

    def show() -> str:
        return "n" if v is None else str(v)

    first = show()
    v = 3
    return first + show()


print(captured())
print(with_siblings())
print(by_lambda())
print(returned()())
print(Holder().run())
print(takes_none(None))
print(widened())
