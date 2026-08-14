# A function stored in a Callable-typed field and called back. Execution is
# needed because three layers had to come off for this to run and each one
# could be undone without the others noticing: a wrong callee (the field
# dispatched as a method), a wrong target (a primitive-i64 clone offered as a
# callable value), and a store refused for comparing a runtime representation
# against a logical contract. Only the printed values say all three are gone.
#
# Both call syntaxes are here -- through the field directly and through a local
# bound from it -- because they took different paths to the same defect, and
# both result types, since the int one is the only one with a primitive clone.

from typing import Callable


def seven() -> int:
    return 7


def greeting() -> str:
    return "hi"


def twice(n: int) -> int:
    return n * 2


class Holder:
    def __init__(self, f: Callable[[], int]) -> None:
        self._f: Callable[[], int] = f

    def direct(self) -> int:
        return self._f()

    def through_local(self) -> int:
        g: Callable[[], int] = self._f
        return g()


class Greeter:
    def __init__(self, f: Callable[[], str]) -> None:
        self._f: Callable[[], str] = f

    def call(self) -> str:
        return self._f()


class Doubler:
    def __init__(self, f: Callable[[int], int]) -> None:
        self._f: Callable[[int], int] = f

    def call(self, n: int) -> int:
        return self._f(n)


h = Holder(seven)
print(h.direct(), h.through_local())
print(Greeter(greeting).call())
print(Doubler(twice).call(5), Doubler(twice).call(0))

# A plain local callable, which already worked: a repair that reroutes it is
# caught here rather than in whatever uses it next.
g: Callable[[], int] = seven
print(g())
