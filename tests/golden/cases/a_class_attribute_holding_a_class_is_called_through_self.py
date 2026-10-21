# WHAT: `self.resultclass(...)` where `resultclass` is a class attribute an
# instance may replace with a subclass (with keywords), `self._f()` on an
# `Optional` callable field after a test against None, and a class attribute
# an `__init__` overwrites only under `if limit is not None`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which class constructs and
# which value the field holds are run-time answers. Each was refused: the call
# as a missing method 'resultclass', the field call as a missing method '_f',
# and the overwritten attribute as `int | None` with no `__mul__`.
from collections.abc import Callable


class Res:
    def __init__(self, n: int, *, extra: int | None = None) -> None:
        self.n = n
        self.extra = extra


class Res2(Res):
    pass


class Runner:
    resultclass = Res

    def __init__(self, resultclass: type[Res] | None = None) -> None:
        if resultclass is not None:
            self.resultclass = resultclass

    def make(self) -> Res:
        return self.resultclass(3, extra=4)


def hello() -> None:
    print("hello")


class Holder:
    def __init__(self, f: Callable[[], None] | None = None) -> None:
        self._f = f

    def go(self) -> None:
        if self._f is not None:
            self._f()
        else:
            print("nothing")


class Limit:
    limit = 5

    def __init__(self, limit: int | None = None) -> None:
        if limit is not None:
            self.limit = limit

    def twice(self) -> int:
        return self.limit * 2


made = Runner().make()
print(type(made).__name__, made.n, made.extra)
print(type(Runner(Res2).make()).__name__)
Holder(hello).go()
Holder().go()
print(Limit().twice(), Limit(7).twice())
