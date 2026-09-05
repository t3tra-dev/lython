# What: a `try` around a call the dispatch cannot resolve to one target. The
# dispatch puts each candidate's call in a block it creates, and the call-site
# marker that ties a call to its handler is keyed on the insertion block -- so
# the exception walked out of the try. Runtime values, because the question is
# whether the handler runs: the same program with ONE candidate target catches,
# which is why every smaller spelling of it was right.
from typing import Callable


class Flaky:
    def __init__(self, fail_times: int) -> None:
        self.remaining: int = fail_times
        self.calls: int = 0

    def run(self, v: int) -> int:
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ValueError("not yet")
        return v * 2


class Steady:
    def run(self, v: int) -> int:
        return v + 1


def guarded(op: Callable[[int], int], v: int) -> int:
    try:
        return op(v)
    except ValueError:
        return -1


def retry(op: Callable[[int], int], v: int, attempts: int) -> str:
    for _ in range(attempts):
        try:
            return str(op(v))
        except ValueError:
            continue
    return "gave up"


def two_guards(op: Callable[[int], int], v: int) -> str:
    try:
        first = op(v)
        second = op(v)
        return str(first + second)
    except ValueError:
        return "caught"


flaky = Flaky(1)
steady = Steady()
print(guarded(flaky.run, 3), guarded(steady.run, 3))
print(retry(Flaky(2).run, 5, 4))
print(retry(Flaky(9).run, 5, 2))
print(two_guards(Flaky(1).run, 4), two_guards(steady.run, 4))
