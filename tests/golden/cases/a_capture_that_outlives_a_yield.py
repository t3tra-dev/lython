# What: a generator whose nested def captures values that must survive a yield.
# Each capture is read back out of the function object's closure store, and what
# that slot holds is what the contract's `box` primitive returns -- for a bool a
# three-word singleton header rather than the truth bit. Runtime values, because
# the question is which value each capture hands back once the frame resumes.
from typing import Iterator


def flagged(n: int) -> Iterator[str]:
    flag = True
    off = False
    name = "v"
    base = 10
    ratio = 0.5

    def pick(v: int) -> str:
        if off:
            return "off"
        if flag:
            return name + str(v + base) + str(ratio)
        return "-"

    for i in range(n):
        yield pick(i)


class Gate:
    def rows(self, n: int) -> Iterator[int]:
        allowed = True

        def check(v: int) -> int:
            return v if allowed else 0

        for i in range(n):
            yield check(i)


def both(n: int) -> Iterator[int]:
    on = True
    off = False

    def sign(v: int) -> int:
        if off:
            return 0
        return v if on else -v

    for i in range(n):
        yield sign(i - 1)


print(list(flagged(3)))
print(list(Gate().rows(3)))
print(list(both(4)))
