# What: a closure over a local bound to a bool, RETURNED from the function that
# built it. The capture rides out as a trailing result lane, and a bool's lane
# is a truth bit -- no header for an ownership contract to anchor, and nothing
# for a release to discharge. Runtime values, because the question is which
# value each returned closure reads; the same closure over an int, a str or a
# float returns fine, and so does one over a bool PARAMETER.
from typing import Callable, Iterator


def sign() -> Callable[[int], int]:
    flag = True

    def pick(v: int) -> int:
        return v if flag else -v

    return pick


def two_flags() -> Callable[[int], int]:
    on = True
    off = False

    def pick(v: int) -> int:
        if off:
            return 0
        return v if on else -v

    return pick


def mixed() -> Callable[[int], str]:
    flag = True
    base = 10
    name = "v"

    def describe(v: int) -> str:
        if flag:
            return name + str(v + base)
        return "off"

    return describe


def from_a_parameter(flag: bool) -> Callable[[int], int]:
    def pick(v: int) -> int:
        return v if flag else -v

    return pick


class Gate:
    def opener(self) -> Callable[[int], int]:
        allowed = True

        def check(v: int) -> int:
            return v if allowed else 0

        return check


def through_a_generator(n: int) -> Iterator[int]:
    f = sign()
    for i in range(n):
        yield f(i - 1)


print(sign()(3), sign()(-3))
print(two_flags()(4))
print(mixed()(2))
print(from_a_parameter(True)(5), from_a_parameter(False)(5))
print(Gate().opener()(7))
print(list(through_a_generator(3)))
