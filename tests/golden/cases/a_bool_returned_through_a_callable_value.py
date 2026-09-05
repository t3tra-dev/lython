# What: a bool-returning function reached through a value the dispatch cannot
# name statically -- a list element, a cell, a field, a dict entry. The miss arm
# of that dispatch materializes a dead result, and a bool's dead value is a
# truth bit rather than a header the frame allocated. Runtime values, because
# the question is which answer each call gives; the same containers of int- and
# str-returning functions have always compiled.
from typing import Callable


def is_zero(k: int) -> bool:
    return k == 0


def is_positive(k: int) -> bool:
    return k > 0


def through_list(k: int) -> list[bool]:
    fs: list[Callable[[int], bool]] = [is_zero, is_positive]
    return [f(k) for f in fs]


def through_cell(k: int) -> bool:
    f = is_zero
    f = is_positive

    def ask(v: int) -> bool:
        return f(v)

    return ask(k)


def through_dict(k: int) -> bool:
    d: dict[str, Callable[[int], bool]] = {"z": is_zero, "p": is_positive}
    return d["z"](k)


class Holder:
    def __init__(self, f: Callable[[int], bool]) -> None:
        self.f: Callable[[int], bool] = f

    def ask(self, k: int) -> bool:
        return self.f(k)


print(through_list(0), through_list(3))
print(through_cell(0), through_cell(3))
print(through_dict(0), through_dict(3))
print(Holder(is_positive).ask(0), Holder(is_positive).ask(3))
