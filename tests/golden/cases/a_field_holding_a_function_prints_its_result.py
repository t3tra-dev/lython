# WHAT: a call through a function held in a field, printed directly, prints
# the result the function returned. Inference typed `b.k(3)` as `object`, so
# print took its one-object path and handed the union result to the lowering,
# which refused it; `r = b.k(3)` then `print(r)` rendered it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: what failed is which path
# print renders the value through, and the answer is the printed text.
from typing import Callable


def g(x: int) -> int | None:
    return x + 1


def h(x: int) -> int | float:
    return x * 1.5


class UBox:
    def __init__(self, k: Callable[[int], int | None]) -> None:
        self.k = k


class FBox:
    def __init__(self, k: Callable[[int], int | float]) -> None:
        self.k = k


b = UBox(g)
print(b.k(3))
print(FBox(h).k(3))
