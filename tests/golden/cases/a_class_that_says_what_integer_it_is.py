# What: the three protocols a class uses to say "here is my integer" --
# `__index__` for a sequence subscript, and `__floor__` / `__ceil__` /
# `__trunc__` for the math functions. Running it is what shows each one reached
# the class's own method: the manifest overloads these calls would otherwise
# take answer with a number too, just the wrong one.
import math


class Position:
    def __init__(self, n: int) -> None:
        self.n = n

    def __index__(self) -> int:
        return self.n

    def __floor__(self) -> int:
        return self.n - 1

    def __ceil__(self) -> int:
        return self.n + 1

    def __trunc__(self) -> int:
        return self.n


values = [10, 20, 30, 40]
print(values[Position(1)], "abcd"[Position(2)], (7, 8, 9)[Position(0)])
print(math.floor(Position(5)), math.ceil(Position(5)), math.trunc(Position(5)))
print(math.floor(1.7), math.ceil(1.2), math.trunc(-1.7))
