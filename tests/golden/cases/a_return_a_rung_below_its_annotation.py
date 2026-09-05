# What: a return value that stands a rung below the return annotation keeps
# its own type, exactly as a local, a parameter and a parameter default
# already do. Runtime values, because the whole question is which type the
# call answers with: `half(7)` is CPython's int 3, not 3.0, and `positive(3)`
# is True, not 1.


def half(n: int) -> float:
    return n // 2


def positive(n: int) -> int:
    return n > 0


def scaled(n: int) -> float:
    total = 0
    for i in range(n):
        total += i
    return total


def widened(n: int) -> float:
    return n / 2


class Meter:
    def __init__(self, ticks: int) -> None:
        self.ticks = ticks

    def reading(self) -> float:
        return self.ticks * 2


print(half(7), half(8))
print(positive(3), positive(-1))
print(scaled(4), widened(7))
print(Meter(3).reading(), Meter(3).reading() + 0.5)
