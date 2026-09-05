# What: a local whose annotation stands a rung above the value bound to it.
# The value keeps its own type -- CPython's `v: float = 1` is the int 1 -- and
# everything downstream of the name has to agree with that, including a return
# whose own annotation is the same rung above. Runtime values, because the
# question is which type each name answers with.


def declared_then_returned() -> float:
    v: float = 1
    return v


def declared_then_rebound() -> float:
    v: float = 1
    v = 2.5
    return v


def through_a_cell() -> float:
    v: float = 1

    def bump() -> None:
        nonlocal v
        v = 2

    bump()
    return v


def flagged() -> int:
    n: int = True
    return n


def in_a_loop() -> float:
    total: float = 0
    for i in range(4):
        total = total + i
    return total


print(declared_then_returned(), declared_then_rebound())
print(through_a_cell(), flagged())
print(in_a_loop())
