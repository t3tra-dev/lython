# What this pins: an argument standing a rung below its declared parameter in
# the numeric tower reaches a body emitted FOR THAT RUNG, so nothing is
# converted at the boundary and the value keeps its own type all the way
# through -- including out of the return, whose annotation is inert.
#
# Why this needs to run rather than assert on a diagnostic: the refusal it
# replaces was a diagnostic, but the repair is only correct if the ANSWERS are
# CPython's, and the wrong repair (convert at the boundary) also removes the
# diagnostic. `print(f(3))` is 6 and not 6.0, `print(n)` for a bool parameter
# is True and not 1: only running can tell those apart.
#
# Every expected line is python3.14's.

# --- the same function reached at two rungs gets two bodies ----------------
def double(x: float) -> float:
    return x * 2


print(double(3))
print(double(3.0))
print(double(True))
print(double(3), double(3.0))


# --- the argument keeps its own type where the body can OBSERVE it ---------
def show(x: float) -> None:
    print(x)


show(3)
show(3.5)
show(True)


def count(n: int) -> None:
    print(n)


count(True)
count(False)
count(7)


# --- one call may mix rungs, and each position decides on its own ----------
def total(a: float, b: int, c: float) -> float:
    return a + b + c


print(total(1, 2, 3))
print(total(1.0, 2, 3))
print(total(1, 2, 3.5))
print(total(1.0, 2, 3.5))
print(total(True, True, True))


# --- a declared rung that is already met is untouched ----------------------
def half(x: float) -> float:
    return x / 2


print(half(5.0))
print(half(5))


# --- the specialized body is a real body, and it branches -------------------
# Not RECURSES: a specialized body that calls itself has to type that call
# against the DECLARED signature, which is the refusal the specialization
# exists to lift, so `def power(base: float, n: int)` reached by `power(2, 3)`
# is still refused. Recorded with the mechanism in
# tests/probe/wb_argument_boundary_numeric_tower.py rather than pinned here.
def clamp(x: float, lo: float) -> float:
    if x < lo:
        return lo
    return x


print(clamp(3, 5))
print(clamp(7, 5))
print(clamp(3.0, 5.0))


# --- a parameter the tower does not reach still goes down the declared body -
class Animal:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Dog(Animal):
    def __init__(self, n: int) -> None:
        self.n = n


def take(a: Animal) -> int:
    return a.n


print(take(Dog(4)))
print(take(Animal(9)))


def text(v: str) -> int:
    return len(v)


print(text("abcd"))
