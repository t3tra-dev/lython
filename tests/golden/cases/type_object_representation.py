# What this pins: a `type[X]` value travels through a parameter, a field, a
# local, a return and a generator suspension, and constructing through it
# builds X. Its physical shape is EMPTY -- which class it is, is decided by its
# type -- so every one of those positions carries it for free.
#
# Why this needs to run rather than assert on a diagnostic: the refusals it
# replaces were diagnostics, but the property that matters is that the class
# CONSTRUCTED is the one the value names. A representation that carried the
# wrong class would compile and print the wrong instance, and only the values
# below can tell those apart.
#
# Every expected line is python3.14's.


class A:
    def __init__(self, n: int) -> None:
        self.n: int = n
        self.tag: str = "A"


class B:
    def __init__(self, n: int) -> None:
        self.n: int = n * 100
        self.tag: str = "B"


# --- a local binding, and rebinding it to a different class ----------------
t = A
a = t(1)
print(a.n, a.tag)
t2 = B
b = t2(1)
print(b.n, b.tag)


# --- a parameter -----------------------------------------------------------
def build(t: type[A], n: int) -> A:
    return t(n)


print(build(A, 3).n)


def build_b(t: type[B], n: int) -> B:
    return t(n)


print(build_b(B, 3).n)


# --- a return value, then called ------------------------------------------
def chosen(t: type[A]) -> type[A]:
    return t


picked = chosen(A)
print(picked(4).n)
print(chosen(A)(5).n)


# --- a field, read back and called ----------------------------------------
class Registry:
    def __init__(self, t: type[A], label: str) -> None:
        self.t: type[A] = t
        self.label: str = label

    def make(self, n: int) -> A:
        return self.t(n)


r = Registry(A, "reg")
print(r.label)
print(r.make(6).n)
print(r.t(7).n, r.t(7).tag)


# --- the receiver of a field read still runs ------------------------------
made: int = 0


def fresh(label: str) -> Registry:
    global made
    made += 1
    return Registry(A, label)


print(fresh("one").t(8).n)
print(made)


# --- a generator constructing at more than one yield ----------------------
# One construction always worked: the shared py.type.object died before the
# suspend. Two or more left it live across one, and a type object had no lane.
from typing import Iterator


def three() -> Iterator[A]:
    yield A(1)
    yield A(2)
    yield A(3)


total = 0
for item in three():
    total += item.n
print(total)


def counted() -> Iterator[B]:
    i = 0
    while i < 3:
        yield B(i)
        i += 1


parts = 0
for item in counted():
    parts += item.n
print(parts)
