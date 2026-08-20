# What this pins: `@dataclass(order=True)`.
#
#     @dataclass(order=True)
#     class Rec: ...
#     # dataclass argument 'order' is not supported
#
# Which is a refusal of `sorted(rows)` over a record type -- the reason to
# reach for a dataclass in the first place. CPython's dataclasses builds
# `(self.f0, ...) < (other.f0, ...)` and lets tuple's own ordering decide, so
# the four methods are synthesized as exactly that expression: the answer comes
# from the manifest tuple comparison rather than from a second implementation
# of lexicographic order, and it agrees with `sorted(key=...)` over the same
# fields by construction.
#
# Why this must run: field order is the answer. Two records that agree on the
# first field and differ on the second are what separates "compares the tuple"
# from "compares the first field", and a sort is the only thing that shows the
# whole ordering at once.
#
# ⛔ CPython returns NotImplemented for a foreign operand and lets the reflected
# method try; here `other` is typed as this class, so a foreign operand is
# refused at the call instead -- earlier, and with the class named.
#
# ⛔ `max(rows)` and `min(rows)` are still refused ("needs an element type the
# fold can seed"): the fold seeds its accumulator with a value of the element
# type that the seen-flag keeps it from reading, and a user class has no such
# constant. `sorted(rows)[-1]` is the spelling that works.
from dataclasses import dataclass


@dataclass(order=True)
class Rec:
    score: int
    name: str


@dataclass(order=True)
class One:
    x: int


@dataclass(order=False)
class Plain:
    x: int


rows = [Rec(2, "b"), Rec(1, "a"), Rec(2, "a"), Rec(1, "z")]
print(sorted(rows))
print(sorted(rows, reverse=True))
print(Rec(1, "a") < Rec(2, "a"), Rec(2, "a") < Rec(2, "b"), Rec(2, "b") < Rec(2, "a"))
print(Rec(2, "b") <= Rec(2, "b"), Rec(3, "z") > Rec(2, "a"), Rec(1, "a") >= Rec(1, "a"))
print(Rec(1, "a") == Rec(1, "a"), Rec(1, "a") == Rec(1, "b"))
print(sorted([One(3), One(1), One(2)]), One(1) < One(2))
print(Plain(1) == Plain(1))
print(sorted(rows)[-1], sorted(rows)[0])

hits = 0
i = 0
while i < 200:
    left = Rec(i % 3, "a")
    right = Rec(i % 2, "b")
    if left < right:
        hits += 1
    i += 1
print("hits", hits)
