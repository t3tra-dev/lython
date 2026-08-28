# WHAT: a three-level class chain reached through a base-typed receiver -- a
# method that calls `super()` at every level, and a `@property` each level
# overrides.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which body ran is the
# output. A dispatch that picks the wrong level compiles and prints a string
# that reads like an answer, and a `super()` chain that stops one level early
# is a shorter string, not a diagnostic.
#
# ⛔ TWO LEVELS IS NOT THE TEST. With one subclass the arm for it resolves
# directly and no second dispatcher is built; the middle class's `super()` only
# loses its context once a THIRD class makes that arm dispatch again.
import sys


class A:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def describe(self) -> str:
        return "A" + str(self.n)

    @property
    def scaled(self) -> int:
        return self.n


class B(A):
    def describe(self) -> str:
        return "B(" + super().describe() + ")"

    @property
    def scaled(self) -> int:
        return self.n * 10


class C(B):
    def describe(self) -> str:
        return "C[" + super().describe() + "]"

    @property
    def scaled(self) -> int:
        return self.n * 100


# A subclass that overrides NEITHER inherits the nearest declaration.
class D(B):
    pass


sys.stdout.write(C(1).describe() + "\n")

xs: "list[A]" = [A(1), B(2), C(3), D(4)]
for x in xs:
    sys.stdout.write(x.describe() + " " + str(x.scaled) + "\n")


def through_a_parameter(a: A) -> str:
    return a.describe() + "/" + str(a.scaled)


sys.stdout.write(through_a_parameter(A(5)) + "\n")
sys.stdout.write(through_a_parameter(C(6)) + "\n")
sys.stdout.write(through_a_parameter(D(7)) + "\n")

total = 0
for x in xs:
    total += x.scaled
sys.stdout.write(str(total) + "\n")
