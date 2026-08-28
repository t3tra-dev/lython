# WHAT: the two halves of ordering a class. A class written to be SORTED
# defines one method -- `__lt__` -- and Python answers all four spellings by
# REFLECTING it; and a NamedTuple orders like the tuple it is, field by field.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the reflected form has to
# swap its operands, and a swap that does not happen is not a diagnostic -- it
# is `b > a` answering what `a > b` asks. `sorted()` over either kind then
# prints a plausible order that is the reverse of CPython's.
import sys
from typing import NamedTuple


class Money:
    cents: int

    def __init__(self, cents: int) -> None:
        self.cents = cents

    def __lt__(self, other: "Money") -> bool:
        return self.cents < other.cents

    def __repr__(self) -> str:
        return "$" + str(self.cents)


class Span:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    # Only the non-strict direction, which reflects to the other non-strict
    # one and not to `<`.
    def __le__(self, other: "Span") -> bool:
        return self.n <= other.n


class Rec(NamedTuple):
    key: str
    n: int


a = Money(100)
b = Money(200)
# `a < b` is direct; `b > a` is the same method with its operands swapped.
sys.stdout.write(str(a < b) + " " + str(b > a) + " " + str(a > b) + " "
                 + str(b < a) + "\n")

s1 = Span(1)
s2 = Span(2)
sys.stdout.write(str(s1 <= s2) + " " + str(s2 >= s1) + " " + str(s2 <= s1)
                 + "\n")

# A class that DEFINES the direct method keeps it: reflecting past one would
# answer a different question than the program wrote.
class Both:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def __lt__(self, other: "Both") -> bool:
        return self.n < other.n

    def __gt__(self, other: "Both") -> bool:
        return self.n > other.n


sys.stdout.write(str(Both(1) > Both(2)) + " " + str(Both(2) > Both(1)) + "\n")

monies: "list[Money]" = [b, a, Money(150)]
sys.stdout.write(repr(sorted(monies)) + "\n")

# A NamedTuple compares field by field, in declaration order.
recs: "list[Rec]" = [Rec("b", 2), Rec("a", 9), Rec("a", 1)]
sys.stdout.write(repr(sorted(recs)) + "\n")
sys.stdout.write(str(Rec("a", 1) < Rec("a", 2)) + " "
                 + str(Rec("a", 1) < Rec("b", 0)) + " "
                 + str(Rec("b", 0) <= Rec("b", 0)) + " "
                 + str(Rec("b", 1) > Rec("b", 0)) + "\n")

# Ordering and equality agree, and the hash still keys a dict.
table: "dict[Rec, str]" = {}
table[Rec("a", 1)] = "first"
sys.stdout.write(table[Rec("a", 1)] + " " + str(Rec("a", 1) == Rec("a", 1))
                 + "\n")
