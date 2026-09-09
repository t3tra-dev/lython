# A field whose only assignment is an EMPTY container literal has no element
# type of its own, and the operations that give it one are usually in another
# method -- which nothing looked at:
#
#     class Bag:
#         def __init__(self) -> None:
#             self.xs = []
#         def put(self, n: int) -> None:
#             self.xs.append(n)
#     # operand type 'builtins.object' does not match selected evidence
#
# That is how a container held by a class is written when nobody annotates it,
# and even `self.xs = []` followed by `self.xs.append(0)` in the SAME
# constructor was unseeded: the scan that answers this for a local was keyed on
# a bare NAME, so a field was never asked about at all.
#
# Why execution: the element type decides what may be read back out, so the
# program has to DECODE what it stored -- the arithmetic and the concatenations
# below are the assertions, not the fact that it compiles.
#
# ⭐ ONE SCAN AGAIN. It takes the container as an ATTRIBUTE of a receiver now,
# and is asked once per method with that method's parameters in scope.
#
# ⛔ Two methods that disagree leave the field erased, where it started -- the
# same rule the scan already applies to two seeds in one suite.
#
# ⛔ Still open, and for the same reason this one was: a module GLOBAL filled
# inside a function, and an outer local filled inside a nested def. Both are on
# the other side of a callable boundary that no walk crosses --
# tests/probe/wb_empty_container_seeded_from_another_scope.py.


class Bag:
    def __init__(self) -> None:
        self.xs = []
        self.seen = set()
        self.index = {}

    def put(self, n: int) -> None:
        self.xs.append(n)
        self.seen.add(n)

    def label(self, key: str, n: int) -> None:
        self.index[key] = n

    def total(self) -> int:
        t = 0
        for v in self.xs:
            t += v
        return t

    def labelled(self, key: str) -> int:
        return self.index[key] + 1


class Log:
    def __init__(self) -> None:
        self.lines = []
        self.lines.append("start")

    def add(self, text: str) -> None:
        self.lines.append(text)

    def joined(self) -> str:
        return "|".join(self.lines)


b = Bag()
b.put(3)
b.put(4)
b.label("a", 10)
print(b.total(), b.xs[0] + 1)
print(sorted(b.seen))
print(b.labelled("a"))

g = Log()
g.add("middle")
g.add("end")
print(g.joined())
print(g.lines[1] + "!")
