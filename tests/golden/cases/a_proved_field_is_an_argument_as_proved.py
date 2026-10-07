# WHAT: a field or a module global a guard has proved present is handed to
# sorted, list, tuple, max, min, a comprehension and the rest as the member
# the guard proved, as CPython hands over the object: `if self.s is not None:
# print(sorted(self.s))` sorts the set, and `max(_seen)` after the same guard
# on a global finds the largest.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: these were refused at
# emit, and the claim is that each builtin computes the value CPython does
# from the proved member -- the sorted order, the largest, the doubled list --
# which only running shows.

class C:
    def __init__(self) -> None:
        self.s: set[int] | None = None

    def show(self) -> None:
        if self.s is not None:
            print(sorted(self.s), len(self.s), list(self.s), max(self.s))
            print(sum(self.s), self.s | {9}, [x * 2 for x in self.s])
            print(tuple(self.s), min(self.s) + 1, any(x > 1 for x in self.s))


c = C()
c.show()
c.s = {3, 1, 2}
c.show()

_seen: set[int] | None = None
def note(x: int) -> None:
    global _seen
    if _seen is None:
        _seen = set()
    _seen.add(x)
def report() -> list[int]:
    if _seen is None:
        return []
    return sorted(_seen)
for v in [5, 1, 3, 1]:
    note(v)
print(report(), max(_seen) if _seen is not None else -1)
class Grid:
    def __init__(self) -> None:
        self.rows: list[list[int]] | None = None
    def build(self, n: int) -> None:
        self.rows = [[i * j for j in range(n)] for i in range(n)]
    def total(self) -> int:
        if self.rows is None:
            return 0
        return sum(sum(r) for r in self.rows) + len([r for r in self.rows if len(r) > 0])
    def widths(self) -> list[int]:
        if self.rows is not None:
            return [len(r) for r in self.rows]
        return []
g = Grid()
print(g.total(), g.widths())
g.build(3)
print(g.total(), g.widths())
