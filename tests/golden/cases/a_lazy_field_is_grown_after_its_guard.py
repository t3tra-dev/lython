# WHAT: an Optional container field created on first use -- `if self.xs is
# None: self.xs = []` then `self.xs.append(s)`, and the same for a dict and a
# set, and a class attribute through the class -- grows on every call, the
# calls that skip the creation included, and `+=` on a proved int field adds.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the append was refused,
# and the obvious repair compiled into a wrong answer -- the SECOND call
# printed `['a', None]`, because the empty literal's evidence reached the
# path that had not built it. Only calling twice and printing the list shows
# which one this is.

class C:
    def __init__(self) -> None:
        self.v: int | None = None
        self.xs: list[str] | None = None
    def bump(self) -> None:
        if self.v is None:
            self.v = 0
        self.v += 1
        if self.xs is None:
            self.xs = []
        self.xs.append("a")
c = C()
c.bump()
c.bump()
print(c.v, c.xs)
class C2:
    def __init__(self) -> None:
        self.xs: list[str] | None = None
    def bump(self) -> None:
        if self.xs is None:
            self.xs = []
        self.xs.append("a")
c2 = C2()
c2.bump()
c2.bump()
print(c2.xs)

class Cache:
    def __init__(self) -> None:
        self.items: list[str] | None = None
        self.index: dict[str, int] | None = None
        self.seen: set[int] | None = None
    def add(self, s: str) -> None:
        if self.items is None:
            self.items = []
        self.items.append(s)
        if self.index is None:
            self.index = {}
        self.index[s] = len(self.items)
        if self.seen is None:
            self.seen = set()
        self.seen.add(len(s))
    def reset(self, keep: bool) -> None:
        if not keep:
            self.items = None
            self.index = None
c = Cache()
for w in ["a", "bb", "a", "ccc"]:
    c.add(w)
print(c.items, c.index)
seen = c.seen
if seen is not None:
    print(sorted(seen))
c.reset(False)
c.add("z")
c.reset(True)
c.add("y")
print(c.items, c.index)
for i in range(200):
    c.add(str(i))
if c.items is not None:
    print(len(c.items))
class Reg3:
    items: list[str] | None = None
    last: "Reg3 | None" = None
    count: int | None = None
    def __init__(self, n: str) -> None:
        self.n = n
def register3(s: str) -> None:
    if Reg3.items is None:
        Reg3.items = []
    Reg3.items.append(s)
    Reg3.last = Reg3(s)
    if Reg3.count is None:
        Reg3.count = 0
    Reg3.count += 1
for i in range(50):
    register3("x" + str(i))
print(len(Reg3.items) if Reg3.items is not None else 0, Reg3.count)
if Reg3.last is not None:
    print(Reg3.last.n)
Reg3.items = None
Reg3.last = None
print(Reg3.items, Reg3.last)

