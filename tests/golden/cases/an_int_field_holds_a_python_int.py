# WHAT: a class field annotated `int` holding values a machine word cannot --
# through the constructor, through a later store, through arithmetic that grows
# past 64 bits mid-loop, and read back out.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is the VALUE, in
# full precision. A field that keeps only the low 64 bits compiles, runs, and
# prints a number -- a different one -- and the store that used to refuse this
# raised OverflowError from a line the author did not write.
#
# ⛔ `bool` is still a body word: its runtime shape is `i1`, so there is no
# address a box could hold. Nothing is lost -- a bool has two values.
class Counter:
    total: int
    step: int
    seen: bool

    def __init__(self, step: int) -> None:
        self.total = 0
        self.step = step
        self.seen = False

    def add(self, n: int) -> int:
        self.total = self.total + n
        self.seen = True
        return self.total


big = 2 ** 70
c = Counter(big)
print(c.step)
print(c.add(big))
print(c.add(1))
print(c.total, c.seen)

c.total = -(10 ** 30)
print(c.total)
print(c.total // 7)
print(c.total * c.total)

grow = Counter(1)
i = 0
while i < 4:
    grow.total = grow.total * (10 ** 12) + 3
    i += 1
print(grow.total)
print(grow.total > 2 ** 64, grow.total % 1000)


class Pair:
    a: int
    b: int

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def swap(self) -> None:
        t = self.a
        self.a = self.b
        self.b = t


p = Pair(3, 2 ** 100)
p.swap()
print(p.a, p.b)
p.swap()
print(p.a, p.b)
