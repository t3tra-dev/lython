# WHAT: an instance's int and float fields hold the value itself where it has
# one, and every way of reading a field back -- straight after the store, in a
# later method, after a rebinding, through an Optional or a union field, from
# an object the constructor built in a loop -- reads the same number.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: whether a field holds an
# immediate or an object is decided at run time by the value, and the reads
# that follow a store used to be answered from the stored object, which the
# field no longer keeps. This passes on the build before the change too; it
# guards the readers.
class Point:
    def __init__(self, x: int, y: float) -> None:
        self.x = x
        self.y = y
        self.tag: int | None = None
        self.either: int | str = "none"

    def shift(self, dx: int) -> None:
        self.x = self.x + dx
        self.y = self.y * 2.0

    def describe(self) -> str:
        return f"{self.x} {self.y} {self.tag} {self.either}"


p = Point(2 ** 62 - 1, 0.5)
print(p.x, p.y, p.x + 1)
p.shift(1)
print(p.x, p.y)
p.tag = 7000
p.either = -12345
print(p.describe())
p.tag = None
p.either = "s"
print(p.describe())
pts = [Point(i * 1000, i / 4) for i in range(5)]
print([q.x for q in pts], [q.y for q in pts], sum(q.x for q in pts))
for q in pts:
    q.shift(-1)
print([(q.x, q.y) for q in pts])
big = Point(2 ** 70, 1e300)
print(big.x, big.y, big.x == 2 ** 70)
