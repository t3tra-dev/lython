# What: `x.v = n` on a base-typed receiver runs the setter the RUNTIME class
# declares, the way `x.v` one line over already ran its getter. The write used
# to inline the base's setter, so the value stored and the value read back
# disagreed about which class the object is -- and nothing said so.
#
# Only running it shows which setter ran: each one stores a different function
# of `n`, and the read is what reports it. The third class overrides nothing,
# which pins that an inheriting subclass lands in its parent's arm; the
# augmented form is here because `x.v += 1` is a read AND a write, and the two
# halves used to resolve to different classes.
class Cell:
    def __init__(self) -> None:
        self._v = 0

    @property
    def v(self) -> int:
        return self._v

    @v.setter
    def v(self, n: int) -> None:
        self._v = n


class Doubling(Cell):
    @property
    def v(self) -> int:
        return self._v

    @v.setter
    def v(self, n: int) -> None:
        self._v = n * 2


class Inheriting(Doubling):
    pass


cells: list[Cell] = [Cell(), Doubling(), Inheriting()]
for cell in cells:
    cell.v = 5
print("stored", [c.v for c in cells])

for cell in cells:
    cell.v += 1
print("augmented", [c.v for c in cells])


def store(c: Cell, n: int) -> int:
    c.v = n
    return c.v


print("through a parameter", [store(c, 3) for c in cells])
