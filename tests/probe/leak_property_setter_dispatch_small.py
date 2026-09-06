# WHAT: a property WRITE through a base-typed receiver in a loop. The write goes
# through a synthesized dispatcher that takes the receiver and the value, so the
# receiver crosses a call boundary on every iteration; the stored string is
# sized past the probe floor.
class Cell:
    def __init__(self) -> None:
        self._v = ""

    @property
    def v(self) -> str:
        return self._v

    @v.setter
    def v(self, s: str) -> None:
        self._v = s


class Louder(Cell):
    @property
    def v(self) -> str:
        return self._v

    @v.setter
    def v(self, s: str) -> None:
        self._v = s + "!"


i = 0
while i < 300:
    c: Cell = Louder()
    c.v = "z" * 4096
    i += 1
print("done")
