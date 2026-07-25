# probe: a method stores into self's tuple[int, int] field; the caller reads it back
# axes: acquire=self width=tuple op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: (1, 2)

class Box:
    def __init__(self, v: tuple[int, int]) -> None:
        self.f: tuple[int, int] = v

    def set(self) -> None:
        fresh: tuple[int, int] = (1, 2)
        self.f = fresh


def mk() -> Box:
    v: tuple[int, int] = (0, 0)
    return Box(v)


o = mk()
o.set()
print(o.f)
