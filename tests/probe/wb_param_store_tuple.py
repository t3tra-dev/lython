# probe: callee stores into a borrowed receiver's tuple[int, int] field; caller reads it back
# axes: acquire=param width=tuple op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: (1, 2)

class Box:
    def __init__(self, v: tuple[int, int]) -> None:
        self.f: tuple[int, int] = v


def mk() -> Box:
    v: tuple[int, int] = (0, 0)
    return Box(v)


def rebind(b: Box) -> None:
    fresh: tuple[int, int] = (1, 2)
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
