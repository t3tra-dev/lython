# probe: callee stores into and reads back a borrowed receiver's tuple[int, int] field, then the caller reads it
# axes: acquire=param width=tuple op=rebind flow=straight observe=writeback-both
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: in callee: (1, 2) / in caller: (1, 2)

class Box:
    def __init__(self, v: tuple[int, int]) -> None:
        self.f: tuple[int, int] = v


def mk() -> Box:
    v: tuple[int, int] = (0, 0)
    return Box(v)


def rebind(b: Box) -> None:
    fresh: tuple[int, int] = (1, 2)
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
