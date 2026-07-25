# probe: callee stores into a borrowed receiver's set[int] field; caller reads it back
# axes: acquire=param width=set op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: set[int]) -> None:
        self.f: set[int] = v


def mk() -> Box:
    v: set[int] = set()
    return Box(v)


def rebind(b: Box) -> None:
    fresh: set[int] = {1, 2}
    b.f = fresh


o = mk()
rebind(o)
print(len(o.f))
