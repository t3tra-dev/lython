# probe: callee stores into a borrowed receiver's int field; caller reads it back
# axes: acquire=param width=int op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 42

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    v: int = 0
    return Box(v)


def rebind(b: Box) -> None:
    fresh: int = 42
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
