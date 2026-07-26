# probe: a class with two float fields; the callee stores only into the second
# axes: acquire=param width=float x2 op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 1.0 9.5

class Box:
    def __init__(self, a: float, b: float) -> None:
        self.a: float = a
        self.b: float = b


def mk() -> Box:
    return Box(1.0, 2.0)


def rebind(x: Box) -> None:
    fresh: float = 9.5
    x.b = fresh


o = mk()
rebind(o)
print(o.a, o.b)
