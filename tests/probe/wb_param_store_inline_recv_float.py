# probe: same store-through-parameter, but the caller built the receiver inline
# axes: acquire=param(inline recv) width=float op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def rebind(b: Box) -> None:
    fresh: float = 1.5
    b.f = fresh


v: float = 0.0
o = Box(v)
rebind(o)
print(o.f)
