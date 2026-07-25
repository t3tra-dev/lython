# probe: callee stores into a borrowed receiver's Other field; caller reads it back
# axes: acquire=param width=obj op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 7

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v


def mk() -> Box:
    v: Other = Other(0)
    return Box(v)


def rebind(b: Box) -> None:
    fresh: Other = Other(7)
    b.f = fresh


o = mk()
rebind(o)
print(o.f.n)
