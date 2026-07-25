# probe: callee stores into a borrowed receiver's str field; caller reads it back
# axes: acquire=param width=str op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


def rebind(b: Box) -> None:
    fresh: str = "abcd"
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
