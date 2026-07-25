# probe: callee stores into and reads back a borrowed receiver's float field, then the caller reads it
# axes: acquire=param width=float op=rebind flow=straight observe=writeback-both
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: in callee: 1.5 / in caller: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    v: float = 0.0
    return Box(v)


def rebind(b: Box) -> None:
    fresh: float = 1.5
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
