# probe: callee stores into and reads back a borrowed receiver's int field, then the caller reads it
# axes: acquire=param width=int op=rebind flow=straight observe=writeback-both
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: in callee: 42 / in caller: 42

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    v: int = 0
    return Box(v)


def rebind(b: Box) -> None:
    fresh: int = 42
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
