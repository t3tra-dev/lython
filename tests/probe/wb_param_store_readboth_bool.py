# probe: callee stores into and reads back a borrowed receiver's bool field, then the caller reads it
# axes: acquire=param width=bool op=rebind flow=straight observe=writeback-both
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: in callee: True / in caller: True

class Box:
    def __init__(self, v: bool) -> None:
        self.f: bool = v


def mk() -> Box:
    v: bool = False
    return Box(v)


def rebind(b: Box) -> None:
    fresh: bool = True
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
