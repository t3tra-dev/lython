# probe: callee stores into and reads back a borrowed receiver's Other field, then the caller reads it
# axes: acquire=param width=obj op=rebind flow=straight observe=writeback-both
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: in callee: 7 / in caller: 7

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
    o = b
    print("in callee:", o.f.n)


o = mk()
rebind(o)
print("in caller:", o.f.n)
