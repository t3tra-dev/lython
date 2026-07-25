# probe: callee stores into and reads back a borrowed receiver's set[int] field, then the caller reads it
# axes: acquire=param width=set op=rebind flow=straight observe=writeback-both
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: in callee: 2 / in caller: 2

class Box:
    def __init__(self, v: set[int]) -> None:
        self.f: set[int] = v


def mk() -> Box:
    v: set[int] = set()
    return Box(v)


def rebind(b: Box) -> None:
    fresh: set[int] = {1, 2}
    b.f = fresh
    o = b
    print("in callee:", len(o.f))


o = mk()
rebind(o)
print("in caller:", len(o.f))
