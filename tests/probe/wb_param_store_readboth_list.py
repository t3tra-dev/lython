# probe: callee stores into and reads back a borrowed receiver's list[int] field, then the caller reads it
# axes: acquire=param width=list op=rebind flow=straight observe=writeback-both
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: in callee: 2 / in caller: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


def rebind(b: Box) -> None:
    fresh: list[int] = [1, 2]
    b.f = fresh
    o = b
    print("in callee:", len(o.f))


o = mk()
rebind(o)
print("in caller:", len(o.f))
