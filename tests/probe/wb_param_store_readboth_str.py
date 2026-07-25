# probe: callee stores into and reads back a borrowed receiver's str field, then the caller reads it
# axes: acquire=param width=str op=rebind flow=straight observe=writeback-both
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: in callee: abcd / in caller: abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


def rebind(b: Box) -> None:
    fresh: str = "abcd"
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
