# probe: callee stores into and reads back a borrowed receiver's dict[str, int] field, then the caller reads it
# axes: acquire=param width=dict op=rebind flow=straight observe=writeback-both
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: in callee: 1 / in caller: 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


def rebind(b: Box) -> None:
    fresh: dict[str, int] = {"a": 1}
    b.f = fresh
    o = b
    print("in callee:", len(o.f))


o = mk()
rebind(o)
print("in caller:", len(o.f))
