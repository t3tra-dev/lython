# probe: callee stores into a borrowed receiver's dict[str, int] field; caller reads it back
# axes: acquire=param width=dict op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


def rebind(b: Box) -> None:
    fresh: dict[str, int] = {"a": 1}
    b.f = fresh


o = mk()
rebind(o)
print(len(o.f))
