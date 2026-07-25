# probe: callee stores into a borrowed receiver's int | None field; caller reads it back
# axes: acquire=param width=optional op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 3 loud 拒否 (診断)
#   runtime object header has invalid type 'i64'
# CPython 3.14 expects: 5

class Box:
    def __init__(self, v: int | None) -> None:
        self.f: int | None = v


def mk() -> Box:
    v: int | None = None
    return Box(v)


def rebind(b: Box) -> None:
    fresh: int | None = 5
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
