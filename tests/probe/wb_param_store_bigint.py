# probe: callee stores into a borrowed receiver's int field; caller reads it back
# axes: acquire=param width=bigint op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   OverflowError: int too large to convert to a native 64-bit integer
# CPython 3.14 expects: 12345678901234567890

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    v: int = 0
    return Box(v)


def rebind(b: Box) -> None:
    fresh: int = 12345678901234567890
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
