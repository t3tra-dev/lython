# probe: a method stores into self's int field; the caller reads it back
# axes: acquire=self width=bigint op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   OverflowError: int too large to convert to a native 64-bit integer
# CPython 3.14 expects: 12345678901234567890

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v

    def set(self) -> None:
        fresh: int = 12345678901234567890
        self.f = fresh


def mk() -> Box:
    v: int = 0
    return Box(v)


o = mk()
o.set()
print(o.f)
