# probe: callee stores into and reads back a borrowed receiver's int field, then the caller reads it
# axes: acquire=param width=bigint op=rebind flow=straight observe=writeback-both
# CLASSIFICATION: 3 loud 拒否 (診断)
#   OverflowError: int too large to convert to a native 64-bit integer
# CPython 3.14 expects: in callee: 12345678901234567890 / in caller: 12345678901234567890

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    v: int = 0
    return Box(v)


def rebind(b: Box) -> None:
    fresh: int = 12345678901234567890
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
