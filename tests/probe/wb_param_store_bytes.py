# probe: callee stores into a borrowed receiver's bytes field; caller reads it back
# axes: acquire=param width=bytes op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: b'abcd'

class Box:
    def __init__(self, v: bytes) -> None:
        self.f: bytes = v


def mk() -> Box:
    v: bytes = b""
    return Box(v)


def rebind(b: Box) -> None:
    fresh: bytes = b"abcd"
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
