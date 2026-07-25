# probe: a method stores into self's bytes field; the caller reads it back
# axes: acquire=self width=bytes op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: b'abcd'

class Box:
    def __init__(self, v: bytes) -> None:
        self.f: bytes = v

    def set(self) -> None:
        fresh: bytes = b"abcd"
        self.f = fresh


def mk() -> Box:
    v: bytes = b""
    return Box(v)


o = mk()
o.set()
print(o.f)
