# probe: a method stores into self's int field; the caller reads it back
# axes: acquire=self width=int op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 42

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v

    def set(self) -> None:
        fresh: int = 42
        self.f = fresh


def mk() -> Box:
    v: int = 0
    return Box(v)


o = mk()
o.set()
print(o.f)
