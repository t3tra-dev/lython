# probe: a method stores into self's float field; the caller reads it back
# axes: acquire=self width=float op=rebind flow=straight observe=writeback
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v

    def set(self) -> None:
        fresh: float = 1.5
        self.f = fresh


def mk() -> Box:
    v: float = 0.0
    return Box(v)


o = mk()
o.set()
print(o.f)
