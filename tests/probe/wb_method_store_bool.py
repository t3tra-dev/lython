# probe: a method stores into self's bool field; the caller reads it back
# axes: acquire=self width=bool op=rebind flow=straight observe=writeback
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: True

class Box:
    def __init__(self, v: bool) -> None:
        self.f: bool = v

    def set(self) -> None:
        fresh: bool = True
        self.f = fresh


def mk() -> Box:
    v: bool = False
    return Box(v)


o = mk()
o.set()
print(o.f)
