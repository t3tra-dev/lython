# probe: a method stores into self's complex field; the caller reads it back
# axes: acquire=self width=complex op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: (1+2j)

class Box:
    def __init__(self, v: complex) -> None:
        self.f: complex = v

    def set(self) -> None:
        fresh: complex = 1 + 2j
        self.f = fresh


def mk() -> Box:
    v: complex = 0j
    return Box(v)


o = mk()
o.set()
print(o.f)
