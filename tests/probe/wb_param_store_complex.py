# probe: callee stores into a borrowed receiver's complex field; caller reads it back
# axes: acquire=param width=complex op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: (1+2j)

class Box:
    def __init__(self, v: complex) -> None:
        self.f: complex = v


def mk() -> Box:
    v: complex = 0j
    return Box(v)


def rebind(b: Box) -> None:
    fresh: complex = 1 + 2j
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
