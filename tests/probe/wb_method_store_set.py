# probe: a method stores into self's set[int] field; the caller reads it back
# axes: acquire=self width=set op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: set[int]) -> None:
        self.f: set[int] = v

    def set(self) -> None:
        fresh: set[int] = {1, 2}
        self.f = fresh


def mk() -> Box:
    v: set[int] = set()
    return Box(v)


o = mk()
o.set()
print(len(o.f))
