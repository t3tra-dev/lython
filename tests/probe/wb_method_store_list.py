# probe: a method stores into self's list[int] field; the caller reads it back
# axes: acquire=self width=list op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v

    def set(self) -> None:
        fresh: list[int] = [1, 2]
        self.f = fresh


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
o.set()
print(len(o.f))
