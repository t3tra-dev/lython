# probe: a method stores into self's Other field; the caller reads it back
# axes: acquire=self width=obj op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 7

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v

    def set(self) -> None:
        fresh: Other = Other(7)
        self.f = fresh


def mk() -> Box:
    v: Other = Other(0)
    return Box(v)


o = mk()
o.set()
print(o.f.n)
