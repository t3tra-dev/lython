# probe: a method stores into self's str field; the caller reads it back
# axes: acquire=self width=str op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v

    def set(self) -> None:
        fresh: str = "abcd"
        self.f = fresh


def mk() -> Box:
    v: str = ""
    return Box(v)


o = mk()
o.set()
print(o.f)
