# probe: a method stores into self's dict[str, int] field; the caller reads it back
# axes: acquire=self width=dict op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v

    def set(self) -> None:
        fresh: dict[str, int] = {"a": 1}
        self.f = fresh


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


o = mk()
o.set()
print(len(o.f))
