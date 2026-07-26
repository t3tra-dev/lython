# probe: a method stores into self's int | None field; the caller reads it back
# axes: acquire=self width=optional op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   runtime object header has invalid type 'i64'
# CPython 3.14 expects: 5

class Box:
    def __init__(self, v: int | None) -> None:
        self.f: int | None = v

    def set(self) -> None:
        fresh: int | None = 5
        self.f = fresh


def mk() -> Box:
    v: int | None = None
    return Box(v)


o = mk()
o.set()
print(o.f)
