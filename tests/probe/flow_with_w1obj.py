# probe: field rebind reached through with control flow; field type Other
# axes: acquire=call width=w1obj op=rebind flow=with
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   runtime manifest has no Ctx.__enter__ method
# CPython 3.14 expects: 7

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v


def mk() -> Box:
    v: Other = Other(0)
    return Box(v)


class Ctx:
    def __enter__(self) -> "Ctx":
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        return False


o = mk()
with Ctx():
    x: Other = Other(7)
    o.f = x
print(o.f.n)
