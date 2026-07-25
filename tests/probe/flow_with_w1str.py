# probe: field rebind reached through with control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=with
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   runtime manifest has no Ctx.__enter__ method
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


class Ctx:
    def __enter__(self) -> "Ctx":
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        return False


o = mk()
with Ctx():
    x: str = "ab"
    o.f = x
print(len(o.f))
