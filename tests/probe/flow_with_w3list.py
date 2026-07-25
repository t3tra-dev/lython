# probe: field rebind reached through with control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=with
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   runtime manifest has no Ctx.__enter__ method
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


class Ctx:
    def __enter__(self) -> "Ctx":
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        return False


o = mk()
with Ctx():
    x: list[int] = [1, 2]
    o.f = x
print(len(o.f))
