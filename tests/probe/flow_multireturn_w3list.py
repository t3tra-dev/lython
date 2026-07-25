# probe: field rebind reached through multireturn control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=multireturn
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2 / 3 / 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


def run(k: int) -> int:
    o = mk()
    x: list[int] = [1, 2]
    o.f = x
    if k == 0:
        return len(o.f)
    if k == 1:
        y: list[int] = [3, 4, 5]
        o.f = y
        return len(o.f)
    z: list[int] = [1, 2]
    o.f = z
    return len(o.f)


print(run(0))
print(run(1))
print(run(2))
