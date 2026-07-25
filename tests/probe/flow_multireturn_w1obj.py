# probe: field rebind reached through multireturn control flow; field type Other
# axes: acquire=call width=w1obj op=rebind flow=multireturn
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 7 / 9 / 7

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v


def mk() -> Box:
    v: Other = Other(0)
    return Box(v)


def run(k: int) -> int:
    o = mk()
    x: Other = Other(7)
    o.f = x
    if k == 0:
        return o.f.n
    if k == 1:
        y: Other = Other(9)
        o.f = y
        return o.f.n
    z: Other = Other(7)
    o.f = z
    return o.f.n


print(run(0))
print(run(1))
print(run(2))
