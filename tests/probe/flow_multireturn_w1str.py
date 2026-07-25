# probe: field rebind reached through multireturn control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=multireturn
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 2 / 3 / 2

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


def run(k: int) -> int:
    o = mk()
    x: str = "ab"
    o.f = x
    if k == 0:
        return len(o.f)
    if k == 1:
        y: str = "cde"
        o.f = y
        return len(o.f)
    z: str = "ab"
    o.f = z
    return len(o.f)


print(run(0))
print(run(1))
print(run(2))
