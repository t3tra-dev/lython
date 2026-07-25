# probe: field rebind reached through earlyreturn control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=earlyreturn
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2 / 3

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


def run(flag: bool) -> int:
    o = mk()
    x: str = "ab"
    o.f = x
    if flag:
        return len(o.f)
    y: str = "cde"
    o.f = y
    return len(o.f)


print(run(True))
print(run(False))
