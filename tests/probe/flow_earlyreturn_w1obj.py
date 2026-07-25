# probe: field rebind reached through earlyreturn control flow; field type Other
# axes: acquire=call width=w1obj op=rebind flow=earlyreturn
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 7 / 9

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v


def mk() -> Box:
    v: Other = Other(0)
    return Box(v)


def run(flag: bool) -> int:
    o = mk()
    x: Other = Other(7)
    o.f = x
    if flag:
        return o.f.n
    y: Other = Other(9)
    o.f = y
    return o.f.n


print(run(True))
print(run(False))
