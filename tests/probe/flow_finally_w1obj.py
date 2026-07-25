# probe: field rebind reached through finally control flow; field type Other
# axes: acquire=call width=w1obj op=rebind flow=finally
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
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


o = mk()
try:
    pass
finally:
    x: Other = Other(7)
    o.f = x
print(o.f.n)
