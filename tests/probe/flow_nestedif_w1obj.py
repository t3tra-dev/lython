# probe: field rebind reached through nestedif control flow; field type Other
# axes: acquire=call width=w1obj op=rebind flow=nestedif
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
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
n = len("abc")
if n > 1:
    if n == 3:
        x: Other = Other(7)
        o.f = x
    else:
        y: Other = Other(9)
        o.f = y
print(o.f.n)
