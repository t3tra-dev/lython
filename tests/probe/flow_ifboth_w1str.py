# probe: field rebind reached through ifboth control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=ifboth
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


o = mk()
n = len("ab")
if n == 2:
    x: str = "ab"
    o.f = x
else:
    y: str = "cde"
    o.f = y
print(len(o.f))
