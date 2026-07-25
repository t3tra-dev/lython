# probe: field rebind reached through while control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=while
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


o = mk()
k = 0
while k < 2:
    x: str = "ab"
    o.f = x
    k = k + 1
print(len(o.f))
