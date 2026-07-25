# probe: field rebind reached through ifone control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=ifone
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
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
print(len(o.f))
