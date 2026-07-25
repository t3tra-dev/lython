# probe: field rebind reached through excepthandler control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=excepthandler
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


o = mk()
try:
    raise ValueError("x")
except ValueError:
    x: str = "ab"
    o.f = x
print(len(o.f))
