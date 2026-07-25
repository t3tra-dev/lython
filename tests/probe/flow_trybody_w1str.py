# probe: field rebind reached through trybody control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=trybody
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


o = mk()
try:
    x: str = "ab"
    o.f = x
except ValueError:
    print("unreachable")
print(len(o.f))
