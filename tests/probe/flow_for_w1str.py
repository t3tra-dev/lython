# probe: field rebind reached through for control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=for
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


o = mk()
for _ in range(2):
    x: str = "ab"
    o.f = x
print(len(o.f))
