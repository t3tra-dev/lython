# probe: field rebind -- receiver from an inline constructor in the same frame; field type str
# axes: acquire=inline width=w1str op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 4 abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


v: str = ""
o = Box(v)
fresh: str = "abcd"
o.f = fresh
print(len(o.f), o.f)
