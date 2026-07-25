# probe: field rebind -- receiver from an inline constructor in the same frame; field type int
# axes: acquire=inline width=w3int op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 123456

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


v: int = 0
o = Box(v)
fresh: int = 123456
o.f = fresh
print(o.f)
