# probe: field rebind -- receiver from an inline constructor in the same frame; field type list[int]
# axes: acquire=inline width=w3list op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


v: list[int] = []
o = Box(v)
fresh: list[int] = [1, 2]
o.f = fresh
print(len(o.f), o.f[0])
