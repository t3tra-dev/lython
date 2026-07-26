# probe: field rebind reached through while control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=while
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
k = 0
while k < 2:
    x: list[int] = [1, 2]
    o.f = x
    k = k + 1
print(len(o.f))
