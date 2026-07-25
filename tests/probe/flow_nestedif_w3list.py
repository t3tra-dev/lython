# probe: field rebind reached through nestedif control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=nestedif
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
n = len("abc")
if n > 1:
    if n == 3:
        x: list[int] = [1, 2]
        o.f = x
    else:
        y: list[int] = [3, 4, 5]
        o.f = y
print(len(o.f))
