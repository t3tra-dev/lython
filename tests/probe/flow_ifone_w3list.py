# probe: field rebind reached through ifone control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=ifone
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
n = len("ab")
if n == 2:
    x: list[int] = [1, 2]
    o.f = x
print(len(o.f))
