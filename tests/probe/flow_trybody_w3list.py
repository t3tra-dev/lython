# probe: field rebind reached through trybody control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=trybody
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
try:
    x: list[int] = [1, 2]
    o.f = x
except ValueError:
    print("unreachable")
print(len(o.f))
