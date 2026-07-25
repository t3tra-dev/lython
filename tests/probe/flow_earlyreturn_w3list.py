# probe: field rebind reached through earlyreturn control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=earlyreturn
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 2 / 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


def run(flag: bool) -> int:
    o = mk()
    x: list[int] = [1, 2]
    o.f = x
    if flag:
        return len(o.f)
    y: list[int] = [3, 4, 5]
    o.f = y
    return len(o.f)


print(run(True))
print(run(False))
