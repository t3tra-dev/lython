# probe: in-place field mutation inside a while loop (dsetitem)
# axes: acquire=call width=w3list/w1dict op=dsetitem flow=while
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


o = mk()
k = 0
while k < 3:
    o.f["k"] = k
    k = k + 1
print(len(o.f))
