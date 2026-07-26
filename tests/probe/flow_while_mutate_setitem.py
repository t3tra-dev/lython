# probe: in-place field mutation inside a while loop (setitem)
# axes: acquire=call width=w3list/w1dict op=setitem flow=while
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [0]
    return Box(v)


o = mk()
k = 0
while k < 3:
    o.f[0] = k
    k = k + 1
print(o.f[0])
