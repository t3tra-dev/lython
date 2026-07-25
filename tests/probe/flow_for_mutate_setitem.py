# probe: in-place field mutation inside a for loop (setitem)
# axes: acquire=call width=w3list/w1dict op=setitem flow=for
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [0]
    return Box(v)


o = mk()
for k in range(3):
    o.f[0] = k
print(o.f[0])
