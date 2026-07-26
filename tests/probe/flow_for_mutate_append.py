# probe: in-place field mutation inside a for loop (append)
# axes: acquire=call width=w3list/w1dict op=append flow=for
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
for k in range(3):
    o.f.append(k)
print(len(o.f))
