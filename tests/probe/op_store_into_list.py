# probe: an object obtained from a call is stored into a list, then mutated through the list
# axes: acquire=call width=w3list op=store-into-container flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
xs: list[Box] = []
xs.append(o)
fresh: list[int] = [7, 8]
xs[0].f = fresh
print(len(xs[0].f), len(o.f))
