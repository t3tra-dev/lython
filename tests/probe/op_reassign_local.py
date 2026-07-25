# probe: the local naming the object is reassigned to a second object after a rebind
# axes: acquire=call width=w3list op=reassign-local flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2 / 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
a: list[int] = [1, 2]
o.f = a
print(len(o.f))
o = mk()
b: list[int] = [1, 2, 3]
o.f = b
print(len(o.f))
