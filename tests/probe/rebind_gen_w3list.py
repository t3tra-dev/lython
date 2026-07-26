# probe: field rebind -- receiver from a generator yield; field type list[int]
# axes: acquire=gen width=w3list op=rebind flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   source generator next lowering currently supports int yields
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


from typing import Iterator


def gen() -> Iterator[Box]:
    yield mk()


for o in gen():
    fresh: list[int] = [1, 2]
    o.f = fresh
    print(len(o.f), o.f[0])
