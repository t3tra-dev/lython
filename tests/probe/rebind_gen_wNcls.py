# probe: field rebind -- receiver from a generator yield; field type Wide
# axes: acquire=gen width=wNcls op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   source generator next lowering currently supports int yields
# CPython 3.14 expects: 1 2.5 z

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Wide:
    def __init__(self, a: int, b: float, c: str) -> None:
        self.a: int = a
        self.b: float = b
        self.c: str = c


class Box:
    def __init__(self, v: Wide) -> None:
        self.f: Wide = v


def mk() -> Box:
    v: Wide = Wide(0, 0.0, "")
    return Box(v)


from typing import Iterator


def gen() -> Iterator[Box]:
    yield mk()


for o in gen():
    fresh: Wide = Wide(1, 2.5, "z")
    o.f = fresh
    print(o.f.a, o.f.b, o.f.c)
