# probe: field rebind -- receiver from a generator yield; field type int
# axes: acquire=gen width=w3int op=rebind flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   source generator next lowering currently supports int yields
# CPython 3.14 expects: 123456

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    v: int = 0
    return Box(v)


from typing import Iterator


def gen() -> Iterator[Box]:
    yield mk()


for o in gen():
    fresh: int = 123456
    o.f = fresh
    print(o.f)
