# probe: field rebind -- receiver from a generator yield; field type dict[str, int]
# axes: acquire=gen width=w1dict op=rebind flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   source generator next lowering currently supports int yields
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


from typing import Iterator


def gen() -> Iterator[Box]:
    yield mk()


for o in gen():
    fresh: dict[str, int] = {"a": 1, "b": 2}
    o.f = fresh
    print(len(o.f), o.f["a"])
