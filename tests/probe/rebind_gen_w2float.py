# probe: field rebind -- receiver from a generator yield; field type float
# axes: acquire=gen width=w2float op=rebind flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   source generator next lowering currently supports int yields
# CPython 3.14 expects: 1.5
# FIXED 2026-08-15: the yield lane, not the payload type -- see
# tests/probe/wb_source_generator_non_int_yield.py for the whole cluster.

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    v: float = 0.0
    return Box(v)


from typing import Iterator


def gen() -> Iterator[Box]:
    yield mk()


for o in gen():
    fresh: float = 1.5
    o.f = fresh
    print(o.f)
