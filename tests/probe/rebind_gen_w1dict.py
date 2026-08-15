# probe: field rebind -- receiver from a generator yield; field type dict[str, int]
# axes: acquire=gen width=w1dict op=rebind flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   source generator next lowering currently supports int yields
# CPython 3.14 expects: 2 1
# FIXED 2026-08-15: the yield lane, not the payload type -- see
# tests/probe/wb_source_generator_non_int_yield.py for the whole cluster.

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
