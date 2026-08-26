# probe: callee stores into a borrowed receiver's `list[int] | None` field; caller reads it back
# axes: acquire=param width=optional-boxed op=rebind flow=straight observe=writeback
# CLASSIFICATION @ 2026-08-27: 1 正しい
#
# The same shape as wb_param_store_optional, one annotation apart: its field is
# `int | None`, which stays INLINE because an int has no entity to put in a box,
# and it is still refused for the reason recorded there -- the store writes the
# callee's copy of the lanes. This one's member IS an entity, so the field is a
# BOX, the store lands in the instance, and the caller sees it.
#
# CPython 3.14 expects: [5]


class Box:
    def __init__(self, v: list[int] | None) -> None:
        self.f: list[int] | None = v


def mk() -> Box:
    v: list[int] | None = None
    return Box(v)


def rebind(b: Box) -> None:
    fresh: list[int] | None = [5]
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
