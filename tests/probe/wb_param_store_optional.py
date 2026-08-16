# probe: callee stores into a borrowed receiver's int | None field; caller reads it back
# axes: acquire=param width=optional op=rebind flow=straight observe=writeback
# CLASSIFICATION @ 2026-08-17: 3 loud 拒否 (診断)
#   storing into field 'f' of a receiver that arrived as a parameter is not
#   supported for this field's type: the store writes the receiver's own value
#   lanes, so the caller would not see it
#
# ⛔ RECLASSIFIED. It used to be "runtime object header has invalid type
# 'i64'", which was the shape the store produced before the refusal was moved
# to a static boundary. The message now names the mechanism -- a union field is
# stored INLINE, so a store through a parameter writes the callee's copy of the
# lanes -- which is the same inline-union-field item the five-symptom cluster
# is about.
# CPython 3.14 expects: 5

class Box:
    def __init__(self, v: int | None) -> None:
        self.f: int | None = v


def mk() -> Box:
    v: int | None = None
    return Box(v)


def rebind(b: Box) -> None:
    fresh: int | None = 5
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
