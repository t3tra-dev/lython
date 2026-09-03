# OPEN, the RESIDUE of a repair. A generator's yields are typed by a walk with
# no flow facts, and the annotation is now taken whenever every inferred yield
# is a union CONTAINING it (cases/a_generator_that_yields_inside_a_guard). Two
# shapes are left, both because the walk cannot even TYPE the yield expression:
#
#     def gen(xs: list[str | None]) -> Iterator[str]:
#         for v in xs:
#             if v is not None:
#                 yield v.upper()      # infers None: 'upper' is not on the union
#
#     def gen(xs: list[A]) -> Iterator[str]:
#         for x in xs:
#             if isinstance(x, B):
#                 yield x.tag()        # same, through a class narrowing
#
# `yield v * 2` is the shape that works, and only because `__mul__` on the
# union still infers SOMETHING (`int | None`) for the walk to recognise. A
# method or attribute lookup fails outright and the walk records None.
#
# ⛔ MEASURED AND DROPPED: taking the annotation unconditionally. It accepts
# all of these, and it moves a genuinely wrong generator -- `-> Iterator[str]`
# with `yield v` for an int `v` -- from a clean emit diagnostic to "Failed to
# run lowering pipeline", because the per-yield coercion does not refuse int
# where str is declared. A worse diagnostic for a wrong program is not the
# trade; what these need is a narrowing-aware yield walk, which is the same
# mechanism [[lython-where-a-proof-is-spent]] wants for a FIELD.
#
# ⛔ And one shape moved rather than closed: `def gen(v: int | None) ->
# Iterator[int]` with `if v is not None: yield v` now reaches the generator
# lowering and is refused there ("supports only straight-line pure int yield
# bodies") -- the union PARAMETER cannot ride the frame. Loud either way; the
# annotation mismatch was masking it.
from typing import Iterator


class A:
    pass


class B(A):
    def tag(self) -> str:
        return "b"


def tags(xs: list[A]) -> Iterator[str]:
    for x in xs:
        if isinstance(x, B):
            yield x.tag()


print(list(tags([A(), B()])))
