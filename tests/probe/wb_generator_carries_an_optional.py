# A generator with an `X | None` local LIVE across a yield is refused; the same
# local assigned after the yield is fine, and the same loop in a plain function
# is fine. The idiom is the pairwise walk:
#
#     prev: "int | None" = None
#     for x in xs:
#         if prev is not None:
#             yield prev + x
#         prev = x
#
# The decline reason names it exactly: "a value of type !py.union<int, None> is
# live across a yield and has no generator frame lane (a frame lane is keyed on
# a runtime contract)". A union has no contract name, so `laneEligibleContract`
# answers empty and the whole generator drops to the tier below, which refuses
# it for its own single-lane limit.
#
# Measured neighbours: a plain `int` accumulator across a yield is fine (b3), a
# union local written only AFTER the yield is fine (c3), and the identical loop
# returning a list instead of yielding is fine (c1). What the yield changes is
# only that the value has to survive a suspension.
#
# ⭐ THE SHAPE OF THE REPAIR IS THE BOX, and it exists. `X | None` whose payload
# is one runtime lane is already stored BOXED in a class field
# (`classFieldStoredBoxed`), the box's entity word being zero IS the tag, and
# `unionValuesFromBoxWords` rebuilds a union from box words under a tag guard.
# So the frame lane can be keyed on the box rather than on the union: box at
# the suspend, rebuild at the resume. A WIDER union needs a real tag word and
# is the union-lane mechanism proper, not this.
#
# ⛔ NOT recorded as ready: this is one half of the union-lane mechanism, which
# is budgeted on its own rather than picked up mid-round.
def pairwise(xs: "list[int]"):
    prev: "int | None" = None
    for x in xs:
        if prev is not None:
            yield prev + x
        prev = x


print(list(pairwise([1, 2, 3])))
