# What this pins: a container of ints where a container of floats is declared.
#
#     def f() -> list[float]:
#         return [1]
#     print(sum(f()))        # printed 5e-324; CPython prints 1
#
# A container's ELEMENT type is its storage here, so retyping `list[int]` to
# `list[float]` leaves int boxes in float slots and the read decodes the int's
# words as a double. `coerceValue` already declines the SCALAR retyping for this
# exact reason ("int, float and bool share no representation"); the container case
# is the same lie one level in, and it was still being emitted as a
# `py.class.upcast`.
#
# Every shape that printed CPython's answer did so because nothing had decoded an
# element yet. `t: tuple[float, float] = (1, 2)` printed `(1, 2)` while
# `t[0] + 0.5` printed 0.5 in the same program, and `return [1]` printed `[1]`
# while `sum(...)` of it printed 5e-324. So declining the retyping gives back no
# working ground: it turns silent wrong answers into a mismatch the store, the
# return or the call reports.
#
# Why an errors case rather than a case: the answer is a refusal. What the program
# should print is `1`, and reaching that needs the element to STAY an int in a
# float-declared container -- which is the representation question recorded in
# tests/probe/wb_grid_leftovers_2026_08_16.py, not something this refusal decides.
#
# ⛔ Two channels still mis-execute and neither is this one: a module-global
# container (`xs: list[float] = [1]` at module scope) goes through the static
# attribute initializer, and a tuple literal builds `tuple[float, float]` directly
# from the positional expectations. Both still print 5e-324, and both are recorded
# with their measurements.


def f() -> list[float]:
    return [1]


print(sum(f()))
