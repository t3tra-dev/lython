# HALF FIXED 2026-09-05. The CELL half is closed: `xs: list[int] = [1, 2]` at
# module scope, a class attribute's `P.v[0] = True`, a global written from
# inside a function, and `d["b"] = True` on a `dict[str, int]` are all refused
# at emit now, naming the cell and the two representations. They printed
# `[True, 2] 0` -- the container's repr decodes by TAG and was right, the
# subscript decoded by the DECLARED element type and was not.
#
# ⛔ A LOCAL is not a cell and deliberately keeps working: the emitter refines
# the name's element type at the store, so `xs: list[int] = [1, 2]; xs[0] =
# True; print(xs[0])` inside a function prints True and is correct.
#
# STILL OPEN, and the shape below is what it is: once the element type has been
# refined to bool, USING it picks the dispatch the declaration named --
#
#   "cannot adapt builtins.bool to runtime input 0 of builtins.int.__add__
#    [values: 'memref<3xi64>', expected 'memref<2xi64>']"
#
# -- loud, but from the LOWERING and about the compiler. A bool is 3 lanes and
# an int is 2, which is the whole reason the cell case could not be made to
# work either. What would close it is the union lane machinery: a `list[int]`
# that can hold a bool needs its reads to decode by tag, which is what the
# container's repr already does and what the subscript cannot.
#
# ⛔ The parameter boundary must NOT get the cell's rule: `def take(n: int)`
# called with `True` is correct and works (`take(True)` prints 2).


def used_in_a_function() -> int:
    xs: list[int] = [1, 2]
    xs[0] = True
    v = xs[0]
    return v + 1


def appended_in_a_function() -> int:
    ys: list[int] = []
    ys.append(True)
    v = ys[0]
    return v + 1


print(used_in_a_function(), appended_in_a_function())
