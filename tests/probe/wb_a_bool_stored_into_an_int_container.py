# probe: a bool written into a container whose ELEMENT type is int
# CLASSIFICATION @ 82993ec1: 2 silent, 2 loud-but-internal
#   xs: list[int] = [1, 2]; xs[0] = True   -> printed 0        (module scope)
#   ys: list[int] = []; ys.append(True)    -> printed 0        (module scope)
#   the same two in a function             -> "cannot adapt builtins.bool to
#                                             runtime input 0 of
#                                             builtins.int.__add__" as soon as
#                                             the element is USED
#   xs = [1, 2]; xs[0] = True (no annotation) -> print(xs) gives [True, 2], but
#                                             xs[0] + 1 fails the same way
# CPython 3.14 expects: True / 2 / True / bool
#
# The DECLARATION channels are closed (82993ec1+): `xs: list[int] = [True]` and
# `class P: v: list[int] = [True]` are refused at emit, naming the container's
# element representation. This is the MUTATION channel of the same fact, and it
# is still open: nothing adapts or reports the element written by
# `__setitem__`, `append`, `insert` or `extend`.
#
# ⛔ Why not the same refusal, mechanically: the declaration channels each have
# ONE cell write to test, and the mutation channel is every manifest method
# that takes an element -- the check belongs where an argument is adapted to a
# receiver's own type argument, and there is no such single place today. The
# parameter boundary must NOT get the same rule: `def take(n: int)` called with
# `True` is correct and works (`take(True)` prints 2).


def used_in_a_function() -> int:
    xs: list[int] = [1, 2]
    xs[0] = True
    v = xs[0]
    return v + 1


rows: list[int] = [1, 2]
rows[0] = True
print(rows[0])
print(used_in_a_function())
