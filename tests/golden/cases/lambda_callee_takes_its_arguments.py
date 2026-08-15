# What this pins: an unannotated lambda in CALLEE position gets its parameter
# types from the arguments it is applied to. `(lambda v: v * 2)(5)` and
# `max(xs, key=lambda p: p[1])` were both "lambda requires a Callable
# annotation because its type contains unresolved Unknown" -- the callee is
# emitted before the operands, so nothing had told the lambda what it takes.
#
# Why this needs to run rather than assert on a diagnostic: the parameter type
# is what the BODY is compiled against, so getting it wrong is not a refusal,
# it is a different program. `key=lambda p: p[1]` over `list[tuple[str, int]]`
# has to reach `p[1]` as an int for the fold to seed and compare; against
# `object` it does not compile, and against the wrong tuple slot it would
# compile and pick the wrong element. Only the chosen values separate those.
#
# ⛔ What still needs an annotation, and is the boundary this does not move:
# a lambda bound to a NAME (`f = lambda v: v * 2`). The assignment is not an
# application, so there are no argument types to read, and the refusal there
# is the same one with nothing to replace it.
#
# Every expected line is python3.14's.

# --- a lambda applied directly ---------------------------------------------
print((lambda v: v * 2)(5))
print((lambda a, b: a + b)(3, 4))
print((lambda s: s.upper())("ab"))
print((lambda p: p[0] + p[1])((10, 20)))


# --- min()/max() with a lambda key -----------------------------------------
# The fold seeds an accumulator with the key's type, so the key has to answer
# something the seed can build. It could not see through a lambda at all.
nums = [3, 1, 2]
print(max(nums, key=lambda v: -v), min(nums, key=lambda v: -v))

pairs = [("b", 2), ("a", 3), ("c", 1)]
print(max(pairs, key=lambda p: p[1]), min(pairs, key=lambda p: p[1]))
print(max(pairs, key=lambda p: p[0]), min(pairs, key=lambda p: p[0]))

words = ["bb", "a", "ccc"]
print(max(words, key=lambda w: len(w)), min(words, key=lambda w: len(w)))


# --- the shapes that already worked, as the control ------------------------
# A named key and a builtin key must keep working: "the lambda now resolves"
# would also be satisfied by resolving everything to the same wrong thing.
def second(p: tuple[str, int]) -> int:
    return p[1]


print(max(pairs, key=second), min(pairs, key=second))
print(max(words, key=len), min(words, key=len))
print(sorted(pairs, key=lambda p: p[1]))
print(list(map(lambda v: v * 3, nums)))
