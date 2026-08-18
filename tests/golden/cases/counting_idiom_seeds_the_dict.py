# What this pins: the word-count idiom over a dict that was never annotated.
#
#     counts = {}
#     for w in words:
#         counts[w] = counts.get(w, 0) + 1
#     # !py.union<int, object> does not provide manifest method '__add__'
#
# An empty literal has no element type of its own, so the emitter scans the rest
# of the suite for a store that says what goes in. That scan SKIPS any stored
# expression mentioning the name -- reading `counts` while deciding what
# `counts` holds reads it at the type being decided -- and in this idiom the
# self-referential store is the only one there is.
#
# The `.get(key, default)` inside it carries the answer: the default is what the
# value is when the key is absent. Binding that provisionally and re-inferring
# the WHOLE stored expression is what makes `+ 1` an int and `+ 1.5` a float,
# rather than taking the default's type as the answer.
#
# Why this must run: the seed decides the dict's storage, and a wrong seed is
# not a refusal -- it is an int slot holding a float, or the other way round.
# What says it landed right is the printed dict and the arithmetic on the way in.
#
# ⛔ The seed is a fallback, not a preference: a store that does not mention the
# name is better evidence and still wins (`later` has one of each), and two
# disagreeing stores are still a disagreement.
#
# ⛔ `counts.get(w, 0) + 1.5` over a dict of FLOATS stays refused: get's third
# overload answers `V | D`, and `float | int` has no `__add__`. Writing the
# default as `0.0` is the fix, and it is CPython's own advice about the type of
# a default.


def count(words: list[str]) -> dict[str, int]:
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts


print(count(["x", "y", "x"]))

totals = {}
for w in ["a", "b", "a"]:
    totals[w] = totals.get(w, 0.0) + 1.5
print(totals)

joined = {}
for w in ["ab", "cd", "ae"]:
    joined[w[0]] = joined.get(w[0], []) + [w]
print(joined)

later = {}
later["seed"] = 0
for w in ["a", "b"]:
    later[w] = later.get(w, 0) + 1
print(later)
