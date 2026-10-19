# WHAT: tuples whose lengths are known concatenated -- a display with a
# function's result, two names, a one-element display -- and the result
# indexed, measured and formatted with `%`, each member keeping its own type.
#
# WHY THIS IS RUN: the members and their order are the result. The
# concatenation joined the two element types, and `int` with `str` joined to
# `object`, so `fmt % ((i,) + pair)` -- how unittest reports the first
# differing element -- was refused.
def pair(a: str, b: str) -> tuple[str, str]:
    return a, b


i = 3
t = (i,) + pair("x", "y")
print(t[0] + 1, t[1].upper(), t[2], len(t))
left = (1, "a")
right = (2.5, True)
both = left + right
print(both, both[2] * 2, not both[3])
print("element %d:\n%s\n%s" % ((i,) + pair("first", "second")))
