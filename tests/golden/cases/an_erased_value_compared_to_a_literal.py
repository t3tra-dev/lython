# What this pins: `==` and `!=` between a type-erased `object` and a concrete
# value -- the shape every read out of a list[object], a dict value or e.args
# lands in.
#
#     xs: list[object] = [1, "a"]
#     print(xs[0] == 1)
#     # cannot pass concrete object builtins.int as builtins.object runtime
#     # input 1 of builtins.object.__eq__; box the object at the owning ABI
#     # boundary first
#
# object.__eq__ compares two payload BOXES and dispatches on the box's class
# id, so the concrete side needs the same box the container slot beside it
# already builds. Boxing was allowed only for a source-defined class; a builtin
# was refused, and a literal int was refused twice over, because it carries its
# value in an i64 with no handle to box at all until it is materialized.
#
# Why this must run: the answer is decided by a runtime dispatch on the box's
# class id -- the same erased read is True against one literal and False against
# another of a different type, which is exactly what no static check can see.
# The loop runs the comparison often enough that a box leaked or over-released
# per comparison would show; tests/leak_gate.py reads 0 for this file.
#
# ⛔ THE ERASED SIDE MUST BE THE RECEIVER, so `1 == xs[0]` swaps. == and != are
# the two operators where that costs nothing: both dispatch on the box and both
# are symmetric under it. The ORDERING operators keep their refusal -- `<` has
# no boxed dispatcher to be symmetric under, and swapping would reverse the
# comparison rather than answer it.
#
# ⛔ A None in an erased container is a separate defect and is not here: `xs:
# list[object] = [1, None]; print(xs[1])` aborts in Ly_DecRef with no comparison
# in it at all.
def compare(v: object) -> str:
    if v == 1:
        return "one"
    if v == "a":
        return "a"
    return "other"


xs: list[object] = [1, "a", 2.5, True, 7]
print(xs[0] == 1, xs[1] == "a", xs[2] == 2.5, xs[3] == True, xs[4] == 7)
print(xs[0] == 2, xs[1] == "b", xs[0] == "a", xs[1] == 1)
print(1 == xs[0], "a" == xs[1], 2 == xs[0], "b" == xs[1])
print(xs[0] != 1, xs[0] != 2, 1 != xs[0], 2 != xs[0])
print(compare(1), compare("a"), compare(9))

d: dict[str, object] = {"n": 3, "s": "z"}
print(d["n"] == 3, d["s"] == "z", d["n"] == "z")

try:
    raise ValueError("boom", 2)
except ValueError as e:
    print(e.args[0] == "boom", e.args[1] == 2, e.args[0] == 2)

hits = 0
i = 0
while i < 300:
    if xs[0] == 1:
        hits += 1
    if xs[1] == "a":
        hits += 1
    if 2.5 == xs[2]:
        hits += 1
    i += 1
print("hits", hits)
