# WHAT: reading an element whose static type is a union out of a list, a tuple
# and a dict -- by subscript, by iteration, and through `.values()` -- and then
# deciding which member is there.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is which member
# the tag names, and a wrong tag is not a refusal. Every read below decodes the
# element with `isinstance` or prints it, so a member decoded as the wrong one
# prints the wrong thing rather than failing to compile.
#
# ⛔ THE `None` MEMBER IS THE DEFAULT TAG, not a matched one: it has no lanes
# and no class id, so no comparison can select it and the tag starts there.
# `[1, None, 3]` printed `1 0 3` while that default was 0.
xs: "list[int | str]" = ["k", 1]
print(xs[0], xs[1])
for v in xs:
    print(type(v).__name__, v)

optional: "list[int | None]" = [1, None, 3]
for o in optional:
    print(o)

pair: "tuple[int | str, ...]" = (7, "b", 9)
for p in pair:
    if isinstance(p, str):
        print("str", p.upper())
    else:
        print("int", p + 1)

d: "dict[str, int | str]" = {"a": 1, "b": "z"}
for k in sorted(d):
    w = d[k]
    if isinstance(w, str):
        print(k, "str", w)
    else:
        print(k, "int", w * 2)
for w2 in d.values():
    print(type(w2).__name__)

rows: "list[list[int | str]]" = [[1, "a"], ["b", 2]]
for row in rows:
    for cell in row:
        print(type(cell).__name__, cell)
