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

# ⛔ A UNION GOES BACK IN AS THE MEMBER ITS TAG NAMES. The read has to take a
# reference per member and PIN the container past it: a tuple a function
# returned was released between the union's build and the retain, so the retain
# ran on a string already at zero -- and with the write in place one such
# program answered instead of aborting.
def split(line: str) -> "tuple[str, int | str]":
    key, _, raw = line.partition("=")
    if raw.isdigit():
        return (key, int(raw))
    return (key, raw)


table: "dict[str, int | str]" = {}
for line in ["port=8080", "host=local"]:
    name, parsed = split(line)
    table[name] = parsed
for name in sorted(table):
    stored = table[name]
    print(name, stored, type(stored).__name__)

copied: "list[int | str]" = []
for cell in xs:
    copied.append(cell)
print(copied)

head, *tail = xs
print(head, tail)

rows: "list[list[int | str]]" = [[1, "a"], ["b", 2]]
for row in rows:
    for cell in row:
        print(type(cell).__name__, cell)
