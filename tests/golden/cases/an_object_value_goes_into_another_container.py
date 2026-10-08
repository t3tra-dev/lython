# WHAT: a value typed `object` goes into another container as itself: set(xs)
# and frozenset(xs) over a list[object], an annotated set[object] literal, a
# set[object] sorted or copied into a dict, comprehensions over a list[object]
# (filtered, or reading type(x).__name__), and values that outlive the list
# they came from.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the slot keeps the box's
# entity, so its reference count is the entity's; a count off by one prints
# the right answer and then frees an object still held -- the leak gate and
# the values read after the source is cleared are the checks.

def keep(v: object, into: list[object]) -> None:
    into.append(v)

def make() -> list[object]:
    return ["w" * 800, 7 * 10**30, ("t", "u" * 900)]

bag: list[object] = []
for o in make():
    keep(o, bag)
    keep(o, bag)
s: set[object] = set()
for o in bag:
    s.add(o)
print(len(bag), len(s))
first = bag[0]
bag.clear()
print(len(str(first)), len(s), first in s, "w" * 800 in s)
d: dict[str, object] = {}
for o in s:
    d[str(len(str(o)))] = o
s.clear()
lens: list[int] = []
for k in d:
    lens.append(len(str(d[k])))
print(sorted(lens))
fz = frozenset(make())
print(len(fz), 7 * 10**30 in fz)

t: set[object] = {1, "a", 1, (2, 3), None}
print(len(t), 1 in t, "a" in t, (2, 3) in t, None in t, 2 in t)
u: set[object] = {1}
u.add("b")
print(len(u), "b" in u)
v: set[int | str] = {1, "x", 1}
print(len(v))
def take(s: set[object]) -> int:
    return len(s)
print(take({1, "z"}))

xs: list[object] = ["s" * 700, 10**40, 1.5, None, (1, "x" * 600), [1, 2], "b"]
s = set([x for x in xs if not isinstance(x, list)])
print(len(s))
out = sorted([x for x in xs if isinstance(x, str)], key=len)
print([len(o) for o in out])
names = sorted({type(x).__name__ for x in xs})
print(names)
by_prefix = {str(x)[:3]: len(str(x)) for x in xs}
print(sorted(by_prefix.items()))
